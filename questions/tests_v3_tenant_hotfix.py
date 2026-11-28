"""Permanent multi-tenant regression tests for the V3 extraction hotfix.

Uses isolated disposable SQLite databases ('default' and 'tenant_31') so all
multi-DB routing, threading, on_commit, concurrency, and retry assertions run
without touching PostgreSQL or external APIs.
"""
import os
import tempfile
import threading
import unittest
from datetime import timedelta
from unittest import mock

from django.apps import apps as django_apps
from django.contrib.contenttypes.models import ContentType
from django.db import connections, transaction
from django.utils import timezone

from accounts.models import Institute, User, UserInstituteMembership
from accounts.tenant_middleware import activate_tenant_db
from accounts.utils import clear_current_db, get_current_db, set_current_db
from exams.models import Exam
from patterns.models import ExamPattern, PatternSection
from questions.models import ExtractedQuestion, ExtractionJob, OCRResult, Question
from questions.services.paper_extraction import ExtractionResult, PaperExtractionError
from questions.tasks import extract_questions_v3_task


class V3MultiTenantHotfixTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._orig_databases = {k: v.copy() for k, v in connections.databases.items()}
        for alias, fname in (('default', 'default.sqlite3'), ('tenant_31', 'tenant_31.sqlite3')):
            if hasattr(connections._connections, alias):
                getattr(connections._connections, alias).close()
                delattr(connections._connections, alias)
            connections.databases[alias] = {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': os.path.join(cls._tmpdir.name, fname),
                'ATOMIC_REQUESTS': False,
                'AUTOCOMMIT': True,
                'CONN_MAX_AGE': 0,
                'CONN_HEALTH_CHECKS': False,
                'OPTIONS': {},
                'TIME_ZONE': None,
                'USER': '',
                'PASSWORD': '',
                'HOST': '',
                'PORT': '',
                'TEST': {'NAME': None},
            }
            all_models = list(django_apps.get_models())
            models_to_create = all_models
            with connections[alias].schema_editor() as editor:
                for model in models_to_create:
                    editor.create_model(model)

    @classmethod
    def tearDownClass(cls):
        for alias in ('default', 'tenant_31'):
            if hasattr(connections._connections, alias):
                getattr(connections._connections, alias).close()
                delattr(connections._connections, alias)
        connections.databases.clear()
        connections.databases.update(cls._orig_databases)
        cls._tmpdir.cleanup()
        super().tearDownClass()

    def setUp(self):
        self.tearDown()
        set_current_db('default')
        self.inst = Institute.objects.using('default').create(
            name='KIITIS', contact_email='k@example.com', db_name='tenant_31'
        )
        self.admin = User.objects.create_user(
            username='kadmin', email='kadmin@example.com', password='pw',
            role='super_admin', institute=self.inst,
        )
        now = timezone.now()
        self.pattern = ExamPattern.objects.using('tenant_31').create(
            name='Physics Pattern', institute=self.inst, created_by=self.admin,
            total_questions=25, total_duration=60, total_marks=100,
        )
        PatternSection.objects.using('tenant_31').create(
            pattern=self.pattern, name='Physics', subject='Physics',
            question_type='single_mcq', start_question=1, end_question=25,
            marks_per_question=4,
        )
        self.exam = Exam.objects.using('tenant_31').create(
            title='Tenant Exam', institute=self.inst, created_by=self.admin,
            pattern=self.pattern, duration_minutes=60,
            start_date=now, end_date=now + timedelta(hours=2),
        )
        self.addCleanup(clear_current_db)

    def tearDown(self):
        clear_current_db()
        ExtractedQuestion.objects.using('tenant_31').all().delete()
        ExtractionJob.objects.using('tenant_31').all().delete()
        Exam.objects.using('tenant_31').all().delete()
        PatternSection.objects.using('tenant_31').all().delete()
        ExamPattern.objects.using('tenant_31').all().delete()
        ExtractedQuestion.objects.using('default').all().delete()
        ExtractionJob.objects.using('default').all().delete()
        User.objects.using('default').all().delete()
        Institute.objects.using('default').all().delete()

    def _fake_result(self):
        return ExtractionResult(
            questions=[{
                'number': 1, 'subject': 'Physics', 'type': 'single_mcq',
                'stem': 'Q1 stem', 'options': ['A', 'B', 'C', 'D'],
                'answer': 'A', 'review': [],
            }],
            model='gemini-test', input_tokens=10, output_tokens=5, warnings=[],
        )

    def test_tenant_thread_extraction_writes_only_to_tenant_db(self):
        job = ExtractionJob.objects.using('tenant_31').create(
            exam=self.exam, pattern=self.pattern, created_by=self.admin,
            file_name='paper.pdf', file_type='application/pdf', file_size=100,
            file_path='paper.pdf', status='pending',
        )
        self.assertFalse(ExtractionJob.objects.using('default').filter(id=job.id).exists())

        errors = []
        observed_db = []

        def worker():
            # Fresh thread starts without inherited tenant context ('default')
            observed_db.append(get_current_db())
            activate_tenant_db('tenant_31')
            try:
                with mock.patch('questions.services.paper_extraction.extract_questions',
                                return_value=self._fake_result()):
                    extract_questions_v3_task(str(job.id), tenant_db='tenant_31')
            except Exception as exc:
                errors.append(exc)
            finally:
                clear_current_db()
                connections['tenant_31'].close()

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=10)

        self.assertEqual(errors, [])
        self.assertEqual(observed_db, ['default'])
        job.refresh_from_db(using='tenant_31')
        self.assertEqual(job.status, 'completed')
        self.assertEqual(job.progress_percent, 100)
        self.assertEqual(ExtractedQuestion.objects.using('tenant_31').filter(job=job).count(), 1)
        self.assertEqual(ExtractedQuestion.objects.using('default').count(), 0)
        self.assertEqual(ExtractionJob.objects.using('default').count(), 0)

    def test_failed_extraction_marks_tenant_job_failed_even_after_context_clear(self):
        job = ExtractionJob.objects.using('tenant_31').create(
            exam=self.exam, pattern=self.pattern, created_by=self.admin,
            file_name='bad.pdf', file_type='application/pdf', file_size=100,
            file_path='bad.pdf', status='pending',
        )
        outer_caught = []

        def worker():
            activate_tenant_db('tenant_31')
            try:
                with mock.patch('questions.services.paper_extraction.extract_questions',
                                side_effect=PaperExtractionError('OCR empty')):
                    extract_questions_v3_task(str(job.id), tenant_db='tenant_31')
            except Exception as exc:
                # extract_questions_v3_task restores caller's previous tenant context ('tenant_31')
                self.assertEqual(get_current_db(), 'tenant_31')
                outer_caught.append(exc)
                _j = ExtractionJob.objects.using('tenant_31').get(id=job.id)
                if _j.status != 'failed':
                    _j.mark_failed(str(exc))
            finally:
                clear_current_db()
                connections['tenant_31'].close()

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=10)

        self.assertEqual(len(outer_caught), 1)
        job.refresh_from_db(using='tenant_31')
        self.assertEqual(job.status, 'failed')
        self.assertIn('OCR empty', job.error_message)

    def test_on_commit_delays_start_and_rollback_cancels(self):
        started = []
        with transaction.atomic(using='tenant_31'):
            job = ExtractionJob.objects.using('tenant_31').create(
                exam=self.exam, pattern=self.pattern, created_by=self.admin,
                file_name='c.pdf', file_type='application/pdf', file_size=10,
                file_path='c.pdf', status='pending',
            )
            transaction.on_commit(lambda: started.append(str(job.id)), using='tenant_31')
            self.assertEqual(started, [])  # must not run before commit
        self.assertEqual(started, [str(job.id)])

        rolled_back = []
        with transaction.atomic(using='tenant_31'):
            job2 = ExtractionJob.objects.using('tenant_31').create(
                exam=self.exam, pattern=self.pattern, created_by=self.admin,
                file_name='rb.pdf', file_type='application/pdf', file_size=10,
                file_path='rb.pdf', status='pending',
            )
            transaction.on_commit(lambda: rolled_back.append(str(job2.id)), using='tenant_31')
            transaction.set_rollback(True, using='tenant_31')
        self.assertEqual(rolled_back, [])

    def test_simultaneous_duplicate_execution_runs_only_once(self):
        job = ExtractionJob.objects.using('tenant_31').create(
            exam=self.exam, pattern=self.pattern, created_by=self.admin,
            file_name='dup.pdf', file_type='application/pdf', file_size=100,
            file_path='dup.pdf', status='pending',
        )
        barrier = threading.Barrier(2)
        results = []
        lock = threading.Lock()

        def worker():
            barrier.wait(timeout=5)
            with mock.patch('questions.services.paper_extraction.extract_questions',
                            return_value=self._fake_result()):
                r = extract_questions_v3_task(str(job.id), tenant_db='tenant_31')
            with lock:
                results.append(r)
            connections['tenant_31'].close()

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        ran = [r for r in results if r.get('success')]
        skipped = [r for r in results if r.get('skipped')]
        self.assertEqual(len(ran), 1)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(ExtractedQuestion.objects.using('tenant_31').filter(job=job).count(), 1)

    def test_retry_after_partial_failure_preserves_imported_and_avoids_duplicates(self):
        job = ExtractionJob.objects.using('tenant_31').create(
            exam=self.exam, pattern=self.pattern, created_by=self.admin,
            file_name='retry.pdf', file_type='application/pdf', file_size=100,
            file_path='retry.pdf', status='failed', error_message='prior failure',
        )
        imported_row = ExtractedQuestion.objects.using('tenant_31').create(
            job=job, question_text='Already imported Q0', question_type='single_mcq',
            options=['A', 'B', 'C', 'D'], is_imported=True,
        )
        ExtractedQuestion.objects.using('tenant_31').create(
            job=job, question_text='Stale partial Q1', question_type='single_mcq',
            options=['A', 'B', 'C', 'D'], is_imported=False,
        )
        with mock.patch('questions.services.paper_extraction.extract_questions',
                        return_value=self._fake_result()):
            res = extract_questions_v3_task(str(job.id), tenant_db='tenant_31')

        self.assertTrue(res['success'])
        qs = list(ExtractedQuestion.objects.using('tenant_31').filter(job=job).order_by('id'))
        self.assertEqual(len(qs), 2)
        self.assertEqual(qs[0].id, imported_row.id)
        self.assertTrue(qs[0].is_imported)
        self.assertEqual(qs[1].question_text, 'Q1 stem')
        self.assertFalse(qs[1].is_imported)

    def test_early_duplicate_return_restores_previous_tenant_context(self):
        job = ExtractionJob.objects.using('tenant_31').create(
            exam=self.exam, pattern=self.pattern, created_by=self.admin,
            file_name='done.pdf', file_type='application/pdf', file_size=100,
            file_path='done.pdf', status='completed',
        )
        set_current_db('default')
        res = extract_questions_v3_task(str(job.id), tenant_db='tenant_31')
        self.assertTrue(res['skipped'])
        self.assertEqual(get_current_db(), 'default')

    def test_missing_job_restores_previous_tenant_context(self):
        set_current_db('default')
        with self.assertRaises(ExtractionJob.DoesNotExist):
            extract_questions_v3_task('00000000-0000-0000-0000-000000000099', tenant_db='tenant_31')
        self.assertEqual(get_current_db(), 'default')

    def test_tenant_activation_failure_restores_context(self):
        set_current_db('default')
        with mock.patch('accounts.tenant_middleware.activate_tenant_db', side_effect=RuntimeError('bad tenant')):
            with self.assertRaisesRegex(RuntimeError, 'bad tenant'):
                extract_questions_v3_task('00000000-0000-0000-0000-000000000099', tenant_db='tenant_31')
        self.assertEqual(get_current_db(), 'default')
