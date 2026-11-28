from datetime import timedelta

from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from accounts.models import Institute, User
from exams.models import Exam
from patterns.models import ExamPattern
from questions.models import Question


class ExamBulkCleanupTests(TestCase):
    def setUp(self):
        self.institute = Institute.objects.create(name='Test Inst', contact_email='inst@example.com')
        self.admin = User.objects.create_user(
            username='admin1', email='admin1@example.com', password='pw',
            role='super_admin', institute=self.institute,
        )
        self.student = User.objects.create_user(
            username='stu1', email='stu1@example.com', password='pw',
            role='student', institute=self.institute,
        )
        now = timezone.now()
        self.pattern = ExamPattern.objects.create(
            name='Test Pattern', institute=self.institute, created_by=self.admin,
            total_questions=25, total_duration=120, total_marks=100,
        )
        self.exam1 = Exam.objects.create(
            title='Exam 1', institute=self.institute, created_by=self.admin, pattern=self.pattern,
            duration_minutes=60, start_date=now, end_date=now + timedelta(hours=2),
        )
        self.exam2 = Exam.objects.create(
            title='Exam 2', institute=self.institute, created_by=self.admin, pattern=self.pattern,
            duration_minutes=60, start_date=now, end_date=now + timedelta(hours=2),
        )
        self.q1 = Question.objects.create(
            exam=self.exam1, institute=self.institute, created_by=self.admin, question_number=1, question_text='Q1',
            question_type='single_mcq', options=['A', 'B', 'C', 'D'], correct_answer='A',
        )
        self.q2 = Question.objects.create(
            exam=self.exam1, institute=self.institute, created_by=self.admin, question_number=2, question_text='Q2',
            question_type='single_mcq', options=['A', 'B', 'C', 'D'], correct_answer='B',
        )
        self.other_q = Question.objects.create(
            exam=self.exam2, institute=self.institute, created_by=self.admin, question_number=1, question_text='Other Q1',
            question_type='single_mcq', options=['A', 'B', 'C', 'D'], correct_answer='C',
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.admin)

    def test_scope_all_delete_questions(self):
        resp = self.client.post(
            f'/api/questions/exams/{self.exam1.id}/bulk-cleanup/',
            {'scope': 'all', 'delete_questions': True, 'delete_answers': False},
            format='json',
            HTTP_HOST='localhost',
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {'exam_id': self.exam1.id, 'deleted_questions': 2, 'cleared_answers': 0})
        self.assertEqual(Question.objects.filter(exam=self.exam1).count(), 0)
        self.assertEqual(Question.objects.filter(exam=self.exam2).count(), 1)

    def test_scope_all_delete_answers_only(self):
        resp = self.client.post(
            f'/api/questions/exams/{self.exam1.id}/bulk-cleanup/',
            {'scope': 'all', 'delete_questions': False, 'delete_answers': True},
            format='json',
            HTTP_HOST='localhost',
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {'exam_id': self.exam1.id, 'deleted_questions': 0, 'cleared_answers': 2})
        self.q1.refresh_from_db()
        self.q2.refresh_from_db()
        self.other_q.refresh_from_db()
        self.assertEqual(self.q1.correct_answer, '')
        self.assertEqual(self.q2.correct_answer, '')
        self.assertEqual(self.other_q.correct_answer, 'C')

    def test_scope_specific_is_strictly_exam_scoped(self):
        resp = self.client.post(
            f'/api/questions/exams/{self.exam1.id}/bulk-cleanup/',
            {'scope': 'specific', 'delete_questions': True, 'delete_answers': False,
             'question_ids': [self.q1.id, self.other_q.id]},
            format='json',
            HTTP_HOST='localhost',
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {'exam_id': self.exam1.id, 'deleted_questions': 1, 'cleared_answers': 0})
        self.assertFalse(Question.objects.filter(id=self.q1.id).exists())
        self.assertTrue(Question.objects.filter(id=self.q2.id).exists())
        self.assertTrue(Question.objects.filter(id=self.other_q.id).exists())

    def test_invalid_payloads_return_400(self):
        r1 = self.client.post(
            f'/api/questions/exams/{self.exam1.id}/bulk-cleanup/',
            {'scope': 'all', 'delete_questions': False, 'delete_answers': False},
            format='json',
            HTTP_HOST='localhost',
        )
        self.assertEqual(r1.status_code, 400)

        r2 = self.client.post(
            f'/api/questions/exams/{self.exam1.id}/bulk-cleanup/',
            {'scope': 'specific', 'delete_questions': True, 'delete_answers': False, 'question_ids': []},
            format='json',
            HTTP_HOST='localhost',
        )
        self.assertEqual(r2.status_code, 400)

    def test_student_forbidden(self):
        self.client.force_authenticate(user=self.student)
        resp = self.client.post(
            f'/api/questions/exams/{self.exam1.id}/bulk-cleanup/',
            {'scope': 'all', 'delete_questions': True, 'delete_answers': False},
            format='json',
            HTTP_HOST='localhost',
        )
        self.assertEqual(resp.status_code, 403)
