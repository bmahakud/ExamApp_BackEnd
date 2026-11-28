"""Endpoints for answer-key extraction: upload a PDF, review extracted answers, apply to questions."""

import logging
import os
import uuid as uuid_lib

from django.conf import settings
from django.db import transaction
from django.shortcuts import get_object_or_404
from django.utils.text import get_valid_filename
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from accounts.utils import get_current_db
from exams.models import Exam
from questions.models import AnswerExtractionJob, ExtractedAnswer, ExtractedQuestion, Question
from questions.tasks import extract_answers_task

logger = logging.getLogger('extraction')


def _user_owns_exam(user, exam: Exam) -> bool:
    """Permission check: the user's institute must match the exam's institute."""
    user_institute = getattr(user, 'institute', None)
    return user_institute is not None and exam.institute_id == user_institute.id


def _serialize_row(row: ExtractedAnswer) -> dict:
    return {
        'id': row.id,
        'question_number': row.question_number,
        'subject': row.subject,
        'extracted_answer': row.extracted_answer,
        'current_answer': row.current_answer,
        'match_status': row.match_status,
        'matched_question_id': row.matched_question_id,
        'review_flags': row.review_flags or [],
        'skip': row.skip,
        'is_applied': row.is_applied,
        'apply_error': row.apply_error,
    }


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def upload_answer_key(request):
    """Upload an answer-key file for an exam and start the extraction job.

    POST /api/questions/answer-keys/
    form-data: file, exam_id
    """
    uploaded_file = request.FILES.get('file')
    exam_id = request.data.get('exam_id') or request.data.get('exam')

    if not uploaded_file:
        return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)
    if not exam_id:
        return Response({'error': 'exam_id is required'}, status=status.HTTP_400_BAD_REQUEST)

    exam = get_object_or_404(Exam, id=exam_id)
    if not _user_owns_exam(request.user, exam):
        return Response({'error': 'You do not have permission to upload an answer key for this exam'},
                        status=status.HTTP_403_FORBIDDEN)

    upload_dir = os.path.join(settings.MEDIA_ROOT, 'answer_key_uploads')
    os.makedirs(upload_dir, exist_ok=True)

    safe_name = get_valid_filename(uploaded_file.name)
    base, ext = os.path.splitext(safe_name)
    unique_name = f"{base}_{uuid_lib.uuid4().hex[:8]}{ext}"
    full_path = os.path.join(upload_dir, unique_name)

    with open(full_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    job = AnswerExtractionJob.objects.create(
        exam=exam,
        created_by=request.user,
        file_name=uploaded_file.name,
        file_type=uploaded_file.content_type or '',
        file_size=uploaded_file.size,
        file_path=full_path,
        status='pending',
    )

    tenant_db = get_current_db()
    import threading
    job_id_str = str(job.id)

    def _run_answer_extraction():
        try:
            extract_answers_task(job_id_str, tenant_db=tenant_db)
        except Exception as exc:
            logger.error(f"Background answer-key extraction failed for job {job_id_str}: {exc}", exc_info=True)
            try:
                _j = AnswerExtractionJob.objects.get(id=job_id_str)
                if _j.status != 'failed':
                    _j.mark_failed(str(exc))
            except Exception:
                pass

    t = threading.Thread(target=_run_answer_extraction, daemon=True, name=f"answer-key-{job_id_str[:8]}")
    t.start()

    return Response(
        {'job_id': job_id_str, 'status': job.status, 'message': 'Answer-key extraction started'},
        status=status.HTTP_201_CREATED,
    )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_answer_key_status(request, job_id):
    """Return current status of an answer-key extraction job.

    GET /api/questions/answer-keys/{job_id}/status/
    """
    job = get_object_or_404(AnswerExtractionJob, id=job_id)
    if not _user_owns_exam(request.user, job.exam):
        return Response({'error': 'Forbidden'}, status=status.HTTP_403_FORBIDDEN)

    return Response({
        'job_id': str(job.id),
        'status': job.status,
        'progress_percent': job.progress_percent,
        'error_message': job.error_message,
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_answer_key_preview(request, job_id):
    """Return all extracted answer rows for review, with a summary count.

    GET /api/questions/answer-keys/{job_id}/preview/
    """
    job = get_object_or_404(AnswerExtractionJob, id=job_id)
    if not _user_owns_exam(request.user, job.exam):
        return Response({'error': 'Forbidden'}, status=status.HTTP_403_FORBIDDEN)

    rows = list(job.extracted_answers.all())
    matched_count = sum(1 for r in rows if r.match_status == 'matched')
    flagged_count = sum(1 for r in rows if r.review_flags)

    return Response({
        'job_id': str(job.id),
        'status': job.status,
        'summary': {
            'total': len(rows),
            'matched': matched_count,
            'unmatched': len(rows) - matched_count,
            'flagged': flagged_count,
        },
        'rows': [_serialize_row(r) for r in rows],
    })


@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
@parser_classes([JSONParser])
def update_extracted_answer(request, pk):
    """Edit an extracted answer row before apply. Accepts extracted_answer and/or skip.

    PATCH /api/questions/extracted-answers/{id}/
    """
    row = get_object_or_404(ExtractedAnswer.objects.select_related('job__exam'), pk=pk)
    if not _user_owns_exam(request.user, row.job.exam):
        return Response({'error': 'Forbidden'}, status=status.HTTP_403_FORBIDDEN)
    if row.is_applied:
        return Response({'error': 'Row has already been applied and cannot be edited'},
                        status=status.HTTP_400_BAD_REQUEST)

    update_fields = []
    if 'extracted_answer' in request.data:
        new_answer = str(request.data['extracted_answer'] or '').strip()
        if not new_answer:
            return Response({'error': 'extracted_answer cannot be empty'},
                            status=status.HTTP_400_BAD_REQUEST)
        row.extracted_answer = new_answer
        update_fields.append('extracted_answer')
    if 'skip' in request.data:
        row.skip = bool(request.data['skip'])
        update_fields.append('skip')

    if update_fields:
        update_fields.append('updated_at')
        row.save(update_fields=update_fields)
    return Response(_serialize_row(row))


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([JSONParser])
def apply_answer_key(request, job_id):
    """Write extracted answers onto the matched Question rows.

    POST /api/questions/answer-keys/{job_id}/apply/

    Overwrites Question.correct_answer for every matched, non-skipped, not-yet-applied row.
    Question.save() handles label-to-option-text normalization.
    """
    job = get_object_or_404(AnswerExtractionJob, id=job_id)
    if not _user_owns_exam(request.user, job.exam):
        return Response({'error': 'Forbidden'}, status=status.HTTP_403_FORBIDDEN)
    if job.status != 'completed':
        return Response({'error': f'Job is not ready to apply (status: {job.status})'},
                        status=status.HTTP_400_BAD_REQUEST)

    rows = list(job.extracted_answers.select_related('matched_question').filter(is_applied=False))

    updated = 0
    skipped = 0
    unmatched = 0
    errors = []

    with transaction.atomic():
        for row in rows:
            if row.match_status != 'matched' or row.matched_question_id is None:
                unmatched += 1
                continue
            if row.skip:
                skipped += 1
                continue

            question = row.matched_question
            try:
                question.correct_answer = row.extracted_answer
                question.save(update_fields=['correct_answer', 'updated_at'])
                row.is_applied = True
                row.apply_error = ''
                row.save(update_fields=['is_applied', 'apply_error', 'updated_at'])
                updated += 1
            except Exception as e:
                logger.error(f"Apply failed for ExtractedAnswer {row.id}: {e}", exc_info=True)
                row.apply_error = str(e)
                row.save(update_fields=['apply_error', 'updated_at'])
                errors.append({'row_id': row.id, 'error': str(e)})

    return Response({
        'job_id': str(job.id),
        'updated': updated,
        'skipped': skipped,
        'unmatched': unmatched,
        'errors': errors,
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([JSONParser])
def exam_bulk_cleanup(request, exam_id):
    """Delete questions or clear answers for a specific exam.

    POST /api/questions/exams/<int:exam_id>/bulk-cleanup/
    """
    exam = get_object_or_404(Exam, id=exam_id)
    user_role = (getattr(request.user, 'role', '') or '').lower()
    if not _user_owns_exam(request.user, exam) or user_role == 'student':
        return Response({'error': 'Forbidden'}, status=status.HTTP_403_FORBIDDEN)

    scope = request.data.get('scope')
    if scope not in ('all', 'specific'):
        return Response({'error': "scope must be 'all' or 'specific'"}, status=status.HTTP_400_BAD_REQUEST)

    delete_questions = bool(request.data.get('delete_questions', False))
    delete_answers = bool(request.data.get('delete_answers', False))
    if not delete_questions and not delete_answers:
        return Response(
            {'error': 'At least one of delete_questions or delete_answers must be true'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    question_ids = request.data.get('question_ids')
    if scope == 'specific':
        if (not isinstance(question_ids, list) or not question_ids or
                not all(isinstance(q, int) and not isinstance(q, bool) for q in question_ids)):
            return Response(
                {'error': 'question_ids must be a non-empty list of integers when scope is specific'},
                status=status.HTTP_400_BAD_REQUEST,
            )

    with transaction.atomic():
        target_qs = Question.objects.filter(exam=exam)
        if scope == 'specific':
            target_qs = target_qs.filter(id__in=question_ids)

        deleted_questions = 0
        cleared_answers = 0

        if delete_questions:
            target_ids = list(target_qs.values_list('id', flat=True))
            if target_ids:
                ExtractedQuestion.objects.filter(
                    job__exam=exam, imported_question_id__in=target_ids
                ).update(is_imported=False, imported_question=None)
                ExtractedAnswer.objects.filter(
                    job__exam=exam, matched_question_id__in=target_ids
                ).update(is_applied=False, matched_question=None)
                deleted_questions = len(target_ids)
                Question.objects.filter(exam=exam, id__in=target_ids).delete()
        elif delete_answers:
            target_ids = list(target_qs.values_list('id', flat=True))
            if target_ids:
                cleared_answers = Question.objects.filter(exam=exam, id__in=target_ids).update(correct_answer='')
                ExtractedAnswer.objects.filter(
                    job__exam=exam, matched_question_id__in=target_ids
                ).update(is_applied=False)
            if scope == 'all':
                ExtractedAnswer.objects.filter(job__exam=exam, is_applied=True).update(is_applied=False)

    return Response({
        'exam_id': exam.id,
        'deleted_questions': deleted_questions,
        'cleared_answers': cleared_answers,
    })
