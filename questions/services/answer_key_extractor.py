"""Extract (question_number, answer) pairs from an answer-key file using Mathpix OCR + structured Gemini."""

import logging
import re
from typing import Dict, List, Optional, Tuple

from django.conf import settings

from questions.services.paper_extraction import (
    PaperExtractionError,
    call_gemini,
    map_subject,
    ocr_document,
)

logger = logging.getLogger('extraction')

ANSWER_KEY_SCHEMA = {
    'type': 'object',
    'properties': {
        'answers': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'question_number': {'type': 'integer'},
                    'subject': {'type': ['string', 'null']},
                    'answer': {'type': 'string'},
                },
                'required': ['question_number', 'answer'],
            },
        },
    },
    'required': ['answers'],
}

ANSWER_KEY_SYSTEM_PROMPT = """You extract printed answer keys from exam documents into JSON. Transcribe faithfully: do not solve, calculate, work out, or invent answers.

Return ONLY answers that are explicitly printed in the document (such as in an answer-key table, answer list, or marked solutions).
If the document contains no printed answer key or solutions, return an empty list: {"answers": []}.
NEVER solve the questions yourself or guess answers.

The OCR markdown is authoritative for content: copy every character, value, formula, and expression from it.
The attached paper/image is authoritative for structure and question numbering: follow it where the OCR scrambles multi-column layouts, table columns, or question numbers.
Never copy text from the attached paper's own text layer: it flattens mathematical expressions, exponents, and subscripts (for example, 10^{3} becomes 103).

Rules:
- question_number: the printed question number (integer).
- subject: the subject heading this question belongs to (e.g. Physics, Chemistry, Mathematics), or null if no subject heading is present or the key is a flat list.
- answer: the exact answer printed.
  * For multiple-choice questions (MCQs), use the option letter(s) (A, B, C, D). If multiple options are correct, separate them with a comma (e.g. "A, C").
  * For numerical questions, output the exact value as printed (e.g. "80", "24.5", "156").
  * If the key gives numeric labels (1, 2, 3, 4) for MCQ choices, convert to (A, B, C, D).
- Skip entries where the answer is missing, cancelled, or shown as a dash/asterisk unless a specific answer is stated."""

MCQ_LETTER_PATTERN = re.compile(r'\(?([A-Da-d])\)?')
NUMERIC_ANSWER_PATTERN = re.compile(r'^-?\d+(\.\d+)?$')


class AnswerKeyExtractionError(PaperExtractionError):
    """Raised when answer-key extraction fails."""


def normalize_answer(answer: str, is_mcq: bool = False) -> str:
    """Normalize extracted answer string.

    MCQ answers: '(A)' -> 'A', '(a), (c)' -> 'A, C'.
    Numerical answers: trimmed raw string.
    """
    raw = str(answer or '').strip()
    if not raw:
        return ''

    letters = MCQ_LETTER_PATTERN.findall(raw)
    # Check if raw string is an MCQ letter sequence (e.g., "(A)", "B, C", "a / c")
    cleaned_test = re.sub(r'[\s,\(\)\[\]/|&;.]+', '', raw)
    if letters and (is_mcq or (cleaned_test and all(c.upper() in 'ABCDEFGH' for c in cleaned_test))):
        return ', '.join(l.upper() for l in letters)

    return raw


def extract_answer_key(file_path: str, content_type: str = '') -> List[Dict]:
    """Run Mathpix OCR + structured Gemini call on an answer-key document.

    Returns a list of dicts:
        [{'question_number': int, 'subject': Optional[str], 'answer': str}]
    """
    markdown = ocr_document(file_path)
    if not markdown or len(markdown.strip()) < 10:
        raise AnswerKeyExtractionError('OCR found no text in the document.')

    model = getattr(settings, 'EXTRACTION_GEMINI_MODEL', getattr(settings, 'GEMINI_MODEL', 'gemini-3.1-flash-lite'))
    try:
        data, usage = call_gemini(
            system_prompt=ANSWER_KEY_SYSTEM_PROMPT,
            markdown=markdown,
            model=model,
            file_path=file_path,
            schema=ANSWER_KEY_SCHEMA,
        )
    except PaperExtractionError:
        raise
    except Exception as e:
        logger.error(f"Gemini answer-key call failed: {e}", exc_info=True)
        raise AnswerKeyExtractionError(f"Answer-key extraction failed: {e}")

    raw_answers = data.get('answers') or []
    entries: List[Dict] = []
    for item in raw_answers:
        if not isinstance(item, dict):
            continue
        try:
            q_no = int(item.get('question_number'))
        except (TypeError, ValueError):
            continue

        subj = item.get('subject')
        if subj is not None:
            subj = str(subj).strip() or None

        ans = normalize_answer(str(item.get('answer') or ''))
        if not ans:
            continue

        entries.append({
            'question_number': q_no,
            'subject': subj,
            'answer': ans,
        })

    return entries


def match_answers_to_exam(entries: List[Dict], exam) -> List[Dict]:
    """Match extracted answer entries to questions in the exam.

    Returns a list of dicts for creating ExtractedAnswer rows:
        [{
            'question_number': int,
            'subject': str,
            'extracted_answer': str,
            'matched_question': Optional[Question],
            'current_answer': str,
            'match_status': 'matched' | 'unmatched',
            'review_flags': List[str],
        }]
    """
    questions = list(
        exam.questions.filter(is_active=True)
        .order_by('pattern_section_id', 'question_number', 'id')
    )

    exam_subjects = list(dict.fromkeys(q.subject for q in questions if q.subject))
    if exam.pattern:
        pattern_subjects = list(dict.fromkeys(exam.pattern.sections.values_list('subject', flat=True)))
        for s in pattern_subjects:
            if s and s not in exam_subjects:
                exam_subjects.append(s)

    by_subject_and_num: Dict[Tuple[str, int], any] = {}
    by_subject_and_pattern_num: Dict[Tuple[str, int], any] = {}
    by_subject_and_paper_num: Dict[Tuple[str, int], any] = {}
    by_question_number: Dict[int, any] = {}
    by_paper_question_number: Dict[int, any] = {}
    by_subject_ordered: Dict[str, List[any]] = {}

    for q in questions:
        norm_subj = map_subject(q.subject, exam_subjects).lower() if q.subject else ''
        if norm_subj and q.question_number is not None:
            by_subject_and_num[(norm_subj, q.question_number)] = q
        if norm_subj and q.question_number_in_pattern is not None:
            by_subject_and_pattern_num[(norm_subj, q.question_number_in_pattern)] = q
        if norm_subj and getattr(q, 'paper_question_number', None) is not None:
            by_subject_and_paper_num[(norm_subj, q.paper_question_number)] = q

        if q.question_number is not None and q.question_number not in by_question_number:
            by_question_number[q.question_number] = q
        if getattr(q, 'paper_question_number', None) is not None and q.paper_question_number not in by_paper_question_number:
            by_paper_question_number[q.paper_question_number] = q

        if norm_subj:
            by_subject_ordered.setdefault(norm_subj, []).append(q)

    # Track duplicates in the extracted entries
    key_counts: Dict[Tuple[Optional[str], int], int] = {}
    num_counts: Dict[int, int] = {}
    for entry in entries:
        s_key = (map_subject(entry['subject'], exam_subjects).lower() if entry.get('subject') else None)
        key_counts[(s_key, entry['question_number'])] = key_counts.get((s_key, entry['question_number']), 0) + 1
        num_counts[entry['question_number']] = num_counts.get(entry['question_number'], 0) + 1

    results: List[Dict] = []
    for entry in entries:
        q_no = entry['question_number']
        raw_subj = entry.get('subject')
        norm_subj = map_subject(raw_subj, exam_subjects).lower() if raw_subj else None
        answer = entry['answer']

        matched_q = None

        # Priority 1: Match with subject if subject is specified
        if norm_subj:
            matched_q = by_subject_and_paper_num.get((norm_subj, q_no))
            if not matched_q:
                matched_q = by_subject_and_pattern_num.get((norm_subj, q_no))
            if not matched_q:
                matched_q = by_subject_and_num.get((norm_subj, q_no))
            if not matched_q and norm_subj in by_subject_ordered:
                subj_qs = by_subject_ordered[norm_subj]
                if 1 <= q_no <= len(subj_qs):
                    matched_q = subj_qs[q_no - 1]

        # Priority 2: Match without subject or fallback
        if not matched_q:
            matched_q = by_paper_question_number.get(q_no)
        if not matched_q:
            matched_q = by_question_number.get(q_no)
        if not matched_q:
            if 1 <= q_no <= len(questions):
                matched_q = questions[q_no - 1]

        effective_subject = raw_subj or (matched_q.subject if matched_q else '')
        flags: List[str] = []

        # Duplicate flag
        is_dupe = (key_counts.get((norm_subj, q_no), 0) > 1) or (norm_subj is None and num_counts.get(q_no, 0) > 1)
        if is_dupe:
            subj_desc = f" in {effective_subject}" if effective_subject else ""
            flags.append(f"duplicate question number {q_no}{subj_desc}")

        if not matched_q:
            subj_desc = f" for {effective_subject}" if effective_subject else ""
            flags.append(f"question number {q_no} not found in exam{subj_desc}")
            results.append({
                'question_number': q_no,
                'subject': effective_subject,
                'extracted_answer': answer,
                'matched_question': None,
                'current_answer': '',
                'match_status': 'unmatched',
                'review_flags': flags,
            })
            continue

        # Matched question validations
        if matched_q.question_type in ('single_mcq', 'multiple_mcq'):
            opts = matched_q.options or []
            normalized_mcq = normalize_answer(answer, is_mcq=True)
            if normalized_mcq:
                answer = normalized_mcq
            if len(opts) < 2:
                flags.append('MCQ has fewer than 2 options')
            letters = re.findall(r'[A-Za-z]', answer)
            if letters:
                for l in letters:
                    if ord(l.upper()) - ord('A') >= len(opts):
                        flags.append(f"answer {answer} does not match the {len(opts)} options")
                        break
            elif not any(str(opt).strip().lower() == answer.strip().lower() for opt in opts):
                flags.append(f"answer \"{answer}\" is not a valid option label")

        elif matched_q.question_type == 'numerical':
            if not NUMERIC_ANSWER_PATTERN.match(answer.strip()):
                flags.append(f"numerical answer \"{answer}\" is not a number")

        results.append({
            'question_number': q_no,
            'subject': effective_subject,
            'extracted_answer': answer,
            'matched_question': matched_q,
            'current_answer': matched_q.correct_answer or '',
            'match_status': 'matched',
            'review_flags': flags,
        })

    return results
