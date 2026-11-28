"""
Question-paper extraction: Mathpix OCR + one structured Gemini call.

The pipeline is deliberately short, and no step depends on a particular paper's layout:
  1. ocr_document()    OCR the whole file once with Mathpix (cached by file hash in OCRResult).
  2. call_gemini()     One Gemini call turns the OCR markdown into JSON constrained by SCHEMA,
                       covering every subject at once. The original PDF is attached so the model
                       can correct OCR mis-numbering (e.g. a renumbered answer key).
  3. clean_question()  Deterministic, layout-independent fixes to the model output.
  4. review_reasons()  Checks that only flag a question for human review. Nothing is retried or
                       escalated automatically; each flag is a readable reason stored on the row.

Why this route (benchmark, Oct 2026, 15-page PCM paper with answer key): Mathpix + PDF ->
gemini-3.1-flash-lite (thinking 'low') found all 75 questions in 6/6 runs and matched the printed key in
5/6 (74/75 once) at ~$0.007/page; 90% of its remaining errors were flagged by review_reasons().
Text-only input inherited an OCR-renumbered answer key (60/75); 'minimal' thinking dropped questions and
copied flattened text from the PDF; vision-only models made silent option errors; gemini-3.6/3.8-flash
made no errors in 5 runs but cost 1.5x and took ~2x longer (set EXTRACTION_GEMINI_MODEL to switch).
A regex parser with routing gates was brittle and saved almost nothing.
"""
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
from django.conf import settings

logger = logging.getLogger('extraction')

QUESTION_TYPES = ['single_mcq', 'multiple_mcq', 'numerical', 'subjective', 'true_false', 'fill_blank']
MCQ_TYPES = ('single_mcq', 'multiple_mcq')
GEMINI_URL = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
# Inline request data is capped at 20 MB after base64 (+33%); larger files are sent as OCR text only.
MAX_ATTACH_BYTES = 14 * 1024 * 1024
ATTACH_MIME = {'.pdf': 'application/pdf', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}

SCHEMA = {
    'type': 'object',
    'properties': {
        'questions': {'type': 'array', 'items': {
            'type': 'object',
            'properties': {
                'question_number': {'type': 'integer'},
                'subject': {'type': 'string'},
                'question_type': {'type': 'string', 'enum': QUESTION_TYPES},
                'question_text': {'type': 'string'},
                'options': {'type': 'array', 'items': {'type': 'string'}},
                'sub_parts': {'type': 'array', 'items': {'type': 'string'}},
                'answer_from_source': {'type': ['string', 'null']},
            },
            'required': ['question_number', 'subject', 'question_type', 'question_text', 'options'],
        }},
        'answer_key': {'type': 'array', 'items': {
            'type': 'object',
            'properties': {'question_number': {'type': 'integer'}, 'answer': {'type': 'string'}},
            'required': ['question_number', 'answer'],
        }},
    },
    'required': ['questions'],
}

SYSTEM_PROMPT = """You convert the OCR markdown of an exam paper into JSON. Transcribe faithfully: do not solve,
correct, paraphrase, translate or add anything. Return every question in the paper, including questions
printed only as images.

The OCR markdown is authoritative for content: copy every character and formula from it. The attached paper
is only for checking structure - follow it where the OCR mis-numbers questions or answer-key entries, or
scrambles which options belong to which question. Never copy text from the attached paper's own text layer:
it drops superscripts, subscripts and symbols (for example 10^{{3}} becomes 103).

Rules:
- question_number: the printed number. Numbers can appear before, beside or after their stem or options
  because of layout; group each stem and its options with the number they belong to.
- question_text: the stem only - no question number, no options, no marks annotation such as "[2 marks]".
  NEVER repeat the answer choices/options (whether numbered 1, 2, 3, 4 or labeled A, B, C, D) inside question_text.
  Write every mathematical/chemical expression in LaTeX inside $...$ (for example $x^2$, $10^{{-6}}\\text{{ C}}$, $\\frac{{a}}{{b}}$).
  Never output raw LaTeX commands or power expressions like ^{{...}} outside $...$.
- Image links ![](url): every link that belongs to a question must appear, unchanged, in that question's
  question_text (a diagram in the stem) or in an option (an option that is a picture).
  If the choices (A)-(D) are each a graph/diagram image ![](url), put those 4 image links in options in order
  and do NOT put them in question_text.
- options: the choices in order (A, B, C, D), text only without the label. If the selectable choices are
  bare labels that refer to content printed elsewhere (for example "(A) 1." referring to "(1) <content>", or
  "(A) A." referring to "(A) ![](url)"), output the referenced content/image URL. If the same set of choices is
  printed twice, output it once. Use \\epsilon_{{0}} (never \\in_{{0}}) for permittivity.
- question_type: single_mcq (exactly one correct choice), multiple_mcq (the paper says one or more may be
  correct), numerical (numeric answer, no choices - e.g. "Find the value of n", "in μC", "in N/C", "ratio b/a"),
  fill_blank, true_false, subjective (written or constructed proof/essay answer).
- sub_parts: labelled parts (a), (b), (i)... of a non-MCQ question, in order, without labels; never put
  them in options.
- answer_from_source: only an answer printed in the paper (answer key or solution) - never work it out.
  Use the option letter (A-D) for MCQs, the value for numerical questions. Otherwise null.
  If an answer-key table is present, also return every entry in answer_key.
- subject: the nearest subject heading above the question; if there is none, choose from: {subjects}.
- Ignore page headers/footers, logos, instructions and section titles."""


class PaperExtractionError(Exception):
    """Extraction could not produce a usable result; the message is shown to the user."""


@dataclass
class ExtractionResult:
    questions: List[Dict]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    warnings: List[str] = field(default_factory=list)


def extract_questions(file_path: str, subjects: List[str]) -> ExtractionResult:
    """OCR `file_path` and return cleaned questions, each with a `review` list of reasons (empty = clean)."""
    markdown = ocr_document(file_path)
    if len(markdown.strip()) < 20:
        raise PaperExtractionError('OCR found no text in the document.')

    model = settings.EXTRACTION_GEMINI_MODEL
    data, usage = call_gemini(SYSTEM_PROMPT.format(subjects=', '.join(subjects)), markdown, model, file_path)

    answers_printed = prints_answers(markdown)
    answer_key = {k['question_number']: k['answer'] for k in data.get('answer_key') or []} if answers_printed else {}
    questions = []
    for raw in data.get('questions') or []:
        q = clean_question(raw, subjects, answers_printed, answer_key)
        q['review'] = review_reasons(q, markdown)
        questions.append(q)

    warnings = _numbering_warnings(questions)
    not_extracted = sorted(set(answer_key) - {q['number'] for q in questions})
    if not_extracted:
        warnings.append(f'answer key lists questions that were not extracted: {not_extracted[:20]}')
    for w in warnings:
        logger.warning(f'Extraction check: {w}')
    return ExtractionResult(
        questions=questions, model=model, warnings=warnings,
        input_tokens=usage.get('promptTokenCount', 0),
        output_tokens=usage.get('candidatesTokenCount', 0) + usage.get('thoughtsTokenCount', 0),
    )


# --------------------------------------------------------------------------- OCR + model calls
def ocr_document(file_path: str) -> str:
    """Mathpix markdown for the whole file (cached by file hash, so re-uploads are free)."""
    from questions.services.mathpix_service import MathpixService

    lower = file_path.lower()
    if lower.endswith('.txt'):
        with open(file_path, encoding='utf-8', errors='replace') as f:
            return f.read()
    service = MathpixService()
    if lower.endswith(('.png', '.jpg', '.jpeg')):
        text, _ = service.extract_image_with_cache(file_path)
    else:
        text, _ = service.extract_pdf_with_cache(file_path)
    return text


def _file_part(file_path: Optional[str]) -> Optional[Dict]:
    """The original PDF/image as an inline part, so the model can check the OCR against the page."""
    mime = ATTACH_MIME.get(os.path.splitext(file_path or '')[1].lower())
    if not mime or not os.path.isfile(file_path) or os.path.getsize(file_path) > MAX_ATTACH_BYTES:
        return None
    with open(file_path, 'rb') as f:
        return {'inlineData': {'mimeType': mime, 'data': base64.b64encode(f.read()).decode()}}


def call_gemini(
    system_prompt: str,
    markdown: str,
    model: str,
    file_path: Optional[str] = None,
    schema: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """One structured-output call. Returns (parsed JSON, usageMetadata). Raises PaperExtractionError."""
    parts = [{'text': 'OCR MARKDOWN:\n' + markdown}]
    file_part = _file_part(file_path)
    if file_part:
        parts.insert(0, file_part)
    body = {
        'systemInstruction': {'parts': [{'text': system_prompt}]},
        'contents': [{'role': 'user', 'parts': parts}],
        'generationConfig': {
            'responseMimeType': 'application/json',
            'responseJsonSchema': SCHEMA if schema is None else schema,
            'thinkingConfig': {'thinkingLevel': settings.EXTRACTION_GEMINI_THINKING_LEVEL},
        },
    }
    resp, last_error = None, None
    for attempt in range(3):
        try:
            resp = requests.post(GEMINI_URL.format(model=model), json=body, timeout=300,
                                 headers={'x-goog-api-key': settings.GEMINI_API_KEY})
        except requests.RequestException as e:
            resp, last_error = None, e
        if resp is not None and resp.status_code not in (429, 500, 503):
            break
        time.sleep(5 * (attempt + 1))
    if resp is None:
        raise PaperExtractionError(f'Could not reach Gemini: {last_error}')

    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    if resp.status_code != 200:
        message = (payload.get('error') or {}).get('message') or resp.text[:300]
        raise PaperExtractionError(f'Gemini ({model}) returned HTTP {resp.status_code}: {message}')
    candidates = payload.get('candidates') or []
    if not candidates:
        raise PaperExtractionError(f'Gemini returned no output: {payload.get("promptFeedback")}')
    finish = candidates[0].get('finishReason')
    if finish not in (None, 'STOP'):
        raise PaperExtractionError(f'Gemini stopped early ({finish}); the output would be incomplete.')
    text = ''.join(p.get('text', '') for p in candidates[0].get('content', {}).get('parts', []) if not p.get('thought'))
    try:
        return json.loads(text), payload.get('usageMetadata', {})
    except ValueError as e:
        raise PaperExtractionError(f'Gemini returned invalid JSON: {e}')


# --------------------------------------------------------------------------- cleaning
# A LaTeX command whose backslash the model did not escape in JSON decodes to a control character:
# \frac -> form feed + "rac", \times -> tab + "imes", \beta -> backspace + "eta", \rightarrow -> CR + "ightarrow".
_LF_COMMANDS = (r'(?:eq|eg|e|u|abla|ot|otin|i|mid|parallel|leq|geq|subseteq|supseteq|exists|cong|sim|'
                r'leftarrow|rightarrow|Leftarrow|Rightarrow|ewline)')
_SUBJECT_ALIASES = {'math': 'mathematics', 'maths': 'mathematics', 'phy': 'physics', 'chem': 'chemistry',
                    'bio': 'biology', 'cs': 'computer science', 'eng': 'english'}
_ANSWERS_PRINTED = re.compile(
    r'^\s*#*\s*\**\s*answer\s*key\b|\b(?:ans|answer|correct\s+(?:option|answer))\s*[:.\-]\s*\(?[A-Da-d0-9-]',
    re.I | re.M)


def _bare_image_urls(text: str) -> str:
    """'![](url)' -> url on its own line: the student view renders bare image URLs, not markdown."""
    return re.sub(r'[ \t]*!\[[^\]]*\]\((\S+?)\)[ \t]*', r'\n\1\n', text).strip()


_UNICODE_MATH_MAP = {
    'μ': r'\mu ', 'α': r'\alpha ', 'β': r'\beta ', 'γ': r'\gamma ', 'δ': r'\delta ',
    'θ': r'\theta ', 'λ': r'\lambda ', 'ε': r'\varepsilon ', 'π': r'\pi ',
    'σ': r'\sigma ', 'τ': r'\tau ', 'φ': r'\phi ', 'ω': r'\omega ', 'Ω': r'\Omega ',
    'Δ': r'\Delta ', '×': r'\times ', '÷': r'\div ', '±': r'\pm ', '≤': r'\leq ',
    '≥': r'\geq ', '≠': r'\neq ', '≈': r'\approx ', '→': r'\rightarrow ', '°': r'^{\circ}',
}
_UNIT_RE = re.compile(
    r'(?:^|\s)[+-]?\d+(?:\.\d+)?\s*[μmnkKMGT]?(?:C|F|Ω|V|A|N|J|W|Pa|Hz|eV|mol|g|kg|m|s|cm|mm|km)'
    r'(?:/(?:m|s|kg|C|N|mol|L|m²|s²))?(?:$|\s)'
)


def _normalize_math_delimiters(text: str) -> str:
    """Convert \\(...\\)/\\[...\\] to $...$/$$...$$ and fix Mathpix's \\in_0 permittivity typo."""
    t = re.sub(r'\\\(([\s\S]*?)\\\)', lambda m: f'${m.group(1).strip()}$', text)
    t = re.sub(r'\\\[([\s\S]*?)\\\]', lambda m: f'$${m.group(1).strip()}$$', t)
    # Mathpix sometimes OCRs \epsilon_0 as \in_0; set membership \in never takes a numeric subscript.
    t = re.sub(r'\\in(?=\s*(?:_\{?\s*0\b|_0\b))', r'\\epsilon', t)
    return t


def _to_latex_chars(text: str) -> str:
    for ch, rep in _UNICODE_MATH_MAP.items():
        if ch in text:
            text = text.replace(ch, rep)
    return re.sub(r' {2,}', ' ', text).strip()


def fix_escapes(text) -> str:
    t = str(text or '')
    t = t.replace('\x0c', '\\f').replace('\x08', '\\b').replace('\r', '\\r')
    t = re.sub(r'\t(?=[A-Za-z])', r'\\t', t)
    # A newline is only ambiguous inside math, where "\n" + "eq" etc. was meant to be \neq.
    t = re.sub(r'\$\$[\s\S]*?\$\$|\$[^$]+?\$',
               lambda m: re.sub(r'\n(?=' + _LF_COMMANDS + r'(?![A-Za-z]))', r'\\n', m.group(0)), t)
    return _normalize_math_delimiters(t)


def _wrap_math(text: str) -> str:
    """'\\frac{1}{2}' -> '$\\frac{1}{2}$' when the model left an all-math field without delimiters."""
    t = text.strip()
    if not t or '$' in t or 'http://' in t or 'https://' in t:
        return t
    # LaTeX command or superscript/subscript
    if re.search(r'\\[A-Za-z]+|\^|_\{|\^[0-9]', t):
        return f'${t}$'
    # Math equation or assignment like 'T=m g+E q' or 'x = 2'
    if re.search(r'^[A-Za-z]\s*=\s*.*[+\-*/]', t):
        return f'${t}$'
    # Plain-text Physics/Chemistry unit/symbol option like '2 μC', '10 Ω', 'α + β'
    if any(ch in t for ch in _UNICODE_MATH_MAP) and len(t) <= 40:
        return f'${_to_latex_chars(t)}$'
    return t


def _wrap_stem_inline_math(stem: str) -> str:
    """Wrap isolated mathematical/chemical expressions outside $...$ in $...$."""
    if not stem:
        return ''
    # Merge '4 × $10^{23}$' -> '$4 \times 10^{23}$'
    stem = re.sub(r'(\d+(?:\.\d+)?)\s*[×x]\s*\$([^$\n]+)\$', r'$\1 \\times \2$', stem)
    # Split by existing $...$ blocks so we only inspect non-math text
    parts = re.split(r'(\$\$[\s\S]*?\$\$|\$[^$\n]+?\$)', stem)
    for i in range(0, len(parts), 2):
        # Skip URLs while wrapping inline math tokens
        seg = parts[i]
        tokens = re.split(r'(https?://\S+)', seg)
        for j in range(0, len(tokens), 2):
            tok = tokens[j]
            # Wrap braced powers/subscripts like '-(2/5)^{-3/2} q' or 'H_{2}O'
            tok = re.sub(r'([+-]?(?:\([^)]+\)|[A-Za-z0-9])+(?:\^|_)\{[^}]+\}(?:\s*[A-Za-z]+)?)', r'$\1$', tok)
            # Wrap simple unbraced powers/subscripts like 'x^2', 'H_2O' (avoid snake_case identifiers)
            tok = re.sub(r'(?<![A-Za-z0-9_])([A-Za-z0-9]+\^[0-9]+|[A-Z][a-z]?_[0-9]+(?:\^\{?[0-9+-]+\}?)?)', r'$\1$', tok)
            tokens[j] = tok
        parts[i] = ''.join(tokens)
    return ''.join(parts)


def _plain(text: str) -> str:
    """Formatting-insensitive form for comparisons: no LaTeX wrappers, braces, $ or whitespace."""
    t = re.sub(r'\\(?:mathrm|mathbf|text|left|right|displaystyle)(?![A-Za-z])', '', str(text or ''))
    t = re.sub(r'\\(?:rightarrow|to)(?![A-Za-z])', '→', t).replace('\\dfrac', '\\frac')
    return re.sub(r'[\s${}]', '', t).lower()


def _numbers(text: str) -> set:
    """Integers in the text; braces are separators so 'x^{2}' and 'x^2' both give {'2'}."""
    return set(re.findall(r'\d+', re.sub(r'[{}]', ' ', str(text or ''))))


def _strip_repeated_options(stem: str, options: List[str]) -> str:
    """Drop an options list the model repeated at the end of the stem."""
    if len(options) < 2:
        return stem
    m = re.search(r'\s*(\(\s*(?:1|a|A)\s*\)|(?<![A-Za-z0-9])[1Aa][.)])\s*\S[\s\S]*$', stem)
    if m and _plain(options[0]) in _plain(m.group(0)) and _plain(options[-1]) in _plain(m.group(0)):
        return stem[:m.start()].rstrip()
    return stem


def map_subject(subject: Optional[str], subjects: List[str]) -> str:
    """Map a heading/model subject name onto the pattern's subject names ('Math' -> 'Mathematics')."""
    s = (subject or '').strip()
    key = _SUBJECT_ALIASES.get(s.lower(), s.lower())
    for candidate in subjects:
        if _SUBJECT_ALIASES.get(candidate.lower(), candidate.lower()) == key:
            return candidate
    return s


def prints_answers(markdown: str) -> bool:
    """True only if the paper itself prints answers ('Answer key' heading or 'Ans: B' markers)."""
    return bool(_ANSWERS_PRINTED.search(markdown or ''))


def _extract_image_options_from_stem(stem: str, options: List[str]) -> Tuple[str, List[str]]:
    """If options are bare labels (A, B, C, D) and stem ends with labeled option images, move them into options."""
    if len(options) < 2:
        return stem, options
    letters = [re.sub(r'[()\[\].\s]', '', o).upper() for o in options]
    if letters != [chr(65 + i) for i in range(len(letters))]:
        return stem, options
    pattern = re.compile(
        r'\s*(?:\(\s*A\s*\)|(?<![A-Za-z0-9])A[.)])\s*(https?://\S+)\s*'
        + ''.join(rf'(?:\(\s*{chr(65 + i)}\s*\)|(?<![A-Za-z0-9]){chr(65 + i)}[.)])\s*(https?://\S+)\s*' for i in range(1, len(letters)))
        + r'$',
        re.I,
    )
    m = pattern.search(stem)
    if not m:
        return stem, options
    return stem[:m.start()].rstrip(), [m.group(i + 1) for i in range(len(letters))]


def clean_question(raw: Dict, subjects: List[str], answers_printed: bool, answer_key: Dict) -> Dict:
    number = raw.get('question_number')
    options = [_bare_image_urls(_wrap_math(fix_escapes(o))) for o in raw.get('options') or []]
    stem = _bare_image_urls(fix_escapes(raw.get('question_text')))
    stem, options = _extract_image_options_from_stem(stem, options)
    stem = _wrap_stem_inline_math(_strip_repeated_options(stem, options))
    parts = [re.sub(r'^(?:\([a-zA-Z0-9]+\)|[a-zA-Z0-9]+[.)])\s+', '', _bare_image_urls(_wrap_math(fix_escapes(p))))
             for p in raw.get('sub_parts') or []]
    if parts:
        stem = '\n'.join([stem] + [f'({chr(97 + i)}) {p}' for i, p in enumerate(parts)])
    qtype = raw.get('question_type') if raw.get('question_type') in QUESTION_TYPES else 'subjective'

    answer = ''
    if answers_printed:
        answer = str(answer_key.get(number) or raw.get('answer_from_source') or '').strip()
        if qtype in MCQ_TYPES and re.fullmatch(r'\(?[A-Da-d]\)?(\s*,\s*\(?[A-Da-d]\)?)*', answer):
            answer = ','.join(re.findall(r'[A-Da-d]', answer)).upper()
    return {'number': number, 'subject': map_subject(raw.get('subject'), subjects), 'type': qtype,
            'stem': stem, 'options': options, 'answer': answer}


# --------------------------------------------------------------------------- review flags
def _renders(text: str) -> bool:
    """Mirrors the student view (LaTeXRenderer.tsx): KaTeX only inside $..$/$$..$$, plain text elsewhere."""
    outside = re.sub(r'\$\$[\s\S]*?\$\$|\$[^$\n]+?\$', ' ', text)
    outside = re.sub(r'!\[[^\]]*\]\([^)]*\)|https?://\S+', ' ', outside)
    return '$' not in outside and not re.search(r'\\[A-Za-z]+|\^\{|_\{', outside)


_IMAGE_URL = re.compile(r'https?://\S+\.(?:png|jpe?g|gif|webp)', re.I)
# Narrow on purpose: 'graph'/'diagram' also appear in text-only questions ("a graph is plotted between...").
_MENTIONS_FIGURE = re.compile(r'\b(?:figure|shown|depicted)\b', re.I)


def review_reasons(q: Dict, markdown: str) -> List[str]:
    """Readable reasons a reviewer should look at this question; empty when nothing looks wrong."""
    reasons = []
    opts = q['options']
    if not q['stem'].strip():
        reasons.append('question text is empty')
    if q['type'] in MCQ_TYPES:
        if len(opts) < 2:
            reasons.append('MCQ has fewer than 2 options')
        letters = [re.sub(r'[()\[\].]', '', _plain(o)) for o in opts]
        if len(letters) >= 2 and letters == [chr(97 + i) for i in range(len(letters))]:
            reasons.append('options are only labels (A, B, C, D); the real options may be in an image')
        if re.fullmatch(r'[A-D](,[A-D])*', q['answer']) and \
                any(ord(a) - ord('A') >= len(opts) for a in q['answer'].split(',')):
            reasons.append(f'answer {q["answer"]} does not match the {len(opts)} options')
    elif opts:
        reasons.append(f'{q["type"]} question has options')
    if _MENTIONS_FIGURE.search(q['stem']) and not any(_IMAGE_URL.search(f) for f in [q['stem']] + opts):
        reasons.append('mentions a figure but has no image')
    if q['type'] == 'numerical' and q['answer'] and not re.fullmatch(r'-?\d+(\.\d+)?', q['answer']):
        reasons.append(f'numerical answer "{q["answer"]}" is not a number')
    if not all(_renders(f) for f in [q['stem']] + opts):
        reasons.append('some math is outside $...$ and will not render')
    missing = sorted(_numbers(' '.join([q['stem']] + opts)) - _numbers(markdown), key=int)
    if missing:
        reasons.append(f'numbers not found in the OCR text: {", ".join(missing[:5])}')
    return reasons


def _numbering_warnings(questions: List[Dict]) -> List[str]:
    """Duplicate or skipped question numbers within a subject (subject-wise numbering is allowed)."""
    warnings, by_subject = [], {}
    for q in questions:
        if isinstance(q['number'], int):
            by_subject.setdefault(q['subject'], []).append(q['number'])
    for subject, numbers in by_subject.items():
        dupes = sorted({n for n in numbers if numbers.count(n) > 1})
        gaps = sorted(set(range(min(numbers), max(numbers) + 1)) - set(numbers))
        if dupes:
            warnings.append(f'{subject}: duplicate question numbers {dupes}')
            for q in questions:
                if q['subject'] == subject and q['number'] in dupes:
                    q['review'].append('duplicate question number')
        if gaps:
            warnings.append(f'{subject}: question numbers missing {gaps[:20]}')
    return warnings
