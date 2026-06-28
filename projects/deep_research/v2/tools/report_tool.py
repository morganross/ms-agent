# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import json
from ms_agent.llm.utils import Tool
from ms_agent.tools.base import ToolBase
from ms_agent.utils.utils import file_lock, render_markdown_todo

try:
    from evidence_tool import _parse_note_from_md  # type: ignore
except Exception:  # pragma: no cover
    from .evidence_tool import _parse_note_from_md  # type: ignore


def _now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S')


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_read_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _write_text(path: str, content: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        _ensure_dir(parent)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def _normalize_report_content(content: str) -> str:
    text = str(content or '').strip()
    if not text:
        return ''

    if '\\n' in text:
        text = text.replace('\\n', '\n')
    if "\\'" in text:
        text = text.replace("\\'", "'")
    text = _demote_extra_top_level_headings(text)

    lines = text.splitlines()
    if lines and lines[0].startswith('# '):
        lines[0] = lines[0].replace(' (Draft)', '').replace('(Draft)', '').rstrip()
        text = '\n'.join(lines)
    return text.rstrip() + '\n'


def _demote_heading_line(line: str) -> str:
    match = re.match(r'^(#{1,5})(\s+.+)$', line)
    if not match:
        return line
    return f"{match.group(1)}#{match.group(2)}"


def _demote_all_headings(text: str) -> str:
    return '\n'.join(_demote_heading_line(line) for line in text.splitlines())


def _demote_extra_top_level_headings(text: str) -> str:
    lines = text.splitlines()
    seen_title = False
    has_extra_h1 = False
    for line in lines:
        if not line.startswith('# '):
            continue
        if not seen_title:
            seen_title = True
        else:
            has_extra_h1 = True
            break
    if not has_extra_h1:
        return text

    normalized_lines: List[str] = []
    seen_title = False
    for line in lines:
        if line.startswith('# ') and not seen_title:
            seen_title = True
            normalized_lines.append(line)
        elif seen_title:
            normalized_lines.append(_demote_heading_line(line))
        else:
            normalized_lines.append(line)
    return '\n'.join(normalized_lines)


def _heading_title(line: str) -> Optional[str]:
    match = re.match(r'^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$', line)
    if not match:
        return None
    return match.group(1).strip()


def _canonical_heading_text(text: str) -> str:
    normalized = re.sub(r'\s+', ' ', str(text or '').strip().lower())
    normalized = re.sub(r'^chapter\s+\d+\s*[:.\-]?\s*', '', normalized)
    normalized = re.sub(r'^\d+(?:\.\d+)*\s*[.)]?\s*', '', normalized)
    return normalized


def _ensure_chapter_heading(content: str, chapter_title: str) -> str:
    text = str(content or '').strip()
    title = str(chapter_title or '').strip()
    if not text or not title:
        return text.rstrip() + '\n' if text else ''

    lines = text.splitlines()
    first_content_index: Optional[int] = None
    for index, line in enumerate(lines):
        if line.strip():
            first_content_index = index
            break
    if first_content_index is None:
        return f'## {title}\n'

    first_heading = _heading_title(lines[first_content_index])
    if first_heading is not None:
        if _canonical_heading_text(first_heading) == _canonical_heading_text(title):
            lines[first_content_index] = f'## {title}'
            return '\n'.join(lines).rstrip() + '\n'

    lines.insert(first_content_index, f'## {title}')
    return '\n'.join(lines).rstrip() + '\n'


def _chapter_toc_label(chapter_id: Any, title: str) -> str:
    title_text = str(title or '').strip()
    if re.match(rf'^{re.escape(str(chapter_id))}\s*[.)]\s+', title_text):
        return title_text
    return f'Chapter {chapter_id}: {title_text}'


def _normalize_chapter_content(content: str) -> str:
    text = str(content or '').strip()
    if not text:
        return ''
    if '\\n' in text:
        text = text.replace('\\n', '\n')
    if "\\'" in text:
        text = text.replace("\\'", "'")
    if any(line.startswith('# ') for line in text.splitlines()):
        text = _demote_all_headings(text)
    return text.rstrip() + '\n'


_LOCAL_REFERENCE_PATTERNS = [
    re.compile(r'(?<![A-Za-z0-9_:/.-])(?:evidence|reports)/[^\s\])>]+',
               re.IGNORECASE),
    re.compile(r'(?<![A-Za-z0-9_:/.-])(?:final_report|draft|report)\.md\b',
               re.IGNORECASE),
    re.compile(r'\b(?:analysis|conclusion|note)_[0-9a-f]{6,}\.md\b',
               re.IGNORECASE),
    re.compile(r'\[[0-9a-f]{6,}\]', re.IGNORECASE),
    re.compile(r'\b(?:Note ID|Analysis ID|note_id|analysis_id)\b',
               re.IGNORECASE),
]


def _invalid_local_report_references(text: str) -> List[str]:
    """Return local evidence/analysis references that must not enter reports."""
    invalid: List[str] = []
    for pattern in _LOCAL_REFERENCE_PATTERNS:
        for match in pattern.finditer(text or ''):
            invalid.append(match.group(0))
    return list(dict.fromkeys(invalid))


def _validate_note_ids(_field_name: str, note_ids: List[str],
                       valid_note_ids: set[str]) -> List[str]:
    invalid = []
    for note_id in note_ids or []:
        if not isinstance(note_id, str) or note_id.strip() not in valid_note_ids:
            invalid.append(str(note_id))
    return invalid


def _invalid_cited_urls(cited_urls: Optional[List[str]]) -> List[str]:
    invalid = []
    for url in cited_urls or []:
        value = str(url or '').strip()
        if (not value or not re.match(r'^https?://', value, re.IGNORECASE)
                or _invalid_local_report_references(value)):
            invalid.append(value)
    return invalid


def _format_validation_error(message: str,
                             invalid_items: List[str]) -> Dict[str, Any]:
    return {
        'status': 'error',
        'message': message,
        'invalid_items': invalid_items[:20],
    }


def _render_outline_md(outline: Dict[str, Any]) -> str:
    """Render outline as Markdown."""
    lines = [f"# {outline.get('title', 'Report Outline')}", '']

    for ch in outline.get('chapters', []):
        status_icon = {
            'pending': '⏳',
            'in_progress': '🔄',
            'completed': '✅'
        }.get(ch.get('status', 'pending'), '⏳')

        lines.append(
            f"## Chapter {ch['chapter_id']}: {ch['title']} {status_icon}")

        if ch.get('goals'):
            lines.append('')
            lines.append('**Goals:**')
            for goal in ch['goals']:
                lines.append(f'- {goal}')

        if ch.get('sections_description'):
            lines.append('')
            lines.append('**Chapter structure:**')
            lines.append(ch['sections_description'])

        if ch.get('candidate_evidence'):
            lines.append('')
            lines.append(
                f"**Related evidence:** {', '.join(ch['candidate_evidence'])}")

        lines.append('')

    return '\n'.join(lines)


def _render_outline_progress_md(outline: Dict[str, Any]) -> str:
    """Render a concise outline progress view for terminal logs."""
    chapters = outline.get('chapters', [])
    total = len(chapters)
    completed = sum(1 for ch in chapters if ch.get('status') == 'completed')
    in_progress = sum(1 for ch in chapters
                      if ch.get('status') == 'in_progress')
    pending = total - completed - in_progress

    lines = [f"# {outline.get('title', 'Report Outline')}", '']
    lines.append(
        f'Progress: {completed}/{total} completed | {in_progress} in progress | {pending} pending'
    )
    lines.append('')
    lines.append('## Chapters')
    lines.append('')

    for ch in chapters:
        status = ch.get('status', 'pending')
        status_icon = {
            'pending': '⏳',
            'in_progress': '🔄',
            'completed': '✅'
        }.get(status, '⏳')
        lines.append(
            f"- {status_icon} Chapter {ch['chapter_id']}: {ch['title']}")

    lines.append('')
    return '\n'.join(lines)


class ReportTool(ToolBase):
    """
    Report generation tool for DeepResearch Reporter agent.

    Provides structured workflow for generating reports:
    1. commit_outline - Generate chapter structure bound to evidence
    2. update_outline - Update chapter information
    3. get_status - Check progress
    4. prepare_chapter_bundle - Prepare evidence for chapter writing
    5. commit_chapter - Write chapter content
    6. commit_conflict - Record evidence conflicts
    7. finalize_report - Assemble final report

    Storage:
    - reports/outline.json: Chapter structure with evidence bindings
    - reports/outline.md: Markdown render of outline
    - reports/chapters/chapter_XX.md: Chapter content
    - reports/chapters/chapter_XX_meta.json: Chapter metadata
    - reports/conflict.json: Recorded conflicts
    - reports/report.md: Final assembled report
    """

    SERVER_NAME = 'report_generator'

    def __init__(self, config, **kwargs):
        super().__init__(config)
        tool_cfg = getattr(getattr(config, 'tools'), 'report_generator')
        self.exclude_func(tool_cfg)

        # Configurable paths
        self._reports_dir = getattr(tool_cfg, 'reports_dir',
                                    'reports') if tool_cfg else 'reports'
        self._evidence_dir = getattr(tool_cfg, 'evidence_dir',
                                     'evidence') if tool_cfg else 'evidence'
        self._lock_subdir = getattr(tool_cfg, 'lock_subdir',
                                    '.locks') if tool_cfg else '.locks'

    async def connect(self) -> None:
        """Initialize directory structure."""
        _ensure_dir(self.output_dir)
        _ensure_dir(
            os.path.join(self.output_dir, self._reports_dir, 'chapters'))
        _ensure_dir(os.path.join(self.output_dir, self._lock_subdir))

    def _paths(self) -> Dict[str, str]:
        return {
            'outline_json':
            os.path.join(self.output_dir, self._reports_dir, 'outline.json'),
            'outline_md':
            os.path.join(self.output_dir, self._reports_dir, 'outline.md'),
            'outline_progress_md':
            os.path.join(self.output_dir, self._reports_dir,
                         'outline_progress.md'),
            'chapters_dir':
            os.path.join(self.output_dir, self._reports_dir, 'chapters'),
            'conflict_json':
            os.path.join(self.output_dir, self._reports_dir, 'conflict.json'),
            'draft_md':
            os.path.join(self.output_dir, self._reports_dir, 'draft.md'),
            'report_md':
            os.path.join(self.output_dir, self._reports_dir, 'report.md'),
            'final_report_md':
            os.path.join(self.output_dir, 'final_report.md'),
            'evidence_index':
            os.path.join(self.output_dir, self._evidence_dir, 'index.json'),
            'evidence_notes_dir':
            os.path.join(self.output_dir, self._evidence_dir, 'notes'),
            'lock_dir':
            os.path.join(self.output_dir, self._lock_subdir),
        }

    async def _get_tools_inner(self) -> Dict[str, Any]:
        tools: Dict[str, List[Tool]] = {
            self.SERVER_NAME: [
                Tool(
                    tool_name='commit_outline',
                    server_name=self.SERVER_NAME,
                    description=
                    ('Generate the report outline with chapter structure. '
                     'Each chapter must be bound to relevant evidence (note_ids). '
                     'Ensures all evidence is covered by at least one chapter.'
                     ),
                    parameters={
                        'type': 'object',
                        'properties': {
                            'title': {
                                'type': 'string',
                                'description': 'Title of the report.',
                            },
                            'summary': {
                                'type':
                                'string',
                                'description':
                                ('Short report summary to place immediately after the title '
                                 'and before the table of contents. State what failed, '
                                 'why it matters, and what the report will prove.'),
                            },
                            'chapters': {
                                'type': 'array',
                                'description': 'List of chapter definitions.',
                                'items': {
                                    'type':
                                    'object',
                                    'properties': {
                                        'title': {
                                            'type': 'string',
                                            'description': 'Chapter title.',
                                        },
                                        'goals': {
                                            'type':
                                            'array',
                                            'items': {
                                                'type': 'string'
                                            },
                                            'description':
                                            'Main objectives of this chapter.',
                                        },
                                        'sections_description': {
                                            'type':
                                            'string',
                                            'description':
                                            ('Detailed section-by-section plan for this chapter '
                                             '(NOT a single-sentence summary). '
                                             'Write subsections as a numbered list in markdown. '
                                             'For EACH subsection include: '
                                             '(a) subsection title, (b) 2-5 bullet key '
                                             'points / questions to answer, '
                                             '(c) expected output form: narrative synthesis is required; '
                                             'optionally add an artifact '
                                             '(e.g., table/checklist) to support the narrative.'
                                             ),
                                        },
                                        'candidate_evidence': {
                                            'type':
                                            'array',
                                            'items': {
                                                'type': 'string'
                                            },
                                            'description':
                                            'List of note_ids relevant to this chapter.',
                                        },
                                    },
                                    'required': [
                                        'title', 'goals',
                                        'sections_description',
                                        'candidate_evidence'
                                    ],
                                },
                            },
                        },
                        'required': ['title', 'summary', 'chapters'],
                        'additionalProperties': False,
                    },
                ),
                Tool(
                    tool_name='prepare_chapter_bundle',
                    server_name=self.SERVER_NAME,
                    description=
                    ('Prepare metadata and evidence content for writing a specific chapter. '
                     'Returns the chapter info with full evidence details for review. '
                     'Call this before commit_chapter to review evidence quality.'
                     ),
                    parameters={
                        'type': 'object',
                        'properties': {
                            'chapter_id': {
                                'type': 'integer',
                                'description': 'The chapter number (1-based).',
                            },
                            'relevant_evidence': {
                                'type':
                                'array',
                                'items': {
                                    'type': 'string'
                                },
                                'description':
                                ('List of note_ids maybe used in this chapter. '
                                 'The note_ids in this list will be loaded for review.'
                                 ),
                            },
                            # 'need_raw_chunks': {
                            #     'type': 'boolean',
                            #     'description': 'Whether to load raw chunk content for deeper analysis.',
                            #     'default': False,
                            # },
                        },
                        'required': ['chapter_id', 'relevant_evidence'],
                        'additionalProperties': False,
                    },
                ),
                Tool(
                    tool_name='commit_chapter',
                    server_name=self.SERVER_NAME,
                    description=
                    ('Write the content of a specific chapter. '
                     'The chapter will be saved as chapter_XX.md and status updated to completed.'
                     ),
                    parameters={
                        'type':
                        'object',
                        'properties': {
                            'chapter_id': {
                                'type': 'integer',
                                'description': 'The chapter number (1-based).',
                            },
                            'reranked_evidence': {
                                'type':
                                'array',
                                'items': {
                                    'type': 'string'
                                },
                                'description':
                                'List of note_ids reranked and chosen for this chapter.',
                            },
                            'content': {
                                'type':
                                'string',
                                'description':
                                ('The markdown content of the chapter. '
                                 'The content should include citations to the resources used in this chapter.'
                                 'Make sure the content is based on the reranked evidence.'
                                 ),
                            },
                            'cited_urls': {
                                'type':
                                'array',
                                'items': {
                                    'type': 'string'
                                },
                                'description':
                                ('List of resource urls actually cited in this chapter.'
                                 'Keep the same order as cited in content.'),
                            },
                        },
                        # Keep schema consistent with Python signature (reranked_evidence has no default)
                        'required': [
                            'chapter_id', 'reranked_evidence', 'content',
                            'cited_urls'
                        ],
                        'additionalProperties':
                        False,
                    },
                ),
                Tool(
                    tool_name='load_chunk',
                    server_name=self.SERVER_NAME,
                    description=
                    ('Load raw chunk content when evidence summaries are insufficient. '
                     'Reserved for future implementation.'),
                    parameters={
                        'type': 'object',
                        'properties': {
                            'chunk_ids': {
                                'type': 'array',
                                'items': {
                                    'type': 'string'
                                },
                                'description': 'List of chunk IDs to load.',
                            },
                        },
                        'required': ['chunk_ids'],
                        'additionalProperties': False,
                    },
                ),
                Tool(
                    tool_name='commit_conflict',
                    server_name=self.SERVER_NAME,
                    description=
                    'Record a conflict or contradiction between evidence.',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'description': {
                                'type': 'string',
                                'description': 'Description of the conflict.',
                            },
                            'evidence_ids': {
                                'type':
                                'array',
                                'items': {
                                    'type': 'string'
                                },
                                'description':
                                'Note IDs involved in the conflict.',
                            },
                            'chapter_id': {
                                'type': 'integer',
                                'description':
                                'Optional: Related chapter number.',
                            },
                            'resolution': {
                                'type':
                                'string',
                                'description':
                                'Optional: How the conflict was resolved.',
                            },
                        },
                        'required': ['description', 'evidence_ids'],
                        'additionalProperties': False,
                    },
                ),
                Tool(
                    tool_name='update_outline',
                    server_name=self.SERVER_NAME,
                    description=
                    'Update a specific chapter in the outline (title, goals, or evidence bindings).',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'chapter_id': {
                                'type': 'integer',
                                'description': 'The chapter number to update.',
                            },
                            'updates': {
                                'type': 'object',
                                'description':
                                'Fields to update (title, goals, sections_description, candidate_evidence).',
                                'properties': {
                                    'title': {
                                        'type': 'string',
                                        'description': 'Title of the chapter.',
                                    },
                                    'goals': {
                                        'type':
                                        'array',
                                        'items': {
                                            'type': 'string'
                                        },
                                        'description':
                                        'Main objectives of this chapter.',
                                    },
                                    'sections_description': {
                                        'type':
                                        'string',
                                        'description':
                                        ('Detailed section-by-section plan for '
                                         'this chapter (NOT a single-sentence summary). '
                                         'Write subsections as a numbered list in markdown. '
                                         'For EACH subsection include: '
                                         '(a) subsection title, (b) 2-5 bullet key '
                                         'points / questions to answer, '
                                         '(c) expected output form: narrative synthesis '
                                         'is required; optionally add an artifact '
                                         '(e.g., table/checklist) to support the narrative.'
                                         ),
                                    },
                                    'candidate_evidence': {
                                        'type':
                                        'array',
                                        'items': {
                                            'type': 'string'
                                        },
                                        'description':
                                        'List of note_ids relevant to this chapter.',
                                    },
                                },
                            },
                        },
                        'required': ['chapter_id', 'updates'],
                        'additionalProperties': False,
                    },
                ),
                Tool(
                    tool_name='assemble_draft',
                    server_name=self.SERVER_NAME,
                    description=
                    ('Assemble all chapters into a draft (draft.md) with TOC and references. '
                     'Returns the draft path along with a summary of recorded conflicts. '
                     'The model should then review the draft and conflicts and call finalize_report.'
                     ),
                    parameters={
                        'type': 'object',
                        'properties': {
                            'include_toc': {
                                'type': 'boolean',
                                'description':
                                'Whether to include table of contents.',
                                'default': True,
                            },
                            'include_references': {
                                'type': 'boolean',
                                'description':
                                'Whether to include references section.',
                                'default': True,
                            },
                        },
                        'required': [],
                        'additionalProperties': False,
                    },
                ),
                Tool(
                    tool_name='finalize_report',
                    server_name=self.SERVER_NAME,
                    description=
                    ('Create the final report artifact after chapters are complete. '
                     'If final_content is omitted, this promotes the assembled draft after '
                     'removing draft markers and normalizing escaped newlines. '
                     'Writes both reports/report.md and final_report.md.'),
                    parameters={
                        'type': 'object',
                        'properties': {
                            'final_content': {
                                'type':
                                'string',
                                'description':
                                ('Optional complete final report markdown. '
                                 'When omitted, the current assembled draft is used.'),
                            },
                        },
                        'required': [],
                        'additionalProperties': False,
                    },
                ),
                Tool(
                    tool_name='get_status',
                    server_name=self.SERVER_NAME,
                    description='Get current report generation progress.',
                    parameters={
                        'type': 'object',
                        'properties': {},
                        'required': [],
                        'additionalProperties': False,
                    },
                ),
            ]
        }
        return tools

    async def call_tool(self, server_name: str, *, tool_name: str,
                        tool_args: dict) -> str:
        return await getattr(self, tool_name)(**(tool_args or {}))

    def _load_outline(self, paths: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Load outline.json."""
        return _safe_read_json(paths['outline_json'])

    def _save_outline(self,
                      paths: Dict[str, str],
                      outline: Dict[str, Any],
                      render: bool = True) -> None:
        """Save outline.json and render outline.md."""
        outline['updated_at'] = _now_iso()
        _write_text(paths['outline_json'], _json_dumps(outline))
        _write_text(paths['outline_md'], _render_outline_md(outline))
        _write_text(paths['outline_progress_md'],
                    _render_outline_progress_md(outline))

        if render:
            render_markdown_todo(
                paths['outline_progress_md'],
                title='CURRENT REPORT OUTLINE',
                use_pager=False)

    def _load_evidence_index(self, paths: Dict[str, str]) -> Dict[str, Any]:
        """Load evidence index."""
        data = _safe_read_json(paths['evidence_index'])
        if data is None:
            return {'notes': {}}
        return data

    def _load_note_content(self, paths: Dict[str, str],
                           note_id: str) -> Optional[Dict[str, Any]]:
        """Load a single note's full content from markdown file."""
        note_path = os.path.join(paths['evidence_notes_dir'],
                                 f'note_{note_id}.md')
        if not os.path.exists(note_path):
            return None

        with open(note_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse note from markdown
        note = _parse_note_from_md(content, note_id)
        note['raw_content'] = content
        return note

    def _load_conflict(self, paths: Dict[str, str]) -> Dict[str, Any]:
        """Load conflict.json."""
        data = _safe_read_json(paths['conflict_json'])
        if data is None:
            return {'updated_at': _now_iso(), 'conflicts': []}
        return data

    def _save_conflict(self, paths: Dict[str, str],
                       conflict: Dict[str, Any]) -> None:
        """Save conflict.json."""
        conflict['updated_at'] = _now_iso()
        _write_text(paths['conflict_json'], _json_dumps(conflict))

    async def commit_outline(
        self,
        title: str,
        summary: str,
        chapters: List[Dict[str, Any]],
    ) -> str:
        """Generate report outline with chapter structure."""
        paths = self._paths()
        _ensure_dir(paths['chapters_dir'])
        _ensure_dir(paths['lock_dir'])

        # Load evidence index to validate coverage
        evidence_index = self._load_evidence_index(paths)
        all_note_ids = set(evidence_index.get('notes', {}).keys())
        invalid_candidates: List[str] = []
        for ch in chapters:
            invalid_candidates.extend(
                _validate_note_ids('candidate_evidence',
                                   ch.get('candidate_evidence', []),
                                   all_note_ids))
        if invalid_candidates:
            return _json_dumps(
                _format_validation_error(
                    'Outline candidate_evidence must contain existing evidence note_ids only. '
                    'Do not use analysis IDs, local file paths, or missing notes.',
                    list(dict.fromkeys(invalid_candidates))))

        # Build outline
        outline_chapters = []
        covered_evidence = set()

        for idx, ch in enumerate(chapters, start=1):
            candidate = ch.get('candidate_evidence', [])
            covered_evidence.update(candidate)

            outline_chapters.append({
                'chapter_id':
                idx,
                'title':
                ch.get('title', f'Chapter {idx}'),
                'goals':
                ch.get('goals', []),
                'sections_description':
                ch.get('sections_description', ''),
                'candidate_evidence':
                candidate,
                'status':
                'pending',
            })

        # Check coverage
        uncovered = all_note_ids - covered_evidence
        coverage_warning = None
        if uncovered:
            coverage_warning = (
                f'Warning: the following evidence is not covered by any chapter: {list(uncovered)}'
            )

        outline = {
            'title': title,
            'summary': str(summary or '').strip(),
            'created_at': _now_iso(),
            'updated_at': _now_iso(),
            'chapters': outline_chapters,
        }

        with file_lock(paths['lock_dir'], 'report_outline'):
            self._save_outline(paths, outline)

        result = {
            'status': 'ok',
            'outline_path': os.path.relpath(paths['outline_json'],
                                            self.output_dir),
            'chapters_count': len(outline_chapters),
            'total_evidence': len(all_note_ids),
            'covered_evidence': len(covered_evidence),
        }

        if coverage_warning:
            result['warning'] = coverage_warning

        return _json_dumps(result)

    async def prepare_chapter_bundle(
        self,
        chapter_id: int,
        relevant_evidence: List[str],
        need_raw_chunks: bool = False,
    ) -> str:
        """Prepare chapter metadata and evidence content."""
        paths = self._paths()
        _ensure_dir(paths['lock_dir'])

        # Load outline
        outline = self._load_outline(paths)
        if outline is None:
            return _json_dumps({
                'status':
                'error',
                'message':
                'Outline not created yet. Please call commit_outline first.'
            })

        # Find chapter
        chapter = None
        for ch in outline.get('chapters', []):
            if ch['chapter_id'] == chapter_id:
                chapter = ch
                break

        if chapter is None:
            return _json_dumps({
                'status': 'error',
                'message': f'Chapter {chapter_id} not found.'
            })

        # Load evidence content
        evidence_index = self._load_evidence_index(paths)
        notes_meta = evidence_index.get('notes', {})
        all_note_ids = set(notes_meta.keys())
        requested_note_ids = list(
            dict.fromkeys(
                list(chapter.get('candidate_evidence', []))
                + list(relevant_evidence or [])))
        invalid_requested = _validate_note_ids('relevant_evidence',
                                               requested_note_ids,
                                               all_note_ids)
        if invalid_requested:
            return _json_dumps(
                _format_validation_error(
                    'Chapter evidence must contain existing evidence note_ids only. '
                    'Do not use analysis IDs, local file paths, or missing notes.',
                    list(dict.fromkeys(invalid_requested))))

        notes_content = []
        seen_note_ids = set()
        for note_id in chapter.get('candidate_evidence', []):
            meta = notes_meta.get(note_id, {})
            note_data = self._load_note_content(paths, note_id)

            if note_data:
                notes_content.append({
                    'note_id':
                    note_id,
                    'title':
                    meta.get('title', note_data.get('title', '')),
                    'content':
                    note_data.get('content', ''),
                    'contradicts':
                    note_data.get('contradicts', ''),
                    'summary':
                    meta.get('summary', note_data.get('summary', '')),
                    'sources':
                    meta.get('sources', note_data.get('sources', [])),
                    'quality_score':
                    meta.get('quality_score', note_data.get('quality_score')),
                    'tags':
                    meta.get('tags', note_data.get('tags', [])),
                })
            else:
                return _json_dumps({
                    'status':
                    'error',
                    'message':
                    f'Evidence note {note_id} is indexed but its note file is missing.',
                })

            seen_note_ids.add(note_id)

        for note_id in relevant_evidence or []:
            if note_id not in seen_note_ids:
                meta = notes_meta.get(note_id, {})
                note_data = self._load_note_content(paths, note_id)

                if note_data:
                    notes_content.append({
                        'note_id':
                        note_id,
                        'title':
                        meta.get('title', note_data.get('title', '')),
                        'content':
                        note_data.get('content', ''),
                        'contradicts':
                        note_data.get('contradicts', ''),
                        'summary':
                        meta.get('summary', note_data.get('summary', '')),
                        'sources':
                        meta.get('sources', note_data.get('sources', [])),
                        'quality_score':
                        meta.get('quality_score',
                                 note_data.get('quality_score')),
                        'tags':
                        meta.get('tags', note_data.get('tags', [])),
                    })
                else:
                    return _json_dumps({
                        'status':
                        'error',
                        'message':
                        f'Evidence note {note_id} is indexed but its note file is missing.',
                    })

        # Build meta
        meta = {
            'chapter_id': chapter_id,
            'chapter_title': chapter['title'],
            'chapter_goals': chapter.get('goals', []),
            'sections_description': chapter.get('sections_description', ''),
            'candidate_evidence': requested_note_ids,
            'need_raw_chunks': need_raw_chunks,
            'loaded_chunks': [],
            'created_at': _now_iso(),
        }

        # Save meta.json
        meta_path = os.path.join(paths['chapters_dir'],
                                 f'chapter_{chapter_id:02d}_meta.json')
        with file_lock(paths['lock_dir'], f'chapter_{chapter_id}_meta'):
            _write_text(meta_path, _json_dumps(meta))

        # Update outline status
        chapter['status'] = 'in_progress'
        with file_lock(paths['lock_dir'], 'report_outline'):
            self._save_outline(paths, outline)

        return _json_dumps({
            'status':
            'ok',
            'chapter_id':
            chapter_id,
            'chapter_title':
            chapter['title'],
            'chapter_goals':
            chapter.get('goals', []),
            'evidence_count':
            len(notes_content),
            'meta_path':
            os.path.relpath(meta_path, self.output_dir),
            'notes_content':
            notes_content,
        })

    async def commit_chapter(
        self,
        chapter_id: int,
        reranked_evidence: List[str],
        content: str,
        cited_urls: Optional[List[str]] = None,
    ) -> str:
        """Write chapter content."""
        paths = self._paths()
        _ensure_dir(paths['lock_dir'])

        # Validate outline exists
        outline = self._load_outline(paths)
        if outline is None:
            return _json_dumps({
                'status': 'error',
                'message': 'Outline not created yet.'
            })

        # Find chapter before mutating report state.
        chapter_found = False
        chapter_title = ''
        target_chapter: Optional[Dict[str, Any]] = None
        for ch in outline.get('chapters', []):
            if ch['chapter_id'] == chapter_id:
                chapter_found = True
                chapter_title = ch['title']
                target_chapter = ch
                break

        if not chapter_found:
            return _json_dumps({
                'status': 'error',
                'message': f'Chapter {chapter_id} not found.'
            })

        evidence_index = self._load_evidence_index(paths)
        all_note_ids = set(evidence_index.get('notes', {}).keys())
        invalid_evidence = _validate_note_ids('reranked_evidence',
                                              reranked_evidence or [],
                                              all_note_ids)
        if invalid_evidence:
            return _json_dumps(
                _format_validation_error(
                    'reranked_evidence must contain existing evidence note_ids only. '
                    'Do not use analysis IDs, local file paths, or missing notes.',
                    list(dict.fromkeys(invalid_evidence))))

        invalid_urls = _invalid_cited_urls(cited_urls)
        if invalid_urls:
            return _json_dumps(
                _format_validation_error(
                    'cited_urls must be public http(s) source URLs. '
                    'Do not cite local evidence files, analysis files, or note IDs.',
                    invalid_urls))

        content = _ensure_chapter_heading(
            _normalize_chapter_content(content),
            chapter_title,
        )
        invalid_content_refs = _invalid_local_report_references(content)
        if invalid_content_refs:
            return _json_dumps(
                _format_validation_error(
                    'Chapter content contains local evidence, analysis, or note placeholders. '
                    'Use only numbered citations that point to public source URLs.',
                    invalid_content_refs))

        # Write chapter file
        chapter_path = os.path.join(paths['chapters_dir'],
                                    f'chapter_{chapter_id:02d}.md')

        with file_lock(paths['lock_dir'], f'chapter_{chapter_id}'):
            _write_text(chapter_path, content)

        if target_chapter is not None:
            target_chapter['status'] = 'completed'
            target_chapter['reranked_evidence'] = list(reranked_evidence or [])
            target_chapter['cited_urls'] = list(cited_urls or [])

        with file_lock(paths['lock_dir'], 'report_outline'):
            self._save_outline(paths, outline)

        meta_path = os.path.join(paths['chapters_dir'],
                                 f'chapter_{chapter_id:02d}_meta.json')
        meta = _safe_read_json(meta_path)
        meta = meta if isinstance(meta, dict) else {}
        meta['reranked_evidence'] = list(reranked_evidence or [])
        meta['cited_urls'] = list(cited_urls or [])
        with file_lock(paths['lock_dir'], f'chapter_{chapter_id}_meta'):
            _write_text(meta_path, _json_dumps(meta))

        return _json_dumps({
            'status':
            'ok',
            'chapter_id':
            chapter_id,
            'chapter_title':
            chapter_title,
            'path':
            os.path.relpath(chapter_path, self.output_dir),
            'content_length':
            len(content),
            'reranked_evidence':
            reranked_evidence or [],
            'cited_urls':
            cited_urls or [],
        })

    async def load_chunk(self, chunk_ids: List[str]) -> str:
        """Load raw chunk content. Reserved for future implementation."""
        return _json_dumps({
            'status': 'not_implemented',
            'message':
            'Chunk storage not enabled in this version. Use evidence notes directly.',
            'chunk_ids': chunk_ids,
        })

    async def commit_conflict(
        self,
        description: str,
        evidence_ids: List[str],
        chapter_id: Optional[int] = None,
        resolution: Optional[str] = None,
    ) -> str:
        """Record evidence conflict."""
        paths = self._paths()
        _ensure_dir(paths['lock_dir'])

        conflict_id = f'conflict_{uuid.uuid4().hex[:6]}'

        conflict_entry = {
            'id': conflict_id,
            'description': description,
            'evidence_ids': evidence_ids,
            'chapter_id': chapter_id,
            'resolution': resolution,
            'created_at': _now_iso(),
        }

        with file_lock(paths['lock_dir'], 'conflict'):
            conflicts = self._load_conflict(paths)
            conflicts['conflicts'].append(conflict_entry)
            self._save_conflict(paths, conflicts)

        return _json_dumps({
            'status':
            'ok',
            'conflict_id':
            conflict_id,
            'total_conflicts':
            len(conflicts['conflicts']),
            'conflict_path':
            os.path.relpath(paths['conflict_json'], self.output_dir),
        })

    async def update_outline(
        self,
        chapter_id: int,
        updates: Dict[str, Any],
    ) -> str:
        """Update a chapter in the outline."""
        paths = self._paths()
        _ensure_dir(paths['lock_dir'])

        with file_lock(paths['lock_dir'], 'report_outline'):
            outline = self._load_outline(paths)
            if outline is None:
                return _json_dumps({
                    'status': 'error',
                    'message': 'Outline not created yet.'
                })

            chapter_found = False
            for ch in outline.get('chapters', []):
                if ch['chapter_id'] == chapter_id:
                    if 'candidate_evidence' in updates:
                        evidence_index = self._load_evidence_index(paths)
                        all_note_ids = set(
                            evidence_index.get('notes', {}).keys())
                        invalid_evidence = _validate_note_ids(
                            'candidate_evidence',
                            updates.get('candidate_evidence') or [],
                            all_note_ids)
                        if invalid_evidence:
                            return _json_dumps(
                                _format_validation_error(
                                    'candidate_evidence must contain existing evidence note_ids only. '
                                    'Do not use analysis IDs, local file paths, or missing notes.',
                                    list(dict.fromkeys(invalid_evidence))))
                    if 'title' in updates:
                        ch['title'] = updates['title']
                    if 'goals' in updates:
                        ch['goals'] = updates['goals']
                    if 'sections_description' in updates:
                        ch['sections_description'] = updates[
                            'sections_description']
                    if 'candidate_evidence' in updates:
                        ch['candidate_evidence'] = updates[
                            'candidate_evidence']
                    chapter_found = True
                    break

            if not chapter_found:
                return _json_dumps({
                    'status':
                    'error',
                    'message':
                    f'Chapter {chapter_id} not found.'
                })

            self._save_outline(paths, outline)

        return _json_dumps({
            'status': 'ok',
            'chapter_id': chapter_id,
            'updates_applied': list(updates.keys()),
        })

    async def assemble_draft(
        self,
        include_toc: bool = True,
        include_references: bool = True,
    ) -> str:
        """Assemble draft from all chapters."""
        paths = self._paths()

        outline = self._load_outline(paths)
        if outline is None:
            return _json_dumps({
                'status': 'error',
                'message': 'Outline not created yet.'
            })

        # Collect chapter contents
        chapters_content = []
        missing_chapters = []
        invalid_chapters: List[Dict[str, Any]] = []

        for ch in outline.get('chapters', []):
            chapter_path = os.path.join(paths['chapters_dir'],
                                        f"chapter_{ch['chapter_id']:02d}.md")
            if os.path.exists(chapter_path):
                with open(chapter_path, 'r', encoding='utf-8') as f:
                    content = _ensure_chapter_heading(
                        _normalize_chapter_content(f.read()),
                        ch['title'],
                    )
                    invalid_refs = _invalid_local_report_references(content)
                    invalid_urls = _invalid_cited_urls(
                        ch.get('cited_urls', []))
                    if invalid_refs or invalid_urls:
                        invalid_chapters.append({
                            'chapter_id':
                            ch['chapter_id'],
                            'invalid_references':
                            invalid_refs[:20],
                            'invalid_cited_urls':
                            invalid_urls[:20],
                        })
                    chapters_content.append({
                        'id':
                        ch['chapter_id'],
                        'title':
                        ch['title'],
                        'content':
                        content,
                        'reranked_evidence':
                        ch.get('reranked_evidence', []),
                        'cited_urls':
                        ch.get('cited_urls', []),
                    })
            else:
                missing_chapters.append(ch['chapter_id'])

        if missing_chapters:
            return _json_dumps({
                'status':
                'error',
                'message':
                f'The following chapters are not completed yet: {missing_chapters}',
            })

        if invalid_chapters:
            return _json_dumps({
                'status':
                'error',
                'message':
                'Cannot assemble draft because one or more chapters contain local evidence, analysis, or note placeholders.',
                'invalid_chapters':
                invalid_chapters,
            })

        # Build draft
        draft_lines = [
            f"# {outline.get('title', 'Research Report')} (Draft)", ''
        ]
        summary_text = str(outline.get('summary') or '').strip()
        if summary_text:
            draft_lines.append(summary_text)
            draft_lines.append('')

        # Table of contents
        if include_toc:
            draft_lines.append('## Table of Contents')
            draft_lines.append('')
            for ch in chapters_content:
                anchor = ch['title'].replace(' ', '-').lower()
                draft_lines.append(
                    f"- [{_chapter_toc_label(ch['id'], ch['title'])}](#{anchor})")
            draft_lines.append('')

        # Chapters
        for ch in chapters_content:
            draft_lines.append(ch['content'])
            draft_lines.append('')

        # References
        if include_references:
            # evidence_index = self._load_evidence_index(paths)
            # notes_meta = evidence_index.get('notes', {})

            cited_urls = []
            seen_urls = set()
            for ch in chapters_content:
                for url in (ch.get('cited_urls') or []):
                    if url not in seen_urls:
                        cited_urls.append(url)
                        seen_urls.add(url)

            all_cited = set()
            for ch in chapters_content:
                all_cited.update(ch.get('reranked_evidence', []))

            # Also include candidate evidence if no explicit citations
            if not all_cited:
                for ch in outline.get('chapters', []):
                    all_cited.update(ch.get('candidate_evidence', []))

            if cited_urls:
                draft_lines.append('## References')
                draft_lines.append('')

                ref_idx = 1
                for url in cited_urls:
                    draft_lines.append(f'[{ref_idx}] {url}')
                    ref_idx += 1

                draft_lines.append('')

        # Write draft
        draft_content = '\n'.join(draft_lines)
        _write_text(paths['draft_md'], draft_content)

        # Load conflicts for summary
        conflicts_data = self._load_conflict(paths)
        conflicts_list = conflicts_data.get('conflicts', [])
        conflicts_summary = []
        for c in conflicts_list:
            conflicts_summary.append({
                'id': c.get('id'),
                'description': c.get('description'),
                'chapter_id': c.get('chapter_id'),
                'resolution': c.get('resolution'),
            })

        return _json_dumps({
            'status':
            'ok',
            'draft_path':
            os.path.relpath(paths['draft_md'], self.output_dir),
            'chapters_count':
            len(chapters_content),
            'content_length':
            len(draft_content),
            'conflicts_count':
            len(conflicts_list),
            'conflicts_summary':
            conflicts_summary,
            'next_step_reminder':
            ('Review the draft and conflicts, then generate the final report. '
             'Note: the draft cannot be used as the final report; '
             'do not replace report content with references or pointers to other content or files '
             '(e.g., "details are in chapter_2.md", "see draft.md for more details").'
             ),
        })

    async def finalize_report(self, final_content: Optional[str] = None) -> str:
        """Write the final report from supplied content or the assembled draft."""
        paths = self._paths()

        if final_content and final_content.strip():
            report_content = _normalize_report_content(final_content)
        else:
            if not os.path.exists(paths['draft_md']):
                draft_result = json.loads(await self.assemble_draft())
                if draft_result.get('status') != 'ok':
                    return _json_dumps(draft_result)
            with open(paths['draft_md'], 'r', encoding='utf-8') as f:
                report_content = _normalize_report_content(f.read())

        if not report_content.strip():
            return _json_dumps({
                'status': 'error',
                'message': 'Final report content is empty.',
            })

        invalid_refs = _invalid_local_report_references(report_content)
        if invalid_refs:
            return _json_dumps(
                _format_validation_error(
                    'Final report contains local evidence, analysis, or note placeholders. '
                    'Rewrite the report with only numbered citations to public sources.',
                    invalid_refs))

        _write_text(paths['report_md'], report_content)
        _write_text(paths['final_report_md'], report_content)

        return _json_dumps({
            'status': 'ok',
            'report_path': os.path.relpath(paths['report_md'], self.output_dir),
            'final_report_path': os.path.relpath(paths['final_report_md'], self.output_dir),
            'content_length': len(report_content),
        })

    async def get_status(self) -> str:
        """Get current report generation progress."""
        paths = self._paths()

        outline = self._load_outline(paths)
        conflicts = self._load_conflict(paths)

        if outline is None:
            return _json_dumps({
                'status':
                'not_started',
                'outline_exists':
                False,
                'chapters': [],
                'conflicts_count':
                len(conflicts.get('conflicts', [])),
            })

        chapters_status = []
        for ch in outline.get('chapters', []):
            chapter_path = os.path.join(paths['chapters_dir'],
                                        f"chapter_{ch['chapter_id']:02d}.md")
            chapters_status.append({
                'chapter_id':
                ch['chapter_id'],
                'title':
                ch['title'],
                'status':
                ch.get('status', 'pending'),
                'file_exists':
                os.path.exists(chapter_path),
                'candidate_evidence_count':
                len(ch.get('candidate_evidence', [])),
            })

        completed = sum(1 for ch in chapters_status
                        if ch['status'] == 'completed')
        total = len(chapters_status)

        return _json_dumps({
            'status':
            'in_progress' if completed < total else 'completed',
            'outline_exists':
            True,
            'report_title':
            outline.get('title', ''),
            'progress':
            f'{completed}/{total}',
            'chapters':
            chapters_status,
            'conflicts_count':
            len(conflicts.get('conflicts', [])),
            'draft_exists':
            os.path.exists(paths['draft_md']),
            'report_exists':
            os.path.exists(paths['report_md']),
            'final_report_exists':
            os.path.exists(paths['final_report_md']),
        })
