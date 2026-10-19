# -*- coding: utf-8 -*-
"""Read user-visible Claude Code JSONL transcripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from ...harnesses.events import HarnessHistoryItem, HarnessHistoryKind
from ..models import SourceSession
from ._utils import parse_datetime

_MAX_TRANSCRIPT_BYTES = 64 * 1024 * 1024
_MAX_TRANSCRIPT_LINE_BYTES = 4 * 1024 * 1024
_MAX_TRANSCRIPT_LINES = 100_000
_MAX_HISTORY_ITEMS = 50_000
MAX_DISCOVERED_TRANSCRIPTS = 5_000
_MAX_SCANNED_TRANSCRIPTS = 10_000


@dataclass(frozen=True)
class ClaudeTranscript:
    """One user-visible Claude Code JSONL transcript."""

    source_id: str
    path: Path
    modified_at: datetime


def discover_claude_transcripts(claude_home: Path) -> list[ClaudeTranscript]:
    """Discover direct project transcripts, excluding Claude subagents."""
    projects = claude_home.expanduser() / "projects"
    if not projects.is_dir() or projects.is_symlink():
        return []
    candidates: list[ClaudeTranscript] = []
    scanned = 0
    try:
        project_dirs = projects.iterdir()
    except OSError:
        return []
    for project in project_dirs:
        if project.is_symlink() or not project.is_dir():
            continue
        for candidate in _transcripts_in(project):
            scanned += 1
            if scanned > _MAX_SCANNED_TRANSCRIPTS:
                break
            candidates.append(candidate)
        if scanned > _MAX_SCANNED_TRANSCRIPTS:
            break
    return sorted(
        candidates,
        key=lambda item: item.modified_at,
        reverse=True,
    )[:MAX_DISCOVERED_TRANSCRIPTS]


def _transcripts_in(directory: Path) -> Iterator[ClaudeTranscript]:
    try:
        for path in directory.glob("*.jsonl"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                modified_at = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
            except OSError:
                continue
            yield ClaudeTranscript(
                source_id=path.stem,
                path=path,
                modified_at=modified_at,
            )
    except OSError:
        return


def read_claude_transcript(
    transcript: ClaudeTranscript,
) -> tuple[SourceSession | None, list[str]]:
    """Parse a bounded Claude Code transcript into a portable conversation."""
    history: list[HarnessHistoryItem] = []
    warnings: list[str] = []
    timestamps: list[datetime] = []
    source_id = transcript.source_id
    cwd = ""
    first_user_text = ""
    malformed = 0
    records: list[dict[str, Any]] = []
    try:
        if transcript.path.stat().st_size > _MAX_TRANSCRIPT_BYTES:
            return None, [f"Skipped Claude session {source_id}: file is too large."]
        with transcript.path.open("rb") as stream:
            for line_number in range(1, _MAX_TRANSCRIPT_LINES + 1):
                line = stream.readline(_MAX_TRANSCRIPT_LINE_BYTES + 1)
                if not line:
                    break
                if len(line) > _MAX_TRANSCRIPT_LINE_BYTES:
                    return None, [
                        f"Skipped Claude session {source_id}: "
                        f"line {line_number} is too large.",
                    ]
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(raw, dict):
                    continue
                records.append(raw)
            else:
                if stream.read(1):
                    return None, [
                        f"Skipped Claude session {source_id}: too many JSONL lines.",
                    ]
    except OSError as exc:
        return None, [f"Could not read Claude session {source_id}: {exc}"]

    for raw in _current_branch_records(records):
        source_id = str(raw.get("sessionId") or source_id)
        cwd = cwd or _absolute_path(raw.get("cwd"))
        timestamp = parse_datetime(raw.get("timestamp"))
        if timestamp is not None:
            timestamps.append(timestamp)
        history.extend(_raw_history_items(raw))
        if len(history) > _MAX_HISTORY_ITEMS:
            return None, [
                f"Skipped Claude session {source_id}: history is too large.",
            ]
        if not first_user_text and raw.get("type") == "user":
            first_user_text = _content_text((raw.get("message") or {}).get("content"))

    if malformed:
        warnings.append(
            f"Claude session {source_id} ignored {malformed} malformed JSONL line(s).",
        )
    if not history:
        return None, [
            *warnings,
            f"Skipped Claude transcript {transcript.path.name} because it "
            "contains no supported conversation messages.",
        ]
    title = _title_from_text(first_user_text) or f"Claude {source_id[:8]}"
    return (
        SourceSession(
            source_id=source_id,
            title=title[:200],
            cwd=cwd,
            created_at=min(timestamps) if timestamps else None,
            updated_at=max(timestamps) if timestamps else transcript.modified_at,
            history=history,
            metadata={
                "source": "claude",
                "source_path": str(transcript.path),
            },
        ),
        warnings,
    )


def _current_branch_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the latest non-sidechain branch in Claude's message graph."""
    nodes = {
        str(record.get("uuid")): record
        for record in records
        if record.get("uuid") and not record.get("isSidechain", False)
    }
    if not nodes:
        return records
    parents = {
        str(record.get("parentUuid"))
        for record in nodes.values()
        if record.get("parentUuid")
    }
    if not parents:
        return records
    leaves = [node_id for node_id in nodes if node_id not in parents]
    if not leaves:
        return records
    leaf = next(
        (
            str(record.get("uuid"))
            for record in reversed(records)
            if str(record.get("uuid") or "") in leaves
        ),
        leaves[-1],
    )
    branch: list[dict[str, Any]] = []
    visited: set[str] = set()
    while leaf and leaf not in visited:
        visited.add(leaf)
        record = nodes.get(leaf)
        if record is None:
            break
        branch.append(record)
        leaf = str(record.get("parentUuid") or "")
    return list(reversed(branch))


def _raw_history_items(raw: dict[str, Any]) -> list[HarnessHistoryItem]:
    message = raw.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    message_type = str(raw.get("type") or message.get("role") or "")
    message_id = str(raw.get("uuid") or "")
    blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
    items: list[HarnessHistoryItem] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "")
        item_id = str(block.get("id") or block.get("tool_use_id") or "")
        if block_type == "text":
            text = str(block.get("text") or "")
            if text:
                items.append(
                    HarnessHistoryItem(
                        kind=(
                            HarnessHistoryKind.USER
                            if message_type == "user"
                            else HarnessHistoryKind.MESSAGE
                        ),
                        item_id=message_id,
                        text=text,
                    ),
                )
        elif block_type == "thinking":
            items.append(
                HarnessHistoryItem(
                    kind=HarnessHistoryKind.REASONING,
                    item_id=message_id,
                    text=str(block.get("thinking") or ""),
                ),
            )
        elif block_type == "tool_use":
            items.append(
                HarnessHistoryItem(
                    kind=HarnessHistoryKind.TOOL_CALL,
                    item_id=item_id,
                    tool_name=str(block.get("name") or "tool"),
                    data={"arguments": block.get("input") or {}},
                ),
            )
        elif block_type == "tool_result":
            items.append(
                HarnessHistoryItem(
                    kind=HarnessHistoryKind.TOOL_OUTPUT,
                    item_id=item_id,
                    text=_content_text(block.get("content")),
                    data={"is_error": bool(block.get("is_error"))},
                ),
            )
    return items


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _absolute_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    path = Path(value).expanduser()
    return str(path) if path.is_absolute() else ""


def _title_from_text(text: str) -> str:
    return " ".join(text.split())[:200]


__all__ = [
    "ClaudeTranscript",
    "MAX_DISCOVERED_TRANSCRIPTS",
    "discover_claude_transcripts",
    "read_claude_transcript",
]
