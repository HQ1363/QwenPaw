# -*- coding: utf-8 -*-
"""Claude Code transcript migration coverage."""

from __future__ import annotations

import json
from pathlib import Path

from qwenpaw.harnesses.events import HarnessHistoryKind
from qwenpaw.portability.providers.claude_sessions import (
    _current_branch_records,
    discover_claude_transcripts,
    read_claude_transcript,
)


def test_reads_user_visible_claude_transcript(tmp_path: Path) -> None:
    home = tmp_path / ".claude"
    project = home / "projects" / "-workspace"
    project.mkdir(parents=True)
    session_path = project / "session-1.jsonl"
    session_path.write_text(
        "\n".join(
            json.dumps(item)
            for item in [
                {
                    "type": "user",
                    "uuid": "user-1",
                    "sessionId": "session-1",
                    "cwd": "/workspace",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {"content": "Please inspect this project."},
                },
                {
                    "type": "assistant",
                    "uuid": "assistant-1",
                    "sessionId": "session-1",
                    "timestamp": "2026-01-01T00:01:00Z",
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": "I will inspect it."},
                            {
                                "type": "tool_use",
                                "id": "tool-1",
                                "name": "Read",
                                "input": {"path": "README.md"},
                            },
                            {"type": "text", "text": "The project is ready."},
                        ],
                    },
                },
                {
                    "type": "user",
                    "uuid": "tool-result-1",
                    "sessionId": "session-1",
                    "timestamp": "2026-01-01T00:01:01Z",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-1",
                                "content": "README content",
                            },
                        ],
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    subagent = project / "subagents"
    subagent.mkdir()
    (subagent / "child.jsonl").write_text("{}", encoding="utf-8")

    transcripts = discover_claude_transcripts(home)

    assert [item.source_id for item in transcripts] == ["session-1"]
    session, warnings = read_claude_transcript(transcripts[0])
    assert warnings == []
    assert session is not None
    assert session.title == "Please inspect this project."
    assert session.cwd == "/workspace"
    assert [item.kind for item in session.history] == [
        HarnessHistoryKind.USER,
        HarnessHistoryKind.REASONING,
        HarnessHistoryKind.TOOL_CALL,
        HarnessHistoryKind.MESSAGE,
        HarnessHistoryKind.TOOL_OUTPUT,
    ]


def test_keeps_only_the_current_claude_message_branch() -> None:
    records = [
        {"uuid": "root", "message": {"content": "First prompt"}},
        {
            "uuid": "obsolete",
            "parentUuid": "root",
            "message": {"content": "Discarded response"},
        },
        {
            "uuid": "current",
            "parentUuid": "root",
            "message": {"content": "Current response"},
        },
    ]

    branch = _current_branch_records(records)

    assert [item["uuid"] for item in branch] == ["root", "current"]
