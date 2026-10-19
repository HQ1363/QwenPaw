# -*- coding: utf-8 -*-
"""Claude Code migration provider."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ...utils.io_utils import run_sync_io
from ..models import ProviderInventory, SourceLocation, SourceSession
from .base import (
    ProgressReporter,
    SessionReadBudget,
    make_inventory,
    progress_milestone,
)
from .claude_sessions import (
    MAX_DISCOVERED_TRANSCRIPTS,
    discover_claude_transcripts,
    read_claude_transcript,
)
from .external_state import discover_project_memory
from .locator import resolve_source_location

_READ_CONCURRENCY = 4
_SESSION_SCAN_TIMEOUT_SECONDS = 300


class ClaudeMigrationProvider:  # pylint: disable=too-few-public-methods
    """Read Claude Code's local, user-visible project conversations."""

    provider_id = "claude"

    def __init__(
        self,
        workspace: Any,
        *,
        claude_home: Path | None = None,
        source_location: SourceLocation | None = None,
    ) -> None:
        self._workspace = workspace
        if source_location is None:
            source_location = resolve_source_location(
                "claude",
                source_home=claude_home,
            )
        if claude_home is not None:
            source_location.data_home = str(claude_home.expanduser())
            source_location.data_home_source = "injected"
            source_location.data_home_exists = claude_home.is_dir()
        self._source_location = source_location
        self._claude_home = Path(source_location.data_home)

    async def inventory(
        self,
        *,
        limit: int,
        progress: ProgressReporter | None = None,
        include_sessions: bool = True,
        session_ids: set[str] | None = None,
    ) -> ProviderInventory:
        """Discover bounded Claude Code conversations without copying config."""
        records, memory_projects = await asyncio.gather(
            (
                run_sync_io(discover_claude_transcripts, self._claude_home)
                if include_sessions
                else _empty_records()
            ),
            run_sync_io(discover_project_memory, self._claude_home),
        )
        if session_ids is not None:
            records = [
                record for record in records if record.source_id in session_ids
            ]
        records = records[:limit]
        warnings = [
            "Claude Code settings, credentials, Skills, MCP configuration, "
            "and subagent transcripts are not imported. Project memory is "
            "imported separately.",
        ]
        if include_sessions and len(records) >= MAX_DISCOVERED_TRANSCRIPTS:
            warnings.append(
                "Claude transcript discovery reached its safety limit; older "
                "candidate files were not scanned.",
            )
        if include_sessions and progress is not None:
            await progress(
                f"发现 {len(records)} 个 Claude Code 会话候选文件，"
                "正在读取用户可见对话（最多 4 个同时读取）…",
            )

        sessions: list[SourceSession] = []
        budget = SessionReadBudget()
        completed = 0
        progress_lock = asyncio.Lock()

        async def read_one(
            record,
        ) -> tuple[SourceSession | None, list[str]]:
            nonlocal completed
            try:
                result = await run_sync_io(read_claude_transcript, record)
            except Exception as exc:  # pylint: disable=broad-except
                result = (
                    None,
                    [f"Could not read Claude session {record.source_id}: {exc}"],
                )
            async with progress_lock:
                completed += 1
                if progress is not None and progress_milestone(
                    completed,
                    len(records),
                ):
                    await progress(
                        f"Claude Code 会话读取进度：{completed}/{len(records)}",
                    )
            return result

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _SESSION_SCAN_TIMEOUT_SECONDS
        for start in range(0, len(records), _READ_CONCURRENCY):
            remaining = deadline - loop.time()
            if remaining <= 0:
                warnings.append(
                    "Claude session scan reached its 300 second time limit; "
                    "remaining histories were not read.",
                )
                break
            batch = records[start : start + _READ_CONCURRENCY]
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*(read_one(record) for record in batch)),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                warnings.append(
                    "Claude session scan reached its 300 second time limit; "
                    "remaining histories were not read.",
                )
                break
            for session, session_warnings in results:
                warnings.extend(session_warnings)
                if session is None:
                    continue
                budget_warning = budget.add(session)
                if budget_warning:
                    warnings.append(budget_warning)
                    if budget.exhausted:
                        break
                    continue
                sessions.append(session)
            if budget.exhausted:
                break

        return make_inventory(
            "claude",
            detected=self._claude_home.is_dir(),
            locator=str(self._claude_home),
            sessions=sessions,
            memory_projects=memory_projects,
            warnings=warnings,
        )


async def _empty_records() -> list[Any]:
    """Keep concurrent inventory discovery uniform when sessions are omitted."""
    return []


__all__ = ["ClaudeMigrationProvider"]
