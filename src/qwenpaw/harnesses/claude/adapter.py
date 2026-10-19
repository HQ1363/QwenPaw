# -*- coding: utf-8 -*-
"""Claude Code implementation of the third-party agent adapter."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ...utils.io_utils import read_json, write_json_atomic_async
from ..base import HarnessAdapter, HarnessOperationNotSupportedError
from ..capabilities import HarnessRuntimeCapabilities
from ..events import (
    HarnessAttachment,
    HarnessDiscoveredSkill,
    HarnessEvent,
    HarnessEventKind,
    HarnessProvider,
)
from .discovery import ClaudeBinaryResolution, resolve_claude_binary_info

_STATUS_TIMEOUT_SECONDS = 10
_CLAUDE_NOT_FOUND_MESSAGE = (
    "Claude Code runtime not found. Install the Claude Code CLI or provide "
    "its executable path."
)


class ClaudeAdapter(HarnessAdapter):
    """Run Claude Code CLI turns in a workspace-scoped conversation."""

    def __init__(self, state_dir: Path, binary: str | None = None) -> None:
        self._state_dir = state_dir
        self._session_path = state_dir / "claude_sessions.json"
        self._binary = binary
        self._sessions = self._load_sessions()
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._login_process: asyncio.subprocess.Process | None = None
        self._started_tools: dict[str, str] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}

    async def status(self) -> HarnessProvider:
        """Return Claude Code installation and authentication status."""
        resolution = self._resolution()
        if resolution is None:
            return HarnessProvider(
                id="claude",
                name="Claude Code",
                available=True,
                installed=False,
                error=_CLAUDE_NOT_FOUND_MESSAGE,
            )
        try:
            output, returncode = await self._run_cli("auth", "status")
        except (OSError, asyncio.TimeoutError) as exc:
            return self._provider(resolution, error=str(exc))
        output_lower = output.lower()
        if returncode and "not logged" not in output_lower:
            return self._provider(
                resolution,
                error=output.strip() or "Claude Code status check failed.",
            )
        authenticated = returncode == 0 and "not logged" not in output_lower
        return self._provider(resolution, authenticated=authenticated)

    async def start_login(self, device_code: bool = False) -> dict[str, Any]:
        """Start the provider-owned Claude Code login flow."""
        del device_code
        resolution = self._require_resolution()
        if (
            self._login_process is None
            or self._login_process.returncode is not None
        ):
            self._login_process = await asyncio.create_subprocess_exec(
                str(resolution.path),
                "auth",
                "login",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        return {
            "type": "external",
            "loginId": f"claude-{uuid.uuid4().hex}",
            "command": f"{resolution.path} auth login",
        }

    async def logout(self) -> None:
        """Explain Claude Code's interactive logout limitation."""
        raise HarnessOperationNotSupportedError(
            "Run `claude logout` in a terminal to sign out of Claude Code.",
        )

    async def discover_skills(
        self,
        cwd: Path,
    ) -> list[HarnessDiscoveredSkill]:
        """Discover project and user Skills without loading their contents."""
        discovered: list[HarnessDiscoveredSkill] = []
        seen: set[Path] = set()
        for root in (cwd / ".claude" / "skills", Path.home() / ".claude" / "skills"):
            if not root.is_dir() or root.is_symlink():
                continue
            try:
                candidates = root.iterdir()
            except OSError:
                continue
            for candidate in candidates:
                skill_file = candidate / "SKILL.md"
                if (
                    candidate.is_symlink()
                    or not candidate.is_dir()
                    or not skill_file.is_file()
                    or skill_file.is_symlink()
                    or candidate in seen
                ):
                    continue
                seen.add(candidate)
                discovered.append(
                    HarnessDiscoveredSkill(
                        name=candidate.name,
                        provider_id="claude",
                        source=str(candidate),
                    ),
                )
        return discovered

    async def run_command(
        self,
        *,
        session_id: str,
        command: str,
        arguments: str,
        cwd: Path,
        settings: dict[str, Any],
    ) -> list[HarnessEvent]:
        """Claude Code commands are submitted through the normal turn path."""
        del session_id, command, arguments, cwd, settings
        return await super().run_command(
            session_id="",
            command="",
            arguments="",
            cwd=Path(),
            settings={},
        )

    def run_turn(
        self,
        *,
        session_id: str,
        prompt: str,
        cwd: Path,
        settings: dict[str, Any],
        attachments: list[HarnessAttachment] | None = None,
    ) -> AsyncIterator[HarnessEvent]:
        """Run a Claude CLI turn and normalize its stream-json events."""
        if attachments:
            async def unsupported_attachments() -> AsyncIterator[HarnessEvent]:
                yield HarnessEvent(
                    kind=HarnessEventKind.ERROR,
                    text="Claude Code attachments are not supported yet.",
                )

            return unsupported_attachments()
        return self._run_turn(session_id, prompt, cwd, settings)

    async def reset_session(self, session_id: str) -> None:
        """Forget a local mapping without deleting Claude-owned history."""
        self._sessions.pop(session_id, None)
        await write_json_atomic_async(self._session_path, self._sessions)

    async def stop(self) -> None:
        """Terminate active Claude processes owned by this adapter."""
        for process in tuple(self._processes.values()):
            if process.returncode is None:
                process.terminate()
                await process.wait()
        self._processes.clear()
        if (
            self._login_process is not None
            and self._login_process.returncode is None
        ):
            self._login_process.terminate()
            await self._login_process.wait()
        self._login_process = None

    async def _run_turn(
        self,
        session_id: str,
        prompt: str,
        cwd: Path,
        settings: dict[str, Any],
    ) -> AsyncIterator[HarnessEvent]:
        resolution = self._require_resolution()
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            arguments = [
                "--output-format",
                "stream-json",
                "--verbose",
                "--include-partial-messages",
                "--permission-mode",
                str(settings.get("permission_mode") or "default"),
            ]
            existing_session = self._sessions.get(session_id)
            if existing_session:
                arguments.extend(["--resume", existing_session])
            model = str(settings.get("model") or "").strip()
            if model:
                arguments.extend(["--model", model])
            mcp_config, disallowed_tools = self._mcp_config(settings)
            if mcp_config:
                arguments.extend(
                    [
                        "--mcp-config",
                        json.dumps({"mcpServers": mcp_config}),
                        "--strict-mcp-config",
                    ],
                )
            if disallowed_tools:
                arguments.extend(["--disallowedTools", ",".join(disallowed_tools)])
            arguments.extend(["--print", prompt])
            process = await asyncio.create_subprocess_exec(
                str(resolution.path),
                *arguments,
                cwd=str(cwd),
                env=os.environ.copy(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._processes[session_id] = process
            original_source_session = self._sessions.get(session_id)
            completed = False
            try:
                assert process.stdout is not None
                async for line in process.stdout:
                    for event in self._events_from_line(line, session_id):
                        if event.kind == HarnessEventKind.COMPLETED:
                            completed = True
                        yield event
                stderr = await process.stderr.read() if process.stderr else b""
                returncode = await process.wait()
                if returncode and not completed:
                    yield HarnessEvent(
                        kind=HarnessEventKind.ERROR,
                        text=stderr.decode(errors="replace").strip()
                        or f"Claude Code exited with status {returncode}.",
                    )
                elif not completed:
                    yield HarnessEvent(kind=HarnessEventKind.COMPLETED)
                if self._sessions.get(session_id) != original_source_session:
                    await write_json_atomic_async(self._session_path, self._sessions)
            except asyncio.CancelledError:
                if process.returncode is None:
                    process.terminate()
                    await asyncio.shield(process.wait())
                raise
            finally:
                self._processes.pop(session_id, None)

    def _events_from_line(
        self,
        line: bytes,
        session_id: str,
    ) -> list[HarnessEvent]:
        """Convert one Claude stream-json record to normalized events."""
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []
        if payload.get("type") == "stream_event":
            event = payload.get("event")
            return self._stream_events(event) if isinstance(event, dict) else []
        if payload.get("type") == "assistant":
            return self._assistant_events(payload.get("message"))
        if payload.get("type") == "user":
            return self._tool_result_events(payload.get("message"))
        if payload.get("type") == "result":
            return self._result_events(payload, session_id)
        return []

    @staticmethod
    def _stream_events(event: dict[str, Any]) -> list[HarnessEvent]:
        if event.get("type") != "content_block_delta":
            return []
        delta = event.get("delta")
        if not isinstance(delta, dict):
            return []
        if delta.get("type") == "text_delta":
            return [
                HarnessEvent(
                    kind=HarnessEventKind.TEXT_DELTA,
                    text=str(delta.get("text") or ""),
                ),
            ]
        if delta.get("type") == "thinking_delta":
            return [
                HarnessEvent(
                    kind=HarnessEventKind.REASONING_DELTA,
                    text=str(delta.get("thinking") or ""),
                ),
            ]
        return []

    def _assistant_events(self, message: Any) -> list[HarnessEvent]:
        if not isinstance(message, dict):
            return []
        events: list[HarnessEvent] = []
        for block in message.get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            events.append(
                HarnessEvent(
                    kind=HarnessEventKind.TOOL_STARTED,
                    item_id=str(block.get("id") or ""),
                    tool_name=str(block.get("name") or "tool"),
                    data={"arguments": block.get("input") or {}},
                ),
            )
            self._started_tools[str(block.get("id") or "")] = str(
                block.get("name") or "tool",
            )
        return events

    def _tool_result_events(self, message: Any) -> list[HarnessEvent]:
        """Finish streamed tool calls when Claude returns their results."""
        if not isinstance(message, dict):
            return []
        content = message.get("content")
        blocks = content if isinstance(content, list) else []
        events: list[HarnessEvent] = []
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            item_id = str(block.get("tool_use_id") or "")
            events.append(
                HarnessEvent(
                    kind=HarnessEventKind.TOOL_COMPLETED,
                    item_id=item_id,
                    tool_name=self._started_tools.pop(item_id, "tool"),
                    text=self._content_text(block.get("content")),
                    data={"is_error": bool(block.get("is_error"))},
                ),
            )
        return events

    def _result_events(
        self,
        payload: dict[str, Any],
        session_id: str,
    ) -> list[HarnessEvent]:
        source_session = str(payload.get("session_id") or "")
        if source_session:
            self._sessions[session_id] = source_session
        if payload.get("is_error"):
            return [
                HarnessEvent(
                    kind=HarnessEventKind.ERROR,
                    text=str(payload.get("result") or "Claude Code failed."),
                ),
            ]
        return [HarnessEvent(kind=HarnessEventKind.COMPLETED)]

    def _mcp_config(
        self,
        settings: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        capabilities = settings.get("_runtime_capabilities")
        if not isinstance(capabilities, HarnessRuntimeCapabilities):
            return {}, []
        result: dict[str, Any] = {}
        disallowed_tools: list[str] = []
        for server in capabilities.mcp_servers:
            if server.tools is None and server.default_policy == "deny":
                continue
            denied_tools = [
                f"mcp__{server.name}__{name}"
                for name, policy in server.tool_policies.items()
                if policy == "deny"
            ]
            if server.default_policy == "deny":
                denied_tools.append(f"mcp__{server.name}__*")
            if server.default_policy == "deny":
                # Claude Code CLI 无法安全表达“默认拒绝但个别工具例外”的
                # MCP 策略，因此不注入该服务，避免未授权工具被调用。
                continue
            disallowed_tools.extend(denied_tools)
            if server.transport == "stdio":
                result[server.name] = {
                    "command": server.command,
                    "args": server.args,
                    "env": server.revealed_env(),
                }
            else:
                result[server.name] = {
                    "type": "sse" if server.transport == "sse" else "http",
                    "url": server.url,
                    "headers": server.revealed_headers(),
                }
        return result, disallowed_tools

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(block.get("text") or "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""

    async def _run_cli(self, *arguments: str) -> tuple[str, int]:
        resolution = self._require_resolution()
        process = await asyncio.create_subprocess_exec(
            str(resolution.path),
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            async with asyncio.timeout(_STATUS_TIMEOUT_SECONDS):
                stdout, _ = await process.communicate()
        except BaseException:
            if process.returncode is None:
                process.kill()
            await asyncio.shield(process.wait())
            raise
        return stdout.decode(errors="replace"), process.returncode or 0

    def _resolution(self) -> ClaudeBinaryResolution | None:
        return resolve_claude_binary_info(self._binary)

    def _require_resolution(self) -> ClaudeBinaryResolution:
        resolution = self._resolution()
        if resolution is None:
            raise RuntimeError(_CLAUDE_NOT_FOUND_MESSAGE)
        return resolution

    @staticmethod
    def _provider(
        resolution: ClaudeBinaryResolution,
        *,
        authenticated: bool = False,
        error: str | None = None,
    ) -> HarnessProvider:
        return HarnessProvider(
            id="claude",
            name="Claude Code",
            available=True,
            installed=True,
            authenticated=authenticated,
            runtime_path=str(resolution.path),
            runtime_source=resolution.source,
            error=error,
        )

    def _load_sessions(self) -> dict[str, str]:
        try:
            raw = read_json(self._session_path)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {
            str(key): str(value) for key, value in raw.items() if key and value
        }


__all__ = ["ClaudeAdapter"]
