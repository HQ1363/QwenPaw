# -*- coding: utf-8 -*-
"""Cross-platform discovery for the Claude Code CLI runtime."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ClaudeBinaryResolution:
    """Resolved Claude Code CLI path and discovery source."""

    path: Path
    source: str


def resolve_claude_binary(
    binary: str | None = None,
    **kwargs,
) -> Path | None:
    """Resolve only the Claude Code executable path."""
    resolution = resolve_claude_binary_info(binary, **kwargs)
    return resolution.path if resolution is not None else None


def resolve_claude_binary_info(
    binary: str | None = None,
    *,
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ClaudeBinaryResolution | None:
    """Resolve an explicit, environment-provided, or PATH CLI."""
    environment = environ if environ is not None else os.environ
    if binary:
        resolved = _resolve_configured(binary, platform_name=platform_name)
        return (
            ClaudeBinaryResolution(resolved, "configured")
            if resolved is not None
            else None
        )

    environment_binary = environment.get("CLAUDE_BINARY")
    if environment_binary:
        resolved = _resolve_configured(
            environment_binary,
            platform_name=platform_name,
        )
        return (
            ClaudeBinaryResolution(resolved, "environment")
            if resolved is not None
            else None
        )

    on_path = shutil.which("claude")
    if on_path:
        return ClaudeBinaryResolution(Path(on_path).resolve(), "path")
    return None


def _resolve_configured(
    binary: str,
    *,
    platform_name: str | None = None,
) -> Path | None:
    """Resolve a configured executable path without searching arbitrary roots."""
    candidate = Path(binary).expanduser()
    if not candidate.is_absolute():
        located = shutil.which(binary)
        candidate = Path(located) if located else candidate
    if not _is_executable(candidate, platform_name=platform_name):
        return None
    return candidate.resolve()


def _is_executable(
    path: Path,
    *,
    platform_name: str | None = None,
) -> bool:
    """Return whether a candidate is usable as a Claude CLI executable."""
    platform_value = platform_name or sys.platform
    if not path.is_file():
        return False
    if platform_value == "win32":
        return path.suffix.lower() in {".exe", ".cmd", ".bat"}
    return os.access(path, os.X_OK)


__all__ = [
    "ClaudeBinaryResolution",
    "resolve_claude_binary",
    "resolve_claude_binary_info",
]
