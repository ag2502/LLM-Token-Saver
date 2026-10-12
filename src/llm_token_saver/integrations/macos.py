"""macOS conveniences: notifications and a launchd agent for the weekly report."""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

LABEL = "io.github.ag2502.tokensaver.weekly"
_NOTIFY_SCRIPT = (
    "on run argv\ndisplay notification (item 1 of argv) with title (item 2 of argv)\nend run"
)


def is_macos() -> bool:
    return sys.platform == "darwin"


def notify(message: str, title: str = "Token Saver") -> bool:
    """Show a notification. Text travels as arguments, never spliced into the script."""
    if not is_macos():
        return False
    result = subprocess.run(
        ["osascript", "-e", _NOTIFY_SCRIPT, message, title],
        capture_output=True,
        check=False,
        timeout=10,
    )
    return result.returncode == 0


def launch_agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def weekly_report_plist(
    program: list[str], *, weekday: int, hour: int, minute: int, log_dir: Path
) -> bytes:
    """A launchd agent that runs `program` once a week (weekday 0 = Sunday, 1 = Monday)."""
    return plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": program,
            "StartCalendarInterval": {"Weekday": weekday, "Hour": hour, "Minute": minute},
            "StandardOutPath": str(log_dir / "weekly.log"),
            "StandardErrorPath": str(log_dir / "weekly.log"),
            "ProcessType": "Background",
        }
    )


def install_agent(plist: bytes, directory: Path | None = None) -> Path:
    """Write the agent and (re)load it for the current user."""
    directory = directory or launch_agents_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{LABEL}.plist"
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", domain, str(path)], capture_output=True, check=False)
    path.write_bytes(plist)
    subprocess.run(["launchctl", "bootstrap", domain, str(path)], capture_output=True, check=True)
    return path


def remove_agent(directory: Path | None = None) -> Path | None:
    path = (directory or launch_agents_dir()) / f"{LABEL}.plist"
    if not path.exists():
        return None
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", domain, str(path)], capture_output=True, check=False)
    path.unlink()
    return path
