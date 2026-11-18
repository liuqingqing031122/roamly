"""Best-effort Git metadata identifying the code that produced an evaluation run."""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# The installed package lives inside the repository during development, so Git can
# find the repository root from here even when it is not the working directory.
PACKAGE_DIR = Path(__file__).resolve().parent

GIT_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class GitInfo:
    # None when the value cannot be determined; never guessed.
    commit: str | None = None
    dirty: bool | None = None


def read_git_info(directory: Path = PACKAGE_DIR) -> GitInfo:
    """Return the HEAD commit and whether the working tree has uncommitted changes.

    Untracked files that are not ignored count as changes. Any failure (Git missing,
    not a repository, no commits yet) yields None for that value instead of an error.
    """
    commit = _git(directory, "rev-parse", "HEAD")
    status = _git(directory, "status", "--porcelain")
    info = GitInfo(commit=commit or None, dirty=None if status is None else status != "")
    if info.commit is None or info.dirty is None:
        logger.info("Git metadata unavailable for %s; recording null", directory)
    return info


def _git(directory: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(directory), *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
