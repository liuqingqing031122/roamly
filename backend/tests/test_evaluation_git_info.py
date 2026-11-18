import shutil
import subprocess
from pathlib import Path

import pytest

from roamly.evaluation import git_info
from roamly.evaluation.git_info import GitInfo, read_git_info

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository whose root is above the `backend/` directory, as in this project."""
    root = tmp_path / "repo"
    (root / "backend").mkdir(parents=True)
    git(root, "init", "--quiet")
    (root / ".gitignore").write_text("eval_runs/\n", encoding="utf-8")
    (root / "backend" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", "initial")
    return root


@requires_git
def test_clean_repository_from_a_subdirectory(repo: Path) -> None:
    info = read_git_info(repo / "backend")

    assert info == GitInfo(commit=git(repo, "rev-parse", "HEAD"), dirty=False)
    assert info.commit is not None and len(info.commit) == 40


@requires_git
@pytest.mark.parametrize(
    "change",
    [
        lambda root: (root / "backend" / "app.py").write_text("changed\n", encoding="utf-8"),
        lambda root: (root / "backend" / "new.py").write_text("new\n", encoding="utf-8"),
    ],
    ids=["modified_file", "untracked_file"],
)
def test_uncommitted_changes_are_dirty(repo: Path, change: object) -> None:
    change(repo)  # type: ignore[operator]

    assert read_git_info(repo / "backend").dirty is True


@requires_git
def test_ignored_files_do_not_make_the_tree_dirty(repo: Path) -> None:
    (repo / "eval_runs").mkdir()
    (repo / "eval_runs" / "run.json").write_text("{}", encoding="utf-8")

    assert read_git_info(repo / "backend").dirty is False


@requires_git
def test_outside_a_repository_gives_null(tmp_path: Path) -> None:
    outside = tmp_path / "plain"
    outside.mkdir()

    assert read_git_info(outside) == GitInfo(commit=None, dirty=None)


@requires_git
def test_repository_without_commits_has_no_commit(tmp_path: Path) -> None:
    git(tmp_path, "init", "--quiet")

    assert read_git_info(tmp_path).commit is None


@pytest.mark.parametrize(
    "error", [FileNotFoundError("git"), subprocess.TimeoutExpired(["git"], 10)]
)
def test_git_unavailable_gives_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(git_info.subprocess, "run", fail)

    assert read_git_info(tmp_path) == GitInfo(commit=None, dirty=None)


@requires_git
def test_default_directory_finds_this_repository() -> None:
    # The tests run from a checkout of this project.
    assert read_git_info().commit is not None
