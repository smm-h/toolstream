"""Tests for toolstream._builtin_tools -- standalone tool implementations."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from toolstream._builtin_tools import (
    _resolve_path,
    bash,
    edit,
    glob_files,
    grep,
    read,
    write,
)


# ============================================================
# _resolve_path
# ============================================================


class TestResolvePath:
    def test_absolute_path_unchanged(self):
        result = _resolve_path("/absolute/path.txt", "/some/cwd")
        assert result == Path("/absolute/path.txt")

    def test_relative_resolved_against_cwd(self):
        result = _resolve_path("relative/file.txt", "/my/cwd")
        assert result == Path("/my/cwd/relative/file.txt")

    def test_bare_filename_resolved_against_cwd(self):
        result = _resolve_path("file.txt", "/my/cwd")
        assert result == Path("/my/cwd/file.txt")

    def test_dot_relative_resolved(self):
        result = _resolve_path("./sub/file.txt", "/my/cwd")
        assert result == Path("/my/cwd/sub/file.txt")


# ============================================================
# read
# ============================================================


class TestRead:
    async def test_basic_read(self, tmp_path: Path):
        f = tmp_path / "hello.txt"
        f.write_text("line one\nline two\nline three\n")
        result = await read(str(f), cwd=str(tmp_path))
        assert "1: line one" in result
        assert "2: line two" in result
        assert "3: line three" in result

    async def test_offset_and_limit(self, tmp_path: Path):
        f = tmp_path / "numbers.txt"
        f.write_text("\n".join(f"line {i}" for i in range(10)))
        result = await read(str(f), cwd=str(tmp_path), offset=2, limit=3)
        assert "3: line 2" in result
        assert "4: line 3" in result
        assert "5: line 4" in result
        assert "1: line 0" not in result
        assert "6: line 5" not in result

    async def test_relative_path(self, tmp_path: Path):
        f = tmp_path / "relative.txt"
        f.write_text("content here\n")
        result = await read("relative.txt", cwd=str(tmp_path))
        assert "1: content here" in result

    async def test_nonexistent_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            await read("/nonexistent/path.txt", cwd=str(tmp_path))


# ============================================================
# write
# ============================================================


class TestWrite:
    async def test_basic_write(self, tmp_path: Path):
        target = tmp_path / "output.txt"
        result = await write(str(target), "hello world", cwd=str(tmp_path))
        assert "Wrote" in result
        assert target.read_text() == "hello world"

    async def test_creates_parent_dirs(self, tmp_path: Path):
        target = tmp_path / "sub" / "dir" / "file.txt"
        await write(str(target), "nested", cwd=str(tmp_path))
        assert target.read_text() == "nested"

    async def test_reports_byte_count(self, tmp_path: Path):
        target = tmp_path / "count.txt"
        result = await write(str(target), "12345", cwd=str(tmp_path))
        assert "5 bytes" in result


# ============================================================
# bash
# ============================================================


class TestBash:
    async def test_basic_command(self, tmp_path: Path):
        result = await bash("echo hello", cwd=str(tmp_path), env=dict(os.environ))
        assert "hello" in result

    async def test_timeout_handling(self, tmp_path: Path):
        result = await bash("sleep 999", cwd=str(tmp_path), env=dict(os.environ), timeout=1)
        assert "timed out" in result

    async def test_output_truncation(self, tmp_path: Path):
        # Generate output larger than 50K chars
        result = await bash("python3 -c \"print('x' * 60000)\"", cwd=str(tmp_path), env=dict(os.environ))
        assert "truncated" in result
        assert len(result) < 60000

    async def test_truncation_reports_original_length(self, tmp_path: Path):
        # The truncation message should report the full original length
        result = await bash("python3 -c \"print('x' * 60000)\"", cwd=str(tmp_path), env=dict(os.environ))
        assert "60001 total chars" in result  # 60000 x's + newline from print

    async def test_stderr_captured(self, tmp_path: Path):
        result = await bash("echo err >&2", cwd=str(tmp_path), env=dict(os.environ))
        assert "err" in result

    async def test_uses_cwd(self, tmp_path: Path):
        result = await bash("pwd", cwd=str(tmp_path), env=dict(os.environ))
        assert str(tmp_path) in result

    async def test_env_passed_to_subprocess(self, tmp_path: Path):
        env = {"TOOLSTREAM_TEST_VAR": "test_value_42", "PATH": os.environ.get("PATH", "")}
        result = await bash("echo $TOOLSTREAM_TEST_VAR", cwd=str(tmp_path), env=env)
        assert "test_value_42" in result

    async def test_env_isolation(self, tmp_path: Path):
        # With an empty env, process env vars should NOT be inherited
        result = await bash("echo $PATH", cwd=str(tmp_path), env={})
        # PATH is not in the env dict, so the shell should expand $PATH to empty
        assert os.environ.get("PATH", "") not in result or result.strip() == ""


# ============================================================
# edit
# ============================================================


class TestEdit:
    async def test_basic_edit(self, tmp_path: Path):
        f = tmp_path / "edit_me.txt"
        f.write_text("foo bar baz")
        result = await edit(str(f), "bar", "qux", cwd=str(tmp_path))
        assert "Edited" in result
        assert f.read_text() == "foo qux baz"

    async def test_old_string_not_found(self, tmp_path: Path):
        f = tmp_path / "edit_me.txt"
        f.write_text("foo bar baz")
        result = await edit(str(f), "nonexistent", "qux", cwd=str(tmp_path))
        assert "Error" in result

    async def test_first_occurrence_only(self, tmp_path: Path):
        """Regression test: edit must replace only the FIRST occurrence."""
        f = tmp_path / "multi.txt"
        f.write_text("hello world\nhello again\nhello once more\n")
        result = await edit(str(f), "hello", "world", cwd=str(tmp_path))
        assert "Edited" in result
        content = f.read_text()
        assert content == "world world\nhello again\nhello once more\n"

    async def test_first_occurrence_three_on_same_line(self, tmp_path: Path):
        """Edit replaces only the first of three identical tokens."""
        f = tmp_path / "triple.txt"
        f.write_text("hello hello hello")
        await edit(str(f), "hello", "world", cwd=str(tmp_path))
        assert f.read_text() == "world hello hello"

    async def test_relative_path(self, tmp_path: Path):
        f = tmp_path / "rel.txt"
        f.write_text("old content")
        result = await edit("rel.txt", "old", "new", cwd=str(tmp_path))
        assert "Edited" in result
        assert f.read_text() == "new content"


# ============================================================
# grep
# ============================================================


class TestGrep:
    async def test_basic_grep(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("def hello():\n    pass\n")
        (tmp_path / "b.py").write_text("def goodbye():\n    pass\n")
        result = await grep("hello", path=str(tmp_path), cwd=str(tmp_path))
        assert "hello" in result
        assert "goodbye" not in result

    async def test_grep_with_include(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("target here\n")
        (tmp_path / "b.txt").write_text("target here\n")
        result = await grep(
            "target", path=str(tmp_path), cwd=str(tmp_path), include="*.py",
        )
        assert "a.py" in result
        assert "b.txt" not in result

    async def test_grep_without_include(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("found\n")
        (tmp_path / "b.txt").write_text("found\n")
        result = await grep("found", path=str(tmp_path), cwd=str(tmp_path))
        assert "a.py" in result
        assert "b.txt" in result

    async def test_grep_no_match(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("nothing relevant\n")
        result = await grep("nonexistent_pattern", path=str(tmp_path), cwd=str(tmp_path))
        # grep returns exit code 1 on no match
        assert "nonexistent_pattern" not in result


# ============================================================
# glob_files
# ============================================================


class TestGlobFiles:
    async def test_basic_glob(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = await glob_files(str(tmp_path / "*.py"), cwd=str(tmp_path))
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    async def test_glob_limit_100(self, tmp_path: Path):
        # Create 120 files, only first 100 should appear
        for i in range(120):
            (tmp_path / f"file_{i:03d}.txt").write_text("")
        result = await glob_files(str(tmp_path / "*.txt"), cwd=str(tmp_path))
        lines = [line for line in result.split("\n") if line.strip()]
        assert len(lines) == 100

    async def test_glob_recursive(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("")
        (tmp_path / "top.py").write_text("")
        result = await glob_files(str(tmp_path / "**" / "*.py"), cwd=str(tmp_path))
        assert "deep.py" in result
        assert "top.py" in result

    async def test_glob_no_matches(self, tmp_path: Path):
        result = await glob_files(str(tmp_path / "*.xyz"), cwd=str(tmp_path))
        assert result == ""

    async def test_relative_pattern_respects_cwd(self, tmp_path: Path):
        """Relative patterns should find files relative to cwd."""
        (tmp_path / "one.txt").write_text("")
        (tmp_path / "two.txt").write_text("")
        (tmp_path / "skip.py").write_text("")
        result = await glob_files("*.txt", cwd=str(tmp_path))
        assert "one.txt" in result
        assert "two.txt" in result
        assert "skip.py" not in result

    async def test_absolute_pattern_ignores_cwd(self, tmp_path: Path):
        """Absolute patterns should not be affected by cwd."""
        target_dir = tmp_path / "target"
        target_dir.mkdir()
        (target_dir / "found.txt").write_text("")
        decoy_dir = tmp_path / "decoy"
        decoy_dir.mkdir()
        (decoy_dir / "nope.txt").write_text("")
        result = await glob_files(str(target_dir / "*.txt"), cwd=str(decoy_dir))
        assert "found.txt" in result
        assert "nope.txt" not in result
