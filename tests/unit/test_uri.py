from __future__ import annotations

import pytest

from seekvfs.exceptions import VFSError
from seekvfs.uri import SCHEME, is_dir_uri, parse_uri, with_scheme


def test_scheme_constant() -> None:
    assert SCHEME == "seekvfs://"


def test_parse_uri_basic() -> None:
    assert parse_uri("seekvfs://memories/foo.md") == "memories/foo.md"


def test_parse_uri_empty_path() -> None:
    assert parse_uri("seekvfs://") == ""


def test_parse_uri_preserves_case() -> None:
    assert parse_uri("seekvfs://Memories/Foo.md") == "Memories/Foo.md"


def test_parse_uri_invalid_scheme() -> None:
    with pytest.raises(VFSError):
        parse_uri("file:///etc/passwd")


def test_parse_uri_missing_scheme() -> None:
    with pytest.raises(VFSError):
        parse_uri("memories/foo.md")


def test_parse_uri_non_string() -> None:
    with pytest.raises(VFSError):
        parse_uri(123)  # type: ignore[arg-type]


def test_is_dir_uri_true_when_trailing_slash() -> None:
    assert is_dir_uri("seekvfs://memories/") is True


def test_is_dir_uri_false_without_trailing_slash() -> None:
    assert is_dir_uri("seekvfs://memories/foo.md") is False


def test_with_scheme_prepends() -> None:
    assert with_scheme("memories/foo.md") == "seekvfs://memories/foo.md"


def test_with_scheme_idempotent() -> None:
    assert with_scheme("seekvfs://foo") == "seekvfs://foo"
