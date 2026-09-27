"""
The release version is set in pyproject.toml and repeated in CITATION.cff
(which GitHub's "Cite this repository" button reads). Fail if a release bump
updates one and forgets the other; see docs/releasing.md.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _first_version(path, pattern):
    match = re.search(pattern, path.read_text(), flags=re.MULTILINE)
    assert match, f"no version line found in {path.name}"
    return match.group(1)


def test_citation_cff_version_matches_pyproject():
    citation = ROOT / "CITATION.cff"
    if not citation.exists():  # e.g. running from an sdist, which doesn't ship it
        pytest.skip("CITATION.cff not present")
    pyproject_version = _first_version(ROOT / "pyproject.toml", r'^version = "([^"]+)"')
    citation_version = _first_version(citation, r"^version: ['\"]?([^'\"\s]+)")
    assert citation_version == pyproject_version
