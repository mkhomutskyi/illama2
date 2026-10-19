"""Publication checks must catch metadata leaks without printing their values."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "public_scan", Path(__file__).resolve().parents[1] / "scripts/check_public_secrets.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
scan = _module.scan


def test_personal_email_in_commit_metadata_is_reported_without_value():
    email = "example-owner" + "@" + "gmail.com"
    findings = scan(f"Author: Example <{email}>\n".encode(), "commit-metadata")
    assert findings == [
        {"location": "commit-metadata", "line": 1, "rule": "personal-email"}
    ]
    assert email not in repr(findings)


def test_github_noreply_identity_is_allowed():
    assert scan(b"Author: owner <123+owner@users.noreply.github.com>", "metadata") == []


def test_credential_finding_never_contains_token():
    token = "ghp_" + "x" * 36
    findings = scan(token.encode(), "fixture")
    assert findings == [{"location": "fixture", "line": 1, "rule": "github-token"}]
    assert token not in repr(findings)
