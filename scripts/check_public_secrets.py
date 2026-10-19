"""Check tracked text and reachable Git history for credentials/personal emails.

Findings identify locations only; matching values are never printed. This is a
bounded pattern check, not a guarantee that every possible secret is detected.
"""

import argparse
import json
import pathlib
import re
import subprocess


PATTERNS = {
    "private-key": re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
    ),
    "github-token": re.compile(
        r"\bgh[pousr]_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{60,}\b"
    ),
    "huggingface-token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "openai-token": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "slack-token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{25,}\b"),
    "personal-email": re.compile(
        r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
        r"(?:gmail|googlemail|outlook|hotmail|live|yahoo|icloud|aol|protonmail|proton)\."
        r"(?:com|net|me)\b", re.IGNORECASE
    ),
}
FORBIDDEN_NAMES = re.compile(
    r"(^|/)(?:id_rsa|id_ed25519(?:_[^/]*)?|\.env|credentials|known_hosts)(?:$|\.)"
)


def git(*args):
    return subprocess.check_output(["git", *args])


def scan(content, location):
    if b"\x00" in content:
        return []
    text = content.decode("utf-8", errors="replace")
    findings = []
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append(
                {
                    "location": location,
                    "line": text.count("\n", 0, match.start()) + 1,
                    "rule": label,
                }
            )
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    paths = git("ls-files", "-z").decode().split("\x00")
    findings = []
    count = 0
    for name in filter(None, paths):
        if name.endswith((".env.example", ".env.sample", ".env.template")):
            pass
        elif FORBIDDEN_NAMES.search(name):
            findings.append({"location": name, "line": 0, "rule": "sensitive-filename"})
        p = pathlib.Path(name)
        if p.is_file():
            findings.extend(scan(p.read_bytes(), name))
            count += 1
    blobs = 0
    if args.history:
        findings.extend(scan(git("log", "--all", "--format=fuller"), "commit-metadata"))
        for line in git("rev-list", "--objects", "--all").decode().splitlines():
            oid, _, name = line.partition(" ")
            if git("cat-file", "-t", oid).strip() != b"blob":
                continue
            findings.extend(scan(git("cat-file", "blob", oid), f"{oid[:12]}:{name}"))
            blobs += 1
    print(
        json.dumps(
            {"tracked_files": count, "history_blobs": blobs, "findings": findings},
            indent=2,
        )
    )
    raise SystemExit(1 if findings else 0)


if __name__ == "__main__":
    main()
