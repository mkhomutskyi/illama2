# Security and public sharing

Illama is intended for local or controlled-network inference. The runner binds
inside its container to `0.0.0.0`; Docker's published ports may be reachable
from other machines. Limit exposure using host/network rules or an appropriate
authenticated gateway. Set `ILLAMA_API_KEY` privately when clients require
authentication. Do not expose an unauthenticated service to the Internet.

Keep `.env`, HF tokens, private SSH keys, downloaded model caches, user inputs,
raw operator logs, and server-specific deployment details outside tracked
files. `.env.example` contains variable names and empty/example values only.
Never put a real credential into an issue, screenshot or example command.
The CLI reports whether a key is configured without echoing its value.

`illama pull` may download licensed third-party weights and execute supported
model conversion code. Review model provenance, licenses and any remote-code
requirements before using a new repository. Image URL retrieval is subject to
SSRF checks; keep private-image URL access disabled unless intentionally needed
inside your trusted network.

## Before publishing

```bash
python scripts/check_public_secrets.py --history
git diff --cached --name-only
```

The scanner checks common private-key, provider-token and personal-email patterns
in tracked text, reachable Git blobs and commit metadata. It prints locations,
never matching values. Use a GitHub-provided no-reply email for published commits;
check both author and committer before pushing. Keep repository-local Git identity
settings aligned with the intended publishing account.
It cannot prove the absence of arbitrary passwords or every credential format.
Review staged files and history as well; do not add `docs/local` or private logs.

If a real secret is exposed, revoke/rotate it promptly. Removing the current
file does not remove it from Git history, forks, caches or previous downloads.
Contact the repository owner privately through GitHub's available private
security-reporting channel; if unavailable, arrange a private channel first.
Do not include the secret in a public issue.
