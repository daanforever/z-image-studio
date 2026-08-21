---
description: Create a git commit using Conventional Commits
---

# commit

Invoking this command **is** an explicit request to create a git commit now.

## Source of truth

Before staging or committing, **read** `.cursor/rules/conventional-commits.mdc` and follow it **exactly**. That file is the only source of truth for commit message format, types, scopes, and examples. Do not invent a different style, and do not skip a requirement in that file.

Also follow the git safety protocol named in that rule (and the workspace git-commit user rules):

- Never update git config
- Never force-push, hard-reset, or other destructive git commands unless the user asked in this invocation
- Never `--no-verify`, `--no-gpg-sign`, or skip hooks unless the user asked
- Never `git commit --amend` unless every amend condition in the safety protocol is met
- Never interactive git (`-i`)
- Do not push unless the user asked
- Do not commit secrets (`.env`, credentials, keys). Warn if the user asked to include them
- If there is nothing to commit, do not create an empty commit
- If a hook rejects the commit, fix the issue and create a **new** commit (do not amend)

## Workflow

Run these in parallel:

1. `git status` — untracked, staged, unstaged
2. `git diff` and `git diff --staged` — all changes that would be committed
3. `git log -8 --oneline` — match this repo's subject style **within** Conventional Commits

Then:

1. Pick **one** `type` and optional `scope` from the rule that matches the primary intent. Prefer `feat`/`fix` over `chore` when behavior changes.
2. Draft the message from the rule: English, imperative, lowercase after the colon, no trailing period, ~50 characters, max 72. Optional body: blank line after the subject, wrap at 72, explain **why** not what.
3. Stage only the relevant files for this commit. Do not dump unrelated dirty files into it.
4. Commit with a HEREDOC (no `-i`, no `--no-verify`):

```bash
git commit -m "$(cat <<'EOF'
<type>(<optional-scope>): <description>

<optional body>
EOF
)"
```

5. Run `git status` after the commit and confirm it succeeded.

If the working tree is clean, say so and stop.
