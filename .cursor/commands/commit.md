---
description: Create a single git commit using Conventional Commits with a detailed description of all changes
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

1. Pick **one** primary `type` and optional `scope` from the rule that matches the most impactful change. Since this will be a single commit for all current changes, choose the type that best represents the overall update (prefer `feat`/`fix` over `chore`).
2. Draft the message: 
   - **Subject:** English, imperative, lowercase after the colon, no trailing period, ~50 chars, max 72.
   - **Body (REQUIRED):** Must have a blank line after the subject. You **MUST** write a highly detailed explanation covering **all** modifications included in this commit. Use bullet points to list the changes. Explain both **what** was changed and **why**. Wrap text at 72 characters.
3. Stage **all** modified, added, and deleted files (e.g., using `git add .`). Do not split changes into multiple commits; group everything into exactly ONE commit.
4. Commit with a HEREDOC (no `-i`, no `--no-verify`):

```bash
git commit -m "$(cat <<'EOF'
<type>(<optional-scope>): <description>

<detailed body with bullet points explaining ALL changes: what and why>
EOF
)"
