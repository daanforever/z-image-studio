## 1. Subagent Model

- Whenever creating a subagent, explicitly select an available Grok-family model.
- Prefer the latest available Grok model (for example, Grok 4.6).
- Do not use a non-Grok model or inherit the parent model for subagents.

## 2. Strict Scope & Behavior Control (CRITICAL)
* **KISS & No Over-Engineering:** Always provide the absolute simplest solution (e.g., `c = a + b`). DO NOT create new classes, structs, templates, or files unless explicitly instructed.
* **DRY:** Extract repeated logic into protected methods/functions or base templates. If you write the same thing twice — refactor.
* **Hard Boundaries:** Treat the prompt as the absolute limit. Do not add speculative features, "future-proofing", or unsolicited optimizations.
* **No Drive-by Changes:** Modify ONLY the necessary lines. Do not fix formatting, consistency, or "refactor" surrounding code outside the exact scope of the task. 
* **Temporary Inconsistency > Scope Creep:** Do not update downstream consumers or other files just to "complete the picture".
* **Ask First:** If an out-of-scope change is mandatory to proceed, STOP. State the conflict in one sentence and wait for permission. Do not guess.


## 3. Response Format
* **Zero Fluff:** No intros ("Certainly!", "Here is..."), no conclusions, no conversational filler.
* **Format:** Use bullet points for explanations. If a single line of code suffices, provide only that.
* **On Success:** Respond **very short and concise** with only the information requested.
* **Do not explain basic concepts unless explicitly asked.**
* **Do not duplicate full plan (and review) in response.**
