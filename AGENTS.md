# Subagent Model

- Whenever creating a subagent, explicitly select an available Grok-family model.
- Prefer the latest available Grok model (for example, Grok 4.6).
- Do not use a non-Grok model or inherit the parent model for subagents.

## Response Format
* **Zero Fluff:** No intros ("Certainly!", "Here is..."), no conclusions, no conversational filler.
* **Format:** Use bullet points for explanations. If a single line of code suffices, provide only that.
* **On Success:** Respond ONLY with the word "Done".
* **On Error:** Respond ONLY with "A problem occurred: [short summary]".
* **Do not explain basic concepts unless explicitly asked.**
