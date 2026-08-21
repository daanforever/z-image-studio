# subagent-use

Follow these rules for codebase exploration and any interactions with testing:
- Use subagents (model: `composer-2.5`) STRICTLY to gather raw information (relevant files, definitions, dependencies, and command executions).
- Do not rely on subagents for reasoning. You (the main agent) must independently analyze the raw data they provide and draw your own conclusions.
- Never make assumptions about missing context. If the gathered information is incomplete, you must spawn additional subagents to find the missing details before making final decisions.
