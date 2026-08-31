# code-review
Review and optimize code instead of blindly editing or appending to it.

# Instructions
Review the relevant code with a senior developer's code review mindset. Prioritize bugs, behavioral regressions, security issues, and missing tests.

# Code Modification Philosophy
When evaluating a task, proposing solutions, or making changes (if explicitly asked), absolutely avoid "additive programming" (just appending new lines of code). Instead, you must apply the following sequence:
1. **Evaluate:** Analyze the existing codebase context and architecture first.
2. **Reuse (DRY):** Actively search for existing functions, types, components, or utilities. Reuse them instead of reinventing the wheel.
3. **Refactor & Delete:** If your new logic replaces old logic, explicitly remove the outdated code. Be bold in deleting dead, redundant, or verbose code rather than piling new logic on top of it.
4. **Optimize:** Aim for an elegant solution. Reduce overall code size, lower complexity, and consolidate repetitive logic.

# Output Format
Findings must be the primary focus, ordered by severity. When suggesting a fix, clearly state your approach (e.g., "I will reuse function X and delete the obsolete Y"). Do not make code changes directly unless the user explicitly asks for them.
