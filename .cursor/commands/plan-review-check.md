# plan-review-check
Verify plan review results, filter out false positives, and select the optimal solutions.

# Instructions

Act as a Senior Architect and secondary reviewer to validate the previous `plan-review` findings against the original project requirements and constraints. 

- **Filter & Validate:** Critically evaluate each raised concern, flaw, or missing step. Eliminate false positives, over-engineering, hallucinations, and unverified assumptions. Keep only genuinely confirmed logic gaps, real scope issues, and practical problems.
- **Select the Optimal Solution:** Where the previous review or original plan proposed multiple alternative solutions or approaches, analyze them and decisively select the *single, most optimal* solution based on simplicity, maintainability, and best practices.
- **Finalize:** Present a refined, final list of verified issues along with your chosen concrete actions. Output a clear, unambiguous, and actionable updated implementation plan without unnecessary complexity.

Never update the plan directly.
