# plan-review-update

Update the orchestrator-ready plan by incorporating the latest review feedback. Maintain a clean, strictly actionable format designed for an Orchestrator AI managing Subagents.

# Instructions
1. **Preserve High-Level Context**: Keep "Context", "Objective", and "Root cause" (if present) unchanged unless they explicitly conflict with new findings from the review.
2. **Update Subagent Tasks**: When revising the "Step-by-Step Orchestration Plan", ensure every added, kept, or modified step strictly adheres to the Subagent delegation format. Do not break the structure. Each step must clearly define:
   - **Task Objective**
   - **Target Files / Context**
   - **Acceptance Criteria**
3. **Maintain Blueprint Purity**: Do not add conversational filler, status markers (e.g., "Done already", "Task 1 completed", "Still broken"), or scratchpad notes. Do not output a changelog of what you changed. Simply output the *complete, revised blueprint* containing only the actionable steps the Orchestrator currently needs to execute or re-verify.
4. **Update Final Testing**: Revise the end-to-end verification steps if the review feedback alters the expected final state.

If any questions or ambiguities remain that might block the Orchestrator from successfully executing the updated plan, please list them clearly at the very end.
