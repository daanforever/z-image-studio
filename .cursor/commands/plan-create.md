# fix-plan
Draft an orchestrator-ready fix plan with clarified context, delegatable tasks, and testing strategy

# Instructions
Create a structured remediation plan based on the confirmed issues. **Crucially, this plan must be designed for execution by an Orchestrator AI. The Orchestrator's role is to spawn independent Subagents for each step of the plan and monitor/verify their successful completion.**

Always format your response with the following three mandatory sections:

1. "Context / Objective": Take the user's original message or reason that prompted this task and rephrase it into a clear, professional summary. This serves as the high-level mission for the Orchestrator.

2. "Step-by-Step Orchestration Plan": Provide an actionable plan prioritized by severity. Each step must be formulated as a self-contained task ready to be delegated to a Subagent. For each step, explicitly define:
   - **Task Objective**: Clear instructions on what the Subagent needs to achieve.
   - **Target Files / Context**: Specific files, functions, or dependencies the Subagent needs to work with.
   - **Acceptance Criteria**: What exactly the Orchestrator must check to confirm the Subagent succeeded before moving to the next step.
   *(Do not write the code implementation yourself unless explicitly requested. Focus on the blueprint for the Subagents).*

3. "Testing": Outline how the Orchestrator should verify the final state after all Subagents have finished their tasks. Suggest specific end-to-end test cases, integration checks, and edge cases to ensure all sub-tasks tie together correctly.
