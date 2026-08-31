# Fix test

Here is the generalized prompt in English, focusing on the core concept of tests being tailored/fitted to mask broken production logic:

***

### Task: Audit and Fix Overfitted Tests and Hidden Codebase Bugs

**Context:**
Sometimes, tests are written or modified to pass against a buggy or logically incorrect codebase. Instead of fixing the underlying bug, the test assertions are loosened, tolerances are inflated, or critical validation steps are omitted. This creates a false sense of security where the test suite passes, but the production code remains broken.

**Your Goal:**
Analyze the test file `[FILENAME]` and the production code it covers. Identify where assertions have been loosely "fitted" to mask underlying logical or architectural bugs, fix the production code, and refactor the tests to be strictly robust.

---

### Instructions:

#### 1. Audit `[FILENAME]` for "Assertion Fitting"
* Examine the assertions in `[FILENAME]`. Look for red flags of tailored tests:
  * Unusually high or arbitrary tolerances (e.g., extremely large absolute/relative tolerances).
  * Trivial thresholds (e.g., asserting a difference is `> 1e-6` when a real failure would still pass this check).
  * Missing assertions for critical outputs, side effects, or edge cases.
  * Tests that verify trivial properties (like determinism or basic types) while ignoring the correctness of the actual data flow.

#### 2. Trace and Audit the Production Code
* Deeply analyze the production logic covered by `[FILENAME]`. 
* Look past the passing tests. Trace the data flow, state changes, and component interactions to find hidden bugs, logical discrepancies, or architectural flaws (e.g., incorrect batching, missing masks, state leakage, or improper boundary handling).

#### 3. Fix the Production Code
* Implement a clean, minimal, and correct fix for the underlying logical bug in the production codebase.
* Ensure the fix adheres to the existing coding style and architectural patterns of the project.

#### 4. Refactor the Test Suite in `[FILENAME]`
* Tighten the assertions in `[FILENAME]` so they are mathematically and logically robust.
* Replace loose tolerances with precise, invariant metrics (e.g., normalized differences, cosine similarity, or strict equality where appropriate).
* Ensure the tests are designed to fail if the bug is reintroduced, but pass cleanly with the correct implementation.

#### 5. Verify the Solution
* Run the updated test suite to verify that all tests pass and that no regressions have been introduced in related modules.
