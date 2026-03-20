# Convokit LLM Log - Team 055

## Iteration v2.8 (Conditional Blending)
- **Precision:** 0.7368
- **Recall:** 0.6667
- **F1:** 0.7000
- **Note:** Implementation of "Fast-Pass" logic for extreme scores. While it successfully protected latency for clear cases, it did not provide a metrics boost and slightly regressed F1 compared to v2.5.

## Iteration v2.7 (LLM Weight Shift + Coping Examples)
- **Precision:** 0.7568
- **Recall:** 0.6667
- **F1:** 0.7089
- **Note:** Attempted to fix the Precision drop from v2.6 by increasing LLM weight (0.7) and adding "safe coping" examples. It over-corrected, causing Recall to drop significantly as the LLM became too dismissive of risk.

## Iteration v2.6 (Subtle Ideation + 0.35 Threshold)
- **Precision:** 0.6316
- **Recall:** 0.8571
- **F1:** 0.7273
- **Note:** Successfully achieved high Recall (85%) by adding examples of "daily dark thoughts" to the prompt and lowering the threshold. However, Precision suffered as more safe "venting" was caught in the net.

## Iteration v2.5 (Lower Threshold 0.4) - **Peak Performance**
- **Precision:** 0.7111
- **Recall:** 0.7619
- **F1:** 0.7356
- **Note:** This was the best balance. The Grand Composite score with a slightly more sensitive threshold (0.4) captured enough risk without overwhelming the system with false positives.

## Iteration v2.4 (Grand Composite Scoring)
- **Precision:** 0.7941
- **Recall:** 0.6429
- **F1:** 0.7105
- **Note:** Moved to continuous score blending: `(Cassandre * 0.4) + (LLM * 0.6)`. Blending proved much more effective than binary "veto" logic used in v2.1-v2.3.

---

# Root Cause Analysis: The Plateau (v2.6 - v2.8)

In the final iterations, F1 stalled around **0.70 - 0.73**. Here is what went wrong:

1. **The "Assistant Risk" Leak:** Cassandre's multi-turn model accumulates risk across the whole conversation. In several cases, the *Assistant's* helpful turns (e.g., "Would you like to explore art clubs?") were flagged as high risk because the previous *User* turn was very dark. Blending these scores with the LLM helped, but didn't completely decouple the speaker roles.
2. **Few-Shot Over-Correction:** We found the LLM is highly sensitive to few-shot examples. 
    *   Adding "Subtle Ideation" examples (v2.6) made it great at Recall but too paranoid (low Precision).
    *   Adding "Coping" examples (v2.7) made it too strict (low Recall).
    *   We reached a point where fixing one metric directly broke the other because the boundary between "Safe Venting" and "Active Ideation" is linguistically very thin.
3. **Threshold vs. Weight Tug-of-War:** We attempted to optimize two variables simultaneously (thresholds and weights). This made it difficult to find the global optimum. v2.5 likely hit a local maximum that was difficult to exceed without more significant changes to how Cassandre handles speaker-specific risk.

### **Final Recommendation:**
Iteration **v2.5** is the strongest candidate for submission, offering the best balance of safety (76% recall) and accuracy (71% precision).
