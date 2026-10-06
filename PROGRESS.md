# Progress

## Current status

**Abandoned.** One submission on 2026-04-07 which returned no public score, and
the competition closed on 2024-08-12 - so it never could be scored. The entry
is unranked among 1,849 teams.

What the repo holds is the approach: Gemma2 QLoRA fine-tuning for 3-class
preference prediction, with training and inference notebooks for Kaggle GPUs.

## Last session (2026-10-06)

- Corrected the README, which presented "Expected Performance" without saying
  that nothing had ever been measured.
- Added this file.

## Open issues

- **The competition is closed, so no score can ever be obtained.** This is a
  record of an approach, not of a result, and the README now says so.
- Nothing in the repo is tested.

## Next steps (prioritized)

1. **Decide what this is for.** Either keep it as a documented approach and
   take it off the portfolio, or re-point the same pipeline at a live
   preference-prediction competition where it can actually score.

## Decisions & rationale

- Gemma2 QLoRA rather than a cross-encoder: the task is preference between two
  long responses, which needs the full context the decoder already handles.
