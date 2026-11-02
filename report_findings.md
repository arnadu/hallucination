# Does "do not hallucinate" make an LLM more reliable?

## Problem

Language models sometimes state false things with confidence. A folk remedy is to add "do not hallucinate" to the prompt. We test whether that sentence, or a longer set of instructions ("say so instead of guessing", "point out a false premise"), makes a model more reliable than a plain prompt.

"Reliable" means the right behavior for the question, and a judge model passes or fails each answer. There are three kinds of question:
- **Answerable:** the answer is available. Reliable = the correct answer.
- **Unanswerable:** the answer is missing or the thing does not exist. Reliable = say so instead of guessing.
- **False premise:** the question assumes something untrue. Reliable = correct the assumption.

A prompt can fail by inventing too much (*too bold*) or by refusing too much (*too cautious*), so all three kinds are included.

## Conclusions

Numbers are reliable rates pooled over 118 questions (5 trials each); intervals are 95% bootstrap CIs.

| Model | Plain prompt | "Do not hallucinate" | Longer instructions |
|---|---:|---:|---:|
| gemma-3-4b-it | 67.8% | 67.9% | 75.9% |
| gemma-4-31b-it, no reasoning | 94.3% | 95.7% | 97.6% |
| gemma-4-31b-it, reasoning on | 96.4% | 97.9% | 99.2% |

1. **The model matters far more than the prompt.** The gap between the two models (67.8% vs 94.3%) is much larger than any prompt effect. (The models differ in generation as well as size.)
2. **Longer instructions help, and this is the most credible finding.** Against the plain prompt: +8.1 points [+1.2, +15.2] for gemma-3-4b, +3.3 [+1.4, +5.7] for gemma-4-31b, +2.8 [+0.7, +5.2] with reasoning. The direction is the same for both models and both question sets.
3. **"Do not hallucinate" alone helps little and inconsistently.** gemma-3-4b: +0.1 [-5.0, +5.3], hiding a split: it helped closed-book (+6.6) and hurt with a source document (-6.3). gemma-4-31b: about +1.5, borderline significant.
4. **The gain is on questions that cannot be answered.** On those, longer instructions added +26.8 points (4b) and about +10 (31b). On answerable and false-premise questions they added nothing measurable.
5. **Prompting reduces fabrication but does not remove it.** On unknowable closed-book questions, gemma-3-4b still invented an answer 64% of the time with the best prompt (95% with the plain one); gemma-4-31b 10% (20% plain).
6. **The small model pays a price.** With a source document, gemma-3-4b under the longer instructions started refusing answerable and false-premise questions it could have answered (about 15% of them). The larger model did not.
7. **Reasoning adds about 2 points at every prompt and does not change the ranking of prompts.** This comparison is confounded with provider (see Limitations).

## What was tested

- **Open-book (v3, 61 questions):** the model gets a fictional fact sheet and a question, and must stay within the sheet. **Closed-book (v4, 57 questions):** the model gets only the question, which may ask about something invented (a paper, an award, a pandas function), so the right answer is "I don't know that".
- **Prompts:** plain; plain + "do not hallucinate"; a longer instruction set (admit uncertainty, correct false premises).
- **Models:** google/gemma-3-4b-it, and google/gemma-4-31b-it without and with reasoning forced on (`--reasoning high`, pinned to one provider). A probe showed that without a setting no provider returned any reasoning for gemma-4.
- **Judge:** openai/gpt-oss-120b, 3 passes, majority vote. Five trials per question, temperature 0.7.
- **Statistics:** per-question reliable rates pooled over both question sets, bootstrap CIs resampling questions (`pooled_stats.py`; also the viewer's Reliability tab).

## Judge audit

A human (the author) labelled 100 stratified answers blind.
- Weighted agreement with the judge: **93%**. The judge rarely passed a bad answer (about 3% of passes).
- It failed answers a human would pass in about 30% of its fails, concentrated on a few questions with strict or ambiguous ground truths (M16, M13, A14, P06). True reliability is probably about 2 points higher than reported.
- If the judge is harsher on hedged answers, the gain from longer instructions is understated. Not established.

## Limitations

1. Two models from one family, which differ in generation as well as size.
2. Questions and ground truths were written by an LLM; the invented v4 entities were not checked on the web.
3. The judge audit used one rater and 100 answers; the ground truths flagged in it were not corrected and the runs were not regraded.
4. Reasoning on/off is confounded with provider: the no-reasoning runs used mixed providers, the reasoning runs used CoreWeave only. A clean test needs `--reasoning off` on CoreWeave.
5. The analysis was designed after seeing early results, with no multiple-comparison correction; isolated significant cells are suggestive. gemma-4-31b is near the ceiling (94-99%), so its differences are small.

## Reproducing

```
.\.venv\Scripts\python.exe pooled_stats.py <v3 run> <v4 run> --out report_stats.md
```
Runs: gemma-3-4b `20261101T023119_693775Z` + `20261101T053954_671012Z`; gemma-4-31b `20261101T113934_269424Z` + `20261101T140919_612505Z`; with reasoning `20261101T192908_122904Z` + `20261101T224130_899749Z`. Do not pool runs of different models or reasoning settings.
Full tables: `report_stats.md`, `report_stats_gemma4.md`, `report_stats_gemma4_reasoning.md`. The previous long version is `report_findings_detailed.md`.
