# Does "do not hallucinate" make an LLM more reliable?

An experiment testing whether adding "do not hallucinate", or a longer set of instructions about admitting uncertainty and correcting false premises, makes a language model more reliable than a plain prompt. Models are called through OpenRouter and answers are graded by an LLM judge.

**Start with the report: [report_findings.md](report_findings.md).** A longer version with all tables is in [report_findings_detailed.md](report_findings_detailed.md).

## Contents

| Path | What it is |
|---|---|
| `hallucination.py` | Runner: asks the model each question under each prompt, then grades the answers with the judge |
| `pooled_stats.py` | Pooled reliability statistics across the two question sets (writes `report_stats*.md`) |
| `analysis.py` | Per-run analysis written to each run's `analysis.txt` |
| `viewer.html` | Standalone viewer: open it in a browser and load a result folder |
| `data/v3`, `data/v4` | Question sets: v3 is open-book (with a source document), v4 is closed-book |
| `results/` | The six runs used in the report (`answers.jsonl`, `metadata.json`, summaries) |
| `build_audit.py`, `audit_score.py`, `audit_sample.xlsx`, `audit_key.csv` | Human audit of the judge (100 answers) |

## Setup

```
python -m venv .venv
.\.venv\Scripts\pip install openai python-dotenv openpyxl
```
Put your key in a `.env` file (not committed): `OPENROUTER_API_KEY=...`

## Run an experiment

```
python hallucination.py --model google/gemma-4-31b-it --judge-model openai/gpt-oss-120b `
  --trials 5 --judge-passes 3 --data-dir data/v3 --provider-only CoreWeave --reasoning high
```
Useful options: `--data-dir` (question set), `--reasoning default|off|on|low|medium|high`, `--provider-only` / `--provider-ignore`, `--regrade <run folder>` (re-run only the judge). Each run writes a folder under `results/`.

## Reproduce the report statistics

```
python pooled_stats.py results\20261101T023119_693775Z results\20261101T053954_671012Z --out report_stats.md
```
The run folders for each model and reasoning setting are listed in the report. Do not pool runs of different models or reasoning settings.

## Notes

All names, companies and products in the question sets are fictional. Questions and ground truths were written by an LLM assistant and have not been independently verified.

## License

MIT, see [LICENSE](LICENSE). This covers the code, question sets and results in this repository.
