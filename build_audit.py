"""Build a blind sample of judged answers for a human audit of the LLM judge.

python build_audit.py <run dirs...> [--n 100] [--seed 0] [--out audit]
Writes audit_sample.xlsx (to label: no judge verdict shown) and audit_key.csv
(judge verdicts + stratum sizes, for scoring with audit_score.py).
"""
import argparse, csv, json, random
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

GROUP = {"answerable": "answer", "knowable_control": "answer", "missing_information": "hold_back",
         "unknowable": "hold_back", "false_premise": "correct", "false_premise_known": "correct"}

def stratum(r):
    g = r.get("grade") or {}
    votes = [x["grade"]["pass"] for x in r.get("judge_runs") or [] if x.get("grade")]
    if g.get("pass"):
        kind = "pass_split" if votes and not all(votes) else "pass"
    elif g.get("unnecessary_abstention"): kind = "fail_cautious"
    elif g.get("unsupported_claim"): kind = "fail_bold"
    else: kind = "fail_noflag"
    if kind == "pass": kind += "_" + GROUP.get(r["category"], "other")
    return kind

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="audit")
    a = ap.parse_args()
    pool = {}
    for run in a.runs:
        d = Path(run); meta = json.loads((d / "metadata.json").read_text(encoding="utf-8-sig"))
        for line in (d / "answers.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("error") or not r.get("grade"): continue
            r["_run"] = d.name; r["_data"] = meta["data_dir"]; r["_model"] = meta["model"]
            r["_reasoning"] = meta.get("reasoning", "default")
            pool.setdefault(stratum(r), []).append(r)
    # failures and split votes are rare and informative: take up to quota; fill the rest with passes
    quota = {"fail_noflag": 10, "fail_cautious": 10, "fail_bold": 25, "pass_split": 10}
    rng = random.Random(a.seed); picked = []
    for k, q in quota.items():
        rows = pool.get(k, []); picked += [(k, r) for r in rng.sample(rows, min(q, len(rows)))]
    pk = sorted(k for k in pool if k.startswith("pass_") and k != "pass_split")
    left = a.n - len(picked)
    for i, k in enumerate(pk):
        q = left // len(pk) + (1 if i < left % len(pk) else 0)
        picked += [(k, r) for r in rng.sample(pool[k], min(q, len(pool[k])))]
    rng.shuffle(picked)
    wb = Workbook(); ws = wb.active; ws.title = "audit"
    with open(a.out + "_key.csv", "w", newline="", encoding="utf-8-sig") as g:
        w = ws; k = csv.writer(g)
        w.append(["audit_id", "data_set", "question_id", "question", "ground_truth", "answer",
                    "human_pass (1/0)", "human_failure (bold/cautious/other)", "notes"])
        k.writerow(["audit_id", "run", "model", "reasoning", "variant", "trial", "stratum", "stratum_size",
                    "sampled_in_stratum", "judge_pass", "judge_bold", "judge_cautious", "judge_votes_pass", "judge_reason"])
        n_in = {s: sum(1 for t, _ in picked if t == s) for s in pool}
        for i, (s, r) in enumerate(picked, 1):
            gr = r["grade"]; votes = [x["grade"]["pass"] for x in r.get("judge_runs") or [] if x.get("grade")]
            w.append([i, r["_data"], r["question_id"], r["question"], r["ground_truth"], r["answer"], "", "", ""])
            k.writerow([i, r["_run"], r["_model"], r["_reasoning"], r["variant"], r["trial"], s, len(pool[s]),
                        n_in[s], int(gr["pass"]), int(gr["unsupported_claim"]), int(gr["unnecessary_abstention"]),
                        f"{sum(votes)}/{len(votes)}", gr.get("reason", "")])
    for c, wd in zip("ABCDEFGHI", (8, 10, 10, 45, 55, 80, 14, 18, 30)): ws.column_dimensions[c].width = wd
    for row in ws.iter_rows():
        for c in row: c.alignment = Alignment(wrap_text=True, vertical="top")
    for c in ws[1]: c.font = Font(bold=True)
    ws.freeze_panes = "A2"
    wb.save(a.out + "_sample.xlsx")
    print(len(picked), "rows;", {s: (n_in[s], len(pool[s])) for s in sorted(pool)})

main()
