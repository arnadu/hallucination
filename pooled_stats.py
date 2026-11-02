"""Pooled reliability statistics across result runs (same method as the viewer's Reliability tab).

Usage:
    python pooled_stats.py results/<run_a> results/<run_b> [--out report_stats.md] [--resamples 10000]

An answer is "reliable" if the judge passed it. Unreliable answers are split, in this order, into
cautious (unnecessary_abstention flagged), bold (unsupported_claim flagged) and other (no flag).
Question categories map to three groups (should answer / hold back / correct). Within each
(run, group) block the mean per-question change B-A is taken over trials graded in both prompts,
and blocks are averaged with equal weight. Intervals resample questions within each block.
"""

import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

GROUPS = {"answer": "Should answer", "hold": "Should hold back", "correct": "Should correct"}
DEFAULT_GROUP = {
    "answerable": "answer", "knowable_control": "answer",
    "missing_information": "hold", "unknowable": "hold",
    "false_premise": "correct", "false_premise_known": "correct",
}
OUTCOMES = ["reliable", "bold", "cautious", "other"]
OUTCOME_LABEL = {"reliable": "Reliable", "bold": "Too bold", "cautious": "Too cautious", "other": "Other failure"}


def load_run(path):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text("utf-8"))
    rows = [json.loads(line) for line in (path / "answers.jsonl").read_text("utf-8").splitlines() if line.strip()]
    for r in rows:
        g = r.get("grade")
        r["graded"] = bool(g) and not r.get("error")
        if r["graded"]:
            r["outcome"] = ("reliable" if g["pass"] else "cautious" if g["unnecessary_abstention"]
                            else "bold" if g["unsupported_claim"] else "other")
        r["group"] = DEFAULT_GROUP.get(r["category"], "skip")
    variants = [v for v in meta["prompts"] if any(r["variant"] == v for r in rows)]
    return {"name": path.name, "meta": meta, "rows": rows, "variants": variants}


def pts(x):
    return f"{x * 100:+.1f}"


def pct(x):
    return f"{x * 100:.1f}%"


def ci_str(ci):
    return f"[{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]"


def pctl(values):
    s = sorted(values)
    n = len(s)
    return s[int(0.025 * n)], s[min(n - 1, int(0.975 * n))]


def sign_p(b, c):
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def build_blocks(run, a, b):
    """Return ({group: [question dicts]}, trial-pair counts) for matched, graded question-trial pairs."""
    idx = {(r["variant"], r["question_id"], r["trial"]): r for r in run["rows"] if r["variant"] in (a, b)}
    per_q = {}
    only_a = only_b = pairs = 0
    for (v, q, t), ra in idx.items():
        if v != a:
            continue
        rb = idx.get((b, q, t))
        if rb is None or not ra["graded"] or not rb["graded"] or ra["group"] == "skip":
            continue
        pairs += 1
        pa, pb = ra["outcome"] == "reliable", rb["outcome"] == "reliable"
        only_a += pa and not pb
        only_b += pb and not pa
        o = per_q.setdefault(q, {"group": ra["group"], "n": 0, "a": Counter(), "b": Counter()})
        o["n"] += 1
        o["a"][ra["outcome"]] += 1
        o["b"][rb["outcome"]] += 1
    blocks = defaultdict(list)
    for q, o in per_q.items():
        ra = {k: o["a"][k] / o["n"] for k in OUTCOMES}
        rb = {k: o["b"][k] / o["n"] for k in OUTCOMES}
        blocks[o["group"]].append({"q": q, "a": ra, "b": rb, "d": {k: rb[k] - ra[k] for k in OUTCOMES}})
    return blocks, {"pairs": pairs, "only_a": only_a, "only_b": only_b}


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def compare_pair(runs, a, b, resamples, seed=0):
    rng = random.Random(seed)
    strata = []  # dicts: run index, group, qs
    pair_counts = {}
    for ri, run in enumerate(runs):
        blocks, pc = build_blocks(run, a, b)
        pair_counts[ri] = pc
        for g in GROUPS:
            if blocks.get(g):
                strata.append({"ri": ri, "group": g, "qs": blocks[g]})
    # bootstrap each stratum: boot[i][outcome] = list of resampled mean changes
    boot = []
    for s in strata:
        n = len(s["qs"])
        d = {k: [q["d"][k] for q in s["qs"]] for k in OUTCOMES}
        cols = {k: [] for k in OUTCOMES}
        for _ in range(resamples):
            pick = rng.choices(range(n), k=n)
            for k in OUTCOMES:
                dk = d[k]
                cols[k].append(sum(dk[i] for i in pick) / n)
        boot.append(cols)

    def summ(idxs):
        out = {"a": {}, "b": {}, "d": {}, "ci": {}}
        for k in OUTCOMES:
            out["a"][k] = mean([mean([q["a"][k] for q in strata[i]["qs"]]) for i in idxs])
            out["b"][k] = mean([mean([q["b"][k] for q in strata[i]["qs"]]) for i in idxs])
            out["d"][k] = out["b"][k] - out["a"][k]
            arr = [sum(col) / len(idxs) for col in zip(*[boot[i][k] for i in idxs])]
            out["ci"][k] = pctl(arr)
        return out

    all_idx = list(range(len(strata)))
    res = {
        "strata": [{"ri": s["ri"], "group": s["group"], "n": len(s["qs"]), "sum": summ([i])} for i, s in enumerate(strata)],
        "pooled": summ(all_idx),
        "n_questions": sum(len(s["qs"]) for s in strata),
        "pair_counts": pair_counts,
    }
    res["runs"] = []
    for ri in range(len(runs)):
        idxs = [i for i in all_idx if strata[i]["ri"] == ri]
        if idxs:
            res["runs"].append({"ri": ri, "idxs": idxs, "sum": summ(idxs),
                                "n": sum(len(strata[i]["qs"]) for i in idxs)})
    res["groups"] = []
    for g in GROUPS:
        idxs = [i for i in all_idx if strata[i]["group"] == g]
        if idxs:
            res["groups"].append({"g": g, "sum": summ(idxs), "n": sum(len(strata[i]["qs"]) for i in idxs)})
    res["hetero"] = []
    for x, y in itertools.combinations(res["runs"], 2):
        ax = [sum(col) / len(x["idxs"]) for col in zip(*[boot[i]["reliable"] for i in x["idxs"]])]
        ay = [sum(col) / len(y["idxs"]) for col in zip(*[boot[i]["reliable"] for i in y["idxs"]])]
        res["hetero"].append({"x": x["ri"], "y": y["ri"], "d": x["sum"]["d"]["reliable"] - y["sum"]["d"]["reliable"],
                              "ci": pctl([p - q for p, q in zip(ax, ay)])})
    # question-level view per run (equal weight per question, as in analysis.txt)
    res["qlevel"] = []
    for ri, run in enumerate(runs):
        blocks, _ = build_blocks(run, a, b)
        qs = [q for g in blocks.values() for q in g]
        if not qs:
            continue
        d = [q["d"]["reliable"] for q in qs]
        boots = [mean([d[i] for i in rng.choices(range(len(d)), k=len(d))]) for _ in range(resamples)]
        hi = sum(x > 1e-12 for x in d)
        lo = sum(x < -1e-12 for x in d)
        res["qlevel"].append({"ri": ri, "n": len(d), "d": mean(d), "ci": pctl(boots), "higher": hi, "lower": lo,
                              "tied": len(d) - hi - lo, "p": sign_p(hi, lo)})
    return res


def sig(ci):
    return "*" if ci[0] > 0 or ci[1] < 0 else ""


def delta_cell(d, ci):
    return f"{pts(d)}{sig(ci)} {ci_str(ci)}"


def pair_section(runs, a, b, res):
    L = [f"### {b} vs {a}  (B = {b}, A = {a})", ""]
    P = res["pooled"]
    L += [f"**Pooled over {res['n_questions']} question blocks in {len(res['strata'])} strata (equal weight per stratum).** "
          "Changes are B minus A in percentage points; `*` means the 95% interval excludes 0.", "",
          "| Outcome | A | B | Change (B-A) [95% CI] |", "|---|---:|---:|---|"]
    for k in OUTCOMES:
        L.append(f"| {OUTCOME_LABEL[k]} | {pct(P['a'][k])} | {pct(P['b'][k])} | {delta_cell(P['d'][k], P['ci'][k])} |")
    L += ["", "**By run and group** (reliable rate for A and B, then the change in each outcome)", "",
          "| Run (data set) | Group | Questions | A reliable | B reliable | Change reliable | Change bold | Change cautious | Change other |",
          "|---|---|---:|---:|---:|---|---|---|---|"]

    def row(label, grp, n, s):
        return (f"| {label} | {grp} | {n} | {pct(s['a']['reliable'])} | {pct(s['b']['reliable'])} | "
                f"{delta_cell(s['d']['reliable'], s['ci']['reliable'])} | {delta_cell(s['d']['bold'], s['ci']['bold'])} | "
                f"{delta_cell(s['d']['cautious'], s['ci']['cautious'])} | {delta_cell(s['d']['other'], s['ci']['other'])} |")

    for r in res["runs"]:
        name = f"{runs[r['ri']]['meta']['data_dir']}"
        for s in res["strata"]:
            if s["ri"] == r["ri"]:
                L.append(row(name, GROUPS[s["group"]], s["n"], s["sum"]))
        L.append(row(f"**{name} (run average)**", "all groups", r["n"], r["sum"]))
    for g in res["groups"]:
        L.append(row("**pooled**", f"**{GROUPS[g['g']]}**", g["n"], g["sum"]))
    L.append(row("**pooled**", "**all groups**", res["n_questions"], P))

    if res["hetero"]:
        L += ["", "**Do the runs agree?** Difference between runs in the change in reliable rate (first minus second):", ""]
        for h in res["hetero"]:
            verdict = "the runs DISAGREE (interval excludes 0)" if sig(h["ci"]) else "no reliable disagreement"
            L.append(f"- {runs[h['x']]['meta']['data_dir']} vs {runs[h['y']]['meta']['data_dir']}: {pts(h['d'])} pts {ci_str(h['ci'])} - {verdict}")
    L += ["", "**Question-level view per run** (every question equal weight, the same method as the Compare tab and `analysis.txt`)", "",
          "| Run | Questions | Change in reliable [95% CI] | B higher / lower / tied | Sign test p | Trial pairs only-A / only-B | Pair sign test p |",
          "|---|---:|---|---|---:|---|---:|"]
    for q in res["qlevel"]:
        pc = res["pair_counts"][q["ri"]]
        L.append(f"| {runs[q['ri']]['meta']['data_dir']} | {q['n']} | {delta_cell(q['d'], q['ci'])} | {q['higher']} / {q['lower']} / {q['tied']} | "
                 f"{q['p']:.3f} | {pc['only_a']} / {pc['only_b']} | {sign_p(pc['only_b'], pc['only_a']):.3f} |")
    L.append("")
    return L


def descriptive(runs):
    L = ["## 2. Outcome rates per prompt (all graded answers, not matched)", ""]
    for run in runs:
        L += [f"### {run['meta']['data_dir']}  ({run['meta']['model']})", "",
              "| Prompt | Group | Graded answers | Reliable | Too bold | Too cautious | Other |", "|---|---|---:|---:|---:|---:|---:|"]
        for v in run["variants"]:
            for g in list(GROUPS) + ["all"]:
                rs = [r for r in run["rows"] if r["variant"] == v and r["graded"] and (g == "all" or r["group"] == g)]
                if not rs:
                    continue
                c = Counter(r["outcome"] for r in rs)
                label = "all" if g == "all" else GROUPS[g]
                L.append(f"| {v} | {label} | {len(rs)} | " + " | ".join(pct(c[k] / len(rs)) for k in OUTCOMES) + " |")
        L.append("")
    return L


def data_quality(runs):
    L = ["## 1. Runs and data quality", "",
         "| Run | Model | Data set | Judge (passes) | Trials | Temp | Questions | Rows | Graded | Errors | Providers |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for run in runs:
        m, rows = run["meta"], run["rows"]
        prov = Counter(r.get("provider") for r in rows if r.get("provider"))
        L.append(f"| {run['name']} | {m['model']} (reasoning: {m.get('reasoning', 'default')}) | {m['data_dir']} | {m.get('judge_model_effective') or m['judge_model']} ({m['judge_passes']}) | "
                 f"{m['trials']} | {m['temperature']} | {len({r['question_id'] for r in rows})} | {len(rows)} | "
                 f"{sum(r['graded'] for r in rows)} | {sum(bool(r.get('error')) for r in rows)} | "
                 f"{', '.join(f'{p} ({n})' for p, n in prov.items())} |")
    L += ["", "Judge consistency and unflagged failures:", "",
          "| Data set | Answers with 3 judge passes | Pass verdict split | Failed with no flag | Both flags set | Unnecessary abstention on non-answer groups |",
          "|---|---:|---:|---:|---:|---:|"]
    for run in runs:
        multi = [r for r in run["rows"] if len(r.get("judge_runs") or []) > 1]
        split = [r for r in multi if len({bool(x["grade"] and x["grade"]["pass"]) for x in r["judge_runs"]}) > 1]
        graded = [r for r in run["rows"] if r["graded"]]
        noflag = [r for r in graded if r["outcome"] == "other"]
        both = [r for r in graded if r["grade"]["unsupported_claim"] and r["grade"]["unnecessary_abstention"]]
        caut_non = [r for r in graded if r["outcome"] == "cautious" and r["group"] != "answer"]
        L.append(f"| {run['meta']['data_dir']} | {len(multi)} | {len(split)} ({pct(len(split) / len(multi)) if multi else 'n/a'}) | "
                 f"{len(noflag)} | {len(both)} | {len(caut_non)} |")
    L.append("")
    return L


def behaviour(runs):
    L = ["## 4. Behavioural observations", "", "**Answer stability**: share of (question, prompt) cells where every trial gave the same answer text.", "",
         "| Data set | Prompt | Identical in every trial | Mean distinct answers per question (of 5 trials) |", "|---|---|---:|---:|"]
    for run in runs:
        for v in run["variants"]:
            cells = defaultdict(list)
            for r in run["rows"]:
                if r["variant"] == v and not r.get("error"):
                    cells[r["question_id"]].append(" ".join(str(r["answer"]).lower().split()))
            same = sum(len(set(a)) == 1 for a in cells.values())
            L.append(f"| {run['meta']['data_dir']} | {v} | {same}/{len(cells)} ({pct(same / len(cells))}) | {mean([len(set(a)) for a in cells.values()]):.2f} |")
    L += ["", "**Questions that were never reliable under any prompt or trial**", ""]
    for run in runs:
        ok = defaultdict(int)
        cat = {}
        for r in run["rows"]:
            cat[r["question_id"]] = r["category"]
            ok[r["question_id"]] += r["graded"] and r["outcome"] == "reliable"
        never = sorted(q for q in cat if ok[q] == 0)
        by_cat = Counter(cat[q] for q in never)
        L.append(f"- {run['meta']['data_dir']}: {len(never)} of {len(cat)} questions ({', '.join(f'{c}: {n}' for c, n in by_cat.items())}): {', '.join(never)}")
    L += ["", "**Too-cautious answers outside the should-answer group** (refused or said the source lacked an answer that it had)", ""]
    for run in runs:
        c = Counter((r["variant"], r["question_id"]) for r in run["rows"] if r["graded"] and r["outcome"] == "cautious" and r["group"] != "answer")
        L.append(f"- {run['meta']['data_dir']}: " + (", ".join(f"{v}/{q} x{n}" for (v, q), n in sorted(c.items())) or "none"))
    L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="report_stats.md")
    ap.add_argument("--resamples", type=int, default=10000)
    args = ap.parse_args()
    runs = [load_run(p) for p in args.runs]
    models = {f"{r['meta']['model']} (reasoning: {r['meta'].get('reasoning', 'default')})" for r in runs}
    L = ["# Pooled reliability statistics", "",
         f"Generated by `pooled_stats.py` from: {', '.join(r['name'] for r in runs)}. Bootstrap resamples: {args.resamples}, seed 0.", ""]
    if len(models) > 1:
        L += [f"> WARNING: the runs differ in model or reasoning setting ({'; '.join(sorted(models))}); pooled numbers mix those effects with the prompt effect.", ""]
    L += data_quality(runs) + descriptive(runs)
    L += ["## 3. Prompt comparisons", "",
          "Reliable = the judge passed the answer. Too bold = failed with an unsupported claim. Too cautious = failed with an unnecessary abstention "
          "(takes precedence when both are flagged). Other = failed with no flag. Groups: should answer (answerable, knowable_control), "
          "should hold back (missing_information, unknowable), should correct (false_premise, false_premise_known).", ""]
    variants = [v for v in runs[0]["variants"] if all(v in r["variants"] for r in runs)]
    for a, b in itertools.combinations(variants, 2):
        L += pair_section(runs, a, b, compare_pair(runs, a, b, args.resamples))
    L += behaviour(runs)
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
