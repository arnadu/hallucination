"""Summaries and a detailed comparison report for a finished run.

Usage (no API calls needed):
    python analysis.py results/<run_id>
"""
import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 0
REFERENCE_VARIANT = "baseline"
# Empty-answer rates further apart than this make raw pass rates unfair.
EMPTY_RATE_WARNING_GAP = 0.05


def error_type(record):
    if not record.get("error"):
        return None
    if record.get("error_type"):
        return record["error_type"]
    # Runs made before error types were recorded.
    return "empty_answer" if "empty answer" in record["error"] else "other"


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["variant"], "all")].append(record)
        groups[(record["variant"], record["category"])].append(record)

    summary = []
    for (variant, category), rows in sorted(groups.items()):
        graded = [r for r in rows if r["grade"] is not None]
        n = len(graded)
        passed = sum(r["grade"]["pass"] for r in graded)
        unsupported = sum(r["grade"]["unsupported_claim"] for r in graded)
        abstained = sum(r["grade"]["unnecessary_abstention"] for r in graded)
        summary.append({
            "variant": variant,
            "category": category,
            "attempted": len(rows),
            "graded": n,
            "errors": len(rows) - n,
            "empty_answers": sum(
                error_type(r) == "empty_answer" for r in rows
            ),
            "recovered_by_retry": sum(
                1 for r in rows
                if r["answer"] is not None and r.get("empty_responses")
            ),
            "passed": passed,
            "pass_rate": passed / n if n else None,
            "pass_rate_errors_as_fail": passed / len(rows) if rows else None,
            "unsupported_claims": unsupported,
            "unsupported_claim_rate": unsupported / n if n else None,
            "unnecessary_abstentions": abstained,
        })
    return summary


def write_summary(out, summary):
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


# ---------------------------------------------------------------- statistics

def mean(values):
    return sum(values) / len(values)


def sign_test_p(a, b):
    """Two-sided exact sign test (McNemar's exact test) on a vs b wins."""
    n = a + b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_ci(values):
    """95% percentile bootstrap CI for the mean of values."""
    if not values:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    means = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(BOOTSTRAP_SAMPLES)
    )
    return (
        means[int(0.025 * BOOTSTRAP_SAMPLES)],
        means[int(0.975 * BOOTSTRAP_SAMPLES) - 1],
    )


def fraction(count, total):
    return f"{count}/{total} ({count / total:.1%})" if total else "n/a"


def pts(value):
    return f"{value * 100:+.1f} pts"


# ------------------------------------------------------------------- report

def index_by_key(records):
    by_key = defaultdict(dict)
    for r in records:
        by_key[(r["question_id"], r["trial"])][r["variant"]] = r
    return by_key


def per_question(pairs, field):
    """Map question id -> (reference values, other values) over the pairs."""
    grouped = defaultdict(lambda: ([], []))
    for ref_row, other_row in pairs:
        ref_vals, other_vals = grouped[ref_row["question_id"]]
        ref_vals.append(ref_row["grade"][field])
        other_vals.append(other_row["grade"][field])
    return grouped


def data_quality_section(records, variants):
    lines = ["1. DATA QUALITY (rows that could not be graded)"]
    lines.append(
        f"  {'variant':20} {'attempted':>9} {'graded':>7} {'empty':>6} "
        f"{'other_err':>9} {'retry_ok':>8}"
    )
    empty_rates = {}
    for variant in variants:
        rows = [r for r in records if r["variant"] == variant]
        graded = sum(r["grade"] is not None for r in rows)
        empty = sum(error_type(r) == "empty_answer" for r in rows)
        other = len(rows) - graded - empty
        retried = sum(
            1 for r in rows
            if r["answer"] is not None and r.get("empty_responses")
        )
        empty_rates[variant] = empty / len(rows) if rows else 0.0
        lines.append(
            f"  {variant:20} {len(rows):>9} {graded:>7} {empty:>6} "
            f"{other:>9} {retried:>8}"
        )

    gap = max(empty_rates.values()) - min(empty_rates.values())
    if gap > EMPTY_RATE_WARNING_GAP:
        lines.append(
            f"  WARNING: empty-answer rates differ by {gap:.1%} between "
            "variants, so pass rates over graded rows compare different "
            "question sets. Use the matched-pair results below."
        )

    for variant in variants:
        providers = Counter(
            r.get("provider") for r in records
            if r["variant"] == variant and r.get("provider")
        )
        if providers:
            lines.append(f"  Answering provider, {variant}: {dict(providers)}")

    empties = [e for r in records for e in r.get("empty_responses") or []]
    if empties:
        lines.append(
            f"  Empty responses logged: {len(empties)} "
            f"(finish_reason: {dict(Counter(e['finish_reason'] for e in empties))}; "
            f"provider: {dict(Counter(e.get('provider') for e in empties))}; "
            f"content: {dict(Counter(repr(e['content']) for e in empties))})"
        )
    return lines


def three_views_section(records, variants, by_key):
    lines = ["", "2. PASS RATE, THREE WAYS"]
    matched = [
        k for k, v in by_key.items()
        if all(var in v and v[var]["grade"] for var in variants)
    ]
    header = " " * 26 + "".join(f"{v:>26}" for v in variants)
    lines.append(header)

    def row(label, cell):
        return f"  {label:24}" + "".join(f"{cell(v):>26}" for v in variants)

    def rows_of(variant):
        return [r for r in records if r["variant"] == variant]

    lines.append(row("graded rows only", lambda v: fraction(
        sum(r["grade"]["pass"] for r in rows_of(v) if r["grade"]),
        sum(r["grade"] is not None for r in rows_of(v)),
    )))
    lines.append(row("errors count as fail", lambda v: fraction(
        sum(r["grade"]["pass"] for r in rows_of(v) if r["grade"]),
        len(rows_of(v)),
    )))
    lines.append(row(
        f"matched ({len(matched)} pairs)",
        lambda v: fraction(
            sum(by_key[k][v]["grade"]["pass"] for k in matched), len(matched)
        ),
    ))
    lines.append(
        "  'matched' keeps only question-trial pairs graded in every variant."
    )
    return lines


def comparison_section(records, ref, other, by_key):
    pairs = [
        (v[ref], v[other]) for v in by_key.values()
        if ref in v and other in v and v[ref]["grade"] and v[other]["grade"]
    ]
    lines = [
        "",
        f"3. PAIRED COMPARISON: {other} vs {ref} "
        f"({len(pairs)} matched question-trial pairs)",
    ]
    if not pairs:
        return lines + ["  No pairs were graded in both variants."]

    n = len(pairs)
    metrics = [("pass", "Pass"), ("unsupported_claim", "Unsupported claim")]
    if any(
        a["grade"]["unnecessary_abstention"]
        or b["grade"]["unnecessary_abstention"]
        for a, b in pairs
    ):
        metrics.append(("unnecessary_abstention", "Unnecessary abstention"))
    for field, label in metrics:
        ref_only = sum(
            a["grade"][field] and not b["grade"][field] for a, b in pairs
        )
        other_only = sum(
            b["grade"][field] and not a["grade"][field] for a, b in pairs
        )
        ref_total = sum(a["grade"][field] for a, _ in pairs)
        other_total = sum(b["grade"][field] for _, b in pairs)
        lines.append(
            f"  {label}: {ref} {fraction(ref_total, n)}, "
            f"{other} {fraction(other_total, n)}"
        )
        lines.append(
            f"    true only for {ref}: {ref_only}; only for {other}: "
            f"{other_only}; exact sign-test p={sign_test_p(ref_only, other_only):.3f}"
            " (treats every pair as independent, so it overstates certainty)"
        )

        grouped = per_question(pairs, field)
        diffs = [mean(o) - mean(r) for r, o in grouped.values()]
        low, high = bootstrap_ci(diffs)
        higher = sum(d > 0 for d in diffs)
        lower = sum(d < 0 for d in diffs)
        lines.append(
            f"    By question ({len(diffs)} questions, equal weight): "
            f"{other} - {ref} = {pts(mean(diffs))}, "
            f"95% bootstrap CI [{pts(low)}, {pts(high)}]"
        )
        lines.append(
            f"    {other} higher on {higher} questions, lower on {lower}, "
            f"tied on {len(diffs) - higher - lower} "
            f"(sign-test p={sign_test_p(higher, lower):.3f})"
        )
        if low > 0:
            verdict = f"{other} has a reliably HIGHER {label.lower()} rate."
        elif high < 0:
            verdict = f"{other} has a reliably LOWER {label.lower()} rate."
        else:
            verdict = (
                "No reliable difference detected (the CI includes 0)."
            )
        lines.append(f"    -> {verdict}")

    lines.append("  Pass rate by category (matched pairs):")
    for category in sorted({a["category"] for a, _ in pairs}):
        rows = [(a, b) for a, b in pairs if a["category"] == category]
        lines.append(
            f"    {category:22} n={len(rows):<4} "
            f"{ref} {fraction(sum(a['grade']['pass'] for a, _ in rows), len(rows)):>16}  "
            f"{other} {fraction(sum(b['grade']['pass'] for _, b in rows), len(rows)):>16}"
        )

    lines.append("")
    lines.append(f"4. PER-QUESTION DETAIL: {other} vs {ref} (pass counts, matched trials)")
    questions = {r["question_id"]: r for r in records}
    grouped = per_question(pairs, "pass")
    differing = sorted(
        (mean(o) - mean(r), q) for q, (r, o) in grouped.items()
        if mean(o) != mean(r)
    )
    width = max(len(ref), len(other))
    if differing:
        lines.append(
            f"  {'id':5} {'category':20} {ref:>{width}} {other:>{width}} "
            f"{'diff':>10}  question"
        )
    for diff, q in differing:
        r, o = grouped[q]
        text = questions[q]["question"]
        lines.append(
            f"  {q:5} {questions[q]['category']:20} "
            f"{f'{sum(r)}/{len(r)}':>{width}} {f'{sum(o)}/{len(o)}':>{width}} "
            f"{pts(diff):>10}  "
            f"{text[:40] + '...' if len(text) > 40 else text}"
        )
    both_fail = sorted(
        q for q, (r, o) in grouped.items() if not any(r) and not any(o)
    )
    both_pass = sum(1 for r, o in grouped.values() if all(r) and all(o))
    lines.append(
        f"  Failed every matched trial in BOTH variants: "
        f"{', '.join(both_fail) if both_fail else 'none'}"
    )
    lines.append(
        f"  Passed every matched trial in both variants: {both_pass} "
        f"of {len(grouped)} questions"
    )
    return lines


def judge_section(records):
    judged = [r for r in records if r.get("judge_runs")]
    multi = [r for r in judged if len(r["judge_runs"]) > 1]
    lines = ["", "5. JUDGE RELIABILITY"]
    if not multi:
        lines.append(
            "  The judge ran once per answer, so its own consistency is "
            "unmeasured. Use --judge-passes 3 (or --regrade with it) to "
            "measure how often it disagrees with itself."
        )
        return lines

    split = [
        r for r in multi
        if len({run["grade"]["pass"] for run in r["judge_runs"]}) > 1
    ]
    lines.append(
        f"  {len(multi[0]['judge_runs'])} judge passes per answer; the pass "
        f"verdict was split on {fraction(len(split), len(multi))} of answers."
    )
    if split:
        lines.append(
            "  Split answers: "
            + ", ".join(
                f"{r['question_id']}/{r['variant']}/t{r['trial']}"
                for r in split[:12]
            )
            + (" ..." if len(split) > 12 else "")
        )
    return lines


def build_report(records, metadata=None):
    metadata = metadata or {}
    variants = sorted({r["variant"] for r in records})
    ref = REFERENCE_VARIANT if REFERENCE_VARIANT in variants else variants[0]
    by_key = index_by_key(records)

    lines = ["=" * 78, "DETAILED ANALYSIS", "=" * 78]
    lines.append(
        f"model={metadata.get('model')}  "
        f"judge={metadata.get('judge_model_effective')}  "
        f"judge_passes={metadata.get('judge_passes', 1)}  "
        f"trials={metadata.get('trials')}"
    )
    lines.append(
        f"{len({r['question_id'] for r in records})} questions, "
        f"{len(records)} rows"
        + (
            f"  (regraded from {metadata['regraded_from']})"
            if metadata.get("regraded_from") else ""
        )
    )
    lines.append("")
    lines += data_quality_section(records, variants)
    lines += three_views_section(records, variants, by_key)
    for other in variants:
        if other != ref:
            lines += comparison_section(records, ref, other, by_key)
    lines += judge_section(records)
    return lines


def print_report(lines):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print("\n".join(lines))


def load_run(run_dir):
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text("utf-8"))
    records = [
        json.loads(line)
        for line in (run_dir / "answers.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    return records, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="a results/<run_id> folder")
    args = parser.parse_args()

    records, metadata = load_run(args.run_dir)
    lines = build_report(records, metadata)
    (Path(args.run_dir) / "analysis.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print_report(lines)


if __name__ == "__main__":
    main()
