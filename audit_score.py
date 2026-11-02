"""Score the judge against human labels: python audit_score.py audit_sample.xlsx audit_key.csv
Reports raw agreement, a stratum-weighted estimate of judge accuracy, and the disagreements."""
import csv, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from openpyxl import load_workbook
from collections import defaultdict
_ws = load_workbook(sys.argv[1]).active
_h = [c.value for c in _ws[1]]
lab = {}
for _r in _ws.iter_rows(min_row=2, values_only=True):
    d = {k: ("" if v is None else str(v)) for k, v in zip(_h, _r)}
    lab[d["audit_id"]] = d
key = {r["audit_id"]: r for r in csv.DictReader(open(sys.argv[2], encoding="utf-8-sig"))}
rows = []
for i, k in key.items():
    h = lab[i]["human_pass (1/0)"].strip()
    if h in ("0", "1"): rows.append((k, int(h)))
print(f"{len(rows)} of {len(key)} labelled")
agree = [(k, int(k["judge_pass"]) == h) for k, h in rows]
print(f"raw agreement {sum(a for _, a in agree)}/{len(agree)}")
st = defaultdict(list)
for k, a in agree: st[k["stratum"]].append(a)
tot = sum(int(next(k for k, _ in agree if k["stratum"] == s)["stratum_size"]) for s in st)
w = sum(int(next(k for k, _ in agree if k["stratum"] == s)["stratum_size"]) * sum(v) / len(v) for s, v in st.items()) / tot
print(f"stratum-weighted agreement (whole population) {w:.1%}")
for s, v in sorted(st.items()): print(f"  {s:15s} {sum(v)}/{len(v)}")
fp = [k for k, h in rows if int(k["judge_pass"]) == 1 and h == 0]
fn = [k for k, h in rows if int(k["judge_pass"]) == 0 and h == 1]
print(f"judge passed but human failed (missed hallucination): {len(fp)}; judge failed but human passed: {len(fn)}")
for k in fp + fn: print("  disagree", k["audit_id"], k["stratum"], k["run"], k["judge_reason"][:100])
