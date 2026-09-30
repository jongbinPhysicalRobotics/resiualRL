"""외란 격자 표: push_tab.py file.json → 방향 × 크기. 위상 여러 개면 성공 수/시행 수."""
import json, sys
d = json.load(open(sys.argv[1]))
tab = {}
for r in d:
    p = r.get("push")
    if p is None:
        continue
    fx, fy = p[0], p[1]
    dirn = "앞 (+x)" if fx > 0 else "뒤 (−x)" if fx < 0 else "왼 (+y)" if fy > 0 else "오른 (−y)"
    mag = abs(fx) + abs(fy)
    tab.setdefault(dirn, {}).setdefault(mag, []).append((bool(r["pass"]), r["survived"]))
mags = sorted({m for v in tab.values() for m in v})
print("| 방향 | " + " | ".join(f"{m:g} N" for m in mags) + " |")
print("|---|" + "---|" * len(mags))
def cell(lst):
    if lst is None:
        return "—"
    if len(lst) == 1:
        return "✓" if lst[0][0] else f"✗ {lst[0][1]:.1f}s"
    n = sum(ok for ok, _ in lst)
    return f"{n}/{len(lst)}" + (" ✓" if n == len(lst) else "")
for dirn in ["앞 (+x)", "뒤 (−x)", "왼 (+y)", "오른 (−y)"]:
    if dirn in tab:
        print(f"| {dirn} | " + " | ".join(cell(tab[dirn].get(m)) for m in mags) + " |")
