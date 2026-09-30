import json, sys, collections
d = json.load(open(sys.argv[1]))
rows = collections.defaultdict(dict)
for r in d:
    kv = dict(x.split("=") for x in r["var"].split(","))
    key = (float(kv.get("tcycle", 0.5)), float(kv.get("tswing", 0.17)), kv.get("hor"), float(kv["khalf"]))
    c = ",".join(f"{x:g}" for x in r["cmd"])
    rows[key][c] = r
print("주기 스윙 구간 k_half | 1.0 | 0.6 | 0.6+0.5 | 제자리")
for key in sorted(rows):
    cells = []
    for c in ("1,0,0", "0.6,0,0", "0.6,0,0.5", "0,0,0"):
        r = rows[key].get(c)
        if r is None:
            cells.append("—"); continue
        s = f"✓ {r['track']:.0f}% ({r['vx']:.2f}) slip {r['slip_mm']:.1f} sat {100*r['tau_sat_frac']:.1f}%" if r["pass"] else f"✗ {r['survived']:.1f}s ({r['vx']:.2f})"
        cells.append(s)
    print(f"{key[0]} {key[1]} {key[2]} {key[3]:.3f} | " + " | ".join(cells))
