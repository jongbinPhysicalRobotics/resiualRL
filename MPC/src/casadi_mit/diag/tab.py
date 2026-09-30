"""11_gait_quality JSON 요약: tab.py file.json [strip_prefix]"""
import json, sys
d = json.load(open(sys.argv[1])); pre = sys.argv[2] if len(sys.argv) > 2 else ""
for r in d:
    v = r["var"].replace(pre, "") if pre else r["var"]
    c = ",".join(f"{x:g}" for x in r["cmd"])
    ok = "✓" if r["pass"] else "✗"
    s = f"{c:>12s} {v:<34s} {ok} {r['survived']:6.1f}s trk {r['track']:4.0f}% vx {r['vx']:+.2f} vy {r['vy']:+.2f} wz {r['wz']:+.2f}"
    s += f" slip {r['slip_mm']:5.1f} µ95 {r['fric_cone_p95']:.2f} R/L {r['load_RL']:.2f} ms {r['solve_ms_med']:.1f} it {r['iters_med']:.0f}"
    print(s)
