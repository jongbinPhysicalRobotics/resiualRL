import json, collections, sys
P="profile=ramp,fix=half+yawref,ypath=0.3,init=warm,khalf=0.3,yawtraj=1.0,yawff=2,psioff=0,rai=0.2,vdir=0.7,admm=25,nbs=5,rho=5e-5,hor=0.44,fref=1e-3,armsw=0.3,xmax=1.0,rvwb=1,replan=1,yawlead=0.5"
d=json.load(open(sys.argv[1]))
agg=collections.defaultdict(lambda: collections.defaultdict(list))
for r in d:
    v=r["var"].replace(P,"").split(",push")[0] or "(A8)"
    agg[v][r["push"][1]].append(r["pass"])
fs=sorted({f for dd in agg.values() for f in dd}, key=lambda f: (abs(f), -f))
print("변형 | " + " | ".join(("왼 " if f>0 else "오른 ")+f"{abs(f):g}" for f in fs) + " | 합")
for v,dd in agg.items():
    print(f"{v} | " + " | ".join(f"{sum(dd[f])}/{len(dd[f])}" for f in fs) + f" | {sum(sum(x) for x in dd.values())}/{sum(len(x) for x in dd.values())}")
if len(sys.argv)>2:
    for r in json.load(open(sys.argv[2])):
        v=r["var"].replace(P,"") or "(A8)"; c=",".join(f"{x:g}" for x in r["cmd"])
        print(f"{c:>10s} {v:<26s} {'✓' if r['pass'] else '✗'} {r['survived']:5.1f}s trk {r['track']:4.0f}% vy {r['vy']:+.3f} slip {r['slip_mm']:.1f}")
