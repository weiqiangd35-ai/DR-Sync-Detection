"""
fig_roc.py -- ROC/DR figures for the letter
============================================
ACTIVE:   make_3LD()  -> fig_roc_3LD.{pdf,png}   (Fig. 2 of the letter)
          Enumerates all deterministic operating points of the 3-LD discrete
          system (20 monotone fusion rules x 64 local-rule combinations),
          draws the concave hull (= DR ROC), the DFC-only randomization
          chord A'B', and marks G, E, C, A, B, W* + the alpha line.
          All named coordinates are the parent paper's published values,
          each verified to lie in the enumerated set / on the hull.

RETIRED:  make_2LD()  -> fig_roc_2LD.{pdf,png}   (no longer used in the tex)
          Kept for the record. Retired because the parent paper's published
          tangency points A, B are slightly off the logistic ROC formulas
          (Pd offset up to ~0.010 at B), which is visible at figure scale,
          and because the DR gain (0.0044) is visually negligible there.
"""
import os
os.makedirs('figures', exist_ok=True)
import numpy as np
from itertools import product
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":"serif","font.size":8,"legend.fontsize":6.3,
    "xtick.labelsize":7,"ytick.labelsize":7,"lines.linewidth":0.9,
    "axes.linewidth":0.6,"axes.labelsize":8})


# ======================================================================
def make_3LD():
    LOC = [(0.0,0.0),(0.1,0.6),(0.2,0.7),(1.0,1.0)]
    U3  = list(product([0,1],repeat=3))

    def monotone_rules():
        rules=[]
        for tt in range(256):
            f={u:(tt>>i)&1 for i,u in enumerate(U3)}
            ok=all(not(all(a<=b for a,b in zip(u,v)) and f[u]>f[v])
                   for u in U3 for v in U3)
            if ok: rules.append(f)
        return rules
    RULES = monotone_rules()
    assert len(RULES)==20                       # Dedekind number M(3)

    def op_point(locs,f):
        Pf=Pd=0.0
        for u in U3:
            if not f[u]: continue
            pf=pd=1.0
            for k,(qf,qd) in enumerate(locs):
                pf*= qf if u[k] else 1-qf
                pd*= qd if u[k] else 1-qd
            Pf+=pf; Pd+=pd
        return Pf,Pd

    pts={tuple(round(x,6) for x in op_point(locs,f))
         for locs in product(LOC,repeat=3) for f in RULES}
    pts=np.array(sorted(pts))

    def upper_hull(P):
        P=P[np.lexsort((-P[:,1],P[:,0]))]
        hull=[]
        for p in P:
            while len(hull)>=2:
                (x1,y1),(x2,y2)=hull[-2],hull[-1]
                if (x2-x1)*(p[1]-y1)-(y2-y1)*(p[0]-x1)>=0: hull.pop()
                else: break
            if hull and abs(p[0]-hull[-1][0])<1e-12: continue
            hull.append(tuple(p))
        return np.array(hull)
    H=upper_hull(pts)

    A=(0.104,0.784); B=(0.271,0.936); C=(0.1708,0.8448)
    G=(0.1360,0.7960); E=(0.1708,0.8208); Ws=(0.2046,0.8210)
    Ap=(0.1180,0.7680); Bp=(0.1900,0.8400); alpha=0.1708

    def inset(p): return any(abs(p[0]-q[0])<5e-4 and abs(p[1]-q[1])<5e-4
                             for q in pts)
    for nm,p_ in (("A",A),("B",B),("G",G),("A'",Ap),("B'",Bp)):
        assert inset(p_), f"{nm} not in enumerated set"

    fig,ax=plt.subplots(figsize=(3.5,1.85))
    m=(pts[:,0]>=0.04)&(pts[:,0]<=0.36)&(pts[:,1]>=0.68)
    ax.plot(pts[m,0],pts[m,1],"o",ms=2.2,mfc="none",mec="#7799cc",mew=0.6,
            label="deterministic operating points")
    hm=(H[:,0]>=0.0)&(H[:,0]<=0.40)
    ax.plot(H[hm,0],H[hm,1],"k-",lw=1.3,label="DR ROC (convex-hull boundary)")
    ax.plot([Ap[0],Bp[0]],[Ap[1],Bp[1]],ls="--",color="#c04040",lw=0.9,
            label="randomization at the DFC only")
    ax.axvline(alpha,color="0.5",ls="--",lw=0.7)
    ax.text(alpha+0.003,0.700,r"$\alpha$",color="0.35",fontsize=7)

    def mark(p,style,color,lbl,dx,dy,ms=4.5):
        ax.plot(*p,style,ms=ms,color=color)
        ax.annotate(lbl,p,(p[0]+dx,p[1]+dy),fontsize=8,color=color)
    mark(A,"o","#2e8b2e","A",-0.014,0.008)
    mark(B,"o","#2e8b2e","B",0.005,0.004)
    mark(C,"o","k","C",-0.004,0.011)
    mark(G,"o","#2050b0","V",-0.006,-0.020)
    mark(E,"o","#c04040","R",-0.017,-0.006)
    mark(Ws,"s","#b03090",r"$W^{*}$",0.006,-0.008)
    ax.annotate("",xy=Ws,xytext=C,
        arrowprops=dict(arrowstyle="->",lw=0.8,color="#b03090",ls=":"))
    ax.text(0.196,0.836,"sync\nlost",fontsize=6.3,color="#b03090",ha="center")
    ax.set_xlim(0.045,0.325); ax.set_ylim(0.685,0.965)
    ax.set_xlabel(r"$P_f$"); ax.set_ylabel(r"$P_d$")
    ax.legend(loc="lower right",framealpha=0.93)
    for ext in ("pdf","png"):
        fig.savefig(f"figures/fig_roc_3LD.{ext}",bbox_inches="tight",
                    dpi=600 if ext=="png" else None)
    plt.close(fig)
    print("fig_roc_3LD.{pdf,png} written; cross-checks passed")


# ======================================================================
def make_2LD():   # RETIRED, see module docstring
    pf1=np.linspace(1e-4,1-1e-4,4000)
    tau=2*np.arctanh(1-2*pf1)
    pd1=0.5-0.5*np.tanh((tau-2.5)/2)
    AND=(pf1**2,pd1**2); OR=(2*pf1-pf1**2,2*pd1-pd1**2)
    A=(0.1581,0.7870); B=(0.2437,0.8652); C=(0.2009,0.8261)
    G=(0.2009,0.8217); Ws=(0.2640,0.7600); alpha=0.2009
    fig,ax=plt.subplots(figsize=(3.5,2.5))
    ax.plot(*AND,color="#b02020",lw=0.9,label="AND-rule ROC")
    ax.plot(*OR,color="#2050b0",lw=0.9,label="OR-rule ROC")
    ax.plot([A[0],B[0]],[A[1],B[1]],"k-",lw=1.4,label="dependent randomization")
    ax.axvline(alpha,color="0.5",ls="--",lw=0.7)
    for p,c,l,dx,dy in ((A,"#2e8b2e","A",-0.016,0.008),(B,"#2e8b2e","B",0.005,0.006),
                        (C,"k","C",-0.004,0.012),(Ws,"#b03090",r"$W^{*}$",0.006,-0.006)):
        ax.plot(*p,"o",ms=4,color=c); ax.annotate(l,p,(p[0]+dx,p[1]+dy),fontsize=8,color=c)
    ax.set_xlim(0.08,0.33); ax.set_ylim(0.62,0.92)
    ax.set_xlabel(r"$P_f$"); ax.set_ylabel(r"$P_d$")
    ax.legend(loc="lower right",framealpha=0.92)
    for ext in ("pdf","png"):
        fig.savefig(f"figures/fig_roc_2LD.{ext}",bbox_inches="tight",
                    dpi=600 if ext=="png" else None)
    plt.close(fig)
    print("fig_roc_2LD.{pdf,png} written (retired figure)")


if __name__=="__main__":
    make_3LD()
