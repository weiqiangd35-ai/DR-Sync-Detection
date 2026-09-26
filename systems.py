"""Benchmark systems of Dong & Kam, IEEE/CAA JAS 8(2):361-376, 2021.

A system is n local detectors (LDs) and a data fusion center (DFC) that
hop, via a shared coin s_t (A w.p. p), between two deterministic
strategies. `locA`/`locB` are the per-LD (P_f, P_d) under each strategy;
`thrA`/`thrB` are the k-out-of-n fusion thresholds. All laws are joint
laws of the DFC record (s_t, U_t) under H_0 (h=0) or H_1 (h=1).

Failure modes (Sec. II of the letter):
  G        -- the LD group follows a common coin independent of s_t
  P{...}   -- the listed LDs are decoupled (partial mode)
  F        -- all LDs decoupled
"""
import numpy as np
from itertools import product


class Sys:
    def __init__(s,n,locA,locB,thrA,thrB,p,alpha,A,B):
        s.n,s.locA,s.locB,s.thrA,s.thrB=n,locA,locB,thrA,thrB
        s.p,s.alpha,s.A,s.B=p,alpha,A,B
        s.U=list(product([0,1],repeat=n))
    def bit(s,op,u,h=0): return (op[h] if u else 1-op[h])
    def prod(s,op,Uv,h=0):
        r=1.0
        for u in Uv: r*=s.bit(op,u,h)
        return r
    def fuse(s,cs,Uv): return 1 if sum(Uv)>=(s.thrA if cs==0 else s.thrB) else 0
    def f_sync(s,p=None,h=0):
        p=s.p if p is None else p
        return {(cs,Uv):ps*s.prod(op,Uv,h)
                for cs,ps,op in ((0,p,s.locA),(1,1-p,s.locB)) for Uv in s.U}
    def f_G(s,p=None,h=0):
        p=s.p if p is None else p
        g={Uv:p*s.prod(s.locA,Uv,h)+(1-p)*s.prod(s.locB,Uv,h) for Uv in s.U}
        return {(cs,Uv):ps*g[Uv] for cs,ps in ((0,p),(1,1-p)) for Uv in s.U}
    def f_part(s,yb,p=None,h=0):
        p=s.p if p is None else p
        law={}
        for cs,ps,ops in ((0,p,s.locA),(1,1-p,s.locB)):
            for Uv in s.U:
                pr=ps
                for k,u in enumerate(Uv):
                    pr *= (p*s.bit(s.locA,u,h)+(1-p)*s.bit(s.locB,u,h)) if k in yb \
                          else s.bit(ops,u,h)
                law[(cs,Uv)]=law.get((cs,Uv),0)+pr
        return law
    def modes(s,p=None,h=0):
        M={"G":s.f_G(p,h)}
        for sub in product(*[[0,1]]*s.n):
            yb={i for i in range(s.n) if sub[i]}
            if not yb: continue
            key="F" if len(yb)==s.n else "P{"+",".join(f"LD{i+1}" for i in sorted(yb))+"}"
            M[key]=s.f_part(yb,p,h)
        return M

def kl(fm,f0): return sum(v*np.log(v/f0[k]) for k,v in fm.items() if v>0)


SYS2 = Sys(2,(0.3976,0.8871),(0.1304,0.6328),2,1,0.5,0.2009,(0.1581,0.7870),(0.2437,0.8652))
SYS3 = Sys(3,(0.2,0.7),(0.1,0.6),2,1,0.6,0.1708,(0.104,0.784),(0.271,0.936))
