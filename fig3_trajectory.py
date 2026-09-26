"""Fig. 3: one closed-loop realization of the mode-G CUSUM statistic
(2-LD system, all slots under H0) at the MC-calibrated threshold h*,
read from results/final_2LD.json (run pipeline.py 2LD A first)."""
import json, os
import trajectory_plot as T
os.makedirs("figures", exist_ok=True)
h = json.load(open("results/final_2LD.json"))["h_star"]
T.fig_main(T.SYS2, nu=150, T_total=500, h=h, win=100, fname="figures/fig_main_2LD")
