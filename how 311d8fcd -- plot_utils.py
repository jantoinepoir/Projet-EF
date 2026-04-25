[33mcommit 311d8fcd40ddb8add853f0f8e00e495797fbfbd8[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmain[m[33m)[m
Author: edoyle <eloise.doyle@student.uclouvain.be>
Date:   Thu Apr 23 10:51:26 2026 +0200

    chgt d'échelle

[1mdiff --git a/__pycache__/dirichlet.cpython-312.pyc b/__pycache__/dirichlet.cpython-312.pyc[m
[1mindex f6a5bac6..9908bdc0 100644[m
Binary files a/__pycache__/dirichlet.cpython-312.pyc and b/__pycache__/dirichlet.cpython-312.pyc differ
[1mdiff --git a/__pycache__/gmsh_utils.cpython-312.pyc b/__pycache__/gmsh_utils.cpython-312.pyc[m
[1mindex 8c25c580..16d762cf 100644[m
Binary files a/__pycache__/gmsh_utils.cpython-312.pyc and b/__pycache__/gmsh_utils.cpython-312.pyc differ
[1mdiff --git a/__pycache__/mass.cpython-312.pyc b/__pycache__/mass.cpython-312.pyc[m
[1mindex bf78d839..f9907a00 100644[m
Binary files a/__pycache__/mass.cpython-312.pyc and b/__pycache__/mass.cpython-312.pyc differ
[1mdiff --git a/__pycache__/stiffness.cpython-312.pyc b/__pycache__/stiffness.cpython-312.pyc[m
[1mindex 2ef41925..eea7a0fa 100644[m
Binary files a/__pycache__/stiffness.cpython-312.pyc and b/__pycache__/stiffness.cpython-312.pyc differ
[1mdiff --git a/main_krogh.py b/main_krogh.py[m
[1mindex 67d7710d..b2275582 100644[m
[1m--- a/main_krogh.py[m
[1m+++ b/main_krogh.py[m
[36m@@ -22,6 +22,7 @@[m [mUsage examples:[m
 import argparse[m
 import numpy as np[m
 import matplotlib.pyplot as plt[m
[32m+[m[32mfrom matplotlib.ticker import ScalarFormatter[m[41m [m
 [m
 from gmsh_utils import ([m
     getPhysical,gmsh_init, gmsh_finalize,[m
[36m@@ -80,7 +81,7 @@[m [mdef main():[m
 [m
     # Physique[m
     # Remplacer dans le parser :[m
[31m-    parser.add_argument("--D",  type=float, default=2.4e-11, help="Diffusion coefficient [m^2/s]")[m
[32m+[m[32m    parser.add_argument("--D",  type=float, default=3.6e-10, help="Diffusion coefficient [m^2/s]")[m
     parser.add_argument("--P",  type=float, default=3.0e-8,  help="Vascular permeability [m/s]")[m
     parser.add_argument("--kr", type=float, default=2.0e-4,  help="Reaction/uptake rate [s^-1]")[m
     parser.add_argument("--order", type=int, default=1, help="Polynomial order (1=P1, 2=P2)")[m
[36m@@ -302,10 +303,18 @@[m [mdef main():[m
             ax.set_xlim([-1.1*args.Rt, 1.1*args.Rt])[m
             ax.set_ylim([-1.1*args.Rt, 1.1*args.Rt])[m
             ax.axis('equal')[m
[32m+[m[32m            for axis in [ax.xaxis, ax.yaxis]:[m
[32m+[m[32m                fmt = ScalarFormatter(useMathText=True)[m
[32m+[m[32m                fmt.set_powerlimits((0, 0))   # force la notation 10^n pour toutes les valeurs[m
[32m+[m[32m                axis.set_major_formatter(fmt)[m
[32m+[m
[32m+[m[32m            # Ticks explicites pour x et y[m
[32m+[m[32m            ticks = [-1e-4, -5e-5, 0, 5e-5, 1e-4][m
[32m+[m[32m            ax.set_xticks(ticks)[m
[32m+[m[32m            ax.set_yticks(ticks)[m
             plt.pause(0.001)[m
[31m-[m
[31m-    plt.ioff()[m
[31m-    plt.show()[m
[32m+[m[32m            plt.ioff()[m
[32m+[m[32m            plt.show()[m
 [m
     fig1, ax1 = plt.subplots()[m
     ax1.plot(times, avg_conc * 1e3, '-o')[m
