# dirichlet.py
import numpy as np
from scipy.sparse.linalg import spsolve

def apply_dirichlet_by_reduction(K, F, dirichlet_dofs, dirichlet_values):
    dirichlet_dofs = np.asarray(dirichlet_dofs, dtype=int)
    dirichlet_values = np.asarray(dirichlet_values, dtype=float)

    n = len(F)
    mask = np.ones(n, dtype=bool)
    mask[dirichlet_dofs] = False
    free_dofs = np.nonzero(mask)[0]

    K_FF = K[free_dofs, :][:, free_dofs]
    K_FD = K[free_dofs, :][:, dirichlet_dofs]

    F_F = F[free_dofs]
    F_red = F_F - K_FD.dot(dirichlet_values)

    U_full = np.zeros(n, dtype=float)
    U_full[dirichlet_dofs] = dirichlet_values

    return K_FF, F_red, free_dofs, U_full

def theta_step(M, K, F_n, F_np1, U_n, dt, theta, dirichlet_dofs, dir_vals_np1):
    A = M + theta * dt * K
    B = M - (1.0 - theta) * dt * K
    rhs = B.dot(U_n) + dt * (theta * F_np1 + (1.0 - theta) * F_n)

    A_red, rhs_red, free_dofs, U_full = apply_dirichlet_by_reduction(A, rhs, dirichlet_dofs, dir_vals_np1)
    U_free = spsolve(A_red.tocsr(), rhs_red)
    U_full[free_dofs] = U_free
    U_full[dirichlet_dofs] = dir_vals_np1
    return U_full