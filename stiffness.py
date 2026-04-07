# stiffness.py
import numpy as np
from scipy.sparse import lil_matrix


def assemble_stiffness_and_rhs(elemTags, conn, jac, det, xphys, w, N, gN, kappa_fun, rhs_fun, tag_to_dof):
    """
    Assemble global stiffness matrix and load vector for:
        -d/dx (kappa(x) du/dx) = f(x)

    K_ij = ∫ kappa * grad(N_i)·grad(N_j) dx
    F_i  = ∫ f * N_i dx

    Notes:
    - gmsh gives gN in reference coordinates; we map with inv(J).
    - For 1D line embedded in 3D, gmsh provides a 3x3 Jacobian; we keep the same approach.

    Returns
    -------
    K : lil_matrix (nn x nn)
    F : ndarray (nn,)
    """
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
    nn = int(np.max(tag_to_dof) + 1)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)

    K = lil_matrix((nn, nn), dtype=np.float64)
    F = np.zeros(nn, dtype=np.float64)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]
        for g in range(ngp):
            xg = xphys[e, g]
            wg = w[g]
            detg = det[e, g]
            invjacg = np.linalg.inv(jac[e, g])

            kappa_g = float(kappa_fun(xg))
            f_g = float(rhs_fun(xg))

            for a in range(nloc):
                Ia = int(dof_indices[a])
                F[Ia] += wg * f_g * N[g, a] * detg

                gradNa = invjacg @ gN[g, a]
                for b in range(nloc):
                    Ib = int(dof_indices[b])
                    gradNb = invjacg @ gN[g, b]
                    K[Ia, Ib] += wg * kappa_g * float(np.dot(gradNa, gradNb)) * detg

    return K, F

def assemble_rhs_neumann(F, elemTags, conn, jac, det, xphys, w, N, gN, g_neu_fun, tag_to_dof):
    ne = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac = np.asarray(jac, dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn = np.asarray(conn, dtype=np.int64).reshape(ne, nloc)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)

    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]
        for g in range(ngp):
            xg = xphys[e, g]
            wg = w[g]
            detg = det[e, g]

            g_neu_g = float(g_neu_fun(xg))

            for a in range(nloc):
                Ia = int(dof_indices[a])
                N_a = N[g, a]
                F[Ia] += wg * g_neu_g * N_a * detg

    return F

def assemble_robin_matrix(elemTags, conn, det, xphys, w, N, P, tag_to_dof, nn):
    """
    Assemble the *constant* stiffness contribution from the Robin BC:
 
        -D grad(c).n = P (c_plasma - c)   on Gamma_v
 
    The term  P * c  on the left-hand side adds to the global stiffness:
 
        K_robin[i,j] = integral_Gamma_v  P * N_i * N_j  ds
 
    This matrix does NOT depend on c_plasma, so it is assembled once and
    added to the diffusion stiffness K.
 
    Parameters
    ----------
    elemTags      : (ne,) boundary element tags
    conn          : flattened boundary connectivity (ne * nloc_bnd)
    det           : flattened boundary det(J) from get_jacobians on the boundary
    xphys         : flattened physical coords of Gauss points (ne * ngp * 3)
    w             : Gauss weights (ngp,)
    N             : flattened boundary basis values (ngp * nloc_bnd)
    P             : float, vascular permeability  [m/s]
    tag_to_dof    : array mapping gmsh node tag -> dof index
    nn            : total number of dofs
 
    Returns
    -------
    K_robin : lil_matrix (nn x nn)
    """
    ne  = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
 
    det   = np.asarray(det,   dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    conn  = np.asarray(conn,  dtype=np.int64  ).reshape(ne, nloc)
    N     = np.asarray(N,     dtype=np.float64).reshape(ngp, nloc)
 
    K_robin = lil_matrix((nn, nn), dtype=np.float64)
 
    for e in range(ne):
        dof_indices = tag_to_dof[conn[e, :]]
        for g in range(ngp):
            wg   = w[g]
            detg = det[e, g]
            for a in range(nloc):
                Ia = int(dof_indices[a])
                for b in range(nloc):
                    Ib = int(dof_indices[b])
                    K_robin[Ia, Ib] += wg * P * N[g, a] * N[g, b] * detg
 
    return K_robin
 
 
def assemble_robin_rhs(elemTags, conn, det, xphys, w, N, P, c_plasma_fun, tag_to_dof, nn):
    """
    Assemble the *time-varying* RHS contribution from the Robin BC:
 
        -D grad(c).n = P (c_plasma - c)   on Gamma_v
 
    The term  P * c_plasma  on the right-hand side contributes:
 
        F_robin[i] = integral_Gamma_v  P * c_plasma(x, t) * N_i  ds
 
    Must be called at each time step when c_plasma is time-dependent.
 
    Parameters
    ----------
    c_plasma_fun  : callable  c_plasma_fun(xyz) -> float
                    Plasma concentration at position xyz (length-3 array).
                    For a uniform plasma concentration c0, use: lambda x: c0
 
    Returns
    -------
    F_robin : ndarray (nn,)
    """
    ne  = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
 
    det   = np.asarray(det,   dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    conn  = np.asarray(conn,  dtype=np.int64  ).reshape(ne, nloc)
    N     = np.asarray(N,     dtype=np.float64).reshape(ngp, nloc)
 
    F_robin = np.zeros(nn, dtype=np.float64)
 
    for e in range(ne):
        dof_indices = tag_to_dof[conn[e, :]]
        for g in range(ngp):
            xg   = xphys[e, g]
            wg   = w[g]
            detg = det[e, g]
            cp   = float(c_plasma_fun(xg))
            for a in range(nloc):
                Ia = int(dof_indices[a])
                F_robin[Ia] += wg * P * cp * N[g, a] * detg
 
    return F_robin