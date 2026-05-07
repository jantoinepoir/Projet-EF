# stiffness.py
import numpy as np
from scipy.sparse import lil_matrix


def assemble_stiffness_and_rhs(elemTags, conn, jac, det, xphys, w, N, gN, kappa_fun, rhs_fun, tag_to_dof):
    """
    Assemble la matrice de diffusion et le second membre volumique.

    Matrice de diffusion :
        K_ij = ∫_Ω kappa ∇φ_j · ∇φ_i dΩ

    Second membre volumique :
        F_i = ∫_Ω f φ_i dΩ

    Dans notre projet, kappa correspond au coefficient de diffusion D.
    Le terme f est nul dans les modèles étudiés, mais il est gardé dans la fonction pour conserver un assemblage général.
    """
    ne = len(elemTags) # nombre d'éléments
    ngp = len(w) # nombre de points de Gauss par élément
    nloc = int(len(conn) // ne) # nombre de noeuds locaux par élément
    nn = int(np.max(tag_to_dof) + 1) # nombre total de degrés de liberté

    # Reshape des entrées pour faciliter l'assemblage
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
            
            # Les gradients fournis par Gmsh sont exprimés sur l'élément de référence. On les ramène dans l'élément réel avec inv(J).
            invjacg = np.linalg.inv(jac[e, g])

            kappa_g = float(kappa_fun(xg))
            f_g = float(rhs_fun(xg))

            for a in range(nloc):
                Ia = int(dof_indices[a])
                gradNa = invjacg @ gN[g, a]

                F[Ia] += wg * f_g * N[g, a] * detg
                

                for b in range(nloc):
                    Ib = int(dof_indices[b])
                    gradNb = invjacg @ gN[g, b]

                    K[Ia, Ib] += wg * kappa_g * float(np.dot(gradNa, gradNb)) * detg

    return K, F



def assemble_robin_matrix(elemTags, conn, det, xphys, w, N, P, tag_to_dof, nn):
    """
    Assemble la matrice associée à la condition de Robin.

    Pour la condition :
        -D ∇c · n = P(c_plasma - c)

    la formulation faible ajoute au membre gauche :
        R_ij = ∫_{Γ_v} P φ_j φ_i dΓ

    Cette matrice ne dépend pas de la concentration plasmatique. Elle peut donc
    être assemblée une seule fois, puis ajoutée à la matrice de diffusion.
    """
    ne  = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
 
    det   = np.asarray(det,   dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    conn  = np.asarray(conn,  dtype=np.int64  ).reshape(ne, nloc)
    N     = np.asarray(N,     dtype=np.float64).reshape(ngp, nloc)
 
    R = lil_matrix((nn, nn), dtype=np.float64)
 
    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]

        for g in range(ngp):
            wg   = w[g]
            detg = det[e, g]

            for a in range(nloc):
                Ia = int(dof_indices[a])

                for b in range(nloc):
                    Ib = int(dof_indices[b])
                    R[Ia, Ib] += wg * P * N[g, a] * N[g, b] * detg
 
    return R
 


def assemble_robin_rhs(elemTags, conn, det, xphys, w, N, P, c_plasma_fun, tag_to_dof, nn):
    """
    Assemble le second membre associé à la condition de Robin.

    Pour la condition :
        -D ∇c · n = P(c_plasma - c)

    la formulation faible ajoute au membre droit :
        G_i = ∫_{Γ_v} P c_plasma φ_i dΓ

    La concentration plasmatique est donnée sous forme de fonction afin de
    pouvoir utiliser soit une valeur constante, soit une valeur dépendant de
    la position ou du temps.
    """
    ne  = len(elemTags)
    ngp = len(w)
    nloc = int(len(conn) // ne)
 
    det   = np.asarray(det,   dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    conn  = np.asarray(conn,  dtype=np.int64  ).reshape(ne, nloc)
    N     = np.asarray(N,     dtype=np.float64).reshape(ngp, nloc)
 
    G = np.zeros(nn, dtype=np.float64)
 
    for e in range(ne):
        element_tags = conn[e, :]
        dof_indices = tag_to_dof[element_tags]

        for g in range(ngp):
            xg   = xphys[e, g]
            wg   = w[g]
            detg = det[e, g]
            cp   = float(c_plasma_fun(xg))

            for a in range(nloc):
                Ia = int(dof_indices[a])
                G[Ia] += wg * P * cp * N[g, a] * detg
 
    return G