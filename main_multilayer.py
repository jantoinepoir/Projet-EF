#main_multilayer.py
"diffusion de la doxorubicine depuis un vaisseau sanguin à travers une peau multicouche (tissu sain, tissu tumoral, tissu nécrosé)"

import argparse
import numpy as np
import matplotlib.pyplot as plt
import gmsh
from scipy.sparse import lil_matrix, csr_matrix, diags  # <--- Ajout de diags pour le Lumping

from gmsh_utils import (
    gmsh_init, gmsh_finalize,
    prepare_quadrature_and_basis,
    build_multilayer_rect_mesh,
    get_jacobians,
    getPhysical
)
from stiffness import (
    assemble_stiffness_and_rhs,
    assemble_robin_matrix,
    assemble_robin_rhs,
)
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import plot_fe_solution_2d, plot_mesh_2d


#pour pouvoir faire le graphe de la concentration moyenne à différents moments
def linear_profile(dof_coords, U, L, nbins=60):
    # dof_coords[:, 0] contient les coordonnées x de tes nœuds
    x = np.asarray(dof_coords, dtype=float)[:, 0]
    
    # On normalise la distance par rapport à la largeur totale L
    rho = x / L 
    bins = np.linspace(0.0, 1.0, nbins + 1)
    inds = np.digitize(rho, bins) - 1

    profile = np.zeros(nbins, dtype=float)
    counts = np.zeros(nbins, dtype=int)
    
    for i, idx in enumerate(inds):
        if 0 <= idx < nbins:
            profile[idx] += U[i]
            counts[idx] += 1

    nonzero = counts > 0
    profile[nonzero] /= counts[nonzero]
    centers = 0.5 * (bins[:-1] + bins[1:])

    return centers[nonzero], profile[nonzero]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=float, default=0.00015)
    parser.add_argument("--H", type=float, default=0.005)
    parser.add_argument("--cl", type=float, default=0.000015)
    parser.add_argument("--dt", type=float, default=20.0)
    parser.add_argument("--nsteps", type=int, default=100)
    parser.add_argument("--theta", type=float, default=1.0)
    args = parser.parse_args()

    #Propiétés des différentes couches (diffusion et capture)
    layer_props = { #D est le coefficient de diffusion, kr est le taux de capture (consommation) de la doxorubicine par les cellules du tissu
        "Layer1": {"D": 1.0e-10, "kr": 5.0e-6}, #tissu dense près du vaisseau (la diffusion y est plus lente car forte capture)
        "Layer2": {"D": 5.0e-10, "kr": 2.0e-7}, #tissu plus lâche au milieu/tumoral standard (la diffusion y est plus rapide car moins de capture)
        "Layer3": {"D": 9.96e-10, "kr": 5.0e-8} #tissu tumoral nécrosé au centre (la diffusion y est très rapide car quasi pas de capture, moins de cellules viables)--> la pénétration y est facilitée
    }
    #conditions aux limites (condition de Robin au niveau de la paroi du vaisseau):
    P = 2.78e-6 #perméabilité pour la doxorubicine libre
    C_PLASMA = 5.0e-3 

    gmsh_init("skin_diffusion")
    
    # 1. Construction du maillage
    (elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags) = build_multilayer_rect_mesh(L=args.L, H=args.H, cl=args.cl, order=1)

    # 2. Mapping DoFs
    unique_dofs_tags = np.unique(elemNodeTags)
    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
    
    # POur le graphe de la concentration moyenne à différents moments, on a besoin de faire le lien entre les DoFs et leurs coordonnées physiques (pour faire le profil linéaire)
    # On reshape les coordonnées pour avoir (N, 3)
    all_coords = nodeCoords.reshape(-1, 3)

    # On crée le tableau pour stocker les coordonnées ordonnées des DoFs
    dof_coords = np.zeros((num_dofs, 3))

    # Création d'un petit dictionnaire pour faire le lien entre le tag  
    # et la position dans la liste des coordonnées
    tag_to_idx = {tag: i for i, tag in enumerate(nodeTags)}

    # On remplit dof_coords en utilisant l'ordre des DoFs
    for i, tag in enumerate(unique_dofs_tags):
        idx = tag_to_idx[tag]
        dof_coords[i] = all_coords[idx]
    # --------------------------------------------------


    # 3. Assemblage
    K_total = lil_matrix((num_dofs, num_dofs))
    M_total = lil_matrix((num_dofs, num_dofs))

    for name, props in layer_props.items():
        try:
            elTypeL, elTagsL, elNodeTagsL, entityTag = getPhysical(name)
            xi, w, N, gN = prepare_quadrature_and_basis(elTypeL, 1)
            jac, det, coords = get_jacobians(elTypeL, xi, tag=entityTag)

            K_layer, _ = assemble_stiffness_and_rhs(
                elTagsL, elNodeTagsL, jac, det, coords, w, N, gN,
                kappa_fun=lambda x: props["D"], rhs_fun=lambda x: 0.0,
                tag_to_dof=tag_to_dof
            )
            M_layer = assemble_mass(elTagsL, elNodeTagsL, det, w, N, tag_to_dof)
            K_total += K_layer + props["kr"] * M_layer
            M_total += M_layer
        except: continue

    # Condition Robin
    (elemTypeV, elemTagsV, elemNodeTagsV, entityTagV) = getPhysical("VesselWall")
    xiV, wV, NV, gNV = prepare_quadrature_and_basis(elemTypeV, 1)
    jacV, detV, coordsV = get_jacobians(elemTypeV, xiV, tag=entityTagV)
    K_robin = assemble_robin_matrix(elemTagsV, elemNodeTagsV, detV, coordsV, wV, NV, 
                                    P=P, tag_to_dof=tag_to_dof, nn=num_dofs)
    
    # 4. Finalisation des matrices (Stiff + Robin)
    K = (K_total + K_robin).tocsr()
    
    # APPLICATION DU MASS LUMPING (Stabilisation)
    # On transforme la matrice de masse cohérente en matrice diagonale pour éviter de faire trop de liens avec les autres éléments lors de la résolution du système linéaire, ce qui peut causer des instabilités numériques (concentration négative).
    M_consistent = M_total.tocsr()
    M = diags(np.array(M_consistent.sum(axis=1)).flatten())
    

    # 5. Visualisation initiale
    fig_mesh, ax_mesh = plt.subplots(figsize=(10, 5))
    plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, ax=ax_mesh)
    
    ax_mesh.set_title("Maillage ")
    plt.tight_layout()
    plt.show()
    
    # 6. Simulation
    U = np.zeros(num_dofs)
    plt.ion()
    fig, ax = plt.subplots()

    # 1.))  Initialisation (pour graphe concentration moyenne vs temps)
    # Calcul de la masse totale pour normaliser (somme des éléments de M)
    mass_total = float(M.sum()) 
    # Listes pour stocker les données de temps et de concentration
    times = np.zeros(args.nsteps + 1)
    avg_conc = np.zeros(args.nsteps + 1)
    # Valeur initiale à t=0
    times[0] = 0.0
    avg_conc[0] = M.dot(U).sum() / mass_total


    #pour le graphe de la concentration moyenne a differents moments 
    # Initialisation pour le suivi à des moments précis
    target_times = [100.0, 200.0, 300.0] 
    snapshots = {}

    F_robin = assemble_robin_rhs(
            elemTagsV, elemNodeTagsV, detV, coordsV, wV, NV,
            P=P,
            c_plasma_fun=lambda x: C_PLASMA,
            tag_to_dof=tag_to_dof,
            nn=num_dofs
        )

    cbar = None
    for step in range(args.nsteps):

        U = theta_step(M, K, F_robin, F_robin, U,
                       dt=args.dt, theta=args.theta,
                       dirichlet_dofs=np.array([], dtype=int),
                       dir_vals_np1=np.array([], dtype=float))

        print(f"Étape {step} | Max U : {np.max(U):.4e}")

        # 2.)) calcul de la concentration moyenne à chaque étape:
        times[step + 1] = (step + 1) * args.dt
        avg_conc[step + 1] = M.dot(U).sum() / mass_total

        for t_target in target_times:
            if abs(times[step + 1] - t_target) < (args.dt / 2):
                snapshots[t_target] = U.copy()

        ax.clear()
        vmin, vmax = 0, C_PLASMA
        contour = plot_fe_solution_2d(
            elemTags, elemNodeTags, nodeCoords, nodeTags, U,
            tag_to_dof, ax=ax, cmap='turbo', vmin=vmin, vmax=vmax   
        )
        
        if cbar is None:
            cbar = plt.colorbar(contour, ax=ax, label="Concentration")

        ax.set_aspect('auto')
        ax.set_title(f"Temps: {step * args.dt:.0f} s")
        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.show()

    # 3. trace le graphique de l'évolution de la concentration moyenne dans le tissu au cours du temps:
    plt.figure(figsize=(8, 5))
    plt.plot(times, avg_conc, label='Conc. Moyenne', color='blue', linewidth=2)
    plt.xlabel('Temps [s]')
    plt.ylabel('Concentration Moyenne [mol/m³]')
    plt.title('Évolution de la concentration moyenne dans le tissu')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 4. trace les profils de concentration à différents moments (pour voir comment la doxorubicine pénètre dans le tissu au cours du temps):
    plt.figure()
    for t_val, U_snap in snapshots.items():
        rho, profile = linear_profile(dof_coords, U_snap, args.L) 
        plt.plot(rho, profile, label=f't = {t_val:.0f} s')
    plt.xlabel('Distance normalisée (0=vaisseau, 1=limite)')
    plt.ylabel('Concentration [mol/m³]')
    plt.title('Profil de pénétration par couches')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()