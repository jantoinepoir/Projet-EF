#main_multilayer.py
"""
Modèle multicouche : diffusion de la doxorubicine depuis un vaisseau vers un tissu hétérogène.

Équation résolue dans chaque couche i :
    dc/dt - div(D_i grad(c)) + kr_i c = 0

Conditions aux limites :
    VesselWall : -D grad(c).n = P(c_plasma - c)   condition de Robin
    OuterBoundary : -D grad(c).n = 0              condition de Neumann homogène

Forme matricielle :
    M dU/dt + K U = F

avec :
    K = somme_i(K_diff_i + K_reac_i) + R
    F = F0 + G

où R et G viennent de la condition de Robin.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
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
from plot_utils import plot_fe_solution_2d_multicouche, plot_mesh_2d


def linear_profile(dof_coords, U, L, nbins=60):
    """
    Calcule un profil moyen selon la direction x.

    Les nœuds sont regroupés selon la distance normalisée :
        rho = x / L

    Pour chaque intervalle, on calcule la moyenne des concentrations nodales
    situées dans cet intervalle.
    """
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



def print_diffusion_metrics(layer_props, L, ratios):
    """
    Affiche deux indicateurs utiles pour interpréter les couches :
    - une vitesse apparente de diffusion : v_app = 2D/d ;
    - une longueur de pénétration : lambda = sqrt(D/kr).
    """
    print("\n" + "="*50)
    print("ANALYSE DES PARAMÈTRES DE DIFFUSION")
    print("="*50)
    print(f"{'Couche':<15} | {'Vitesse (μm/s)':<15} | {'Pénétration (mm)':<15}")
    print("-"*50)
    
    for i, (name, props) in enumerate(layer_props.items()):
        d = ratios[i] * L  # Épaisseur en mètres
        D = props["D"]
        kr = props["kr"]
        
        # Calculs
        # Temps caractéristique : t = d² / (2*D)
        # Vitesse moyenne : v = d / t = 2*D / d
        v_moy = (2 * D) / d if d > 0 else 0
        # Longueur de pénétration : lambda = sqrt(D / kr)
        lambd = np.sqrt(D / kr) if kr > 0 else float('inf')
        
        # Conversion pour l'affichage (m -> μm et m -> mm)
        print(f"{name:<15} | {v_moy * 1e6:<15.2f} | {lambd * 1e3:<15.2f}")
    
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser()
    
    # Paramètres géométriques et de maillage
    parser.add_argument("--L", type=float, default=0.00015)
    parser.add_argument("--H", type=float, default=0.00010)
    parser.add_argument("--cl", type=float, default=0.000015)
    
    # Paramètres temporels
    parser.add_argument("--order", type=int, default=1, help="Ordre des éléments finis")
    parser.add_argument("--dt", type=float, default=30.0)
    parser.add_argument("--nsteps", type=int, default=1000)
    parser.add_argument("--theta", type=float, default=1.0)
    
    args = parser.parse_args()

    L = args.L
    H = args.H
    dt = args.dt
    theta = args.theta
    T = dt * args.nsteps

    C_PLASMA = 5.0e-3  # mol/m³
    P = 3.0e-8         # m/s (DOX libre, Eikenberry Table 1)

    #Propiétés des différentes couches (diffusion et capture)
    layer_props = {#D est le coefficient de diffusion, kr est le taux de capture (consommation) de la doxorubicine par les cellules du tissu
    "Layer1": {"D": 1.58e-10, "kr": 2.0e-4},  # péri-vasculaire dense #tissu dense près du vaisseau (la diffusion y est plus lente car forte capture)
    "Layer2": {"D": 4.17e-10, "kr": 1.0e-4},  # tumoral proliférant #tissu plus lâche au milieu/tumoral standard (la diffusion y est plus rapide car moins de capture)
    "Layer3": {"D": 9.96e-10, "kr": 5.0e-6},  # nécrosé #tissu tumoral nécrosé au centre (la diffusion y est très rapide car quasi pas de capture, moins de cellules viables)--> la pénétration y est facilitée
    }

       
    # 1. Maillage du domaine multicouche
    gmsh_init("skin_diffusion")
    (
        elemType, 
        nodeTags, 
        nodeCoords, 
        elemTags, 
        elemNodeTags, 
        bnds, 
        bnds_tags
    ) = build_multilayer_rect_mesh(
        L=L, 
        H=H, 
        cl=args.cl, 
        order=args.order)
    
    fig_mesh, ax_mesh = plt.subplots(figsize=(10, 5))
    plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags, ax=ax_mesh)
    
    ax_mesh.set_title("Maillage multicouche")
    plt.tight_layout()
    plt.show()


    # 2. Construction de la correspondance entre tags Gmsh et indices de degrés de liberté
    unique_dofs_tags = np.unique(elemNodeTags)
    nn = len(unique_dofs_tags)
    
    max_tag = int(np.max(nodeTags))
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    tag_to_idx = np.zeros(max_tag + 1, dtype=int)
    
    # On reshape les coordonnées pour avoir (N, 3)
    all_coords = nodeCoords.reshape(-1, 3)
    dof_coords = np.zeros((nn, 3))

    # tag_to_idx relie un tag Gmsh à sa position dans nodeCoords
    for i, tag in enumerate(nodeTags):
        tag_to_idx[int(tag)] = i

    # tag_to_dof relie un tag Gmsh à son indice de degré de liberté
    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i] = all_coords[tag_to_idx[int(tag)]]


    # 3. Assemblage couche par couche
    K_lil = lil_matrix((nn, nn))
    M_lil = lil_matrix((nn, nn))
    F0 = np.zeros(nn, dtype=float)

    for name, props in layer_props.items():
        try:
            elTypeL, elTagsL, elNodeTagsL, entityTag = getPhysical(name)
            
            xi, w, N, gN = prepare_quadrature_and_basis(elTypeL, 1)
            jac, det, coords = get_jacobians(elTypeL, xi, tag=entityTag)

            # Matrice de diffusion de la couche et second membre volumique.
            # Ici F_layer est nul car il n'y a pas de source volumique.
            K_diff_lil, F_layer = assemble_stiffness_and_rhs(
                elTagsL, elNodeTagsL, jac, det, coords, w, N, gN,
                kappa_fun=lambda x: props["D"], rhs_fun=lambda x: 0.0,
                tag_to_dof=tag_to_dof
            )

            # Matrice de masse de la couche
            M_layer_lil = assemble_mass(elTagsL, elNodeTagsL, det, w, N, tag_to_dof)
            
            # Matrice de réaction de la couche : K_reac_i = kr_i M_i.
            K_reac_lil = props["kr"] * M_layer_lil
            
            K_lil += K_diff_lil + K_reac_lil
            M_lil += M_layer_lil
            F0 += F_layer

        except: continue

    # 4. Données de quadrature sur la frontière VesselWall
    (elemTypeV, elemTagsV, elemNodeTagsV, entityTagV) = getPhysical("VesselWall")
    
    xiV, wV, NV, gNV = prepare_quadrature_and_basis(elemTypeV, 1)
    jacV, detV, coordsV = get_jacobians(elemTypeV, xiV, tag=entityTagV)
    
    
    # 5. Assemblage de la condition de Robin
    # Matrice de Robin R, ajoutée au mambre gauche
    R_lil = assemble_robin_matrix(elemTagsV, elemNodeTagsV, detV, coordsV, wV, NV, 
                                    P=P, tag_to_dof=tag_to_dof, nn=nn)
    
    K_lil += R_lil

    K = K_lil.tocsr()

    # Second membre de Robin G, ajouté au membre droit.
    G = assemble_robin_rhs(
        elemTagsV,
        elemNodeTagsV,
        detV,
        coordsV,
        wV,
        NV,
        P=P,
        c_plasma_fun=lambda x: C_PLASMA,
        tag_to_dof=tag_to_dof,
        nn=nn,
    )

    F = F0 + G
    
    # Mass lumping : la matrice de masse cohérente est remplacée par une matrice diagonale obtenue par sommation des lignes.
    M_consistent = M_lil.tocsr()
    M = diags(np.array(M_consistent.sum(axis=1)).flatten())
    

    # 6. Condition initiale
    U = np.zeros(nn)

    # 7. Préparation du stockage desrésultats
    print(f"Starting time integration: {args.nsteps} steps, dt={dt} s, T={T:.1f} s")
    print(f"P={P:.2e} m/s")
    print(f"L={L:.2e} m, H={H:.2e} m")

    mass_total = float(M.sum()) 

    times = np.zeros(args.nsteps + 1)
    avg_conc = np.zeros(args.nsteps + 1)

    times[0] = 0.0
    avg_conc[0] = M.dot(U).sum() / mass_total


    # Instants auxquels on garde une copie pour tracer les profils
    target_profile_times = [100.0, 500.0, 1000.0, 2000.0] 
    snapshot_steps = sorted(
        {
            min(args.nsteps, max(0, int(round(target_time / dt))))
            for target_time in target_profile_times
        }
    )

    snapshots = {}


    # 8. Affichage interactif de la diffusion
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 6))
    cbar = None

    for step in range(args.nsteps):

        U = theta_step(M, K, F, F, U,
                       dt=args.dt, theta=args.theta,
                       dirichlet_dofs=np.array([], dtype=int),
                       dir_vals_np1=np.array([], dtype=float))


        times[step + 1] = (step + 1) * dt
        avg_conc[step + 1] = M.dot(U).sum() / mass_total

        for t_target in target_profile_times:
            if abs(times[step + 1] - t_target) < (args.dt / 2):
                snapshots[t_target] = U.copy()

        ax.clear()
        vmin, vmax = 0, 1
        contour = plot_fe_solution_2d_multicouche(
            elemTags, elemNodeTags, nodeCoords, nodeTags, U,
            tag_to_dof, ax=ax, cmap='turbo', vmin=vmin, vmax=vmax   
        )

        #on rajoute des lignes pour marquer les interfaces entre les couches
        x_bounds = np.cumsum([0, 0.13, 0.53, 0.34]) * args.L
        for x in x_bounds[1:-1]:  # on ne marque pas les bords extérieurs
            ax.axvline(x=x, color='black', linestyle='-', linewidth=1)         


        if cbar is None:
            cbar = plt.colorbar(contour, ax=ax, label="Concentration")

        ax.set_aspect('auto')
        ax.set_title(f"Temps: {(step + 1) * args.dt:.0f} s")
        plt.draw()
        plt.pause(0.1)


    plt.ioff()
    plt.show()

    # 9. Courbe de concentration moyenne
    fig1, ax1 = plt.subplots()
    ax1.plot(times, avg_conc * 1e3, "-o", markersize=3)
    ax1.set_xlabel("Temps [s]")
    ax1.set_ylabel("Concentration moyenne [mmol/m³]")
    ax1.set_title("Concentration moyenne du tissu en fonction du temps")
    ax1.grid(True)

    # 10. Profils de pénétration par couches
    fig2, ax2 = plt.subplots()

    for t_val, U_snap in snapshots.items():
        rho, profile = linear_profile(dof_coords, U_snap, args.L)
        ax2.plot(rho, profile / C_PLASMA, label=f"t = {t_val:.0f} s")

    layer_ratios = [0.13, 0.53, 0.34]
    layer_bounds = np.cumsum(layer_ratios)

    for x_layer in layer_bounds[:-1]:
        ax2.axvline(
            x=x_layer,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
        )

    y_text = 0.225
    x_couche1 = 0.045
    x_couche2 = (0.13 + 0.66) / 2
    x_couche3 = (0.66 + 1.0) / 2

    ax2.text(x_couche1, y_text, "Couche 1", ha='center', va='center', fontsize=11)
    ax2.text(x_couche2, y_text, "Couche 2", ha='center', va='center', fontsize=11)
    ax2.text(x_couche3, y_text, "Couche 3", ha='center', va='center', fontsize=11)

    ax2.set_xlabel("Distance normalisée (0 = vaisseau, 1 = limite)")
    ax2.set_ylabel(r"$c/c_{\mathrm{plasma}}$")
    ax2.set_title("Profils de pénétration par couches")
    ax2.legend()
    ax2.grid(True)
    
    fig1.savefig("concentration_moyenne_multicouche.png", dpi=300, bbox_inches="tight")
    fig2.savefig("profils_multicouche.png", dpi=300, bbox_inches="tight")
    
    print_diffusion_metrics(layer_props, args.L, layer_ratios)
    print("Done.")
    plt.show()
    gmsh_finalize()



if __name__ == "__main__":
    main()