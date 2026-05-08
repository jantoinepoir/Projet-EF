# main_krogh.py
"""
Modèle de Krogh : diffusion de la doxorubicine depuis un vaisseau vers le tissu.

Équation résolue :
    dc/dt - div(D grad(c)) + kr c = 0 dans Ω

Conditions aux limites :
    Γ_v : -D grad(c).n = P(c_plasma - c)   condition de Robin
    Γ_t : -D grad(c).n = 0                 condition de Neumann homogène

Forme matricielle :
    M dU/dt + K U = F

avec :
    K = K_diff + K_reac + R
    F = F0 + G

où R et G viennent de la condition de Robin.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Circle

from gmsh_utils import (
    getPhysical,
    gmsh_init,
    gmsh_finalize,
    prepare_quadrature_and_basis,
    build_annular_mesh,
    get_jacobians,
)

from stiffness import (
    assemble_stiffness_and_rhs,
    assemble_robin_matrix,
    assemble_robin_rhs,
)

from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import plot_mesh_2d, plot_fe_solution_2d_krogh


def radial_profile(dof_coords, U, Rv, Rt, nbins=60):
    """
    Calcule un profil radial moyen de concentration.

    Les nœuds sont regroupés selon leur rayon normalisé :
        rho = (r - Rv) / (Rt - Rv)

    Pour chaque intervalle radial, on calcule la moyenne des concentrations
    nodales situées dans cet intervalle.
    """
    coords = np.asarray(dof_coords, dtype=float)
    r = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)

    # On ne considère que les nœuds situés dans le domaine physique (entre Rv et Rt)
    mask = (r >= Rv) & (r <= Rt)
    if np.sum(mask) == 0:
        return np.array([]), np.array([])

    r = r[mask]
    U = np.asarray(U, dtype=float)[mask]

    rho = (r - Rv) / (Rt - Rv)

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
    parser = argparse.ArgumentParser(description="Krogh cylinder – drug diffusion")

    # Paramètres géométriques et de maillage
    parser.add_argument("--Rv", type=float, default=8e-6, help="Vessel radius [m]")
    parser.add_argument("--Rt", type=float, default=1e-4, help="Tissue radius [m]")
    parser.add_argument("--clv", type=float, default=0.5e-6, help="Mesh size at vessel wall [m]")
    parser.add_argument("--clt", type=float, default=5e-6, help="Mesh size at tissue boundary [m]")

    # Paramètres physiques
    parser.add_argument("--D", type=float, default=3.73e-10, help="Diffusion coefficient [m^2/s]")
    parser.add_argument("--P", type=float, default=3.0e-8, help="Vascular permeability [m/s]")
    parser.add_argument("--kr", type=float, default=2.0e-4, help="Reaction/uptake rate [s^-1]")

    # Paramètres temporels
    parser.add_argument("--order", type=int, default=1, help="Polynomial order")
    parser.add_argument("--theta", type=float, default=1.0, help="Theta-scheme")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step [s]")
    parser.add_argument("--nsteps", type=int, default=100, help="Number of time steps")

    args = parser.parse_args()

    Rv = args.Rv
    Rt = args.Rt
    D = args.D
    P = args.P
    kr = args.kr
    dt = args.dt
    theta = args.theta
    T = dt * args.nsteps

    C_PLASMA = 5.0e-3  # mol/m³


    # 1. Maillage du cylindre de Krogh
    gmsh_init("krogh_cylinder")
    (
        elemType,
        nodeTags,
        nodeCoords,
        elemTags,
        elemNodeTags,
        bnds,
        bnds_tags,
    ) = build_annular_mesh(
        Rv=Rv,
        Rt=Rt,
        cl_v=args.clv,
        cl_t=args.clt,
        order=args.order,
    )

    plot_mesh_2d(
        elemType,
        nodeTags,
        nodeCoords,
        elemTags,
        elemNodeTags,
        bnds,
        bnds_tags,
    )


    # 2. Construction de la correspondance entre tags Gmsh et indices de degrés de liberté
    unique_dofs_tags = np.unique(elemNodeTags)
    nn = len(unique_dofs_tags)

    max_tag = int(np.max(nodeTags))
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    tag_to_idx = np.zeros(max_tag + 1, dtype=int)

    all_coords = nodeCoords.reshape(-1, 3)
    dof_coords = np.zeros((nn, 3), dtype=float)

    # tag_to_idx relie un tag Gmsh à sa position dans nodeCoords
    for i, tag in enumerate(nodeTags):
        tag_to_idx[int(tag)] = i

    # tag_to_dof relie un tag Gmsh à son indice de degré de liberté 
    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i] = all_coords[tag_to_idx[int(tag)]]


    # 3. Données de quadrature volume
    xi, w, N, gN = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords = get_jacobians(elemType, xi)


    # 4. Données de quadrature bord Robin (frontière du vaisseau)
    elemTypeV, elemTagsV, elemNodeTagsV, entityTagV = getPhysical("VesselWall")

    xiV, wV, NV, _ = prepare_quadrature_and_basis(elemTypeV, args.order)
    _, detV, coordsV = get_jacobians(elemTypeV, xiV, tag=entityTagV)


    # 5. Assemblage des matrices et vecteurs
    #Matrice de masse M
    M_lil = assemble_mass(
        elemTags,
        elemNodeTags,
        det,
        w,
        N,
        tag_to_dof,
    )

    # Matrice de diffusion K_diff et second membre volumique F0
    # Ici F0 est nul car il n'y a pas de source volumique
    K_diff_lil, F0 = assemble_stiffness_and_rhs(
        elemTags,
        elemNodeTags,
        jac,
        det,
        coords,
        w,
        N,
        gN,
        kappa_fun=lambda x: D,
        rhs_fun=lambda x: 0.0,
        tag_to_dof=tag_to_dof,
    )

    # Matrice de réaction K_reac = kr * M
    K_reac_lil = kr * M_lil

    # Matrice de Robin R, ajoutée au membre gauche
    R_lil = assemble_robin_matrix(
        elemTagsV,
        elemNodeTagsV,
        detV,
        coordsV,
        wV,
        NV,
        P=P,
        tag_to_dof=tag_to_dof,
        nn=nn,
    )

    K_lil = K_diff_lil + K_reac_lil + R_lil

    M = M_lil.tocsr()
    K = K_lil.tocsr()

    # Second membre de Robin G, ajouté au membre droit
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

    # Second membre total 
    F = F0 + G


    # 6. Condition initiale
    U = np.zeros(nn, dtype=float)


    # 7. Préparation du stockage desrésultats
    print(f"Starting time integration: {args.nsteps} steps, dt={dt} s, T={T:.1f} s")
    print(f"D={D:.2e} m²/s, P={P:.2e} m/s, kr={kr:.2e} s⁻¹")
    print(f"Rv={Rv:.2e} m, Rt={Rt:.2e} m")

    mass_total = float(M.sum())

    times = np.zeros(args.nsteps + 1, dtype=float)
    avg_conc = np.zeros(args.nsteps + 1, dtype=float)

    times[0] = 0.0
    avg_conc[0] = M.dot(U).sum() / mass_total

    # Instants auxquels on garde une copie pour tracer les profils radiaux
    target_profile_times = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0, T]
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

    # Échelle adaptée aux résultats en mmol/m³.
    c_max_display = 0.02


    # 9. Boucle temporelle
    for step in range(args.nsteps):
        t_np1 = (step + 1) * dt

        U = theta_step(
            M,
            K,
            F,
            F,
            U,
            dt=dt,
            theta=theta,
            dirichlet_dofs=[],
            dir_vals_np1=[]
        )

        times[step + 1] = t_np1
        avg_conc[step + 1] = M.dot(U).sum() / mass_total

        if step + 1 in snapshot_steps:
            snapshots[t_np1] = U.copy()

        if step % 2.0 == 0 or step == args.nsteps - 1:
            ax.clear()

            # Conversion mol/m³ -> mmol/m³ pour l'affichage.
            U_display = np.clip(U * 1e3, 1e-7, None)  # mol/m³ -> mmol/m³, évite log(0)
            
            contour = plot_fe_solution_2d_krogh(
                elemTags=elemTags,
                elemNodeTags=elemNodeTags,
                nodeTags=nodeTags,
                nodeCoords=nodeCoords,
                U=U_display,
                tag_to_dof=tag_to_dof,
                show_mesh=False,
                ax=ax,
                vmin=0.0,
                vmax=c_max_display,
                cmap='viridis',
            )

            # Le disque central correspondant au vaisseau est masqué pour ne pas fausser l'échelle de couleur
            hole = Circle(
                (0.0, 0.0),
                Rv,
                facecolor="white",
                edgecolor="white",
                linewidth=1.0,
                zorder=20,
            )
            ax.add_patch(hole)

            if cbar is None:
                cbar = fig.colorbar(contour, ax=ax, label="Concentration [mmol/m³]")
            else:
                cbar.update_normal(contour)

            ax.set_title(
                f"Cylindre de Krogh — Doxorubicine — t = {t_np1:.0f} s\n"
                f"D={D:.1e} m²/s, P={P:.1e} m/s, kr={kr:.1e} s⁻¹\n"
                f"min={U_display.min():.3f}, max={U_display.max():.3f} mmol/m³"
            )

            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_xlim([-1.1 * Rt, 1.1 * Rt])
            ax.set_ylim([-1.1 * Rt, 1.1 * Rt])
            ax.set_aspect("equal")

            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)

            ticks = [-Rt, -Rt / 2, 0.0, Rt / 2, Rt]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            fig.savefig(f"diffusion_krogh_{int(t_np1)}s.png", dpi=300, bbox_inches="tight")

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

    plt.ioff()
    plt.show()

    # 10. Courbe de concentration moyenne
    fig1, ax1 = plt.subplots()
    ax1.plot(times, avg_conc * 1e3, "-o", markersize=3)
    ax1.set_xlabel("Temps [s]")
    ax1.set_ylabel("Concentration moyenne [mmol/m³]")
    ax1.set_title("Concentration moyenne du tissu en fonction du temps")
    ax1.grid(True)


    # 11. Profils radiaux normalisés
    fig2, ax2 = plt.subplots()

    for time in sorted(snapshots):
        rho, profile = radial_profile(dof_coords, snapshots[time], Rv, Rt)

        if rho.size > 0:
            ax2.plot(rho, profile / C_PLASMA, label=f"t={time:.0f}s")

    ax2.set_xlabel(r"Rayon normalisé $\rho=(r-R_v)/(R_t-R_v)$")
    ax2.set_ylabel(r"$c/c_{\mathrm{plasma}}$")
    ax2.set_title("Profils de pénétration radiaux")
    ax2.legend()
    ax2.grid(True)
    
    fig1.savefig("concentration_moyenne_krogh.png", dpi=300, bbox_inches="tight")
    fig2.savefig("profils_krogh.png", dpi=300, bbox_inches="tight")

    print("Done.")

    plt.show()
    gmsh_finalize()


if __name__ == "__main__":
    main()