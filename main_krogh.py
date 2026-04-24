# main_krogh.py
"""
Krogh cylinder model — drug diffusion from capillary to tissue.

PDE:
    dc/dt - D * Delta(c) + kr * c = 0 in Omega

Boundary conditions:
    Gamma_v: -D grad(c).n = P * (c_plasma - c)  [Robin]
    Gamma_t: -D grad(c).n = 0                   [Neumann homog.]

Matrix form:
    M dU/dt + (K_diff + K_reac + R_robin) U = G_robin
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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
from plot_utils import plot_mesh_2d, plot_fe_solution_2d


def radial_profile(dof_coords, U, Rv, Rt, nbins=60):
    coords = np.asarray(dof_coords, dtype=float)
    r = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)

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

    # Géométrie
    parser.add_argument("--Rv", type=float, default=8e-6, help="Vessel radius [m]")
    parser.add_argument("--Rt", type=float, default=1e-4, help="Tissue radius [m]")
    parser.add_argument("--clv", type=float, default=0.5e-6, help="Mesh size at vessel wall [m]")
    parser.add_argument("--clt", type=float, default=5e-6, help="Mesh size at tissue boundary [m]")

    # Physique
    parser.add_argument("--D", type=float, default=3.6e-10, help="Diffusion coefficient [m^2/s]")
    parser.add_argument("--P", type=float, default=3.0e-8, help="Vascular permeability [m/s]")
    parser.add_argument("--kr", type=float, default=2.0e-4, help="Reaction/uptake rate [s^-1]")

    # Discrétisation
    parser.add_argument("--order", type=int, default=1, help="Polynomial order")
    parser.add_argument("--theta", type=float, default=1.0, help="Theta-scheme")
    parser.add_argument("--dt", type=float, default=10.0, help="Time step [s]")
    parser.add_argument("--nsteps", type=int, default=360, help="Number of time steps")

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

    # ------------------------------------------------------------------
    # 1. Maillage
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Mapping Gmsh node tags -> DoFs compacts
    # ------------------------------------------------------------------
    unique_dofs_tags = np.unique(elemNodeTags)
    nn = len(unique_dofs_tags)

    max_tag = int(np.max(nodeTags))
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    tag_to_idx = np.zeros(max_tag + 1, dtype=int)

    all_coords = nodeCoords.reshape(-1, 3)
    dof_coords = np.zeros((nn, 3), dtype=float)

    for i, tag in enumerate(nodeTags):
        tag_to_idx[int(tag)] = i

    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i] = all_coords[tag_to_idx[int(tag)]]

    # ------------------------------------------------------------------
    # 3. Quadrature volume
    # ------------------------------------------------------------------
    xi, w, N, gN = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords = get_jacobians(elemType, xi)

    # ------------------------------------------------------------------
    # 4. Quadrature bord Robin
    # ------------------------------------------------------------------
    elemTypeV, elemTagsV, elemNodeTagsV, entityTagV = getPhysical("VesselWall")

    xiV, wV, NV, _ = prepare_quadrature_and_basis(elemTypeV, args.order)
    _, detV, coordsV = get_jacobians(elemTypeV, xiV, tag=entityTagV)

    # ------------------------------------------------------------------
    # 5. Assemblage
    # ------------------------------------------------------------------

    M_lil = assemble_mass(
        elemTags,
        elemNodeTags,
        det,
        w,
        N,
        tag_to_dof,
    )

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

    K_reac_lil = kr * M_lil

    K_robin_lil = assemble_robin_matrix(
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

    K_total_lil = K_diff_lil + K_reac_lil + K_robin_lil

    M = M_lil.tocsr()
    K = K_total_lil.tocsr()

    G_robin = assemble_robin_rhs(
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

    F = F0 + G_robin

    # ------------------------------------------------------------------
    # 6. Condition initiale
    # ------------------------------------------------------------------
    U = np.zeros(nn, dtype=float)

    # ------------------------------------------------------------------
    # 7. Préparation résultats
    # ------------------------------------------------------------------
    print(f"Starting time integration: {args.nsteps} steps, dt={dt} s, T={T:.1f} s")
    print(f"D={D:.2e} m²/s, P={P:.2e} m/s, kr={kr:.2e} s⁻¹")
    print(f"Rv={Rv:.2e} m, Rt={Rt:.2e} m")

    mass_total = float(M.sum())

    times = np.zeros(args.nsteps + 1, dtype=float)
    avg_conc = np.zeros(args.nsteps + 1, dtype=float)

    times[0] = 0.0
    avg_conc[0] = M.dot(U).sum() / mass_total

    target_profile_times = [100.0, 500.0, 1000.0, T]
    snapshot_steps = sorted(
        {
            min(args.nsteps, max(0, int(round(target_time / dt))))
            for target_time in target_profile_times
        }
    )

    snapshots = {}

    # ------------------------------------------------------------------
    # 8. Affichage interactif
    # ------------------------------------------------------------------
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 6))
    cbar = None

    # Échelle adaptée aux résultats en mmol/m³.
    # La solution finale est autour de 0.6 mmol/m³ avec les paramètres par défaut.
    c_max_display = 0.6

    # ------------------------------------------------------------------
    # 9. Boucle temporelle
    # ------------------------------------------------------------------
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

        if step % 10 == 0 or step == args.nsteps - 1:
            print(
                f"step {step + 1}, t={t_np1:.1f}, "
                f"min={U.min():.3e}, max={U.max():.3e}"
            )

            ax.clear()

            U_display = U * 1e3  # mol/m³ -> mmol/m³

            contour = plot_fe_solution_2d(
                elemTags=elemTags,
                elemNodeTags=elemNodeTags,
                nodeTags=nodeTags,
                nodeCoords=nodeCoords,
                U=U_display,
                tag_to_dof=tag_to_dof,
                show_mesh=False,
                ax=ax,
                vmin=U_display.min(),
                vmax=U_display.max(),
                cmap="viridis",
            )

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

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

    plt.ioff()

    # ------------------------------------------------------------------
    # 10. Concentration moyenne
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots()
    ax1.plot(times, avg_conc * 1e3, "-o", markersize=3)
    ax1.set_xlabel("Temps [s]")
    ax1.set_ylabel("Concentration moyenne [mmol/m³]")
    ax1.set_title("Concentration moyenne du tissu en fonction du temps")
    ax1.grid(True)

    # ------------------------------------------------------------------
    # 11. Profils radiaux
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots()

    for time in sorted(snapshots):
        rho, profile = radial_profile(dof_coords, snapshots[time], Rv, Rt)

        if rho.size > 0:
            ax2.plot(rho, profile / C_PLASMA, label=f"t={time:.0f}s")

    ax2.set_xlabel(r"Rayon normalisé $\rho=(r-R_v)/(R_t-R_v)$")
    ax2.set_ylabel(r"$c/c_{\mathrm{plasma}}$")
    ax2.set_title("Profils radiaux de concentration")
    ax2.legend()
    ax2.grid(True)

    print("Done.")

    plt.show()
    gmsh_finalize()


if __name__ == "__main__":
    main()