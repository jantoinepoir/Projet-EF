# main_krogh.py
"""
Krogh cylinder model — drug diffusion from capillary to tissue.

PDE:
    dc/dt - D * Delta(c) = 0    in Omega (annular tissue domain)

Boundary conditions:
    Gamma_v (inner, vessel wall):   -D grad(c).n = P * (c_plasma(t) - c)   [Robin]
    Gamma_t (outer, tissue limit):  -D grad(c).n = 0                        [Neumann homog.]

Discretisation:
    Space  : Lagrange P1 (or P2) triangles via Gmsh
    Time   : theta-scheme  (theta=1 => implicit Euler, theta=0.5 => Crank-Nicolson)

Usage examples:
    python main_krogh.py                        # default parameters
    python main_krogh.py --theta 0.5 --dt 5e-3  # Crank-Nicolson
    python main_krogh.py -order 2               # P2 elements
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from gmsh_utils import (
    getPhysical,gmsh_init, gmsh_finalize,
    prepare_quadrature_and_basis,
    build_annular_mesh,
    get_jacobians,
    border_dofs_from_tags 
    
)
from stiffness import (
    assemble_stiffness_and_rhs,
    assemble_robin_matrix,
    assemble_robin_rhs,
)
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import setup_interactive_figure, plot_mesh_2d, plot_fe_solution_2d


def main():
    parser = argparse.ArgumentParser(description="Krogh cylinder – drug diffusion (Gmsh FE)")

<<<<<<< HEAD
    # Géométrie
    parser.add_argument("--Rv",  type=float, default=8e-6,   help="Vessel radius [m]")
    parser.add_argument("--Rt",  type=float, default=1e-4,   help="Tissue radius [m]")
    parser.add_argument("--clv", type=float, default=0.5e-6, help="Mesh size at vessel wall")
    parser.add_argument("--clt", type=float, default=5e-6,   help="Mesh size at tissue boundary")
=======
    # Geometry
    parser.add_argument("--Rv",  type=float, default=8e-5,  help="Vessel radius [m]")
    parser.add_argument("--Rt",  type=float, default=50e-5, help="Tissue radius [m]")
    parser.add_argument("--clv", type=float, default=1e-6,  help="Mesh size at vessel wall")
    parser.add_argument("--clt", type=float, default=5e-6,  help="Mesh size at tissue boundary")
>>>>>>> 42d070d189f358df25c9fde730f4d1c5890ee4bc

    # Physique
    # Remplacer dans le parser :
    parser.add_argument("--D",  type=float, default=2.4e-11, help="Diffusion coefficient [m^2/s]")
    parser.add_argument("--P",  type=float, default=3.0e-8,  help="Vascular permeability [m/s]")
    parser.add_argument("--kr", type=float, default=2.0e-4,  help="Reaction/uptake rate [s^-1]")
    parser.add_argument("--order", type=int, default=1, help="Polynomial order (1=P1, 2=P2)")
    parser.add_argument("--theta", type=float, default=1.0, help="Theta-scheme: 1=Euler implicite, 0.5=Crank-Nicolson")

    # Temps  (T = 3600 s avec dt=10 s => 360 pas)
    parser.add_argument("--dt",     type=float, default=10.0)
    parser.add_argument("--nsteps", type=int,   default=360)

    # ------------------------------------------------------------------
    # Physical parameters
    # ------------------------------------------------------------------
    args = parser.parse_args() 
    D=   args.D
    P  = args.P
    kr = args.kr
    T  = args.dt * args.nsteps   # total simulation time  [s]

    # Plasma concentration: step function at t=0 (bolus injection)
    # Feel free to change to a time-varying profile.
    C_PLASMA = 5.0e-3   # mol/m³  (5 µM après correction unités)

    def c_plasma(t):
        """Concentration plasmatique constante (pic post-injection maintenu)."""
        return C_PLASMA

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------
    gmsh_init("krogh_cylinder")

    (elemType, nodeTags, nodeCoords,
     elemTags, elemNodeTags,
     bnds, bnds_tags) = build_annular_mesh(
        Rv=args.Rv, Rt=args.Rt,
        cl_v=args.clv, cl_t=args.clt,
        order=args.order
    )

    # Show mesh with boundary labels
    plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags)

    # ------------------------------------------------------------------
    # DOF mapping: gmsh node tag  ->  compact index 0 … nn-1
    # ------------------------------------------------------------------
    unique_dofs_tags = np.unique(elemNodeTags)
    num_dofs = len(unique_dofs_tags)
    max_tag  = int(np.max(nodeTags))

    dof_coords  = np.zeros((num_dofs, 3))
    all_coords  = nodeCoords.reshape(-1, 3)
    tag_to_dof  = np.full(max_tag + 1, -1, dtype=int)
    tag_to_idx  = np.zeros(max_tag + 1, dtype=int)   # tag -> position in nodeTags array

    for i, tag in enumerate(nodeTags):
        tag_to_idx[int(tag)] = i

    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i] = all_coords[tag_to_idx[int(tag)]]

    nn = num_dofs

    # ------------------------------------------------------------------
    # Quadrature and Jacobians (volume elements)
    # ------------------------------------------------------------------
    xi,  w,  N,  gN  = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords = get_jacobians(elemType, xi)

    # ------------------------------------------------------------------
    # Boundary data for the vessel wall (Robin condition)
    # ------------------------------------------------------------------
    # getPhysical returns: elemType, elemTags, elemNodeTags, entityTag
    (elemTypeV, elemTagsV,
     elemNodeTagsV, entityTagV) = getPhysical("VesselWall")

    xiV,  wV,  NV,  gNV  = prepare_quadrature_and_basis(elemTypeV, args.order)
    jacV, detV, coordsV  = get_jacobians(elemTypeV, xiV, tag=entityTagV)

    # ------------------------------------------------------------------
    # Assemble constant matrices
    # ------------------------------------------------------------------

    # Volume: diffusion stiffness  K[i,j] = integral D grad(N_i).grad(N_j) dOmega
    # F0 = volume source term (zero here since f=0)
    K_vol_lil, F0 = assemble_stiffness_and_rhs(
        elemTags, elemNodeTags,
        jac, det, coords,
        w, N, gN,
        kappa_fun=lambda x: D,   # kappa = D (diffusion coeff.)
        rhs_fun=lambda x: 0.0,   # no volumetric source
        tag_to_dof=tag_to_dof
    )

    # Robin boundary: K_robin[i,j] = integral_Gamma_v  P * N_i * N_j ds
    # This is constant (does not depend on c_plasma), assembled once.
    K_robin_lil = assemble_robin_matrix(
        elemTagsV, elemNodeTagsV,
        detV, coordsV, wV, NV,
        P=P, tag_to_dof=tag_to_dof, nn=nn
    )

    # Mass matrix: M[i,j] = integral N_i * N_j dOmega
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)

    # K_reaction[i,j] = kr * integral N_i * N_j dOmega = kr * M
    K_total_lil = K_vol_lil + K_robin_lil + kr * M_lil

    # Convert to CSR for efficient linear algebra
    K = K_total_lil.tocsr()
    M = M_lil.tocsr()

    # ------------------------------------------------------------------
    # Initial condition: tissue is drug-free at t=0
    # ------------------------------------------------------------------
    U = np.zeros(nn, dtype=float)

    # ------------------------------------------------------------------
    # No Dirichlet BC in this model (Robin on Gamma_v, Neumann on Gamma_t)
    # ------------------------------------------------------------------
    dir_dofs = np.array([], dtype=int)
    dir_vals = np.array([], dtype=float)

    # ------------------------------------------------------------------
    # Time integration (theta-scheme)
    # ------------------------------------------------------------------
    _, ax = setup_interactive_figure()
    print(f"Starting time integration: {args.nsteps} steps, dt={args.dt} s, T={T:.1f} s")
    print(f"D={D:.2e} m²/s,  P={P:.2e} m/s,  Rv={args.Rv:.2e} m,  Rt={args.Rt:.2e} m")

    fig, ax = plt.subplots()
    plt.ion()
    c_max_display = C_PLASMA * 1e3

    for step in range(args.nsteps):
        t     = step * args.dt
        t_np1 = t + args.dt

        F_robin_n = assemble_robin_rhs(
            elemTagsV, elemNodeTagsV,
            detV, coordsV, wV, NV,
            P=P,
            c_plasma_fun=lambda x, _t=t: c_plasma(_t),
            tag_to_dof=tag_to_dof, nn=nn
        )
        F_robin_np1 = assemble_robin_rhs(
            elemTagsV, elemNodeTagsV,
            detV, coordsV, wV, NV,
            P=P,
            c_plasma_fun=lambda x, _t=t_np1: c_plasma(_t),
            tag_to_dof=tag_to_dof, nn=nn
        )

        Fn   = F0 + F_robin_n
        Fnp1 = F0 + F_robin_np1

        U = theta_step(
            M, K,
            Fn, Fnp1, U,
            dt=args.dt,
            theta=args.theta,
            dirichlet_dofs=dir_dofs,
            dir_vals_np1=dir_vals
        )

        if step % 5 == 0:
            fig.clf()
            ax = fig.add_subplot(111)

            U_display = np.clip(U * 1e3, 1e-7, None)  # mol/m³ -> mmol/m³, évite log(0)

            contour = plot_fe_solution_2d(
                elemNodeTags=elemNodeTags,
                nodeTags=nodeTags,
                nodeCoords=nodeCoords,
                U=U_display,
                tag_to_dof=tag_to_dof,
                show_mesh=False,
                ax=ax,
                vmin=1e-7,
                vmax=c_max_display,
                cmap='inferno',
            )

            plt.colorbar(contour, ax=ax, label="Concentration [mmol/m³]")
            ax.set_title(
                f"Cylindre de Krogh — Doxorubicine — t = {t_np1:.0f} s\n"
                f"D={D:.1e} m²/s,  P={P:.1e} m/s,  kr={kr:.1e} s⁻¹"
            )
            ax.set_xlabel("x  [m]")
            ax.set_ylabel("y  [m]")
            ax.set_xlim([-1.1*args.Rt, 1.1*args.Rt])
            ax.set_ylim([-1.1*args.Rt, 1.1*args.Rt])
            ax.axis('equal')
            plt.pause(0.01)

    print("Done.")
    plt.ioff()
    plt.show()
    gmsh_finalize()


if __name__ == "__main__":
    main()
