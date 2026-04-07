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

    # Geometry
    parser.add_argument("--Rv",  type=float, default=8e-6,  help="Vessel radius [m]")
    parser.add_argument("--Rt",  type=float, default=50e-6, help="Tissue radius [m]")
    parser.add_argument("--clv", type=float, default=1e-6,  help="Mesh size at vessel wall")
    parser.add_argument("--clt", type=float, default=5e-6,  help="Mesh size at tissue boundary")

    # Physics
    parser.add_argument("-D",    type=float, default=1e-10, help="Diffusion coefficient [m^2/s]")
    parser.add_argument("-P",    type=float, default=1e-6,  help="Vascular permeability [m/s]")

    # FE
    parser.add_argument("-order",  type=int,   default=1)
    parser.add_argument("--theta", type=float, default=1.0,
                        help="Theta-scheme: 1=implicit Euler, 0.5=Crank-Nicolson")
    parser.add_argument("--dt",    type=float, default=1.0)   # seconds
    parser.add_argument("--nsteps",type=int,   default=200)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Physical parameters
    # ------------------------------------------------------------------
    D  = args.D   # diffusion coefficient  [m^2/s]
    P  = args.P   # vascular permeability  [m/s]
    T  = args.dt * args.nsteps   # total simulation time  [s]

    # Plasma concentration: step function at t=0 (bolus injection)
    # Feel free to change to a time-varying profile.
    def c_plasma(t):
        """Uniform plasma concentration (assumed constant or slowly varying)."""
        return 1.0   # normalised units

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

    # Total stiffness:  K_total = K_vol + K_robin
    K_total_lil = K_vol_lil + K_robin_lil

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

    for step in range(args.nsteps):
        t    = step * args.dt
        t_np1 = t + args.dt

        # Robin RHS at time n:   F_robin_n[i] = integral P * c_plasma(t) * N_i ds
        F_robin_n   = assemble_robin_rhs(
            elemTagsV, elemNodeTagsV,
            detV, coordsV, wV, NV,
            P=P,
            c_plasma_fun=lambda x, _t=t:    c_plasma(_t),
            tag_to_dof=tag_to_dof, nn=nn
        )

        # Robin RHS at time n+1
        F_robin_np1 = assemble_robin_rhs(
            elemTagsV, elemNodeTagsV,
            detV, coordsV, wV, NV,
            P=P,
            c_plasma_fun=lambda x, _t=t_np1: c_plasma(_t),
            tag_to_dof=tag_to_dof, nn=nn
        )

        # Total RHS = volume source (zero) + Robin flux
        Fn   = F0 + F_robin_n
        Fnp1 = F0 + F_robin_np1

        # One theta-scheme step
        # (M + theta*dt*K) U^{n+1} = (M-(1-theta)*dt*K) U^n + dt*(theta*F^{n+1}+(1-theta)*F^n)
        # No Dirichlet => empty dir_dofs
        U = theta_step(
            M, K,
            Fn, Fnp1, U,
            dt=args.dt,
            theta=args.theta,
            dirichlet_dofs=dir_dofs,
            dir_vals_np1=dir_vals
        )

        # ------ Plot every 5 steps to avoid slowing down ------
        if step % 5 == 0:
            ax.clear()
            contour = plot_fe_solution_2d(
                elemNodeTags=elemNodeTags,
                nodeTags=nodeTags,
                nodeCoords=nodeCoords,
                U=U,
                tag_to_dof=tag_to_dof,
                show_mesh=False,
                ax=ax
            )
            ax.set_title(
                f"Krogh cylinder  —  t = {t_np1:.1f} s  (step {step+1}/{args.nsteps})\n"
                f"D={D:.1e} m²/s, P={P:.1e} m/s, theta={args.theta}"
            )
            ax.set_xlabel("x  [m]")
            ax.set_ylabel("y  [m]")
            ax.axis('equal')

            # Add a colourbar on the first step
            if step == 0:
                plt.colorbar(contour, ax=ax, label="Drug concentration  [normalised]")

            plt.pause(0.01)

    print("Done.")
    plt.ioff()
    plt.show()

    gmsh_finalize()


if __name__ == "__main__":
    main()
