# plot_utils.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import gmsh


def setup_interactive_figure(xlim=None, ylim=None):
    plt.ion()
    fig, ax = plt.subplots()
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig, ax


def plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags,
                 bnds, bnds_tags, tag_to_index=None, ax=None):
    """
    Affiche le maillage 2D et les bords physiques.

    Dans notre projet :
    - VesselWall = bord intérieur, condition de Robin
    - TissueBoundary = bord extérieur, Neumann homogène
    """
    coords = nodeCoords.reshape(-1, 3)
    x = coords[:, 0]
    y = coords[:, 1]

    if tag_to_index is None:# Gmsh numérote les nœuds avec des tags arbitraires
        # Ce tableau de correspondance convertit un tag Gmsh en indice dans le tableau de coordonnées.
        max_node_tag = int(np.max(nodeTags))
        tag_to_index = np.zeros(max_node_tag + 1, dtype=int)
        for i, tag in enumerate(nodeTags):
            tag_to_index[int(tag)] = i

    num_elements = len(elemTags)
    nodes_per_elem = len(elemNodeTags) // num_elements

    # take only the first 3 nodes (=geometric nodes that form the triangles)
    # Pour des éléments d'ordre 2 (6 nœuds par triangle), seuls les 3 premiers nœuds sont
    # les sommets géométriques. On les extrait pour construire la triangulation.    
    all_nodes = elemNodeTags.reshape(num_elements, nodes_per_elem)
    corner_nodes = all_nodes[:, :3]

    # Map to indices
    tri_indices = tag_to_index[corner_nodes.astype(int)]
    mesh_triang = tri.Triangulation(x, y, tri_indices)#Construit la triangulation
    #matplotlib à partir des coordonnées et de la connectivité

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the skeleton
    ax.triplot(mesh_triang, color="black", lw=0.5, alpha=0.4)

    colors = ["red", "darkblue", "orange", "mediumpurple", "pink"]

    for i, (name, dim) in enumerate(bnds):
        tags = bnds_tags[i]
        indices = tag_to_index[tags.astype(int)]
        #Pour chaque frontière physique, les nœuds correspondants sont affichés en couleur 
        # distincte afin de visualiser où s'appliquent les conditions limites
        ax.scatter(
            x[indices], y[indices],
            label=name,
            s=15,
            zorder=3,
            marker="o",
            facecolor="None",
            edgecolor=colors[i % len(colors)]
        )

    ax.set_aspect("equal")
    ax.legend(frameon=True, framealpha=1)
    ax.axis("off")

    return ax


def plot_fe_solution_2d_krogh(elemTags, elemNodeTags, nodeCoords, nodeTags, U, tag_to_dof,
                        show_mesh=False, ax=None, cmap="hot", vmin=None, vmax=None, norm=None):
    """
    Affiche la solution éléments finis 2D.

    U contient les valeurs nodales de la concentration.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    num_dofs = len(U)
    coords_mapped = np.zeros((num_dofs, 2))

    if nodeCoords.size != len(nodeTags) * 3:
        _, coords_to_use, _ = gmsh.model.mesh.getNodes()
        all_coords = coords_to_use.reshape(-1, 3)
    else:
        all_coords = nodeCoords.reshape(-1, 3)

    for i, tag in enumerate(nodeTags):
        dof_idx = tag_to_dof[int(tag)]# Il convertit un tag Gmsh en indice de degré de liberté (DDL) dans le vecteur solution U
         #C'est la correspondance entre la numérotation Gmsh et la numérotation interne du solveur éléments finis.
        if dof_idx != -1:
            coords_mapped[dof_idx] = all_coords[i, :2]

    x = coords_mapped[:, 0]
    y = coords_mapped[:, 1]

    num_elements = len(elemTags)
    nodes_per_elem = len(elemNodeTags) // num_elements

    conn_reshaped = elemNodeTags.reshape(-1, nodes_per_elem)
    triangles = tag_to_dof[conn_reshaped[:, :3].astype(int)]

    U = np.array(U).flatten()

    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin, vmax, 100)
        contour = ax.tricontourf(
            x, y, triangles, U,
            levels=levels,
            cmap=cmap,
            extend="both",
        )
    else:
        contour = ax.tricontourf(
            x, y, triangles, U,
            levels=100,
            cmap=cmap,
            norm=norm
        )

    if show_mesh:
        ax.triplot(x, y, triangles, color="white", linewidth=0.2, alpha=0.3)

    ax.set_aspect("equal")
    ax.axis("off")

    return contour

def plot_fe_solution_2d_multicouche(elemTags, elemNodeTags, nodeCoords, nodeTags, U, tag_to_dof,
                        show_mesh=False, ax=None, cmap="hot", vmin=None, vmax=None, norm=None):
    """
    Affiche la solution éléments finis 2D.
 
    U contient les valeurs nodales de la concentration.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
 
    num_dofs = len(U)
    coords_mapped = np.zeros((num_dofs, 2))#Ce tableau stocke les coordonnées (x,y) de chaque degré de liberté (DDL) correspondant à la solution U.
    if nodeCoords.size != len(nodeTags) * 3:
        _, coords_to_use, _ = gmsh.model.mesh.getNodes()
        all_coords = coords_to_use.reshape(-1, 3)
    else:
        all_coords = nodeCoords.reshape(-1, 3)
 
    for i, tag in enumerate(nodeTags):
        dof_idx = tag_to_dof[int(tag)]# Il convertit un tag Gmsh en indice de degré de liberté (DDL) dans le vecteur solution U
         #C'est la correspondance entre la numérotation Gmsh et la numérotation interne du solveur éléments finis.
        if dof_idx != -1:
            coords_mapped[dof_idx] = all_coords[i, :2]
 
    x = coords_mapped[:, 0]
    y = coords_mapped[:, 1]
 
    num_elements = len(elemTags)
    nodes_per_elem = len(elemNodeTags) // num_elements
 
    conn_reshaped = elemNodeTags.reshape(-1, nodes_per_elem)
    triangles = tag_to_dof[conn_reshaped[:, :3].astype(int)]
    #La connectivité (qui dit quels nœuds forment chaque triangle) 
    # est re-exprimée en indices de DDL plutôt qu'en tags Gmsh
 
    U = np.array(U).flatten()
 
    norm = None
    contour = ax.tricontourf(x, y, triangles, U, levels=100, cmap=cmap, norm=norm)
    #Trace le champ scalaire U en interpolant les valeurs nodales sur toute la surface triangulée,
    #avec 100 niveaux de couleur
    if show_mesh:
        ax.triplot(x, y, triangles, color="white", linewidth=0.2, alpha=0.3)
 
    ax.set_aspect("equal")
    ax.axis("off")
 
    return contour