# gmsh_utils.py
import numpy as np
import gmsh


def gmsh_init(model_name="fem1d"):
    gmsh.initialize()
    gmsh.model.add(model_name)

def gmsh_finalize():
    gmsh.finalize()

def prepare_quadrature_and_basis(elemType, order):
    """
    Returns:
      xi (flattened uvw), w (ngp), N (flattened bf), gN (flattened gbf)
    """
    rule = f"Gauss{2 * order}"
    xi, w = gmsh.model.mesh.getIntegrationPoints(elemType, rule)
    _, N, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "Lagrange")
    _, gN, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "GradLagrange")
    return xi, np.asarray(w, dtype=float), N, gN


def get_jacobians(elemType, xi, tag=-1):
    """
    Wrapper around gmsh.getJacobians.
    Returns (jacobians, dets, coords)
    """
    jacobians, dets, coords = gmsh.model.mesh.getJacobians(elemType, xi, tag=tag)
    return jacobians, dets, coords

def border_dofs_from_tags(l_tags, tag_to_dof):
    """
    Converts a list of GMSH node tags into the corresponding 
    compact matrix indices (DoFs).
    """
    # Ensure tags are integers
    l_tags = np.asarray(l_tags, dtype=int)
    
    # Filter out any tags that might not be in our DoF mapping (like geometry points)
    # then map them to our 0...N-1 indices
    valid_mask = (tag_to_dof[l_tags] != -1)
    l_dofs = tag_to_dof[l_tags[valid_mask]]
    return l_dofs

def getPhysical(name):
    """
    Get the physical group elements and nodes for a given name and dimension.
    """
    
    dimTags = gmsh.model.getEntitiesForPhysicalName(name)
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=dimTags[0][0], tag=dimTags[0][1])
    elemType = elemTypes[0]  # Assuming one element type per physical group
    elemTags = elemTags[0]
    elemNodeTags = elemNodeTags[0]
    entityTag = dimTags[0][1]
    return elemType, elemTags, elemNodeTags, entityTag

def build_annular_mesh(Rv=0.01, Rt=0.08, cl_v=0.002, cl_t=0.008, order=1):
    """
    Build and mesh a 2D annular domain (Krogh cylinder cross-section):
        Omega = {(x,y) : Rv^2 <= x^2+y^2 <= Rt^2}
 
    The inner boundary Gamma_v (vessel wall) and outer boundary Gamma_t (tissue
    limit) are identified automatically by comparing bounding box radii.
 
    Parameters
    ----------
    Rv    : float  Inner (vessel) radius
    Rt    : float  Outer (tissue) radius
    cl_v  : float  Characteristic mesh length at the vessel wall (fine)
    cl_t  : float  Characteristic mesh length at the tissue boundary (coarser)
    order : int    Polynomial order of elements (1 = P1 triangles)
 
    Returns
    -------
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags
      - bnds      : list of (name, dim) pairs
      - bnds_tags : list of node-tag arrays, one per boundary
    """
    assert Rv < Rt, "Rv must be strictly smaller than Rt"
 
    # --- Use OCC kernel for boolean operations
    disk_outer = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, Rt, Rt)
    disk_inner = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, Rv, Rv)
 
    # Annulus = outer disk minus inner disk
    out_dimtags, _ = gmsh.model.occ.cut(
        [(2, disk_outer)],   # tool
        [(2, disk_inner)],   # cutters
        removeObject=True,
        removeTool=True
    )
    gmsh.model.occ.synchronize()
 
    surf_tag = out_dimtags[0][1]
 
    # --- Mesh size fields: fine near vessel wall, coarser at tissue edge
    # We use a Distance field from the inner circle boundary to smoothly vary cl.
    curves = gmsh.model.getBoundary([(2, surf_tag)], oriented=False)
 
    # Identify inner vs outer curve by bounding-box radius
    inner_tag = None
    outer_tag = None
    for dim_tag in curves:
        ctag = abs(dim_tag[1])
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, ctag)
        r_max = max(abs(xmax), abs(ymax), abs(xmin), abs(ymin))
        if r_max < 0.5 * (Rv + Rt):      # small radius  => vessel wall
            inner_tag = ctag
        else:                              # large radius  => tissue boundary
            outer_tag = ctag
 
    if inner_tag is None or outer_tag is None:
        raise RuntimeError(
            "Could not identify inner/outer boundaries. "
            f"Curves found: {curves}"
        )
 
    # --- Mesh size using gmsh Fields
    # Field 1: Distance from inner circle
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", [inner_tag])
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)
 
    # Field 2: Threshold – cl_v near the wall, cl_t far away
    f_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh, "InField",   f_dist)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin",   cl_v)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax",   cl_t)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin",   0.2 * (Rt - Rv))
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax",   0.8 * (Rt - Rv))
 
    gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
 
    # --- Physical groups
    gmsh.model.addPhysicalGroup(1, [inner_tag], tag=1)
    gmsh.model.setPhysicalName(1, 1, "VesselWall")       # Gamma_v  (Robin)
 
    gmsh.model.addPhysicalGroup(1, [outer_tag], tag=2)
    gmsh.model.setPhysicalName(1, 2, "TissueBoundary")   # Gamma_t  (Neumann homog.)
 
    gmsh.model.addPhysicalGroup(2, [surf_tag], tag=1)
    gmsh.model.setPhysicalName(2, 1, "Tissue")
 
    # --- Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
 
    # --- Collect data
    elemType   = gmsh.model.mesh.getElementType("triangle", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags,  elemNodeTags  = gmsh.model.mesh.getElementsByType(elemType)
 
    bnds = [("VesselWall", 1), ("TissueBoundary", 1)]
    bnds_tags = []
    for name, dim in bnds:
        pg_tag = -1
        for t in gmsh.model.getPhysicalGroups(dim):
            if gmsh.model.getPhysicalName(dim, t[1]) == name:
                pg_tag = t[1]
                break
        if pg_tag == -1:
            raise ValueError(f"Physical group '{name}' not found.")
        bnds_tags.append(
            gmsh.model.mesh.getNodesForPhysicalGroup(dim, pg_tag)[0]
        )
 
    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags

def build_multilayer_rect_mesh(L=0.1, H=0.05, layer_ratios=[0.13, 0.53, 0.34], cl=0.002, order=1):
    """
    Crée un maillage rectangulaire 2D composé de 3 couches de tissus.
    """
    # Si on sait que L=150 μm
    # Layer 1 (VesselWall/Proche) : ~0 à 20 μm (13% de L) 
    # Layer 2 (Tissu viable)      : ~20 à 100 μm (53% de L)
    # Layer 3 (Hypoxique/Nécrosé) : ~100 à 150 μm (34% de L)

    # 1. Calcul des positions x des interfaces
    x_bounds = np.cumsum([0] + layer_ratios) * L
    
    surf_tags = []
    # 2. Création des 3 rectangles initiaux
    for i in range(3):
        width = x_bounds[i+1] - x_bounds[i]
        tag = gmsh.model.occ.addRectangle(x_bounds[i], 0, 0, width, H)
        surf_tags.append(tag)

    # 3. La "soudure" (Fragment)
    # On fragmente pour que les interfaces soient partagées.
    # 'out' contiendra la liste des nouvelles surfaces créées.
    out, out_map = gmsh.model.occ.fragment([(2, s) for s in surf_tags], [])
    
    gmsh.model.occ.synchronize()

    # 4. Identification des Groupes Physiques (Surfaces)
    # 'out' est une liste de (dim, tag). On récupère uniquement les surfaces (dim=2).
    new_surfaces = [entity[1] for entity in out if entity[0] == 2]
    
    # On les trie par leur position en X pour être sûr que Layer1 est à gauche
    new_surfaces.sort(key=lambda s: gmsh.model.occ.getCenterOfMass(2, s)[0])

    for i, s_tag in enumerate(new_surfaces):
        pg_tag = i + 1
        gmsh.model.addPhysicalGroup(2, [s_tag], tag=pg_tag)
        gmsh.model.setPhysicalName(2, pg_tag, f"Layer{i+1}")

    # 5. Les Bords (Frontières)
    curves = gmsh.model.getEntities(1)
    vessel_curves = []
    outer_curves = []

    for dim, c_tag in curves:
        mass = gmsh.model.occ.getCenterOfMass(dim, c_tag)
        # On utilise une petite tolérance (1e-6) au lieu de np.isclose pour plus de sûreté
        if abs(mass[0] - 0) < 1e-6: # Bord gauche
            vessel_curves.append(c_tag)
        elif abs(mass[0] - L) < 1e-6: # Bord droit
            outer_curves.append(c_tag)

    gmsh.model.addPhysicalGroup(1, vessel_curves, tag=10)
    gmsh.model.setPhysicalName(1, 10, "VesselWall")

    gmsh.model.addPhysicalGroup(1, outer_curves, tag=11)
    gmsh.model.setPhysicalName(1, 11, "OuterBoundary")

    # 6. Maillage
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), cl)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    # --- Extraction des données pour le solveur ---
    elemType = gmsh.model.mesh.getElementType("triangle", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)

    bnds = [("VesselWall", 1), ("OuterBoundary", 1)]
    bnds_tags = []
    for name, dim in bnds:
        tag_pg = -1
        for t in gmsh.model.getPhysicalGroups(dim):
            if gmsh.model.getPhysicalName(dim, t[1]) == name:
                tag_pg = t[1]
                break
        if tag_pg != -1:
            bnds_tags.append(gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag_pg)[0])

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags
