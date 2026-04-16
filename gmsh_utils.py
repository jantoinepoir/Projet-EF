# gmsh_utils.py

#géométrie / maillage /quadrature

import numpy as np
import gmsh


def gmsh_init(model_name="fem1d"):
    gmsh.initialize()
    gmsh.model.add(model_name)


def gmsh_finalize():
    gmsh.finalize()


def build_1d_mesh(L=1.0, cl1=0.02, cl2=0.10, order=1):
    """
    Build and mesh a 1D segment [0,L] with different characteristic lengths.
    Returns (line_tag, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags).
    """
    p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, cl1)
    p1 = gmsh.model.geo.addPoint(L, 0.0, 0.0, cl2)
    line = gmsh.model.geo.addLine(p0, p1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.setOrder(order)

    elemType = gmsh.model.mesh.getElementType("line", order)

    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)

    return line, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags


#projet
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


def end_dofs_from_nodes(nodeCoords):
    """
    Robustly identify first/last node dofs from coordinates (x-min, x-max).
    nodeCoords is flattened [x0,y0,z0, x1,y1,z1, ...]
    Returns (left_dof, right_dof) as 0-based indices.
    """
    X = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)[:, 0]
    left = int(np.argmin(X))
    right = int(np.argmax(X))
    return left, right

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
    

def open_2d_mesh(msh_filename, order=1):
    """
    Load a .msh file.

    Parameters
    ----------
    msh_filename : str
        Path to the .msh file
    order : int
        Polynomial order of elements

    Returns
    -------
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags
    """

    import gmsh

    # --- load geometry
    gmsh.open(msh_filename)

    # --- high order
    gmsh.model.mesh.setOrder(order)

    # --- element type (triangles)
    elemType = gmsh.model.mesh.getElementType("triangle", order)

    # --- nodes
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()

    # --- elements
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)

    surf = gmsh.model.getEntities(2)[0][1]

    curve_tags = gmsh.model.getBoundary([(2, surf)], oriented=False)
    
    gmsh.model.addPhysicalGroup(1, [curve_tags[0][1]], tag=1)
    gmsh.model.setPhysicalName(1, 1, "OuterBoundary")

    gmsh.model.addPhysicalGroup(1, [curve_tags[1][1]], tag=2)
    gmsh.model.setPhysicalName(1, 2, "InnerBoundary")

    bnds = [('OuterBoundary', 1),('InnerBoundary', 1)]

    bnds_tags = []
    for name, dim in bnds:
        tag = -1
        for t in gmsh.model.getPhysicalGroups(dim):
            if gmsh.model.getPhysicalName(dim, t[1]) == name:
                tag = t[1]
                break
        if tag == -1:
            raise ValueError(f"Physical group '{name}' not found in mesh.")
        bnds_tags.append(gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0])

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags
