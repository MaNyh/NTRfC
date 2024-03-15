import os

import pyvista as pv


def read_vtk(path_to_mesh):
    """Read a vtk/vtp/vtu file and return the corresponding mesh object.

    Parameters
    ----------
    path_to_mesh : str
        The file path to the vtk/vtp/vtu file to be read.

    Returns
    -------
    mesh : pyvista.UnstructuredGrid or pyvista.StructuredGrid
        The mesh object constructed from the input file.
    """
    return pv.read(path_to_mesh)


def read_vtm(path_to_mesh):
    """Read a vtm file and return the corresponding mesh object.

    Parameters
    ----------
    path_to_mesh : str
        The file path to the vtm file to be read.

    Returns
    -------
    mesh : pyvista.UnstructuredGrid or pyvista.StructuredGrid
        The combined mesh object constructed from all the blocks in the vtm file.
    """
    multi_block_mesh = pv.read(path_to_mesh)
    return multi_block_mesh.combine()


def read_cgns(path_to_mesh):
    """Read a cgns file and return the corresponding mesh object.

    Parameters
    ----------
    path_to_mesh : str
        The file path to the cgns file to be read.

    Returns
    -------
    mesh : pyvista.UnstructuredGrid or pyvista.StructuredGrid
        The combined mesh object constructed from all the unstructured or structured grids in the cgns file.
    """
    cgns_reader = pv.get_reader(path_to_mesh)
    cgns = cgns_reader.read()
    nbase = cgns.n_blocks
    basenames = [cgns.get_block_name(i) for i in range(nbase)]
    bases = [cgns[b] for b in basenames]
    mesh = pv.UnstructuredGrid()
    for base in bases:
        ndomains = base.n_blocks
        domainnames = [base.get_block_name(i) for i in range(ndomains)]
        domains = [base[i] for i in domainnames]
        if all(isinstance(grid, pv.UnstructuredGrid) for grid in domains):
            for domain in domains:
                if isinstance(domain, pv.UnstructuredGrid):
                    mesh = mesh.merge(domain)
                if isinstance(domain, pv.StructuredGrid):
                    mesh = mesh.merge(domain)
        else:
            # not all elements are UnstructuredGrid objects
            for domain in domains:
                nblocks = domain.n_blocks
                blocks = [domain[b] for b in range(nblocks)]
                for block in blocks:
                    if isinstance(block, pv.UnstructuredGrid):
                        mesh = mesh.merge(block)
                    if isinstance(block, pv.StructuredGrid):
                        mesh = mesh.merge(block)
    return mesh


def load_mesh(path_to_mesh):
    """Load a mesh file and return the corresponding mesh object.

    Parameters
    ----------
    path_to_mesh : str
        The file path to the mesh file to be loaded.
        The file extension should be one of:

        ".vtk", ".vtp", ".vtu", ".vtm", ".cgns", ".msh".

    Returns
    -------
    mesh : pyvista.UnstructuredGrid or pyvista.StructuredGrid
        The mesh object constructed from the input file.
    """
    assert os.path.isfile(path_to_mesh), f"{path_to_mesh} is not a valid file"
    extension = os.path.splitext(path_to_mesh)[1]
    if extension == ".vtk" or extension == ".vtp" or extension == ".vtu":
        mesh = read_vtk(path_to_mesh)
    elif extension == ".vtm":
        mesh = read_vtm(path_to_mesh)
    elif extension == ".cgns":
        mesh = read_cgns(path_to_mesh)
    elif extension == ".msh":
        mesh = pv.read(path_to_mesh)
    return mesh
