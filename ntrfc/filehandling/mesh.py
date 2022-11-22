import os

import pyvista as pv


def load_mesh(path_to_mesh):
    assert os.path.isfile(path_to_mesh), path_to_mesh + " is not a valid file"
    extension = os.path.splitext(path_to_mesh)[1]
    if extension == ".vtk" or extension == ".vtp" or extension == ".vtu":
        mesh = pv.read(path_to_mesh)
    elif extension == ".vtm":
        multiBlockMesh = pv.read(path_to_mesh)
        mesh = multiBlockMesh.combine()
    elif extension == ".cgns":
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
            for domain in domains:
                nblocks = domain.n_blocks
                blocks = [domain[b] for b in range(nblocks)]
                for block in blocks:
                    if type(block) == pv.UnstructuredGrid:
                        mesh = mesh.merge(block)
                    if type(block) == pv.StructuredGrid:
                        mesh = mesh.merge(block)
    return mesh
