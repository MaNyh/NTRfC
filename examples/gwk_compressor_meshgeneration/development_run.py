#import os
#os.system('snakemake -c1 03_meshgeometry/tfd_gwk_compressor_blocklines.pdf')
from ntrfc.preprocessing.mesh_creation.domain_creation import create_2d_blocklines

create_2d_blocklines(r"D:\CodingProjects\NTRfC\examples\gwk_compressor",["tfd_gwk_compressor"])
