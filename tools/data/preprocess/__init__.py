'''
Adopted from SPIN: https://github.com/nkolot/SPIN
Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop
Nikos Kolotouros*, Georgios Pavlakos*, Michael J. Black, Kostas Daniilidis
ICCV 2019
'''


from .h36m import h36m_extract
from .pw3d import pw3d_extract
from .mpi_inf_3dhp import mpi_inf_3dhp_extract
from .lsp_dataset import lsp_dataset_extract
from .lsp_dataset_original import lsp_dataset_original_extract
from .hr_lspet import hr_lspet_extract
from .mpii import mpii_extract
from .coco import coco_extract
