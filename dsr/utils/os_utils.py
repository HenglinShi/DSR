import os

import shutil
import os.path as osp
from loguru import logger
from shutil import copytree, ignore_patterns

def copy_code(output_folder, curr_folder, code_folder='code'):
    code_folder = osp.join(output_folder, code_folder)
    if not osp.exists(code_folder):
        os.makedirs(code_folder)

    # Copy code
    logger.info('Copying main files ...')

    for f in [x for x in os.listdir(curr_folder) if x.endswith('.py')]:
        mainpy_path = osp.join(curr_folder, f)
        dest_mainpy_path = osp.join(code_folder, f)
        shutil.copy2(mainpy_path, dest_mainpy_path)

    logger.info('Copying the rest of the source code ...')
    for f in ['dsr', 'configs']:
        src_folder = osp.join(curr_folder, f)
        dest_folder = osp.join(code_folder, osp.split(src_folder)[1])
        if os.path.exists(dest_folder):
            shutil.rmtree(dest_folder)
        shutil.copytree(src_folder, dest_folder, ignore=ignore_patterns('*.pyc', 'tmp*', '__pycache__'))
