# ===============================================
# setup.py
# Description: Setup file for Cython compilation
# ===============================================

import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='levenshtein',
    ext_modules=cythonize('watermark/exp_edit/cython_files/levenshtein.pyx'),
    include_dirs = [np.get_include()]
)