import os
import sys
import subprocess
import shutil
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py
from Cython.Build import cythonize
import numpy as np

# Check if running within SageMath
try:
    from sage.env import SAGE_INC, SAGE_LIB
    using_sage = True
except ImportError:
    SAGE_INC = SAGE_LIB = None
    using_sage = False

# Custom command to preprocess .sage files
class PreprocessSageFiles(_build_py):
    def run(self):
        # Use 'sage -python' to run the preprocessing script
        self.preprocess_sage_files()
        super().run()
    def preprocess_sage_files(self):
        # Run the preprocessing script
        subprocess.check_call([sys.executable, 'scripts/preprocess_sage_files.py'])
        print("Preprocessing of .sage files complete.")




# Custom build_ext command to ensure preprocessing happens before compiling extensions
class build_ext(_build_ext):
    def run(self):
        # Preprocess .sage files before building extensions
        self.run_command('build_py')
        super().run()

# Define the Cython extension module
include_dirs = [np.get_include()]
library_dirs = []
libraries = []
extra_compile_args = ['-fopenmp']
extra_link_args = ['-fopenmp']

if using_sage:
    include_dirs.append(SAGE_INC)
    library_dirs.append(SAGE_LIB)
    libraries.append('sage')

ext_modules = cythonize([
    Extension(
        "pyPlumbing.cython_l_norm",
        ["src/pyPlumbing/cython_l_norm.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
])

setup(
    name='pyPlumbing',
    version='0.2.0',
    description='A plumbing utilities package for SageMath',
    author='Davide Passaro',
    author_email='dpassaro@caltech.edu',
    url='https://github.com/d-passaro/pyPlumbing',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    cmdclass={
        'build_py': PreprocessSageFiles,
        'build_ext': build_ext,
    },
    ext_modules=ext_modules,
    install_requires=[
        'numpy<2',
        'Cython',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Sage',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
