import numpy as np
from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

ext_modules = [
    Extension(
        "cython_l_norm",
        ["cython_l_norm.pyx"],
        include_dirs=[np.get_include()],  # Add NumPy's header directory
        extra_compile_args=['-fopenmp'],  # Enable OpenMP support
        extra_link_args=['-fopenmp'],     # Link with OpenMP
    ),
]

setup(
    name='cython_l_norm',
    ext_modules=cythonize(ext_modules),
)

