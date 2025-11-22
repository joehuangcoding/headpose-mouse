from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "cpu_nms",
        sources=["cpu_nms.pyx", "cpu_nms.c"],  # Cython and C files
        include_dirs=[np.get_include()],       # Include NumPy headers
    )
]

setup(
    name="cpu_nms",
    ext_modules=cythonize(extensions)
)