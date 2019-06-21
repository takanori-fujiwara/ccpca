import glob
from distutils.core import setup

cpca_cpp_so = glob.glob('cpca_cpp*.so')[0]
ccpca_cpp_so = glob.glob('ccpca_cpp*.so')[0]

setup(
    name='ccpca',
    version=0.1,
    packages=[''],
    package_dir={'': '.'},
    package_data={'': [cpca_cpp_so, ccpca_cpp_so]},
    py_modules = ['cpca_cpp', 'ccpca_cpp', 'cpca', 'ccpca']
)
