from distutils.core import setup

setup(name='fc-view',
      version=0.1,
      packages=[''],
      package_dir={'': '.'},
      install_requires=['numpy', 'scipy'],
      py_modules=['opt_sign_flip', 'mat_reorder'])
