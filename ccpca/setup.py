import os
import sys
import glob
from distutils.core import setup

if sys.platform.startswith('darwin'):
    if os.system('which brew') > 0:
        print('installing homebrew')
        os.system(
            '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"'
        )
    print('installing python3, eigen, pybind11')
    os.system('brew install python3 eigen pybind11')
    print('processing cmake')
    os.system('rm CMakeCache.txt')
    os.system('cmake .')
    print('processing make')
    os.system('make')
elif sys.platform.startswith('linux'):
    print('installing pybind11')
    os.system('pip3 install pybind11')
    print('building cPCA')
    os.system(
        'c++ -O3 -Wall -mtune=native -march=native -shared -std=c++11 -I/usr/include/eigen3/ -fPIC `python3 -m pybind11 --includes` cpca.cpp cpca_wrap.cpp -o cpca_cpp`python3-config --extension-suffix`'
    )
    print('building ccPCA')
    os.system(
        'c++ -O3 -Wall -mtune=native -march=native -shared -std=c++11 -I/usr/include/eigen3/ -fPIC `python3 -m pybind11 --includes` cpca.cpp cpca_wrap.cpp ccpca.cpp ccpca_wrap.cpp -o ccpca_cpp`python3-config --extension-suffix`'
    )
else:
    print(
        f'ccPCA only supports macos and linux. Your platform: {sys.platform}')

cpca_cpp_so = glob.glob('cpca_cpp*.so')[0]
ccpca_cpp_so = glob.glob('ccpca_cpp*.so')[0]

setup(name='ccpca',
      version=0.12,
      packages=[''],
      package_dir={'': '.'},
      package_data={'': [cpca_cpp_so, ccpca_cpp_so]},
      install_requires=['numpy'],
      py_modules=['cpca_cpp', 'ccpca_cpp', 'cpca', 'ccpca'])

# path = os.path.abspath(__file__)
# clone = "git clone https://github.com/takanori-fujiwara/ccpca.git"

# os.chdir(path)
# os.system(clone)
