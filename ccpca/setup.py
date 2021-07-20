import os
import sys
import glob
from distutils.core import setup

for cpca_cpp in glob.glob('cpca_cpp*.so'):
    try:
        os.remove(cpca_cpp)
    except OSError:
        print("Error while deleting cpca_cpp shared object files")

for ccpca_cpp in glob.glob('ccpca_cpp*.so'):
    try:
        os.remove(ccpca_cpp)
    except OSError:
        print("Error while deleting ccpca_cpp shared object files")

if sys.platform.startswith('darwin'):
    if os.system('which brew') > 0:
        print('installing homebrew')
        os.system(
            '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"'
        )
    print('installing python3, eigen, pybind11')
    os.system('brew install pkg-config python3 eigen pybind11')
    ## This part can be used to build with CMake (but for anaconda env, this doesn't work well)
    # print('processing cmake')
    # os.system('rm -f CMakeCache.txt')
    # os.system('cmake .')
    # print('processing make')
    # os.system('make')
    print('building cPCA')
    os.system(
        'c++ -O3 -Wall -mtune=native -march=native -shared -std=c++11 -undefined dynamic_lookup -I/usr/local/include/eigen3/ $(python3 -m pybind11 --includes) cpca.cpp cpca_wrap.cpp -o cpca_cpp$(python3-config --extension-suffix)'
    )
    print('building ccPCA')
    os.system(
        'c++ -O3 -Wall -mtune=native -march=native -shared -std=c++11 -undefined dynamic_lookup -I/usr/local/include/eigen3/ $(python3 -m pybind11 --includes) cpca.cpp cpca_wrap.cpp ccpca.cpp ccpca_wrap.cpp -o ccpca_cpp$(python3-config --extension-suffix)'
    )
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
      version=0.13,
      packages=[''],
      package_dir={'': '.'},
      package_data={'': [cpca_cpp_so, ccpca_cpp_so]},
      install_requires=['numpy'],
      py_modules=['cpca_cpp', 'ccpca_cpp', 'cpca', 'ccpca'])
