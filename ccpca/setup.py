import sysconfig
import sys
from distutils.core import setup
from shutil import which


def cpp_compiler_check(command_name='c++'):
    err_msg = '''C++ compiler is not found and cannot install ccpca.
Install C++ compiler. For example, for macOS: run "xcode-select --install" in terminal'''

    if which(command_name) is None:
        sys.exit(err_msg)


extension_suffix = sysconfig.get_config_var('EXT_SUFFIX')

# remove existing shared objects, etc.
for removing_extension in ['so', 'exp', 'lib', 'obj', 'pyd', 'dll']:
    for removing_file in glob.glob(f'*.{removing_extension}'):
        try:
            os.remove(removing_file)
        except OSError:
            print("Error while deleting existing compiled files")

if sys.platform.startswith('darwin'):
    cpp_compiler_check()

    if os.system('which brew') > 0:
        print('installing homebrew')
        os.system(
            '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"'
        )
    print('installing python3, eigen, pybind11')
    os.system('brew install pkg-config python3 eigen')
    print('installing pybind11')
    os.system('pip3 install pybind11')
    ## This part can be used to build with CMake (but for anaconda env, this doesn't work well)
    # print('processing cmake')
    # os.system('rm -f CMakeCache.txt')
    # os.system('cmake .')
    # print('processing make')
    # os.system('make')
    print('building cPCA')
    os.system(
        f'c++ -O3 -Wall -mtune=native -march=native -shared -std=c++11 -undefined dynamic_lookup -I/usr/local/include/eigen3/ $(python3 -m pybind11 --includes) cpca.cpp cpca_wrap.cpp -o cpca_cpp{extension_suffix}'
    )
    print('building ccPCA')
    os.system(
        f'c++ -O3 -Wall -mtune=native -march=native -shared -std=c++11 -undefined dynamic_lookup -I/usr/local/include/eigen3/ $(python3 -m pybind11 --includes) cpca.cpp cpca_wrap.cpp ccpca.cpp ccpca_wrap.cpp -o ccpca_cpp{extension_suffix}'
    )
elif sys.platform.startswith('linux'):
    cpp_compiler_check()

    print('installing pybind11')
    os.system('pip3 install pybind11')
    print('building cPCA')
    os.system(
        f'c++ -O3 -Wall -mtune=native -march=native -shared -std=c++11 -I/usr/include/eigen3/ -fPIC `python3 -m pybind11 --includes` cpca.cpp cpca_wrap.cpp -o cpca_cpp{extension_suffix}'
    )
    print('building ccPCA')
    os.system(
        f'c++ -O3 -Wall -mtune=native -march=native -shared -std=c++11 -I/usr/include/eigen3/ -fPIC `python3 -m pybind11 --includes` cpca.cpp cpca_wrap.cpp ccpca.cpp ccpca_wrap.cpp -o ccpca_cpp{extension_suffix}'
    )
elif sys.platform.startswith('win'):
    cpp_compiler_check('cl')

    print('installing pybind11 requests')
    os.system('pip3 install pybind11 requests')

    print('downloading eigen')
    import requests
    import zipfile

    eigen_ver = '3.4.0'
    eigen_name = f'eigen-{eigen_ver}'
    eigen_zip = f'{eigen_name}.zip'
    url = f'https://gitlab.com/libeigen/eigen/-/archive/{eigen_ver}/{eigen_zip}'
    req = requests.get(url)
    with open(eigen_zip, 'wb') as of:
        of.write(req.content)

    with zipfile.ZipFile(eigen_zip, 'r') as zip_ref:
        zip_ref.extractall()

    print('preparing env info')
    import subprocess
    pybind_includes = subprocess.check_output('python -m pybind11 --includes')
    pybind_includes = pybind_includes.decode()
    pybind_includes = pybind_includes[:-2]  # exclude /r/n
    # add double quotes to handle spaces in file paths
    pybind_includes = ' '.join([
        f'/I"{pybind_include}"'
        for pybind_include in pybind_includes.split('-I')[1:]
    ]).replace(' "', '"')

    pyver = f"python{sys.version_info.major}{sys.version_info.minor}"
    pythonlib_path = os.path.dirname(sys.executable) + f'\\libs\\{pyver}.lib'

    # requires VS C++ compiler (https://aka.ms/vs/17/release/vs_BuildTools.exe)
    # also, use an appropriate command prompt for VS (e.g., x64 instead of x86 if using 62-bit Python3)
    print('building cpca')
    os.system(f'cl /c /O2 /std:c11 /EHsc /I./{eigen_name}/ cpca.cpp')
    os.system(
        f'cl /c /O2 /std:c11 /EHsc /I./{eigen_name}/ {pybind_includes} cpca_wrap.cpp'
    )
    os.system(
        f'link cpca.obj cpca_wrap.obj "{pythonlib_path}" /DLL /OUT:cpca_cpp{extension_suffix}'
    )

    print('building ccpca')
    os.system(f'cl /c /O2 /std:c11 /EHsc /I./{eigen_name}/ ccpca.cpp')
    os.system(
        f'cl /c /O2 /std:c11 /EHsc /I./{eigen_name}/ {pybind_includes} ccpca_wrap.cpp'
    )
    os.system(
        f'link cpca.obj cpca_wrap.obj ccpca.obj ccpca_wrap.obj "{pythonlib_path}" /DLL /OUT:ccpca_cpp{extension_suffix}'
    )
    ####
    # cl /c /O2 /std:c11 /EHsc /I./eigen-3.4.0/ cpca.cpp
    # cl /c /O2 /std:c11 /EHsc /I./eigen-3.4.0/ /I"C:\\Users\\Takanori Fujiwara\\AppData\\Local\\Programs\\Python\\Python310\\Include" /I"C:\\Users\\Takanori Fujiwara\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\pybind11\\include" cpca_wrap.cpp
    # link cpca.obj cpca_wrap.obj "C:\\Users\\Takanori Fujiwara\\AppData\\Local\\Programs\\Python\\Python310\\libs\\python310.lib" /DLL /OUT:cpca_cpp.cp310-win_amd64.pyd

    ####
    # cl /c /O2 /std:c11 /EHsc /I./eigen-3.4.0/ ccpca.cpp
    # cl /c /O2 /std:c11 /EHsc /I./eigen-3.4.0/ /I"C:\\Users\\Takanori Fujiwara\\AppData\\Local\\Programs\\Python\\Python310\\Include" /I"C:\\Users\\Takanori Fujiwara\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\pybind11\\include" ccpca_wrap.cpp
    # link cpca.obj cpca_wrap.obj ccpca.obj ccpca_wrap.obj "C:\\Users\\Takanori Fujiwara\\AppData\\Local\\Programs\\Python\\Python310\\libs\\python310.lib" /DLL /OUT:ccpca_cpp.cp310-win_amd64.pyd
else:
    print(
        f'ccPCA only supports macos, linux, windows. Your platform: {sys.platform}'
    )

cpca_cpp_so = f'cpca_cpp{extension_suffix}'
ccpca_cpp_so = f'ccpca_cpp{extension_suffix}'

setup(name='ccpca',
      version=0.16,
      packages=[''],
      package_dir={'': '.'},
      package_data={'': [cpca_cpp_so, ccpca_cpp_so]},
      install_requires=['numpy'],
      py_modules=['cpca_cpp', 'ccpca_cpp', 'cpca', 'ccpca'])
