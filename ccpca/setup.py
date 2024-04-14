import sysconfig
from setuptools import setup

extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")

cpca_cpp_so = f"cpca_cpp{extension_suffix}"
ccpca_cpp_so = f"ccpca_cpp{extension_suffix}"

setup(
    name="ccpca",
    version=0.16,
    packages=[""],
    package_dir={"": "."},
    package_data={"": [cpca_cpp_so, ccpca_cpp_so]},
    install_requires=["numpy"],
    py_modules=["cpca_cpp", "ccpca_cpp", "cpca", "ccpca"],
)
