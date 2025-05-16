from setuptools import setup

setup(
    name="ccpca",
    version="0.2.3",
    packages=["cpca", "ccpca", "fc_view"],
    package_dir={"cpca": "cpca", "ccpca": "ccpca", "fc_view": "fc_view"},
    install_requires=["numpy", "scipy", "multiprocess"],
    py_modules=["cpca", "ccpca", "fc_view"],
)
