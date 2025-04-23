from setuptools import setup

setup(
    name="ccpca",
    version=0.20,
    packages=[""],
    package_dir={"": "."},
    install_requires=["numpy", "scipy"],
    py_modules=["cpca", "ccpca"],
)
