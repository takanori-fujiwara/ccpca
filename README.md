## ccPCA: contrasting clusters in PCA - C++ Library and Python Module

About
-----
* ccPCA and feature contribution visualization from: ***Supporting Analysis of Dimensionality Reduction Results with Contrastive Learning***.
Takanori Fujiwara, Oh-Hyun Kwon, and Kwan-Liu Ma.
IEEE Transactions on Visualization and Computer Graphics and IEEE VIS 2019 (VAST).
DOI: [10.1109/TVCG.2019.2934251](https://doi.org/10.1109/TVCG.2019.2934251)

* Demonstration of a system using ccPCA: http://kwonoh.net/ccpca/

* Features
  * Fast C++ implementation with Eigen3 of Contrastive PCA (cPCA) from [Abid and Zhang et al., 2018].
    * A. Abid, M. J. Zhang, V. K. Bagaria, and J. Zou. Exploring patterns enriched in a dataset with contrastive principal component analysis, Nature Communicationsvolume,Vol. 9, No. 1, pp. 2134, 2018.
    * https://github.com/abidlabs/contrastive

  * An extended version of cPCA (ccPCA) for contrasting clusters.

  * Algorithms for generating an effective, scalable feature contribution visualization, including optimal sign flipping of (c)PCA, matrix reordering, and aggregation.

* Use case examples
  * Analysis of dimensionality reduction results (compare some points with others)
  * Analysis of clustering results (compare some cluster with others)
  * Analysis of labeled data (compare some label with others)

******

Requirements
-----
* C++11 compiler, Python3, Eigen3, Pybind11, Numpy

* Note: Tested on macOS Mojave. Linux is also supported (tested on Ubuntu 19.0.4 LTS).

******

Setup
-----
#### Mac OS with Homebrew

##### 1) Installation of ccPCA

* Install required libraries

    `brew install python3`

    `brew install eigen`

    `brew install pybind11`

    `pip3 install numpy`

* Move to "ccpca/ccpca" directory

* Build source codes with cmake

    `cmake .`

    `make`

* This generates a shared library, "cpca_cpp.xxxx.so" and "ccpca_cpp.xxxx.so" (e.g., ccpca_cpp.cpython-37m-darwin.so).

* Install the modules with pip3.

    `pip3 install .`

##### 2) Installation of fc-view

* If you want to use the algorithms for scalable visualization of features' contributions, please follow the next steps.

* Move to "ccpca/fc_view" directory

* Install the modules with pip3

    `pip3 install . `

#### Linux (will be tested on Ubuntu 18.0.4 LTS. Not tested yet)

##### 1) Installation of ccPCA

* Install libraries

    `sudo apt update`

    `sudo apt install libeigen3-dev`

    `sudo apt install python3-pip python3-dev`

    `pip3 install pybind11`

    `pip3 install numpy`

* Move to "ccpca/ccpca" directory

* Build source codes with (need to run both two commands below):

    Building cPCA:

    ``c++ -O3 -Wall -mtune=native -march=native -shared -std=c++11 -I/usr/include/eigen3/ -fPIC `python3 -m pybind11 --includes` cpca.cpp cpca_wrap.cpp -o cpca_cpp`python3-config --extension-suffix` ``

    Building ccPCA:

    ``c++ -O3 -Wall -mtune=native -march=native -shared -std=c++11 -I/usr/include/eigen3/ -fPIC `python3 -m pybind11 --includes` cpca.cpp cpca_wrap.cpp ccpca.cpp ccpca_wrap.cpp -o ccpca_cpp`python3-config --extension-suffix` ``

* This generates a shared library, "cpca_cpp.xxxx.so" and "ccpca_cpp.xxxx.so" (e.g., ccpca_cpp.cpython-37m-x86_64-linux-gnu.so).

* Install the modules with pip3.

    `pip3 install .`

* You can test with sample.py

    `pip3 install matplotlib`

    `pip3 install sklearn`

    `python3 sample.py`

##### 2) Installation of fc-view

* If you want to use the algorithms for scalable visualization of features' contributions, please follow the next steps.

* Move to "ccpca/fc_view" directory

* Install the modules with pip3

    `pip3 install . `

******

Usage
-----
* With Python3
    * Import installed modules from python (e.g., `from ccpca import CCPCA`). See sample.ipynb for examples.

* With C++
    * Include header files (e.g., ccpca.hpp) in C++ code with cpp files (e.g., ccpca.cpp).

******

## How to Cite
Please, cite:    
Takanori Fujiwara, Oh-Hyun Kwon, and Kwan-Liu Ma, "Supporting Analysis of Dimensionality Reduction Results with Contrastive Learning".
IEEE Transactions on Visualization and Computer Graphics, 2019.
DOI: [10.1109/TVCG.2019.2934251](https://doi.org/10.1109/TVCG.2019.2934251)
