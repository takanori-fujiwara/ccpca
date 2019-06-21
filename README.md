## ccPCA: PCA for contrasting clusters - C++ Library and Python Module

About
-----
* ccPCA and feature contribution visualization from: ***Supporting Analysis of Dimensionality Reduction Results with Contrastive Learning***    
Takanori Fujiwara, Oh-Hyun Kwon, and Kwan-Liu Ma   
arXiv preprint, 2019 (also currently conditionally accepted to IEEE VAST 2019).

* Features
  * Fast C++ implementation with Eigen3 of Contrastive PCA (cPCA) from [Abid and Zhang et al., 2018].
    * A. Abid, M. J. Zhang, V. K. Bagaria, and J. Zou. Exploring patterns enriched in a dataset with contrastive principal component analysis, Nature Communicationsvolume,Vol. 9, No. 1, pp. 2134, 2018.
    * https://github.com/abidlabs/contrastive

  * An extended version of cPCA (ccPCA) for contrastive clusters.

  * Algorithms for generating an effective, scalable feature contribution visualization, including optimal sign flipping of (c)PCA, matrix reordering, and aggregation.

******

Requirements
-----
* C++11 compiler, Python3, Eigen3, Pybind11, Numpy

* Note: Tested on macOS Mojave. Linux is also supported (testing will be done soon).

******

Setup
-----
#### Mac OS with Homebrew
* Install libraries

    `brew install python3`

    `brew install eigen`

    `brew install pybind11`

    `pip3 install numpy`

* Build source codes in "ccpca" directory with cmake

    `mv /path/to/directory-of-CmakeLists.txt`

    `cmake .`

    `make`

* This generates a shared library, "cpca_cpp.xxxx.so" and "ccpca_cpp.xxxx.so" (e.g., ccpca_cpp.cpython-37m-darwin.so).

* If you want to run samples in this directory. You need to install additional libraries.

    `pip3 install matplotlib`

#### Linux (tested on Ubuntu 18.0.4 LTS)
* Install libraries

    `sudo apt update`

    `sudo apt install libeigen3-dev`

    `sudo apt install python3-pip python3-dev`

    `pip3 install pybind11`

    `pip3 install numpy`

* Build (TODO)

<!-- * Move to 'inc_pca' directory then compile with:

    ``c++ -O3 -Wall -mtune=native -march=native -shared -std=c++11 -I/usr/include/eigen3/ -fPIC `python3 -m pybind11 --includes` inc_pca.cpp inc_pca_wrap.cpp -o inc_pca_cpp`python3-config --extension-suffix` ``

* This generates a shared library, "inc_pca_cpp.xxxx.so" (e.g., inc_pca_cpp.cpython-37m-x86_64-linux-gnu.so).

* If you want to run sample.py in this directory. You need to install additional libraries.

    `sudo apt install python3-tk`

    `pip3 install matplotlib`

    `pip3 install sklearn` -->

******

Usage
-----
* With Python3
    * Place cpca_cpp.xxxx.so, cpca.py, ccpca_cpp.xxxx.so, ccpca.py in the same directory.

    * Import "ccpca" from python. See sample.ipynb for examples.

* With C++
    * Include ccpca.hpp in C++ code with ccpca.cpp.

* Note: Also, it is possible to use cPCA without ccPCA. If you want, import "cpca" from python or include cpca.hpp in C++ code.

******

## How to Cite
Please, cite:    
Takanori Fujiwara, Oh-Hyun Kwon, and Kwan-Liu Ma, "Supporting Analysis of Dimensionality Reduction Results with Contrastive Learning".
arXiv preprint, 2019.
(We prefer citing the version for IEEE VAST 2019 once it is accepted).
