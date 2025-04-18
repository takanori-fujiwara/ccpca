## ccPCA: contrasting clusters in PCA - C++ Library and Python Module

New
-----
* Supported installation using virtual environments (due to the use of venv became a default from Python3.12) (2024-04-14).
* Now, all Mac OS, Linux, Windows are supported (2022-04-27).

About
-----
* ccPCA and feature contribution visualization from: ***Supporting Analysis of Dimensionality Reduction Results with Contrastive Learning***.
Takanori Fujiwara, Oh-Hyun Kwon, and Kwan-Liu Ma.
IEEE Transactions on Visualization and Computer Graphics and IEEE VIS 2020 (VAST 2019).
DOI: [10.1109/TVCG.2019.2934251](https://doi.org/10.1109/TVCG.2019.2934251)

* Demonstration of a system using ccPCA: http://kwonoh.net/ccpca/

* Features
  * Fast C++ implementation with Eigen3 of Contrastive PCA (cPCA) from [Abid and Zhang et al., 2018].<br />
    * A. Abid, M. J. Zhang, V. K. Bagaria, and J. Zou. Exploring patterns enriched in a dataset with contrastive principal component analysis, Nature Communicationsvolume,Vol. 9, No. 1, pp. 2134, 2018.
    * https://github.com/abidlabs/contrastive
  * Also, full automatic selection of contrastive parameter alpha from [Fujiwara et al., 2022].
    * T. Fujiwara, J. Zhao, F. Chen, Y. Xu, and K.-L. Ma. Network Comparison with Interpretable Contrastive Network Representation Learning, Journal of Data Science, Statistics, and Visualisation, 2022.

  * An extended version of cPCA (ccPCA) for contrasting clusters.

  * Algorithms for generating an effective, scalable feature contribution visualization, including optimal sign flipping of (c)PCA, matrix reordering, and aggregation.

* Use case examples
  * Analysis of dimensionality reduction results (compare some points with others)
  * Analysis of clustering results (compare some cluster with others)
  * Analysis of labeled data (compare some label with others)

******

Requirements
-----
* All major OSs are supported (macOS, Linux, Windows)

* C++11 compiler, Python3, Eigen3, Pybind11, Numpy

* Note: Tested on macOS Sequoia, Ubuntu 22.0.4 LTS, Windows 10. Currently, usage in <span style="color:#8888ff">Google Colab is not supported</span>. (This is because when using ccpca via Python, ccpca needs to import shared libraries produced by Pybind11 ('cpca_cpp.so' and 'ccpca_cpp.so').  I will appreciate if somebody can help solve this problem.)

******

Setup
-----
#### Mac OS with Homebrew

##### 1) Installation of ccPCA

* Make sure if you have C++ compiler. For example,

  `which c++`

  should return the c++ compiler path (e.g., /usr/bin/c++) if it exists. If it does not exist, run:

  `xcode-select --install`

* Move to "ccpca/ccpca" directory

* Run presetup.py

    `python3 presetup.py`

    - Note (2024-04-14): For installation using virtualenv, proccesses listed in "presetup.py" are separated from "setup.py". This is due to venv's bug (module involved commands do not work correctly when using pip3 under venv).
    - This python script generates C++ shared library of inc-pca also installs homebrew, pkg-config, python3, eigen, pybind11 if they do not exist.


* Install the modules with pip3 (this installs numpy if it does not exist).

    `pip3 install .`

* You can test with sample.py

    `pip3 install matplotlib scikit-learn`

    `python3 sample.py`

##### 2) Installation of fc-view

* If you want to use the algorithms for scalable visualization of features' contributions, please follow the next steps.

* Move to "ccpca/fc_view" directory

* Install the modules with pip3

    `pip3 install . `

#### Linux (tested on Ubuntu 22.0.4 LTS)

##### 1) Installation of ccPCA

* Install libraries

    `sudo apt update`

    `sudo apt install libeigen3-dev python3-pip python3-dev`

    * Note: Replace apt commands based on your Linux OS.

* Move to "ccpca/ccpca" directory

* Install the modules with pip3.

    `python3 presetup.py`

    `pip3 install .`

    * Note: If installation does not work, check setup.py and replace c++ commands based on your environment.

* You can test with sample.py

    `pip3 install matplotlib scikit-learn`

    `python3 sample.py`

##### 2) Installation of fc-view

* If you want to use the algorithms for scalable visualization of features' contributions, please follow the next steps.

* Move to "ccpca/fc_view" directory

* Install the modules with pip3
    
    `pip3 install . `

#### Windows (tested on Windows 10, <span style="color:#ff8888">requiring MSVC as a C++ compiler</span>)

##### 1) Installation of ccPCA

* Install required compiler and library

    - Install MSVC (Microsoft C++): For example, you can download from https://visualstudio.microsoft.com/downloads/?q=build+tools
      (note: other compilers are not supported, e.g., MinGW)

    - Install Python3 (https://www.python.org/downloads/windows/)


* Move to "ccpca/ccpca" directory

* Install the modules with pip3 in "*Command Prompt for VS*". <span style="color:#ff8888">Note: if you installed 64-bit Python3, use *x64 Native Command Prompt for VS*</span>.

    `python presetup.py`
    
    `pip3 install .`


* You can test with sample.py

    `pip3 install matplotlib scikit-learn`

    `python sample.py`

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
IEEE Transactions on Visualization and Computer Graphics, 2020.
DOI: [10.1109/TVCG.2019.2934251](https://doi.org/10.1109/TVCG.2019.2934251)

If you use cPCA with automatic selection of contrastive parameter alpha, please cite:
Takanori Fujiwara, Jian Zhao, Francine Chen, Yaoliang Xu, and Kwan-Liu Ma,
"Network Comparison with Interpretable Contrastive Network Representation Learning". 	Journal of Data Science, Statistics, and Visualisation, 2022.
DOI: [10.52933/jdssv.v2i5.56](https://doi.org/10.52933/jdssv.v2i5.56)
