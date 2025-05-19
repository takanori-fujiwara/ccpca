## ccPCA: contrasting clusters in PCA - C++ Library and Python Module

New
-----
* Python implementation of cPCA, ccPCA is completely changed to only use python for easiler implementation (2025-04-23).
    - Compared with previous C++ binded version, this version has much faster when data size is large (but slightly slower for small data). Also, Python version supports any float precisions.
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
  * Fast C++ and Python implementations of Contrastive PCA (cPCA) from [Abid and Zhang et al., 2018].<br />
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

* Python3 (latest)

* Note: Tested on macOS Sequoia, Ubuntu 24.04 LTS, Google Colab, and Windows 10. 

******

Setup (Python implementation)
-----

* Install via PyPI:

  `pip3 install ccpca`

  Or, manual installation: download this repository, move to the downloaded repository, and then:

    `pip3 install .`

* You can test with sample.py

    `pip3 install matplotlib scikit-learn`

    `python3 sample.py`

    Note: when using Linux, you may need to install PyQt: e.g., `pip3 install PyQt6`

******

Usage
-----
* With Python3
    * Import installed modules from python, for example: 
        - `from cpca import CPCA`
        - `from ccpca import CCPCA`
        - `from fc_view import OptSignFlip, MatReorder`
    * See ccpca/sample.py and sample.ipynb for examples.
    * Also, there are detailed documentations in cpca/cpca.py, ccpca/ccpca.py, fc_view/mat_reorder.py, and fc_view/opt_sign_flip.py

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
