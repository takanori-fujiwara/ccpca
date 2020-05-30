#  Because pybind11 cannot generate default parameters well, this code is to set them

import ccpca_cpp
import numpy as np


class CCPCA(ccpca_cpp.CCPCA):
    """ccPCA. A variation of cPCA for contrasting the target cluster to the
    others

    Parameters
    ----------
    n_components: int, optional, (default=2)
        A number of componentes to take.
    standardize: boo, optional, (default=True)
        Whether standardize input matrices or not.
    Attributes
    ----------
    None.
    ----------
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from sklearn import datasets

    >>> from ccpca import CCPCA

    >>> dataset = datasets.load_iris()
    >>> X = dataset.data
    >>> y = dataset.target

    >>> # get dimensionality reduction result with the best alpha
    >>> ccpca = CCPCA()
    >>> ccpca.fit(X[y == 0], X[y != 0], var_thres_ratio=0.5, max_log_alpha=0.5)
    >>> X_new = ccpca.transform(X)

    >>> # plot result
    >>> plt.figure()
    >>> colors = ['navy', 'turquoise', 'darkorange']
    >>> lw = 2
    >>> for color, i, target_name in zip(colors, [0, 1, 2], [0, 1, 2]):
    ...     plt.scatter(
    ...         X_new[y == i, 0],
    ...         X_new[y == i, 1],
    ...         color=color,
    ...         alpha=.8,
    ...         lw=lw,
    ...         label=target_name)
    >>> plt.legend(loc='best', shadow=False, scatterpoints=1)
    >>> plt.title('cPCA of IRIS dataset with automatic alpha =' +
    ...       str(dc.get_best_alpha()))
    >>> plt.show()
    Notes
    -----
    For python version, fit_transform is not implemented because it does not
    look like work properly via pybind11 right now.
    """

    def __init__(self, n_components=2, standardize=True):
        super().__init__(n_components, standardize)

    # def fit_transform(self,
    #                   K,
    #                   R,
    #                   auto_alpha_selection=True,
    #                   alpha=None,
    #                   var_thres_ratio=0.5,
    #                   parallel=True,
    #                   n_alphas=40,
    #                   max_log_alpha=3.0,
    #                   keep_reports=False):
    #     """TODO: some bug while binding C++ code with pybind11, fit_transform
    #     deos not seem to work properly
    #
    #     If using auto alpha selection, find the best contrast parameter alpha
    #     first. Otherwise, input alpha value is used for fit.
    #     Then, fit using cPCA with the (best) alpha, and transform a matrix
    #     concatenating K and R with cPCs. For cPCA, a matrix E concatenating K
    #     and R will be used as a foreground dataset and R will be used as a
    #     background dataset.
    #
    #     Parameters
    #     ----------
    #     K: array-like, shape(n_samples, n_components)
    #         A target cluster.
    #     R: array of array-like, n_groups x shape(n_samples, n_components)
    #         Background datasets.
    #     auto_alpha_selection: bool, optional (default=True)
    #         If True, the best alpha is automatically found and used for fit().
    #         If False, input alpha is used for fit().
    #     alpha: None or float, optional (default=None)
    #         If using manual alpha selection (i.e., auto_alpha_selection is
    #         False), this alpha value is used for fit(). If alpha is None, alpha
    #         is automatically set to 0.0.
    #     var_thres_ratio: float, optional, (default=0.5)
    #         Ratio threshold of variance of K to keep (the parameter gamma in
    #         our paper).
    #     parallel: bool, optional, (default=True)
    #         If True, multithread implemented in C++ will be used for
    #         calculation.
    #     n_alphas: int, optional, (default=40)
    #         A number of alphas to check to find the best one.
    #     max_log_alpha: float, optional, (default=3.0)
    #         10.0 ** max_log_alpha is the maximum value of alpha will be used.
    #         Even though this default parameter (i.e., 3.0) follows the original
    #         cPCA by [Abid and Zhang et al., 2018], you may want to set a much
    #         smaller value based on the dataset (e.g., around 0.5 or 1.0 works
    #         well from our experience).
    #     keep_reports: bool, optional, (default=False)
    #         If True, while automatic alpha selection, reports are recorded. The
    #         reports include "alpha", "discrepancy score", "variance score",
    #         "1D projection of K", "1D projection of R", and "cPC loadings".
    #         These reports can be obtained via get_reports() method.
    #         Use parallel=False togerther. Currently, this function does not
    #         support when running in parallel.
    #     Returns
    #     -------
    #     None
    #     """
    #     if alpha == None:
    #         alpha = 0.0
    #
    #     return super().fit_transform(K, R, auto_alpha_selection, alpha,
    #                                  var_thres_ratio, parallel, n_alphas,
    #                                  max_log_alpha, keep_reports)

    def fit(self,
            K,
            R,
            auto_alpha_selection=True,
            alpha=None,
            var_thres_ratio=0.5,
            parallel=True,
            n_alphas=40,
            max_log_alpha=3.0,
            keep_reports=False):
        """If using auto alpha selection, find the best contrast parameter alpha
        first. Otherwise, input alpha value is used for fit. Then, fit using
        cPCA with the (best) alpha. For cPCA, a matrix E concatenating K and R
        will be used as a foreground dataset and R will be used as a background
        dataset.

        Parameters
        ----------
        K: array-like, shape(n_samples, n_components)
            A target cluster.
        R: array of array-like, n_groups x shape(n_samples, n_components)
            Background datasets.
        auto_alpha_selection: bool, optional (default=True)
            If True, the best alpha is automatically found and used for fit().
            If False, input alpha is used for fit().
        alpha: None or float, optional (default=None)
            If using manual alpha selection (i.e., auto_alpha_selection is
            False), this alpha value is used for fit(). If alpha is None, alpha
            is automatically set to 0.0 and auto_alpha_selection is used.
        var_thres_ratio: float, optional, (default=0.5)
            Ratio threshold of variance of K to keep (the parameter gamma in
            our paper).
        parallel: bool, optional, (default=True)
            If True, multithread implemented in C++ will be used for
            calculation.
        n_alphas: int, optional, (default=40)
            A number of alphas to check to find the best one.
        max_log_alpha: float, optional, (default=3.0)
            10.0 ** max_log_alpha is the maximum value of alpha will be used.
            Even though this default parameter (i.e., 3.0) follows the original
            cPCA by [Abid and Zhang et al., 2018], you may want to set a much
            smaller value based on the dataset (e.g., 0.5 or 1.0 works well
            from our experience).
        keep_reports: bool, optional, (default=False)
            If True, while automatic alpha selection, reports are recorded. The
            reports include "alpha", "discrepancy score", "variance score",
            "1D projection of K", "1D projection of R", and "cPC loadings".
            These reports can be obtained via get_reports() method.
            Use parallel=False togerther. Currently, this function does not
            support when running in parallel.
        Returns
        -------
        self.
        """
        if alpha == None:
            alpha = 0.0
            auto_alpha_selection = True
        else:
            auto_alpha_selection = False

        super().fit(K, R, auto_alpha_selection, alpha, var_thres_ratio,
                    parallel, n_alphas, max_log_alpha, keep_reports)

        return self

    def transform(self, X):
        """Obtaining transformed result Y with X and current cPCs.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Testing data, where n_samples is the number of samples and
            n_features is the number of features. n_features must be the same
            size with the traiding data's features used for partial_fit.
        Returns
        -------
        Y : array-like, shape (n_samples, n_components)
            Returns the transformed (or projected) result.
        """
        return super().transform(X)

    def fit_transform(self,
                      K,
                      R,
                      auto_alpha_selection=True,
                      alpha=None,
                      var_thres_ratio=0.5,
                      parallel=True,
                      n_alphas=40,
                      max_log_alpha=3.0,
                      keep_reports=False):
        """ Applying fit first and then return transformed matrix of E (i.e.,
        vertical stack of K and R).
        If using auto alpha selection, find the best contrast parameter alpha
        first. Otherwise, input alpha value is used for fit. Then, fit using
        cPCA with the (best) alpha. For cPCA, a matrix E concatenating K and R
        will be used as a foreground dataset and R will be used as a background
        dataset.

        Parameters
        ----------
        K: array-like, shape(n_samples, n_components)
            A target cluster.
        R: array of array-like, n_groups x shape(n_samples, n_components)
            Background datasets.
        auto_alpha_selection: bool, optional (default=True)
            If True, the best alpha is automatically found and used for fit().
            If False, input alpha is used for fit().
        alpha: None or float, optional (default=None)
            If using manual alpha selection (i.e., auto_alpha_selection is
            False), this alpha value is used for fit(). If alpha is None, alpha
            is automatically set to 0.0 and auto_alpha_selection is used.
        var_thres_ratio: float, optional, (default=0.5)
            Ratio threshold of variance of K to keep (the parameter gamma in
            our paper).
        parallel: bool, optional, (default=True)
            If True, multithread implemented in C++ will be used for
            calculation.
        n_alphas: int, optional, (default=40)
            A number of alphas to check to find the best one.
        max_log_alpha: float, optional, (default=3.0)
            10.0 ** max_log_alpha is the maximum value of alpha will be used.
            Even though this default parameter (i.e., 3.0) follows the original
            cPCA by [Abid and Zhang et al., 2018], you may want to set a much
            smaller value based on the dataset (e.g., 0.5 or 1.0 works well
            from our experience).
        keep_reports: bool, optional, (default=False)
            If True, while automatic alpha selection, reports are recorded. The
            reports include "alpha", "discrepancy score", "variance score",
            "1D projection of K", "1D projection of R", and "cPC loadings".
            These reports can be obtained via get_reports() method.
            Use parallel=False togerther. Currently, this function does not
            support when running in parallel.
        Returns
        -------
        Y : array-like, shape (n_samples, n_components)
            Returns the transformed (or projected) result.
        """
        self.fit(K, R, auto_alpha_selection, alpha, var_thres_ratio, parallel,
                 n_alphas, max_log_alpha, keep_reports)

        return self.transform(np.vstack((K, R)))

    def best_alpha(self,
                   K,
                   R,
                   var_thres_ratio=0.5,
                   parallel=True,
                   n_alphas=40,
                   max_log_alpha=3.0,
                   keep_reports=False):
        """Finds the best contrast parameter alpha which has high discrepancy
        score between the dimensionality reduced K and the dimensionality
        reduced R while keeping the variance of K with the ratio threshold
        var_thres_ratio.
        For cPCA, a matrix E concatenating K and R will be used as a foreground
        dataset and R will be used as a background dataset.
        Parameters
        ----------
        K: array-like, shape(n_samples, n_components)
            A target cluster.
        R: array of array-like, n_groups x shape(n_samples, n_components)
            Background datasets.
        var_thres_ratio: float, optional, (default=0.5)
            Ratio threshold of variance of K to keep (the parameter gamma in
            our paper).
        parallel: bool, optional, (default=True)
            If True, multithread implemented in C++ will be used for
            calculation.
        n_alphas: int, optional, (default=40)
            A number of alphas to check to find the best one.
        max_log_alpha: float, optional, (default=3.0)
            10.0 ** max_log_alpha is the maximum value of alpha will be used.
            Even though this default parameter (i.e., 3.0) follows the original
            cPCA by [Abid and Zhang et al., 2018], you may want to set a much
            smaller value based on the dataset (e.g., 0.5 or 1.0 works well
            from our experience).
        keep_reports: bool, optional, (default=False)
            If True, while automatic alpha selection, reports are recorded. The
            reports include "alpha", "discrepancy score", "variance score",
            "1D projection of K", "1D projection of R", and "cPC loadings".
            These reports can be obtained via get_reports() method.
            Use parallel=False togerther. Currently, this function does not
            support when running in parallel.
        Returns
        -------
        best_alpha: float
            The found best alpha.
        """
        return super().best_alpha(K, R, var_thres_ratio, parallel, n_alphas,
                                  max_log_alpha, keep_reports)

    def get_feat_contribs(self):
        """Returns feature contributions from current cPCA result. The
        feature contributions are the same value with the first cPC loading.
        Parameters
        ----------
        None
        Returns
        -------
        feat_contribs: array-like, shape(n_features, 1)
            Feature contributions.
        """
        return super().get_feat_contribs()

    def get_scaled_feat_contribs(self):
        """Returns scaled feature contributions from current cPCA result.
        Scaled feature contributions are in the range from -1 to 1 by dividing
        each feature contribution by the maximum absolute value of the FCs
        (e.g., the original range from -0.1 to 0.5 will be changed to the range
        from -0.2 to 1.0)
        Parameters
        ----------
        None
        Returns
        -------
        feat_contribs: array-like, shape(n_features, 1)
            Feature contributions.
        """
        return super().get_scaled_feat_contribs()

    def get_loadings(self):
        """Returns current principal component loadings.

        Parameters
        ----------
        None.
        Returns
        -------
        components: array-like, shape(n_features, n_components)
            Contrastive principal component loadings.
        """
        return super().get_loadings()

    def get_loading(self, index):
        """Returns i-th principal component loading.

        Parameters
        ----------
        index: int
            Indicates i-th principal component loading.
        Returns
        -------
        component: array-like, shape(1, n_components)
            i-th principal component loading.
        """
        return super().get_loading(index)

    def get_components(self):
        """Returns current components.

        Parameters
        ----------
        None.
        Returns
        -------
        components: array-like, shape(n_features, n_components)
            Contrastive principal components.
        """
        return super().get_components()

    def get_component(self, index):
        """Returns i-th component.

        Parameters
        ----------
        index: int
            Indicates i-th component.
        Returns
        -------
        component: array-like, shape(1, n_components)
            i-th contrastive principal component.
        """
        return super().get_component(index)

    def get_first_component(self):
        """Returns the firsrt PC from current cPCA result.
        Parameters
        ----------
        None
        Returns
        -------
        pc: array, shape(n_features)
            The first principal component.
        """
        return super().get_feat_contribs()

    def get_best_alpha(self):
        """Returns best alpha found with best_alpha()
        Parameters
        ----------
        None
        -------
        best_alpha: float
            The found best alpha.
        """
        return super().get_best_alpha()

    def get_reports(self):
        """Returns the reports kept while automatic selection of alpha. To get
        reports, you need to set keep_reports=True in the corresponding method.
        Parameters
        ----------
        None
        -------
        reports: tuple (float, float, float, array-like(n_samples, 1),
            array-like(n_samples, 1), array-like(n_features, 1).
            The order of tuple is "discrepancy score", "variance score",
            "1D projection of K", "1D projection of R", and "cPC loadings".
        """
        return super().get_reports()
