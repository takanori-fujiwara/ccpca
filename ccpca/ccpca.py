import os
import multiprocess as mp

import numpy as np
from scipy.linalg import eigh
from cpca import CPCA


def _scaled_var(a, b):
    min_val = np.min((a.min(), b.min()))
    max_val = np.max((a.max(), b.max()))
    val_range = np.max((max_val - min_val, np.finfo(a.dtype).tiny))
    var_a = np.mean(((a - np.mean(a)) / val_range) ** 2)
    var_b = np.mean(((b - np.mean(b)) / val_range) ** 2)

    return var_a, var_b


def _bin_width_scott(vals):
    sd = 0.0
    if vals.size > 1:
        sd = np.sqrt(((vals - vals.mean()) ** 2).sum() / float(vals.size - 1))
    denom = np.max((np.power(vals.size, 1.0 / 3.0), np.finfo(vals.dtype).tiny))
    bin_width = 3.5 * sd / denom

    return bin_width


def _hist_intersect(a, b):
    min_val = np.min((a.min(), b.min()))
    max_val = np.max((a.max(), b.max()))
    val_range = np.max((max_val - min_val, np.finfo(a.dtype).tiny))

    tmp_a = (a - min_val) / val_range
    tmp_b = (b - min_val) / val_range
    tmp_ab = np.hstack((tmp_a, tmp_b))

    bin_w = np.max((_bin_width_scott(tmp_ab), np.finfo(tmp_ab.dtype).tiny))
    n_bins = int(1.0 / bin_w) + 1

    counts_a = np.zeros(n_bins, dtype=int)
    counts_b = np.zeros(n_bins, dtype=int)
    for val in tmp_a:
        bin_index = int(val / bin_w)
        counts_a[bin_index] += 1
    for val in tmp_b:
        bin_index = int(val / bin_w)
        counts_b[bin_index] += 1

    hist_intersect = 0
    for i in range(n_bins):
        hist_intersect += np.min((counts_a[i], counts_b[i]))

    return hist_intersect


class CCPCA:
    """ccPCA. A variation of cPCA for contrasting the target cluster to the
    others.

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
    ...       str(ccpca.get_best_alpha()))
    >>> plt.show()
    """

    def __init__(self, n_components=2, standardize=True):
        self.components_ = None
        self.eigenvalues_ = None
        self.total_pos_eigenvalue_ = None
        self.loadings_ = None
        self.best_alpha_ = None
        self.feat_contribs_ = None
        self.reports_ = None

        self._cpca = CPCA(n_components, standardize)

    def fit(
        self,
        K,
        R,
        auto_alpha_selection=True,
        alpha=None,
        var_thres_ratio=0.5,
        parallel=False,
        n_alphas=40,
        max_log_alpha=1.0,
        keep_reports=False,
    ):
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
        parallel: bool, optional, (default=False)
            If True, multiprocessing will be used for calculation (advanced use).
            Python implementation after Version 0.2.2 relies on Pathos for multiprocessing.
            But, Pathos sometimes produces an error, and for now, we set this parameter False by default.
        n_alphas: int, optional, (default=40)
            A number of alphas to check to find the best one.
        max_log_alpha: float, optional, (default=1.0)
            10.0 ** max_log_alpha is the maximum value of alpha will be used.
            Although the original cPCA by [Abid and Zhang et al., 2018] set 3.0
            by default, we decided to set 1.0 by default after Version 0.2.4 based on our experience.
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

        if np.any(np.isnan(K)):
            raise ArithmeticError("the target cluster data must not contain any NA.")
        if np.any(np.isnan(R)):
            raise ArithmeticError("the data must not contain any NA.")

        if auto_alpha_selection:
            self._fit_with_best_alpha(
                K, R, var_thres_ratio, parallel, n_alphas, max_log_alpha, keep_reports
            )
        else:
            self._fit_with_manual_alpha(K, R, alpha)

        self._copy_attr_vals_from_cpca()
        if self.loadings_ is not None and self.loadings_.shape[1] > 0:
            self.feat_contribs_ = self.loadings_[:, 0]

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
        return self._cpca.transform(X)

    def fit_transform(
        self,
        K,
        R,
        auto_alpha_selection=True,
        alpha=None,
        var_thres_ratio=0.5,
        parallel=False,
        n_alphas=40,
        max_log_alpha=1.0,
        keep_reports=False,
    ):
        """Applying fit first and then return transformed matrix of E (i.e.,
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
        parallel: bool, optional, (default=False)
            If True, multiprocessing will be used for calculation (advanced use).
            Python implementation after Version 0.2.2 relies on Pathos for multiprocessing.
            But, Pathos sometimes produces an error, and for now, we set this parameter False by default.
        n_alphas: int, optional, (default=40)
            A number of alphas to check to find the best one.
        max_log_alpha: float, optional, (default=1.0)
            10.0 ** max_log_alpha is the maximum value of alpha will be used.
            Although the original cPCA by [Abid and Zhang et al., 2018] set 3.0
            by default, we decided to set 1.0 by default after Version 0.2.4 based on our experience.
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
        self.fit(
            K,
            R,
            auto_alpha_selection,
            alpha,
            var_thres_ratio,
            parallel,
            n_alphas,
            max_log_alpha,
            keep_reports,
        )

        return self.transform(self._cpca._fg)

    # getter methods to be consistent with previous version of CCPCA
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
        return self.feat_contribs_

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
        return self.feat_contribs_ / np.max(np.abs(self.feat_contribs_))

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
        return self.components_

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
        if self.components_ is None:
            print("Run fit() before get_component()")
            return
        return self.components_[:, index]

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
        if self.components_ is None:
            print("Run fit() before get_first_component()")
            return
        return self.components_[:, 0]

    def get_eigenvalues(self):
        """Returns current eigenvalues.

        Parameters
        ----------
        None.
        Returns
        -------
        eigenvalues: array-like, shape(, n_components)
            Contrastive principal components' eigenvalues.
        """
        return self.eigenvalues_

    def get_eigenvalue(self, index):
        """Returns i-th eigenvalue.

        Parameters
        ----------
        index: int
            Indicates i-th eigenvalue.
        Returns
        -------
        eigenvalue: float
            i-th eigenvalue.
        """
        if self.eigenvalues_ is None:
            print("Run fit() before get_eigenvalue()")
            return
        return self.eigenvalues_[index]

    def get_total_pos_eigenvalue(self):
        """Returns the total of n_features positive eigenvalues (not n_components).
        This value can be used to compute the explained ratio of variance of the matrix C.

        Parameters
        ----------
        None
        Returns
        -------
        total_pos_eigenvalue: float
            The total of positive eigenvalues.
        """
        return self.total_pos_eigenvalue_

    def get_loadings(self):
        """Returns current principal component loadings.

        Parameters
        ----------
        None.
        Returns
        -------
        loadings: array-like, shape(n_features, n_components)
            Contrastive principal component loadings.
        """
        return self.loadings_

    def get_loading(self, index):
        """Returns i-th principal component loading.

        Parameters
        ----------
        index: int
            Indicates i-th principal component loading.
        Returns
        -------
        loading: array-like, shape(1, n_components)
            i-th principal component loading.
        """
        if self.loadings_ is None:
            print("Run fit() before get_loading()")
            return
        return self.loadings_[:, index]

    def get_best_alpha(self):
        """Returns best alpha found with fit() with auto_selection_alpha=True
        Parameters
        ----------
        None
        -------
        best_alpha: float
            The found best alpha.
        """
        return self.best_alpha_

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
        return self.reports_

    def _copy_attr_vals_from_cpca(self):
        self.components_ = self._cpca.get_components()
        self.eigenvalues_ = self._cpca.get_eigenvalues()
        self.total_pos_eigenvalue_ = self._cpca.get_total_pos_eigenvalue()
        self.loadings_ = self._cpca.get_loadings()

    def _fit_with_manual_alpha(self, K, R, alpha):
        if K.shape[1] != R.shape[1]:
            raise ValueError("# of cols in K and R must be the same.")
        self._cpca.fit(np.vstack((K, R)), R, auto_alpha_selection=False, alpha=alpha)

        return self

    def _update_best_alpha(
        self,
        K,
        R,
        var_thres_ratio=0.5,
        parallel=False,
        n_alphas=40,
        max_log_alpha=1.0,
        keep_reports=False,
    ):
        """Finds the best contrast parameter alpha which has high discrepancy
        score between the dimensionality reduced K and the dimensionality
        reduced R while keeping the variance of K with the ratio threshold
        var_thres_ratio.
        """
        if K.shape[1] != R.shape[1]:
            raise ValueError("# of cols in K and R must be the same.")

        self._cpca.fit(np.vstack((K, R)), R, auto_alpha_selection=False, alpha=0.0)

        self.best_alpha_ = 0.0
        proj_A = self._cpca.transform(self._cpca._fg)[:, 0]
        proj_K = proj_A[: K.shape[0]]
        proj_R = proj_A[K.shape[0] :]
        base_var_K, _ = _scaled_var(proj_K, proj_R)

        best_discrepancy = 1.0 / np.max(
            (_hist_intersect(proj_K, proj_R), np.finfo(proj_K.dtype).tiny)
        )

        self.reports_ = []
        if keep_reports:
            self.reports_.append(
                (
                    0.0,
                    best_discrepancy,
                    base_var_K,
                    proj_K,
                    proj_R,
                    self._cpca.get_loading(0),
                )
            )
            if parallel:
                parallel = False
                print(
                    "current version keep_reports only support non-parallel running. parallel is turned off."
                )

        alphas = np.logspace(-1, max_log_alpha, num=n_alphas - 1, dtype=K.dtype)
        if not parallel:
            for alpha in alphas:
                self._cpca.update_components(alpha)
                proj_A = self._cpca.transform(self._cpca._fg)[:, 0]
                proj_K = proj_A[: K.shape[0]]
                proj_R = proj_A[K.shape[0] :]

                discrepancy = 1.0 / np.max(
                    (
                        _hist_intersect(proj_K, proj_R),
                        np.finfo(proj_K.dtype).tiny,
                    )
                )
                var_K, _ = _scaled_var(proj_K, proj_R)

                if (var_K >= base_var_K * var_thres_ratio) and (
                    discrepancy > best_discrepancy
                ):
                    best_discrepancy = discrepancy
                    self.best_alpha_ = alpha

                if keep_reports:
                    self.reports_.append(
                        (
                            alpha,
                            best_discrepancy,
                            var_K,
                            proj_K,
                            proj_R,
                            self._cpca.get_loading(0),
                        )
                    )
        else:
            n_workers = np.max((os.cpu_count(), 1))
            A = self._cpca._fg.copy()
            Cov_A = self._cpca._C_fg.copy()
            Cov_R = self._cpca._C_bg.copy()

            def worker_task(alpha):
                # Less locks but fewer steps
                diff_cov = Cov_A - alpha * Cov_R
                _, eigenvectors = eigh(diff_cov)
                component = eigenvectors[:, -1]
                # # More locks but fewer steps
                # self._cpca.update_components(alpha)
                # component = self._cpca.get_component(0)

                proj_A = A @ component
                proj_K = proj_A[: K.shape[0]]
                proj_R = proj_A[K.shape[0] :]
                var_K, _ = _scaled_var(proj_K, proj_R)
                discrepancy = 1.0 / np.max(
                    (_hist_intersect(proj_K, proj_R), np.finfo(proj_K.dtype).tiny)
                )

                return var_K, discrepancy

            with mp.Pool(processes=n_workers) as pool:
                results = pool.map(worker_task, alphas)

            for alpha, (var, discrepancy) in zip(alphas, results):
                if (var >= base_var_K * var_thres_ratio) and (
                    discrepancy > best_discrepancy
                ):
                    best_discrepancy = discrepancy
                    self.best_alpha_ = alpha

        return self.best_alpha_

    def _fit_with_best_alpha(
        self, K, R, var_thres_ratio, parallel, n_alphas, max_log_alpha, keep_reports
    ):
        self._update_best_alpha(
            K, R, var_thres_ratio, parallel, n_alphas, max_log_alpha, keep_reports
        )
        self._cpca.update_components(self.best_alpha_)

        return self
