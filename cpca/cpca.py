import numpy as np
from scipy.linalg import eigh


class CPCA:
    """Contrastive PCA with efficient implemetation and automatic alpha selection.

    Parameters
    ----------
    n_components: int, optional, (default=2)
        A number of componentes to take.
    standardize: boo, optional, (default=True)
        Whether standardize input matrices or not.
    Attributes
    ----------
    None.
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from sklearn import datasets

    >>> from cpca import CPCA

    >>> dataset = datasets.load_iris()
    >>> X = dataset.data
    >>> y = dataset.target

    >>> # apply cPCA
    >>> cpca = CPCA()
    >>> X_new = cpca.fit_transform(fg=X, bg=X[y != 0], alpha=0.84)

    >>> # plot figures
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
    >>> plt.title('cPCA of IRIS dataset with alpha=0.84')
    >>> plt.show()
    """

    def __init__(self, n_components=2, standardize=True):
        self.n_components = n_components
        self.standardize = standardize
        self.components_ = None
        self.eigenvalues_ = None
        self.total_pos_eigenvalue_ = None
        self.loadings_ = None
        self.best_alpha_ = None
        self.reports_ = None

        self._fg = None
        self._bg = None
        self._C_fg = None
        self._C_bg = None

    def fit(
        self,
        fg,
        bg,
        auto_alpha_selection=True,
        alpha=None,
        eta=1e-3,
        convergence_ratio=1e-2,
        max_iter=10,
        keep_reports=False,
    ):
        """Fit the model with a foreground matrix and a background matrix.

        Parameters
        ----------
        fg: array-like, shape (n_samples, n_features)
            A foreground (or target) dataset.
        bg: array-like, shape (n_samples, n_features)
            A background dataset. This column size must be the same size with
            fg. (A row size can be different from fg.)
        auto_alpha_selection:
            If True, find auto_alpha_selection for fit. Otherwise, compute PCs
            based on input alpha.
        eta: float, optional, (default=1e-3)
            Small constant value that will add to covariance matrix of bg when
            applying automatic alpha selection. Smaller eta tends to allow
            a larger alpha as the best alpha.
        convergence_ratio: float, optional, (default=1e-2)
            Threshold of improvement ratio for convergence of automatic alpha
            selection.
        max_iter=10: int, optional, (default=10)
            The number of alpha updates at most.
        keep_reports: bool, optional, (default=False)
            If True, while automatic alpha selection, reports are recorded. The
            reports are the history of "alpha" values.
        Returns
        -------
        self.
        """
        if alpha == None:
            alpha = 0.0
            auto_alpha_selection = True
        else:
            auto_alpha_selection = False

        if auto_alpha_selection:
            self._fit_with_best_alpha(
                fg, bg, alpha, eta, convergence_ratio, max_iter, keep_reports
            )
        else:
            self._fit_with_manual_alpha(fg, bg, alpha)

        return self

    def transform(self, X):
        """Obtaining transformed result Y with X and current PCs.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Testing data, where n_samples is the number of samples and
            n_features is the number of features. n_features must be the same
            size with the traiding data's features used for partial_fit.
        Returns
        -------
        Y : array-like, shape (n_samples, n_components)
            The transformed (or projected) result.
        """
        if self.components_ is None:
            print("Run fit() before transform()")
            return

        return X @ self.components_

    def fit_transform(
        self,
        fg,
        bg,
        auto_alpha_selection=True,
        alpha=None,
        eta=1e-3,
        convergence_ratio=1e-2,
        max_iter=10,
        keep_reports=False,
    ):
        """Fit the model with a foreground matrix and a background matrix and
        then obtain transformed result of fg and current PCs.

        Parameters
        ----------
        fg: array-like, shape (n_samples, n_features)
            A foreground (or target) dataset.
        bg: array-like, shape (n_samples, n_features)
            A background dataset. This column size must be the same size with
            fg. (A row size can be different from fg.)
        auto_alpha_selection:
            If True, find auto_alpha_selection for fit. Otherwise, compute PCs
            based on input alpha.
        alpha: float
            A contrast parameter, which quantifies the trade-off between having
            high target variance and low background variance. alpha must be
            equal to or larger than 0. If 0, the result will be the same with
            the ordinary PCA. If auto_alpha_selection is True, this alpha is
            used as an initial alpha value for auto selection.
        eta: float, optional, (default=1e-3)
            Small constant value that will add to covariance matrix of bg when
            applying automatic alpha selection. Smaller eta tends to allow
            a larger alpha as the best alpha.
        convergence_ratio: float, optional, (default=1e-2)
            Threshold of improvement ratio for convergence of automatic alpha
            selection.
        max_iter=10: int, optional, (default=10)
            The number of alpha updates at most.
        keep_reports: bool, optional, (default=False)
            If True, while automatic alpha selection, reports are recorded. The
            reports are the history of "alpha" values.
        Returns
        -------
        Y : array-like, shape (n_samples, n_components)
            The transformed (or projected) result.
        """
        self.fit(
            fg,
            bg,
            auto_alpha_selection,
            alpha,
            eta,
            convergence_ratio,
            max_iter,
            keep_reports,
        )
        return self.transform(fg)

    def update_components(self, alpha):
        """Update the components with a new contrast parameter alpha. Before
        using this, at least one time, fit() or fit_transform() must be called
        to build an initial result of the components.

        Parameters
        ----------
        alpha: float
            A contrast parameter, which quantifies the trade-off between having
            high target variance and low background variance. alpha must be
            equal to or larger than 0. If 0, the result will be the same with
            the ordinary PCA.
        Returns
        -------
        None.
        """
        if self.components_ is None:
            print("Run fit() at least once before update_components()")
            return
        self._update_components(alpha)

        return self

    # getter methods to be consistent with previous version of CPCA
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

    def get_reports(self):
        """Returns the reports kept while automatic selection of alpha. To get
        reports, you need to set keep_reports=True in the corresponding method.
        Parameters
        ----------
        None
        -------
        reports: array-like(n_alphas, 1),
            alpha values (at the same time optimization scores) obtained through
            best alpha selection".
        """
        return self.reports_

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

    def _update_components(self, alpha):
        w, v = eigh(self._C_fg - alpha * self._C_bg)
        self.components_ = v[:, np.argsort(-np.real(w))[: self.n_components]]
        self.eigenvalues_ = w[np.argsort(-np.real(w))[: self.n_components]]

        # Update related info
        self.total_pos_eigenvalue_ = self.eigenvalues_[self.eigenvalues_ > 0].sum()
        self.loadings_ = self.components_ * np.sqrt(np.abs(self.eigenvalues_))

        return self

    def _fit_with_manual_alpha(self, fg, bg, alpha):
        self._fg = fg
        self._bg = bg

        if self._fg.size == 0 and self._bg.size == 0:
            print("Both target and background matrices are empty.")
        elif self._fg.size == 0:
            # the result will be the same with when alpha is +inf
            self._fg = np.zeros((1, self._bg.shape[1]))
        elif self._bg.size == 0:
            # the result will be the same with ordinary PCA
            self._bg = np.zeros((1, self._fg.shape[1]))

        if self._fg.shape[1] != self._bg.shape[1]:
            print("# of features of foregraound and background must be the same.")

        self._fg -= np.mean(self._fg, axis=0)
        self._bg -= np.mean(self._bg, axis=0)

        if self.standardize:
            self._fg /= np.std(self._fg, axis=0)
            self._bg /= np.std(self._bg, axis=0)

            # NaN to 0.0f
            self._fg[np.isnan(self._fg)] = 0.0
            self._bg[np.isnan(self._bg)] = 0.0

        # Covariance matrix
        self._C_fg = (self._fg.T @ self._fg) / max(
            self._fg.shape[0] - 1, np.finfo(self._fg.dtype).tiny
        )
        self._C_bg = (self._bg.T @ self._bg) / max(
            self._bg.shape[0] - 1, np.finfo(self._bg.dtype).tiny
        )

        self._update_components(alpha)

        return self

    def _update_best_alpha(
        self, fg, bg, init_alpha, eta, convergence_ratio, max_iter, keep_reports
    ):
        """Automatic alpha selection for cPCA to maximize the variance ratio of target and backgroud.
        This implementation is based on:
        - Fujiwara et al., "Network Comparison with Interpretable Contrastive Network Representation Learning", JDSSV, 2022.
        """
        self.reports_ = []
        alpha = init_alpha
        self._fit_with_manual_alpha(fg, bg, alpha)

        # method 2: add small constant to diag of bgCov_ to avoid singular
        # (Note: refer to cpca.cpp to know what is method 1)
        self._C_bg += np.eye(self._C_bg.shape[0]) * eta

        if keep_reports:
            self.reports_.append(alpha)

        for _ in range(max_iter):
            tr_fg = np.trace(self.components_.T @ self._C_fg @ self.components_)
            tr_bg = np.trace(self.components_.T @ self._C_bg @ self.components_)
            tr_bg = max(tr_bg, np.finfo(self._C_bg.dtype).tiny)

            prev_alpha = alpha
            alpha = tr_fg / tr_bg
            self.update_components(alpha)
            if keep_reports:
                self.reports_.append(alpha)

            if (abs(prev_alpha - alpha) / alpha) < convergence_ratio:
                break

        self.best_alpha_ = alpha

        return self.best_alpha_

    def _fit_with_best_alpha(
        self, fg, bg, init_alpha, eta, convergence_ratio, max_iter, keep_reports
    ):
        self.best_alpha_ = self._update_best_alpha(
            fg, bg, init_alpha, eta, convergence_ratio, max_iter, keep_reports
        )
        self._fit_with_manual_alpha(fg, bg, self.best_alpha_)

        return self
