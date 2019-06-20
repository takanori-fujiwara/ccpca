#  Because pybind11 cannot generate default parameters well, this code is to set them
import cpca_cpp


class CPCA(cpca_cpp.CPCA):
    """Contrastive PCA with efficient C++ implemetation with Eigen.

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
    Notes
    -----
    """

    def __init__(self, n_components=2, standardize=True):
        super().__init__(n_components, standardize)

    def initialize(self):
        """Reset components obtained by fit()
        """
        super().initialize()

    def fit(self, fg, bg, alpha):
        """Fit the model with a foreground matrix and a background matrix.

        Parameters
        ----------
        fg: array-like, shape (n_samples, n_features)
            A foreground (or target) dataset.
        bg: array-like, shape (n_samples, n_features)
            A background dataset. This column size must be the same size with
            fg. (A row size can be different from fg.)
        alpha: float
            A contrast parameter, which quantifies the trade-off between having
            high target variance and low background variance. alpha must be
            equal to or larger than 0. If 0, the result will be the same with
            the ordinary PCA.
        Returns
        -------
        None.
        """
        super().fit(fg, bg, alpha)

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
        return super().transform(X)

    def fit_transform(self, fg, bg, alpha):
        """Fit the model with fg and bg, and then apply the dimensionality
        reduction on fg.

        Parameters
        ----------
        fg: array-like, shape (n_samples, n_features)
            A foreground (or target) dataset.
        bg: array-like, shape (n_samples, n_features)
            A background dataset. This column size must be the same size with
            fg. (A row size can be different from fg.)
        alpha: float
            A contrast parameter, which quantifies the trade-off between having
            high target variance and low background variance. alpha must be
            equal to or larger than 0. If 0, the result will be the same with
            the ordinary PCA.
        Returns
        -------
        Y : array-like, shape (n_samples, n_components)
            The transformed (or projected) result.
        """
        return super().fit_transform(fg, bg, alpha)

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
        super().update_components(alpha)

    def logspace(self, start, end, num, base=10.0):
        """Generate logarithmic space.

        Parameters
        ----------
        start: float
            base ** start is the starting value of the sequence.
        end: float
            base ** end is the ending value of the sequence.
        num: int
            Number of samples to generate.
        base : float, optional, (default=10.0)
            The base of the log space. The step size between the elements in
            ln(samples) / ln(base) (or log_base(samples)) is uniform.
        Returns
        -------
        samples : ndarray
            Num samples, equally spaced on a log scale.
        None.
        """
        return super().logspace(start, end, num, base)

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
