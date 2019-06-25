class OptSignFlip():
    """
    The optimal sign flipping introduced in [Fujiwara et al., 2019].
    Parameters
    ----------
    None
    Attributes
    ----------
    None
    Examples
    --------
    Please refer to sample.ipynb (jupyter notebook).
    References
    ----------
    T. Fujiwara, O.-H. Kwon, and K.-L. Ma, Supporting Analysis of Dimensionality
    Reduction Results with Contrastive Learning, arXiv preprint, 2019.
    """

    def __init__(self):
        None

    @classmethod
    def _opt_sym_mat_sign(cls, X, set_diag_zero=True):
        import numpy as np

        if set_diag_zero:
            np.fill_diagonal(X, 0)

        n, _ = X.shape
        col_sum = np.sum(X, axis=0)
        sum_of_col_sum = np.sum(col_sum)

        # flip negative value until sum_of_col_sum > 0
        # counter i is just in case to avoid infinite loop (should not happen though)
        i = 0
        flips = [False] * n
        while np.any(col_sum < 0) and i < n:
            large_neg_val_idx = np.array(np.argsort(col_sum)).flatten()[0]
            X[large_neg_val_idx, :] *= -1
            X[:, large_neg_val_idx] *= -1
            flips[large_neg_val_idx] = not flips[large_neg_val_idx]
            col_sum = np.sum(X, axis=0)
            i += 1

        return flips

    @classmethod
    def opt_sign_flip(cls, first_cpc_mat, feat_contrib_mat=None):
        """
        Apply the optimal sign flipping to the first cPC matrix and feature
        contributions' matrix.
        Parameters
        ----------
        first_cpc_mat: array, shape(n_features, n_clusters)
            The first cPC for each cluster.
        feat_contrib_mat: array, shape(n_features, n_clusters), optional
            The features' contributions for each cluster.
        Returns
        -------
        fcluster : ndarray
            An array of length n_columns. T[i] is the flat cluster number to
            which original observation i belongs.
        """
        import numpy as np
        # generate cosine similarity matrix
        dot_prod_mat = first_cpc_mat.transpose().dot(first_cpc_mat)
        norms = np.linalg.norm(first_cpc_mat, axis=0)
        norm_mat = np.tile(norms, (first_cpc_mat.shape[1], 1))
        norm_mat = norm_mat.transpose() * norms
        cos_mat = dot_prod_mat / norm_mat

        # minimize negative elements by sign flipping
        flips = cls._opt_sym_mat_sign(cos_mat)

        # perform flip
        for i, flip in enumerate(flips):
            if flip:
                first_cpc_mat[:, i] *= -1
        if feat_contrib_mat is not None:
            for i, flip in enumerate(flips):
                if flip:
                    feat_contrib_mat[:, i] *= -1
