# Reordering heatmap with hierarchical clustering with optimal-leaf-ordering
import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist


class MatReorder():
    """Heatmap reordering and aggregation by hierarchical clustering with
    optimal-leaf-ordering.
    Parameters
    ----------
    None
    Attributes
    ----------
    order_row_: List, heatmap's row length
        The order of rows obtained by hierarchical clustering with
        optimal-leaf-ordering.
    order_col_: List, heatmap's column length
        The order of columns obtained by hierarchical clustering with
        optimal-leaf-ordering.
    Z_row_: ndarray
        The hierarchical clustering encoded as a linkage matrix for rows.
    Z_col_: ndarray
        The hierarchical clustering encoded as a linkage matrix for columns.
    Examples
    --------
    >>> import numpy as np; np.random.seed(0)
    >>> import matplotlib.pyplot as plt
    >>> from mat_reorder import MatReorder

    >>> heatmap = np.random.rand(50, 3)

    >>> # reordering
    >>> mr = MatReorder()
    >>> reordered_heatmap = mr.fit_transform(heatmap)

    >>> # aggregation
    >>> agg_reordered_heatmap, label_to_rows, label_to_rep_row = mr.aggregate_rows(heatmap,
    ...                                                                            10,
    ...                                                                            agg_method='abs_max')

    >>> # plot
    >>> fig, ax = plt.subplots(figsize=(4, 8))
    >>> im = ax.imshow(reordered_heatmap, cmap='coolwarm', aspect='auto')
    >>> plt.show()

    >>> fig, ax = plt.subplots(figsize=(4, 8))
    >>> im = ax.imshow(agg_reordered_heatmap, cmap='coolwarm', aspect='auto')
    >>> # add row names
    >>> row_names = label_to_rep_row
    >>> for i, name in enumerate(row_names):
    ...     rows = label_to_rows[i]
    ...     row_names[i] = str(name) + '('
    ...     for row in rows:
    ...        row_names[i] += str(row) + ','
    ...     row_names[i] += ')'
    >>> ax.set_yticks(np.arange(len(row_names)))
    >>> ax.yaxis.tick_right()
    >>> ax.set_yticklabels(row_names)
    >>> plt.show()
    Notes
    -----
    References
    ----------
    """

    def __init__(self):
        self.order_row_ = None
        self.order_col_ = None
        self.Z_row_ = None
        self.Z_col_ = None

    def fit(self,
            X,
            hclust_method='complete',
            optimal_leaf_ordering=True,
            pdist_metric='cosine',
            row_cluster=True,
            col_cluster=True):
        """Fit a heat map X into a clustered, reordered space.

        Parameters
        ----------
        X : array-like, shape (n_rows, n_columns)
            A matrix representing a heatmap.
        hclust_method : str, optional, (default='complete')
            The linkage algorithm to use.
            See the Linkage Methods section in
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html.
        optimal_leaf_ordering: bool, optional, (default=True)
            If True, the linkage matrix will be reordered so that the distance
            between successive leaves is minimal. This results in a more
            intuitive tree structure when the data are visualized. defaults to
            False, because this algorithm can be slow, particularly on large
            datasets. See also the optimal_leaf_ordering function in scipy.
        pdist_metric: str or function, optional, (default='cosine')
            The distance metric to use in the case that X is a collection of
            observation vectors; ignored otherwise. See the pdist function in
            scipy for a list of valid distance metrics. A custom distance
            function can also be used.
        row_cluster: bool, optional, (default=True)
            If True, apply hclust to X's rows. Otherwise, order_row_ will be
            the order from 0 to (n_rows - 1).
        col_cluster: bool, optional, (default=True)
            If True, apply hclust to X's columns. Otherwise, order_col_ will be
            the order from 0 to (n_columns - 1).
        Returns
        -------
        self: object
            hclust results for each row and col will be stored in Z_row_ and
            Z_col_. Also, the obtained order will be stored in order_row_ and
            order_col_.
        """
        if type(X) is not np.ndarray:
            print("input data needs to be a matrix")
        else:
            if not row_cluster:
                self.order_row_ = list(range(X.shape[0]))
            else:
                D_row = pdist(X, pdist_metric)
                self.Z_row_ = linkage(
                    D_row,
                    method=hclust_method,
                    optimal_ordering=optimal_leaf_ordering)
                self.order_row_ = leaves_list(self.Z_row_)

            if not col_cluster:
                self.order_col_ = list(range(X.shape[1]))
            else:
                D_col = pdist(X.transpose(), pdist_metric)
                self.Z_col_ = linkage(
                    D_col,
                    method=hclust_method,
                    optimal_ordering=optimal_leaf_ordering)
                self.order_col_ = leaves_list(self.Z_col_)

    def transform(self, X):
        """
        Return that transformed output with using the result of fit().

        Parameters
        ----------
        X : array-like, shape (n_rows, n_columns)
            A matrix representing a heatmap.
        Returns
        -------
        X_new : array, shape (n_rows, n_columns)
            Reordered X based on order_row_ and order_col_ obtained from fit().
        """
        if type(X) is not np.ndarray:
            print("input data needs to be a matrix")
        else:
            if len(self.order_row_) != X.shape[0] or len(
                    self.order_col_) != X.shape[1]:
                print("matrix shape and fitted order's shape is different")
            else:
                return (X[self.order_row_, :])[:, self.order_col_]

    def fit_transform(self,
                      X,
                      hclust_method='complete',
                      optimal_leaf_ordering=True,
                      pdist_metric='cosine',
                      row_cluster=True,
                      col_cluster=True):
        """
        Fit a heatmap X into a clustered, reordered space and return that
        transformed output.

        Parameters
        ----------
        X : array-like, shape (n_rows, n_columns)
            A matrix representing a heatmap.
        hclust_method : str, optional, (default='complete')
            The linkage algorithm to use.
            See the Linkage Methods section in
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html.
        optimal_leaf_ordering: bool, optional, (default=True)
            If True, the linkage matrix will be reordered so that the distance
            between successive leaves is minimal. This results in a more
            intuitive tree structure when the data are visualized. defaults to
            False, because this algorithm can be slow, particularly on large
            datasets. See also the optimal_leaf_ordering function in scipy.
        pdist_metric: str or function, optional, (default='cosine')
            The distance metric to use in the case that X is a collection of
            observation vectors; ignored otherwise. See the pdist function in
            scipy for a list of valid distance metrics. A custom distance
            function can also be used.
        row_cluster: bool, optional, (default=True)
            If True, apply hclust to X's rows. Otherwise, order_row_ will be
            the order from 0 to (n_rows - 1).
        col_cluster: bool, optional, (default=True)
            If True, apply hclust to X's columns. Otherwise, order_col_ will be
            the order from 0 to (n_columns - 1).
        Returns
        -------
        X_new : array, shape (n_rows, n_columns)
            Reordered X based on order_row_ and order_col_ obtained from fit().
        """
        self.fit(
            X,
            hclust_method=hclust_method,
            optimal_leaf_ordering=optimal_leaf_ordering,
            pdist_metric=pdist_metric,
            row_cluster=row_cluster,
            col_cluster=col_cluster)
        return self.transform(X)

    def get_row_labels(self, n_clusters):
        """
        Get cluster labels for rows with an indicated number of clusters.
        This uses the result from fit(). Therefore, use fit() before this.

        Parameters
        ----------
        n_clusters: int
            n_clusters is the number of clusters you want to get.
        Returns
        -------
        fcluster : ndarray
            An array of length n_rows. T[i] is the flat cluster number to which
            original observation i belongs.
        """
        if self.Z_row_ is not None:
            # fcluster index starts from 1
            return fcluster(self.Z_row_, n_clusters, criterion='maxclust') - 1

    def get_col_labels(self, n_clusters):
        """
        Get cluster labels for columns with an indicated number of clusters.
        This uses the result from fit(). Therefore, use fit() before this.

        Parameters
        ----------
        n_clusters: int
            n_clusters is the number of clusters you want to get.
        Returns
        -------
        fcluster : ndarray
            An array of length n_columns. T[i] is the flat cluster number to
            which original observation i belongs.
        """
        if self.Z_col_ is not None:
            # fcluster index starts from 1
            return fcluster(self.Z_col_, n_clusters, criterion='maxclust') - 1

    def aggregate_rows(self,
                       X,
                       length,
                       agg_method='abs_max',
                       rep_row_method='abs_max'):
        """
        Aggregate a heatmap X's rows based on the fit result.
        This uses the result from fit(). Therefore, use fit() before this.

        Parameters
        ----------
        X: array-like, shape (n_rows, n_columns)
            A matrix representing a heatmap.
        length: int
            A length of rows you want to get after aggregation.
        agg_method: str, optional, (default='abs_max')
            A method to calculate a value for each aggregated cell. 'abs_max'
            or 'mean' is available. If 'abs_max', with in the aggregated cells,
            the value whose absolute value is maximum will be taken. If 'mean',
            the mean value of the aggregated cells will be taken.
        rep_row_method: str, optional, (default='abs_max')
            A method to decide a representative row index for each aggregated
            cell. Only 'abs_max' is available now. If 'abs_max', with in the
            aggregated cells, the row index whose value has the maximum
            absolute value with in the aggregated rows will be taken.
        Returns
        -------
        X_aggregated : array, shape (length, n_columns)
            Aggregated X with # of rows = length.
        label_to_rows: List, a size is length
            Original X's row indices for each aggregated label (corresponding
            to the X_aggregated's row index)
        label_to_rep_row: List, a size is length
            Original X's representative row index for each aggregated label
            (corresponding to the X_aggregated's row index). Only one idex for
            each.
        """
        nrow, ncol = X.shape
        label_to_rows = [[] for i in range(length)]
        row_labels = self.get_row_labels(length)

        # perform aggregation
        agg_mat = np.zeros((length, ncol))
        if agg_method == 'mean':
            count_mat = np.zeros((length, ncol))
            for i in range(nrow):
                label = row_labels[i]
                label_to_rows[label].append(i)
                for j in range(ncol):
                    agg_mat[label, j] += X[i, j]
                    count_mat[label, j] += 1

            for i in range(length):
                for j in range(ncol):
                    count = count_mat[i, j]
                    if (count != 0):
                        agg_mat[i, j] /= count
        elif agg_method == 'abs_max':
            for i in range(nrow):
                label = row_labels[i]
                label_to_rows[label].append(i)
                for j in range(ncol):
                    val_x = X[i, j]
                    val_agg_mat = agg_mat[label, j]
                    if abs(val_x) > abs(val_agg_mat):
                        agg_mat[label, j] = X[i, j]

        # get representative row for each cluster
        label_to_rep_row = [None for i in range(length)]
        if rep_row_method == 'abs_max':
            for i, rows in enumerate(label_to_rows):
                current_abs_max = 0.0
                for j, row in enumerate(rows):
                    if j == 0:
                        label_to_rep_row[i] = row
                    abs_max_val = max(abs(X[row, ]))
                    if abs_max_val > current_abs_max:
                        current_abs_max = abs_max_val
                        label_to_rep_row[i] = row

        agg_mat = agg_mat[:, self.order_col_]

        return agg_mat, label_to_rows, label_to_rep_row
