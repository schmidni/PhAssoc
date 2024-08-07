import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import pair_confusion_matrix as PCM


class ClusterStatistics:
    def __init__(self):
        self._ari = []
        self._pcm = []
        self._perc_eq = []

    def add(self, labels, labels_pred, n_true=None, n_pred=None):
        self._ari.append(ARI(labels, labels_pred))
        self._pcm.append(PCM(labels, labels_pred))
        if n_true and n_pred:
            self._perc_eq.append(abs(n_true-n_pred) / n_true)

    def ari(self):
        return np.round(np.mean(self._ari), 3)

    def pcm(self):
        return np.round(np.mean(self._pcm, axis=0), 3)

    def precision(self):
        return np.round(self.pcm()[1, 1] /
                        (self.pcm()[0, 1] + self.pcm()[1, 1]), 3)

    def recall(self):
        return np.round(self.pcm()[1, 1] /
                        (self.pcm()[1, 0] + self.pcm()[1, 1]), 3)

    def accuracy(self):
        return np.round((self.pcm()[0, 0] +
                         self.pcm()[1, 1]) / self.pcm().sum(), 3)

    def perc_eq(self):
        if not self._perc_eq:
            return None
        # TODO: Better metric
        return np.round(100*(1-np.mean(self._perc_eq)), 3)


color_iter = itertools.cycle(
    ["navy", "c", "cornflowerblue", "gold", "orange", "green",
     "lime", "red", "purple", "blue", "pink", "brown", "black", "gray",
     "magenta", "cyan", "olive", "maroon", "darkslategray", "darkkhaki"])


def plot_arrivals(arrivals, cat, cat_pred, labels, labels_pred):
    fig, ax = plt.subplots(2, sharex=True, figsize=(7, 5))
    # fig.suptitle('Vertically stacked subplots')

    # prediction
    for idx in range(len(np.unique(labels_pred))):
        ax[1].scatter(arrivals.loc[labels_pred == idx, 'time'] / 1000,
                      arrivals.loc[labels_pred == idx, 'dx'],
                      color=color_iter.__next__(), s=80
                      )

    ax[1].scatter(arrivals.loc[labels_pred == -1, 'time'] / 1000,
                  arrivals.loc[labels_pred == -1, 'dx'],
                  color='black', s=20)
    ax[1].scatter(cat_pred['time'] / 1000, cat_pred['dx'],
                  color='darkorange', marker='x')

    # truth
    for idx in range(len(np.unique(labels))):
        ax[0].scatter(arrivals.loc[labels == idx, 'time'] / 1000,
                      arrivals.loc[labels == idx, 'dx'],
                      color=color_iter.__next__(), s=80
                      )
    ax[0].scatter(cat['time'] / 1000, cat['dx'],
                  color='darkorange', marker='x')


def plot_clusters(X, Y_, means, covariances, x, y, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(
            zip(means, covariances, color_iter)):
        if covar.ndim < 2:
            covar = np.diag(covar)
        # use advanced indexing and broadcasting to select
        # the rows and columns corresponding to x and y
        covar = covar[np.ix_([x, y], [x, y])]
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, x], X[Y_ == i, y], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(
            (mean[x], mean[y]), v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title(title)
