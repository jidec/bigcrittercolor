from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from bigcrittercolor.helpers import _bprint

def _findNClusters(values, minmaxn= (3,10), metric="ch", show=True, print_steps=True):
    _bprint(print_steps, "Number of clusters unspecified, started finding optimum by " + metric + "...")

    # create a vector of ns to try for knee assessment
    ns = np.arange(minmaxn[0], minmaxn[1] + 1)

    models = [GaussianMixture(n).fit(values) for n in ns]
    aics = [m.aic(values) for m in models]
    labels_list = [m.fit_predict(values) for m in models]

    ch_scores = [calinski_harabasz_score(values, l) for l in labels_list]
    db_scores = [davies_bouldin_score(values, l) for l in labels_list]
    sil_scores = [silhouette_score(values, l) for l in labels_list]

    if show:
        plt.plot(ns, ch_scores)
        plt.title('Calinski-Harabasz Scores')
        plt.xlabel('Number of clusters')
        plt.ylabel('CH score')
        plt.show()

        plt.plot(ns, db_scores)
        plt.title('Davies-Bouldin Scores')
        plt.xlabel('Number of clusters')
        plt.ylabel('DB score')
        plt.show()

        plt.plot(ns, sil_scores, label='Silhouette score')
        plt.title('Silhouette Scores')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette score')
        plt.show()

    if metric == "ch":
        n = ns[np.argmax(ch_scores)]
    elif metric == "db":
        n = ns[np.argmax(db_scores)]

    return n
