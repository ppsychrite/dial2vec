import numpy as np
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from dataclasses import dataclass, fields
from prettytable import PrettyTable
from time import time
from sklearn.cluster import KMeans


@dataclass
class EvaluationResult(OrderedDict):
    RI: float = 0.          # Adjusted Rand Index
    NMI: float = 0.         # Normalized Mutual Informatio
    acc: float = 0.         # Clustering Accuracy
    purity: float = 0.      # Clutering Purity
    SR: float = 0.          # Semantic Relatedness
    MRR: float = 0.         # 平均倒数排名（Mean Reciprocal Rank, MRR）
    MAP: float = 0.         # Mean Average Precision
    all_mean: float = 0.
    alignment: float = 0.
    adjusted_alignment: float = 0.
    uniformity: float = 0.

    def __post_init__(self):
        self.positive_metrics = ['RI', 'NMI', 'acc', 'purity', 'SR', 'MRR', 'MAP']  # 越大越好的metrics
        self.negative_metrics = []   # 越小越好的metrics
        self.not_metrics = ['all_mean', 'alignment', 'adjusted_alignment', 'uniformity']  # 其它不是metric的属性

        self.all_mean = 0.
        self.mean()
        class_fields = fields(self)
        for field in class_fields:
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v

    def __lt__(self, other):
        return self.purity < other.purity

    def mean(self):
        all_values = []
        for key, value in self.__dict__.items():
            if key in self.positive_metrics:
                all_values.append(value)
            elif key in self.negative_metrics:
                all_values.append(-value)
            else:
                pass

        self.all_mean = sum(all_values) / (len(all_values))     # ignore the 'all_mean' when averaging
        return self.all_mean

    def update(self, new_result):
        """
        :param new_result: EvaluationResult
        :return:
        """
        self.RI = new_result.RI
        self.NMI = new_result.NMI
        self.acc = new_result.acc
        self.purity = new_result.purity
        self.SR = new_result.SR
        self.MRR = new_result.MRR
        self.MAP = new_result.MAP

    def show(self, logger=None, note=None):
        if logger is not None:
            logger.info("\nclustering_task [%s]: RI: %s NMI: %s Acc: %s Purity: %s" % (note, self.RI, self.NMI, self.acc, self.purity))
            logger.info("\nSemantic Relatedness [%s]: SR: %s" % (note, self.SR))
            logger.info("\nSession Retrieval [%s]: MRR: %s MAP: %s" % (note, self.MRR, self.MAP))
            logger.info("\nRepresentation_Evaluation [%s]: Alignment: %.6f Alignment (adjusted): %.6f Uniformity: %.6f" % (note, self.alignment, self.adjusted_alignment, self.uniformity))

            tb = PrettyTable()
            tb.field_names = ['', 'RI', 'NMI', 'Acc', 'Purity', 'SR', 'MRR', 'MAP', 'Alignment', 'Adjusted Alignment', 'Uniformity']
            tb.add_row(['Metrics'] + \
                       ['%.2f' % (v * 100) for v in [self.RI, self.NMI, self.acc, self.purity, self.SR, self.MRR, self.MAP]] + \
                       ['%.2f' % v for v in [self.alignment, self.adjusted_alignment, self.uniformity]])
            logger.info('\n' + tb.__str__())


def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score

        Reference: https://blog.csdn.net/weixin_45727931/article/details/111921581
    """
    # matrix which will hold the majority-voted labels
    y_true = y_true.astype(np.int64)
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def get_accuracy(y_true, y_pred):
    """
    计算聚类的准确率
    """
    y_true = y_true.astype(np.int64)

    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def feature_based_evaluation_at_once(features, labels, n_average=1, tasks=None, dtype='float64'): 
    """
    Evaluate all metrics with features
    :param features:                        numpy.array
    :param labels:                          list
    :param n_average:
    :param tasks:
    :param dtype:
    :return:
    """
    labels = np.array(labels).astype(int)
    features = np.array(features).astype(dtype) if features is not None else None

    # n_classes
    label_set = set()
    for s in labels:
        label_set.add(s)

    # initialize
    RI, NMI, acc, purity = 0., 0., 0., 0.
    clustering_time, RI_time, NMI_time, acc_time, purity_time = 0., 0., 0., 0., 0.
    SR = 0.
    MRR, MAP = 0., 0.
    alignment, adjusted_alignment, uniformity = 0., 0., 0.

    # KMeans
    if 'clustering' in tasks:
        for _ in range(n_average):
            # clustering
            pre = time()
            clf = KMeans(n_clusters=len(label_set), max_iter=500, tol=1e-5)
            clf.fit(features)
            y_pred = clf.predict(features)
            clustering_time += (time() - pre) / n_average

            ## RI
            pre = time()
            RI += adjusted_rand_score(labels, y_pred) / n_average
            RI_time += (time() - pre) / n_average

            ## NMI
            pre = time()
            NMI += normalized_mutual_info_score(labels, y_pred) / n_average
            NMI_time += (time() - pre) / n_average

            ## acc
            pre = time()
            acc += get_accuracy(labels, y_pred) / n_average
            acc_time += (time() - pre) / n_average

            ## purity
            pre = time()
            purity += purity_score(labels, y_pred) / n_average
            purity_time += (time() - pre) / n_average

    return EvaluationResult(
        RI=RI,
        NMI=NMI,
        acc=acc,
        purity=purity,
        SR=SR,
        MRR=MRR,
        MAP=MAP,
        alignment=alignment,
        adjusted_alignment=adjusted_alignment,
        uniformity=uniformity
    )