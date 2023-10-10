from sklearn.metrics import auc as _auc

def true_positive_rate(true_positives: int, false_negatives: int) -> float:
    if true_positives + false_negatives == 0:
        return -1
    return true_positives / (true_positives + false_negatives)

def false_positive_rate(false_positives: int, true_negatives: int) -> float:
    if false_positives + true_negatives == 0:
        return -1
    return false_positives / (false_positives + true_negatives)

def precision(true_positives: int, false_positives: int) -> float:
    if true_positives + false_positives == 0:
        return -1
    return true_positives / (true_positives + false_positives)

def recall(true_positives: int, false_negatives: int) -> float:
    if true_positives + false_negatives == 0:
        return -1
    return true_positives / (true_positives + false_negatives)

def f1(precision: float, recall: float):
    if precision == -1 or recall == -1 or precision == 0 or recall == 0:
        return -1
    return 2 * (precision * recall) / (precision + recall)

def auc_sort(x, y):
    def _unique_x(it):
        seen = set()
        for z in it:
            if z[0] not in seen:
                seen.add(z[0])
                yield z
    assert len(x) == len(y)
    zipped = zip(x, y)
    zipped = filter(lambda z: z[0] >= 0.0 and z[1] >= 0.0, zipped)
    zipped = _unique_x(zipped)
    zipped = sorted(zipped, key=lambda z: z[0])
    return [z[0] for z in zipped], [z[1] for z in zipped]

def auc(x, y):
    x, y = auc_sort(x, y)
    if len(x) < 2:
        return 0.0
    return _auc(x=x, y=y)

def auc_roc(false_positive_rates, true_positive_rates):
    return auc(x=false_positive_rates, y=true_positive_rates)

def auc_prc(true_positive_rates, precisions):
    return auc(x=true_positive_rates, y=precisions)
