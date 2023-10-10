import logging
import os
from os.path import join, abspath

import numpy as np
from scipy.stats import gamma

from utils.logging import init_logger

THRESHOLD_CONFIDENCE_INTERVALS = [0.68, 0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]

logger = logging.Logger("Thresholds")
init_logger(logger)

euclidean_distance = lambda a, b: np.sqrt(np.sum((a - b) ** 2))

def calculate_losses_predictions(labels, predictions, distance_metric=euclidean_distance):
    count = len(labels)
    distances = np.empty(shape=(count,))
    index = 0
    # Calculate distances
    for label, prediction in zip(labels, predictions):
        distance = distance_metric(label, prediction)
        distances.put(index, distance)
        index += 1
    return distances

def save_thresholds(losses, prefix, conf_intervals=THRESHOLD_CONFIDENCE_INTERVALS):

    losses_path = f'{prefix}.losses.csv'
    distribution_path = f'{prefix}.distribution.csv'
    thresholds_path = f'{prefix}.thresholds.csv'

    os.makedirs(abspath(join(thresholds_path, '..')), exist_ok=True)

    with open(losses_path, mode='w', encoding='utf-8', newline='\n') as fp:
        for loss in losses:
            fp.write(f'{loss}\n')

    logger.info("Fitting reconstruction error distribution of %s using Gamma distribution params" % prefix)
    shape, loc, scale = gamma.fit(losses, floc=0)

    with open(distribution_path, mode='w', encoding='utf-8', newline='\n') as fp:
        fp.write(f'shape,loc,scale\n')
        fp.write(f'{shape},{loc},{scale}\n')

    logger.info("Saving thresholds to %s" % thresholds_path)
    with open(thresholds_path, 'w', encoding='utf-8', newline='\n') as fp:
        fp.write(f'confidence,threshold\n')
        for confidence in conf_intervals:
            threshold = gamma.ppf(confidence, shape, loc=loc, scale=scale)
            fp.write(f'{confidence},{threshold}\n')
