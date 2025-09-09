import os
import json
from copy import deepcopy

import pandas as pd
import sklearn.metrics as metrics
import numpy as np

def get_roc_aur(human_z, machine_z):
    assert len(human_z) == len(machine_z)

    baseline_z_scores = np.array(human_z)
    watermark_z_scores = np.array(machine_z)
    all_scores = np.concatenate([baseline_z_scores, watermark_z_scores])

    baseline_labels = np.zeros_like(baseline_z_scores)
    watermarked_labels = np.ones_like(watermark_z_scores)
    all_labels = np.concatenate([baseline_labels, watermarked_labels])

    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc, fpr, tpr, thresholds

def get_tpr(fpr, tpr, error_rate):
    assert len(fpr) == len(tpr)

    value = None
    for f, t in zip(fpr, tpr):
        if f <= error_rate:
            value = t
        else:
            assert value is not None
            return value
        
    assert value == 1.0
    return value