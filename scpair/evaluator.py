from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np;
import pandas as pd
from sklearn import preprocessing, metrics
import scipy

"""
evaluation.py provides evaluation functions that takes in predictions and output gene-wise correlation, peak-wise AUROC and AUPRnorm, as well as FCT (FOSCTTM) score
"""

def compute_pairwise_distances(x, y):
    """
    compute pairwise distance for x and y, used for FOSCTTM distance calculation
    """
    x = np.expand_dims(x, 2)
    y = np.expand_dims(y.T, 0)
    diff = np.sum(np.square(x - y), 1)
    return diff


def FOSCTTM(domain1, domain2, output_full=False):
    """
    [fraction of samples closer than the true match]
    return fraction of samples closer than true match (FOSCTTM/FCT score)
    av_fraction: average FOSCTTM
    sorted_fraction_1to2: FOSCTTM score for each cell query in domain 1
    sorted_fraction_2to1: FOSCTTM score for each cell query in domain 2
    """
    n = domain1.shape[0]
    distances_1to2 = compute_pairwise_distances(domain1, domain2)
    distances_2to1 = distances_1to2.T
    fraction_1to2 = []
    fraction_2to1 = []
    for i in range(n):
        fraction_1to2.append(np.sum(distances_1to2[i, i] > distances_1to2[i, :]) / (n - 1))
        fraction_2to1.append(np.sum(distances_2to1[i, i] > distances_2to1[i, :]) / (n - 1))
    av_fraction = (np.sum(fraction_2to1) / n + np.sum(fraction_1to2) / n) / 2
    if output_full:
        sorted_fraction_1to2 = np.sort(np.array(fraction_1to2))
        sorted_fraction_2to1 = np.sort(np.array(fraction_2to1))
        return sorted_fraction_1to2, sorted_fraction_2to1
    else:
        return av_fraction


def FOSFTTM(query, reference, output_full=False, sort=False):
    """
    fraction of samples farther than true match
    return fraction of samples farther than true match (FOSFTTM score)
    avg_fraction: average FOSFTTM
    domain1: query
    domain2: reference
    fraction: FOSFTTM score for each cell query in domain 1
    """
    n = reference.shape[0]
    distances= compute_pairwise_distances(query, reference)
    fraction = []
    for i in range(n):
        fraction.append(np.sum(distances[i, i] < distances[i, :]) / (n - 1))
    avg_fraction = np.sum(fraction) / n
    if output_full:
        if sort:
            sorted_fraction = np.sort(np.array(fraction))
            return sorted_fraction
        else:
            return fraction
    else:
        return avg_fraction



def plot_auroc_perpeak(matrix_true, matrix_pred):
    """
    return scRNA and scATAC projections on VAE embedding layers 
    """
    matrix_true = preprocessing.binarize(matrix_true)
    fpr, tpr, _thresholds = metrics.roc_curve(matrix_true.flatten(), matrix_pred.flatten())
    auc_flatten = metrics.auc(fpr, tpr)
    auprc = metrics.average_precision_score(matrix_true.flatten(), matrix_pred.flatten())
    pp = np.sum(matrix_true)
    pp = pp/(matrix_true.shape[0] * matrix_true.shape[1])
    auprc_norm_flatten = (auprc-pp)/(1-pp)
    auc_list = []
    auprc_list = []
    npos = []
    for i in range(matrix_true.shape[1]):
        pp = np.sum(matrix_true[:,i])
        npos.append(pp)
        if pp >= 1:
            fpr, tpr, _thresholds = metrics.roc_curve(matrix_true[:,i], matrix_pred[:,i])
            auc = metrics.auc(fpr, tpr)
            auprc = metrics.average_precision_score(matrix_true[:,i], matrix_pred[:,i])
            pp = pp/matrix_true.shape[0]
            auprc_norm = (auprc-pp)/(1-pp)
            auc_list.append(auc)
            auprc_list.append(auprc_norm)
        else:
            auc_list.append(np.nan)
            auprc_list.append(np.nan)  
    return np.array(auc_list), np.array(auprc_list), auc_flatten, auprc_norm_flatten, np.array(npos)


def plot_cor_pergene(x, y, logscale, normlib):
    """
    return pearson correlation coefficient for each gene: pearson_r_list
    flattened pearson correlation coefficient: pearson_r_flatten
    number of positive values in the true profile for each gene
    """
    assert x.shape == y.shape, f"Mismatched shapes: {x.shape} {y.shape}"
    x = np.asarray(x)
    y = np.asarray(y)
    if normlib == 'norm':
        ## compare with normalized true profile
        lib = x.sum(axis=1, keepdims=True)
        x = x / lib
    if logscale:
        x = np.log1p(x)
        y = np.log1p(y)
    pearson_r_flatten, pearson_p_flatten = scipy.stats.pearsonr(x.flatten(), y.flatten())
    pearson_r_list = []
    npos = []
    for i in range(x.shape[1]):
        npos.append(np.sum(x[:,i] > 0))
        if not np.all(x[:,i] == 0) and not np.all(y[:,i] == 0):
            pearson_r, pearson_p = scipy.stats.pearsonr(x[:,i], y[:,i])
            #spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
            pearson_r_list.append(pearson_r)
        else:
            pearson_r_list.append(np.nan)
    return np.array(pearson_r_list), pearson_r_flatten, np.array(npos)
