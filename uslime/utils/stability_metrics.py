"""
This is a script containing bunch of utility functions,
helpful to build the stability indices for Lime
"""

import scipy as sp
import numpy as np
from itertools import combinations
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import WLS


# class LocalModelError(Exception):
#     """Custom Exception raised when the model is not Weighted Ridge Regression"""
#     pass


def compute_WLS_stdevs(X, Y, weights, alpha):
    """Function to calculate standard deviations of Weighted Ridge coefficients
    Args:
        X: dataset containing the explanatory variables
        Y: vector of the response variable
        weights: vector of weights (one for each tuple of the X dataset)
        alpha: regularization parameter
    Returns:
        stdevs_beta: list containing the standard deviations of the coefficients
        """

    # Build Weighted Regression (WLS) model
    X_enh = add_constant(X)
    # print('X.shape:', X.shape)
    # print(y.shape)
    # print(pd.DataFrame(X))
    # print(pd.DataFrame(X_enh))
    # print('X_enh.shape:', X_enh.shape)
    # print(weights.shape)
    wls_model = WLS(Y, X_enh, weights=weights)
    results = wls_model.fit()
    errors_wls_weighted = results.wresid

    # Estimate of the sigma squared quantity
    sigma2 = np.dot(errors_wls_weighted.T, errors_wls_weighted) / (X.shape[0] - X.shape[1])
    weights_matr = np.diag(weights)  # reformulate weights as diagonal matrix

    # Standard deviations of the coefficients
    partial_ = np.linalg.inv(np.linalg.multi_dot([X.T, weights_matr, X]) +
                             alpha * np.diag([1, ] * X.shape[1]))
    variances_beta_matrix = sigma2 * np.linalg.multi_dot(
        [partial_, X.T, weights_matr, X, partial_.T])
    variances_beta = np.diag(variances_beta_matrix)
    stdevs_beta = list(np.sqrt(variances_beta))

    return stdevs_beta


def refactor_confints_todict(means, st_devs, feat_names):
    """Refactor means and confidence intervals into a dictionary
    Args:
        means: list of the means of the WRR coefficients
        st_devs: list of the standard deviations of the WRR coefficients
        feat_names: list of feature names associated with the coefficients
    Returns:
        conf_int: dictionary,
            key = the feature name
            value = confidence interval for the feature (upper, lower bound)
    """

    conf_int = {}
    for name, mean, stdev in zip(feat_names, means, st_devs):
        conf_int[name] = [mean - 1.96 * stdev, mean + 1.96 * stdev]
    return conf_int


def compare_confints(confidence_intervals, index_verbose=False):
    """Function to compare confidence intervals obtained through different WRR,
        which are built with the same number of features (possibly different ones, but the same number).
        Core function of the package: calculates the two complementary indices CSI, VSI.
    Args:
        confidence_intervals: list of dictionaries,
            each dictionary is the output of the confint function.
        index_verbose: Controls for the verbosity at the stability indices level,
            when set to True gives information about partial values related to stability.
    Returns:
        csi: Coefficients stability index
        vsi: Variables stability index
    """

    n_features = len(confidence_intervals[0].keys())
    features_limes = []
    for conf_int in confidence_intervals:
        features_limes.append(conf_int.keys())
    unique_features = list(set([l for ll in features_limes for l in ll]))

    # Calculate CSI
    overlapping_tot = []
    for feat in unique_features:
        conf_int_feat = []
        for conf_int in confidence_intervals:
            if conf_int.get(feat):
                conf_int_feat.append(conf_int.get(feat))

        if len(conf_int_feat) < 2:
            pass
        else:
            overlapping = []
            for pair_intervals in combinations(conf_int_feat, 2):
                i1, i2 = pair_intervals
                is_overlap = True if (i1[0] < i2[1] and i2[0] < i1[1]) else False
                overlapping.append(is_overlap)
            frac_overlapping = round(sum(overlapping) / len(overlapping) * 100, 2)
            overlapping_tot.append(frac_overlapping)
            if index_verbose:
                print("""Percentage of overlapping confidence intervals, variable {}: {}%\n""".format(
                    feat, frac_overlapping))

    csi = round(np.mean(overlapping_tot), 2)
    # print('csi:', csi)
    
    # Calculate VSI
    same_vars = 0
    n_combs = 0
    for pair_vars in combinations(features_limes, 2):
        var1, var2 = pair_vars
        same_vars += len(set(var1) & set(var2))
        n_combs += 1
    # print('n_combs', n_combs)
    # print('n_features', n_features)
    if n_combs != 0:
      vsi = round(same_vars / (n_combs * n_features) * 100, 2)
    else: vsi = -1
    if index_verbose:
        print("""Percentage same variables across repeated LIME calls: {}%\n""".format(vsi))

    return csi, vsi

def confidence_intervals(X, true_labels, weights, alpha, easy_model, feature_names, used_features):
    """
    Method to calculate stability indices of an application of the LIME method
    to a particular unit of the dataset.
    The stability indices are described in the paper:
    "Statistical stability indices for LIME: obtaining reliable explanations for Machine Learning models"
    which can be found in the ArXiv online repository: https://arxiv.org/abs/2001.11757
    The paper is currently under review in the Journal of Operational Research Society (JORS)
    Returns:
        conf_int: list of two values (lower and upper bound of the confidence interval)
    """
    # print('X.shape', X.shape)
    # print('Y.shape', y.shape)
    stdevs_beta = compute_WLS_stdevs(X=X, Y=true_labels, weights=weights, alpha=alpha)

    beta_ridge = easy_model.coef_.tolist()

    feature_ids = used_features
    used_features = [feature_names[i] for i in feature_ids]

    conf_int = refactor_confints_todict(means=beta_ridge, st_devs=stdevs_beta, feat_names=used_features)

    return conf_int
    
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def jaccard_distance(usecase):
    sim = []
    for l in usecase:
        i_sim = []
        for j in usecase:
            i_sim.append(1 - jaccard_similarity(l, j))
        sim.append(i_sim)
    return sim


def standard_deviation(coefs_list, num_features):

    compare_std_list = []
    compare_mean_list = []

    for k, coefs in enumerate(coefs_list):
        # print(len(coefs))
        # print(coefs)
        compare_mean = []
        compare_std = []
        for i in range(num_features):
            # print([print(len(Mx)) for Mx in coefs])
            f1 = np.array([Mx[i] for Mx in coefs])
            # print(f'mean for f{i}:', np.mean(f1))
            # print(f'std for f{i}:', np.std(f1))
            compare_mean.append(np.mean(f1))
            compare_std.append(np.std(f1))


        compare_std_list.append([np.mean(np.array(compare_std))])
        compare_mean_list.append([np.mean(np.array(compare_mean))])

    compare_std_t = np.mean(np.array(compare_std), axis=0)
    compare_mean_t =  np.mean(np.array(compare_mean), axis=0)

    return compare_std_t, compare_mean_t