import numpy as np
from .stability_metrics import *


class Evaluator():
    pass

def check_stability(explainer,
                    target_instance,
                    predict_fn,
                    labels=(1,),
                    top_labels=None,
                    num_features=10,
                    num_samples=5000,
                    distance_metric='euclidean',
                    model_regressor=None,
                    n_calls=10,
                    index_verbose=False,
                    verbose=False,
                    # categorical_features_ = [],
                    # feature_values_ = {},
                    # feature_frequencies_ = {},
                    # model=model,
                    # feature_selection=feature_selection,
                    feature_names=[],
                    # kernel='used_model_distance',
                    # sample_around_instance=False,
                    # discretizer=None,
                    # sampling_method='gaussian',
                    random_state=None,
                    ):

    """
    Method to calculate stability indices for a trained LIME instance.
    The stability indices are relative to the particular data point we are explaining.
    The stability indices are described in the paper:
    "Statistical stability indices for LIME: obtaining reliable explanations for Machine Learning models".
    It can be found in the ArXiv online repository: https://arxiv.org/abs/2001.11757
    The paper is currently under review in the Journal of Operational Research Society (JORS)
    Args:
        data_row: 1d numpy array or scipy.sparse matrix, corresponding to a row
        predict_fn: prediction function. For classifiers, this should be a
            function that takes a numpy array and outputs prediction
            probabilities. For regressors, this takes a numpy array and
            returns the predictions. For ScikitClassifiers, this is
            `classifier.predict_proba()`. For ScikitRegressors, this
            is `regressor.predict()`. The prediction function needs to work
            on multiple feature vectors (the vectors randomly perturbed
            from the data_row).
        explainer: explainer object
        labels: iterable with labels to be explained.
        top_labels: if not None, ignore labels and produce explanations for
            the K labels with highest prediction probabilities, where K is
            this parameter.
        num_features: maximum number of features present in explanation
        num_samples: size of the neighborhood to learn the linear model
        distance_metric: the distance metric to use for weights.
        model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase.
            If not Ridge Regression, the stability indices may not be calculated
            and the method will raise a LocalModelError.
        n_calls: number of repeated Lime calls to perform.
            High number of calls slows down the execution,
            however a meaningful comparison takes place only when
            there is a reasonable number of calls to compare.
        index_verbose: Controls for the verbosity at the stability indices level,
            when set to True gives information about partial values related to stability.
        verbose: Controls for the verbosity at the LocalModel level,
            when set to True, gives information about the repeated calls of WRR.
    Returns:
        csi: index to evaluate the stability of the coefficients of each variable across
        different Lime explanations obtained from the repeated n_calls.
        Ranges from 0 to 100.
        vsi: index to evaluate whether the variables retrieved in different Lime explanations
            are the same. Ranges from 0 to 100.
    """

    # Override verbosity in the LimeBaseOvr instance
    # previous_verbose = self.base.verbose
    # self.base.verbose = verbose

    explainer_confidence_intervals = []
    pred_scores = []
    mse_scores = []
    local_preds = []
    original_preds = []
    simple_models = []
    generated_data = []
    features_set = []
    easy_model_coefs = []

    for i in range(n_calls):
        # print(i)

        # (easy_model, used_features, prediction_score, mse_, local_pred, weights,
        # data_inverse, scaled_data, new_y_proba) 

        exp = explainer.explain_instance(data_row=target_instance,
                                        predict_fn=predict_fn,
                                        labels=labels,
                                        top_labels=top_labels,
                                        num_features=num_features,
                                        num_samples=num_samples,
                                        distance_metric=distance_metric,
                                        model_regressor=model_regressor,
                                        sampling_method='gaussian')
        
        assert len(labels) == 1, "Only one label is supported for stability evaluation"

        (easy_model, weights, used_features, X, y) = exp.get_stability_params()

        for label in labels:
            easy_model = easy_model[label]
            weights = weights[label]
            used_features = used_features[label]
            X = X
            y = y[:, label]
        # print(weights[0].shape, weights[1].shape)
        # # print(easy_model[0].shape, easy_model[1].shape)
        # print(prediction_score[0].shape, prediction_score[1].shape)
        # print(local_pred[0].shape, local_pred[1].shape)
        # print('lime_data_inverse.shape', data_inverse[0].shape)
        # print('our_data_inverse.shape', data_inverse[1].shape)
        # print('new_y_proba[0].shape', new_y_proba[0].shape)
        # print('new_y_proba[1].shape',new_y_proba[1].shape)
        # print(weights[0].shape)

        # data_inverse, scaled_data, new_y_proba = frags
        # The first time check if the local model is WRR, otherwise raise Exception
        if not i:
            if easy_model.alpha is None:
                raise LocalModelError("""Lime Local Model is not a Weighted Ridge Regression (WRR),
                Stability indices may not be computed: the formula is model specific""")

        easy_model_coefs.append(easy_model.coef_)

        explainer_confidence_intervals.append(confidence_intervals(X=X, true_labels=y,
                                              weights=weights, alpha=easy_model.alpha,
                                              easy_model=easy_model, feature_names=feature_names,
                                              used_features=used_features))

        # print(lime_confidence_intervals[0])
        # print(our_confidence_intervals[0])

    csi, vsi = compare_confints(confidence_intervals=explainer_confidence_intervals,
                                index_verbose=index_verbose)
      
    #   pred_scores.append(prediction_score)
    #   mse_scores.append(mse_)
    #   local_preds.append(local_pred)
    #   original_preds.append(new_y_proba)
    #   simple_models.append(easy_model)
    #   generated_data.append(data_inverse)
    #   features_set.append(used_features)



  # Set back the original verbosity
  # self.base.verbose = previous_verbose

    return csi, vsi, easy_model_coefs #, pred_scores, mse_scores, local_preds, original_preds, simple_models, generated_data, features_set


def check_accuracy(explainer,
                    target_instance,
                    predict_fn,
                    labels=(1,),
                    top_labels=None,
                    num_features=10,
                    num_samples=5000,
                    distance_metric='euclidean',
                    model_regressor=None,
                    n_calls=10,
                    index_verbose=False,
                    verbose=False,
                    # categorical_features_ = [],
                    # feature_values_ = {},
                    # feature_frequencies_ = {},
                    # model=model,
                    # feature_selection=feature_selection,
                    feature_names=[],
                    # kernel='used_model_distance',
                    # sample_around_instance=False,
                    # discretizer=None,
                    # sampling_method='gaussian',
                    random_state=None
                    ):
    """
    Compares the accuracy of the Lime explanation with the original model.
    Args:
        explainer: Explanation instance.
        target_instance: data row to explain.
        predict_fn: prediction function of the model.
        labels: labels to be explained.
        top_labels: if not None, labels are considered in explanation.
        num_features: maximum number of features present in explanation.
        num_samples: number of samples to explain.
        distance_metric: metric used to compute the distance between instances in the
            original data.
        model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase.
            If not Ridge Regression, the stability indices may not be calculated
            and the method will raise a LocalModelError.
        n_calls: number of repeated Lime calls to perform.
            High number of calls slows down the execution,
            however a meaningful comparison takes place only when
            there is a reasonable number of calls to compare.
        index_verbose: Controls for the verbosity at the stability indices level,
            when set to True gives information about partial values related to stability.
        verbose: Controls for the verbosity at the LocalModel level,
            when set to True, gives information about the repeated calls of WRR.
    Returns:
        accuracy: accuracy of the Lime explanation with the original model.
    """

    # Override verbosity in the LimeBaseOvr instance
    # previous_verbose = self.base.verbose
    # self.base.verbose = verbose

    explainer_accuracies = []
    pred_scores = []
    mse_scores = []
    local_preds = []
    original_preds = []
    simple_models = []
    generated_data = []
    features_set = []
    easy_model_coefs = []

    for i in range(n_calls):
        # print(i)
        exp = explainer.explain_instance(data_row=target_instance,
                                        predict_fn=predict_fn,
                                        labels=labels,
                                        top_labels=top_labels,
                                        num_features=num_features,
                                        num_samples=num_samples,
                                        distance_metric=distance_metric,
                                        model_regressor=model_regressor,
                                        sampling_method='gaussian')

        
        (easy_model, X, y) = exp.easy_model, exp.X, exp.y


        