"""
Functions for explaining classifiers that use tabular data (matrices).
"""
import collections
import copy
from functools import partial
import json
import warnings

import numpy as np
import scipy as sp
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from pyDOE2 import lhs
from scipy.stats.distributions import norm

from lime.discretize import QuartileDiscretizer
from lime.discretize import DecileDiscretizer
from lime.discretize import EntropyDiscretizer
from lime.discretize import BaseDiscretizer
from lime.discretize import StatsDiscretizer

from sklearn.metrics import pairwise_distances
from . import explanation
from . import lime_base


from lime_codes.lime_tabular import LimeTabularExplainer, TableDomainMapper


class USLime(LimeTabularExplainer):
    """Override of the original LimeTabularExplainer class in lime_tabular
        A new method has been implemented: (our method)
        This method samples filtered data from purturbed data."""

    def __init__(self,
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 discretize_continuous=True,
                 discretizer='quartile',
                 # ----------------------
                 sample_around_instance=True,
                 # ----------------------
                 random_state=None,
                 training_data_stats=None):
                 
        """Inherits from the original LimeTabularExplainer class"""

        super(USLime, self).__init__(training_data, mode, training_labels, feature_names,
                                                      categorical_features, categorical_names, kernel_width,
                                                      kernel, verbose, class_names, feature_selection,
                                                      discretize_continuous, discretizer, sample_around_instance,
                                                      random_state, training_data_stats)

    
    def get_sampling_data(self):
        return (self.first_step_sampling_data, 
                self.second_step_sampling_data,
                self.third_step_sampling_data)


    def explain_instance(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         model_regressor=None,
                         sampling_method='gaussian'):

        """
        Overrides the original method in LimeTabularExplainer.explain_instance()

        Original documentation of the method:

        Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

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
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
            sampling_method: Method to sample synthetic data. Defaults to Gaussian
                sampling. Can also use Latin Hypercube Sampling.

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
            # Preventative code: if sparse, convert to csr format if not in csr format already
            data_row = data_row.tocsr()
        
        # -----------------------------
        # sampling around instance
        data, inverse = self.data_inverse(data_row, 3*num_samples, sampling_method, desired_variance=.5)
        # -----------------------------

        if sp.sparse.issparse(data):
            # Note in sparse case we don't subtract mean since data would become dense
            scaled_data = data.multiply(self.scaler.scale_)
            # Multiplying with csr matrix can return a coo sparse matrix
            if not sp.sparse.isspmatrix_csr(scaled_data):
                scaled_data = scaled_data.tocsr()
        else:
            scaled_data = (data - self.scaler.mean_) / self.scaler.scale_


        target_instance = scaled_data[0]
        scaled_data = np.delete(scaled_data, (0), axis=0)
        self.first_step_sampling_data = inverse.copy()
        yss = predict_fn(inverse)
        self.first_step_yss = yss.copy()
        target_instance_proba = yss[0]
        yss = np.delete(yss, (0), axis=0)

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(yss.shape)))

        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        target_instance_label = int(np.round(yss[0,0]))
        new_y_pred = np.where(yss[:, target_instance_label]>=0.5, 1, 0)
        # calculatin the distance of the generated data to the model
        distance_to_model = [np.abs(0.5-y) for y in yss[:, 0]]  
        distance_to_model = np.array(distance_to_model, dtype='float64')

        kernel_width = np.sqrt(scaled_data.shape[1]) * .75
        to_model_weights = np.sqrt(np.exp(-((distance_to_model)**2)/(kernel_width**2)))

        w_idx_dataset = np.column_stack((np.arange(len(scaled_data)),
                                        new_y_pred,
                                        distance_to_model,
                                        to_model_weights))

        # sorting the new data by the distnaces to model
        sorted_w_idx_dataset = w_idx_dataset[np.argsort(w_idx_dataset[:, -1])][::-1, :]
        # print(sorted_w_idx_dataset)
        n_positive_class = sorted_w_idx_dataset[sorted_w_idx_dataset[:, 1]==1][:num_samples]
        n_negative_class = sorted_w_idx_dataset[sorted_w_idx_dataset[:, 1]==0][:num_samples]
        pn_classes = np.concatenate((n_positive_class, n_negative_class))
        # print('pn_classes.shape: ', pn_classes.shape)

        # new selected data
        new_data = scaled_data[pn_classes[:,0].astype(int)]
        self.second_step_sampling_data = new_data.copy()
        self.second_step_yss = yss[pn_classes[:,0].astype(int)].copy()
        # print('new_data.shape: ', new_data.shape)

        # finding the nearest selected data to the test data
        distance_metric = "euclidean"
        distance_to_target = pairwise_distances(new_data,
                                                data_row.reshape(1, -1),
                                                metric=distance_metric
                                                ).ravel()

        pn_classes_distances = np.column_stack((pn_classes, distance_to_target))
        sorted_w_idx_dataset = pn_classes_distances[np.argsort(pn_classes_distances[:, -1])]
        # print(sorted_w_idx_dataset[sorted_w_idx_dataset[:, 1]==1].shape)
        # print(sorted_w_idx_dataset[sorted_w_idx_dataset[:, 1]==0].shape)
        # if new_y_pred[0] == 1:
        n_positive_class_distance = sorted_w_idx_dataset[sorted_w_idx_dataset[:, 1]==1][:int((2/4)*num_samples)]
        n_negative_class_distance = sorted_w_idx_dataset[sorted_w_idx_dataset[:, 1]==0][:int((2/4)*num_samples)]
        # else:
        #     n_positive_class_distance = sorted_w_idx_dataset[sorted_w_idx_dataset[:, 1]==0][:int((2/4)*num_samples)]
        #     n_negative_class_distance = sorted_w_idx_dataset[sorted_w_idx_dataset[:, 1]==1][:int((2/4)*num_samples)]

        final_pn_classes = np.concatenate((n_positive_class_distance, n_negative_class_distance))
        
        last_filtered_data = scaled_data[final_pn_classes[:, 0].astype(int)]
        self.third_step_sampling_data = last_filtered_data.copy()
        filtered_yss = yss[final_pn_classes[:, 0].astype(int)]
        self.third_step_yss = filtered_yss.copy()

        # # calculating our weights
        # distances = np.array([np.abs(0.5-y) for y in filtered_yss[:, target_instance_label]])

        # test the lime distances
        distance_metric = "euclidean"
        distances = pairwise_distances(last_filtered_data,
                                        data_row.reshape(1, -1),
                                        metric=distance_metric
                                        ).ravel()
        
        kernel_width = np.sqrt(last_filtered_data.shape[1]) * .75
        # exp_kernel_width = .05
        our_weights = np.sqrt(np.exp(-(distances**2)/(kernel_width**2))) #Kernel function
        # print(last_distance_to_model)

        # print('datarow.shape: ', data_row.shape)
        # print('data_row: ', data_row)
        np.append(target_instance, last_filtered_data)
        np.append(target_instance_proba, filtered_yss)
        # print('first instance:', last_filtered_data[0])

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        if sp.sparse.issparse(data_row):
            values = self.convert_and_round(data_row.data)
            feature_indexes = data_row.indices
        else:
            values = self.convert_and_round(data_row)
            feature_indexes = None

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names,
                                          feature_indexes=feature_indexes)

        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)

        if self.mode == "classification":
            ret_exp.predict_proba = filtered_yss[0]
            if top_labels:
                labels = np.argsort(filtered_yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]

        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label],
             # ------------------------------ 
             # added just returned easy model, weights, and used_features
             ret_exp.easy_model[label],
             ret_exp.weights[label],
             ret_exp.used_features[label]
             # ------------------------------ 
             ) = self.base.explain_instance_with_data(
                    last_filtered_data,
                    filtered_yss,
                    distances, # our_weights
                    label,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        # ---------------------------------
        # add the information needed for checking stability to the explainer class
        ret_exp.X = last_filtered_data
        ret_exp.y = filtered_yss
        # ---------------------------------
        return ret_exp
