import numpy as np
import  matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap

def set_plot_style():
    plt.axis([-2,2,-2,2])
    plt.xlabel('x1')
    plt.ylabel('x2')

def plot_data(meshgrid_data, meshgrid_data_y, X, y, weights, simple_model,
             target_instance, used_features, to_save=False, file_name="plot_name", plot_name=""):

    gray_cmap=LinearSegmentedColormap.from_list('gy',[(.3,.3,.3),(.8,.8,.8)], N=2)

    y_proba_em = simple_model.predict(meshgrid_data[:, used_features])
    points_of_boundary = np.where((y_proba_em<=0.51)&(y_proba_em>=0.49))
    # points_of_boundary = np.where(y_proba_em>0.5)
    # point1 = XX[points_of_boundary[0]]
    # point2 = XX[points_of_boundary[-1]]
    x_values = meshgrid_data[points_of_boundary, 0]
    y_values = meshgrid_data[points_of_boundary, 1]

    plt.figure(figsize=(5, 5))

    plt.scatter(meshgrid_data[:,0],meshgrid_data[:,1], c=meshgrid_data_y, cmap=gray_cmap) 
    # plt.scatter(X1[:,0],X1[:,1],s=10,c= W1,cmap="RdYlGn")
    plt.scatter(X[y==0,0],X[y==0,1],c=weights[y==0],cmap="RdYlGn",marker="_",s=80)
    plt.scatter(X[y==1,0],X[y==1,1],c=weights[y==1],cmap="RdYlGn",marker="+",s=80)
    # plotting the line
    # plt.scatter(X1[dist_to_model<=0.03,0],X1[dist_to_model<=0.03,1],c='blue', marker="*",s=80)
    plt.scatter(target_instance[0],target_instance[1], c="orangered", marker="o", s=70, edgecolor='black',linewidth=.8)
    # plt.plot(x_values[0], y_values[0], 'bo', linestyle="--")
    
    # plt.plot([x_values[0][0], x_values[0][-1]],
    #            [y_values[0][0], y_values[0][-1]], 'bo', linestyle="--")
    plt.axis([-2,2,-2,2])
    plt.title(plot_name)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.xticks([], [])
    plt.yticks([], [])

    if to_save:
        plt.savefig(f'/Users/benyaminghn/Researches/XAI/LIME/LIME_Codes/lime/experiments/visualization_plots/plt_{file_name}.jpg', dpi=300)

    plt.show()

def data_inverse(data_row,
                num_samples,
                sampling_method):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model
            sampling_method: 'gaussian' or 'lhs'

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
        is_sparse = sp.sparse.issparse(data_row)
        if is_sparse:
            num_cols = data_row.shape[1]
            data = sp.sparse.csr_matrix((num_samples, num_cols), dtype=data_row.dtype)
        else:
            num_cols = data_row.shape[0]
            data = np.zeros((num_samples, num_cols))
        categorical_features = range(num_cols)
        if self.discretizer is None:
            instance_sample = data_row
            scale = self.scaler.scale_
            mean = self.scaler.mean_
            if is_sparse:
                # Perturb only the non-zero values
                non_zero_indexes = data_row.nonzero()[1]
                num_cols = len(non_zero_indexes)
                instance_sample = data_row[:, non_zero_indexes]
                scale = scale[non_zero_indexes]
                mean = mean[non_zero_indexes]

            if sampling_method == 'gaussian':
                data = self.random_state.normal(0, 1, num_samples * num_cols
                                                ).reshape(num_samples, num_cols)
                data = np.array(data)
            elif sampling_method == 'lhs':
                data = lhs(num_cols, samples=num_samples
                           ).reshape(num_samples, num_cols)
                means = np.zeros(num_cols)
                stdvs = np.array([1]*num_cols)
                for i in range(num_cols):
                    data[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(data[:, i])
                data = np.array(data)
            else:
                warnings.warn('''Invalid input for sampling_method.
                                 Defaulting to Gaussian sampling.''', UserWarning)
                data = self.random_state.normal(0, 1, num_samples * num_cols
                                                ).reshape(num_samples, num_cols)
                data = np.array(data)

            if self.sample_around_instance:
                data = data * scale + instance_sample
            else:
                data = data * scale + mean
            if is_sparse:
                if num_cols == 0:
                    data = sp.sparse.csr_matrix((num_samples,
                                                 data_row.shape[1]),
                                                dtype=data_row.dtype)
                else:
                    indexes = np.tile(non_zero_indexes, num_samples)
                    indptr = np.array(
                        range(0, len(non_zero_indexes) * (num_samples + 1),
                              len(non_zero_indexes)))
                    data_1d_shape = data.shape[0] * data.shape[1]
                    data_1d = data.reshape(data_1d_shape)
                    data = sp.sparse.csr_matrix(
                        (data_1d, indexes, indptr),
                        shape=(num_samples, data_row.shape[1]))
            categorical_features = self.categorical_features
            first_row = data_row
        else:
            first_row = self.discretizer.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=num_samples,
                                                      replace=True, p=freqs)
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        if self.discretizer is not None:
            inverse[1:] = self.discretizer.undiscretize(inverse[1:])
        inverse[0] = data_row
        return data, inverse
