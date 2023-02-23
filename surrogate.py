import ConfigSpace
import numpy as np
import logging
import pandas as pd
import pyrfr.regression as reg
import pyrfr.util
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    NumericalHyperparameter,
    Constant,
    OrdinalHyperparameter,
)


class fANOVA_surrogate(object):
    def __init__(
        self,
        X,
        Y,
        config_space=None,
        n_trees=16,
        seed=None,
        bootstrapping=True,
        points_per_tree=None,
        max_features=None,
        min_samples_split=0,
        min_samples_leaf=0,
        max_depth=64,
        cutoffs=(-np.inf, np.inf),
    ):

        """
        Calculate and provide midpoints and sizes from the forest's 
        split values in order to get the marginals
        
        Parameters
        ------------
        X: matrix with the features, either a np.array or a pd.DataFrame (numerically encoded)
        
        Y: vector with the response values (numerically encoded)
        
        config_space : ConfigSpace instantiation
        
        n_trees: number of trees in the forest to be fit
        
        seed: seed for the forests randomness
        
        bootstrapping: whether or not to bootstrap the data for each tree
        
        points_per_tree: number of points used for each tree 
                        (only subsampling if bootstrapping is false)
        
        max_features: number of features to be used at each split, default is 70%
        
        min_samples_split: minimum number of samples required to attempt to split 
        
        min_samples_leaf: minimum number of samples required in a leaf
        
        max_depth: maximal depth of each tree in the forest
        
        cutoffs: tuple of (lower, upper), all values outside this range will be
                 mapped to either the lower or the upper bound. (See:
                 "Generalized Functional ANOVA Diagnostics for High Dimensional
                 Functions of Dependent Variables" by Hooker.)
        """

        pcs = [(np.nan, np.nan)] * X.shape[1]

        # Convert pd.DataFrame to np.array
        if isinstance(X, pd.DataFrame):
            if config_space is not None:
                # Check if column names match parameter names
                bad_input = set(X.columns) - set(
                    config_space.get_hyperparameter_names()
                )
                if len(bad_input) != 0:
                    raise ValueError(
                        "Could not identify parameters %s from pandas dataframes"
                        % str(bad_input)
                    )
                # Reorder dataframe if necessary
                X = X[config_space.get_hyperparameter_names()]
            X = X.to_numpy()

        # if no ConfigSpace is specified, let's build one with all continuous variables
        if config_space is None:
            # if no info is given, use min and max values of each variable as bounds
            config_space = ConfigSpace.ConfigurationSpace()
            for i, (mn, mx) in enumerate(zip(np.min(X, axis=0), np.max(X, axis=0))):
                config_space.add_hyperparameter(
                    UniformFloatHyperparameter("x_%03i" % i, mn, mx)
                )

        self.percentiles = np.percentile(Y, range(0, 100))
        self.cs = config_space
        self.cs_params = self.cs.get_hyperparameters()
        self.n_dims = len(self.cs_params)
        self.n_trees = n_trees
        self._dict = False

        # at this point we have a valid ConfigSpace object
        # check if param number is correct etc:
        if X.shape[1] != len(self.cs_params):
            raise RuntimeError(
                "Number of parameters in ConfigSpace object does not match input X"
            )
        for i in range(len(self.cs_params)):
            if isinstance(self.cs_params[i], NumericalHyperparameter):
                if (np.max(X[:, i]) > self.cs_params[i].upper) or (
                    np.min(X[:, i]) < self.cs_params[i].lower
                ):
                    raise RuntimeError(
                        "Some sample values from X are not in the given interval"
                    )
            elif isinstance(self.cs_params[i], CategoricalHyperparameter):
                unique_vals = set(X[:, i])
                if len(unique_vals) > len(self.cs_params[i].choices):
                    raise RuntimeError(
                        "There are some categoricals missing in the ConfigSpace specification for "
                        "hyperparameter %s:" % self.cs_params[i].name
                    )
            elif isinstance(self.cs_params[i], OrdinalHyperparameter):
                unique_vals = set(X[:, i])
                if len(unique_vals) > len(self.cs_params[i].sequence):
                    raise RuntimeError(
                        "There are some sequence-options missing in the ConfigSpace specification for "
                        "hyperparameter %s:" % self.cs_params[i].name
                    )
            elif isinstance(self.cs_params[i], Constant):
                # oddly, unparameterizedhyperparameter and constant are not supported.
                # raise TypeError('Unsupported Hyperparameter: %s' % type(self.cs_params[i]))
                pass
                # unique_vals = set(X[:, i])
                # if len(unique_vals) > 1:
                #     raise RuntimeError('Got multiple values for Unparameterized (Constant) hyperparameter')
            else:
                raise TypeError(
                    "Unsupported Hyperparameter: %s" % type(self.cs_params[i])
                )

        if not np.issubdtype(X.dtype, np.float64):
            logging.warning("low level library expects X argument to be float")
        if not np.issubdtype(Y.dtype, np.float64):
            logging.warning("low level library expects Y argument to be float")

        # initialize all types as 0
        types = np.zeros(len(self.cs_params), dtype=np.uint)
        # retrieve the types and the bounds from the ConfigSpace
        # TODO: Test if that actually works
        for i, hp in enumerate(self.cs_params):
            if isinstance(hp, CategoricalHyperparameter):
                types[i] = len(hp.choices)
                pcs[i] = (len(hp.choices), np.nan)
            elif isinstance(hp, OrdinalHyperparameter):
                types[i] = len(hp.sequence)
                pcs[i] = (len(hp.sequence), np.nan)
            elif isinstance(self.cs_params[i], NumericalHyperparameter):
                pcs[i] = (hp.lower, hp.upper)
            elif isinstance(self.cs_params[i], Constant):
                types[i] = 1
                pcs[i] = (1, np.nan)
            else:
                raise TypeError("Unsupported Hyperparameter: %s" % type(hp))
        self.pcs = pcs

        # set forest options
        forest = reg.fanova_forest()
        forest.options.num_trees = n_trees
        forest.options.do_bootstrapping = bootstrapping
        forest.options.num_data_points_per_tree = (
            X.shape[0] if points_per_tree is None else points_per_tree
        )
        forest.options.tree_opts.max_features = (
            (X.shape[1] * 7) // 10 if max_features is None else max_features
        )

        forest.options.tree_opts.min_samples_to_split = min_samples_split
        forest.options.tree_opts.min_samples_in_leaf = min_samples_leaf
        forest.options.tree_opts.max_depth = max_depth
        forest.options.tree_opts.epsilon_purity = 1e-8

        # create data container and provide all the necessary information
        if seed is None:
            rng = reg.default_random_engine(np.random.randint(2 ** 31 - 1))
        else:
            rng = reg.default_random_engine(seed)
        data = reg.default_data_container(X.shape[1])

        for i, (mn, mx) in enumerate(pcs):
            if np.isnan(mx):
                data.set_type_of_feature(i, mn)
            else:
                data.set_bounds_of_feature(i, mn, mx)

        for i in range(len(Y)):
            data.add_data_point(X[i].tolist(), Y[i])

        forest.fit(data, rng)

        self.the_forest = forest
        self.data = data

        # initialize a dictionary with parameter dims
        self.variance_dict = dict()

        # getting split values
        forest_split_values = self.the_forest.all_split_values()

        # all midpoints and interval sizes treewise for the whole forest
        self.all_midpoints = []
        self.all_sizes = []

        # compute midpoints and interval sizes for variables in each tree
        for tree_split_values in forest_split_values:
            sizes = []
            midpoints = []
            for i, split_vals in enumerate(tree_split_values):
                if np.isnan(pcs[i][1]):  # categorical parameter
                    # check if the tree actually splits on this parameter
                    if len(split_vals) > 0:
                        midpoints.append(split_vals)
                        sizes.append(np.ones(len(split_vals)))
                    # if not, simply append 0 as the value with the number of categories as the size, that way this
                    # parameter will get 0 importance from this tree.
                    else:
                        midpoints.append((0,))
                        sizes.append((pcs[i][0],))
                else:
                    # add bounds to split values
                    sv = np.array([pcs[i][0]] + list(split_vals) + [pcs[i][1]])
                    # compute midpoints and sizes
                    midpoints.append((1 / 2) * (sv[1:] + sv[:-1]))
                    sizes.append(sv[1:] - sv[:-1])

            self.all_midpoints.append(midpoints)
            self.all_sizes.append(sizes)

        # capital V in the paper
        self.trees_total_variances = []
        # dict of lists where the keys are tuples of the dimensions
        # and the value list contains \hat{f}_U for the individual trees
        # reset all the variance fractions computed
        self.trees_variance_fractions = {}
        self.V_U_total = {}
        self.V_U_individual = {}

        self.cutoffs = cutoffs
        self.set_cutoffs(cutoffs)

    def predict(self, Xtest=None, Ytest=None):

        # Convert pd.DataFrame to np.array
        if isinstance(Xtest, pd.DataFrame):
            Xtest = Xtest.to_numpy()

        datatest = reg.default_data_container(Xtest.shape[1])

        for i, (mn, mx) in enumerate(self.pcs):
            if np.isnan(mx):
                datatest.set_type_of_feature(i, mn)
            else:
                datatest.set_bounds_of_feature(i, mn, mx)

        for i in range(Xtest.shape[0]):
            datatest.add_data_point(Xtest[i].tolist(), 0.5)

        ytesthat = []
        for i in range(Xtest.shape[0]):
            ytesthat.append(self.the_forest.predict(datatest.retrieve_data_point(i)))
        return ytesthat

    def set_cutoffs(self, cutoffs=(-np.inf, np.inf), quantile=None):
        """
        Setting the cutoffs to constrain the input space
        
        To properly do things like 'improvement over default' the
        fANOVA now supports cutoffs on the y values. These will exclude
        parts of the parameters space where the prediction is not within
        the provided cutoffs. This is is specialization of 
        "Generalized Functional ANOVA Diagnostics for High Dimensional
        Functions of Dependent Variables" by Hooker.
        """
        if not (quantile is None):
            percentile1 = self.percentiles[quantile[0]]
            percentile2 = self.percentiles[quantile[1]]
            self.the_forest.set_cutoffs(percentile1, percentile2)
        else:

            self.cutoffs = cutoffs
            self.the_forest.set_cutoffs(cutoffs[0], cutoffs[1])

        # reset all the variance fractions computed
        self.trees_variance_fractions = {}
        self.V_U_total = {}
        self.V_U_individual = {}

        # recompute the trees' total variance
        self.trees_total_variance = self.the_forest.get_trees_total_variances()
