import collections
from typing import Union, List, Any

import numpy as np
import pandas as pd

from scipy.stats import rv_discrete, uniform, randint
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import ParameterSampler

import ConfigSpace
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    NumericalHyperparameter,
)

from bw_select import *
from srcfanova.confspace_utils import get_unimp_hyperparam_configspace


class DiscreteRVWrapper:

    def __init__(self, param_name: str, data: List):
        self.param_name = param_name
        self.data_prime = collections.OrderedDict()
        for value in data:
            if value not in self.data_prime:
                self.data_prime[value] = 0
            self.data_prime[value] += (1.0 / len(data))
        self.prob_distrib = rv_discrete(values=(list(range(len(self.data_prime))), list(self.data_prime.values())))

    def rvs(self, *args: dict, **kwargs: dict) -> int:
        # assumes a samplesize of 1, for random search
        sample = self.prob_distrib.rvs(*args, **kwargs)
        value = list(self.data_prime.keys())[sample]

        return value


class KDEWrapper:

    def __init__(self, hyperparameter: Union[UniformIntegerHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter, NumericalHyperparameter],
                 param_name: str,
                 data: List,
                 oob_strategy: str = 'resample',
                 bandwith: float = 0.1):
        if oob_strategy not in ['resample', 'round', 'ignore']:
            raise ValueError()
        self.oob_strategy = oob_strategy
        self.param_name = param_name
        self.hyperparameter = hyperparameter
        reshaped = np.reshape(data, (len(data), 1))

        if self.hyperparameter.log:
            if isinstance(self.hyperparameter, UniformIntegerHyperparameter):
                raise ValueError(f'Log Integer hyperparameter not supported: {param_name}')
            self.prob_distrib = KernelDensity(kernel='gaussian', bandwidth=bandwith).fit(np.log2(reshaped))
        else:
            self.prob_distrib = KernelDensity(kernel='gaussian', bandwidth=bandwith).fit(reshaped)
        pass

    def pdf(self, x):
        x = np.reshape(x, (len(x), 1))
        if self.hyperparameter.log:
            x = np.log2(x)
        log_dens = self.prob_distrib.score_samples(x)
        return np.exp(log_dens)

    def rvs(self, *args: dict, **kwargs: dict) -> float:
        # assumes a samplesize of 1, for random search
        while True:
            sample = self.prob_distrib.sample(n_samples=1, random_state=kwargs['random_state'])[0][0]
            if self.hyperparameter.log:
                value = np.power(2, sample)
            else:
                value = sample
            if isinstance(self.hyperparameter, UniformIntegerHyperparameter):
                value = int(round(value))

            if self.hyperparameter.lower <= value <= self.hyperparameter.upper:
                return value
            elif self.oob_strategy == 'ignore':
                # TODO: hacky fail safe for some hyperparameters
                if hasattr(self.hyperparameter, 'lower_hard') and self.hyperparameter.lower_hard > value:
                    continue
                if hasattr(self.hyperparameter, 'upper_hard') and self.hyperparameter.upper_hard < value:
                    continue

                return value
            elif self.oob_strategy == 'round':
                if value < self.hyperparameter.lower:
                    return self.hyperparameter.lower
                elif value > self.hyperparameter.upper:
                    return self.hyperparameter.upper


def get_values_for_hyperparam(hyperparam: Union[UniformIntegerHyperparameter, CategoricalHyperparameter, UniformFloatHyperparameter, NumericalHyperparameter],
                              prior_data: np.array,
                              n_samples: int,
                              seed_nb: int,
                              oob_strategy: str = 'resample',
                              bandwidth: float = 0.1) -> List:

    param_grid = dict()
    wrapper = None

    if isinstance(hyperparam, NumericalHyperparameter):
        wrapper = KDEWrapper(hyperparam, hyperparam.name, prior_data, oob_strategy, bandwidth)
    elif isinstance(hyperparam, CategoricalHyperparameter):
        wrapper = DiscreteRVWrapper(hyperparam.name, prior_data)

    param_grid[hyperparam.name] = wrapper
    parameter_iterable = list(ParameterSampler(param_distributions=param_grid,
                                               n_iter=n_samples,
                                               random_state=np.random.RandomState(seed_nb)))
    parameter_iterable_df = pd.DataFrame(parameter_iterable)
    list_values = list(parameter_iterable_df[hyperparam.name])

    return list_values


def get_hyperparam_priors_across_tasks(best_N: int,
                                       task_id_to_leave: int,
                                       param_name: str,
                                       all_data: pd.DataFrame) -> np.array:

    data_task_leave_one = all_data[all_data["task_id"] != task_id_to_leave]
    task_ids_leave_one = sorted(data_task_leave_one["task_id"].unique())

    df_leave_one = None
    for i, task in enumerate(task_ids_leave_one):
        data_task = data_task_leave_one[data_task_leave_one.task_id == task]
        best_N_config = data_task.sort_values('val_binary_accuracy')[-best_N:]

        if df_leave_one is None:
            df_leave_one = best_N_config
        else:
            df_leave_one = pd.concat((df_leave_one, best_N_config))

    hyperparam_prior_vals = np.array(df_leave_one[param_name])

    return hyperparam_prior_vals


def compute_bandwidth_from_priors(prior_data: np.array,
                                  kde_bw_estimator: str) -> float:

    if kde_bw_estimator == 'silverman':
        bw = hsilverman(prior_data)
    elif kde_bw_estimator == 'sj':
        bw = hsj(prior_data)
    else:
        raise ValueError('This bandwidth type is not supported')

    return bw


def get_kde_essentials(task_id: int,
                       config_space: ConfigSpace.ConfigurationSpace,
                       cs_params: List,
                       important_hyperparams_indices: List,
                       all_data: pd.DataFrame,
                       kde_bw: List,
                       kde_bw_estimator: str,
                       best_N: int):

    imp_hyperparams_prior_data = {}

    for h_i in important_hyperparams_indices:
        imp_hyperparam_obj = config_space[cs_params[h_i]]
        imp_hyperparam_name = imp_hyperparam_obj.name
        imp_hyperparam_priors = get_hyperparam_priors_across_tasks(best_N,
                                                                   task_id,
                                                                   imp_hyperparam_name,
                                                                   all_data)
        imp_hyperparams_prior_data[imp_hyperparam_name] = imp_hyperparam_priors

    if kde_bw is not None:
        assert (len(kde_bw) == len(important_hyperparams_indices))
    else:
        kde_bw = []
        for h_i in important_hyperparams_indices:
            imp_hyperparam_obj = config_space[cs_params[h_i]]
            param = imp_hyperparam_obj.name
            if isinstance(imp_hyperparam_obj, NumericalHyperparameter):
                param_priors = imp_hyperparams_prior_data[param]
                bw_param = compute_bandwidth_from_priors(prior_data=param_priors,
                                                         kde_bw_estimator=kde_bw_estimator)
            elif isinstance(imp_hyperparam_obj, CategoricalHyperparameter):
                bw_param = 0  # because a simple RVDiscrete prob. distribution is used which does not need bw.
            
            kde_bw.append(bw_param)
    
    # print(kde_bw)
    
    return imp_hyperparams_prior_data, kde_bw


def get_kde_parameter_distribution(task_id: int,
                                   config_space: ConfigSpace.ConfigurationSpace,
                                   cs_params: List,
                                   important_hyperparams_indices: List,
                                   all_data: pd.DataFrame,
                                   kde_bw: List,
                                   kde_bw_estimator: str,
                                   seed_nb: int,
                                   best_N: int,
                                   n_configs: int):

    imp_hyperparams_prior_data, kde_bw = get_kde_essentials(task_id=task_id,
                                                            config_space=config_space,
                                                            cs_params=cs_params,
                                                            important_hyperparams_indices=important_hyperparams_indices,
                                                            all_data=all_data,
                                                            kde_bw=kde_bw,
                                                            kde_bw_estimator=kde_bw_estimator,
                                                            best_N=best_N)

    unimp_config_space = get_unimp_hyperparam_configspace(important_hyperparams_indices)
    unimp_config_space.seed(seed_nb)
    param_distribution = pd.DataFrame(unimp_config_space.sample_configuration(n_configs))

    # TODO: bad code, fix it so that samples are not sampled at every iteration of HB!
    for j, h_i in enumerate(important_hyperparams_indices):
        imp_hyperparam_obj = config_space[cs_params[h_i]]
        imp_hyperparam_name = imp_hyperparam_obj.name

        hyperparam_bw = kde_bw[j]
        imp_hyperparam_vals = get_values_for_hyperparam(imp_hyperparam_obj,
                                                        imp_hyperparams_prior_data[imp_hyperparam_name],
                                                        n_configs,
                                                        seed_nb,
                                                        'resample',
                                                        hyperparam_bw)
        # print(imp_hyperparam_vals)
        param_distribution[imp_hyperparam_name] = imp_hyperparam_vals

    return param_distribution
