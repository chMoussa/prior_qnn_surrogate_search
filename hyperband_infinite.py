import os
import sys
import pickle
from typing import Callable, List, Optional, Union, Tuple, Any
import pandas as pd

import mpmath
import ConfigSpace
from ConfigSpace.configuration_space import ConfigurationSpace
from surrogate import fANOVA_surrogate
from srcfanova.confspace_utils import integer_encode_dataframe
from prior_utils import get_kde_parameter_distribution

mpmath.mp.dps = 64 # this is important for avoiding issues with numerical precision. For instance,
# if R (max_iter=243 and eta=3, then s_max should be 5 but when you use numpy, we get 4!


class HyperBandOptimiser:

    def __init__(self,
                 eta: int,
                 config_space: ConfigSpace.ConfigurationSpace,
                 optimisation_goal: str,
                 max_k: int,
                 min_or_max: Callable,  # for us it will be max
                 task_id: int,
                 starting_shots_nb: int,
                 search_type: str = 'uniform',
                 important_hyperparams_indices: List = None,
                 best_N: int = None,
                 seed_nb: int = 0,
                 kde_bw_estimator: str = 'silverman',
                 kde_bw: List = None,
                 pickle_path: str = None):

        if max_k is None:
            raise ValueError('For HB, max_k cannot be None')
            
        self.max_k = max_k
        self.min_or_max = min_or_max
        self.config_space = config_space
        self.eta = eta
        self.optimisation_goal = optimisation_goal
        self.task_id = task_id
        self.starting_shots_nb = starting_shots_nb
        self.search_type = search_type
        self.important_hyperparams_indices = important_hyperparams_indices
        self.best_N = best_N
        self.seed_nb = seed_nb
        self.kde_bw = kde_bw
        self.kde_bw_estimator = kde_bw_estimator
        self.pickle_path = pickle_path

        self.eval_history = []
        self.config_history = {}
        self.config_history_w_perf = {}
        self.param_distribution_history = {}

        if pickle_path is None:
            if self.search_type == 'uniform':
                pickle_path = f'./optimiser/uniform/task_id{self.task_id}_search_{self.search_type}_eta_{self.eta}_max_k_{self.max_k}_shots_{self.starting_shots_nb}_seed_{self.seed_nb}.pckl'
            elif self.search_type == 'kde':
                pickle_path = f'./optimiser/kde/task_id{self.task_id}_search_{self.search_type}_bw_{self.kde_bw}_bw_est_{self.kde_bw_estimator}_bestN_{self.best_N}_eta_{self.eta}_max_k_{self.max_k}_shots_{self.starting_shots_nb}_imp_hyp_{self.important_hyperparams_indices}_seed_{self.seed_nb}.pckl'

        self.pickle_path = pickle_path

    def _get_best_n_configs(self, n: int, df_w_config_evals: pd.DataFrame) -> pd.DataFrame:
        is_descending = self.min_or_max
        # TODO: check whether this makes sense!!!
        if is_descending == max:
            flag = False
        else:
            flag = True
        sorted_configs_by_res = df_w_config_evals.sort_values(by=self.optimisation_goal, ascending=flag)

        return sorted_configs_by_res[:n]

    def store_optimiser(self, pickle_path=None):
        if pickle_path is None:
            pickle_path = self.pickle_path
            
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        pickle.dump(self, open(pickle_path, 'wb'))

    def _get_parameter_distribution(self,
                                    config_space: ConfigSpace.ConfigurationSpace,
                                    cs_params: List,
                                    all_data: pd.DataFrame,
                                    n_configs: int) -> Tuple[Any, pd.DataFrame]:

        task_id = self.task_id
        search_type = self.search_type
        important_hyperparams_indices = self.important_hyperparams_indices
        best_N = self.best_N
        seed_nb = self.seed_nb
        kde_bw = self.kde_bw
        kde_bw_estimator = self.kde_bw_estimator

        # param_distribution = pd.DataFrame()

        if search_type == 'uniform':
            config_space.seed(seed_nb)
            param_distribution = pd.DataFrame(config_space.sample_configuration(n_configs))
        if search_type == 'kde' and important_hyperparams_indices != []:
            param_distribution = get_kde_parameter_distribution(task_id=task_id,
                                                                config_space=config_space,
                                                                cs_params=cs_params,
                                                                important_hyperparams_indices=important_hyperparams_indices,
                                                                all_data=all_data,
                                                                kde_bw=kde_bw,
                                                                kde_bw_estimator=kde_bw_estimator,
                                                                seed_nb=seed_nb,
                                                                best_N=best_N,
                                                                n_configs=n_configs)
        
        param_distribution_df = param_distribution.loc[:, [cs_params[j] for j in range(len(cs_params))]]
        # print(param_distribution)
        param_distribution_df_int_encoded = integer_encode_dataframe(param_distribution_df,
                                                                     config_space)

        return param_distribution_df, param_distribution_df_int_encoded

    def run_optimisation(self,
                         model_for_task_id: fANOVA_surrogate,
                         all_data: pd.DataFrame,
                         store_optimiser: bool = True,
                         verbosity: bool = False):

        max_k = self.max_k  # number of times one wants to run HB optimiser
        eta = self.eta  # halving rate
        starting_shots_nb = self.starting_shots_nb

        def log_eta(x: int):
            return mpmath.log(x) / mpmath.log(eta)

        config_space = self.config_space
        cs_params = config_space.get_hyperparameter_names()

        k = 0
        min_configs = eta

        # Exploration-exploitation trade-off management outer loop
        while k < max_k:
            l = 0
            B = starting_shots_nb * 2 ** k

            while int(log_eta(B)) - l > log_eta(l):
                l += 1
            l = max(0, l - 1)

            while l >= 0:
                n_configs = eta ** l

                if n_configs >= min_configs:
                    halvings = max(1, int(mpmath.ceil(log_eta(n_configs))))

                    param_distribution_df, param_distribution_df_int_encoded = self._get_parameter_distribution(
                        config_space=config_space,
                        cs_params=cs_params,
                        all_data=all_data,
                        n_configs=n_configs)
                    
                    self.param_distribution_history[(k, l)] = param_distribution_df

                    config_evals = model_for_task_id.predict(param_distribution_df_int_encoded).copy()
                    param_distribution_df.loc[:, self.optimisation_goal] = config_evals

                    # Successive halving with rate eta
                    for i in range(halvings):
                        n_configs_i = int(n_configs / eta ** i)  # evaluate n_configs_i configurations/arms
                        budget_configs_i = int(
                            B / n_configs_i / halvings)  # each with budget_configs_i resources/budget

                        # Halving: keep best 1/eta of them, which will be allocated more resources/iterations
                        best_remaining_configs = self._get_best_n_configs(n=max(1, int(mpmath.ceil(n_configs_i / eta))),
                                                                          df_w_config_evals=param_distribution_df)
                        best_eval_in_round = self.min_or_max(best_remaining_configs[self.optimisation_goal])
                        best_config_w_perf_in_round = best_remaining_configs[
                            best_remaining_configs[self.optimisation_goal] == self.min_or_max(
                                best_remaining_configs[self.optimisation_goal])]
                        best_config_in_round = best_config_w_perf_in_round.drop(columns=[self.optimisation_goal])

                        self.eval_history.append(best_eval_in_round)
                        self.config_history[(k, i)] = best_config_in_round
                        self.config_history_w_perf[(k, i)] = best_config_w_perf_in_round

                        if verbosity:
                            best_so_far = self.min_or_max(self.eval_history)
                            print(best_so_far)

                l -= 1

                if store_optimiser:
                    self.store_optimiser()

            k += 1
