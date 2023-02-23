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

# Interesting Blog: https://medium.com/data-from-the-trenches/a-slightly-better-budget-allocation-for-hyperband-bbd45af14481


class HyperBandOptimiser:

    def __init__(self,
                 eta: int,
                 config_space: ConfigSpace.ConfigurationSpace,
                 optimisation_goal: str,
                 max_iter: int,
                 min_or_max: Callable,  # for us it will be max
                 task_id: int,
                 search_type: str = 'uniform',
                 important_hyperparams_indices: List = None,
                 best_N: int = None,
                 seed_nb: int = 0,
                 kde_bw_estimator: str = 'silverman',
                 kde_bw: List = None,
                 pickle_path: str = None):

        if max_iter is None:
            raise ValueError('For HB, max_iter cannot be None')
            
        self.max_iter = max_iter
        self.min_or_max = min_or_max
        self.config_space = config_space
        self.eta = eta
        self.optimisation_goal = optimisation_goal
        self.task_id = task_id
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
        self.hblines = []

        if pickle_path is None:
            if self.search_type == 'uniform':
                pickle_path = f'./optimiser/uniform/task_id{self.task_id}_search_{self.search_type}_eta_{self.eta}_max_iter_{self.max_iter}_seed_{self.seed_nb}.pckl'
            elif self.search_type == 'kde':
                pickle_path = f'./optimiser/kde/task_id{self.task_id}_search_{self.search_type}_bw_{self.kde_bw}_bw_est_{self.kde_bw_estimator}_bestN_{self.best_N}_eta_{self.eta}_max_iter_{self.max_iter}_imp_hyp_{self.important_hyperparams_indices}_seed_{self.seed_nb}.pckl'

        self.pickle_path = pickle_path

    def _get_best_n_configs(self, n: int, df_w_config_evals: pd.DataFrame) -> pd.DataFrame:
        is_descending = self.min_or_max
        
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
        param_distribution_df_int_encoded = integer_encode_dataframe(param_distribution_df,
                                                                     config_space)

        return param_distribution_df, param_distribution_df_int_encoded

    def run_optimisation(self,
                         model_for_task_id: fANOVA_surrogate,
                         all_data: pd.DataFrame,
                         store_optimiser: bool = True,
                         verbosity: bool = False):

        R = self.max_iter  
        eta = self.eta  # halving rate

        def log_eta(x: int):
            return mpmath.log(x) / mpmath.log(eta)

        config_space = self.config_space
        cs_params = config_space.get_hyperparameter_names()
        s_max = int(log_eta(R))
        B = (s_max + 1) * R             # total/max resources (without reuse) per execution of Successive Halving
        unused_budget = 0
        best_so_far = 0
        best_config_so_far = pd.DataFrame
        
        for s in reversed(range(s_max + 1)):
            n_configs = int(mpmath.ceil(int(B / R / (s + 1)) * eta ** s))  # initial number of configurations
            r = R * eta ** (-s)                       # initial resources allocated to each evaluator/arm
            lines = []
            
            param_distribution_df, param_distribution_df_int_encoded = self._get_parameter_distribution(config_space=config_space,
                                                                                                        cs_params=cs_params,
                                                                                                        all_data=all_data,
                                                                                                        n_configs=n_configs)
            
            self.param_distribution_history[(s, n_configs, r)] = param_distribution_df
            # print(f'Starting HB iteration {s}') 

            for i in range(s + 1):
                n_configs_i = n_configs * eta ** (-i)
                r_i = r * eta ** i
                lines.append((int(n_configs_i), int(r_i)))
                
                param_distribution_df_int_encoded['epochs'] = int(r_i)
                
                config_evals = model_for_task_id.predict(param_distribution_df_int_encoded).copy()
                param_distribution_df.loc[:, self.optimisation_goal] = config_evals
                
                best_remaining_configs = self._get_best_n_configs(n=int(n_configs_i / eta),
                                                                  df_w_config_evals=param_distribution_df)
                

                try:
                    best_eval_in_round = self.min_or_max(best_remaining_configs[self.optimisation_goal])
                except:
                    continue
                    
                best_config_w_perf_in_round = best_remaining_configs[
                    best_remaining_configs[self.optimisation_goal] == self.min_or_max(best_remaining_configs[self.optimisation_goal])]
                
                best_config_in_round = best_config_w_perf_in_round.drop(columns=[self.optimisation_goal])
                
                self.eval_history.append(best_eval_in_round)
                self.config_history[(s, i)] = best_config_in_round
                self.config_history_w_perf[(s, i)] = best_config_w_perf_in_round
                
                if self.min_or_max == max:
                    if self.min_or_max(self.eval_history) > best_so_far:
                        best_so_far = self.min_or_max(self.eval_history)
                        best_config_so_far = best_config_w_perf_in_round
                else:
                    if self.min_or_max(self.eval_history) < best_so_far:
                        best_so_far = self.min_or_max(self.eval_history)
                        best_config_so_far = best_config_w_perf_in_round

                if verbosity:
                    print(best_so_far)
            
            self.hblines.append(lines)
            
            if store_optimiser:
                self.store_optimiser()
                
        return best_config_so_far
                
                
                
                
                
                
                


            
            
