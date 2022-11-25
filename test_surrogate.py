import pandas as pd
import numpy as np
import itertools as it
from collections import OrderedDict

from surrogate import fANOVA_surrogate
from srcfanova.confspace_utils import get_configspace, integer_encode_dataframe

from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from scipy.stats import spearmanr


def scorer_utils_cc(x, y):
    coef, p = spearmanr(x, y)
    return coef


if __name__ == "__main__":

    # read data - retirve tasks
    dataf = pd.read_csv("./results_crossval_hyper.csv", sep=",")
    task_ids = sorted(dataf["task_id"].unique())
    measure = "val_binary_accuracy"
    n_trees = 128

    # obtain the configuration space to get parameter names and prepare data accordingly
    config_space = get_configspace(bool(1))
    cs_params = config_space.get_hyperparameter_names()

    data = dataf.loc[:, [cs_params[i] for i in range(len(cs_params))]]
    data = integer_encode_dataframe(data, config_space)
    data["task_id"] = dataf.task_id
    data[measure] = dataf[measure]

    scorer_cc = make_scorer(scorer_utils_cc, greater_is_better=True)

    median_for_ntrees = []
    median_for_ntrees_cc = []
    cv_obj = KFold(10, shuffle=True, random_state=0)

    # for each task, fit surrogate and compute metrics
    for t_idx, task_id in enumerate(task_ids):
        data_task = data[data["task_id"] == task_id]
        del data_task["task_id"]

        y_data = data_task[measure].values
        X_data = data_task.copy()
        del X_data[measure]

        result_cv = {"r2": [], "rmse": [], "cc": []}
        for train, test in cv_obj.split(X_data):
            X_data_train = X_data.iloc[train, :]
            y_data_train = y_data[train]
            X_data_test = X_data.iloc[test, :]
            y_data_test = y_data[test]

            model = fANOVA_surrogate(
                X=X_data_train, Y=y_data_train, n_trees=n_trees, seed=t_idx
            )
            ytesthat = model.predict(X_data_test)
            result_cv["r2"].append(r2_score(y_data_test, ytesthat))
            result_cv["rmse"].append(
                mean_squared_error(y_data_test, ytesthat, squared=False)
            )
            result_cv["cc"].append(scorer_utils_cc(y_data_test, ytesthat))
        median_for_ntrees.append(np.mean(result_cv["r2"]))
        median_for_ntrees_cc.append(np.mean(result_cv["cc"]))
        print(
            task_id,
            np.mean(result_cv["r2"]),
            np.mean(result_cv["rmse"]),
            np.mean(result_cv["cc"]),
        )
