import argparse
import pandas as pd
import itertools
import numpy as np
import logging
import os

import fanova.fanova
import fanova.visualizer


from srcfanova.confspace_utils import get_configspace, integer_encode_dataframe


def parse_args():
    parser = argparse.ArgumentParser(description="Choose Hyperparam search.")
    parser.add_argument("--task_idx", type=int, default=None)
    parser.add_argument("--measure", default="val_binary_accuracy", type=str)
    parser.add_argument("--comb_size", type=int, default=1)
    parser.add_argument("--n_trees", default=16, type=int)
    parser.add_argument("--fileres", type=str, default="nonaggresults_hyper.csv")
    parser.add_argument("--exclude_epochs", type=int, default=1)
    parser.add_argument("--resolution", default=100, type=int)
    parser.add_argument("--output_directory", default="./holdoutfanova/", type=str)
    args = parser.parse_args()
    return args


def run(args):

    # Read data and get IDs of tasks
    dataf = pd.read_csv(args.fileres, sep=",")
    task_ids = dataf["task_id"].unique()
    if args.task_idx:
        task_ids = [task_ids[args.task_idx]]

    # make sure data is numerical and in right order for configspace
    config_space = get_configspace(bool(args.exclude_epochs))
    cs_params = config_space.get_hyperparameter_names()

    data = dataf.loc[:, [cs_params[i] for i in range(len(cs_params))]]
    data = integer_encode_dataframe(data, config_space)
    data["task_id"] = dataf.task_id
    data[args.measure] = dataf[args.measure]

    result = list()
    for t_idx, task_id in enumerate(task_ids):
        data_task = data[data["task_id"] == task_id]
        del data_task["task_id"]

        os.makedirs(args.output_directory, exist_ok=True)

        y_data = data_task[args.measure].values
        X_data = data_task.copy()
        del X_data[args.measure]

        evaluator = fanova.fanova.fANOVA(X=X_data, Y=y_data, n_trees=args.n_trees)
        vis = fanova.visualizer.Visualizer(
            evaluator,
            config_space,
            args.output_directory,
            y_label="Predictive Accuracy",
        )

        indices = list(range(len(config_space.get_hyperparameters())))

        for comb_size in range(1, args.comb_size + 1):
            for h_idx in itertools.combinations(indices, comb_size):
                param_names = np.array(cs_params)[np.array(h_idx)]

                logging.info("-- Calculating marginal for %s" % param_names)
                importance = evaluator.quantify_importance(h_idx)[h_idx]
                if comb_size == 1:
                    visualizer_res = vis.generate_marginal(h_idx[0], args.resolution)
                    # visualizer returns mean, std and potentially grid
                    avg_marginal = np.array(visualizer_res[0])

                elif comb_size == 2:
                    visualizer_res = vis.generate_pairwise_marginal(
                        h_idx, args.resolution
                    )
                    # visualizer returns grid names and values
                    avg_marginal = np.array(visualizer_res[1])

                else:
                    raise ValueError(
                        "No support yet for higher dimensions than 2. Got: %d"
                        % comb_size
                    )

                difference_max_min = max(avg_marginal.reshape((-1,))) - min(
                    avg_marginal.reshape((-1,))
                )

                current = {
                    "task_id": task_id,
                    "hyperparameter": " / ".join(param_names),
                    "n_hyperparameters": len(param_names),
                    "importance_variance": importance["individual importance"],
                    "importance_max_min": difference_max_min,
                }

                result.append(current)

    df_result = pd.DataFrame(result)
    result_path = os.path.join(
        args.output_directory, "fanova_depth_" + str(comb_size) + ".csv"
    )
    df_result.to_csv(result_path)
    logging.info("resulting csv: %s" % result_path)


if __name__ == "__main__":

    run(parse_args())
