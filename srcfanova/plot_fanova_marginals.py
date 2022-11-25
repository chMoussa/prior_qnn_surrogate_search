import argparse
import ConfigSpace
import copy
import fanova.fanova
import fanova.visualizer

import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import logging

import os
import itertools
import typing
import pandas as pd

from srcfanova.confspace_utils import get_configspace, integer_encode_dataframe


# to plot: <openml_pimp_root>/examples/plot/plot_fanova_aggregates.py
def read_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="nonaggresults_hyper.csv", type=str)
    parser.add_argument("--output_directory", default="./holdoutfanova/", type=str)
    parser.add_argument("--measure", default="val_binary_accuracy", type=str)
    parser.add_argument("--n_combi_params", default=1, type=int)
    parser.add_argument("--exclude_epochs", type=int, default=1)

    parser.add_argument("--plot_marginals", action="store_true", default=True)
    parser.add_argument("--plot_extension", default="pdf", type=str)
    parser.add_argument("--plot_resolution", default=100, type=int)

    parser.add_argument("--n_trees", default=16, type=int)

    parser.add_argument("--resolution", default=100, type=int)
    parser.add_argument("--task_idx", type=int, default=None)
    parser.add_argument("--task_id_column", default="dataset", type=str)
    parser.add_argument("--show_legend", action="store_true", default=True)
    parser.add_argument("--tick_size", default=12, type=int)
    parser.add_argument("--label_size", default=14, type=int)
    parser.add_argument("--subsample", default=None, type=int)
    args_, misc = parser.parse_known_args()

    return args_


def apply_logscale(X: np.array, config_space: ConfigSpace.ConfigurationSpace):
    X_prime = np.array(X)
    config_space_prime = copy.deepcopy(config_space)
    for idx, hp in enumerate(config_space_prime.get_hyperparameters()):
        if isinstance(hp, ConfigSpace.hyperparameters.NumericalHyperparameter):
            if hp.log:
                X_prime[:, idx] = np.log(X_prime[:, idx])
                hp.lower = np.log(hp.lower)
                hp.upper = np.log(hp.upper)
                hp.log = False
    for idx, hp in enumerate(config_space_prime.get_hyperparameters()):
        if isinstance(hp, ConfigSpace.hyperparameters.NumericalHyperparameter):
            lowest = np.min(X_prime[:, idx])
            highest = np.max(X_prime[:, idx])
            assert hp.lower <= lowest <= highest <= hp.upper
            assert hp.log is False
    return X_prime, config_space_prime


def plot_single_marginal(
    X: np.array,
    y: np.array,
    config_space: ConfigSpace.ConfigurationSpace,
    name_prefix: str,
    hyperparameter_name: str,
    directory: str,
    y_range: typing.Optional[typing.Tuple[int, int]],
    measure_name: str,
    n_trees: int,
    resolution: int,
    tick_size: int,
    label_size: int,
    show_legend: bool,
    plot_extension: str,
):
    evaluator = fanova.fanova.fANOVA(
        X=X, Y=y, config_space=config_space, n_trees=n_trees
    )
    visualizer = fanova.visualizer.Visualizer(
        evaluator, config_space, "/tmp/", y_label=measure_name
    )

    plt.close("all")
    plt.clf()
    plt.rc("xtick", labelsize=tick_size)
    plt.rc("ytick", labelsize=tick_size)
    plt.rc("axes", labelsize=label_size)
    hyperparameter_idx = config_space.get_idx_by_hyperparameter_name(
        hyperparameter_name
    )
    os.makedirs(directory, exist_ok=True)
    outfile_name = os.path.join(
        directory,
        "%s__%s.%s"
        % (name_prefix, hyperparameter_name.replace(os.sep, "_"), plot_extension),
    )
    visualizer.plot_marginal(hyperparameter_idx, resolution=resolution, show=False)

    x1, x2, _, _ = plt.axis()
    if y_range:
        plt.axis((x1, x2, y_range[0], y_range[1]))
    ax = plt.gca()
    ax.set_xlabel(hyperparameter_name.replace("_", " ").capitalize())
    ax.set_ylabel(measure_name.replace("_", " ").capitalize())
    if not show_legend and ax.get_legend() is not None:
        ax.get_legend().remove()

    plt.tight_layout()
    plt.savefig(outfile_name)
    logging.info("saved marginal plot to: %s" % outfile_name)


def plot_pairwise_marginal(
    X: np.array,
    y: np.array,
    config_space: ConfigSpace.ConfigurationSpace,
    name_prefix: str,
    hyperparameter_names: typing.Tuple[str],
    directory: str,
    z_range: typing.Optional[typing.Tuple[int, int]],
    measure_name: str,
    n_trees: int,
    resolution: int,
    tick_size: int,
    label_size: int,
    show_legend: bool,
    plot_extension: str,
):
    X_prime, config_space_prime = apply_logscale(X, config_space)
    evaluator = fanova.fanova.fANOVA(
        X=X_prime, Y=y, config_space=config_space_prime, n_trees=n_trees
    )
    visualizer = fanova.visualizer.Visualizer(
        evaluator, config_space_prime, "/tmp/", y_label=measure_name
    )

    plt.close("all")
    plt.clf()
    plt.rc("xtick", labelsize=tick_size)
    plt.rc("ytick", labelsize=tick_size)
    plt.rc("axes", labelsize=label_size)
    if len(hyperparameter_names) != 2:
        raise ValueError()
    idx1 = config_space.get_idx_by_hyperparameter_name(hyperparameter_names[0])
    idx2 = config_space.get_idx_by_hyperparameter_name(hyperparameter_names[1])

    indices = [(idx1, idx2), (idx2, idx1)]
    for hp1_hp2 in indices:
        hp1_name = config_space_prime.get_hyperparameter_by_idx(hp1_hp2[0])
        hp2_name = config_space_prime.get_hyperparameter_by_idx(hp1_hp2[1])
        os.makedirs(directory, exist_ok=True)
        outfile_name = os.path.join(
            directory,
            "%s__%s__%s.%s"
            % (
                name_prefix,
                hp1_name.replace(os.sep, "_"),
                hp2_name.replace(os.sep, "_"),
                plot_extension,
            ),
        )
        try:
            visualizer.plot_pairwise_marginal(
                hp1_hp2,
                resolution=resolution,
                show=False,
                colormap=matplotlib.cm.viridis,
                add_colorbar=False,
            )
            ax = plt.gca()
            if z_range:
                ax.set_zlim3d(z_range[0], z_range[1])
            # note that we use original config space
            xlabel_log_str = (
                " (log)"
                if hasattr(config_space.get_hyperparameter(hp1_name), "log")
                and config_space.get_hyperparameter(hp1_name).log
                else ""
            )
            ylabel_log_str = (
                " (log)"
                if hasattr(config_space.get_hyperparameter(hp2_name), "log")
                and config_space.get_hyperparameter(hp2_name).log
                else ""
            )
            ax.set_xlabel(hp1_name.replace("_", " ").capitalize() + xlabel_log_str)
            ax.set_ylabel(hp2_name.replace("_", " ").capitalize() + ylabel_log_str)
            # ax.set_zlabel(measure_name.replace('_', ' ').capitalize())
            if not show_legend and ax.get_legend() is not None:
                ax.get_legend().remove()

            plt.savefig(outfile_name, bbox_inches="tight")
            logging.info("saved marginal plot (3D) to: %s" % outfile_name)
        except IndexError as e:
            logging.warning(
                "IndexError with hyperparameters %s and %s: %s"
                % (hp1_name, hp2_name, e)
            )


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logging.info("Start %s: %s" % (os.path.basename(__file__), vars(args)))

    # Read data and get IDs of tasks
    dataf = pd.read_csv(args.dataset_path, sep=",")
    task_ids = dataf[args.task_id_column].unique()
    if args.task_idx:
        task_ids = [task_ids[args.task_idx]]

    if args.measure not in dataf.columns.values:
        raise ValueError("Could not find measure in dataset: %s" % args.measure)

    # make sure data is numerical and in right order for configspace
    config_space = get_configspace(bool(args.exclude_epochs))
    cs_params = config_space.get_hyperparameter_names()

    data = dataf.loc[:, [cs_params[i] for i in range(len(cs_params))]]
    data = integer_encode_dataframe(data, config_space)
    data[args.task_id_column] = dataf[args.task_id_column]

    data[args.measure] = dataf[args.measure]

    cs_params_comb = cs_params.copy()
    if args.n_combi_params == 2:
        cs_params_comb = [
            c1 + "__" + c2 for (c1, c2) in itertools.combinations(cs_params, 2)
        ]

    for t_idx, task_id in enumerate(task_ids):
        logging.info(
            "Running fanova on task %s (%d/%d)" % (task_id, t_idx + 1, len(task_ids))
        )
        data_task = data[data[args.task_id_column] == task_id]
        del data_task[args.task_id_column]
        # now dataset is gone, and all categoricals are converted, we can convert to float
        data_task = data_task.astype(np.float)
        if args.subsample:
            indices = np.random.choice(len(data_task), args.subsample, replace=False)
            data_task = data_task.iloc[indices]
        logging.info(
            "Dimensions: %s (out of (%s)) %s"
            % (
                str(data_task.shape),
                str(data.shape),
                "[Subsampled]" if args.subsample else "",
            )
        )
        assert len(data_task) >= min(
            100, args.subsample if args.subsample is not None else 100
        )
        os.makedirs(args.output_directory, exist_ok=True)
        X_data = data_task[config_space.get_hyperparameter_names()].values
        y_data = data_task[args.measure].values

        for hyperparameters_str in cs_params_comb:
            hyperparameters = hyperparameters_str.split("__")
            logging.info("-- Starting with: %s" % hyperparameters)

            if len(hyperparameters) == 1:
                plot_single_marginal(
                    X=X_data,
                    y=y_data,
                    config_space=config_space,
                    name_prefix=task_id,
                    hyperparameter_name=hyperparameters[0],
                    directory=os.path.join(args.output_directory, "marginal_plots"),
                    y_range=None,
                    measure_name=args.measure,
                    n_trees=args.n_trees,
                    resolution=args.plot_resolution,
                    tick_size=args.tick_size,
                    label_size=args.label_size,
                    show_legend=args.show_legend,
                    plot_extension=args.plot_extension,
                )
            elif len(hyperparameters) == 2:
                plot_pairwise_marginal(
                    X=X_data,
                    y=y_data,
                    config_space=config_space,
                    name_prefix=task_id,
                    hyperparameter_names=hyperparameters,
                    directory=os.path.join(
                        args.output_directory, "marginal_plots_pairwise"
                    ),
                    z_range=None,
                    measure_name=args.measure,
                    n_trees=args.n_trees,
                    resolution=args.plot_resolution,
                    tick_size=args.tick_size,
                    label_size=args.label_size,
                    show_legend=args.show_legend,
                    plot_extension=args.plot_extension,
                )
            else:
                raise ValueError(
                    "No support yet for higher dimensions than 2. Got: %d"
                    % len(hyperparameters)
                )


if __name__ == "__main__":
    run(read_cmd())
