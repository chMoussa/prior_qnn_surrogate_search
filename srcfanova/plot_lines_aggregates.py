import argparse
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import logging
import math
import openml
import Orange
import os
import pandas as pd
import seaborn as sns
import typing


def read_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fanova_result_file", default="./holdoutfanova/fanova_depth_1.csv", type=str
    )
    parser.add_argument("--output_directory", default="./holdoutfanova/", type=str)
    parser.add_argument("--measure", default="val_binary_accuracy", type=str)
    parser.add_argument("--font_size", default=16, type=int)
    parser.add_argument("--n_combi_params", default=1, type=int)
    parser.add_argument("--plot_extension", default="pdf", type=str)
    parser.add_argument("--log_scale", action="store_false", default=False)
    parser.add_argument("--plot_nemenyi", action="store_true", default=True)
    parser.add_argument("--nemenyi_width", default=12, type=int)
    parser.add_argument("--nemenyi_textspace", default=3, type=int)
    args_, misc = parser.parse_known_args()

    return args_


def critical_dist(numModels, numDatasets):
    # confidence values for alpha = 0.05. Index is the number of models (minimal two)
    alpha005 = [
        -1,
        -1,
        1.959964233,
        2.343700476,
        2.569032073,
        2.727774717,
        2.849705382,
        2.948319908,
        3.030878867,
        3.10173026,
        3.16368342,
        3.218653901,
        3.268003591,
        3.312738701,
        3.353617959,
        3.391230382,
        3.426041249,
        3.458424619,
        3.488684546,
        3.517072762,
        3.543799277,
        3.569040161,
        3.592946027,
        3.615646276,
        3.637252631,
        3.657860551,
        3.677556303,
        3.696413427,
        3.71449839,
        3.731869175,
        3.748578108,
        3.764671858,
        3.780192852,
        3.795178566,
        3.809663649,
        3.823679212,
        3.837254248,
        3.850413505,
        3.863181025,
        3.875578729,
        3.887627121,
        3.899344587,
        3.910747391,
        3.921852503,
        3.932673359,
        3.943224099,
        3.953518159,
        3.963566147,
        3.973379375,
        3.98296845,
        3.992343271,
        4.001512325,
        4.010484803,
        4.019267776,
        4.02786973,
        4.036297029,
        4.044556036,
        4.05265453,
        4.060596753,
        4.068389777,
        4.076037844,
        4.083547318,
        4.090921028,
        4.098166044,
        4.105284488,
        4.112282016,
        4.119161458,
        4.125927056,
        4.132582345,
        4.139131568,
        4.145576139,
        4.151921008,
        4.158168297,
        4.164320833,
        4.170380738,
        4.176352255,
        4.182236797,
        4.188036487,
        4.19375486,
        4.199392622,
        4.204952603,
        4.21043763,
        4.215848411,
        4.221187067,
        4.22645572,
        4.23165649,
        4.236790793,
        4.241859334,
        4.246864943,
        4.251809034,
        4.256692313,
        4.261516196,
        4.266282802,
        4.270992841,
        4.275648432,
        4.280249575,
        4.284798393,
        4.289294885,
        4.29374188,
        4.298139377,
        4.302488791,
    ]
    return alpha005[numModels] * math.sqrt(
        (numModels * (numModels + 1)) / (6 * numDatasets)
    )


def nemenyi_plot(
    df: pd.DataFrame, output_file: str, nemenyi_width: int, nemenyi_textspace: int
):
    # nemenyi test
    pivoted = df.pivot(
        index="task_id", columns="hyperparameter", values="importance_variance"
    )
    avg_ranks = (
        pivoted.rank(axis=1, method="average", ascending=False).sum(axis=0)
        / pivoted.shape[0]
    )

    cd = critical_dist(pivoted.shape[1], pivoted.shape[0])

    # print some statistics, for sanity checking
    logging.info("cd = %f, %s" % (cd, avg_ranks.to_dict()))

    # and plot
    Orange.evaluation.scoring.graph_ranks(
        list(avg_ranks.to_dict().values()),
        list(avg_ranks.to_dict().keys()),
        cd=cd,
        filename=output_file,
        width=nemenyi_width,
        textspace=nemenyi_textspace,
    )
    logging.info("stored figure to %s" % output_file)


def boxplots_variance_contrib(
    df: pd.DataFrame, output_file: str, n_combi_params: int, log_scale: bool
):
    medians = df.groupby("hyperparameter")[
        ["n_hyperparameters", "importance_variance", "importance_max_min"]
    ].median()
    df = df.join(medians, on="hyperparameter", how="left", rsuffix="_median")

    # vanilla boxplots
    plt.clf()
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 10))
    cutoff_value_variance = (
        calculate_cutoff_value(medians, "importance_variance", n_combi_params)
        - 0.000001
        if n_combi_params > 0
        else 1.0
    )
    df = df.query(
        "n_hyperparameters == 1 or importance_variance_median >= %f"
        % cutoff_value_variance
    ).sort_values("importance_variance_median")
    ticks = list(dict.fromkeys(df["hyperparameter"].values))

    # sns.boxplot(
    #     x="hyperparameter",
    #     y="importance_variance",
    #     data=df.query(
    #         "n_hyperparameters == 1 or importance_variance_median >= %f"
    #         % cutoff_value_variance
    #     ).sort_values("importance_variance_median"),
    #     ax=ax1,
    # )

    sns.lineplot(
        x="hyperparameter",
        y="importance_variance",
        data=df,
        hue="task_name",
        legend=True,
    )
    ax1.set_xticklabels(ticks, rotation=45, ha="right")
    ax1.set_ylabel("Variance Contribution")
    ax1.set_xlabel(None)
    if log_scale:
        ax1.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_file)
    logging.info("stored figure to %s" % output_file)


def boxplots_minmax(
    df: pd.DataFrame, output_file: str, n_combi_params: int, log_scale: bool
):
    medians = df.groupby("hyperparameter")[
        ["n_hyperparameters", "importance_variance", "importance_max_min"]
    ].median()
    df = df.join(medians, on="hyperparameter", how="left", rsuffix="_median")

    plt.clf()
    fig, ax2 = plt.subplots(1, 1, figsize=(8, 10))
    cutoff_value_max_min = (
        calculate_cutoff_value(medians, "importance_max_min", n_combi_params) - 0.000001
        if n_combi_params > 0
        else 100.0
    )
    df = df.query(
        "n_hyperparameters == 1 or importance_variance_median >= %f"
        % cutoff_value_max_min
    ).sort_values("importance_max_min_median")
    ticks = list(dict.fromkeys(df["hyperparameter"].values))
    # sns.boxplot(
    #     x="hyperparameter",
    #     y="importance_max_min",
    #     data=df.query(
    #         "n_hyperparameters == 1 or importance_max_min_median >= %f"
    #         % cutoff_value_max_min
    #     ).sort_values("importance_max_min_median"),
    #     ax=ax2,
    # )
    sns.lineplot(
        x="hyperparameter",
        y="importance_max_min",
        data=df,
        hue="task_name",
        legend=True,
    )
    ax2.set_xticklabels(ticks, rotation=45, ha="right")
    ax2.set_ylabel("max(marginal) - min(marginal)")
    ax2.set_xlabel(None)
    if log_scale:
        ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_file)
    logging.info("stored figure to %s" % output_file)


def best_per_dataset_scatter(df: pd.DataFrame):
    # best per data feature
    # qualities = pd.DataFrame.from_dict(openml.tasks.list_tasks(task_id=df['task_id'].unique()), orient='index')
    # qualities = qualities[['tid', 'NumberOfInstances', 'NumberOfFeatures']]
    # logging.info('stored figure to %s' % output_file)
    pass


def calculate_cutoff_value(
    medians: pd.DataFrame, column_name: str, n_combi_params: typing.Optional[int]
):
    medians_sorted = medians[medians["n_hyperparameters"] > 1].sort_values(column_name)
    cutoff = 0.0
    if n_combi_params is not None and len(medians_sorted) > n_combi_params:
        cutoff = medians_sorted[column_name][-1 * n_combi_params]
    return cutoff


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # and do the plotting
    matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
    params = {
        "text.usetex": False,
        "font.size": args.font_size,
        "font.family": "lmodern",
    }
    matplotlib.rcParams.update(params)
    os.makedirs(args.output_directory, exist_ok=True)

    df = pd.read_csv(args.fanova_result_file)
    df["hyperparameter"] = df["hyperparameter"].apply(lambda x: x.replace("_", " "))
    qualities = pd.DataFrame.from_dict(
        openml.tasks.list_tasks(task_id=df["task_id"].unique()), orient="index"
    )
    qualities["task_name"] = qualities["tid"].apply(
        lambda x: openml.tasks.get_task(x).get_dataset().name
    )
    qualities = qualities[["tid", "task_name", "NumberOfInstances", "NumberOfFeatures"]]
    qualities["task_id"] = qualities["tid"]
    del qualities["tid"]
    df = df.join(qualities, on="task_id", how="left", rsuffix="_qualities")

    output_file_variance = os.path.join(
        args.output_directory,
        "linefanova_variancecontrib%s.%s"
        % (
            "_log" if args.log_scale else "",
            args.plot_extension,
        ),
    )
    output_file_maxmin = os.path.join(
        args.output_directory,
        "linefanova_maxmin%s.%s"
        % (
            "_log" if args.log_scale else "",
            args.plot_extension,
        ),
    )
    boxplots_variance_contrib(
        df, output_file_variance, args.n_combi_params, args.log_scale
    )
    boxplots_minmax(df, output_file_maxmin, args.n_combi_params, args.log_scale)


if __name__ == "__main__":
    run(read_cmd())
