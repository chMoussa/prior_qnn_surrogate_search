import pandas as pd

from typing import List

import ConfigSpace
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    Constant,
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    NumericalHyperparameter,
)


def integer_encode_dataframe(
        df: pd.DataFrame, config_space: ConfigSpace.ConfigurationSpace
) -> pd.DataFrame:
    for column_name in df.columns.values:
        if column_name in config_space.get_hyperparameter_names():
            hyperparameter = config_space.get_hyperparameter(column_name)
            if isinstance(
                    hyperparameter, ConfigSpace.hyperparameters.NumericalHyperparameter
            ):
                # numeric hyperparameter, don't do anything
                pass
            elif isinstance(
                    hyperparameter,
                    (
                            ConfigSpace.hyperparameters.Constant,
                            ConfigSpace.hyperparameters.UnParametrizedHyperparameter,
                    ),
            ):
                # encode as constant value. can be retrieved from config space later
                df[column_name] = 0
                df[column_name] = pd.to_numeric(df[column_name])
            elif isinstance(
                    hyperparameter, ConfigSpace.hyperparameters.CategoricalHyperparameter
            ):
                # added the if else statement to make the function compatible with get_unimp_hyperparam_configspace()
                if df[column_name].dtypes == int:
                    pass
                else:
                    df[column_name] = df[column_name].apply(
                    lambda x: hyperparameter.choices.index(x)
                    )
                    df[column_name] = pd.to_numeric(df[column_name])
            else:
                raise NotImplementedError(
                    "Function not implemented for "
                    "Hyperparameter: %s" % type(hyperparameter)
                )
    return df


def get_configspace(exclude_epochs: bool = False):
    cs = ConfigurationSpace()

    depth = UniformIntegerHyperparameter(
        name="depth", lower=1, upper=10, default_value=1
    )

    is_data_encoding_hardware_efficient = CategoricalHyperparameter(
        name="is_data_encoding_hardware_efficient",
        choices=[True, False],
        default_value=True,
    )

    have_less_rotations = CategoricalHyperparameter(
        name="have_less_rotations", choices=[True, False], default_value=True
    )

    map_type = CategoricalHyperparameter(
        name="map_type", choices=["ring", "full", "pairs"], default_value="ring"
    )
    entangler_operation = CategoricalHyperparameter(
        name="entangler_operation", choices=["cz", "sqiswap"], default_value="cz"
    )

    use_reuploading = CategoricalHyperparameter(
        name="use_reuploading", choices=[True, False], default_value=False
    )

    output_circuit = CategoricalHyperparameter(
        name="output_circuit",
        choices=["2Z", "mZ"],
        default_value="2Z",
    )

    input_activation_function = CategoricalHyperparameter(
        name="input_activation_function",
        choices=["linear", "tanh"],
        default_value="linear",
    )

    learning_rate = UniformFloatHyperparameter(
        name="learning_rate",
        lower=0.0001,
        upper=0.5,
        default_value=0.0001,
        log=True,
    )

    batchsize = CategoricalHyperparameter(
        name="batchsize", choices=[16, 32, 64], default_value=32
    )

    epochs = UniformIntegerHyperparameter(
        name="epochs", lower=1, upper=100, default_value=100
    )
    list_params = [
        batchsize,
        depth,
        entangler_operation,
        have_less_rotations,
        input_activation_function,
        is_data_encoding_hardware_efficient,
        learning_rate,
        map_type,
        output_circuit,
        use_reuploading,
    ]

    if not exclude_epochs:
        list_params.append(epochs)
    cs.add_hyperparameters(
        list_params
    )
    return cs


def get_unimp_hyperparam_configspace(important_hyperparam_indices: List):
    cs = ConfigurationSpace()

    depth = UniformIntegerHyperparameter(
        name="depth", lower=1, upper=10, default_value=1
    )

    is_data_encoding_hardware_efficient = CategoricalHyperparameter(
        name="is_data_encoding_hardware_efficient",
        choices=[True, False],
        default_value=True,
    )

    have_less_rotations = CategoricalHyperparameter(
        name="have_less_rotations", choices=[True, False], default_value=True
    )

    map_type = CategoricalHyperparameter(
        name="map_type", choices=["ring", "full", "pairs"], default_value="ring"
    )
    entangler_operation = CategoricalHyperparameter(
        name="entangler_operation", choices=["cz", "sqiswap"], default_value="cz"
    )

    use_reuploading = CategoricalHyperparameter(
        name="use_reuploading", choices=[True, False], default_value=False
    )

    output_circuit = CategoricalHyperparameter(
        name="output_circuit",
        choices=["2Z", "mZ"],
        default_value="2Z",
    )

    input_activation_function = CategoricalHyperparameter(
        name="input_activation_function",
        choices=["linear", "tanh"],
        default_value="linear",
    )

    learning_rate = UniformFloatHyperparameter(
        name="learning_rate",
        lower=0.0001,
        upper=0.5,
        default_value=0.0001,
        log=True,
    )

    batchsize = CategoricalHyperparameter(
        name="batchsize", choices=[16, 32, 64], default_value=32
    )

    list_hparams = [
        batchsize,
        depth,
        entangler_operation,
        have_less_rotations,
        input_activation_function,
        is_data_encoding_hardware_efficient,
        learning_rate,
        map_type,
        output_circuit,
        use_reuploading,
    ]
    
    # reverse is True in order to delete hyperparameters if an important_hyperparam_indices is not sorted
    # this is crucial because otherwise some other index might be deleted from what was intended.
    for h_i in sorted(important_hyperparam_indices, reverse=True):
        del list_hparams[h_i]
    cs.add_hyperparameters(list_hparams)

    return cs
