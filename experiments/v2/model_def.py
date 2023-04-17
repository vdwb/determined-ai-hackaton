"""
This example demonstrates how to run TensorFlow's Boosted Trees Estimator. Due to the nature of the
model, this example is meant to run as a single-GPU model or a hyperparameter search; it does NOT
support distributed training.

Example based on this tutorial:
    https://www.tensorflow.org/tutorials/estimator/boosted_trees

"""

from typing import Callable, Dict, List, Tuple

import pandas as pd
import tensorflow as tf
import numpy as np

from determined.estimator import EstimatorTrial, EstimatorTrialContext


class BoostedTreesTrial(EstimatorTrial):
    def __init__(self, context: EstimatorTrialContext) -> None:
        self.context = context

        # Load Dataset.
        (
            self.dftrain,
            self.dfeval,
            self.y_train,
            self.y_eval,
            self.feature_columns,
        ) = self.load_dataset()

        # Wrap Optimizer (required by Determined but not used by this specific model).
        self.context.wrap_optimizer(None)

        # Set Hyperparameters - this is being populated at runtime from the .yaml configuration file.
        self.n_trees = context.get_hparam("n_trees")
        self.max_depth = context.get_hparam("max_depth")
        self.learning_rate = context.get_hparam("learning_rate")
        self.l1_regularization = context.get_hparam("l1_regularization")
        self.l2_regularization = context.get_hparam("l2_regularization")
        self.min_node_weight = context.get_hparam("min_node_weight")

    def build_estimator(self) -> tf.estimator.Estimator:
        # Since data fits into memory, use entire dataset per layer.
        n_batches = 1

        est = tf.estimator.BoostedTreesClassifier(
            self.feature_columns,
            n_batches_per_layer=n_batches,
            n_trees=self.n_trees,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            l1_regularization=self.l1_regularization,
            l2_regularization=self.l2_regularization,
            min_node_weight=self.min_node_weight,
        )

        return est

    def make_input_fn(self, X, y, shuffle=True):
        def input_fn():
            NUM_EXAMPLES = len(y)

            dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
            dataset = self.context.wrap_dataset(dataset)

            if shuffle:
                dataset = dataset.shuffle(NUM_EXAMPLES)
            dataset = dataset.repeat(1)
            dataset = dataset.batch(NUM_EXAMPLES)

            return dataset

        return input_fn

    def build_train_spec(self) -> tf.estimator.TrainSpec:
        return tf.estimator.TrainSpec(self.make_input_fn(self.dftrain, self.y_train, shuffle=True))

    def build_validation_spec(self) -> tf.estimator.EvalSpec:
        return tf.estimator.EvalSpec(
            self.make_input_fn(self.dfeval, self.y_eval, shuffle=False), steps=None
        )

    def load_dataset(self):

        dftrain = pd.read_csv(self.context.get_data_config()["truck_dataset"]["train"], delimiter=";", on_bad_lines='skip')
        dfeval = pd.read_csv(self.context.get_data_config()["truck_dataset"]["eval"], delimiter=";", on_bad_lines='skip')
        print(dftrain.columns)
        dftrain.info()
        #dftrain.columns.tolist()
        #dfeval.columns.tolist()
        #dftrain.info
        #dftrain = dftrain.pop("Planned_ETA")
        #dftrain = dftrain.pop("trip_start_date")
        #dfeval = dfeval.pop("Planned_ETA")
        #dfeval = dfeval.pop("trip_start_date")
        dftrain["Driver_Name"] = dftrain["Driver_Name"].fillna('unknown')
        dfeval["Driver_Name"] = dfeval["Driver_Name"].fillna('unknown')

        dftrain["vehicleType"] = dftrain["vehicleType"].fillna('mystery')
        dfeval["vehicleType"] = dfeval["vehicleType"].fillna('mystery')

        y_train = dftrain.pop("ontime")
        y_eval = dfeval.pop("ontime")
        #dftrain = np.array(dftrain)
        #dfeval = np.array(dfeval)
        #y_train = np.array(y_train)
        #y_eval = np.array(y_eval)
        #dftrain = tf.convert_to_tensor(dftrain)
        #dfeval = tf.convert_to_tensor(dfeval)
        #y_train = tf.convert_to_tensor(y_train)
        #y_eval = tf.convert_to_tensor(y_eval)

        CATEGORICAL_COLUMNS = [
            "Market/Regular",
            "Origin_Location",
            "Destination_Location",
            "vehicleType",
            "customerID",
            "supplierID",
            "MaterialShipped",
            "Driver_Name",
        ]
        NUMERIC_COLUMNS = ["TRANSPORTATION_DISTANCE_IN_KM", "expected_travelhours"]

        def one_hot_cat_column(feature_name, vocab):
            return tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab)
            )

        feature_columns = []

        for feature_name in CATEGORICAL_COLUMNS:
            # Need to one-hot encode categorical features.
            vocabulary = dftrain[feature_name].unique()
            feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

        for feature_name in NUMERIC_COLUMNS:
            feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

        #for feature_name in DATE_COLUMNS:
            #feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.int64) )

        return dftrain, dfeval, y_train, y_eval, feature_columns
