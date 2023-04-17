# Truck deliverly service analysis
## objective
* Builld a model that predict ontime/delay
* Identify features that influece the predication


I use the dataset provided by RAM THAIAGU. Thanks!
Dataset downloaded from Kaggle: https://www.kaggle.com/datasets/ramakrishnanthiyagu/delivery-truck-trips-data <br>
Following notebook is used as starting point for this project: https://www.kaggle.com/code/yujiyamamoto/logistics-service-analysis

## Methodology

We first started with some data exploration in the jupyter notebook<br>
After which we cleaned the data in 3 different ways.<br>
For each cleaned dataset we did an inital classification training and evaluation using the classifier of XGBBoost library <br>
We also experimented with a library call 'TPOT' which is an autoML that optimizes machine learning pipelines using genetic programming. No better results were obtained. <br> <br>

Based on the evaluation of the 3 datasets using XGBClassifier we decided on our final dataset to further train and create experiments using Determined AI's platform. <br>

## Data Sample explanation

### Before cleaning
```code
Dataset:

GpsProvider - Vendor who provides GPS<br>
BookingID - Unique Identification for a trip<br>
Market/Regular - Type of trip. Regular - Vendors with whom we will have contract. Market - Vendor with whom we will not have contract<br>
BookingID_Date - Date when booking was created<br>
vehicle_no - Truck Number<br>
Origin_Location - Trip start place<br>
Destination_Location - Trip end place<br>
Org_lat_lon - Latitude/Longitude of start place<br>
Des_lat_lon - Latitude/Longitude of end place<br>
Data_Ping_time - Time when we receive GPS ping<br>
Planned_ETA - Planned Estimated Time of Arrival<br>
Current_Location - Live location<br>
DestinationLocation - Repeat of destination location<br>
actual_eta - Time when the truck arrived<br>
Curr_lat - current latitude - changes each time when we receive GPS ping<br>
Curr_lon - current longitude - changes each time when we receive GPS ping<br>
ontime - If the truck arrived on time - calculated based on Planned and Actual ETA<br>
delay - If the truck arrived with a delay - calculated based on Planned and Actual ETA<br>
OriginLocation_Code - Origin code<br>
DestinationLocation_Code - Destination code<br>
trip_start_date - Date/Time when trip started<br>
trip_end_date Date/Time when trip ended - based on documentation (cant be considered for calculating delay)\
TRANSPORTATION_DISTANCE_IN_KM - Total KM of travel<br>
vehicleType - Type of Truck<br>
Minimum_kms_to_be_covered_in_a_day - Minimum KM the driver needs to cover in a day<br>
Driver_Name - Driver details<br>
Driver_MobileNo - Driver details<br>
customerID - Customer details<br>
customerNameCode - Customer details<br>
supplierID - Supplier - Who provides the vehicle<br>
supplierNameCode - Supplier - Who provides the vehicle<br>
Material_Shipped - Material shipped by vehicle
```

### After cleaning

```code
Market/Regular - Type of trip. Regular - Vendors with whom we will have contract. Market - Vendor with whom we will not have contract<br>
Origin_Location - Trip start place<br>
Destination_Location - Trip end place<br>
TRANSPORTATION_DISTANCE_IN_KM - Total KM of travel<br>
vehicleType - Type of Truck<br>
Driver_Name - Driver details<br>
customerID - Customer details<br>
supplierID - Supplier - Who provides the vehicle<br>
Material_Shipped - Material shipped by vehicle<br>
expected_travelhours - difference of planned ETA and start date<br>
ontime - If the truck arrived on time - calculated based on Planned and Actual ETA<br>
```

## Model Architecture

We used a decision tree model. 
We configured the first training on our dataset with adaptive ASHA.
Underneath you can find our hyperparameter and configuration definition.

### model_def.py
Model_def.py contains the model definition and data processing.
```code
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
```
### adaptive.yaml
The file adaptive.yaml was used to define the configuration for hyperparameter search
```code
name: boosted_trees_estimator_adaptive_search
data:
  truck_dataset:
    train: "tensordataTrain.csv"
    eval: "tensordataTest.csv"
hyperparameters:
  n_trees:
    type: int
    minval: 100
    maxval: 1000
  max_depth:
    type: int
    minval: 5
    maxval: 20
  learning_rate:
    type: log
    base: 10
    minval: -4.0
    maxval: -2.0
  l1_regularization:
    type: log
    base: 10
    minval: -4.0
    maxval: -2.0
  l2_regularization:
    type: log
    base: 10
    minval: -4.0
    maxval: -2.0
  min_node_weight:
    type: double
    minval: 0.0
    maxval: 0.5
  global_batch_size: 8
searcher:
  name: adaptive_asha
  metric: accuracy
  smaller_is_better: false
  max_length:
    batches: 500
  max_trials: 100
entrypoint: model_def:BoostedTreesTrial
scheduling_unit: 1
environment:
  image: "determinedai/environments:py-3.8-pytorch-1.12-tf-2.8-cpu-0.21.0"
```

## Version 1
### const.yaml
The file const.yaml was used to define the configuration for running the model
```code
name: boosted_trees_estimator_const_truck_v1
data:
  truck_dataset:
    train: "tensordataTrain.csv"
    eval: "tensordataTest.csv"
hyperparameters:
  n_trees: 200
  max_depth: 10
  learning_rate: 0.01
  l1_regularization: 0.01
  l2_regularization: 0.01
  min_node_weight: 0.1
  global_batch_size: 8
searcher:
  name: single
  metric: accuracy
  max_length:
    batches: 100
  smaller_is_better: false
entrypoint: model_def:BoostedTreesTrial
scheduling_unit: 1
environment:
  image: "determinedai/environments:py-3.8-pytorch-1.12-tf-2.8-cpu-0.21.0"
```

## Version 2
### const.yaml
The file const.yaml was used to define the configuration for running the model
```code
name: boosted_trees_estimator_const_truck_v2
data:
  truck_dataset:
    train: "tensordataTrain.csv"
    eval: "tensordataTest.csv"
hyperparameters:
  n_trees: 653
  max_depth: 11
  learning_rate: 0.0003368435590495161
  l1_regularization: 0.0005158795301189985
  l2_regularization: 0.0013997404534321738
  min_node_weight: 0.0003359855025574565
  global_batch_size: 8
searcher:
  name: single
  metric: accuracy
  max_length:
    batches: 500
  smaller_is_better: false
entrypoint: model_def:BoostedTreesTrial
scheduling_unit: 1
environment:
  image: "determinedai/environments:py-3.8-pytorch-1.12-tf-2.8-cpu-0.21.0"
```

## Metrics and evaluation
  
  ![screenshot of best metrics](metrics.png "screenshot of best metrics")
  
<br> The searcher was configured to improve the accurary metric. Also loss and precision was taken into account choosing the best model.

## Reproduce results

1. Download the zip of the git repo
2. Unpack folder
3. Go via the cli to the directory determined-ai-hackaton-main -> experiments -> v1 OR v2
```code
\determined-ai-hackaton-main\experiments\v1
OR
\determined-ai-hackaton-main\experiments\v2
```
4. Execute this command to run the original experiment (v1) or the optimised version (v2) !Note! Set determined master ip if you don't intent to run this locally.
```code
#export DET_MASTER="example.org:8888"

det e create -f const.yaml .
```
5. To run the adaptive hyperparameter search run this command in either the v1 or v2 directory
```code
det e create -f adaptive.yaml .
```
You can find these results in our cluster:
```code
https://cluster-ihaceat4.org-yz40t65c.det-cloud.com/
```