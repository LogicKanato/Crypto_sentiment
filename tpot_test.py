import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8268265957030485
exported_pipeline = make_pipeline(
    make_union(
        make_union(
            FunctionTransformer(copy),
            FunctionTransformer(copy)
        ),
        StackingEstimator(estimator=MLPClassifier(activation="identity", alpha=0.01, hidden_layer_sizes=100, learning_rate="constant", max_iter=500, solver="lbfgs"))
    ),
    StackingEstimator(estimator=MLPClassifier(activation="tanh", alpha=0.0001, hidden_layer_sizes=150, learning_rate="invscaling", max_iter=500, solver="adam")),
    StackingEstimator(estimator=MLPClassifier(activation="tanh", alpha=0.1, hidden_layer_sizes=250, learning_rate="invscaling", max_iter=500, solver="lbfgs")),
    MLPClassifier(activation="relu", alpha=1e-05, hidden_layer_sizes=250, learning_rate="adaptive", max_iter=500, solver="sgd")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
