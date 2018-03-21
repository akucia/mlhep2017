import pandas as pd
import pathlib

from sklearn.pipeline import make_pipeline, Pipeline

from ml_hep_tracking.models import SKLearnForest, LGBM, DeepNeuralNet
from ml_hep_tracking.preprocessing import data_pipeline
from ml_hep_tracking.utils import get_data
from sklearn.model_selection import train_test_split, KFold, cross_validate
import logging
from pprint import pprint

logging.basicConfig(format='%(asctime)s:[%(name)s]:%(levelname)s:%(message)s', level=logging.DEBUG)

logging.info("Starting app")
DATASET_PATH = pathlib.Path('data_mlhep2017')

train_df, test_df = get_data(DATASET_PATH)
pprint(train_df.head())
X = train_df.drop(['signal'], axis=1)
y = train_df['signal']

kf = KFold(n_splits=5, shuffle=False)

classifier = DeepNeuralNet((X.shape[1], ), (1,), layers=1)
classifier = Pipeline(data_pipeline()+[
    ('classifier', classifier)
])
scoring = ['roc_auc', 'accuracy', 'neg_log_loss']

scores = cross_validate(
    classifier,
    X, y,
    scoring=scoring,
    cv=3,
    return_train_score=True,
    verbose=3,
    n_jobs=1
)

pprint(scores)

logging.info(scores)