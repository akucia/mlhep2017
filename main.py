import logging
import pathlib
import time
from pprint import pprint

import keras
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from ml_hep_tracking.models import DeepNeuralNet, LGBM
from ml_hep_tracking.preprocessing import data_pipeline
from ml_hep_tracking.utils import get_data, shuffle_arrays

logging.basicConfig(format='%(asctime)s:[%(name)s]:%(levelname)s:%(message)s',
                    level=logging.DEBUG)

logging.info("Starting app")
DATASET_PATH = pathlib.Path('data_mlhep2017/DS_1_train.csv')

train_df, test_df = get_data(DATASET_PATH)

pprint(train_df.head())
train_df.dropna(inplace=True)

rus = RandomUnderSampler(return_indices=True)

X = train_df.drop(['signal', 'index', 'event_id'], axis=1)
y = train_df[['signal']]

X_resampled, y_resampled, idx_resampled = rus.fit_sample(X, y)

X_resampled, y_resampled = shuffle_arrays(X_resampled, y_resampled)
X_resampled = pd.DataFrame(X_resampled, columns=X.columns)


classifier = DeepNeuralNet((X.shape[1],), (1,), layers=3, neurons=32)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, verbose=1
    ),
    keras.callbacks.TensorBoard(
        log_dir=f'./logs/Simple_DNN/{time.ctime()}',
        histogram_freq=0,
        write_graph=True,
    )
]

pipeline = Pipeline(data_pipeline() + [('classifier', classifier)])


scoring = ['roc_auc', 'accuracy', 'neg_log_loss']

scores_dnn = cross_validate(
    pipeline,
    X_resampled, y_resampled,
    scoring=scoring,
    cv=3,
    return_train_score=True,
    verbose=3,
    n_jobs=1,
    fit_params={
        'classifier__epochs': 50,
        'classifier__batch_size': 1000,
        'classifier__callbacks': callbacks,
        'classifier__validation_split': 0.25
    },
)

scores_lgbm = cross_validate(
    LGBM(),
    X_resampled, y_resampled,
    scoring=scoring,
    cv=3,
    return_train_score=True,
    verbose=3,
    n_jobs=1,
)

pprint(scores_dnn)
pprint(scores_lgbm)
