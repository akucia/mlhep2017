import pathlib
import pandas as pd
from typing import Tuple, Generator
import logging
import numpy as np

from sklearn.model_selection import KFold





def get_data(dataset_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info(f"Reading data from {dataset_path}")
    train_set = dataset_path / 'DS_1_train.csv'
    test_set = dataset_path / 'DS_1_test.csv'
    train_df = pd.read_csv(train_set.as_posix())
    test_df = pd.read_csv(test_set.as_posix())

    train_df.drop(columns=['index', 'event_id'], inplace=True)
    train_df = train_df.astype({'signal': int})

    logging.debug(f"Train set: {train_df.shape}  Test set: {test_df.shape}")
    logging.debug(f"Train columns: {list(train_df.columns)}")
    return train_df, test_df


def k_folds(data: pd.DataFrame, y_column='signal', n_splits=3, shuffle=False)\
     -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    kf = KFold(n_splits=n_splits, shuffle=shuffle)

    for train_index, test_index in kf.split(data):
        logging.info("TRAIN:", train_index, "TEST:", test_index)

        X: pd.DataFrame = data.drop([y_column], axis=1)
        y: pd.DataFrame = data[[y_column]]
        X_train, X_test = X.loc[train_index].values, X.loc[test_index].values
        y_train, y_test = y.loc[train_index].values, y.loc[test_index].values

        yield X_train, y_train, X_test, y_test
