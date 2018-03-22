import logging
import pathlib
from typing import Tuple, Generator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def get_data(
    dataset_path: pathlib.Path,
    seed: int = 137,
    test_size: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    logging.info(f"Reading data from {dataset_path}")
    train_df = pd.read_csv(dataset_path.as_posix())

    train_df.drop(columns=['index', 'event_id'], inplace=True)
    train_df = train_df.astype({'signal': int})
    train_df, test_df = train_test_split(
        train_df,
        test_size=test_size,
        random_state=seed
    )
    logging.debug(f"Train set: {train_df.shape}  Test set: {test_df.shape}")
    logging.debug(f"Train columns: {train_df.columns.values}")
    return train_df, test_df


def k_folds(data: pd.DataFrame, y_column='signal', n_splits=3, shuffle=False) \
     -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    kf = KFold(n_splits=n_splits, shuffle=shuffle)

    for train_index, test_index in kf.split(data):
        logging.info("TRAIN:", train_index, "TEST:", test_index)

        X: pd.DataFrame = data.drop([y_column], axis=1)
        y: pd.DataFrame = data[[y_column]]
        X_train, X_test = X.loc[train_index].values, X.loc[test_index].values
        y_train, y_test = y.loc[train_index].values, y.loc[test_index].values

        yield X_train, y_train, X_test, y_test


def shuffle_arrays(x, y):
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    return x[s], y[s]
