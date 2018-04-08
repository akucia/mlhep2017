import logging
import pathlib
import random
from typing import Tuple, Generator

from keras.utils import Sequence

from keras.preprocessing.sequence import TimeseriesGenerator as KerasTimeSeriesGenerator
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

MAX_TRACKS = 2500

def get_data(
    dataset_path: pathlib.Path,
    seed: int = 137,
    test_size: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    logging.info(f"Reading data from {dataset_path}")
    train_df = pd.read_csv(dataset_path.as_posix())

    # train_df.drop(columns=['index', 'event_id'], inplace=True)
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


def load_events(dataset_path: pathlib.Path):
    data_df, _ = get_data(dataset_path, test_size=0)
    true_events = data_df[data_df['event_id'] != -999].copy()

    events = true_events.drop(['index'], axis=1)
    events.sort_values('event_id')
    events['track_id'] = events.groupby('event_id').cumcount()

    index = pd.MultiIndex.from_arrays([events['event_id'], events['track_id']])

    new_events = events.set_index(index)
    new_events = new_events.sort_index()
    new_events = new_events.drop(['event_id', 'track_id'], axis='columns')

    event_idx = pd.Index(events['event_id'].unique())
    track_idx = pd.RangeIndex(0, MAX_TRACKS)

    new_index = pd.MultiIndex.from_product([event_idx, track_idx])

    final_events = new_events.reindex(index=new_index)
    fake_events = data_df[data_df['event_id'] == -999][
        ['X', 'Y', 'Z', 'TX', 'TY', 'chi2', 'signal']].copy()

    fake_events = fake_events.sample(len(fake_events))

    wrong_inputs = fake_events.sample(
        len(np.isnan(final_events)), replace=False
    ).values

    new_full_values = np.where(
        np.isnan(final_events.values),
        wrong_inputs,
        final_events.values
    )

    new_values_df = pd.DataFrame(
        data=new_full_values,
        index=final_events.index,
        columns=final_events.columns
    )

    return new_values_df


class SequenceGenerator(Sequence):
    """Utility class for generating batches of temporal data.
        Similar to keras.preprocessing.sequence.TimeseriesGenerator

        This class takes in a sequence of data-points gathered at
        equal intervals, along with time series parameters such as
        stride, length of history, etc., to produce batches for
        training/validation.

        # Arguments
            data: Indexable generator (such as list or Numpy array)
                containing consecutive data points (timesteps).
                The data should be at 2D, and axis 0 is expected
                to be the time dimension.
            targets: Targets corresponding to timesteps in `data`.
                It should have same length as `data`.
            length: Length of the output sequences (in number of timesteps).
            sampling_rate: Period between successive individual timesteps
                within sequences. For rate `r`, timesteps
                `data[i]`, `data[i-r]`, ... `data[i - length]`
                are used for create a sample sequence.
            stride: Period between successive output sequences.
                For stride `s`, consecutive output samples would
                be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
            start_index, end_index: Data points earlier than `start_index`
                or later than `end_index` will not be used in the output sequences.
                This is seful to reserve part of the data for test or validation.
            shuffle: Whether to shuffle output samples,
                or instead draw them in chronological order.
            reverse: Boolean: if `true`, timesteps in each output sample will be
                in reverse chronological order.
            batch_size: Number of timeseries samples in each batch
                (except maybe the last one).
    """

    def __init__(self, data, targets, length,
                 sampling_rate=1,
                 stride=1,
                 start_index=0,
                 end_index=None,
                 shuffle=False,
                 reverse=False,
                 batch_size=128,
                 verbose=False
                 ):

        self.data = data
        self.targets = targets
        self.length = length
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.start_index = start_index
        if end_index is None:
            end_index = len(data) - 1
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size
        self.verbose = verbose

    def __len__(self):
        return int(np.ceil(
            (self.end_index - self.start_index) /
            (self.batch_size * self.stride)))

    def _empty_batch(self, num_rows):
        samples_shape = [num_rows, self.length // self.sampling_rate]
        samples_shape.extend(self.data.shape[1:])
        targets_shape = [num_rows, self.length // self.sampling_rate]
        targets_shape.extend(self.targets.shape[1:])
        return np.empty(samples_shape), np.empty(targets_shape)

    def __getitem__(self, index):
        i = self.start_index + self.batch_size * self.stride * index
        if self.verbose:
            logging.debug(f"i: {i}")
            logging.debug(
                f"{i, (i + self.batch_size * self.stride, self.end_index), self.stride}")
        rows = np.arange(i, min(i + self.batch_size *
                                self.stride, self.end_index), self.stride)
        if self.verbose:
            logging.debug(f"rows: {rows}")

        samples, targets = self._empty_batch(len(rows))
        for j, row in enumerate(rows):
            indices = np.arange(rows[j], rows[j] + self.length, self.sampling_rate)
            if self.shuffle:
                # indices = (indices)
                np.random.shuffle(indices)
            if self.verbose:
                logging.debug(f"indices: {indices}")
            samples[j] = self.data[indices]
            targets[j] = self.targets[indices]
        if self.reverse:
            return samples[:, ::-1, ...], targets
        return samples, targets

# todo def balance_dataset(x,y)

# todo def metrics_report(model, x,y)

# todo def history_plot(model)