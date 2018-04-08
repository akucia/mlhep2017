import logging
import pathlib
import time

import keras
import numpy as np
import pandas as pd
from keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight

from ml_hep_tracking.models import DeepNeuralNet
from ml_hep_tracking.preprocessing import data_pipeline
from ml_hep_tracking.utils import load_events, MAX_TRACKS, \
    SequenceGenerator

N_FEATURES = 7
logging.basicConfig(format='%(asctime)s:[%(name)s]:%(levelname)s:%(message)s',
                    level=logging.DEBUG)

logging.info("Starting app")
DATASET_PATH = pathlib.Path('data_mlhep2017/DS_1_train.csv')

data_df = load_events(DATASET_PATH)

events_array = data_df.values.reshape((-1, MAX_TRACKS, N_FEATURES))

train, test = train_test_split(
    events_array,
    test_size=0.3,
    random_state=137
)

train_df = pd.DataFrame(train.reshape(-1, N_FEATURES), columns=data_df.columns)
test_df = pd.DataFrame(test.reshape(-1, N_FEATURES), columns=data_df.columns)


normalization_pipeline = Pipeline(data_pipeline())

train_data = normalization_pipeline.fit_transform(train_df)
test_data = normalization_pipeline.transform(test_df)

# print(train_data.head())

train_x = train_data.drop(['signal'], axis='columns').values
test_x = train_data.drop(['signal'], axis='columns').values
train_y = train_data[['signal']].values
test_y = train_data[['signal']].values
class_weight = class_weight.compute_class_weight('balanced', (np.unique(train_y.ravel())), train_y.ravel())

logging.info(f" signals in test set {np.mean(test_y)}")

batch_size = 10
train_series = SequenceGenerator(
    train_x, train_y,
    length=MAX_TRACKS,
    stride=MAX_TRACKS,
    batch_size=batch_size,
    shuffle=True
)
test_series = SequenceGenerator(
    test_x, test_y,
    length=MAX_TRACKS,
    stride=MAX_TRACKS,
    batch_size=batch_size,
    shuffle=True
)
#
# print(len(train_series), len(train_x) // MAX_TRACKS // batch_size)
# assert len(train_series) == len(train_x) // MAX_TRACKS // batch_size
# x, y = train_series[0]
# assert len(x) == batch_size
#
# assert x.shape == (batch_size, MAX_TRACKS, N_FEATURES - 1)
# # assert np.array_equal(train_x[:MAX_TRACKS], x[0])
#
# x, y = train_series[1]
# assert len(x) == batch_size
#
# assert x.shape == (batch_size, MAX_TRACKS, N_FEATURES - 1)


# assert np.array_equal(train_x[MAX_TRACKS*batch_size:MAX_TRACKS*batch_size+MAX_TRACKS], x[0])


class RNN(DeepNeuralNet):
    def __init__(self, input_shape, output_shape,
                 layers=3,
                 neurons=10,
                 bidirectional=None,
                 activation='relu',
                 recurrent_activation='hard_sigmoid',
                 loss_metric='binary_crossentropy',
                 optimizer='adam',
                 batch_norm=True,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 metrics='accuracy',
                 last_layer_act='sigmoid',
                 kernel_initializer='VarianceScaling',
                 **kwargs):

        self.bidirectional = bidirectional
        self.recurrent_dropout = recurrent_dropout
        self.recurrent_activation = recurrent_activation
        super().__init__(input_shape, output_shape,
                         layers,
                         neurons,
                         activation,
                         loss_metric,
                         optimizer,
                         batch_norm,
                         dropout,
                         metrics,
                         last_layer_act,
                         kernel_initializer,
                         **kwargs)

    def __call__(self):
        inp = keras.layers.Input(self.input_shape)

        layer = inp
        for i in range(self.layers):
            layer = keras.layers.GRU(
                self.neurons,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                recurrent_dropout=self.recurrent_dropout,
                recurrent_activation=self.recurrent_activation,
                return_sequences=True
            )(layer)
            # if self.batch_norm:
            #     layer = keras.layers.BatchNormalization()(layer)
            if self.bidirectional is not None:
                layer = Bidirectional(layer, merge_mode=self.bidirectional)
            if self.dropout > 0:
                layer = keras.layers.core.Dropout(self.dropout)(layer)

        # layer = keras.layers.Dense(
        #     self.output_shape[-1],
        #     activation=self.last_layer_act,
        #     kernel_initializer=self.kernel_initializer
        # )(layer)

        layer = keras.layers.GRU(
            self.output_shape[-1],
            activation=self.last_layer_act,
            kernel_initializer=self.kernel_initializer,
            recurrent_dropout=self.recurrent_dropout,
            recurrent_activation=self.recurrent_activation,
            return_sequences=True
        )(layer)

        model = keras.models.Model(inputs=[inp], outputs=[layer])
        model.compile(optimizer=self.optimizer, loss=self.loss_metric,
                      metrics=self.metrics)
        self.model = model
        return model


# time.ctime()
dnn_callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=f'./logs/DNN/{time.ctime()}',
        histogram_freq=0,
        write_graph=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1
    ),
]

DNNclf = DeepNeuralNet(
    (MAX_TRACKS, N_FEATURES - 1),
    (MAX_TRACKS, 1),
    neurons=32,
    layers=3,
    dropout=0,
    loss_metric='binary_crossentropy',
    metrics=['accuracy'],
    last_layer_act='sigmoid',
    kernel_initializer='he_normal',
    optimizer='adam',
    batch_norm=True,
    activation='relu',
)
#
model = DNNclf()
print(model.summary())
# model.fit_generator(train_series, epochs=100, callbacks=dnn_callbacks, validation_data=test_series)

rnn_callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=f'./logs/RNN/{time.ctime()}',
        histogram_freq=0,
        write_graph=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1
    ),
]

rnn_clf = RNN(
    (MAX_TRACKS, N_FEATURES - 1),
    (MAX_TRACKS, 1),
    neurons=10,
    layers=2,
    dropout=0,
    loss_metric='binary_crossentropy',
    metrics='accuracy',
    last_layer_act='sigmoid',
    kernel_initializer='he_normal',
    optimizer='adam',
    batch_norm=True,
    activation='tanh',
)

rnn_model = rnn_clf()
print(rnn_model.summary())
rnn_model.fit_generator(train_series, epochs=200, callbacks=rnn_callbacks, validation_data=test_series, class_weight=class_weight)
