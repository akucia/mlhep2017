import keras
import lightgbm as lgbm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


class SKLearnForest(RandomForestClassifier):
    pass


class LGBM(lgbm.LGBMClassifier):
    pass


class DeepNeuralNet(BaseEstimator, KerasClassifier):
    def __init__(self, input_shape, output_shape,
                 layers=3,
                 neurons=10,
                 activation='relu',
                 loss_metric='binary_crossentropy',
                 optimizer='adam',
                 batch_norm=True,
                 dropout=0.0,
                 metrics='accuracy',
                 last_layer_act='sigmoid',
                 kernel_initializer='VarianceScaling',
                 **kwargs
                 ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers = layers
        self.neurons = neurons
        self.activation = activation
        self.loss_metric = loss_metric
        self.optimizer = optimizer
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.kernel_initializer = kernel_initializer
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.last_layer_act = last_layer_act

        super().__init__(**kwargs)

    def __call__(self):
        inp = keras.layers.Input(self.input_shape)

        layer = inp
        for i in range(self.layers):
            layer = keras.layers.Dense(
                self.neurons,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer
            )(layer)
            if self.batch_norm:
                layer = keras.layers.BatchNormalization()(layer)
            if self.dropout > 0:
                layer = keras.layers.core.Dropout(self.dropout)(layer)

        layer = keras.layers.Dense(
            self.output_shape[-1],
            activation=self.last_layer_act,
            kernel_initializer=self.kernel_initializer
        )(layer)
        model = keras.models.Model(inputs=[inp], outputs=[layer])
        model.compile(optimizer=self.optimizer, loss=self.loss_metric,
                      metrics=self.metrics)
        self.model = model
        return model

    def fit(self, x, y, **fit_params):
        return super().fit(x, y, **fit_params)
