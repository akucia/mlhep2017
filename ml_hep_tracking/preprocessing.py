from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class FeatureTransform(BaseEstimator, TransformerMixin):

    def __init__(self, input_feature: str, output_feature: Optional[str] = None,
                 transformation: Callable = None):
        self.input_feature = input_feature
        self.output_feature = self.input_feature
        self.transformation = transformation
        if output_feature is not None:
            self.output_feature = output_feature
        self.scaler = StandardScaler

    def _apply_transformaton(self, X: pd.DataFrame) -> np.ndarray:
        if self.transformation is not None:
            data = self.transformation(X[self.input_feature].values)
        else:
            data = X[self.input_feature].values
        return data.reshape(-1, 1)

    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureTransform':
        X = X.copy()
        self.scaler = self.scaler()
        data = self._apply_transformaton(X)
        self.scaler.fit(data)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        data = self._apply_transformaton(X)
        X[self.output_feature] = self.scaler.transform(data)
        return X


def data_pipeline():
    return [
        ('Y', FeatureTransform('Y')),
        ('X', FeatureTransform('X')),
        ('Z', FeatureTransform('Z')),
        ('TY', FeatureTransform('TY')),
        ('TX', FeatureTransform('TX')),
        ('chi2', FeatureTransform('chi2', transformation=np.sqrt)),
    ]
