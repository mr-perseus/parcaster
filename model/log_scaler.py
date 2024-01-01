import numpy as np
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline


def log_transform(x):
    return np.log1p(x)


def inverse_log_transform(x):
    return np.expm1(x)


class LogScaler:
    def __init__(self):
        self.pipeline = Pipeline([
            ('log_transformer', FunctionTransformer(func=log_transform, inverse_func=inverse_log_transform)),
            ('scaler', MinMaxScaler()),
        ])

    def fit_transform(self, data):
        return self.pipeline.fit_transform(data)

    def transform(self, data):
        return self.pipeline.transform(data)

    def inverse_transform(self, data):
        print(f"data: {data}")
        inverted_data = self.pipeline.named_steps['scaler'].inverse_transform(data)
        inverted_data = self.pipeline.named_steps['log_transformer'].inverse_transform(inverted_data)
        print(f"inverted_data: {inverted_data}")

        return inverted_data


if __name__ == "__main__":
    logScaler = LogScaler()
    X = np.array([[1], [2], [3], [4], [5], [100], [200], [300]]).astype(float)
    X_2 = np.array([[300]]).astype(float)

    X_transformed = logScaler.fit_transform(X)
    X_2_transformed = logScaler.transform(X_2)

    print("Transformed Data X:", X_transformed)
    print("Transformed Data X_2:", X_2_transformed)

    X_transformed_back = logScaler.inverse_transform(X_transformed)
    X_2_transformed_back = logScaler.inverse_transform(X_2_transformed)

    print("Transformed back Data X:", X_transformed_back)
    print("Transformed back Data X_2:", X_2_transformed_back)
