import joblib


class Scaler:
    def __init__(self, scaler):
        self.scaler = scaler

    def scale(self, X_train, X_val, X_test, y_train, y_val, y_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        y_train_scaled = self.scaler.fit_transform(y_train)
        y_val_scaled = self.scaler.transform(y_val)
        y_test_scaled = self.scaler.transform(y_test)

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def save(self, path):
        joblib.dump(self.scaler, path)

    @classmethod
    def load(cls, path):
        scaler = joblib.load(path)
        return cls(scaler)
