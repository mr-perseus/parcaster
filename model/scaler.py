import joblib


class Scaler:
    def __init__(self, scaler):
        self.scaler = scaler

    def scale(self, train, val, test):
        train_scaled = self.scaler.fit_transform(train)
        val_scaled = self.scaler.transform(val)
        test_scaled = self.scaler.transform(test)

        return train_scaled, val_scaled, test_scaled

    def fit_transform(self, data):
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def save(self, path):
        joblib.dump(self.scaler, path)

    @classmethod
    def load(cls, path):
        scaler = joblib.load(path)
        return cls(scaler)
