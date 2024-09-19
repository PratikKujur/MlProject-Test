# Data Transformation Class
from sklearn.preprocessing import StandardScaler

class DataTransformation:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def transform(self):
        array = self.dataframe.values
        X = array[:, 0:8]
        Y = array[:, 8]

        # Feature scaling
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, Y