import pandas as pd

class DataIngestion:
    def __init__(self, url):
        self.url = url

    def load_data(self):
        names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
        dataframe = pd.read_csv(self.url, names=names)
        return dataframe