import pandas as pd


class DataIngest:
    def __init__(self) -> None:
        self.data_path = None

    def load_data(self, data_path) -> pd.DataFrame:
        self.data_path = data_path
        df = pd.read_csv(self.data_path, encoding="ISO-8859-1")
        return df
