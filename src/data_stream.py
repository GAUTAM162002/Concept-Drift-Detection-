import pandas as pd

class DataStream:
    def __init__(self, file_path):
        try:
            self.data = pd.read_csv(file_path)
        except Exception as e:
            print("Error loading file:", e)
            self.data = pd.DataFrame()

        self.index = 0 

    def has_more_data(self):
        return self.index < len(self.data)

    def get_next_instance(self):
        if self.has_more_data():
            row = self.data.iloc[self.index]
            self.index += 1
            return row
        else:
            return None