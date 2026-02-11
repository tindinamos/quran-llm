# A terrible splitter because it doesn't shuffle the data, but it works for now
# The Quran has shorter verses at the end, so leaving validation for the last verses isn't going
# to verify that the model learned the big ones, even though it's trained on them.
class UniformDataSplitter:
    def __init__(self, data, split_ratio=0.9):
        self.data = data
        self.split_ratio = split_ratio

    def split(self):
        split_index = int(len(self.data) * self.split_ratio)
        train_data = self.data[:split_index]
        val_data = self.data[split_index:]
        return train_data, val_data
