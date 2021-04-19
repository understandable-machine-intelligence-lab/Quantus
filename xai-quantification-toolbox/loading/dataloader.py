import numpy as np
import random


class DataLoader:
    """ Encapsulates the dataloading functionality.
    -----------
    Attributes
    -----------
    dataset: object of type Dataset
        Dataset object to apply dataloading to
    batch_size: int
        size of batches to split the dataset into, default: 32
    shuffle: bool
        indicator if the dataset should be shuffled before making batches
    startidx: int
        when set, the dataset is loaded from startidx to endidx
    endidx: int
        when set, the dataset is loaded from startidx to endidx
    """

    def __init__(self, dataset, batch_size=32, shuffle=False, startidx=0, endidx=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        if endidx > 0:
            assert (startidx < endidx)
            if endidx > len(self.dataset):
                self.idx = list(range(startidx, len(self.dataset)))
            else:
                self.idx = list(range(startidx, endidx))
        else:
            self.idx = list(range(len(self.dataset)))

    def __iter__(self):
        """ Iterator of the given dataset. """
        if self.shuffle:
            random.shuffle(self.idx)

        self.batches = [self.idx[i:i+self.batch_size] for i in range(0, len(self.idx), self.batch_size)]

        self.current_batch = 0

        return self

    def __next__(self):
        """ Provide next batch. """

        if self.current_batch >= len(self.batches):
            self.epoch_started = False
            raise StopIteration
        else:
            samples = []

            for i in self.batches[self.current_batch]:
                sample = self.dataset[i]
                samples.append(sample)

            self.current_batch += 1

            return np.array(samples)
