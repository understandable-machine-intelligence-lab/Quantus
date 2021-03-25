from abc import ABC, abstractmethod


class DataSample(ABC):
    """ Represents a single sample of a dataset. """

    def __init__(self, datum, filename):
        """ Initializes a new DataSample instance.
        Supported attributes:
            datum   - sample data
            filename    - key
            label   - list of class labels
            one_hot_label   - one hot encoded class labels
            binary_mask     - dict of type label -> binary mask
        """
        self.datum = datum
        self.filename = filename


class Dataset(ABC):
    """ Abstract Interface for custom datasets.
    -----------
    Attributes
    -----------
    datapath: str
        filepath to the dataset
    partition: str
        one of the options train or val
    samples: list
        list of Datasamples of type DataSample
    mode: str
        mode of the dataset to control which attributes of DataSample need to be prepared
    cmap:   list or dict
        information to determine a classname to index mapping or vice versa
    classes: list
        list of classnames to be used in this dataset object
    (optional?)
    labels/anns: list
        list of labels assigned to the samples
    """

    def __init__(self, datapath, partition):
        """ Initialize the model. """
        self.datapath = datapath

        assert partition in ["train", "val"]
        self.partition = partition
        self.samples = []
        self.mode = "raw"
        super().__init__()

    def __len__(self):
        """ Returns the length of the dataset as int. """
        return len(self.samples)

    @abstractmethod
    def __getitem__(self, index):
        """ Retrieves the element with the specified index. """
        return NotImplementedError

    def set_mode(self, mode):
        """ Set the mode of the dataset to determine return values"""
        if mode not in ["raw", "preprocessed", "binary_mask"]:
            raise ValueError(f"mode {mode} not in the set of valid options")

        self.mode = mode

    @abstractmethod
    def classname_to_idx(self, class_name):
        """ convert a classname to an index. """
        return NotImplementedError

    # @abstractmethod
    # def get_dataset_partition(self, startidx, endidx, batched=False):
    #     """ Retrieve a partition from the dataset ranging from startidx to endidx. """
    #     return NotImplementedError
    #
    # @abstractmethod
    # def preprocess_image(self, image):
    #     """ Preprocess a single image. """
    #     return NotImplementedError
    #
    # @abstractmethod
    # def preprocess_data(self, data, labels):
    #     """ Preprocess the presented data. """
    #     preprocessed = [self.preprocess_image(image, label) for image, label in np.column_stack(data, labels)]
    #
    #     return preprocessed[:, 0], preprocessed[:, 1]
