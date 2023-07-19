import os
import json

from PIL import Image
import torch.utils.data as tdata

class CustomDataset(tdata.Dataset):

    def __init__(self, root, transform, **kwargs):
        """
        Dataset for downloaded Data.
        Expected folder layout is
            root/mode/readable-class-name/sample.*
        """

        # Stores the arguments for later use
        self.transform = transform

        self.mode = kwargs.get("mode", "train")
        self.classes = kwargs.get("classes", "all")
        self.accepted_filetypes = kwargs.get("accepted_filetypes", ["png", "jpeg", "jpg"])
        self.labelmap_path = kwargs.get("labelmap_path", None)

        assert isinstance(self.mode, str)
        assert self.labelmap_path is not None

        self.root = os.path.join(root, self.mode)

        # Loads the label map into memory
        with open(self.labelmap_path) as label_map_file:
            # label map should map from class name to class idx
            self.label_map = json.load(label_map_file)

        # create list of (sample, label) tuples.
        self.samples = []

        print("DATA_ROOT", self.root)

        # Build samples with correct classes first
        for cl in self.label_map.keys():
            cl_dir = os.path.join(self.root, cl)
            if os.path.exists(cl_dir) and self.classes == "all" or cl in self.classes:
                for fname in [f for f in os.listdir(cl_dir) if os.path.isfile(os.path.join(cl_dir, f))]:
                    if fname.split(".")[-1].lower() in self.accepted_filetypes:
                        self.samples.append((os.path.join(cl_dir, fname), self.label_map[cl]["label"]))


    def __len__(self):
        """
        Retrieves the number of samples in the dataset.

        Returns
        -------
            int
                Returns the number of samples in the dataset.
        """

        return len(self.samples)

    def __getitem__(self, index):
        """
        Retrieves the element with the specified index.

        Parameters
        ----------
            index: int
                The index of the element that is to be retrieved.

        Returns
        -------
            Sample
                Returns the sample with the specified index
        """

        # Gets the path name and the WordNet ID of the sample
        file, label = self.samples[index]

        # Loads the image from file
        sample = Image.open(file)
        sample = sample.convert('RGB')

        # If the user specified a transform for the samples, then it is applied
        sample = self.transform(sample)

        # Returns the sample
        return sample, label
