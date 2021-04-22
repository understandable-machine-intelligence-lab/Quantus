from abc import ABC, abstractmethod


class ModelInterface(ABC):
    """ Abstract Interface of the Model Class
    -----------
    Attributes
    -----------
    path: str
        path, where the model is stored
    name: str
        name of the model e.g. vgg16
    type: str
        backend that the model is build on (tensorflow / pytorch)
    """

    def __init__(self, modelpath, modelname, modeltype):
        """ Initialize the model. """
        self.path = modelpath
        self.name = modelname
        self.type = modeltype
        super().__init__()

    # @abstractmethod
    # def evaluate(self, data, labels):
    #     """ Evaluate the performance of the model on the given data. """
    #     return NotImplementedError

    @abstractmethod
    def predict(self, data):
        """ Compute model predictions for the given data (model output).
        Parameters:
            data: numpy array
                numpy array of data to compute predictions for
        Returns
            numpy array of predictions of shape len(data) * num_classes
        """
        return NotImplementedError

    @abstractmethod
    def get_layer_names(self, with_weights_only=False):
        """ Get a list of model layer names.
         Parameters:
             with_weights_only: boolean
                whether to include only layers containing weights
        """
        return NotImplementedError

    @abstractmethod
    def randomize_layer_weights(self, layer_name):
        """ Randomizes the weights of the chosen layer.
        Parameters:
            layer_name: str
                name/key to the chosen layer
        """
        return NotImplementedError

    @abstractmethod
    def compute_relevance(self, batch, layer_names, neuron_selection, xai_method, additional_parameter):
        """ Computes attributions for the given data and labels.
        Parameters
            batch: numpy array
                numpy array of data to compute attributions for
            layer_names: list of str
                list of layer names to compute attributions for
            neuron_selection: int
                selected output neuron for attribution
            xai_method: str
                xai_method key for method mapping
            addidional_parameter: int
                as some xai method can be modified using additional parameters (tbd)
        Returns
            numpy array of attributions
        """
        return NotImplementedError
