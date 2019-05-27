
class ModelConfig:
    """A class to hold the configurations for the DRNN model
       this class is passed around the training file to provide
       functions within with the necessary arguments.

       Args
         Epochs (int): number of training epochs
         Batch (int): number of samples per training epoch
         dropout (float): percentage of nuerons to dropout at each layer
         learning_rate (float): learning rate of the adam optimizer
         name (str): name of the model

    """

    def __init__(self, epochs, batch, dropout, name, learning_rate):
        self.epochs = epochs
        self.batch = batch
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.name = name
        self.num_layers = 3
        self.layer_sizes = [128, 64, 32]

    def set_layers(self, num_layers, layer_sizes):
        try:
            if num_layers != len(layer_sizes):
                raise Exception("Number of layers and sizes given are inconsistent")
            else:
                self.num_layers = num_layers
                self.layer_sizes = layer_sizes
        except Exception as e:
            print(e)

    def set_data_paths(self, train_path, infer_path):
        self.train_path = train_path
        self.infer_path = infer_path

