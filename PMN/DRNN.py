
from keras import backend as K
from keras import models
from keras import layers
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import TensorBoard
from time import time

from modelconfig import ModelConfig

import pandas as pd
import numpy as np
import tensorflow as tf


def build_model(train_shape):
    """Build a deep neural net for regression. The adam optimizer is used with
       fairly common settings, although lr is changed as a parameter. ReLU
       activation is used at each layer and a dropout layer is added after
       each Dense layer.

       Args
         train_shape (tuple): The input shape of the training data

      Returns
        DRNN built in Keras with the parameter specifications
    """

    model = models.Sequential()

    # add first layer
    model.add(layers.Dense(CONF.layer_sizes[0], activation='relu',
                           input_shape=(train_shape[1],)))
    model.add(layers.Dropout(CONF.dropout))

    # Add layers to DRNN model with dropout
    for layer in CONF.num_layers[1:]:
        model.add(layers.Dense(layer, activation='relu'))
        model.add(layers.Dropout(CONF.dropout))

    # Add final layer
    model.add(layers.Dense(2))

    adamopt = optimizers.Adam(lr=CONF.learning_rate, beta_1=0.9, beta_2=0.999,
                              epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=adamopt, loss='mse', metrics=['mae', "accuracy"])
    return model


def get_data(data_path, train_data=True):
    """Gets the training and inference data for the MPO training process

       Args
         data_path (str): path to inference or training_data
         train_data (bool): if true, split data into train and test

      Returns
         A list of either training or inference data
         train data will have four elements for training and testing
         inference data will have two elements(samples and targets)
    """

    all_data = pd.read_csv(data_path)
    all_targets = all_data[["KH", "KHTH"]].copy()
    all_data.drop(['KH', 'KHTH'], axis=1, inplace=True)

    if train_data:
        X_train, x_test, Y_train, y_test = train_test_split(all_data, all_targets, test_size=0.1)
        data = [X_train, x_test, Y_train, y_test]
        return data
    return [all_data, all_targets]


def train_model(model, train_data):
    """Collect low-resolution training data and train PMN model
       Returns model to be used for higher resolution inference

       Args
         model (Keras Model): model to be trained on low resolution data
         train_data (list): a list of samples and targets split into train
                            and test (4 total elements)

      Returns
        A keras model trained on low resolution data
    """

    # Collect train and test data
    X_train = train_data[0]
    x_test = train_data[1]
    Y_train = train_data[2]
    y_test = train_data[3]

    # init TensorBoard
    tensorboard = TensorBoard(log_dir="logs/{}".format(CONF.name))

    callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                 ModelCheckpoint(filepath=CONF.name + "_model.h5",
                                 monitor='val_loss', save_best_only=True),
                 tensorboard]

    # train PMN model
    model.fit(X_train,
              Y_train,
              epochs=CONF.epochs,
              callbacks=callbacks,
              validation_split=0.1,
              batch_size=CONF.batch,
              verbose=2)

    total_loss, mse, val_acc = model.evaluate(x_test, y_test)
    print("total loss: %e" % total_loss)
    print("Validation MSE: %e" % mse)
    print("validation accuracy: %e" % val_acc)

    # Field of Merit for hyperparameter optimization
    print("FoM: %e" % total_loss)

    return model


def create_mapping_values(model, inference_data):
    """Creation of the mapping samples for the MPO pipeline. Samples
       of a higher resolution than the samples used in training are
       given to the DRNN trained on low resolution data for inference.
       the results are recorded and returned

       Args
         model (Keras model): A DRNN model trained on low res data
         inference_data (list): a list of samples and targets for the
                                higher resolution data.

       Returns
         A dataframe of the predicted and actual parameter spaces for the
         model resolution provided.

    """

    med_train = inference_data[0]
    med_targets = inference_data[1]

    med_pred = pd.DataFrame(model.predict(med_train))
    med_tar = pd.DataFrame(med_targets.copy())

    for i in range(5):
        print("Querying PMN for predictions")
        # create mutliple prediction instances
        predicted = pd.DataFrame(model.predict(med_train))
        med_pred = pd.concat([predicted, med_pred])
        med_tar = pd.concat([med_tar, med_targets])

    print("---- Real ----")
    print(med_tar)
    print("---- Pred ----")
    print(med_pred)

    med_tar["pred_KH"] = med_pred[0]
    med_tar["pred_KHTH"] = med_pred[1]

    return med_tar

def write_results(results):
    results_file = CONF.name + ".csv"
    results.to_csv(results_file, index=False)


def main():
    """the entire training and inference process of the MPO pipeline

       1) Parameter mapping network(PMN) is trained on low resolution data
       2) Samples of a higher resolution are provided to PMN trained on low res data
       3) The mapping samples generated in part 2 are recorded and returned for
          use in the Parameter Mapping Transform(PMT) in the data-analysis directory


       Please refer to the documentation for more information
    """

    training_data = get_data(CONF.train_path)
    training_data_shape = training_data[0].shape

    # Create and train a PMN model
    model = build_model(training_data_shape)
    trained_model = train_model(model, training_data)

    # Use PMN trained on low resolution data to create mapping samples
    # from high resolution data
    inference_data = get_data(CONF.infer_path, train_data=False)
    mapping_samples = create_mapping_values(trained_model, inference_data)
    write_results(mapping_samples)



if __name__ == "__main__":
    import os
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--learning_rate", type=float, default=0.01)
    argparser.add_argument("--layer_sizes", type=list, default=[128, 64, 32])
    argparser.add_argument("--num_layers", type=int, default=3)
    argparser.add_argument("--epochs", type=int, default=200)
    argparser.add_argument("--batch", type=int, default=100)
    argparser.add_argument("--dropout", type=float, default=.2)
    argparser.add_argument("--lowres", type=str, default="../processed-data/full/low-ke-full.csv")
    argparser.add_argument("--highres", type=str, default="../processed-data/full/med-ke-full.csv")
    argparser.add_argument("--name", type=str, default="drnn_model")
    args = argparser.parse_args()

    # A class to hold all of the DRNN model parameters
    CONF = ModelConfig(args.epochs, args.batch, args.dropout, args.learning_rate, args.name)
    CONF.set_data_paths(args.lowres, args.highres)

    # Set Parallelism options for tensorflow backend
    config = tf.ConfigProto(intra_op_parallelism_threads=48, inter_op_parallelism_threads=2,
                            allow_soft_placement=True, device_count = {'CPU': 48 })
    session = tf.Session(config=config)
    K.set_session(session)

    os.environ["OMP_NUM_THREADS"] = "48"
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    main()


