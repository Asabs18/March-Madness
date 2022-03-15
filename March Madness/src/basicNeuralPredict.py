import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split


def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']

def format_output(data):
    y1 = data.pop('Score1')
    y1 = np.array(y1)
    y2 = data.pop('Score2')
    y2 = np.array(y2)
    return y1, y2

def build_model(train):
    # Define model layers.
    input_layer = Input(shape=(len(train .columns),))
    first_dense = Dense(units='128', activation='relu')(input_layer)
    # Y1 output will be fed from the first dense
    y1_output = Dense(units='1', name='Score1_output')(first_dense)

    second_dense = Dense(units='128', activation='relu')(first_dense)
    # Y2 output will be fed from the second dense
    y2_output = Dense(units='1', name='Score2_output')(second_dense)

    # Define the model with the input layer and a list of output layers
    model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

    return model

def plot_diff(history, y_true, y_pred, title=''):
    plt.plot(y_true, y_pred, 'r*')
    plt.axis([0, 150, 0, 150])
    plt.title(title)
    plt.ylabel("Actual Score")
    plt.xlabel("Predicted Score")
    plt.show()

def main():
    data = pd.read_csv("../data/MarchMadnessData.csv")
    data = data.drop(['Team1'], 1)
    data = data.drop(['Team2'], 1)
    data = data.sample(frac = 1)
    
    train, test = train_test_split(data, test_size=0.2, random_state=1)
    train, val = train_test_split(train, test_size=0.2, random_state=1)

    train_stats = train.describe()
    train_stats.pop('Score1')
    train_stats.pop('Score2')
    train_stats = train_stats.transpose()
    train_Y = format_output(train)
    test_Y = format_output(test)
    val_Y = format_output(val)

    norm_train_X = np.array(norm(train, train_stats))
    norm_test_X = np.array(norm(test, train_stats))
    norm_val_X = np.array(norm(val, train_stats))

    #TRAIN NEW MODEL
    model = build_model(train)

    optimizer = tf.keras.optimizers.SGD(lr=0.001)
    model.compile(optimizer=optimizer,
                loss={'Score1_output': 'mse', 'Score2_output': 'mse'},
                metrics={'Score1_output': tf.keras.metrics.RootMeanSquaredError(),
                        'Score2_output': tf.keras.metrics.RootMeanSquaredError()})
    
    history = model.fit(norm_train_X, train_Y, epochs=10, batch_size=10, validation_data=(norm_test_X, test_Y))
    
    loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x=norm_val_X, y=val_Y)

    print()
    print(f'loss: {loss}')
    print(f'score1_loss: {Y1_loss}')
    print(f'score2_loss: {Y2_loss}')
    print(f'score1_rmse: {Y1_rmse}')
    print(f'score2_rmse: {Y2_rmse}')

    Y_pred = model.predict(norm_test_X)
    price_pred = Y_pred[0]
    ptratio_pred = Y_pred[1]

    plot_diff(history, test_Y[0], Y_pred[0], title='Score1')
    plot_diff(history, test_Y[1], Y_pred[1], title='Score2')


if __name__ == "__main__":
    main()

#Referecnce Links
# https://towardsdatascience.com/guide-to-building-a-college-basketball-machine-learning-model-in-python-1c70b83acb51
# https://medium.com/re-hoop-per-rate/new-and-improved-march-madness-neural-network-for-2020-c154aa1041b7
# https://cloud.google.com/bigquery-ml/docs/bigqueryml-ncaa
# *https://medium.com/re-hoop-per-rate/training-a-neural-network-to-fill-out-my-march-madness-bracket-2e5ee562eab1*
# *https://www.tensorflow.org/tutorials/load_data/pandas_dataframe*
# *https://towardsdatascience.com/multi-output-model-with-tensorflow-keras-functional-api-875dd89aa7c6*