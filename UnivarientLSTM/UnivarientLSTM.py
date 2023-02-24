import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# convert an array of values into a dataset matrix
def univarient_lstm_create_dataset(dataset,time_step):
    """
        Creating dataset for Lstm by considering window size defoult window size is 1
        window size is to decides how many privious points we consider to predict next values
    """
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


def lstm_model_creation(x_train, y_train, x_test, y_test, node1, node2, node3, epochs, batchsize, scaler):
    """
        input parameters:-
        x_train:Splitted data to train the model
        y_train:Splitted data to train the model
        x_test:Splitted data to validate the model loss
        y_test:Splitted data to validate the model loss
        node1:Number of nodes which are added at 1st layer
        node2:Number of nodes which are added at 2nd layer
        node3:Number of nodes which are added at 3rd layer
        epochs:Number os epochs we can run (ex..,10,50,100)
        batchsize:(32,64,100)

        Here we are creating model with LSTM architecture with differents nodes, epochs and batach size

        output:it predicts output values with x_test length

    """
    model = Sequential()
    model.add(LSTM(18, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(44))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batchsize, verbose=1)
    yhat = model.predict(x_test, verbose=0)
    yhat = scaler.inverse_transform(yhat)
    print
    return yhat