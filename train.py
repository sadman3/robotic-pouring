import numpy as np
import random, os, copy
import math, sys
np.set_printoptions(threshold=sys.maxsize)
from sklearn.metrics import mean_squared_error

####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(2)

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import matplotlib.pyplot as plt


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def simple_draw_graph(exp_name, train, valid):
    del train[0]
    del valid[0]

    epochs = list(range(1, len(train)+1))

    plt.plot(epochs, train, label = "Train") 
     
    plt.plot(epochs, valid, label = "Valid") 
    
    # naming the x axis 
    plt.xlabel('Epochs') 
    # naming the y axis 
    plt.ylabel('Loss') 
    # giving a title to my graph 
    plt.title('Train and validation loss') 
    
    # show a legend on the plot 
    plt.legend() 
    
    # function to show the plot 
    #plt.show() 
    plt.savefig('figures/loss-{}.png'.format(exp_name))
    plt.clf()

def draw_graph(input_data, output_data):

    sequence = list(range(1, len(input_data[0])+1))
    # Initialise the subplot function using number of rows and columns 
    
    ax1 = plt.subplot2grid((3, 3), (0,0))
    ax2 = plt.subplot2grid((3, 3), (0,1))
    ax3 = plt.subplot2grid((3, 3), (0,2))
    ax4 = plt.subplot2grid((3, 3), (1,0))
    ax5 = plt.subplot2grid((3, 3), (1,1))
    ax6 = plt.subplot2grid((3, 3), (1,2))
    ax7 = plt.subplot2grid((3, 3), (2,0), colspan=3)
    for i in range(150):
        ax1.plot(sequence, input_data[i, :, 0]) 
        ax1.set_title("$\\theta(t)$") 
        
        ax2.plot(sequence, input_data[i, :, 1]) 
        ax2.set_title("$f(init)$") 
        
        ax3.plot(sequence, input_data[i, :, 2]) 
        ax3.set_title("$f(target)$") 

        ax4.plot(sequence, input_data[i, :, 3]) 
        ax4.set_title("$h(cup)$") 

        ax5.plot(sequence, input_data[i, :, 4]) 
        ax5.set_title("$d(cup)$") 

        ax6.plot(sequence, input_data[i, :, 5]) 
        ax6.set_title("$\hat{\\theta}(t)$") 

        ax7.plot(sequence, output_data[i, :, 0]) 
        ax7.set_title("$f(t)$") 
 
    plt.show()

def load_dataset(file='Robot_Trials_DL.npy'):
    dataset = np.load(file)
    np.random.shuffle(dataset)

    
    n_data = len(dataset)
    train, valid = dataset[:n_data * 85 // 100, :, :], dataset[n_data * 85 // 100:, :, :]
    np.save('test_data.npy', valid)

    y_train, y_valid = train[:, :, 1:2], valid[:, :, 1:2]
    x_train, x_valid = np.delete(train, 1, axis=2), np.delete(valid, 1, axis=2)

    return x_train, y_train, x_valid, y_valid


def main():

    #hyperparams
    epoch = 1500
    lr = 0.003
    opt = 'Adam'
    layer=10000

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")


    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-3,
    #     decay_steps=800,
    #     decay_rate=0.8
    # )

    #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # best=0.003, 1000 epochs, Adam, 4 lstm, first 2 normalization,1 dense, shuffle, batch 16
    if opt == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr,momentum=0.9)
    if opt == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    
    #optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Array of shape (n_samples,n_timesteps,n_features)
    x_train, y_train, x_valid, y_valid = load_dataset()
    
    #print(np.amax(x_train[:,:,0]))
    #print(x_train[:,:,0])
    #draw_graph(x_train, y_train)
    # np.save('ground_truth.npy', y_valid)
        

    std_scale = MinMaxScaler(feature_range=(0,1))
    
    train_non_zero_indices = np.any(x_train, -1)
    valid_non_zero_indices = np.any(x_valid, -1)
    std_scale.fit(x_train[train_non_zero_indices])
    # print(std_scale.data_max_)
    # print(std_scale.data_min_)
    x_train[train_non_zero_indices] = std_scale.transform(x_train[train_non_zero_indices])
    x_valid[valid_non_zero_indices] = std_scale.transform(x_valid[valid_non_zero_indices])
    
    model = tf.keras.Sequential()

    #model.add(tf.keras.layers.Input(shape=(x_train.shape[1], 6)))
    model.add(tf.keras.layers.Masking(mask_value=0.0, input_shape=(x_train.shape[1], x_train.shape[2])))
    # Add a RNN layer with 10 internal units.
    model.add(tf.keras.layers.LSTM(100, return_sequences=True))
    model.add(tf.keras.layers.LayerNormalization())
    #model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(100, return_sequences=True))
    model.add(tf.keras.layers.LayerNormalization())
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(100, return_sequences=True))
    #model.add(tf.keras.layers.LayerNormalization())
    #model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.LSTM(100, return_sequences=True))
    #model.add(tf.keras.layers.LayerNormalization())

    # Add a Dense layer with 1 output unit
    model.add(tf.keras.layers.Dense(1))

    # Compile the model with an optimizer and a loss function
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer)

    # Print a summary of the model
    model.summary()

    # Train the model using samples from the dataset
    # model.fit(x=x_train, y=y_train, epochs=1000,
    #           callbacks=[model_checkpoint_callback],
    #           verbose=1, validation_data=(x_valid, y_valid))

    exp_name = '{}-{}-{}-{}'.format(str(epoch), str(lr), opt, str(layer))
    checkpoint_path = 'checkpoint-' + exp_name + '/'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_loss',
    save_best_only=True)

    history = model.fit(x=x_train, y=y_train, epochs=epoch,
              callbacks=[model_checkpoint_callback],batch_size=16,
              verbose=1, validation_data=(x_valid, y_valid))
    
    
    #simple_draw_graph(exp_name, history.history['loss'], history.history['val_loss'])


    model.load_weights(checkpoint_path)

    # Test the model
    temp_prediction = model.predict(x_valid, verbose=2)
    print('evaluating: ')
    model.evaluate(x_valid, y_valid, verbose=2)

    print()

    prediction = copy.deepcopy(y_valid)
    prediction[valid_non_zero_indices] = temp_prediction[valid_non_zero_indices]
    #print(prediction.shape)
    #print(prediction)
    #print(y_valid.shape)

    prediction = np.reshape(prediction, (prediction.shape[0] * prediction.shape[1], prediction.shape[2]))
    y_valid = np.reshape(y_valid, (y_valid.shape[0] * y_valid.shape[1], y_valid.shape[2]))

    _file = open('check_pred.txt', 'w')
    for i in range(len(prediction)):
        _file.write('{}\t{}\n'.format(y_valid[i][0], prediction[i][0]))
    _file.close()

    prediction_without_zero = prediction[np.any(prediction, -1)]
    y_valid_without_zero = y_valid[np.any(y_valid, -1)]


    print('with zero loss ', mean_squared_error(prediction, y_valid))
    print('without zero loss ', mean_squared_error(prediction_without_zero, y_valid_without_zero))

    # Compute an error and print it
    # error = np.mean(np.abs(output_after - x_train))
    # print(error)


if __name__ == '__main__':
    os.environ['PYTHONHASHSEED'] = str(2)
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    main()
    