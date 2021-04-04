import numpy as np
import os,random
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(2)
from sklearn.preprocessing import MinMaxScaler
import sys,copy
import tensorflow as tf
from sklearn.metrics import mean_squared_error

random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)
# total arguments
n = len(sys.argv)


def load_dataset(file):
    dataset = np.load(file)

    output = dataset[:, :, 1:2]
    input = np.delete(dataset, 1, axis=2)

    return input, output

if n != 2:
    print('wrong format. It should be something like python3 test.py test_trials.npy')

filename = sys.argv[1]
input, output = load_dataset(filename)
non_zero_id = np.any(input, -1)

max_values = [32.60742188, 1.35721388, 0.87896302, 292., 121., 1.79523885]
min_values = [-1.41679688e+02, 8.95937692e-02, 2.64889579e-02, 8.10000000e+01, 6.70000000e+01, -1.15606320e+00]


# normaliztion
for i in range(input.shape[0]):
    for j in range(input.shape[1]):
        for k in range(input.shape[2]):
            input[i][j][k] = (input[i][j][k] - min_values[k]) / (max_values[k] - min_values[k])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
model = tf.keras.Sequential()

model.add(tf.keras.layers.Masking(mask_value=0.0, input_shape=(input.shape[1], input.shape[2])))
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

model.load_weights('checkpoint/')


temp_prediction = model.predict(input)



prediction = np.zeros((output.shape[0], output.shape[1], output.shape[2]))

prediction[non_zero_id] = temp_prediction[non_zero_id]

np.save('test_set_output.npy', prediction)

# print('evaluation: ')
# model.evaluate(input, output)

prediction = np.reshape(prediction, (prediction.shape[0] * prediction.shape[1], prediction.shape[2]))
output = np.reshape(output, (output.shape[0] * output.shape[1], output.shape[2]))


print('mse: ', mean_squared_error(prediction[np.any(output,-1)], output[np.any(output, -1)]))

