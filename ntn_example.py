#!/usr/bin/python

import math
import numpy as np
from sklearn.datasets import load_digits

from keras import backend as K

from keras.optimizers import SGD
from keras.layers import Dense

from keras.layers import Input
from keras.models import Model

from neural_tensor_layer import NeuralTensorLayer


def get_data():
  digits = load_digits()
  L = int(math.floor(digits.data.shape[0] * 0.15))
  X_train = digits.data[:L]
  y_train = digits.target[:L]
  X_test = digits.data[L + 1:]
  y_test = digits.target[L + 1:]
  return X_train, y_train, X_test, y_test


def main():
  input1 = Input(shape=(64,), dtype='float32')
  input2 = Input(shape=(64,), dtype='float32')
  btp = NeuralTensorLayer(output_dim=32, input_dim=64)([input1, input2])

  p = Dense(output_dim=1)(btp)
  model = Model(input=[input1, input2], output=[p])

  sgd = SGD(lr=0.0000000001, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='mean_squared_error', optimizer=sgd)
  X_train, Y_train, X_test, Y_test = get_data()
  X_train = X_train.astype(np.float32)
  Y_train = Y_train.astype(np.float32)
  X_test = X_test.astype(np.float32)
  Y_test = Y_test.astype(np.float32)

  model.fit([X_train, X_train], Y_train, nb_epoch=50, batch_size=5)
  score = model.evaluate([X_test, X_test], Y_test, batch_size=1)
  print score

  print K.get_value(model.layers[2].W)

main()