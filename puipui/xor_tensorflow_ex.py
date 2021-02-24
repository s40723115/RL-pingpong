import tensorflow as tf
import test2
from tensorflow.keras import layers
import pygame
from tensorflow.keras.layers import Activation, Dense

import numpy as np


# X = input of our 3 input XOR gate
# set up the inputs of the neural network (right from the table)
X = np.array(([goal1_x],[goal1_y],[goal2_x],
            [goal2_y])
# y = our output of our neural network
y  = np.array(([circle_x], [circle_y],  [ai_speed])

#define-->compile-->fit-->evaluate-->make Predictions
model = tf.keras.Sequential()#多個網路層線性堆疊的序貫模型

model.add(Dense(4, input_dim=3, activation='relu', use_bias=True))#Dense:output = activation(dot(input, kernel(權重矩陣))+bias)
#model.add(Dense(4,  activation='relu', use_bias=True))
model.add(Dense(1, activation='sigmoid', use_bias=True))

model.compile(loss='mean_squared_error',optimizer='adam',metrics=['binary_accuracy'])#loss:損失函數 optimizer:優化器 metrics:判定model的準確率(評估)；計算預測值與use_bias的符合頻率

print (model.get_weights())

history = model.fit(X, y, epochs=4000, validation_data = (X, y))#modle.fit()用以訓練模型 ; validation_data:在每次訓練結束時，評估損失數據和metrics值


model.summary()#method to display sequential contents


# printing out to file
loss_history = history.history["loss"]# .history紀錄連續迭代的loss值
numpy_loss_history = np.array(loss_history)
np.savetxt("loss_history.txt", numpy_loss_history,delimiter="\n") #字串或字符的分隔列

binary_accuracy_history = history.history["binary_accuracy"]#.history紀錄連續迭代的metrics值
numpy_binary_accuracy = np.array(binary_accuracy_history)
np.savetxt("binary_accuracy.txt", numpy_binary_accuracy, delimiter="\n")


print(np.mean(history.history["binary_accuracy"],dtype=np.float64))#print出每次metrics的平均值

result = model.predict(X).round()#結果預測

print (result)
