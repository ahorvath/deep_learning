from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Nadam,Adam
from keras.callbacks import TensorBoard
from keras.regularizers import l1
from time import time
from keras.layers import Activation, Dense, Dropout, Flatten,Conv1D, LeakyReLU, MaxPooling1D, LSTM, BatchNormalization
import math

def create_model(n_motif, motif_len, seq_length, n_dense_neurons, conv_dropout, dense_dropout, batch_size, X_train, y_train, X_valid, y_valid):
    model = Sequential()
#    model.add(Conv1D(filters=n_motif, kernel_size=motif_len, use_bias = True, activation = "linear",  W_regularizer=l1(0), b_regularizer=l1(0), strides=1, padding='valid', input_shape=(seq_length,4)))
    model.add(Conv1D(filters=n_motif, activation = "relu", kernel_size=motif_len, dilation_rate = 1, use_bias = True, strides=1, padding='valid', input_shape=(seq_length,4)))

    model.add(BatchNormalization())
#i    model.add(LeakyReLU(alpha=0.1))

    #model.add(Activation("relu"))
#   model.add(MaxPooling1D(pool_size=math.ceil(seq_length/motif_len*2),strides=math.ceil(seq_length/motif_len*2)))
#    model.add(MaxPooling1D(pool_size=math.ceil(motif_len/2),strides=math.ceil(motif_len/2)))
#    model.add(MaxPooling1D(pool_size=motif_len,strides=motif_len))
    model.add(MaxPooling1D(pool_size=motif_len,strides=1))
#    model.add(Conv1D(filters=n_motif, kernel_size=motif_len, use_bias = True, strides=motif_len, padding='valid', activation='relu', input_shape=(seq_length,4)))
#    model.add(Activation("relu"))
#    model.add(BatchNormalization())
#    model.add(MaxPooling1D(pool_size=motif_len,strides=motif_len))
#    model.add(LSTM(n_motif, return_sequences=True))
    model.add(Flatten())

    model.add(Dense(n_dense_neurons, use_bias = True))
    model.add(BatchNormalization())
    model.add(Dropout(rate = conv_dropout))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(Nadam(lr=.01), loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(Adam(lr=.1), loss='squared_hinge', metrics=['accuracy'])

    print(model.summary())
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    callbacks = [EarlyStopping(monitor='val_loss', patience=10,  verbose=1,  restore_best_weights=True), ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True), tensorboard]
    model.fit(X_train, y_train, batch_size=batch_size, epochs=200, callbacks = callbacks, validation_data=(X_valid, y_valid), shuffle=True)

    return model
