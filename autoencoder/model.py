#https://blog.keras.io/building-autoencoders-in-keras

from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dense, Reshape
import numpy as np
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

def create_model(n_motif, motif_len, seq_length, batch_size, X_train, X_valid):

    input_img = Input(shape=(seq_length, 4))  # adapt this if using `channels_first` image data format
    print(input_img)
    print(n_motif)
    print("motif_len")
    print(motif_len)
    x = Conv1D(filters=n_motif, kernel_size=motif_len, strides=1, activation='relu', padding='same', name = "first_enc_conv")(input_img)
#    x = MaxPooling1D(pool_size=motif_len,strides=motif_len)(x)
    shape_before_flattening = K.int_shape(x)
    print("shape")
    print(shape_before_flattening[1:])
    x = Flatten()(x)
    encoded = Dense(5, activation='relu', name='encoded')(x)
   
    # Decoder
    x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(encoded)
    # Reshape into an image of the same shape as before our last `Flatten` layer
    x = Reshape(shape_before_flattening[1:])(x)

    x = Conv1D(filters=n_motif, kernel_size=motif_len-1, strides=1, activation='relu', padding='same', name = "second_dec_conv")(x)
    decoded = Conv1D(filters=4, kernel_size=motif_len, activation='sigmoid', padding='same', name = "final_dec_conv")(x)
    print(decoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    print(autoencoder.summary())
#    print(X_train)
    #Let's train it for 100 epochs:
    autoencoder.fit(X_train, X_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(X_valid, X_valid),
                callbacks=[TensorBoard(log_dir='tmp/tb', histogram_freq=0, write_graph=False)])
    return autoencoder
