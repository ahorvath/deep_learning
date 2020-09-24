from __future__ import print_function
import keras
from keras.models import load_model
from keras import backend as K
from read_fasta import fasta_to_hot_array, one_hot_encoder
from make_logos import make_logos
from model import create_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer

np.set_printoptions(threshold=40)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

seq_length = 50
batch_size = 100
motif_len = 10
#padding_size = int(motif_len/2)
padding_size = 0
n_motif = 3
n_dense_neurons = 15

fasta_pos_file = "../CNN_dense/test_data/pos_set_n6000_l50_GGATTACC_and_TTGGGGAA.fasta"
fasta_neg_file = "../CNN_dense/test_data/pos_set_n6000_l50_GGATTACC_or_TTGGGGAA.fasta"
input_pos_id, input_pos_seq, input_pos_features = fasta_to_hot_array(fasta_pos_file, padding_size = padding_size)
input_neg_id, input_neg_seq, input_neg_features = fasta_to_hot_array(fasta_neg_file, padding_size = padding_size)

input_ids = input_pos_id
input_seq = input_pos_seq
input_features = input_pos_features

classes = ["Pos"]
x = np.array(classes)
labels = np.repeat(x, [len(input_pos_features)], axis=0)

lb = LabelBinarizer()
encoded_labels = lb.fit_transform(labels)

###############################################
import pickle
from sklearn.model_selection import train_test_split
indices = np.arange(len(input_features))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(input_features, encoded_labels, indices, test_size=0.2, random_state=1)
X_train, X_valid, y_train, y_valid, train_indices, valid_indices = train_test_split(X_train, y_train, train_indices, test_size=0.1, random_state=1)

model = create_model(n_motif, motif_len, seq_length, batch_size, X_train,X_valid)
#model = load_model('best_model.h5')

t_results = model.evaluate(X_test, X_test, batch_size=batch_size, verbose=1)
print(t_results)
predicted = model.predict(X_test)
print(predicted)
