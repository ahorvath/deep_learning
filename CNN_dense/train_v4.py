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
motif_len = 14
#padding_size = int(motif_len/2)
padding_size = 0
n_motif = 3
n_dense_neurons = 15

#fasta_pos_file = "test_data/pos_l50_s10000_motif_GGGGCCCC.fasta"
#fasta_pos_file = "test_data/pos_set_n6000_l50.fasta"
fasta_pos_file = "test_data/pos_set_n6000_l50_GGATTACC_and_TTGGGGAA.fasta"
#fasta_pos_file = "test_data/pos_set_n6000_l500_GGGGATTACCCC_and_TTGGGGGGGGAA.fasta"
#fasta_neg_file = "test_data/neg_l50_s10000_motif_GGGGCCCC.fasta"
#fasta_neg_file = "test_data/neg_set_n6000_l50.fasta"
fasta_neg_file = "test_data/pos_set_n6000_l50_GGATTACC_or_TTGGGGAA.fasta"
#fasta_neg_file = "test_data/pos_set_n6000_l500_GGGGATTACCCC_or_TTGGGGGGGGAA.fasta"
input_pos_id, input_pos_seq, input_pos_features = fasta_to_hot_array(fasta_pos_file, padding_size = padding_size)
input_neg_id, input_neg_seq, input_neg_features = fasta_to_hot_array(fasta_neg_file, padding_size = padding_size)

input_ids = np.concatenate((input_pos_id, input_neg_id))
input_seq = np.concatenate((input_pos_seq, input_neg_seq))
input_features = np.concatenate((input_pos_features, input_neg_features))

classes = ["Pos", "Neg"]
x = np.array(classes)
labels = np.repeat(x, [len(input_pos_features), len(input_neg_features)], axis=0)

lb = LabelBinarizer()
encoded_labels = lb.fit_transform(labels)

###############################################
import pickle
from sklearn.model_selection import train_test_split
indices = np.arange(len(input_features))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(input_features, encoded_labels, indices, test_size=0.2, random_state=1)
X_train, X_valid, y_train, y_valid, train_indices, valid_indices = train_test_split(X_train, y_train, train_indices, test_size=0.1, random_state=1)

model = create_model(n_motif, motif_len, seq_length+2*padding_size, n_dense_neurons, 0.1, 0.1, batch_size, X_train, y_train, X_valid, y_valid)
#model = load_model('best_model.h5')

t_results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print(t_results)
predicted_classes = model.predict_classes(X_test)
conv_layer = model.layers[0]
conv_output = conv_layer.get_output_at(0)
conv1d_f_max = K.function([model.input], [K.argmax(conv_output, axis=1), K.max(conv_output, axis=1)])

f = open('params.pkl', 'wb')
pickle.dump([input_seq, X_train, X_test, X_valid, y_train, y_test, y_valid, train_indices, test_indices, valid_indices, model], f)
f.close()

y_test_pos = y_test.sum(axis=1) > 0
X_test_pos = X_test[y_test_pos]
y_train_pos = y_train.sum(axis=1) > 0
print( y_train.sum(axis=1))
X_train_pos = X_train[y_train_pos]

matrix = confusion_matrix(y_test, predicted_classes)
print(matrix)
make_logos(n_motif, motif_len, X_train_pos, batch_size = 1, f = conv1d_f_max, name = "motifs")
input_seq[test_indices][y_test_pos]

TP_indices = ((y_test & predicted_classes) == 1).flatten()
TN_indices = ((np.logical_not(y_test) & np.logical_not(predicted_classes)) == 1).flatten()
FP_indices = ((np.logical_not(y_test) & predicted_classes) == 1).flatten()
FN_indices = ((y_test & np.logical_not(predicted_classes)) == 1).flatten()

input_seq[test_indices][TP_indices]; input_seq[test_indices][TN_indices]; input_seq[test_indices][FN_indices]; input_seq[test_indices][FP_indices]
print(len(input_seq[test_indices][TP_indices]))
print(len(input_seq[test_indices][TN_indices]))
#print(input_seq[test_indices][FN_indices])
#print(input_seq[test_indices][FP_indices])
