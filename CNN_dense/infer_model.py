import pickle
from keras import backend as K

with open('params.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    input_seq, X_train, X_test, X_valid, y_train, y_test, y_valid, train_indices, test_indices, valid_indices, model = pickle.load(f)


batch_size = 200

t_results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print(t_results)
predicted_classes = model.predict_classes(X_test)
conv_layer = model.layers[0]
maxpool1d_layer = model.layers[1]
flatten_layer = model.layers[2]
dense_layer = model.layers[3]
# optimize metaparameter

conv_output = conv_layer.get_output_at(0)
maxpool1d_output = maxpool1d_layer.get_output_at(0)
dense_output = dense_layer.get_output_at(0)

maxpool1d_f = K.function([conv_layer.input], conv_output)

conv1d_f_max = K.function([model.input], [K.argmax(conv_output, axis=1), K.max(conv_output, axis=1)])
conv1d_f_all = K.function([model.input], conv_output)
maxpool1d_f_all = K.function([model.input], maxpool1d_output, )
dense_f_all = K.function([model.input], dense_output)

conv1d_f_max_X_test = conv1d_f_max(X_test)
conv1d_f_all_X_test = conv1d_f_all(X_test)
maxpool1d_f_all_X_test = maxpool1d_f_all(X_test)
dense_f_all_X_test = dense_f_all(X_test)

input_seq[test_indices][0]

y_test_pos = y_test.sum(axis=1) > 0
X_test_pos = X_test[y_test_pos]

matrix = confusion_matrix(y_test, predicted_classes)
print(matrix)
make_logos(n_motif, motif_len, X_test_pos, batch_size = batch_size, f = conv1d_f_max, name = "motifs")
input_seq[test_indices][y_test_pos]

TP_indices = ((y_test & predicted_classes) == 1).flatten()
TN_indices = ((np.logical_not(y_test) & np.logical_not(predicted_classes)) == 1).flatten()
FP_indices = ((np.logical_not(y_test) & predicted_classes) == 1).flatten()
FN_indices = ((y_test & np.logical_not(predicted_classes)) == 1).flatten()

input_seq[test_indices][TP_indices]; input_seq[test_indices][TN_indices]; input_seq[test_indices][FN_indices]; input_seq[test_indices][FP_indices]
print(len(input_seq[test_indices][TP_indices]))
print(len(input_seq[test_indices][TN_indices]))
print(input_seq[test_indices][FN_indices])
print(input_seq[test_indices][FP_indices])

