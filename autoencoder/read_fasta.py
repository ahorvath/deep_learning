from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

def one_hot_encoder(sequence, letters):
    onehot_encoder = OneHotEncoder(categories=[letters], handle_unknown = "ignore")
    seq_array = np.array(list(sequence)).reshape(-1, 1)
    onehot_encoded_seq = onehot_encoder.fit_transform(seq_array).toarray()
    return onehot_encoded_seq

def fasta_to_hot_array(fasta_file, padding_size = 0):
    fasta_records = SeqIO.parse(open(fasta_file),'fasta')
    ids = []
    input_sequences = []
    input_features = []
    letters = ['A','C','G','T']
    for fasta_record in fasta_records:
        sequence = (padding_size*'N') + str(fasta_record.seq) + (padding_size*'N')
        ids.append(fasta_record.id)
        one_hot_encoded = one_hot_encoder(sequence, letters)
        input_sequences.append(sequence)
        input_features.append(one_hot_encoded)
    input_features = np.stack(input_features)    
    return ids, input_sequences, input_features


#if __name__ == '__main__':
#    np.set_printoptions(threshold=40)
    #input_pos_features = np.stack(input_pos_features)
#    fasta_pos_file = "pos1.fasta"
    #fasta_neg_file = "neg_set_l4000_l50.fasta"
#    ids,input_sequences,input_features = fasta_to_hot_array(fasta_pos_file, padding_size = 3) 
    #input_neg_features = fasta_to_hot_array(fasta_neg_file, padding_size = 10) 
#    input_features = np.concatenate((input_pos_features, input_neg_features))
#    classes = ["Pos", "Neg"]
#    x = np.array(classes)
#    labels = np.repeat(x, [len(input_pos_features), len(input_neg_features)], axis=0)
#    encoded_labels = one_hot_encoder(np.transpose(labels), classes)

#s = ['a', 'c', 'g', 't']
#le = LabelEncoder()
#ohe = OneHotEncoder(sparse=False, categories='auto')
#s = le.fit_transform(s)
#s = ohe.fit_transform(s.reshape(-1,1))
#print(s)
#inv_s = ohe.inverse_transform(X_test[1])
#inv_s = le.inverse_transform(inv_s.astype(int).ravel())
#inv_s

