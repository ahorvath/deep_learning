import subprocess
import numpy as np

#What do you mean by z[0]? - z is a tuple output from the functional f. z is size 2. The first item contains a matrix of activations from a minibatch input. Suppose x_test is a minibatch of sequences and has the size (100, 1000, 4) (again, 100 is the minibatch size and is completely arbitrary). Then z = f(x_test). z[0] would be a matrix of size (100, 320), and contain integers telling you the location of the maximum activation for each minibatch sample sequence for each filter. For example, z[0][50,12] would tell you the location on sequence 50 where filter 12 achieves its maximum activation. z[1] has the same shape as z[0], but tells you the actual max activation value.

def make_logos(n_motif, motif_len, X, f, batch_size = 100, name = "motifs"):
    motifs = np.zeros((n_motif, motif_len, 4))
    nsites = np.zeros(n_motif)

    for i in range(0, len(X), batch_size):
        x = X[i:i+batch_size]
        z = f([x])
        max_inds = z[0] # N x M matrix, where M is the number of motifs
        max_acts = z[1]
        for m in range(n_motif):
            for n in range(len(x)):
                # Forward strand
                #print ("m=", m, ", n=", n)
                if max_acts[n, m] > 0:
                    #print("max=", max_acts[n, m])
                    nsites[m] += 1
                    motifs[m] += x[n, max_inds[n, m]:max_inds[n, m] + motif_len, :]
        
    print('Making motifs')
    motifs_file_name = name + ".txt" 
    motifs_file = open('motifs.txt', 'w')
    motifs_file.write('MEME version 4.9.0\n\n'
                  'ALPHABET= ACGT\n\n'
                  'strands: + -\n\n'
                  'Background letter frequencies (from uniform background):\n'
                  'A 0.25000 C 0.25000 G 0.25000 T 0.25000\n\n')
    for m in range(n_motif):
        if nsites[m] == 0:
            continue
        motifs_file.write('MOTIF M%i O%i\n' % (m, m))
        motifs_file.write("letter-probability matrix: alength= 4 w= %i nsites= %i E= 1337.0e-6\n" % (motif_len, nsites[m]))
        for j in range(motif_len):
            motifs_file.write("%f %f %f %f\n" % tuple(1.0 * motifs[m, j, 0:4] / np.sum(motifs[m, j, 0:4])))
        motifs_file.write('\n')

    motifs_file.close()
    subprocess.call(["meme2images", motifs_file_name, "."])

