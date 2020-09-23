#set.seed(42)

library(seqinr)

nucl <- c("A", "C", "G", "T")
n <- 4000
#pos.lengths <- rpois(n, 500)
pos.lengths <- rep(50, n)
neg.lengths <- rep(50, n)

#motif1 <- "GGGGCCCCAAAA"
motif1 <- "ATTAGCCG"

neg.seqs <- lapply(1:length(neg.lengths), function(i) {
                    paste0(sample(nucl, neg.lengths[i], replace = TRUE), collapse="")
})

neg.seqs2 <- lapply(1:length(neg.lengths), function(i) {
                    paste0(sample(nucl, neg.lengths[i], replace = TRUE), collapse="")
})


pos.seqs <- lapply(1:length(pos.lengths), function(i) {
                    seq <- paste0(sample(nucl, pos.lengths[i], replace = TRUE), collapse="")

		    startpos1 <- sample(1:(nchar(seq)-nchar(motif1)), 1)
		    substr(seq, startpos1, startpos1+nchar(motif1)) <- motif1

		    #startpos2 <- sample(1:(nchar(seq)-nchar(motif2)), 1)
                    #substr(seq, startpos2, startpos2+nchar(motif2)) <- motif2
		    seq
}) 


write.fasta(pos.seqs, paste0("pos_set", 1:n), paste0("pos_set_l", n, "_l50.fasta"), nbchar = 80)
write.fasta(neg.seqs, paste0("neg_set", 1:n), paste0("neg_set_l", n, "_l50.fasta"), nbchar = 80)
write.fasta(neg.seqs2, paste0("neg_set2_", 1:n), paste0("neg_set2_l", n, "_l50.fasta"), nbchar = 80)

head(lapply(neg.seqs, nchar))
