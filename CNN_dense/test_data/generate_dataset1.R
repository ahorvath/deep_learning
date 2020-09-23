#set.seed(42)

library(seqinr)

nucl <- c("A", "C", "G", "T")
n <- 6000
l <- 50
#pos.lengths <- rpois(n, 500)
pos.lengths <- rep(l, n)
neg.lengths <- rep(l, n)

motif <- "ATTAGCCG"

neg.seqs <- lapply(1:length(neg.lengths), function(i) {
                    paste0(sample(nucl, neg.lengths[i], replace = TRUE), collapse="")
})

neg.seqs2 <- lapply(1:length(neg.lengths), function(i) {
                    paste0(sample(nucl, neg.lengths[i], replace = TRUE), collapse="")
})


pos.seqs <- lapply(1:length(pos.lengths), function(i) {
                    seq <- paste0(sample(nucl, pos.lengths[i], replace = TRUE), collapse="")

		    startpos <- sample(1:(nchar(seq)-nchar(motif)), 1)
		    substr(seq, startpos, startpos+nchar(motif)) <- motif

		    seq
}) 


write.fasta(pos.seqs, paste0("pos_set", 1:n), paste0("pos_set_n", n, "_l", l, ".fasta"), nbchar = 80)
write.fasta(neg.seqs, paste0("neg_set", 1:n), paste0("neg_set_n", n, "_l", l, ".fasta"), nbchar = 80)
write.fasta(neg.seqs2, paste0("neg_set2_", 1:n), paste0("neg_set2_n", n, "_l", l, ".fasta"), nbchar = 80)

head(lapply(neg.seqs, nchar))
