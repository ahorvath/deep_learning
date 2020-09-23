#set.seed(42)

library(seqinr)

nucl <- c("A", "C", "G", "T")
n <- 6000
l <- 500
#pos.lengths <- rpois(n, 500)
pos.lengths <- rep(l, n)
neg.lengths <- rep(l, n)

motif1 <- "GGGGATTACCCC"
motif2 <- "TTGGGGGGGGAA"

neg.seqs <- lapply(1:length(neg.lengths), function(i) {
                    paste0(sample(nucl, neg.lengths[i], replace = TRUE), collapse="")
})

pos.or.seqs <- lapply(1:length(neg.lengths), function(i) {
                    seq <- paste0(sample(nucl, neg.lengths[i], replace = TRUE), collapse="")

		    motif <- sample(c(motif1, motif2), size = 1)
		    startpos <- sample(1:(nchar(seq)-nchar(motif)), 1)
                    substr(seq, startpos, startpos+nchar(motif)) <- motif

		    seq
})

pos.and.seqs <- lapply(1:length(pos.lengths), function(i) {
                    seq <- paste0(sample(nucl, pos.lengths[i], replace = TRUE), collapse="")
		    
		    startpos1 <- sample(1:(nchar(seq)/2-nchar(motif1)), 1)
		    substr(seq, startpos1, startpos1+nchar(motif1)) <- motif1

		    startpos2 <- sample((nchar(seq)/2+1):(nchar(seq)-nchar(motif2)), 1)
                    substr(seq, startpos2, startpos2+nchar(motif2)) <- motif2

		    seq
}) 


write.fasta(pos.and.seqs, paste0("pos_set", 1:n), paste0("pos_set_n", n, "_l", l, "_", motif1, "_and_", motif2, ".fasta"), nbchar = 80)
write.fasta(pos.or.seqs, paste0("pos_set", 1:n), paste0("pos_set_n", n, "_l", l, "_", motif1, "_or_", motif2, ".fasta"), nbchar = 80)
write.fasta(neg.seqs, paste0("neg_set", 1:n), paste0("neg_set_n", n, "_l", l, ".fasta"), nbchar = 80)
