import similarity.pairwise_alignment as pa


def cal_similarity_between_seq(seq1, seq2, similarity_type, dataset_type):
    
    if similarity_type == 'water' and dataset_type == 'protein':
        aln = pa.water(moltype = "prot", qseq = seq1, sseq = seq2, gapextend = 0)
        
    elif similarity_type == 'needle' and dataset_type == 'protein':
        aln = pa.needle(moltype = "prot", qseq = seq1, sseq = seq2, gapextend = 0)

    elif similarity_type == 'water' and dataset_type == 'dna':
        aln = pa.water(moltype = "nucl", qseq = seq1, sseq = seq2, gapextend = 0)
    
    elif similarity_type == 'needle' and dataset_type == 'dna':
        aln = pa.needle(moltype = "nucl", qseq = seq1, sseq = seq2, gapextend = 0)

    else:
        raise ValueError('Unknown similarity type.')
    return aln.pidentity, aln.qaln, aln.saln



if __name__ == "__main__":
    seq1 = "AAA"
    seq2 = "EEE"
