import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

def get_hypothesis_greedy(encoder_out, sample=False):
    
    batch_size = encoder_out.size(0)
    encoder_dim = encoder_out.size(-1)

    # Flatten image
    encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # Tensor to store the previous word at each step; now it is just <start>
    prev_words = torch.LongTensor([[word_map['<start>']]] * batch_size).to(device)  # (batch_size, 1)

    # Tensor to store generated sequences; now it is just <start>
    seqs = prev_words  # (batch_size, 1)

    # Start decoding
    step = 1
    H, C = decoder.init_hidden_state(encoder_out)

    # Keep track of sum top scores for the REINFORCE algorithm.
    sum_top_scores = torch.zeros(batch_size)

    # Note: Batch size changes as generated sequences come to an <end>. 
    while True:

        # Update the indexes of sequences that are incomplete.
        incomplete_inds = [ind for ind, last_word in enumerate(seqs[:,-1]) if last_word != word_map['<end>']]

        embeddings = decoder.embedding(prev_words[incomplete_inds]).squeeze(1)  # (batch_size, embed_dim)

        awe, _ = decoder.attention(encoder_out[incomplete_inds], H[incomplete_inds])  # (batch_size, encoder_dim), (batch_size, num_pixels)

        gate = decoder.sigmoid(decoder.f_beta(H[incomplete_inds]))  # gating scalar, (batch_size, encoder_dim)
        awe = gate * awe

        H[incomplete_inds], C[incomplete_inds] = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (H[incomplete_inds], C[incomplete_inds]))  # (batch_size, decoder_dim)

        scores = decoder.fc(H[incomplete_inds])  # (batch_size, vocab_size)
        scores = F.log_softmax(scores, dim=1) #(batch_size, vocab_size) 

        if sample:
            probs = torch.exp(scores)
            categorical_distribution = torch.distributions.Categorical(probs)
            top_words = categorical_distribution.sample() # (batch_size)
            top_scores = scores[range(scores.size(0)), top_words] # (batch_size) 

        else:
            top_scores, top_words = scores.topk(1, 0, True, True)  # (batch_size) (batch_size)
        
        # Convert unrolled indices to actual indices of scores
        next_word_inds = top_words % vocab_size  # (batch_size)

        # Add new words to sequences
        seqs[incomplete_inds] = torch.cat([seqs[incomplete_inds], next_word_inds.unsqueeze(1)], dim=1)  # (1, step+1)

        # sum scores of actions for incomplete sequences       
        sum_top_scores[incomplete_inds] += top_scores # Keep track of sum top scores for the REINFORCE algorithm.
        
        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

        
    hypotheses = [[w for w in se if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}] for se in seqs]

    return (hypotheses, sum_top_scores)