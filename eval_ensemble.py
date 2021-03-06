import sys
import torch.backends.cudnn as cudnn
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import json
import os
from utils import *
import numpy as np
import json

import torch
torch.manual_seed(0)

with open(sys.argv[1], 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

exp_dir = sys.argv[2]

import numpy as np
from utils import image_comparison_reward, blockPrint, enablePrint

# Parameters
data_folder = cfg['data_folder']  # folder with data files saved by create_input_files.py
data_name = cfg['data_name']  # base name shared by data files

checkpoint = cfg['checkpoint']  # model checkpoint
original_checkpoint = '/data2/adsue/caption_data/checkpoints/BEST_XE_checkpoint_7_coco_5_cap_per_img_5_min_word_freq.pth.tar'
word_map_file = cfg['word_map_file']  # word map, ensure it's the same the data was encoded with and the model was trained with

batch_only = cfg['batch_only']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
original_checkpoint = torch.load(original_checkpoint)

decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()

decoder_original = original_checkpoint['decoder'].module
decoder_original = decoder_original.to(device)
decoder_original.eval()

encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

encoder_original = original_checkpoint['encoder'].module
encoder_original = encoder_original.to(device)
encoder_original.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


# Load word map from JSON
with open(os.path.join('/data2/adsue/caption_data', 'WORDMAP_' + data_name + '.json'), 'r') as j:
    word_map = json.load(j)

# Create the reverse word map
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word    
        
        

def evaluate(beam_size, encoder, decoder, reward_function):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypothesis = list()
    
    image_buffer = list()
    regeneration_reward = list()
        
    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        (img_captions, hyp) = get_captions_and_hypothesis(image, caps, caplens, allcaps, encoder, decoder, decoder_original, beam_size)

        references.append(img_captions)
        hypothesis.append(hyp)
        
        image_buffer.append(image)
        
        assert len(references) == len(hypothesis)

        if (i+1)%32 == 0:
            img_batch = torch.cat(image_buffer).to(device)
            blockPrint()
            regeneration_reward.append(reward_function(img_batch, hypothesis[-32:], save_imgs=batch_only, ground_truth=[sentences[0] for sentences in references[-32:]],split='VAL'))
            enablePrint()
            image_buffer = list()
            if batch_only and (i+1)%32==0:
                break
        
        
    hypothesis_sentences = [' '.join([rev_word_map[ind] for ind in sent]) for sent in hypothesis]
    reference_sentences =  [[' '.join([rev_word_map[ind] for ind in sent]) for sent in ref_sents] for ref_sents in references]
    
    # Calculate CIDER score
    CIDErD = compute_cider(reference_sentences, hypothesis_sentences, split='VAL')
    
    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypothesis)
    
    #Calculate Avg reward
    avg_regeneration_reward = torch.cat(regeneration_reward).mean().item()
    
    return (bleu4, avg_regeneration_reward, CIDErD)


    
def get_captions_and_hypothesis(image, caps, caplens, allcaps, encoder, decoder, decoder_original, beam_size):
    k = beam_size

    # Move to GPU device, if available
    image = image.to(device)  # (1, 3, 256, 256)
    # Encode
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Lists to store completed sequences and scores
    complete_seqs = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)
    h_o, c_o = decoder_original.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)
        
        #------------------------------------------------------------------------------------------------
        embeddings_o = decoder_original.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe_o, _ = decoder_original.attention(encoder_out, h_o)  # (s, encoder_dim), (s, num_pixels)

        gate_o = decoder_original.sigmoid(decoder_original.f_beta(h_o))  # gating scalar, (s, encoder_dim)
        awe_o = gate_o * awe_o

        h_o, c_o = decoder_original.decode_step(torch.cat([embeddings_o, awe_o], dim=1), (h_o, c_o))  # (s, decoder_dim)

        scores_o = decoder_original.fc(h_o)  # (s, vocab_size)
        scores_o = F.log_softmax(scores_o, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + 0.9*scores + 0.1*scores_o  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        h_o = h_o[prev_word_inds[incomplete_inds]]
        c_o = c_o[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 100:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]

    # References
    img_caps = allcaps[0].tolist()
    img_captions = list(
        map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
            img_caps))  # remove <start> and pads

    hyp = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]

    return (img_captions, hyp)

if __name__ == '__main__':
    beam_size = 4
    (bleu4, avg_regeneration_reward, CIDErD) = evaluate(beam_size, encoder, decoder, image_comparison_reward)
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, bleu4))
    print("\nAverage Regeneration-Reward @ beam size of %d is %.4f." % (beam_size, avg_regeneration_reward))
    print("\nCIDER-D score @ beam size of %d is %.4f." % (beam_size, CIDErD))
