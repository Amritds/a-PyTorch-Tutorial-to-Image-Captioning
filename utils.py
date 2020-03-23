import os
import sys
import subprocess
import numpy as np
import h5py
import json
import torch
import pickle
import requests
from torch.nn.modules.loss import _Loss
from torch.nn import CosineSimilarity
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from models import ComparisonEncoder
import torchvision
from nltk.translate.bleu_score import sentence_bleu
from StackGAN.code.main_sampler import sample as image_generator
import yaml
import scipy

with open(sys.argv[1], 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

import torchvision.transforms as transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

# Data parameters
data_folder = cfg['data_folder']  # folder with data files saved by create_input_files.py
data_name = cfg['data_name'] # base name shared by data files

exp_dir = sys.argv[2]

checkpoints_dir = os.path.join(exp_dir,'checkpoints')
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

resnet = torchvision.models.resnet101(pretrained=True)
for p in resnet.parameters():
    p.requires_grad = False

# Use to compute cosine similarity between resnet encodings.
comparison_encoder = resnet.to(device)

cos = CosineSimilarity(dim=1, eps=1e-6)

global word_map, rev_word_map, train_sentence_index

train_sentence_index = None


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def read_img(path):
    img = imread(path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    assert img.shape == (3, 256, 256)
    assert np.max(img) <= 255

    return img

def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}
    
    global word_map, rev_word_map
    
    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Create the reverse word map
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word    
        
    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = read_img(impaths[i])

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    is_best, training_type, reward=None, bleu4=None):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'training_type': training_type,
    		 'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'reward': reward,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    
    filename = training_type + '_checkpoint_' +str(epoch)+'_'+ data_name + '.pth.tar'
    torch.save(state, os.path.join(checkpoints_dir, filename))

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

class RL_loss(_Loss):
    def __init__(self, reward_function):
        super(RL_loss, self).__init__(size_average=None, reduce=None, reduction='mean')
        self.reward_function = reward_function

    def forward(self, imgs, ground_truth, hypothesis, hyp_max, sum_top_scores):
        
        blockPrint() # Avoid verbose print statements
        
        advantage = self.reward_function(imgs, hypothesis, save_imgs=False, ground_truth=ground_truth) - self.reward_function(imgs, hyp_max, save_imgs=False, ground_truth=ground_truth)
            
        enablePrint() # Re-enable print functionality
        
        weighted_sum_top_scores = advantage * sum_top_scores
        return ((-1) * weighted_sum_top_scores.mean()) # Important!!! use negative sum of expected rewards to treat as minimization problem. 

def convert_to_json_and_save(references, hypothesis):
    hyp_sentences = [(i, sent) for (i,sent) in enumerate(hypothesis)]
    ref_sentences = []
    for i, ref_sents in enumerate(references):
        ref_sentences += [(i, sent) for sent in ref_sents]
    
    refs_json = [{'image_id': i, 'caption': r} for (i,r) in ref_sentences]
    hyps_json = [{'image_id': i, 'caption': r} for (i,r) in hyp_sentences]
    
    refs_file = os.path.join(exp_dir, 'refs.json')
    hyps_file = os.path.join(exp_dir, 'hyps.json')
    
    with open(refs_file ,'w') as f:
        f.write(json.dumps(refs_json))
    with open(hyps_file ,'w') as f:
        f.write(json.dumps(hyps_json))    

    
def compute_cider(references, hypothesis):
    # Save json files for CIDER computation.
    convert_to_json_and_save(references, hypothesis)

    # Save params to json for CIDER computation.
    params = {"pathToData" : exp_dir,
              "refName" : "refs.json",
              "candName" : "hyps.json",
              "resultFile" : os.path.join(exp_dir, "results.json"),
              "idf" : "corpus"}
    
    params_file = os.path.join(exp_dir, 'params.json')
    with open(params_file ,'w') as f:
        f.write(json.dumps(params))

    # Compute CIDER scores
    os.system("source activate cider_env && python -u cider/cidereval.py "+exp_dir+"/params.json &> "+exp_dir+"/out_cider &")
    
    # Read CIDER scores
    with open(params['resultFile'] ,'r') as f:
        scores = json.load(f)
    
    # Return CIDER scores
    return (np.mean(scores['CIDEr']), np.mean(scores['CIDErD']))
    
def cider_reward(imgs, hypothesis, save_imgs, ground_truth):
    """
    Note: Uses the sentence index.
    """
    global train_sentence_index
    
    try:
        hypothesis_sentences = [' '.join([rev_word_map[ind] for ind in sent]) for sent in hypothesis]
    except:
        # Load word map from JSON
        with open(os.path.join('/data2/adsue/caption_data', 'WORDMAP_' + data_name + '.json'), 'r') as j:
            word_map = json.load(j)

        # Create the reverse word map
        rev_word_map = {v: k for k, v in word_map.items()}  # ix2word    
        
        hypothesis_sentences = [' '.join([rev_word_map[ind] for ind in sent]) for sent in hypothesis]
        
    if train_sentence_index ==None:
        with open(os.path.join('/data2/adsue/caption_data','TRAIN_CAPTIONS_sentence_index.json'),'r') as f:
            train_sentence_index = json.load(f)
  
    # References
    references_sentences = [train_sentence_index[hyp] for hyp in hypothesis_sentences]
    
    # Calculate CIDER score
    (CIDEr, CIDErD) = compute_cider(reference_sentences, hypothesis_sentences)
    
    return CIDErD
    
def image_comparison_reward(imgs, hypothesis, save_imgs, ground_truth):
    # Note: Ground truth captions not required.
    # Translate and save the hypothesis as plain text.
    
    try:
        sentences = [' '.join([rev_word_map[ind] for ind in sent]) for sent in hypothesis]
    except:
        # Load word map from JSON
        with open(os.path.join('/data2/adsue/caption_data', 'WORDMAP_' + data_name + '.json'), 'r') as j:
            word_map = json.load(j)

        # Create the reverse word map
        rev_word_map = {v: k for k, v in word_map.items()}  # ix2word    
        
        sentences = [' '.join([rev_word_map[ind] for ind in sent]) for sent in hypothesis]
    
    
    # Save the minibatch sentences.
    minibatch_words_path = os.path.join(exp_dir, 'mini_batch_captions.txt')
    with open(minibatch_words_path, 'w') as f:
        for sent in sentences:
            f.write(sent + '\n')

    # Get encoding (saved as a torchfile)
    requests.get('http://0.0.0.0:8080/' + '?path='+exp_dir)
    
    # Generate images from encoded minbatch (saved to file), convert to tensor, scale and normalize.
    recreated_imgs = (image_generator(exp_dir))
    recreated_imgs = torch.Tensor(recreated_imgs).to(device).permute(0,3,1,2)
    recreated_imgs = normalize(recreated_imgs/torch.max(recreated_imgs))
    
    # compute the encoding for recreated and original images
    encoded_original = comparison_encoder(imgs)
    encoded_recreation = comparison_encoder(recreated_imgs)

    if save_imgs:
        save_images_to_folder(imgs, 'original')
        save_images_to_folder(recreated_imgs, 'recreated')
    
    # compute similarity
    similarity = cos(encoded_original, encoded_recreation)

    return similarity

def save_images_to_folder(imgs, file_path):
    
    file_path = os.path.join(exp_dir, file_path)        
        
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    # save images and recreated images.
    x = imgs.permute(0,2,3,1).cpu().numpy()
    for i in range(len(x)):
        im = x[i]
        scipy.misc.imsave(os.path.join(file_path, str(i)+'.jpg'), im)
        
def BLEU_reward(imgs, hypothesis, save_imgs, ground_truth):
    # Note: images not used.
    
    with open(os.path.join('/data2/adsue/caption_data', 'WORDMAP_' + data_name + '.json'), 'r') as j:
        word_map = json.load(j)
    
    img_caps = ground_truth.tolist()
    img_captions = list(
        map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
            img_caps))  # remove <start> and pads
    
    bleu_rewards = torch.Tensor([sentence_bleu([ref], hyp) for (ref, hyp) in zip(img_captions, hypothesis)]).to(device)
    return bleu_rewards
