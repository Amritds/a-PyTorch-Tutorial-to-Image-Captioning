import time
import sys
import os
import gc
import yaml
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from step_wise_decoding import get_hypothesis_greedy
from eval import evaluate
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

#---------------------------------------- Init some global variablles -------------------------------------------------------------

reward_map ={'RL_recreation' : image_comparison_reward,
             'RL_bleu' : BLEU_reward,
             'RL_cider': cider_reward,
             'RL_recreation_cider_balanced': image_comparison_cider_reward_balanced,
             'RL_recreation_cider_not_balanced': image_comparison_cider_reward_not_balanced}

best_bleu4 = 0.  # BLEU-4 score right now
best_reward = 0. # Avg Reward right now

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

#---------------------------------------- Load expriment parameters from CFG file -------------------------------------------------
with open(sys.argv[1], 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# Data parameters
data_folder = cfg['data_folder']  # folder with data files saved by create_input_files.py
data_name = cfg['data_name']  # base name shared by data files

# Model parameters
emb_dim = cfg['emb_dim'] # dimension of word embeddings
attention_dim = cfg['attention_dim']  # dimension of attention linear layers
decoder_dim = cfg['decoder_dim']  # dimension of decoder RNN
dropout = cfg['dropout']

# Training parameters (General)
start_epoch = cfg['start_epoch']
epochs_since_improvement = cfg['epochs_since_improvement']  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = cfg['batch_size']
workers = cfg['workers']  # for data-loading; right now, only 1 works with h5py
encoder_lr = float(cfg['encoder_lr'])  # learning rate for encoder if fine-tuning
decoder_lr = float(cfg['decoder_lr'])  # learning rate for decoder
grad_clip = cfg['grad_clip']  # clip gradients at an absolute value of
alpha_c = cfg['alpha_c']  # regularization parameter for 'doubly stochastic attention', as in the paper

desired_training_type = cfg['training_type']
training_type = desired_training_type

print_freq = cfg['print_freq'] # print training/validation stats every __ batches
fine_tune_encoder = cfg['fine_tune_encoder']  # fine-tune encoder?
checkpoint = cfg['checkpoint'] # path to checkpoint, None if none

# Training parameters (Cross Entropy Maximization)
epochs_XE = cfg['epochs_XE']  # number of epochs to train for (if early stopping is not triggered)

# Training parameters (Expected Reward Maximization)
epochs_RL = cfg['epochs_RL']  # number of epochs to train for (if early stopping is not triggered)

proportion = cfg['proportion']

exp_dir = sys.argv[2]
    
# ------------------------------------ Training and validation code ---------------------------------------------------------------
def main():
    """
    Training and validation.
    """

    global training_type, best_bleu4, best_reward, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, epochs

    print('Loading model now...')
    
    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

        
        num_of_devices = torch.cuda.device_count()
        
        if num_of_devices > 1 and desired_training_type == 'XE': 
            encoder = nn.DataParallel(encoder) #enabling data parallelism
            decoder = nn.DataParallel(decoder)
            
        print('Number of devices available: ', num_of_devices)
    
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        training_type = checkpoint['training_type']
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        best_reward = checkpoint['reward']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

        # Reset training parameters if desired training type and training type do not match.
        if training_type != desired_training_type:
            training_type = desired_training_type
            epochs_since_improvement = 0
            start_epoch=0
            decoder = decoder.module
            encoder = encoder.module
            best_reward=0.

            # Reset optimizers
            decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
   
    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    print('Starting training epochs now...')
    
    # Training to maximize cross entropy and maximize sum of expected rewards.
    training_epochs(encoder, 
                    decoder,
                    encoder_optimizer,
                    decoder_optimizer,
                    train_loader,
                    val_loader,
                    device)



def training_epochs(encoder, decoder, encoder_optimizer, decoder_optimizer, train_loader, val_loader, device):
    """
    Runs training epochs with checkpointing.
     :param train_type: Objective of:
                             (1)Maximizing Cross-Entropy
                             (2)Maximizing Expected Reward using the SCST algorithm.
    """

    global training_type, best_bleu4, best_reward, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, epochs_XE, epochs_RL

    # BEGIN XE Training --------------------------------------------
    if training_type == 'XE':
        for epoch in range(start_epoch, epochs_XE):

            # Decay learning rate after 3 consecutive epochs, and terminate training after 30 if no improvement is seen.
            if epochs_since_improvement == 30:
                break
            if epoch % 3 == 0 and epoch !=0:
                adjust_learning_rate(decoder_optimizer, 0.8)
                if fine_tune_encoder:
                    adjust_learning_rate(encoder_optimizer, 0.8)        
            
            # One epoch's training using the Cross Entropy Loss function ------------------------------------
            criterion = nn.CrossEntropyLoss().to(device)
            
            train_XE(train_loader=train_loader,
                     encoder=encoder,
                     decoder=decoder,
                     criterion=criterion,
                     encoder_optimizer=encoder_optimizer,
                     decoder_optimizer=decoder_optimizer,
                     epoch=epoch)

        #-------------------------------------------------------------------------------------------------

            # One epoch's validation
            try:
                recent_bleu4, _ = validate(encoder=encoder,
                                           decoder=decoder)
            except:
                print('Validation failed.')
                recent_bleu4 = -1
                
            # Check if there was an improvement
            is_best = recent_bleu4 > best_bleu4
            best_bleu4 = max(recent_bleu4, best_bleu4)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            # Save checkpoint
            save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, is_best, training_type, bleu4=recent_bleu4)    

    elif training_type[:5]== 'XE_RL':
        
        for epoch in range(start_epoch, epochs_RL):

            # Decay learning rate after 3 consecutive epochs, and terminate training after 30 if no improvement is seen.
            if epochs_since_improvement == 30:
                break
            if epoch % 3 == 0 and epoch !=0:
                adjust_learning_rate(decoder_optimizer, 0.8)
                if fine_tune_encoder:
                    adjust_learning_rate(encoder_optimizer, 0.8)

            # One epoch's training maximizing the sum of expected rewards loss function
            
            criterion_xe = nn.CrossEntropyLoss().to(device)
            
            reward_function = reward_map[training_type[3:]]
            criterion_rl = RL_loss(reward_function).to(device)
            

            train_XE_RL(train_loader=train_loader,
                        encoder=encoder,
                        decoder=decoder,
                        criterion_xe=criterion_xe,
                        criterion_rl=criterion_rl,
                        encoder_optimizer=encoder_optimizer,
                        decoder_optimizer=decoder_optimizer,
                        epoch=epoch,
                        proportion=proportion)

        #-------------------------------------------------------------------------------------------------

            # One epoch's validation (Computes average reward for the epoch)
            _, recent_reward = validate(encoder=encoder,
                                        decoder=decoder,
                                        reward_function=reward_function)
            
                
            # Check if there was an improvement 
            is_best = recent_reward > best_reward
            best_reward = max(recent_reward, best_reward)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            # Save checkpoint
            save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                            decoder_optimizer, is_best, training_type, reward=recent_reward)  


    # BEGIN RL TRAINING -------------------------------
    elif training_type[:2] == 'RL':

        for epoch in range(start_epoch, epochs_RL):

            # Decay learning rate after 3 consecutive epochs, and terminate training after 30 if no improvement is seen.
            if epochs_since_improvement == 30:
                break
            if epoch % 3 == 0 and epoch !=0:
                adjust_learning_rate(decoder_optimizer, 0.8)
                if fine_tune_encoder:
                    adjust_learning_rate(encoder_optimizer, 0.8)

            # One epoch's training maximizing the sum of expected rewards loss function
            
            reward_function = reward_map[training_type]
            criterion = RL_loss(reward_function).to(device)

            train_RL(train_loader=train_loader,
                     encoder=encoder,
                     decoder=decoder,
                     criterion=criterion,
                     encoder_optimizer=encoder_optimizer,
                     decoder_optimizer=decoder_optimizer,
                     epoch=epoch)

        #-------------------------------------------------------------------------------------------------

            # One epoch's validation (Computes average reward for the epoch)
            _, recent_reward = validate(encoder=encoder,
                                        decoder=decoder,
                                        reward_function=reward_function)
            
                
            # Check if there was an improvement 
            is_best = recent_reward > best_reward
            best_reward = max(recent_reward, best_reward)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            # Save checkpoint
            save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                            decoder_optimizer, is_best, training_type, reward=recent_reward)  


def train_XE(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training for Cross-Entropy maximization.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder modelfrom nltk.translate.bleu_score import corpus_bleu
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)
                
        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        
        #print('Forward prop...')
        
        # Forward prop.
        encoder_out = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoder_out, caps, caplens)

        # Sort gathered results in decreasing order -- corrects for jumbling of using multiple GPUs
        decode_lengths, decode_sort_ind = decode_lengths.sort(dim=0, descending=True)
        decode_lengths = decode_lengths.tolist()
        scores = scores[decode_sort_ind]
        caps_sorted = caps_sorted[decode_sort_ind]
        alphas = alphas[decode_sort_ind]
        sort_ind = sort_ind[decode_sort_ind]

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        #print('Back prop...')
        
        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Maximizing Cross-Entropy\n'
                  'Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
        
            
def train_RL(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training for Expected rewards maximization.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """
    
    decoder.eval()  # evaluation mode (dropout and batchnorm is not used)
    encoder.eval()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per caption)
    

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
    
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        encoder_out = encoder(imgs)
        
        (hypotheses, sum_top_scores) = get_hypothesis_greedy(encoder_out, decoder, sample=True)
        (hyp_max, _) = get_hypothesis_greedy(encoder_out, decoder, sample=False)
        # Calculate loss
        loss = criterion(imgs, caps, hypotheses, hyp_max, sum_top_scores)

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        
        losses.update(loss.item(), batch_size)
       
        batch_time.update(time.time() - start)

        start = time.time()
        
        # Print status
        if i % print_freq == 0:
            print('Maximizing sum of Expected Rewards\n'
                  'Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          loss=losses))
        
        # Free memory.
        del loss
        del sum_top_scores
        del hypotheses
        del hyp_max
        gc.collect()      
        
def train_XE_RL(train_loader, encoder, decoder, criterion_xe, criterion_rl, encoder_optimizer, decoder_optimizer, epoch, proportion):
    """
    Performs one epoch's training for Expected rewards maximization.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """
    
    decoder.eval()  # evaluation mode (dropout and batchnorm is not used)
    encoder.eval()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per caption)
    
    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
    
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
            
        if proportion['rl']!=0:
            # Forward prop.--------------------------------------------------------------------------------- RL
            encoder_out = encoder(imgs)
        
            (hypotheses, sum_top_scores, alphas) = get_hypothesis_greedy(encoder_out, decoder, sample=True)
            (hyp_max, _,_) = get_hypothesis_greedy(encoder_out, decoder, sample=False)
            # Calculate loss
            loss_rl = criterion_rl(imgs, caps, hypotheses, hyp_max, sum_top_scores)
            
            # Add doubly stochastic attention regularization
            loss_rl += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            
        else:
            loss_rl = 0
            
        
        if proportion['xe']!=0:
            # Forward prop------------------------------------------------------------------------------------XE
            encoder_out = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoder_out, caps, caplens)

            # Sort gathered results in decreasing order -- corrects for jumbling of using multiple GPUs
            decode_lengths, decode_sort_ind = decode_lengths.sort(dim=0, descending=True)
            decode_lengths = decode_lengths.tolist()
            scores = scores[decode_sort_ind]
            caps_sorted = caps_sorted[decode_sort_ind]
            alphas = alphas[decode_sort_ind]
            sort_ind = sort_ind[decode_sort_ind]

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss_xe = criterion_xe(scores, targets)

            # Add doubly stochastic attention regularization
            loss_xe += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        else:
            loss_xe = 0
        

        # Back prop.--------------------------------------------------------------------------------------------- XE_RL
        
        loss = (proportion['xe']* loss_xe + proportion['rl']*loss_rl)/ (proportion['xe']+proportion['rl'])
        
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        
        losses.update(loss.item(), batch_size)
       
        batch_time.update(time.time() - start)

        start = time.time()
        
        # Print status
        if i % print_freq == 0:
            print('Maximizing sum of Expected Rewards\n'
                  'Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          loss=losses))
        
        # Free memory.
        del loss_xe
        del loss_rl
        del loss
        del sum_top_scores
        del hypotheses
        del hyp_max
        gc.collect()     
        

def validate(encoder, decoder, reward_function=BLEU_reward):
    """
    Performs one epoch's validation for bleu4 and reward specified.

    """
    beam_size=1
    
    decoder.eval()
    encoder.eval()
    
    (bleu4, avg_regeneration_reward, CIDErD) = evaluate(beam_size, encoder, decoder, reward_function)
    
    validation_file = os.path.join(exp_dir, 'validation.txt')
    with open(validation_file, 'a') as f:
        f.write('BLEU4: ' + str(bleu4)+'     ' +
                reward_function.__name__ + ': ' + str(avg_regeneration_reward) + '     ' +
                'CIDErD: ' + str(CIDErD) +
                '\n')
   
    decoder.train()
    encoder.train()
    
    print('BLEU4: ' + str(bleu4)+'     ' +
          reward_function.__name__ + ': ' + str(avg_regeneration_reward) + '     ' +
          'CIDErD: ' + str(CIDErD) +
          '\n')        

    return (bleu4, avg_regeneration_reward)


if __name__ == '__main__':
    main()
