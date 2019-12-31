import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from step_wise_decoding import get_hypothesis_greedy
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

# Data parameters
data_folder = '/scratch/scratch5/adsue/caption_data'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

print('Device type: ', device)

# Training parameters (General)
start_epoch = 0
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 64
workers = 0  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper

desired_training_type = 'XE'
training_type = desired_training_type

reward_map ={'RL_recreation' : image_comparison_reward,
             'RL_bleu' : BLEU_reward}

best_bleu4 = 0.  # BLEU-4 score right now
best_reward = 0. # Avg Reward right now

print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none

# Training parameters (Cross Entropy Maximization)
epochs_XE = 40  # number of epochs to train for (if early stopping is not triggered)

# Training parameters (Expected Reward Maximization)
epochs_RL = 0  # number of epochs to train for (if early stopping is not triggered)


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
        
        if num_of_devices > 1: 
            encoder = nn.DataParallel(encoder) #enabling data parallelism
            decoder = nn.DataParallel(decoder)
            
        print('Number of devices available: ', num_of_devices)
    
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        training_type = checkpoint['training_type']
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        best_reward = checkpoint['best_reward']
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

            # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
            if epochs_since_improvement == 20:
                break
            if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
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
            recent_bleu4 = validate_XE(val_loader=val_loader,
                                       encoder=encoder,
                                       decoder=decoder,
                                       criterion=criterion)

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


    # BEGIN RL TRAINING -------------------------------
    elif training_type[:2] == 'RL':

        for epoch in range(start_epoch, epochs_RL):

            # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
            if epochs_since_improvement == 20:
                break
            if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
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
            recent_reward = validate_RL(val_loader=val_loader,
                                        encoder=encoder,
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

        # Forward prop.
        encoder_out = encoder(imgs)
        
        (hypotheses, sum_top_scores) = get_hypothesis_greedy(encoder_out, sample=True)
        (hyp_max, _) = get_hypothesis_greedy(encoder_out, sample=False)
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
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Maximizing sum of Expected Rewards\n'
                  'Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


        
        
def validate_XE(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

            
            
        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4

def validate_RL(val_loader, encoder, decoder, reward_function):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    
    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    counter = 0
    sum_avg_rewards = 0

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            encoder_out = encoder(imgs)
        
            # Notice, we are not sampling here! We are using greedy decoding.
            (hypotheses, sum_top_scores) = get_hypothesis_greedy(encoder_out, sample=False)
     
            batch_time.update(time.time() - start)

            batch_avg_reward = reward_function(imgs, hypotheses, caps).mean()
            sum_avg_rewards += batch_avg_reward
            counter +=1

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Minibatch Average Reward:{batch_avg_reward:.3f}')

            
        avg_reward = sum_avg_rewards/counter

        print(
            '\n * Epoch Average Reward- {avg_reward:.3f}\n'.format(
                avg_reward=avg_reward))

    return avg_reward


if __name__ == '__main__':
    main()
