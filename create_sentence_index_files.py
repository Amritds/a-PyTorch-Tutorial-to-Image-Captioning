import json
import os

#data_path = './'
data_path = '/data2/adsue/caption_data/'

with open(os.path.join(data_path,'TRAIN_CAPTIONS_coco_5_cap_per_img_5_min_word_freq.json'),'r') as f:
    encodings = json.load(f)

# Load word map from JSON
with open(os.path.join(data_path, 'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'), 'r') as j:
    word_map = json.load(j)
        
encodings = [[w for w in se if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}] for se in encodings]  
        
# Create the reverse word map
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word    

sentences = [' '.join([rev_word_map[ind] for ind in sent]) for sent in encodings]

sentence_index = {}
sentence_list = []
for (i,s) in enumerate(sentences):
    sentence_list.append(s)
    
    print(i+1,'. ',s)
    
    if (i+1)%5==0:
        print('\n\n')
        sentence_index.update({sent:sentence_list for sent in sentence_list})
        sentence_list = []
        
with open(os.path.join(data_path, 'TRAIN_CAPTIONS_sentence_index.json'),'w') as f:
    json.dump(sentence_index, f)
