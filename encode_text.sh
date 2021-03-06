#!/bin/bash

#
# Extract text embeddings from the encoder
#

CUB_ENCODER=coco_gru18_bs64_cls0.5_ngf128_ndf128_a10_c512_80_net_T.t7 \
CAPTION_PATH=/media/ssd/caption_data/mini_batch_captions \
GPU=1 \

export CUDA_VISIBLE_DEVICES=${GPU}

net_txt=/media/ssd/caption_data/text_encoder/${CUB_ENCODER} \
queries=${CAPTION_PATH}.txt \
filenames=${CAPTION_PATH}.t7 \
th ./get_embedding.lua
