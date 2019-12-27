cd ~

git clone https://github.com/Amritds/a-PyTorch-Tutorial-to-Image-Captioning.git
cd a-PyTorch-Tutorial-to-Image-Captioning
rm -rf StackGAN-Pytorch
git clone https://github.com/Amritds/StackGAN-Pytorch.git


working_dir="/scratch/scratch5/adsue"

cd $working_dir

mkdir caption_data
cd caption_data 

wget cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip
rm -rf caption_datasets.zip

wget images.cocodataset.org/zips/train2014.zip
wget images.cocodataset.org/zips/val2014.zip
unzip train2014.zip
unzip val2014.zip
rm -rf *.zip

## Manually download (use get_drive_file.sh utility) from: https://drive.google.com/file/d/0B0ywwgffWnLLeVNmVVV6OHBDUFE/view to: /media/ssd/caption_data/text_encoder/
## Manually download (use get_drive_file.sh utility) from: https://drive.google.com/file/d/0B0ywwgffWnLLeVNmVVV6OHBDUFE/view to: /media/ssd/caption_data/models/coco/

#cd $working_dir/a-PyTorch-Tutorial-to-Image-Captioning

# Run:  python create_input_files.py (submit job)
# Run:  python train.py (submit job)


