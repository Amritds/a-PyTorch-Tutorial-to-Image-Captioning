from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/scratch/scratch5/adsue/caption_data/dataset_coco.json',
                       image_folder='/scratch/scratch5/adsue/caption_data/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/scratch/scratch5/adsue/caption_data/',
                       max_len=50)
