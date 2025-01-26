from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)

    # TBD - change accordingly if you want to extract the data
    path = '/Users/mouhcine/Documents/cs7643-project-data'

    create_input_files(dataset='coco',
                       karpathy_json_path=path+'/caption_datasets/dataset_coco.json',
                       image_folder=path,
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='./data',
                       max_len=50)
