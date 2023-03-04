import argparse

def get_input_args():
    parser = argparse.ArgumentParser(description = 'Train a neural network')
    parser.add_argument('data_dir', type = str, help = 'path to the directory containing datasets', default = 'flowers/')
    parser.add_argument('--save_dir', type = str, help = 'path to the directory to save checkpoints', default = 'checkpoint/')
    parser.add_argument('--arch', type = str, help = 'Model Architecture (default: vgg16)', default = 'vgg16')
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units in the classifier (default: 4096)', default=4096)
    parser.add_argument('--learning_rate', type=float, help='Learning rate (default: 0.001)', default=0.001)
    parser.add_argument('--epochs', type=int, help='Number of epochs to train (default: 10)', default=10)
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

def get_input_args_predict():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('image_path', type=str, help='path to image file')
    parser.add_argument('checkpoint', type=str, help='path to saved checkpoint file')
    parser.add_argument('--top_k', type=int, default=1, help='return top K most likely classes')
    parser.add_argument('--category_names', type=str, default=None, help='path to mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='use GPU for inference')
    return parser.parse_args()