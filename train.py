from get_input_args import get_input_args
from utils import load_data, preprocess_image
from functions import train_model, command_line_arguments


def main():
    
    in_args = get_input_args()
    command_line_arguments(in_args)
    train_model(in_args.data_dir, in_args.save_dir, in_args.arch, in_args.epochs, in_args.learning_rate, in_args.hidden_units, in_args.gpu)

    
if __name__ == "__main__":
    main()