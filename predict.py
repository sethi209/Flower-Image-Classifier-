from get_input_args import get_input_args_predict
from functions import predict

def main():
    in_args = get_input_args_predict()
    predict(in_args.image_path, in_args.checkpoint, in_args.top_k, in_args.gpu, in_args.category_names)


if __name__ == "__main__":
    main()