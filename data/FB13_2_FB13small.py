from data_transformer import *

if __name__ == '__main__':
    main(src_dataset='./FB13', dst_folder='./FB13_small', new_train_total=10000, new_test_total=100,
         new_valid_total=100)
