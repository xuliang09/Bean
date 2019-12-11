from data_transformer import *

if __name__ == '__main__':
    main(src_dataset='./FB15K', dst_folder='./FB15K_small', new_train_total=20000, new_test_total=200,
         new_valid_total=200)
