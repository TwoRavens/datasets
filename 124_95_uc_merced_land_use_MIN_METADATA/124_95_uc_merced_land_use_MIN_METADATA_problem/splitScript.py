import sys
import numpy as np
import pandas as pd


def build_split_array(y, test_size=0.2):
    num_classes = len(np.unique(y))
    split_array = []
    
    for i in range(num_classes):
        inds = np.where(y == i)[0]
        N = len(inds)
        num_train = int((1-test_size)*N)
        num_test = N-num_train
        split_array.append(['TRAIN']*num_train + ['TEST']*num_test)

    split_array = np.hstack(split_array)        
    return split_array


def write_data_splits_csv_file(learning_data_csv_file, data_splits_csv_file, test_size=0.2):
    ldf = pd.read_csv(learning_data_csv_file)
    split_array = build_split_array(ldf['label'], test_size)
    num_images = len(split_array)

    split_array = split_array[:, None]
    repeat_array = np.zeros((num_images, 1), dtype='int32')
    fold_array = np.zeros((num_images, 1), dtype='int32')
    data = np.concatenate((split_array, repeat_array, fold_array), axis=1)

    df = pd.DataFrame(data, columns=['type', 'repeat', 'fold'])
    df.index.name = 'd3mIndex'
    df.to_csv(data_splits_csv_file)


if __name__ == "__main__":
    write_data_splits_csv_file(sys.argv[1], sys.argv[2])
