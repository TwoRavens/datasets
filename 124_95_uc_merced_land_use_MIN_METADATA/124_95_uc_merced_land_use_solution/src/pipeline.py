import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
import skimage.io as skimg_io
import skimage.transform as skimg_tfm
from sklearn.metrics import accuracy_score
from d3mds import D3MDS

import tensorflow as tf
import keras
from keras.applications.densenet import DenseNet121
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


def load_imgs_from_files(img_files, img_dir):
    X = []
    for i, img_file in enumerate(img_files):
        if i % 1000 == 0:
            print('Loading Image %d' % i)
        img = skimg_io.imread(os.path.join(img_dir, img_file))        
        X.append(img)
    X = np.stack(X, axis=0)
    return X

def resize_imgs(imgs, new_size=(224, 224)):
    imgs_r = []
    for img in imgs:
        img_r = skimg_tfm.resize(img, new_size).astype('float32')
        imgs_r.append(img_r)
    imgs_r = np.stack(imgs_r, axis=0)
    return imgs_r

def build_model(num_classes=21):
    base_model = DenseNet121(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = Dense(num_classes, activation='softmax', name='fc21')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def evaluate_on_test_set(model, X_test, y_test):
    preds = np.argmax(model.predict(X_test), axis=1)
    acc_test = accuracy_score(y_test, preds)
    print('Test Accuracy: %.2f' % (acc_test * 100))
    return acc_test, preds

def write_predictions_csv_file(inds, preds, prediction_filename):     
    df = pd.DataFrame(preds, index=inds, columns=['label'])
    df.to_csv(prediction_filename, index_label='d3mIndex')

def write_scores_csv_file(metric_dict, score_filename):    
    metric_names = []
    metric_values = []
    for metric_name, metric_value in metric_dict.items():
        metric_names.append(metric_name)
        metric_values.append(metric_value)
    metric_names = np.array(metric_names)
    metric_values = np.array(metric_values)

    df = pd.DataFrame(np.concatenate((metric_names[:, None], metric_values[:, None]), axis=1), columns=['metric', 'value'])
    df.to_csv(score_filename)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to train CNN pipeline on UC Merced Land Use dataset.')
    parser.add_argument('--dataset_root', dest='dataset_root',
                        help='Folder containing dataset to run pipeline.')
    parser.add_argument('--model_dir', dest='model_dir', default='./models/',
                        help='Location to load and save model files.')
    parser.add_argument('--use_gpu', dest='use_gpu',
                        action='store_true', help='Use GPU when training model.')
    parser.add_argument('--which_gpu', dest='which_gpu',
                        default='0', help='GPU to use when running.')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dataset_root = args.dataset_root    
    model_dir = args.model_dir
    use_gpu = args.use_gpu
    which_gpu = args.which_gpu
    
    # Fix random seed
    seed = 42
    np.random.seed(seed)

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = which_gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    print('Load DATA')    
    data_path = glob.glob(os.path.join(dataset_root, "*_dataset"))[0]
    problem_path = glob.glob(os.path.join(dataset_root, "*_problem"))[0]
    d3mds = D3MDS(data_path, problem_path)

    print('\nLoad train data')
    df_train = d3mds.get_train_data()
    media_dir = os.path.join(data_path, 'media')
    X_train = load_imgs_from_files(df_train['image'], media_dir)
    y_train = d3mds.get_train_targets()
    X_train = X_train.astype('float32') / 255.0
    print(X_train.shape, y_train.shape)

    print('Load test data')
    df_test = d3mds.get_test_data()
    X_test = load_imgs_from_files(df_test['image'], media_dir)
    y_test = d3mds.get_test_targets()
    X_test = X_test.astype('float32') / 255.0
    print(X_test.shape, y_test.shape)

    print('Resizing train images')
    X_train = resize_imgs(X_train, (224, 224))
    print(X_train.shape)

    print('Resizing test images')
    X_test = resize_imgs(X_test, (224, 224))
    print(X_test.shape)

    num_classes = len(np.unique(y_train))
    model = build_model(num_classes=num_classes)
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,                  
                  optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9),
                  metrics=['accuracy'])

    y_train_one_hot = to_categorical(y_train, num_classes=num_classes)

    batch_size = 32
    epochs = 10

    model_file = 'model_weights.h5'
    model_full_path = os.path.join(model_dir, model_file)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if os.path.exists(model_full_path):
        print('Loading model from %s' % model_full_path)
        model.load_weights(model_full_path)
    else:        
        print('Training model')
        train_datagen = ImageDataGenerator(        
                rotation_range=10,
                width_shift_range=20,
                height_shift_range=20,
                horizontal_flip=True)
        train_generator = train_datagen.flow(X_train, y_train_one_hot, 
                                             batch_size=batch_size, seed=seed)

        model.fit_generator(
                train_generator,
                steps_per_epoch=X_train.shape[0] // batch_size,
                epochs=epochs)

        model.save(model_full_path)

    print('Performance on TEST sest')
    acc_test, preds_test = evaluate_on_test_set(model, X_test, y_test)

    print('Writing predictions to .csv file.')
    cur_dir = os.getcwd()
    predictions_file = os.path.join(cur_dir, 'predictions.csv')
    write_predictions_csv_file(df_test.index, preds_test, predictions_file)

    print('Writing scores to .csv file.')
    metric_dict = {'accuracy': acc_test}
    scores_file = os.path.join(cur_dir, 'scores.csv')
    write_scores_csv_file(metric_dict, scores_file)
