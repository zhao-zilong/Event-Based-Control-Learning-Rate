import numpy as np
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from .datasets import get_data
from .models import get_model
import time
import argparse
from tensorflow.python.lib.io import file_io
from keras.datasets import cifar10, mnist, cifar100
from keras.utils import np_utils
from .EPD import LossLearningRateScheduler
from keras.callbacks import History

def combine_result(h, h_training_epoch):
    h.history['acc'] += h_training_epoch.history['acc']
    h.history['loss'] += h_training_epoch.history['loss']
    h.history['val_acc'] += h_training_epoch.history['val_acc']
    h.history['val_loss'] += h_training_epoch.history['val_loss']
    return h;

def main(job_dir,**args):
    NUM_CLASSES = {'mnist': 10, 'svhn': 10, 'cifar-10': 10, 'cifar-100': 100}
    dataset = "cifar-10"
    X_train, y_train, X_test, y_test, un_selected_index = get_data(dataset, random_shuffle=False)

    image_shape = X_train.shape[1:]
    model = get_model(dataset, input_tensor=None, input_shape=image_shape, num_classes=NUM_CLASSES[dataset])
    optimizer = SGD(lr=0.01, decay=0, momentum=0)


    datagen = ImageDataGenerator(
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip = False,  # randomly flip images
        )
    datagen.fit(X_train)

    epochs_training = 10
    batch_size = 128
    training_steps = 10000
    un_selected_index = range(X_train.shape[0])
    steps = int(np.floor(len(un_selected_index) / training_steps))


    sub_un_selected_list = un_selected_index[0:training_steps]
    X_clean_iteration = X_train[sub_un_selected_list]
    y_clean_iteration = y_train[sub_un_selected_list]
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    h  =   model.fit_generator(datagen.flow(X_clean_iteration, y_clean_iteration, batch_size=batch_size),
                    steps_per_epoch=X_clean_iteration.shape[0]//batch_size, epochs=epochs_training,
                    validation_data=(X_test, y_test), callbacks = [History(), LossLearningRateScheduler(base_lr = 0.01, kd_coef = 5, loss_zero = 0)]
                    )

    for i in np.arange(1, steps):
        if i != steps - 1:
            sub_un_selected_list = un_selected_index[i*training_steps:(i+1)*training_steps]
        else:
            sub_un_selected_list = un_selected_index[i*training_steps:]

        X_clean_iteration = X_train[sub_un_selected_list]
        y_clean_iteration = y_train[sub_un_selected_list]


        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['acc'])
        h_training_epoch  =   model.fit_generator(datagen.flow(X_clean_iteration, y_clean_iteration, batch_size=batch_size),
                        steps_per_epoch=X_clean_iteration.shape[0]//batch_size, epochs=epochs_training,
                        validation_data=(X_test, y_test), callbacks = [History(), LossLearningRateScheduler(base_lr = 0.01, kd_coef = 5, loss_zero = h.history['loss'][0])]
                        )
        h  = combine_result(h, h_training_epoch)


    np.save(file_io.FileIO(job_dir + 'result/EPD_CIFAR10_001.npy', 'w'), h.history)




##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
