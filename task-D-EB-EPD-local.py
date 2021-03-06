import numpy as np
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from trainer.datasets import get_data
from trainer.models import get_model
import time
import argparse
from tensorflow.python.lib.io import file_io
from keras.datasets import cifar10, mnist, cifar100
from keras.utils import np_utils
from trainer.EPD import LossLearningRateScheduler
from trainer.EventBasedLE import EventBasedLearningEpochStopper


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

    epochs_training = 60
    total_epoch_training = 300
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
                    validation_data=(X_test, y_test), callbacks = [LossLearningRateScheduler(base_lr = 0.01, kd_coef = 5, loss_zero = 0, eventbasedLR = True), EventBasedLearningEpochStopper(alpha_threshold = -0.001, loss_zero = 0, lookbackward_epoch = 5, accumulated_epoch = 0, limit_epoch = total_epoch_training, ending_epoch = 10)]
                    )
    round = 0
    turn = 1
    while len(h.history['loss']) < total_epoch_training:
        for i in np.arange(1, steps):
            turn += 1
            if len(h.history['loss']) >= total_epoch_training:
                break
            if i == 0 and round != 0:
                sub_un_selected_list = un_selected_index[0:training_steps]
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
                            validation_data=(X_test, y_test), callbacks = [LossLearningRateScheduler(base_lr = 0.01, kd_coef = 5, loss_zero = h.history['loss'][0], eventbasedLR = True), EventBasedLearningEpochStopper(alpha_threshold = -0.001, loss_zero = h.history['loss'][0], lookbackward_epoch = 5, accumulated_epoch = len(h.history['loss']), limit_epoch = total_epoch_training, ending_epoch = 10)]
                            )
            h  = combine_result(h, h_training_epoch)

        round += 1
    print(turn)
    return h.history




##Running the app
if __name__ == "__main__":
    result_history = main()
    np.save('result_history.npy', result_history)
