from __future__ import absolute_import
from __future__ import print_function

import keras
from keras import backend as K
import numpy as np


class LossLearningRateScheduler(keras.callbacks.History):
    """
    E (Exponential)/PD (Proportional Derivative): A learning rate scheduler that relies on changes in loss function
    eventbasedLR: Turn on/off Event Based Learning Rate
    loss_type: by default, training loss is used, if one wants to use testing loss: loss_type = 'val_loss'
    """

    def __init__(self, base_lr = 0.01, kd_coef = 5.0, loss_zero = 0, eventbasedLR = False, loss_type = 'loss'):

        super(LossLearningRateScheduler, self).__init__()

        self.base_lr = base_lr
        self.kd_coef = kd_coef
        self.loss_type = loss_type
        self.loss_zero = loss_zero
        self.tipping_point = False
        self.last_lr = base_lr
        self.kp = 0.0
        self.eventbasedLR = eventbasedLR


    # this function is called before training
    def on_epoch_begin(self, epoch, logs=None):
        if len(self.epoch) > 1:
            print(self.history[self.loss_type])
        if len(self.epoch) == 1 and self.loss_zero == 0:
            self.loss_zero = self.history[self.loss_type][0]

        if len(self.epoch) > 1 and not self.tipping_point:
            target_loss = self.history[self.loss_type]
            # we should also limit that lr is not bigger than 4
            if target_loss[-1] > target_loss[-2] or self.base_lr*(2**(len(self.epoch))) > 4.0:
                self.tipping_point = True
                self.kp = self.base_lr*(2**(len(self.epoch)-2))
                self.last_lr = self.base_lr*(2**(len(self.epoch)-2))
                print(' '.join(('PD phase (tipping_point triggered): Setting learning rate to', str(self.base_lr*(2**(len(self.epoch)-2))))))
                K.set_value(self.model.optimizer.lr, self.base_lr*(2**(len(self.epoch)-2)))
                return K.get_value(self.model.optimizer.lr)

        if (not self.tipping_point):
            self.last_lr = self.base_lr*(2**len(self.epoch))
            print(' '.join(('E phase: Setting learning rate to', str(self.base_lr*(2**len(self.epoch))))))
            K.set_value(self.model.optimizer.lr, self.base_lr*(2**len(self.epoch)))
        else:
            target_loss = self.history[self.loss_type]
            lr = 0
            if self.eventbasedLR and len(target_loss) > 1 and target_loss[-1] < target_loss[-2]:
                lr = self.last_lr
                print(' '.join(('PD phase (eventbasedLR triggered): Setting learning rate to', str(lr))))
            else:
                lr = self.kp * target_loss[-1] / self.loss_zero - self.base_lr * self.kd_coef * (target_loss[-1] - target_loss[-2]) / self.loss_zero
                if lr < 0:
                    lr = self.kp * target_loss[-1] / self.loss_zero
                print(' '.join(('PD phase: Setting learning rate to', str(lr))))
            self.last_lr = lr

            K.set_value(self.model.optimizer.lr, lr)

        return K.get_value(self.model.optimizer.lr)




def main():
    return

if __name__ == '__main__':
    main()
