from __future__ import absolute_import
from __future__ import print_function

import keras
from keras import backend as K
import numpy as np


class EventBasedLearningEpochStopper(keras.callbacks.History):
    """
    Double EventBased control for learning epochs
    loss_type: by default, training loss is used, if one wants to use testing loss: loss_type = 'val_loss'
    """

    def __init__(self, alpha_threshold = -0.001, loss_zero = 0,lookbackward_epoch = 5, accumulated_epoch = 0, limit_epoch = 300, ending_epoch = 10, loss_type = 'loss'):

        super(EventBasedLearningEpochStopper, self).__init__()

        self.alpha_threshold = alpha_threshold
        self.lookbackward_epoch = lookbackward_epoch
        self.loss_type = loss_type
        self.loss_zero = loss_zero
        self.accumulated_epoch = accumulated_epoch
        self.limit_epoch = limit_epoch
        self.ending_epoch = ending_epoch



    # this function is called before training
    # epoch starts from 0
    # self.epoch and self.history are different in on_epoch_begin and on_epoch_end, see the difference between DoubleEB and EPD
    def on_epoch_end(self, epoch, logs=None):
        self.accumulated_epoch += 1
        if self.accumulated_epoch >= self.limit_epoch:
            self.model.stop_training = True
            return
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        # no need to reset loss0 if it's given
        if epoch == 0 and self.loss_zero == 0:
            self.loss_zero = self.history[self.loss_type][0]
        if epoch >= (self.lookbackward_epoch-1):
            target_loss = self.history[self.loss_type]
            x = np.arange(1,(self.lookbackward_epoch+1),1.0)
            y = np.array([i/self.loss_zero for i in target_loss[-self.lookbackward_epoch:]])
            slope = np.polyfit(x, y, 1)[0]
            print("Slope: "+str(slope))
            if slope > self.alpha_threshold and (self.limit_epoch - self.accumulated_epoch) > self.ending_epoch:
                self.model.stop_training = True

def main():
    return

if __name__ == '__main__':
    main()
