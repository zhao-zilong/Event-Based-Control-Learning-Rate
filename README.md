# Exponentional/Proportional Derivative (E/PD) and Double Event-Based E/PD learning rate algorithms
Code for paper [Feedback Control for Online Training of Neural Networks](https://hal.archives-ouvertes.fr/hal-02115916v2/document) and [Event-Based Control for Online Training of Neural Networks](https://ieeexplore.ieee.org/document/9042857) 
which published in *2019 IEEE Conference on Control Technology and Applications (CCTA)* and *The IEEE Control Systems Letters*.

The basic code structure is designed for google cloud ML-engine platform, the code is based on Python 3.5, Tenserflow and Keras.

## Feedback Control for Online Training of Neural Networks (E/PD)
This paper introduces a Exponentional/Proportional Derivative (E/PD) learning rate algorithm, which dynamically adopt learning rate based on the training/testing loss.
The algorithm is defined in `trainer/EPD.py`. One demo for testing E/PD algorith is provided in `trainer/task-EPD.py`, if user does not want to run it on google cloud, one can run direclty:

```python3
python task-EPD-local.py
```
This code will run E/PD on CIFAR10, the CIFAR10 dataset will be chunked into 5 subsets, each subset will be trained 60 epochs.

## Event-Based Control for Online Training of Neural Networks (Double Event-Based E/PD)
This paper introduces two algorithms:

- Event-Based Learning Rate (EBLR)
- Event-Based Learning Epoch (EBLE)

Code for EBLR is defined in `trainer/EPD.py`, there is a parameter `eventbasedLR`, it will turn on/off this event-based control by setting the parameter True/False.
Code for EBLE is defined in `trainer/EventBasedLE.py`.
One demo for testing these two event-based control is provided in `trainer/task-D-EB-EPD.py`, if user does not want to run it on google cloud, one can run direclty:

```python3
python task-D-EB-EPD-local.py
```
This code will run E/PD and (EBLR & EBLE) on CIFAR10, the CIFAR10 dataset will be chunked into 5 subsets, maximum successive training epochs for one subset is 60, maximum total training epoch sets to 300.

## Usage
Import the customized callbacks:
```
from .EPD import LossLearningRateScheduler
from .EventBasedLE import EventBasedLearningEpochStopper
```

As our algorithms are all based on SGD, so the optimizer must be defined as (value of parameter `lr` here does not matter, because we will adjust its value before the training, setting by our algorithm):
```
optimizer = SGD(lr=0.01, decay=0, momentum=0)
```
To use the callbacks, we can call it as that:

```
h  =   model.fit_generator(datagen.flow(X_clean_iteration, y_clean_iteration, batch_size=batch_size),
                    steps_per_epoch=X_clean_iteration.shape[0]//batch_size, epochs=epochs_training,
                    validation_data=(X_test, y_test), 
                    callbacks = [LossLearningRateScheduler(base_lr = 0.01, kd_coef = 5, eventbasedLR = True), 
                    EventBasedLearningEpochStopper(alpha_threshold = -0.001, loss_zero = 0, lookbackward_epoch = 5, accumulated_epoch = 0, limit_epoch = total_epoch_training, ending_epoch = 10)]
                    )
```

One thing to notice that for parameter `loss_zero` in ***EventBasedLearningEpochStopper***, if this is the first to run the training process, as we do not have loss_zero (the loss value of first training epoch), we must set it to ***0***. Once we have the loss_zero value, we will pass this value when we train the model.

## Citing
If you use E/PD in your research, please cite:
```text
@INPROCEEDINGS{zhao2019epd, 
author={Z. {Zhao} and S. {Cerf} and B. {Robu} and N. {Marchand}}, 
booktitle={2019 IEEE Conference on Control Technology and Applications (CCTA)}, 
title={Feedback Control for Online Training of Neural Networks}, 
year={2019}, 
pages={136-141}, 
doi={10.1109/CCTA.2019.8920662}, 
ISSN={null}, 
month={Aug},}
```
If you use EBLR or EBLE in your research, please cite:
```text
@ARTICLE{zhao2020lcss, 
author={Z. {Zhao} and S. {Cerf} and B. {Robu} and N. {Marchand}}, 
journal={IEEE Control Systems Letters}, 
title={Event-Based Control for Online Training of Neural Networks}, 
year={2020}, 
doi={10.1109/LCSYS.2020.2981984}, 
ISSN={2475-1456}, 
}
```
## License
[Apache 2.0](./LICENSE)



