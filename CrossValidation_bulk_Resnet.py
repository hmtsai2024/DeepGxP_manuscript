from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling1D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import numpy as np
import os
from datetime import datetime as dt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from constant_variables import *

print("TensorFlow version:", tf.__version__)

# If you're using Keras via TensorFlow (recommended):
print("Keras version (via TensorFlow):", tf.keras.__version__)

from tensorflow.keras.optimizers import Adam

def MSEresult(y_train, y_val, y_test, pred_train, pred_val, pred_test):
    mse_train= mean_squared_error(y_train, pred_train)
    mse_val = mean_squared_error(y_val, pred_val)
    mse_test = mean_squared_error(y_test, pred_test)
    return mse_train, mse_val, mse_test

def Pearsonresult(y_train, y_val, y_test, pred_train, pred_val, pred_test):
    pearsonr_train = []
    pearsonp_train = []
    pearsonr_val = []
    pearsonp_val = []
    pearsonr_test = []
    pearsonp_test = []

    for j in range(y_train.shape[1]):
        pearsonr_train.append(pearsonr(y_train.iloc[:, j], pred_train.iloc[:, j])[0])
        pearsonp_train.append(pearsonr(y_train.iloc[:, j], pred_train.iloc[:, j])[1])
        pearsonr_val.append(pearsonr(y_val.iloc[:, j], pred_val.iloc[:, j])[0])
        pearsonp_val.append(pearsonr(y_val.iloc[:, j], pred_val.iloc[:, j])[1])
        pearsonr_test.append(pearsonr(y_test.iloc[:, j], pred_test.iloc[:, j])[0])
        pearsonp_test.append(pearsonr(y_test.iloc[:, j], pred_test.iloc[:, j])[1])

    CorMatrix = pd.DataFrame({
        'rppa': y_train.columns,
        'pearsonr_train': pearsonr_train,
        'pearsonp_train': pearsonp_train,
        'pearsonr_val': pearsonr_val,
        'pearsonp_val': pearsonp_val,
        'pearsonr_test': pearsonr_test,
        'pearsonp_test': pearsonp_test
    })
    return CorMatrix

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=187):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling1D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='linear',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=187):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling1D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='linear',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def ResnetModel(x_train, x_val, x_test, y_train, y_val, y_test, numGenes, version, n):
    x_train_np = x_train.to_numpy().reshape(x_train.shape[0], numGenes, 1)
    x_val_np = x_val.to_numpy().reshape(x_val.shape[0], numGenes, 1)
    x_test_np = x_test.to_numpy().reshape(x_test.shape[0], numGenes, 1)

    print(x_train_np.shape)
    print(x_val_np.shape)
    print(x_test_np.shape)

    input_shape = x_train_np.shape[1:]

    # Training parameters
    batch_size = 100  # orig paper trained all networks with batch_size=128
    epochs = 500
    data_augmentation = False
    num_classes = 187

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = False

    n = n

    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = version

    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    # Model name, depth and version
    model_type = 'ResNet%dv%d' % (depth, version)

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    if version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth)
    else:
        model = resnet_v1(input_shape=input_shape, depth=depth)

    model.compile(loss='mse',
                  optimizer=Adam(lr=lr_schedule(0)))
    #model.summary()
    print(model_type)

    # Prepare model model saving directory.
    # save_dir = os.path.join(os.getcwd(), 'saved_models')
    save_dir = os.path.join(dir_output, 'saved_models2')
    model_name = 'TCGA13995_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    # checkpoint = ModelCheckpoint(filepath=filepath,
    #                              monitor='val_loss',
    #                              verbose=1,
    #                              save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    # callbacks = [checkpoint, lr_reducer, lr_scheduler]

    EStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min')

    callbacks = [lr_reducer, lr_scheduler, EStop]

    # record time
    start_time = dt.now()

    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train_np, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_val_np, y_val),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train_np)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train_np, y_train, batch_size=batch_size),
                            validation_data=(x_val_np, y_val),
                            epochs=epochs, verbose=0, workers=4,
                            callbacks=callbacks)

    # Score trained model.
    scores = model.evaluate(x_test_np, y_test, verbose=0)
    #model.save(dir_output + 'Resnet_v' + str(version) + '.h5')
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])
    print('Test loss:', scores)
    print('Total time taken: {}'.format(dt.now() - start_time))
    print('Predicting....')
    pred_train = pd.DataFrame(model.predict(x_train_np, batch_size=batch_size, verbose=0))
    pred_val = pd.DataFrame(model.predict(x_val_np, batch_size=batch_size, verbose=0))
    pred_test = pd.DataFrame(model.predict(x_test_np, batch_size=batch_size, verbose=0))
    return pred_train, pred_val, pred_test

dir_input = "/home/CBBI/tsaih/data/"
dir_output='/home/CBBI/tsaih/Research/diffMLmethods_comparison/CrossValidation_Xnorm_Z01_example/'

from pathlib import Path
Path(dir_output).mkdir(parents=True, exist_ok=True)

# Read input data (RNA)
ori_exp1=pd.read_csv(dir_input + 'X_data_batch_13995x6813_train_z.txt', sep="\t", index_col=[0])
ori_exp2=pd.read_csv(dir_input + 'X_data_batch_13995x6813_val_z.txt', sep="\t", index_col=[0])
ori_exp3=pd.read_csv(dir_input + 'X_data_batch_13995x6813_test_z.txt', sep="\t", index_col=[0])
ori_exp=pd.merge(pd.merge(ori_exp1, ori_exp2, left_index=True, right_index=True), ori_exp3, left_index=True, right_index=True)
del ori_exp1, ori_exp2, ori_exp3

# Read output data (Protein)
ori_pro1=pd.read_csv(dir_input + 'Y_data_187x6813_train.txt', sep="\t", index_col=[0])
ori_pro2=pd.read_csv(dir_input + 'Y_data_187x6813_val.txt', sep="\t", index_col=[0])
ori_pro3=pd.read_csv(dir_input + 'Y_data_187x6813_test.txt', sep="\t", index_col=[0])
ori_pro=pd.merge(pd.merge(ori_pro1, ori_pro2, left_index=True, right_index=True), ori_pro3, left_index=True, right_index=True)
del ori_pro1, ori_pro2, ori_pro3

ori_pro.index=ori_pro.index.str.upper()
ori_exp_T=ori_exp.T
ori_pro_T=ori_pro.T

pairing=CorrRNAnPro()

# Read sample index for 10-fold CrossValidation
index_train=pd.read_csv(dir_input+'RandomIndex_train_10xCV.txt', sep='\t')
index_val=pd.read_csv(dir_input+'RandomIndex_val_10xCV.txt', sep='\t')
index_test=pd.read_csv(dir_input+'RandomIndex_test_10xCV.txt', sep='\t')

tableResnet = pd.DataFrame(columns=['mse_train','mse_val',  'mse_test', 'cor_train', 'cor_val', 'cor_test'])

version=1
layer=1

# Use fold 1 as example
for i in range(1):
    print('Round' + str(i+1))

    # input_data
    x_train = ori_exp_T.iloc[index_train.iloc[:, i]]
    x_val = ori_exp_T.iloc[index_val.iloc[:, i]]
    x_test = ori_exp_T.iloc[index_test.iloc[:, i]]
    # output_data
    y_train = ori_pro_T.iloc[index_train.iloc[:, i]]
    y_val = ori_pro_T.iloc[index_val.iloc[:, i]]
    y_test = ori_pro_T.iloc[index_test.iloc[:, i]]

    # CNN
    print('.......CNN')
    resultResnet = ResnetModel(x_train, x_val, x_test, y_train, y_val, y_test, 13995, version, layer)
    pearsonResnet=Pearsonresult(y_train, y_val, y_test, resultResnet[0], resultResnet[1], resultResnet[2])
    mse_train, mse_val, mse_test = MSEresult(y_train, y_val, y_test, resultResnet[0], resultResnet[1],
                                             resultResnet[2])

    tableResnet.loc[i, 'mse_train']=mse_train
    tableResnet.loc[i, 'mse_val'] = mse_val
    tableResnet.loc[i, 'mse_test'] = mse_test
    tableResnet.loc[i, 'cor_train']= pearsonResnet['pearsonr_train'].mean()
    tableResnet.loc[i, 'cor_val'] = pearsonResnet['pearsonr_val'].mean()
    tableResnet.loc[i, 'cor_test'] = pearsonResnet['pearsonr_test'].mean()

tableResnet.to_csv(dir_output + 'Table_result_10foldCrossValidation_' + str(1) + '_Resnet.txt', sep='\t')

