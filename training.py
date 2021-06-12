import tensorflow as tf
import tensorflow_addons as tfa
import os
from functions import getMatchPairs, getMisMatchPairs, img_train_test_split, get_run_logdir, euclidean_distance, eucl_dist_output_shape,\
    compute_accuracy, process_pair
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import csv
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2


TRAIN_DIR = 'data/train_mtcnn'
batch_size = 32

train_generator = ImageDataGenerator(rescale=1. / 255,
                                     rotation_range=40)

train_datagen = train_generator.flow_from_directory(
    TRAIN_DIR,
    target_size=(250, 250),
    batch_size=batch_size,
    class_mode='sparse')

TEST_DIR = 'data/validation_mtcnn'
test_generator = ImageDataGenerator(rescale=1. / 255)

test_datagen = test_generator.flow_from_directory(
    TEST_DIR,
    target_size=(250, 250),
    batch_size=batch_size,
    class_mode='sparse')


n_classes = train_datagen.num_classes
n_train_im = train_datagen.samples
EPOCHS = 200

######################## Models ####################################
# input = tf.keras.layers.Input(shape=(250, 250, 3), name="input")
# x = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_normal')(input)
# x = tf.keras.layers.MaxPool2D((2, 2))(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_normal')(x)
# x = tf.keras.layers.MaxPool2D((2, 2))(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal')(x)
# x = tf.keras.layers.MaxPool2D((2, 2))(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal')(x)
# x = tf.keras.layers.MaxPool2D((2, 2))(x)
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x)
# x = tf.keras.layers.Dense(256, activation=None, kernel_initializer='he_normal')(x)
# out = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)  # L2 normalize embeddings
# base_model = tf.keras.Model(inputs=input, outputs=out)
# # base_model.summary()


#################### Transfer learning ########################
pretrained_model = InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')

for layer in pretrained_model.layers[:-100]:
    layer.trainable = False

last_layer = pretrained_model.get_layer('global_average_pooling2d')
last_output = last_layer.output
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(512, activation=None, kernel_initializer='he_normal')(x)
out = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
base_model = tf.keras.Model(inputs=pretrained_model.inputs, outputs=out)


base_model.compile(optimizer=tf.optimizers.Adam(lr=0.0001), loss=tfa.losses.TripletHardLoss())


###### Callbacks ############
early_stop = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
# Creation log_dir
root_logdir = os.path.join(os.curdir, 'tensorboard_logs/face_recognition')
run_logdir = get_run_logdir(root_logdir)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

history = base_model.fit(train_datagen,
                         epochs=EPOCHS,
                         validation_data=test_datagen,
                         callbacks=[early_stop, tensorboard_cb],
                         verbose=2)

base_model.save("models/inceptionResNetv2_semi_freezed100_mtcnn.h5")
# Custom 1:  0.46 epoch 58/// 0.68 epoch 65
# Custom 2:  0.68 epoch 104 // 0.47 epoch 68
# Inception freezed: 0.76 -> pas d'amélioration des loss avec les epochs
# Inception unfreezed 22: 0.35
# Inception unfreezed 52: 0.21 (freezedv2)
# Inception unfreezed 100: 0.16 (inceptionResNetv2_semi_freezed100)
# inceptionResNetv2_semi_freezed100_mtcnn 0.24