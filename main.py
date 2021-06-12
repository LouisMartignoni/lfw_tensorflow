import tensorflow as tf
import tensorflow_addons as tfa
import os
from functions import getMatchPairs, getMisMatchPairs, img_train_test_split, get_run_logdir, euclidean_distance, \
    eucl_dist_output_shape, \
    compute_accuracy, process_pair, show
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import csv
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
#from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from architecture import InceptionResNetV2

############################ Create train/validation folders ######################################
# img_train_test_split('lfw', 0.7)

############################ Pairs evaluation ####################################################
matched_pairs = []
mismatched_pairs = []
labels_temp = []
with open('pairsDevTrain.txt', newline='') as f:
    reader = csv.reader(f, delimiter="\t")
    nb_match = next(reader)[0]
    for i in range(int(nb_match)):
        l = next(reader)
        getMatchPairs(l, matched_pairs)
        labels_temp.append(1.)

    for i in range(int(nb_match)):
        l = next(reader)
        getMisMatchPairs(l, mismatched_pairs)
        labels_temp.append(0.)

l = matched_pairs + mismatched_pairs
# data_X = l
# labels = labels_temp
data_X = []
labels = []
####### mtcnn ########
for row, label in zip(l, labels_temp):
    row[0] = "lfw_mtcnn" + row[0][3:]
    row[1] = "lfw_mtcnn" + row[1][3:]
    if os.path.exists(row[0]) and os.path.exists(row[1]):
        data_X.append(row)
        labels.append(label)
#####################

ds = tf.data.Dataset.from_tensor_slices((data_X, labels))
ds = ds.map(process_pair)
# for images, label in ds:
#     print("2")
ds = ds.shuffle(buffer_size=1500)
ds = ds.batch(32).prefetch(1)

# for images, label in ds:
#     im1 = images[0]
#     im2 = images[1]
#     l = label
#     show(im1[0], label)
#     show(im2[0], label)


################################ Load Previous model ###########################
# base_model = tf.keras.models.load_model("models/base_model.h5") # 0.66
# base_model = tf.keras.models.load_model("models/mtcnn_model.h5") # 0.63
# base_model = tf.keras.models.load_model("models/base_model2.h5") # (mtcnn model) 0.67
# base_model = tf.keras.models.load_model("models/base_model1.h5") # (custom 2 sans mtcnn) 0.71 (threshold 1.26)
#base_model = tf.keras.models.load_model("models/inceptionResNetv2_full_freezed.h5") # transfer learning sur les derniers layers 0.72 (1.16)
# base_model = tf.keras.models.load_model("models/inceptionResNetv2_semi_freezed100.h5") # 0.73(1.33)
# base_model = tf.keras.models.load_model("models/inceptionResNetv2_semi_freezed100_mtcnn.h5") # 0.69(1.3)
#base_model = InceptionResNetV2()
#base_model.load_weights('models/facenet_keras_weights_tf2.h5')
base_model = tf.keras.models.load_model("models/facenet_keras.h5") # 0.69(1.3)


########## Inception ResNetv2 trained on imagenet, not retrained ############
# # 0.73 (threshold 0.69)
# pretrained_model = InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')
# last_layer = pretrained_model.get_layer('global_average_pooling2d')
# last_output = last_layer.output
# x = tf.keras.layers.Flatten()(last_output)
# x = tf.keras.layers.Dense(528, activation='relu', kernel_initializer='he_normal')(x)
# x = tf.keras.layers.Dense(256, activation=None, kernel_initializer='he_normal')(x)
# out = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
# base_model = tf.keras.Model(inputs=pretrained_model.inputs, outputs=out)



############################### Network #######################################
input_A = tf.keras.layers.Input(shape=(250, 250, 3), name="input_A")
input_B = tf.keras.layers.Input(shape=(250, 250, 3), name="input_B")

res_A = base_model(input_A)
res_B = base_model(input_B)

lambda_layer = tf.keras.layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape,
                                      name='lambda_layer')([res_A, res_B])

final_model = tf.keras.Model(inputs=[input_A, input_B], outputs=[lambda_layer])

# ############################### Evaluation ###################################
predictions = np.array([])
labels = np.array([])
for x, y in ds:
    p = final_model.predict(x).squeeze()
    predictions = np.concatenate([predictions, p])
    labels = np.concatenate([labels, y.numpy()])
    #for i, (l, pred) in enumerate(zip(y.numpy(), p)):
    #      if l == (p > 0.2):
    #           im1 = x[0]
    #           im2 = x[1]
    #          print(l,pred)
    #           show(im1[i], l)
    #           show(im2[i], l)
    #           print("done")

train_accuracy = compute_accuracy(labels, predictions, threshold=1.3)
print(train_accuracy)

# print("Loss = {}, Train Accuracy = {} Test Accuracy = {}".format(loss, train_accuracy, test_accuracy))
