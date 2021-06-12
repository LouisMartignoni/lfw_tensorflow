import tensorflow as tf
import os
import random
from shutil import copyfile
from triplet_loss import batch_hard_triplet_loss
import numpy as np
import matplotlib.pyplot as plt


def imgprcs(file, IMG_SIZE=250):
    img = tf.io.read_file(file)
    # oh = tf.image.extract_jpeg_shape(img)
    img = tf.image.decode_jpeg(img)
    # img = tf.cond(tf.less(oh[2],3), lambda: tf.image.grayscale_to_rgb(img), lambda: img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    #img = img / 255.
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    #show(img)
    # img = tf.image.per_image_standardization(img)
    # img = tf.reshape(img, [250,250,3])
    return img


def process_pair(files, label):
    im1 = imgprcs(files[0])
    im2 = imgprcs(files[1])
    return (im1, im2), label


def getMatchPairs(l, matched_pairs):
    num1 = '%04d' % (int(l[1]))
    im_path1 = l[0] + '_' + num1 + '.jpg'
    path1 = os.path.join('lfw', l[0], im_path1)
    num2 = '%04d' % (int(l[2]))
    im_path2 = l[0] + '_' + num2 + '.jpg'
    path2 = os.path.join('lfw', l[0], im_path2)
    # Open images

    matched_pairs.append([path1, path2])


def getMisMatchPairs(l, mismatched_pairs):
    num1 = '%04d' % (int(l[1]))
    im_path1 = l[0] + '_' + num1 + '.jpg'
    path1 = os.path.join('lfw', l[0], im_path1)
    num2 = '%04d' % (int(l[3]))
    im_path2 = l[2] + '_' + num2 + '.jpg'
    path2 = os.path.join('lfw', l[2], im_path2)
    mismatched_pairs.append([path1, path2])


def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("base_model_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


def img_train_test_split(img_source_dir, train_size):
    """
    Randomly splits images over a train and validation folder, while preserving the folder structure

    Parameters
    ----------
    img_source_dir : string
        Path to the folder with the images to be split. Can be absolute or relative path

    train_size : float
        Proportion of the original images that need to be copied in the subdirectory in the train folder
    """
    if not (isinstance(img_source_dir, str)):
        raise AttributeError('img_source_dir must be a string')

    if not os.path.exists(img_source_dir):
        raise OSError('img_source_dir does not exist')

    if not (isinstance(train_size, float)):
        raise AttributeError('train_size must be a float')

    # Set up empty folder structure if not exists
    try:
        os.makedirs('data')
        os.makedirs('data/train')
        os.makedirs('data/validation')
    except OSError:
        pass

    # Get the subdirectories in the main image folder
    subdirs = [subdir for subdir in os.listdir(img_source_dir) if os.path.isdir(os.path.join(img_source_dir, subdir))]

    for subdir in subdirs:
        subdir_fullpath = os.path.join(img_source_dir, subdir)
        if len(os.listdir(subdir_fullpath)) > 30:
            train_subdir = os.path.join('data/train', subdir)
            validation_subdir = os.path.join('data/validation', subdir)

            # Create subdirectories in train and validation folders
            try:
                os.makedirs(train_subdir)
                os.makedirs(validation_subdir)
            except OSError:
                pass

            train_counter = 0
            validation_counter = 0

            # Randomly assign an image to train or validation folder
            for filename in os.listdir(subdir_fullpath):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    fileparts = filename.split('.')

                    if random.uniform(0, 1) <= train_size:
                        copyfile(os.path.join(subdir_fullpath, filename),
                                 os.path.join(train_subdir, str(train_counter) + '.' + fileparts[1]))
                        train_counter += 1
                    else:
                        copyfile(os.path.join(subdir_fullpath, filename),
                                 os.path.join(validation_subdir, str(validation_counter) + '.' + fileparts[1]))
                        validation_counter += 1

            print('Copied ' + str(train_counter) + ' images to data/train/' + subdir)
            print('Copied ' + str(validation_counter) + ' images to data/validation/' + subdir)
        else:
            print('{} not enough images'.format(subdir))


def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def compute_accuracy(y_true, y_pred, threshold=0.14): #1.23
    """Compute classification accuracy with a fixed threshold on distances.
    """
    pred = y_pred.ravel() < threshold
    return np.mean(pred == y_true)


def show(image, label=None):
    plt.figure()
    plt.imshow(image)
    # plt.title(label.numpy())
    plt.axis('off')
    plt.show()
