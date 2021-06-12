import os
from mtcnn import MTCNN
import cv2
import numpy as np
from scipy import misc
from functions import show

min_conf = 0.9
required_size = (150, 150)
margin = 44
steps_threshold = [0.6, 0.6, 0.6]
detector = MTCNN(steps_threshold = steps_threshold)

try:
    os.makedirs('lfw_mtcnn')
    # os.makedirs('data/train_mtcnn')
    # os.makedirs('data/validation_mtcnn')
except OSError:
    pass

# source_train = 'data/train'
# source_test = 'data/validation'
source = 'lfw'

######################### Train embeddings ################################
train_subdirs = [subdir for subdir in os.listdir(source) if os.path.isdir(os.path.join(source, subdir))]
for subdir in train_subdirs:
    subdir_fullpath = os.path.join(source, subdir)
    target_subdir = os.path.join('lfw_mtcnn', subdir)
    try:
        os.makedirs(target_subdir)
    except OSError:
        pass

    for filename in os.listdir(subdir_fullpath):
        im = cv2.imread(os.path.join(subdir_fullpath, filename))
        # RGB_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # show(im)
        # show(RGB_img)
        face = detector.detect_faces(im)
        img_size = np.asarray(im.shape)[0:2]
        if len(face) != 0:
            if face[0]['confidence'] >= min_conf:
                x, y, width, height = face[0]['box']

                x, y = abs(x), abs(y) # bug negative bounding box
                mtcnn_im = im[y:y+height, x:x+width]

                out = cv2.resize(mtcnn_im, required_size)
                final_path = os.path.join(target_subdir, filename)
                cv2.imwrite(final_path, out)




######################### Validation embeddings ################################
# validation_subdirs = [subdir for subdir in os.listdir(source_test) if os.path.isdir(os.path.join(source_test, subdir))]
# for subdir in validation_subdirs:
#     subdir_fullpath = os.path.join(source_test, subdir)
#     target_subdir = os.path.join('data/validation_mtcnn', subdir)
#     try:
#         os.makedirs(target_subdir)
#     except OSError:
#         pass
#
#     for filename in os.listdir(subdir_fullpath):
#         im = cv2.imread(os.path.join(subdir_fullpath, filename))
#         face = detector.detect_faces(im)
#         if len(face) != 0:
#             if face[0]['confidence'] >= min_conf:
#                 x, y, width, height = face[0]['box']
#                 x, y = max(x, 0), max(y, 0) # bug negative bounding box
#                 mtcnn_im = im[y:y+height, x:x+width]
#                 out = cv2.resize(mtcnn_im, required_size)
#                 final_path = os.path.join(target_subdir, filename)
#                 cv2.imwrite(final_path, mtcnn_im)