import os
import gdown
from zipfile import ZipFile
import datetime
from model import CVAE
import tensorflow as tf
import keras
import numpy as np
import tensorflow_datasets as tfds

# (train_data, test_data), info = tfds.load(name = 'celeb_a', split = ['train', 'test'], as_supervised = True, shuffle_files = True, with_info = True)

# url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
# output = "C:/Users/saraa/Desktop/image_gen/celeba_cvae/data.zip"
# gdown.download(url, output, quiet=True)

# with ZipFile(output, "r") as zipobj:
#     zipobj.extractall("celeba_cvae")

NUM_LABELS = 40
BATCH_SIZE = 32
DATASET_SIZE = 202599
NUM_EPOCHS = 50

attributes = {
        '5_o_Clock_Shadow': True,
        'Arched_Eyebrows': True,
        'Attractive': True,
        'Bags_Under_Eyes': True,
        'Bald': True,
        'Bangs': True,
        'Big_Lips': True,
        'Big_Nose': True,
        'Black_Hair': True,
        'Blond_Hair': True,
        'Blurry': True,
        'Brown_Hair': True,
        'Bushy_Eyebrows': True,
        'Chubby': True,
        'Double_Chin': True,
        'Eyeglasses': True,
        'Goatee': True,
        'Gray_Hair': True,
        'Heavy_Makeup': True,
        'High_Cheekbones': True,
        'Male': True,
        'Mouth_Slightly_Open': True,
        'Mustache': True,
        'Narrow_Eyes': True,
        'No_Beard': True,
        'Oval_Face': True,
        'Pale_Skin': True,
        'Pointy_Nose': True,
        'Receding_Hairline': True,
        'Rosy_Cheeks': True,
        'Sideburns': True,
        'Smiling': True,
        'Straight_Hair': True,
        'Wavy_Hair': True,
        'Wearing_Earrings': True,
        'Wearing_Hat': True,
        'Wearing_Lipstick': True,
        'Wearing_Necklace': True,
        'Wearing_Necktie': True,
        'Young': True,
    }

attributes_dict = {}

with open("C:/Users/saraa/Desktop/image_gen/celeba_cvae/list_attr_celeba.txt", "r") as file:
    lines = file.readlines()[2:]
    for line in lines:
        parts = line.split()
        filename = parts[0]
        attributes = list(map(int, parts[1:]))
        attributes_dict[filename] = attributes

@tf.py_function(Tout=tf.float32)
def get_labels(batch_index: tf.Tensor):
    start_idx = batch_index.numpy() * BATCH_SIZE
    labels = []
    for i in range(BATCH_SIZE):
        idx = start_idx + i
        if idx < DATASET_SIZE:
            filename = str.zfill(str(idx+1), 6) + '.jpg'
            labels.append(attributes_dict[filename])
        else:
            break
    return tf.math.maximum(tf.convert_to_tensor(labels, dtype=tf.float32), 0.0)

labels = tf.data.Dataset.range(DATASET_SIZE//BATCH_SIZE+1).map(lambda x: get_labels(x))
dataset = keras.utils.image_dataset_from_directory(
    "C:/Users/saraa/Desktop/image_gen/celeba_cvae", label_mode=None, image_size=(64, 64), batch_size=BATCH_SIZE, shuffle=False, data_format='channels_first')
# scale values down to [0, 1]
dataset = dataset.map(lambda x: x / 255.0)
dataset = tf.data.Dataset.zip(dataset, labels)
model = CVAE(latent_dim=128)
model.generate_images()
for epoch in range(NUM_EPOCHS):
    print(f'{datetime.datetime.now()}: begin epoch {epoch+1}')
    for element in dataset:
        model.train(element[0], element[1])
        if model.steps % 1000 == 0:
            print(f'{datetime.datetime.now()}: completed {model.steps} training steps')
    print(f'{datetime.datetime.now()}: completed {model.steps} training steps, epoch {epoch} complete')
    model.generate_images()
    model.save_checkpoint()