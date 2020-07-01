import os
import tensorflow as tf
import tensorflow_datasets as tfds
from keras_preprocessing.image import ImageDataGenerator

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd

print(tf.__version__)

# Import our GhostNet Model
from ghost_model import GhostNet

# Set seed for reproducability
seed = 1234
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


# (ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'], shuffle_files=True,
#                                          as_supervised=True, with_info=True)
#
# def normalize_img(image, label):
#     """
#     Normalizes images: `uint8` -> `float32`.
#     """
#     return tf.cast(image, tf.float32) / 255., label
#
#
# def normalize_img(image, label):
#     """
#     Normalizes images: `uint8` -> `float32`.
#     """
#     return tf.cast(image, tf.float32) / 255., label

# ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# ds_train = ds_train.cache()
# ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
# ds_train = ds_train.batch(128)
# ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
#
# ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# ds_test = ds_test.batch(128)
# ds_test = ds_test.cache()
# ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# 训练数据集生成器
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
_train_data_dir = 'D:\dl\imageclass\data\\train'
_valid_data_dir = 'D:\dl\imageclass\data\\test'
data_gen = ImageDataGenerator(rescale=1./255)
train_data_gen = data_gen.flow_from_directory(
    _train_data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='sparse')  # "categorical"将是2D one-hot 编码标签,"sparse" 将是 1D 整数标签，
valid_data_gen = data_gen.flow_from_directory(
    _valid_data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='sparse')
print('done')


# Specify number of classes for GhostNet (10 for MNIST)
# model = GhostNet(10)
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer=tf.keras.optimizers.Adam(0.001),
#               metrics=['accuracy'])
#
#
# model.fit(train_data_gen, epochs=6, validation_data=ds_test)

# 仅训练顶层分类器
output_model_file = 'D:\dl\imageclass\model\ghostnet\\best.h5'
callbacks = [
    # tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    # tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),
    tf.keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True)

]
model = GhostNet(10)  # 10 classes
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(
    x=train_data_gen,
    epochs=1,
    validation_data=valid_data_gen,
    callbacks=callbacks)

model.save(output_model_file)

# 学习曲线图
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)