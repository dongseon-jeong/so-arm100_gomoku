import tensorflow as tf
from model import Resnet18, Resnet34
import os
import time
from glob import glob
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import ast 

image_size = 208
batch_size = 64
resnet18 = Resnet18()
resnet18.build((None, image_size,image_size,3))


initial_learning_rate = 0.001
monitor_name = 'val_accuracy'

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = monitor_name,
    factor = 0.6,
    patience = 5,
    cooldown = 0,
    min_lr = 0.00001,
    verbose = 1,
    min_delta = 0.001
)

es_callback = tf.keras.callbacks.EarlyStopping(
    monitor = monitor_name,
    min_delta = 0.001,
    patience =10,
    verbose =1,
    mode = 'auto',
    baseline = None,
    restore_best_weights = True
)

optimizer = tf.keras.optimizers.SGD(learning_rate = initial_learning_rate, momentum=0.9)

model_filename = 'Epoch-{epoch:03d}-{val_accuracy:4f}.weights.h5'
checkpoint_path = os.path.join('./resnet18-models/', model_filename)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, 
                                                 save_weights_only = True, verbose =1, mode = 'auto', 
                                                 save_best_only = True, monitor = monitor_name)



class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, monitor):
        super(MyCallback, self).__init__()
        self.monitor = monitor
    def on_train_begin(self, logs = {}):
        self.times = []
        self.s_time = time.time()
        self.best_duration = 0
        self.best_val_acc = -99.
    def on_epoch_begin(self, batch, logs = {}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, batch, logs = {}):
        self.times.append(time.time()-self.s_time)
        if self.best_val_acc < logs[self.monitor]:
            self.best_val_acc = logs[self.monitor]
            self.best_duration = 0
            print(f'best val acc: {self.best_val_acc:.4f}, === new best === ', end =' ')
            print(f'elapsed: {time.time()-self.epoch_time_start:.2f} ', end = ' ')
            print(f'-{self.times[-1]:.2f}sec')
        else:
            self.best_duration += 1
            print(f'best_val_acc: {self.best_val_acc:.4f}, ', end = ' ')
            print(f'bset duration: {self.best_duration}, ', end = ' ')
            print(f'elapsed: {time.time()-self.epoch_time_start:.2f} ', end = ' ')
            print(f'-{self.times[-1]:.2f}sec')

mycallback2 = MyCallback(monitor_name)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
resnet18.compile(loss = loss_fn, optimizer = optimizer, metrics = ['accuracy'])

# def preprocessing(imgs, img_size, labels):
#     imgs = tf.image.resize(imgs, img_size)
#     return imgs, labels

def load_and_preprocess_image(image_path, label):
    # 이미지 읽고 디코딩
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # PNG이면 decode_png
    image = tf.image.resize(image, (208, 208))
    image = tf.cast(image, tf.float32) / 255.0  # 정규화

    return image, label


def parse_label(label_str):
    # 수동 파싱: 라벨 문자열을 직접 숫자 행렬로 변환
    label_str = label_str.strip("[]")  # 양 끝 대괄호 제거
    rows = label_str.split("],[")      # 행 단위 분할

    matrix = []
    for row in rows:
        row = row.replace("[", "").replace("]", "").strip()  # 괄호 제거
        digits = [int(char) for char in row]
        matrix.append(digits)

    return np.array(matrix, dtype=np.int32)  # shape = (13, 13)




image_list = glob(r"D:\making\so-arm100_gomoku\vision_model\classification\crop_image\classification_data\*.jpg")
image_list2 = glob(r"D:\making\so-arm100_gomoku\vision_model\classification\crop_image\classification_data2\*.jpg")
image_list += image_list2

label_list = pd.read_csv(r"D:\making\so-arm100_gomoku\vision_model\classification\data.csv")
label_str_list  = label_list.iloc[:,0].to_list()

parsed_labels = [parse_label(s) for s in label_str_list]  # [N, 13, 13]
parsed_labels = np.array(parsed_labels, dtype=np.int32).reshape(len(label_str_list), -1)  # [N, 169]

seed = 42
train_image_path, valid_image_path, train_label, valid_label = train_test_split(
    image_list, parsed_labels, test_size=0.05, random_state=seed)

train_dataset = tf.data.Dataset.from_tensor_slices((train_image_path, train_label))
train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(64).prefetch(tf.data.AUTOTUNE)

valid_dataset = tf.data.Dataset.from_tensor_slices((valid_image_path, valid_label))
valid_dataset = valid_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(64).prefetch(tf.data.AUTOTUNE)


history = resnet18.fit(
    train_dataset,
    epochs=25,
    validation_data=valid_dataset,
    validation_freq=1,
    verbose=1,
    callbacks=[es_callback, cp_callback, lr_callback, mycallback2],
    initial_epoch=0
)