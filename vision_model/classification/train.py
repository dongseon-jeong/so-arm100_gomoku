import tensorflow as tf
from model import Resnet18, Resnet34
import os
import time

resnet18 = Resnet18(NUM_CALSSES=10)
resnet18.build((None, 224,224,3))

initial_learning_rate = 0.1
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

optimizer = tf.keras.optimizers.SGD(learing_rate = initial_learning_rate, momentum=0.9)

model_filename = 'Epoch-{epoch:03d}-{val_accuracy:4f}'
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
            print(f'best_val_acc: {self.bset_val_acc:.4f}, ', end = ' ')
            print(f'bset duration: {self.best_duration}, ', end = ' ')
            print(f'elapsed: {time.time()-self.epoch_time_start:.2f} ', end = ' ')
            print(f'-{self.times[-1]:.2f}sec')

mycallback2 = MyCallback(monitor_name)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
resnet18.compile(loss = loss_fn, opimizer = optimizer, metrics = ['accuracy'])

def preprocessing(imgs, img_size, labels):
    imgs = tf.image.resize(imgs, img_size)
    return imgs, labels

batch_size = 64
img_size = (224,224)



# train_dataset
# valid_dataset


# history = resnet18.fit(train_dataset, epochs = 400, validation_data = valid_dataset,
#                        validation_freq =1, verbose = 1,
#                        callbacks = [es_callback, cp_callback, lr_callback, mycallback2],
#                        initial_epoch = 0)