import tensorflow as tf

class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, strides=1):
        super(BasicBlock, self).__init__()
        self.filter_num = filter_num
        self.strides = strides

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filter_num, kernel_size=(3, 3), strides=strides,
            padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            filters=filter_num, kernel_size=(3, 3), strides=1,
            padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        # ⚠️ downsample을 조건부로 나중에 초기화
        self.downsample = None

    def build(self, input_shape):
        # 입력의 채널 수와 출력 채널 수가 다르거나 strides가 1이 아니면 downsample 필요
        if input_shape[-1] != self.filter_num or self.strides != 1:
            self.downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=self.filter_num,
                                       kernel_size=(1, 1),
                                       strides=self.strides,
                                       use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])
        else:
            self.downsample = lambda x: x  # identity

    def call(self, inputs, training=None):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        return tf.nn.relu(tf.keras.layers.add([residual, x]))

    
def make_basic_layer(filter_num, n_layers, strides = 1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, strides = strides))

    for _ in range(1, n_layers):
        res_block.add(BasicBlock(filter_num, strides = 1))

    return res_block


class ResNet_A(tf.keras.Model):
    def __init__(self, layer_params):  # NUM_CLASSES 제거
        super(ResNet_A, self).__init__()
        filter_nums = [64, 128, 256, 512]

        # 초기 conv + batchnorm + maxpool
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same')

        # ResNet 블록
        self.layer1 = make_basic_layer(filter_num=filter_nums[0], n_layers=layer_params[0], strides=1)   # 유지
        self.layer2 = make_basic_layer(filter_num=filter_nums[1], n_layers=layer_params[1], strides=2)   # 유지
        self.layer3 = make_basic_layer(filter_num=filter_nums[2], n_layers=layer_params[2], strides=2)   # 유지
        self.layer4 = make_basic_layer(filter_num=filter_nums[3], n_layers=layer_params[3], strides=1)   # 수정됨 (2 → 1)

        # 평균 풀링 제거, 대신 위치 분류를 위한 conv
        self.fc = tf.keras.layers.Conv2D(filters=3, kernel_size=1, activation=None)  # [batch, 13, 13, 3]
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        x = self.fc(x)  # [batch_size, 13, 13, 3]
        x = tf.reshape(x, [tf.shape(x)[0], -1, 3])  # → [batch_size, 169, 3]

        return x

    
def Resnet18():
    return ResNet_A(layer_params = [2,2,2,2])
def Resnet34():
    return ResNet_A(layer_params = [3,4,6,3])