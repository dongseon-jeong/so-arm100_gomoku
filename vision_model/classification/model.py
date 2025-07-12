import tensorflow as tf

class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, strides = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters = filter_num, kernel_size= (3,3), strides = strides, padding = 'same', use_bias = False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters = filter_num, kernel_size = (3,3), strides = strides , padding = 'same', use_bias = False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        if strides != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(
                tf.keras.layers.Conv2D(filters = filter_num, kernel_size = (1,1), strides = strides, use_bias = False))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x:x
        
    def call(self, inputs, training = None):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training = training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training = training)

        output = tf.nn.relu(tf.keras.layers.add([residual,x]))

        return output
    
def make_basic_layer(filter_num, n_layers, strides = 1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, strides = strides))

    for _ in range(1, n_layers):
        res_block.add(BasicBlock(filter_num, strides = 1))

    return res_block


class ResNet_A(tf.keras.Model):
    def __init__(self, layer_params,    NUM_CLASSES = 10):
        super(ResNet_A, self).__init__()
        filter_nums = [64, 128, 256, 512]
        self.conv1 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7,7), strides = 2, padding = 'same', use_bias = False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = 2, padding = 'same')
        self.layer1 = make_basic_layer(filter_num = filter_nums[0], n_layers = layer_params[0], strides =1)
        self.layer2 = make_basic_layer(filter_num = filter_nums[1], n_layers = layer_params[1], strides =2)
        self.layer3 = make_basic_layer(filter_num = filter_nums[2], n_layers = layer_params[2], strides =2)
        self.layer4 = make_basic_layer(filter_num = filter_nums[3], n_layers = layer_params[3], strides =2)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units = NUM_CLASSES, activation = None)
    
    def call(self, inputs, training = None):
        x = self.conv1(inputs)
        x = self.bn1(x, training = training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training = training)
        x = self.layer2(x, training = training)
        x = self.layer3(x, training = training)
        x = self.layer4(x, training = training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output
    
def Resnet18(NUM_CALSSES):
    return ResNet_A(layer_params = [2,2,2,2], NUM_CLASSES=NUM_CALSSES)

def Resnet34(NUM_CLASSES):
    return ResNet_A(layer_params = [3,4,6,3], NUM_CLASSES=NUM_CLASSES)