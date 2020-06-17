from tensorflow.keras import Sequential
from tensorflow.keras import layers 


class Res_block(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.relu  = layers.ReLU()
        
        self.bn1   = layers.BatchNormalization()
        self.bn2   = layers.BatchNormalization()
        self.add   = layers.Add()
        
    def build(self, input_size):
        self.conv1 = layers.Conv2D(input_size[-1], 3, activation='relu', padding='same', use_bias=False)
        self.conv2 = layers.Conv2D(input_size[-1], 3, activation='relu', padding='same', use_bias=False)
        super().build(input_size)
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.add([x, inputs])
    

def get_toy_ResNet():
    toy_res = Sequential()
    toy_res.add(layers.Conv2D(32, 3, activation='relu'))
    toy_res.add(layers.Conv2D(32, 3, use_bias=False))
    toy_res.add(layers.BatchNormalization())
    toy_res.add(layers.ReLU())
    toy_res.add(layers.MaxPool2D(2))
    toy_res.add(Res_block())
    toy_res.add(layers.Conv2D(64, 3, use_bias=False))
    toy_res.add(layers.BatchNormalization())
    toy_res.add(layers.ReLU())
    toy_res.add(layers.MaxPool2D(2))
    toy_res.add(Res_block())
    toy_res.add(layers.Conv2D(128, 3, use_bias=False))
    toy_res.add(layers.BatchNormalization())
    toy_res.add(layers.ReLU())
    toy_res.add(layers.GlobalAveragePooling2D())
    toy_res.add(layers.Dropout(0.5))
    toy_res.add(layers.Dense(64, activation='relu'))
    toy_res.add(layers.Dropout(0.7))
    toy_res.add(layers.Dense(10, activation='softmax'))
    return toy_res

def get_compiled_toy_ResNet():
    toy_res = get_toy_ResNet()
    toy_res.compile(loss='sparse_categorical_crossentropy',
                    optimizer="RMSProp",
                    metrics=["accuracy"])
    return toy_res