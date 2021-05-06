import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)
  
class MLP(layers.Layer):
    def __init__(self, hidden_dim, dropout_ratio = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_ratio = dropout_ratio
        
    def build(self,shape):
        self.dense1 = layers.Dense(self.hidden_dim)
        self.dense2 = layers.Dense(shape[-1])
        self.dropout1 = layers.Dropout(self.dropout_ratio)
        self.dropout2 = layers.Dropout(self.dropout_ratio)
        
    def call(self, x):
        x = self.dropout1(keras.activations.gelu(self.dense1(x)))
        return self.dropout2(self.dense2(x))

class MixerBlock(layers.Layer):
    def __init__(self, token_dim, channel_dim, dropout_ratio = 0.1):
        super().__init__()
        self.mlp1 = MLP(token_dim,dropout_ratio)
        self.mlp2 = MLP(channel_dim,dropout_ratio)
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()

    def call(self, x):
        x = tf.transpose(self.mlp1(tf.transpose(self.layernorm1(x),[0,2,1])),[0,2,1]) + x
        return self.mlp2(self.layernorm2(x)) + x

class MLPMixer(keras.Model):
    def __init__(self,embed_dim,token_dim, channel_dim,num_layers= 8,patch_size = 16,num_classes = 1000):
        super().__init__()
        self.patch_conv = layers.Conv2D(embed_dim, patch_size, patch_size)
        self.mixer_blocks = [MixerBlock(token_dim, channel_dim) for _ in range(num_layers)]
        self.layer_norm = layers.LayerNormalization()
        self.gap = layers.GlobalAveragePooling1D()
        self.mlp_head = layers.Dense(num_classes)

    def call(self, x):
        x = self.patch_conv(x)
        x = layers.Reshape([-1,x.shape[-1]])(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = self.gap(x)
        return self.mlp_head(x)
      
def test():
    model = MLPMixer(32,10,10)
    model(tf.ones((1,224,224,3)))
