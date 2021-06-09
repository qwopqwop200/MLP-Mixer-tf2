import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)
  
class Patch_image(layers.Layer):
    def __init__(self, embed_dim=32,patch_size=16,patch_method='conv'):
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        if patch_method.lower() == 'conv':
            self.patch_conv = layers.Conv2D(embed_dim, patch_size, patch_size)
        elif patch_method.lower() != 'extract':
            raise Exception('patch_method must be \'conv\',\'extract\'.')            

    def image_to_patches(self,imgs):
        patches = tf.image.extract_patches(imgs, (1,self.patch_size,self.patch_size,1),(1,self.patch_size,self.patch_size,1),(1,1,1,1),"VALID")
        return patches
    
    def call(self,x):
        if self.patch_method == 'conv':
            x = self.patch_conv(x)
            x = layers.Reshape([-1,x.shape[-1]])(x)
        else:
            input_c = x.shape[-1]
            x = self.image_to_patches(x)
            x = layers.Reshape([-1,input_c*self.patch_size**2])(x)
        return x
        

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
    def __init__(self,embed_dim=32,token_dim=32, channel_dim=128,num_layers= 8,patch_size = 16,num_classes = 1000,patch_method='conv'):
        super().__init__()
        self.patch_img = Patch_image(embed_dim,patch_size,patch_method)
        self.mixer_blocks = [MixerBlock(token_dim, channel_dim) for _ in range(num_layers)]
        self.layer_norm = layers.LayerNormalization()
        self.gap = layers.GlobalAveragePooling1D()
        self.mlp_head = layers.Dense(num_classes)

    def call(self, x):
        x = self.patch_img(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = self.gap(x)
        return self.mlp_head(x)
      
def test():
    model = MLPMixer(32,10,10)
    model(tf.ones((1,224,224,3)))
