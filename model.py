import tensorflow as tf
import tensorflow_probability as tfp
from PIL import Image
import numpy as np
import keras
import os
import math

CHKPT_DIR = 'C:/Users/saraa/Desktop/image_gen/checkpoints/'
IMAGE_DIR = 'C:/Users/saraa/Desktop/image_gen/images/'

def lrelu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)

def sigmoid(x):
    return tf.nn.sigmoid(x)

@tf.function
def flatten(x):
    shape = x.shape.as_list()
    flattened_dim = 1
    for i in range(1, len(shape)):
        flattened_dim *= shape[i]
    return tf.reshape(x, [-1, flattened_dim])

class Linear(object):
    def __init__(self, units: int, name: str, activation=None, input_shape=None):
        self.units = units
        self.name = name
        self.activation = activation if activation is not None else lambda x: x
        self.weight = None
        self.bias = None
        if input_shape is not None:
            self.create_dense_vars(input_shape)

    def __call__(self, input: tf.Tensor):
        if self.weight is None or self.bias is None:
            self.create_vars(input.shape.as_list())
        return self.activation(tf.matmul(input, self.weight) + self.bias)
    
    def create_vars(self, input_shape: list | tuple, variance=2.0):
        fan_in = np.prod(input_shape[1:])
        std = math.sqrt(variance / fan_in)
        self.weight = tf.Variable(tf.random.normal(shape=[input_shape[-1], self.units], mean=0.0, stddev=std), name=self.name+"_weight", trainable=True)
        self.bias = tf.Variable(tf.zeros(shape=[1, self.units]), name=self.name+"_bias", trainable=True)

    def get_variables(self) -> list:
        return [self.weight, self.bias]
    
    def get_trainable_variables(self) -> list:
        return [self.weight, self.bias]
    
    def load_variables(self, vars: dict):
        for name, var in vars.items():
            if self.name+"_weight" in name:
                self.weight = var
            elif self.name+"_bias" in name:
                self.bias = var

class Conv2D(object):
    def __init__(self, filters: int, name: str, kernel=(2, 2), stride=(1, 1), padding=(0, 0), dilation=(1, 1), activation=None, input_shape=None):
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        if isinstance(self.padding, (tuple, list)):
            ph, pw = [0, 0], [0, 0]
            if isinstance(self.padding[0], (tuple, list)):
                ph = self.padding[0]
            else:
                ph = [self.padding[0], self.padding[0]]
            if isinstance(self.padding[1], (tuple, list)):
                pw = self.padding[1]
            else:
                pw = [self.padding[1], self.padding[1]]
            self.padding = [[0, 0], [0, 0], ph, pw]
        self.dilation = dilation
        self.name = name
        self.activation = activation if activation is not None else lambda x: x
        self.weight = None
        self.bias = None
        if input_shape is not None:
            self.create_vars(input_shape)

    def __call__(self, input: tf.Tensor):
        if self.weight is None or self.bias is None:
            self.create_vars(input.shape.as_list())
        return self.activation(tf.nn.conv2d(input, self.weight, self.stride, self.padding, "NCHW", self.dilation) + self.bias)
    
    # using HWIO weight format and NCHW
    def create_vars(self, input_shape: tuple | list, variance=2.0):
        fan_in = np.prod(input_shape[1:]) # numpy automatically converts
        std = math.sqrt(variance / fan_in)
        self.weight = tf.Variable(tf.random.normal(shape=[*self.kernel, input_shape[1], self.filters], mean=0.0, stddev=std), name=self.name+"_weight", trainable=True)
        # need to calculate output dims for the bias
        oh, ow = 0, 0
        if self.padding == 'SAME':
            oh = math.ceil(float(input_shape[2])/float(self.stride[0]))
            ow = math.ceil(float(input_shape[3])/float(self.stride[1]))
        else:
            oh = int((input_shape[2] + self.padding[2][0] + self.padding[2][1] - self.dilation[0] * (self.kernel[0] - 1) - 1) / self.stride[0] + 1)
            ow = int((input_shape[3] + self.padding[3][0] + self.padding[3][1] - self.dilation[1] * (self.kernel[1] - 1) - 1) / self.stride[1] + 1)
        self.bias = tf.Variable(tf.zeros(shape=[1, self.filters, oh, ow]), name=self.name+"_bias", trainable=True)

    def get_variables(self) -> list:
        return [self.weight, self.bias]

    def get_trainable_variables(self) -> list:
        return [self.weight, self.bias]
    
    def load_variables(self, vars: dict):
        for name, var in vars.items():
            if self.name+"_weight" in name:
                self.weight = var
            elif self.name+"_bias" in name:
                self.bias = var

class Conv2DTranspose(object):
    def __init__(self, filters: int, name: str, kernel=(2, 2), output_shape = None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), activation=None, input_shape=None):
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        if isinstance(self.padding, (tuple, list)):
            ph, pw = [0, 0], [0, 0]
            if isinstance(self.padding[0], (tuple, list)):
                ph = self.padding[0]
            else:
                ph = [self.padding[0], self.padding[0]]
            if isinstance(self.padding[1], (tuple, list)):
                pw = self.padding[1]
            else:
                pw = [self.padding[1], self.padding[1]]
            self.padding = [[0, 0], [0, 0], ph, pw]
        self.dilation = dilation
        self.output_shape = output_shape
        self.name = name
        self.activation = activation if activation is not None else lambda x: x
        self.weight = None
        self.bias = None
        if input_shape is not None:
            self.create_vars(input_shape)

    def __call__(self, input: tf.Tensor):
        if self.weight is None or self.bias is None:
            self.create_vars(input.shape.as_list())
        # need to redefine it just in case batch size changes
        self.output_shape[0] = input.shape.as_list()[0]
        if not isinstance(self.padding, str):
            # conv2d transpose doesn't support nchw when we have explicit paddings, so we must transpose to nhwc
            out_shape = [self.output_shape[0], self.output_shape[2], self.output_shape[3], self.output_shape[1]]
            pads = [self.padding[0], self.padding[2], self.padding[3], self.padding[1]]
            conv = tf.nn.conv2d_transpose(tf.transpose(input, perm=[0, 2, 3, 1]), self.weight, out_shape, self.stride, pads, "NHWC", self.dilation)
            # then we have to transpose back to nchw
            return self.activation(tf.transpose(conv, perm=[0, 3, 1, 2]) + self.bias)      
        return self.activation(tf.nn.conv2d_transpose(input, self.weight, self.output_shape, self.stride, self.padding, "NCHW", self.dilation) + self.bias)
    
    def create_vars(self, input_shape: tuple | list, variance=2.0):
        fan_in = np.prod(input_shape[1:])
        std = math.sqrt(variance / fan_in)
        self.weight = tf.Variable(tf.random.normal(shape=[*self.kernel, self.filters, input_shape[1]], mean=0.0, stddev=std), name=self.name+"_kernel", trainable=True)
        # need to calculate output dims for the bias
        oh, ow = 1, 1
        if self.padding == 'SAME':
            oh = input_shape[2]*self.stride[0]
            ow = input_shape[3]*self.stride[1]
        else:
            oh = self.stride[0]*(input_shape[2]-1)+self.kernel[0]+(self.kernel[0]-1)*(self.dilation[0]-1)-self.padding[2][0]-self.padding[2][1]
            ow = self.stride[1]*(input_shape[3]-1)+self.kernel[1]+(self.kernel[1]-1)*(self.dilation[1]-1)-self.padding[3][0]-self.padding[3][1]
        self.bias = tf.Variable(tf.zeros(shape=[1, self.filters, oh, ow]), name=self.name+"_bias", trainable=True)
        if self.output_shape is None:
            self.output_shape = [input_shape[0], self.filters, oh, ow]

    def get_variables(self) -> list:
        return [self.weight, self.bias]
    
    def get_trainable_variables(self) -> list:
        return [self.weight, self.bias]
    
    def load_variables(self, vars: dict):
        for name, var in vars.items():
            if self.name+"_weight" in name:
                self.weight = var
            elif self.name+"_bias" in name:
                self.bias = var

class BatchNorm(object):
    # training should always be true upon init
    def __init__(self, axis: int, name: str, momentum=0.99, epsilon=1.0e-7, training=True, input_shape=None):
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.name = name
        self.training = training
        self.gamma, self.beta, self.moving_mean, self.moving_variance = None, None, None, None
        if input_shape is not None:
            self.create_vars(input_shape)

    def __call__(self, input: tf.Tensor):
        if self.gamma is None or self.beta is None:
            self.create_vars(input.shape.as_list())
        return self.batch_norm(input)
    
    @tf.function
    def batch_norm(self, input: tf.Tensor, training: bool, name: str, axis=-1):
        shape = input.shape.as_list()
        reduction_axes = list(range(len(shape)))
        del reduction_axes[self.axis]
        mean, variance = tf.nn.moments(x=input, axes=reduction_axes, keepdims=True)
        # works because we use @tf.function and autograph happens
        if training:
            self.moving_mean = self.moving_mean*self.momentum+mean*(1.0-self.momentum)
            self.moving_variance = self.moving_variance*self.momentum+variance*(1.0-self.momentum)
        else:
            mean = self.moving_mean
            variance = self.moving_variance

        stddev = tf.sqrt(variance + self.epsilon)
        input = (input - mean) / stddev
        input = input * self.gamma + self.beta
        return input
    
    def create_vars(self, input_shape: list | tuple, name: str):
        param_shape = [1] * len(input_shape)
        param_shape[self.axis] = input_shape[self.axis]
        self.gamma = tf.Variable(tf.ones(param_shape), name=name+"_gamma", trainable=True)
        self.beta = tf.Variable(tf.zeros(param_shape), name=name+"_beta", trainable=True)
        self.moving_mean = tf.Variable(tf.zeros(param_shape), name=name+"_moving_mean", trainable=False)
        self.moving_variance = tf.Variable(tf.ones(param_shape), name=name+"_moving_variance", trainable=False)

    def get_variables(self) -> list:
        return [self.gamma, self.beta, self.moving_mean, self.moving_variance]
    
    def get_trainable_variables(self) -> list:
        return [self.gamma, self.beta]
    
    def load_variables(self, vars: dict):
        for name, var in vars.items():
            if self.name+"_gamma" in name:
                self.weight = var
            elif self.name+"_beta" in name:
                self.bias = var
            elif self.name+"_moving_mean" in name:
                self.moving_mean = var
            elif self.name+"_moving_variance" in name:
                self.moving_variance = var
        
class Adam(object):
    def __init__(self, lr=8.75e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7, gamma=0.992):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.gamma = gamma
        self.epsilon = epsilon
        self.trainable_variables, self.moment1, self.moment2, self.updates = {}, {}, {}, {}

    # a quick simple implementation of the adam algorithm, don't need any advanced features from the keras optimizers
    def update_variable(self, gradient: tf.Tensor, variable: tf.Variable):
        if variable.name not in self.trainable_variables:
            self.trainable_variables[variable.name] = variable
            self.moment1[variable.name] = tf.zeros_like(variable)
            self.moment2[variable.name] = tf.zeros_like(variable)
            self.updates[variable.name] = 0
        self.moment1[variable.name] = self.beta_1 * self.moment1[variable.name] + gradient * (1.0 - self.beta_1)
        self.moment2[variable.name] = self.beta_2 * self.moment2[variable.name] + tf.square(gradient) * (1.0 - self.beta_2)    
        corrected_moment1 = self.moment1[variable.name] / (1.0 - self.beta_1 ** (self.updates[variable.name] + 1))
        corrected_moment2 = self.moment2[variable.name] / (1.0 - self.beta_2 ** (self.updates[variable.name] + 1))
        variable.assign_sub(self.lr * corrected_moment1 / (tf.sqrt(corrected_moment2) + self.epsilon))
        self.updates[variable.name] += 1

    def apply_gradients(self, grads_and_vars: zip):
        for grad, var in grads_and_vars:
            if grad is not None:
                self.update_variable(grad, var)

class Encoder(object):
    def __init__(self, latent_dim=2):
        self.latent_dim = latent_dim
        self.conv1 = Conv2D(32, 'encoder_conv1', (4, 4), (2, 2), 'SAME', (1, 1), lrelu)
        self.conv2 = Conv2D(64, 'encoder_conv2', (4, 4), (2, 2), 'SAME', (1, 1), lrelu)
        self.conv3 = Conv2D(128, 'encoder_conv3', (4, 4), (2, 2), 'SAME', (1, 1), lrelu)
        self.dense = Linear(2*self.latent_dim, 'encoder_dense') # outputs latent_dim dimensional vectors of mean and log variance of latent distribution
        self.layers = [self.conv1, self.conv2, self.conv3, self.dense]

    def __call__(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = flatten(x)
        x = self.dense(x)
        return x
    
    def get_trainable_variables(self) -> list:
        vars = []
        for layer in self.layers:
            vars.extend(layer.get_trainable_variables())
        return vars
    
    def get_variables(self) -> list:
        vars = []
        for layer in self.layers:
            vars.extend(layer.get_variables())
        return vars

    def load_checkpoint(self, vars: dict):
        for layer in self.layers:
            layer.load_variables(vars)

class Decoder(object):
    def __init__(self, latent_dim=2):
        self.latent_dim = latent_dim
        self.label_transformer = Linear(self.latent_dim, 'decoder_label_transformer', lrelu) # 218//8 * 178//8 * 64, because transposed convolutions upsample by factor of 2
        self.dense = Linear(8*8*128, 'decoder_dense', lrelu)
        self.conv1t = Conv2DTranspose(64, 'decoder_conv1t', (4, 4), None, (2, 2), 'SAME', (1, 1), lrelu)
        self.conv2t = Conv2DTranspose(128, 'decoder_conv2t', (4, 4), None, (2, 2), 'SAME', (1, 1), lrelu)
        self.conv3t = Conv2DTranspose(256, 'decoder_conv3t', (4, 4), None, (2, 2), 'SAME', (1, 1), lrelu)
        self.conv4 = Conv2D(3, 'decoder_conv4', (4, 4), (1, 1), 'SAME', (1, 1))
        self.layers = [self.dense, self.label_transformer, self.conv1t, self.conv2t, self.conv3t, self.conv4]

    def __call__(self, input, labels):
        x = input + self.label_transformer(labels)
        x = self.dense(x)
        x = tf.reshape(x, shape=[-1, 128, 8, 8])
        x = self.conv1t(x)
        x = self.conv2t(x)
        x = self.conv3t(x)
        x = self.conv4(x)
        return x
    
    def get_trainable_variables(self) -> list:
        vars = []
        for layer in self.layers:
            vars.extend(layer.get_trainable_variables())
        return vars

    def get_variables(self) -> list:
        vars = []
        for layer in self.layers:
            vars.extend(layer.get_variables())
        return vars

    def load_checkpoint(self, vars: dict):
        for layer in self.layers:
            layer.load_variables(vars)

class CVAE(object):
    def __init__(self, latent_dim=2, beta=1.0e-5):
        self.latent_dim = latent_dim
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.latent_dim)
        self.steps = 0
        self.beta = beta # beta parameter used to encourage creativity
        self.optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1.0e-7)

    def __call__(self, input, labels):
        mean, log_var = tf.split(self.encoder(input), num_or_size_splits=2, axis=-1)
        # reparameterization trick
        eps = tf.random.normal(shape=mean.shape) # mean and stddev are 0 and 1
        # multiply by 0.5 turns log variance into log std by laws of logs
        z = eps * tf.exp(log_var * 0.5) + mean
        x = self.decoder(z, labels) # embeds the labels and then adds it to the latent z, then decodes
        return x, mean, log_var
    
    def train(self, input, labels):
        with tf.GradientTape() as tape:
            logits, mean, log_var = self(input, labels)
            logits = flatten(logits)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(flatten(input), logits)
            # reconstruction_loss = tf.reduce_mean(tf.square(logits - input))
            # sum over dim then mean over batch
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(bce, axis=-1))
            kld_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1.0 + log_var - tf.square(mean) - tf.math.exp(log_var), axis=-1))
            loss = reconstruction_loss + kld_loss * self.beta
        variables = self.get_trainable_variables()
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.steps += 1

    # unnecessary now bc we sigmoid
    def normalize_images(self, image_data):
        min_vals = tf.reduce_min(image_data, axis=[1, 2, 3], keepdims=True)
        max_vals = tf.reduce_max(image_data, axis=[1, 2, 3], keepdims=True)
        return (image_data - min_vals) / (max_vals - min_vals)

    def generate_images(self, n=10):
        label = tf.cast(tfp.distributions.Bernoulli(probs=0.5).sample(sample_shape=(n, 40)), dtype=tf.float32)
        image_data = tf.nn.sigmoid(self.decoder(tf.random.normal(shape=[n, self.latent_dim]), label)) * 255.0
        image_data = tf.transpose(image_data, perm=[0, 2, 3, 1]).numpy()
        num_files = len(os.listdir(IMAGE_DIR))
        count = 0
        for image in image_data:
            image = Image.fromarray(image, mode="RGB")
            image.save(IMAGE_DIR+f'image{num_files+count}.jpg')
            count += 1

    def save_checkpoint(self, dir=CHKPT_DIR):
        checkpoint = tf.train.Checkpoint(**self.get_vars_dict())
        checkpoint.save(dir+'cvae_checkpoint')

    def get_vars_dict(self) -> dict:
        # gets all the variables in the model
        vars = {}
        for variable in self.get_all_variables():
            vars[variable.name] = variable
        return vars
    
    def get_trainable_variables(self) -> list:
        # gets all the trainable variables in the model
        vars = self.encoder.get_trainable_variables()
        vars.extend(self.decoder.get_trainable_variables())
        return vars
    
    def get_all_variables(self) -> list:
        # gets all the variables in the model
        vars = self.encoder.get_variables()
        vars.extend(self.decoder.get_variables())
        return vars

    def load_checkpoint(self, dir=CHKPT_DIR):
        vars = {}
        checkpoint_reader = tf.train.load_checkpoint(dir)
        var_to_shape_map = checkpoint_reader.get_variable_to_shape_map()
        for var_name in var_to_shape_map:
            name = var_name[:str.find(var_name, '/')]
            variable = tf.Variable(initial_value=checkpoint_reader.get_tensor(var_name), trainable=True, name=name)
            vars[name] = variable
        self.encoder.load_checkpoint(vars)
        self.decoder.load_checkpoint(vars)
