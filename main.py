import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os.path
import warnings
import tensorflow as tf
import helper


vgg_path = './data/vgg'

print(tf.__version__)

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, path):
    model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph = tf.get_default_graph()
    input_layer = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3 = graph.get_tensor_by_name('layer3_out:0')
    layer4 = graph.get_tensor_by_name('layer4_out:0')
    layer7 = graph.get_tensor_by_name('layer7_out:0')
    return input_layer, keep_prob, layer3, layer4, layer7

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    fcn8 = tf.layers.conv2d(vgg_layer7_out, filters=num_classes, kernel_size=1, name='fcn8')
    
    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=vgg_layer4_out.get_shape().as_list()[-1], kernel_size=4, strides=(2, 2), padding='SAME', name='fcn9')
    
    fcn9_skip = tf.add(fcn9, vgg_layer4_out, name='fcn9_skip')

    fcn10 = tf.layers.conv2d_transpose(fcn9_skip, filters=vgg_layer3_out.get_shape().as_list()[-1], kernel_size=4, strides=(2, 2), padding='SAME', name='fcn10')

    fcn10_skip = tf.add(fcn10, vgg_layer4_out, name='fcn10_skip')

    fcn11 = tf.layers.conv2d_transpose(fcn10_skip, filters=num_classes, kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")
    
    return fcn11

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFlow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    return None, None, None

if __name__ == "__main__":
    data_dir = './data'
    runs_dir = './runs'
    helper.maybe_download_pretrained_vgg(data_dir)
