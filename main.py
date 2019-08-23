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
    

if __name__ == "__main__":
    data_dir = './data'
    runs_dir = './runs'
    helper.maybe_download_pretrained_vgg(data_dir)
