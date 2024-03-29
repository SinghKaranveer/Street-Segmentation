import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os.path
import warnings
import tensorflow as tf
import helper


vgg_path = './data/vgg'
num_classes = 2
image_shape = (160, 576)
EPOCHS = 70
BATCH_SIZE = 16
DROPOUT = 0.75

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

    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

    fcn8 = tf.layers.conv2d(layer7, filters=num_classes, kernel_size=1, name="fcn8")

    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")

    fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

    fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=num_classes,
    kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")

    return fcn11


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    output_reshaped = tf.reshape(nn_last_layer, (-1, num_classes), name="output_reshaped")

    label_reshaped = tf.reshape(correct_label, (-1, num_classes), name="label_reshaped")

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output_reshaped, labels=label_reshaped[:])

    loss = tf.reduce_mean(cross_entropy, name="loss")

    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, name="train")

    return output_reshaped, train, loss


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    for epoch in range(epochs):
        total_loss = 0
        for x, y in get_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, train_op], 
            feed_dict={input_image: x, correct_label: y, keep_prob: 0.75, learning_rate: 0.0001})

            total_loss += loss

        print("EPOCH {} ...".format(epoch + 1))
        print("Loss = {:.3f}".format(total_loss))
        print()


def run():
    num_classes = 2
    learning_rate = 0.00001
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        vgg_path = os.path.join(data_dir, 'vgg')
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        learning_rate = tf.placeholder(tf.float32)
        keep_prob = tf.placeholder(tf.float32)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)

        output = layers(layer3, layer4, layer7, num_classes)

        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)


        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print("MODEL SUCCESSFULLY BUILT")
        print("BEGINNING TRAINING")
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, 
            train_op, cross_entropy_loss, image_input,
            correct_label, keep_prob, learning_rate)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)


        print("COMPLETE")

if __name__ == "__main__":
    run()
