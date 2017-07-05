import tensorflow as tf
from interprettensor.models import alexnet

from tensorflow.contrib.framework import get_or_create_global_step

import os
import numpy as np
from scipy.misc import imread, imresize

from caffe_classes import class_names 

def set_gpu(gpu):
    cuda = True if gpu is not None else False
    use_mult_gpu = isinstance(gpu, list)
    if use_mult_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu).strip('[').strip(']')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu

mu = 114.45155662981172
def load_image(file_path, size=(227,227)):
    im = (imresize(imread(file_path)[:,:,:3], size)).astype(np.float32)
    im = im - mu
    r = im[:,:,0].copy()
    b = im[:,:,2].copy()
    im[:,:,0] = b
    im[:,:,2] = r
    return im.astype(np.float32)

def show_image(im):
    b = im[:,:,0].copy()
    r = im[:,:,2].copy()
    im[:,:,0] = r
    im[:,:,2] = b
    im = im + mu

    f, ax = plt.subplots(1,1)
    ax.imshow(im.astype(np.uint8))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def main():
    set_gpu(gpu=1)

    weights_path = 'bvlc_alexnet.npy'

    graph = tf.Graph()
    tf_global_step = get_or_create_global_step()

    logits_layer_name = 'alexnet/fc8/squeezed:0'

    with graph.as_default():
        images = tf.placeholder(tf.float32, shape=(1,227,227,3))
        net, end_points = alexnet.alexnet(images, is_training=False, 
                pretrained_weights=weights_path)

        sess = tf.Session(graph=graph)
        sess.run(tf.global_variables_initializer())
    
        # construct scalar neuron tensor
        logits = graph.get_tensor_by_name(logits_layer_name)
        neuron_selector = tf.placeholder(tf.int32)
        y = logits[0][neuron_selector]
        
        # construct tensor for predictions
        prediction = tf.argmax(logits, 1)

    # Load the image
    path = 'poodle.png'
    im = load_image(path)
    #show_image(im)

    # Make a prediction. 
    prediction_class, acts = sess.run([prediction, logits], feed_dict = {images: [im]})
    prediction_class = prediction_class[0]

    print("Prediction Class: %d %s\n" % (prediction_class, class_names[prediction_class]))

    num_top = 10
    print('Top-%d Predicted Classes:' % num_top)
    idx = np.argsort(acts[0])[::-1]
    for i in range(num_top):
        print i+1, idx[i], class_names[idx[i]], acts[0][idx[i]]


if __name__ == '__main__':
    main()
