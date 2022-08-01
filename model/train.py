#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import sys

from classes.dataset.Generator import *
from classes.model.pix2code import *


def run(input_path, output_path, is_memory_intensive=False, pretrained_model=None):
    np.random.seed(1234)

    dataset = Dataset()
    dataset.load(input_path, generate_binary_sequences=True)
    dataset.save_metadata(output_path)
    dataset.voc.save(output_path)

    if not is_memory_intensive:
        dataset.convert_arrays()

        input_shape = dataset.input_shape
        output_size = dataset.output_size

        print(len(dataset.input_images), len(dataset.partial_sequences), len(dataset.next_words))
        print(dataset.input_images.shape, dataset.partial_sequences.shape, dataset.next_words.shape)
    else:
        gui_paths, img_paths = Dataset.load_paths_only(input_path)

        input_shape = dataset.input_shape
        output_size = dataset.output_size
        steps_per_epoch = dataset.size / BATCH_SIZE

        voc = Vocabulary()
        voc.retrieve(output_path)

        generator = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE, generate_binary_sequences=True)

    model = pix2code(input_shape, output_size, output_path)

    if pretrained_model is not None:
        model.model.load_weights(pretrained_model)

    if not is_memory_intensive:
        model.fit(dataset.input_images, dataset.partial_sequences, dataset.next_words)
    else:
        model.fit_generator(generator, steps_per_epoch=steps_per_epoch)

if __name__ == "__main__":
    # input_path = 'D:\\again\\pix2code2-master\\pix2code2-master\\datasets\\pix2code_datasets\\web\\all_data'
    input_path = 'D:\\web_database\\final'
    # input_path = 'D:\\again\\pix2code-master\\pix2code-master\\datasets\\pix2code_datasets\\web\\small'
    output_path = '../bin/17/'
    use_generator = False
    pretrained_weigths = '../bin/16/' + '/pix2code.h5'
    # train_autoencoder = True  # False if len(argv) < 3 else True if int(argv[2]) == 1 else False

    run(input_path, output_path, is_memory_intensive=use_generator, pretrained_model=pretrained_weigths)
