from resnet_architecture import *
import tensorflow as tf
import os

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_DEPTH = 6
FLAGS = tf.app.flags.FLAGS

''' Load list of  {filename, label_name, label_index} '''
def load_data(data_dir, data_lst):
    data = []
    train_lst = open(data_lst, 'r').read().splitlines()
    dictionary = open(FLAGS.dictionary, 'r').read().splitlines()
    for img_fn in train_lst:
        fn = os.path.join(data_dir, img_fn)
        label_name = img_fn.split('/')[0]
        label_index = dictionary.index(label_name)
        data.append({
            "filename": fn,
            "label_name": label_name,
            "label_index": label_index
        })
    return data

''' Load input data using queue (feeding)'''


def read_image_from_disk(input_queue):
    class ImageRecord(object):
        pass

    result = ImageRecord()

    result.height = IMAGE_HEIGHT
    result.width = IMAGE_WIDTH
    result.depth = IMAGE_DEPTH

    label_bytes = 0
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    assert record_bytes == 256 * 256 * 6

    value = tf.read_file(input_queue[0])

    record_bytes = tf.decode_raw(value, tf.uint8)

    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result, input_queue[1]


''' Non shuffle inputs , just for evaluation because of slow running  '''
def inputs(data_dir, data_lst):
    data = load_data(data_dir, data_lst)

    filenames = [d['filename'] for d in data]
    label_indexes = [d['label_index'] for d in data]

    input_queue = tf.train.slice_input_producer([filenames, label_indexes], shuffle=False)

    # read image and label from disk
    image, label = read_image_from_disk(input_queue)
    image = tf.cast(image.uint8image, tf.float32)

    ''' Data Augmentation '''
    image = tf.random_crop(image, [FLAGS.input_size, FLAGS.input_size, 4])
    image = tf.image.random_flip_left_right(image)

    # generate batch
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=1,
        capacity=FLAGS.min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=FLAGS.min_queue_examples,
        allow_smaller_final_batch=True)

    return image_batch, tf.reshape(label_batch, [FLAGS.batch_size])


def distorted_inputs(data_dir, data_lst,shuffle=True):
    data = load_data(data_dir, data_lst)

    filenames = [ d['filename'] for d in data ]
    label_indexes = [ d['label_index'] for d in data ]


    input_queue = tf.train.slice_input_producer([filenames, label_indexes], shuffle=shuffle)


    # read image and label from disk
    image, label = read_image_from_disk(input_queue)
    image = tf.cast(image.uint8image, tf.float32)

    print image

    ''' Data Augmentation '''
    image = tf.random_crop(image, [FLAGS.input_size, FLAGS.input_size, 4])
    image = tf.image.random_flip_left_right(image)

    # generate batch
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocess_threads,
        capacity=FLAGS.min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=FLAGS.min_queue_examples,
        allow_smaller_final_batch=True)

    return image_batch, tf.reshape(label_batch, [FLAGS.batch_size])
