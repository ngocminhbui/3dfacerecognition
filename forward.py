import tensorflow as tf
import numpy as np
from exp_config import *
from resnet_architecture import inference
from input import inputs
FLAGS = tf.app.flags.FLAGS

import sys
def main(argv):
    print path
    file_contents = tf.read_file(FLAGS.data_dir + '/'+ path + '_crop.png')
    image = tf.image.decode_png(file_contents, channels=4)
    image=tf.cast(image ,tf.float32)

    image = tf.random_crop(image, [FLAGS.input_size, FLAGS.input_size, 4])
    image = tf.image.random_flip_left_right(image)

    image = tf.reshape(image, [1, 224, 224, 4])

    is_training = tf.placeholder('bool', [], name='is_training')
    logits = inference(image,
                       num_classes=FLAGS.num_classes,
                       is_training=is_training,
                       num_blocks=[3, 4, 6, 3])
    eval_single_img(is_training,logits, image, None)
    return

def eval_single_img(is_training, logits, images, labels):
    predictions = tf.nn.softmax(logits)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

    init = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)

    # restore checkpoint
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        try:
            saver.restore(sess, ckpt.model_checkpoint_path)
        except Exception:
            return
    else:
        print 'No checkpoints found'
        return

    print 'Model:', ckpt.model_checkpoint_path, 'restored.'

    scores_= sess.run([predictions], {is_training: False})
    scores_ = np.asarray(scores_).reshape((-1,10))

    preds_ = scores_.argmax(axis=1).squeeze()

    dictionary = open(FLAGS.dictionary, 'r').read().splitlines()

    print dictionary[preds_]
if __name__ == '__main__':
    global path
    path = sys.argv[1]
    tf.app.run()