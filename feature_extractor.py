import numpy as np
from resnet_architecture import *
import tensorflow as tf
import os
from datetime import datetime
from exp_config import *
from input import distorted_inputs,inputs

FLAGS = tf.app.flags.FLAGS

def top_k_error(predictions, labels, k):
    batch_size =float(FLAGS.batch_size) # predictions.get_shape().as_list()[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=k))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size
def num_correct(predictions,labels,k):
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=k))
    return tf.reduce_sum(in_top1)

def evaluate(is_training, logits, images, labels):
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
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print 'Restored from ', ckpt.model_checkpoint_path

        except Exception:
            return
    else:
        print 'No checkpoints found'
        return

    # evaluation
    coord = tf.train.Coordinator()
    data = None
    data_label = None
    nCorrect=0.0
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
        num_iter = int(np.ceil(NUM_EXAMPLES / FLAGS.batch_size))
        # true_count = 0-
        precision = 0
        # total_sample_count = num_iter * FLAGS.batch_size
        step = 0
        while step < num_iter and not coord.should_stop():
            extracted_features_, label_ = sess.run([logits,labels], {is_training: False})
            if data is None:
                data = extracted_features_
                data_label = label_
            else:

                data = np.concatenate((data, extracted_features_), axis=0)
                data_label = np.concatenate(data, data_label, axis=0)
                if step % 100 == 0:
                    print data.shape, data_label.shape
            step += 1

    except Exception as e:
        coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    # write results
    fn = os.path.join(FLAGS.eval_dir, 'features' + str(global_step) + '.txt')
    np.save(fn, data)

    fn_label = os.path.join(FLAGS.eval_dir, 'features_label' + str(global_step) + '.txt')
    np.save(fn_label, data_label)
    return


def main(_):
    global NUM_EXAMPLES
    NUM_EXAMPLES = len(open(FLAGS.eval_lst, 'r').read().splitlines())


    images, labels = inputs(FLAGS.data_dir, FLAGS.eval_lst)

    is_training = tf.placeholder('bool', [], name='is_training')  # placeholder for the fusion part

    logits = inference(images,
                       num_classes=None,
                       is_training=is_training,
                       num_blocks=[3, 4, 6, 3])
    evaluate(is_training,logits, images, labels)
    return

if __name__ == '__main__':
    tf.app.run(main)
