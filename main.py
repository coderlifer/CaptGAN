"""CaptGan Train/Eval/Infer module.
"""

import logging
import os
import time
import numpy as np
import tensorflow as tf

import configuration
import captiongan_model
import vocabulary


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("vocab_file", "./input/word_to_id.txt",
                       'Vocabulary file path')
tf.flags.DEFINE_string("train_dir", "./train_outputs/",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_integer("number_of_steps", 1000000,
                        "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_integer("num_gpus", 0,
                        "Number of gpus used for training. (0 or 1)")
tf.flags.DEFINE_string("mode", "train",
                       "Running mode, train, eval or infer")


logPath = './log/'
time_str = time.strftime("%Y%m%d-%H%M%S")


def train_op():
    assert FLAGS.input_file_pattern, "--input_file_pattern is required"
    assert FLAGS.train_dir, "--train_dir is required"

    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

    model_config = configuration.ModelConfig()
    model_config.input_file_pattern = FLAGS.input_file_pattern

    training_config = configuration.TrainingConfig()

    # Create training directory.
    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    logger = set_logger(logPath, time_str, os.path.basename(__file__))

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = captiongan_model.CaptionGAN(model_config, mode="train")
        model.build_graph()

        # Set up the learning rate.
        learning_rate_g = tf.constant(training_config.initial_learning_rate_g)
        learning_rate_d = tf.constant(training_config.initial_learning_rate_d)

        num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                 model_config.batch_size)
        # model.saveFreq = num_batches_per_epoch
        # learning rate decay
        if training_config.learning_rate_decay_factor > 0:
            decay_steps_g = int(num_batches_per_epoch *
                                training_config.num_epochs_per_decay)

            decay_steps_d = int(num_batches_per_epoch *
                                training_config.num_epochs_per_decay)

            learning_rate_g = \
                tf.train.exponential_decay(learning_rate_g,
                                           model.global_step,
                                           decay_steps=decay_steps_g,
                                           decay_rate=training_config.learning_rate_decay_factor,
                                           staircase=True)

            learning_rate_d = \
                tf.train.exponential_decay(learning_rate_d,
                                           model.global_step,
                                           decay_steps=decay_steps_d,
                                           decay_rate=training_config.learning_rate_decay_factor,
                                           staircase=True)

        # AdamOptimizer, AdagradOptimizer
        g_solver = tf.train.RMSPropOptimizer(learning_rate=learning_rate_g).\
            minimize(loss=model.g_loss,
                     global_step=model.global_step,
                     var_list=model.generator_variables)

        d_solver = tf.train.RMSPropOptimizer(learning_rate=learning_rate_d). \
            minimize(loss=model.d_loss,
                     global_step=model.global_step,
                     var_list=model.discriminator_variables)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for step in range(FLAGS.number_of_steps):
                if training_config.debug:
                    print('step ' + str(step) + ' begins')

                if training_config.debug:
                    logger.info('g_loss mmd before train: ')
                    logger.info(sess.run(model.g_loss))

                _, gc = sess.run([g_solver, model.g_loss])

                if training_config.debug:
                    logger.info('g_loss, mmd before train: ')
                    logger.info(sess.run(model.g_loss))

                if np.mod(step, training_config.dispFreq) == 0:
                    fake_caption_embedding_ids = sess.run(model.caption_embedding_ids)
                    dc = sess.run(model.d_loss)

                    print(' cost_g ' + str(gc) + ' cost_d ' + str(dc))
                    print(
                        "Generated:" + " ".join(
                            [vocab.id_to_word(x) for x in fake_caption_embedding_ids]))

                if np.mod(step, training_config.dg_ratio) == 0:
                    if training_config.debug:
                        logger.info('model.d_loss before train: ')
                        logger.info(sess.run([model.d_loss]))

                    _, dc = sess.run([d_solver, model.d_loss])

                    if training_config.debug:
                        logger.info('model.d_loss after train: ')
                        logger.info(sess.run([model.d_loss]))

                    if np.mod(step, training_config.dispFreq) == 0:
                        logger.info('Cost D {}'.format(dc))

                if np.mod(step, training_config.saveFreq) == 0:
                    logger.info('Saving model...')

                    save_path = saver.save(sess, logPath + time_str + ".ckpt")

                    logger.info('Model saved in file: %s' % save_path)


def evaluate():
    # predset = sess.run([result3])
    # predset = predset[0]
    # # Check def of this func why need to pass <data> object
    # [bleu2s, bleu3s, bleu4s] = cal_BLEU(predset, test, data, ngram, debug)
    #
    # logger.info('Valid BLEU2 = {}, BLEU3 = {}, BLEU4 = {}'.
    #             format(bleu2s, bleu3s, bleu4s))
    # print ('Valid BLEU (2, 3, 4): ' +
    #        ' '.join([str(round(x, 3)) for x in (bleu2s, bleu3s, bleu4s)]))
    pass


def infer():
    pass


def set_logger(log_Path, timestr, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

    fh = logging.FileHandler(log_Path + timestr + '.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)

    return logger


def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:1'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train_op()
        elif FLAGS.mode == 'eval':
            evaluate()
        elif FLAGS.mode == 'infer':
            # X_infer = helper.load_data(FLAGS.infer_data_path)
            # y_infer = np.ones((X_infer.shape[0],))
            #
            # infer(X_infer, y_infer)
            pass


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()


# def train():
#     assert FLAGS.input_file_pattern, "--input_file_pattern is required"
#     assert FLAGS.train_dir, "--train_dir is required"
#
#     model_config = configuration.ModelConfig()
#     model_config.input_file_pattern = FLAGS.input_file_pattern
#     model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file
#
#     training_config = configuration.TrainingConfig()
#
#     # Create training directory.
#     train_dir = FLAGS.train_dir
#     if not tf.gfile.IsDirectory(train_dir):
#         tf.logging.info("Creating training directory: %s", train_dir)
#         tf.gfile.MakeDirs(train_dir)
#
#     # Build the TensorFlow graph.
#     g = tf.Graph()
#     with g.as_default():
#         # Build the model.
#         model = captiongan_model.CaptionGAN(model_config, mode="train")
#         model.build_graph()
#
#         # Set up the learning rate.
#         learning_rate_g = tf.constant(training_config.initial_learning_rate_g)
#         learning_rate_d = tf.constant(training_config.initial_learning_rate_d)
#
#         learning_rate_decay_fn = None
#         if training_config.learning_rate_decay_factor > 0:
#             num_batches_per_epoch = (training_config.num_examples_per_epoch /
#                                      model_config.batch_size)
#             decay_steps = int(num_batches_per_epoch *
#                               training_config.num_epochs_per_decay)
#
#             def _learning_rate_decay_fn(learning_rate, global_step):
#                 return tf.train.exponential_decay(
#                     learning_rate,
#                     global_step,
#                     decay_steps=decay_steps,
#                     decay_rate=training_config.learning_rate_decay_factor,
#                     staircase=True)
#
#             learning_rate_decay_fn = _learning_rate_decay_fn
#
#         # Set up the training ops.
#         train_op_g = tf.contrib.layers.optimize_loss(
#             loss=model.g_loss,
#             global_step=model.global_step,
#             learning_rate=learning_rate_g,
#             optimizer=training_config.optimizer,
#             clip_gradients=training_config.clip_gradients,
#             learning_rate_decay_fn=learning_rate_decay_fn)
#
#         # Set up the Saver for saving and restoring model checkpoints.
#         saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)
#
#     # Run training.
#     tf.contrib.slim.learning.train(
#         train_op,
#         train_dir,
#         log_every_n_steps=FLAGS.log_every_n_steps,
#         graph=g,
#         global_step=model.global_step,
#         number_of_steps=FLAGS.number_of_steps,
#         init_fn=model.init_fn,
#         saver=saver)
