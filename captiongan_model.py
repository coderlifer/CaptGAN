from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import inputs as input_ops

from collections import OrderedDict


class CaptionGAN(object):
    """CaptionGAN model."""

    def __init__(self, config, mode):
        """CaptionGAN constructor.

        Args:
          config: Hyper-parameters.
          mode: One of 'train', 'eval' and 'infer.
        """
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode

        # Reader for the input data.
        self.reader = tf.TFRecordReader()

        # To match the "Show and Tell" paper we initialize all variables with a
        # random uniform initializer.
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        # An int32 Tensor with shape [batch_size, padded_length].
        self.input_seqs = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.target_seqs = None

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.input_mask = None

        # A float32 Tensor with shape [batch_size, embedding_size].
        self.image_embeddings = None

        # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
        self.seq_embeddings = None

        self.images_and_captions = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_losses = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_loss_weights = None

        # Global step Tensor.
        self.global_step = None

        self._extra_train_ops = []

    # ######## ######## ######## ######## ######## ######## ##
    # ######## ######## Text generation part ######## ########
    # ######## ######## ######## ######## ######## ######## ##

    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def build_inputs(self):
        """Input prefetching, preprocessing and batching.

        Outputs:
          self.input_seqs
          self.target_seqs (training and eval only)
          self.input_mask (training and eval only)
        """
        if self.mode == "inference":
            # In inference mode, image_embeddings and inputs are fed via placeholders.
            image_embeddings_feed = tf.placeholder(dtype=tf.string,
                                                   shape=[],
                                                   name="image_embeddings_feed")
            input_feed = tf.placeholder(dtype=tf.int64,
                                        shape=[None],  # batch_size,      seq_length ?
                                        name="input_feed")

            # Process image_embeddings and insert batch dimensions.
            image_embeddings = tf.expand_dims(image_embeddings_feed, 0)
            input_seqs = tf.expand_dims(input_feed, 1)

            # No target sequences or input mask in inference mode.
            target_seqs = None
            input_mask = None

            self.images_and_captions = [image_embeddings, input_feed]
        else:
            # Prefetch serialized SequenceExample protos.
            input_queue = input_ops.prefetch_input_data(
                self.reader,
                self.config.input_file_pattern,
                is_training=self.is_training(),
                batch_size=self.config.batch_size,
                values_per_shard=self.config.values_per_input_shard,
                input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                num_reader_threads=self.config.num_input_reader_threads)

            # Image processing and random distortion. Split across multiple threads
            # with each thread applying a slightly different distortion.
            assert self.config.num_preprocess_threads % 2 == 0
            images_and_captions = []
            for thread_id in range(self.config.num_preprocess_threads):
                serialized_sequence_example = input_queue.dequeue()

                # encoded_image: [0.1 0.2 0.1 ... 0.2]
                # caption: [1, 7, ... , 4], no padding
                encoded_image, caption = input_ops.parse_sequence_example(
                    serialized_sequence_example,
                    image_feature=self.config.image_feature_name,
                    caption_feature=self.config.caption_feature_name)

                images_and_captions.append([encoded_image, caption])

            # Batch inputs.
            queue_capacity = (2 * self.config.num_preprocess_threads *
                              self.config.batch_size)
            image_embeddings, input_seqs, target_seqs, input_mask = (
                input_ops.batch_with_dynamic_pad(images_and_captions,
                                                 batch_size=self.config.batch_size,
                                                 queue_capacity=queue_capacity))

            self.images_and_captions = [image_embeddings, input_seqs]

        self.image_embeddings = image_embeddings
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.input_mask = input_mask

    def build_seq_embeddings(self):
        """Builds the input sequence embeddings.

        Inputs:
          self.input_seqs

        Outputs:
          self.seq_embeddings
        """
        with tf.variable_scope("seq_embedding", reuse=True), tf.device("/cpu:0"):
            self.embedding_map = tf.get_variable(
                name="seq_embedding",
                shape=[self.config.vocab_size, self.config.embedding_size],
                trainable=False)

        self.seq_embeddings = tf.nn.embedding_lookup(self.embedding_map, self.input_seqs)

    def build_model(self, reuse=False):
        """Builds the model.

        Inputs:
          self.image_embeddings
          self.seq_embeddings
          self.target_seqs (training and eval only)
          self.input_mask (training and eval only)

        Outputs:
          self.total_loss (training and eval only)
          self.target_cross_entropy_losses (training and eval only)
          self.target_cross_entropy_loss_weights (training and eval only)
        """
        # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
        # modified LSTM in the "Show and Tell" paper has no biases and outputs
        # new_c * sigmoid(o).

        with tf.variable_scope('Generator', reuse=reuse):
            with tf.variable_scope('img_dim_transform'):
                # img_embeddings = self.image_embeddings
                self.input_lstm = self._fully_connected(self.image_embeddings,
                                                        self.config.embedding_size,
                                                        'fc_1')

            with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                    num_units=self.config.num_lstm_units, state_is_tuple=True)
                if self.mode == "train":
                    lstm_cell = tf.contrib.rnn.DropoutWrapper(
                        lstm_cell,
                        input_keep_prob=self.config.lstm_dropout_keep_prob,
                        output_keep_prob=self.config.lstm_dropout_keep_prob)

                # Feed the image embeddings to set the initial LSTM state.
                zero_state = lstm_cell.zero_state(
                    batch_size=self.input_lstm.get_shape()[0], dtype=tf.float32)

                _, initial_state = lstm_cell(self.input_lstm, zero_state)

                # Allow the LSTM variables to be reused.
                lstm_scope.reuse_variables()

                if self.mode == "inference":
                    # In inference mode, use concatenated states for convenient feeding and
                    # fetching.
                    # called by name='initial_state' when inferring
                    tf.concat(values=initial_state, axis=1, name="initial_state")

                    # Placeholder for feeding a batch of concatenated states.
                    state_feed = tf.placeholder(dtype=tf.float32,
                                                shape=[None, sum(lstm_cell.state_size)],
                                                name="state_feed")
                    state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

                    # Run a single LSTM step.
                    lstm_outputs, state_tuple = lstm_cell(
                        inputs=tf.squeeze(self.seq_embeddings, axis=[1]),
                        state=state_tuple)

                    # Concatentate the resulting state.
                    tf.concat(values=state_tuple, axis=1, name="state")
                else:
                    # Run the batch of sequence embeddings through the LSTM.
                    sequence_length = tf.reduce_sum(self.input_mask, 1)
                    # lstm_outputs: [batch_size, max_seq_len_this_batch, output_size]
                    lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                        inputs=self.seq_embeddings,
                                                        sequence_length=sequence_length,
                                                        initial_state=initial_state,
                                                        dtype=tf.float32,
                                                        scope=lstm_scope)

            # Stack batches vertically.
            lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

            with tf.variable_scope("logits") as logits_scope:
                logits = tf.contrib.layers.fully_connected(
                    inputs=lstm_outputs,
                    num_outputs=self.config.vocab_size,
                    activation_fn=None,
                    weights_initializer=self.initializer,
                    scope=logits_scope)

                self.logits = tf.reshape(logits,
                                         [self.config.batch_size, -1, self.config.vocab_size])

            if self.mode == "inference":
                tf.nn.softmax(self.logits, name="softmax")
            else:
                # targets = tf.reshape(self.target_seqs, [-1])
                # weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

                # # Compute losses.
                # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                #                                                         logits=self.logits)
                # batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                #                     tf.reduce_sum(weights),
                #                     name="batch_loss")
                # tf.losses.add_loss(batch_loss)
                # total_loss = tf.losses.get_total_loss()
                #
                # # Add summaries.
                # tf.summary.scalar("losses/batch_loss", batch_loss)
                # tf.summary.scalar("losses/total_loss", total_loss)
                for var in tf.trainable_variables():
                    tf.summary.histogram("parameters/" + var.op.name, var)

                    # self.total_loss = total_loss
                    # self.target_cross_entropy_losses = losses  # Used in evaluation.
                    # self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def _build_G(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
        self.build_seq_embeddings()
        self.build_model()
        self.setup_global_step()

        # [batch_size, max_seq_len_this_batch, vocab_size]
        # shp = tf.shape(self.logits)

        # [batch_size, max_seq_len_this_batch]
        self.caption_embedding_ids = tf.arg_max(self.logits, dimension=2)
        # caption_embedding_ids = tf.reshape(caption_embedding_ids, shape=[shp[0], shp[1], 1])

        return self.caption_embedding_ids

    # ######## ######## ######## ######## ######## ######## ##
    # ######## ######## Discriminator part ######## ######## #
    # ######## ######## ######## ######## ######## ######## ##

    def _conv(self, name, x, filter_size, in_channels, out_channels, strides):
        """Convolution."""
        with tf.variable_scope(name):
            # n = filter_size * filter_size * out_filters
            kernel = tf.get_variable('DW',
                                     [filter_size, self.config.embedding_size, in_channels, out_channels],
                                     tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('biases',
                                shape=[out_channels],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(1))

            con2d_op = tf.nn.conv2d(x, kernel, strides, padding='VALID')

            return tf.nn.bias_add(con2d_op, b)

    def _leaky_relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _max_pool(self, x, ksize, strides):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME',
                              name='max_pool')

    def _fully_connected(self, x, out_dim, name):
        """FullyConnected layer for final output."""
        with tf.variable_scope(name):
            x = tf.reshape(x, [self.config.batch_size, -1])
            w = tf.get_variable('DW',
                                [x.get_shape()[1], out_dim],
                                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
            b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer(1))
            return tf.nn.xw_plus_b(x, w, b)

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.summary.histogram(var.op.name, var)

        return tf.multiply(self.config.weight_decay_rate, tf.add_n(costs))

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_D(self, output_lstm, reuse):
        """Build the core generator within the graph.

        :param output_lstm: [batch_size, max_seq_len_this_batch]
        :param reuse: whether reuse parameters or not
        :return:
        """
        # filter_sizes = [3, 4, 5]
        # out_channels = 300
        filter_sizes = self.config.filter_sizes
        out_channels = self.config.out_channels

        params = OrderedDict()
        params['acc_fake_xx'] = np.eye(len(filter_sizes) * out_channels) * 1
        params['acc_real_xx'] = np.eye(len(filter_sizes) * out_channels) * 1
        params['acc_fake_mean'] = np.zeros(len(filter_sizes) * out_channels)
        params['acc_real_mean'] = np.zeros(len(filter_sizes) * out_channels)
        params['seen_size'] = 0
        self.s_params = params

        # [batch_size, max_seq_len_this_batch, embedding_size]
        input_cnn = tf.nn.embedding_lookup(self.embedding_map, output_lstm)

        # [batch_size, max_seq_len_this_batch, embedding_size, 1]
        input_cnn = tf.expand_dims(input=input_cnn, axis=-1)

        with tf.variable_scope('Discriminator', reuse=reuse):
            # discriminator part
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # [batch_size, max_seq_len_this_batch - filter_size + 1, 1, out_channels]
                    conv_op = self._conv('conv',
                                         input_cnn,
                                         filter_size,
                                         1,
                                         out_channels,
                                         [1, 1, 1, 1])

                    bn = tf.contrib.layers.batch_norm(inputs=conv_op,
                                                      is_training=self.mode,
                                                      scope='batch_norm')

                    lrelu = self._leaky_relu(bn, self.config.relu_leakiness)

                    # Max-pooling over the outputs
                    shp = tf.shape(lrelu)

                    # [batch_size, 1, 1, out_channels]
                    pooled = tf.nn.max_pool(
                        lrelu,
                        ksize=[1, shp[1], 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")

                    # len(filter_sizes) * [batch_size, 1, 1, out_channels]
                    pooled_outputs.append(pooled)

            with tf.variable_scope('concat_layer'):
                # Combine all the pooled features
                h_pool = tf.concat(values=pooled_outputs, axis=3)

                # [batch_size, len(filter_sizes) * out_channels]
                captions_features = tf.reshape(h_pool, [-1, out_channels * len(filter_sizes)])
                captions_features = tf.contrib.layers.batch_norm(inputs=captions_features,
                                                                 is_training=self.mode,
                                                                 scope='batch_norm')

            # Final (unnormalized) scores and predictions
            with tf.variable_scope("logit"):
                fc_1 = self._fully_connected(captions_features, 200, 'fc_1')
                captions_features_compressed = tf.nn.sigmoid(fc_1, 'sigmoid_fc1')

                D_logit = self._fully_connected(captions_features_compressed, 1, 'D_logit')
                # D_prob = tf.nn.tanh(D_logit, 'tanh_fc2')
                # D_prob = tf.nn.softmax(D_logit)
                # D_prob = tf.nn.sigmoid(D_logit, 'sigmoid_fc2')

                # mmd compressed features
                capt_feat_mmd = self._fully_connected(captions_features_compressed,
                                                      self.config.dim_mmd,
                                                      'capt_feat_mmd')

                # infoGan part
                proposal = self._fully_connected(captions_features_compressed, 2, 'proposal')

            # encoder part
            with tf.variable_scope('recons_z'):
                # reconstruct z^ (image encoding)
                recons_z = self._fully_connected(captions_features, 1200, 'recons_1')
                recons_z = tf.nn.sigmoid(recons_z, 'recons_1_sigmoid')
                recons_z = self._fully_connected(recons_z, 1200, 'recons_2')
                recons_z = tf.nn.tanh(recons_z, 'recons_2_tanh')

            return D_logit, capt_feat_mmd, proposal, recons_z, captions_features, captions_features_compressed

    # ######## ######## ######## ######## ######## ######## ########
    # ######## ######## Construct the whole graph  ######## ########
    # ######## ######## ######## ######## ######## ######## ########

    def build_train_op(self, D_logit_fake, capt_feat_mmd_fake, D_proposal_fake, recons_z_fake, capt_feat_fake,
                       capt_feat_compressed_fake,
                       D_logit_real, capt_feat_mmd_real, D_proposal_real, recons_z_real, capt_feat_real,
                       capt_feat_compressed_real):
        with tf.variable_scope('Train_OP'):

            n_z = self.config.out_channels * len(self.config.filter_sizes)

            with tf.variable_scope('trainable_variables'):
                self.generator_variables = []
                for var in tf.trainable_variables():
                    if var.op.name.find(r'Generator') > -1:
                        self.generator_variables.append(var)

                self.discriminator_variables = []
                for var in tf.trainable_variables():
                    if var.op.name.find(r'Discriminator') > -1:
                        self.discriminator_variables.append(var)

            with tf.variable_scope('mmd_loss'):
                # sufficient statistics
                cur_size = self.s_params['seen_size'] * 1.0
                identity = tf.eye(self.config.image_embedding_dim) * self.config.diag

                fake_xx = tf.matmul(tf.transpose(capt_feat_fake), capt_feat_fake)
                real_xx = tf.matmul(tf.transpose(capt_feat_real), capt_feat_real)
                acc_fake_xx = (self.s_params['acc_fake_xx'] * cur_size + fake_xx) \
                    / (cur_size + self.config.batch_size)
                acc_real_xx = (self.s_params['acc_real_xx'] * cur_size + real_xx) \
                    / (cur_size + self.config.batch_size)

                fake_mean = tf.reduce_mean(capt_feat_fake, axis=0)
                real_mean = tf.reduce_mean(capt_feat_real, axis=0)
                acc_fake_mean = (self.s_params['acc_fake_mean'] * cur_size +
                                 fake_mean * self.config.batch_size) / (cur_size + self.config.batch_size)
                acc_real_mean = (self.s_params['acc_real_mean'] * cur_size +
                                 real_mean * self.config.batch_size) / (cur_size + self.config.batch_size)

                cov_fake = acc_fake_xx - \
                    tf.matmul(tf.reshape(acc_fake_mean, [-1, 1]),
                              tf.transpose(tf.reshape(acc_fake_mean, [-1, 1]))) + identity
                cov_real = acc_real_xx - \
                    tf.matmul(tf.reshape(acc_real_mean, [-1, 1]),
                              tf.transpose(tf.reshape(acc_real_mean, [-1, 1]))) + identity

                cov_fake_inv = tf.matrix_inverse(cov_fake)
                cov_real_inv = tf.matrix_inverse(cov_real)

                if self.config.feature_match == 'moment':
                    self.fake_obj = tf.reduce_sum(tf.square(fake_mean - real_mean))
                elif self.config.feature_match == 'JSD_acc':
                    temp1 = tf.trace(tf.matmul(cov_fake_inv, cov_real) +
                                     tf.matmul(cov_real_inv, cov_fake))

                    temp2 = tf.matmul(tf.matmul((acc_fake_mean - acc_real_mean),
                                                (cov_fake_inv + cov_real_inv)),
                                      tf.transpose(acc_fake_mean - acc_real_mean))

                    self.fake_obj = temp1 + temp2
                elif self.config.feature_match == 'mmd':
                    # too many nodes, use scan
                    kxx, kxy, kyy = 0, 0, 0
                    dividend = 1
                    dist_x, dist_y = capt_feat_fake / dividend, capt_feat_real / dividend
                    x_sq = tf.reshape(
                        tf.reduce_sum(tf.square(dist_x), axis=1), [-1, 1])  # 64*1
                    y_sq = tf.reshape(
                        tf.reduce_sum(tf.square(dist_y), axis=1), [-1, 1])  # 64*1
                    tempxx = -2 * tf.matmul(dist_x, tf.transpose(dist_x)) + \
                        x_sq + tf.transpose(x_sq)  # (xi -xj)**2
                    tempxy = -2 * tf.matmul(dist_x, tf.transpose(dist_y)) + \
                        x_sq + tf.transpose(y_sq)  # (xi -yj)**2
                    tempyy = -2 * tf.matmul(dist_y, tf.transpose(dist_y)) + \
                        y_sq + tf.transpose(y_sq)  # (yi -yj)**2

                    for sigma in self.config.sigma_range:
                        kxx += tf.reduce_mean(tf.exp(-tempxx / 2 / tf.square(sigma)))
                        kxy += tf.reduce_mean(tf.exp(-tempxy / 2 / tf.square(sigma)))
                        kyy += tf.reduce_mean(tf.exp(-tempyy / 2 / tf.square(sigma)))

                    self.fake_obj = tf.sqrt(kxx + kyy - 2 * kxy)
                elif self.config.feature_match == 'mmd_cov':
                    kxx, kxy, kyy = 0, 0, 0
                    cov_sum = (cov_fake + cov_real) / 2
                    cov_sum_inv = tf.matrix_inverse(cov_sum)

                    dividend = 1
                    dist_x, dist_y = capt_feat_fake / dividend, capt_feat_real / dividend
                    cov_inv_mat = cov_sum_inv
                    x_sq = tf.reshape(
                        tf.reduce_sum(
                            tf.matmul(tf.matmul(dist_x, cov_inv_mat), dist_x), axis=1), [-1, 1])
                    y_sq = tf.reshape(
                        tf.reduce_sum(
                            tf.matmul(tf.matmul(dist_y, cov_inv_mat), dist_y), axis=1), [-1, 1])

                    tempxx = -2 * tf.matmul(
                        tf.matmul(dist_x, cov_inv_mat), tf.transpose(dist_x)) + \
                        x_sq + tf.transpose(x_sq)  # (xi -xj)**2
                    tempxy = -2 * tf.matmul(
                        tf.matmul(dist_x, cov_inv_mat), tf.transpose(dist_y)) + \
                        x_sq + tf.transpose(y_sq)  # (xi -yj)**2
                    tempyy = -2 * tf.matmul(
                        tf.matmul(dist_y, cov_inv_mat), tf.transpose(dist_y)) + \
                        y_sq + tf.transpose(y_sq)  # (yi -yj)**2

                    for sigma in self.config.sigma_range:
                        kxx += tf.reduce_mean(tf.exp(-tempxx / 2 / tf.square(sigma)))
                        kxy += tf.reduce_mean(tf.exp(-tempxy / 2 / tf.square(sigma)))
                        kyy += tf.reduce_mean(tf.exp(-tempyy / 2 / tf.square(sigma)))

                    self.fake_obj = tf.sqrt(kxx + kyy - 2 * kxy)
                elif self.config.feature_match == 'mmd_ld':
                    kxx, kxy, kyy = 0, 0, 0

                    fake_mmd = capt_feat_mmd_fake
                    # mlp_layer_tanh(d_params, capt_feat_fake, prefix='dis_mmd')
                    real_mmd = capt_feat_mmd_real
                    # mlp_layer_tanh(d_params, capt_feat_real, prefix='dis_mmd')

                    dividend = self.config.dim_mmd  # for numerical stability & scale with
                    dist_x, dist_y = fake_mmd / dividend, real_mmd / dividend

                    x_sq = tf.reshape(
                        tf.reduce_sum(tf.square(dist_x), axis=1), [-1, 1])  # 64*1
                    y_sq = tf.reshape(
                        tf.reduce_sum(tf.square(dist_y), axis=1), [-1, 1])  # 64*1

                    tempxx = -2 * tf.matmul(dist_x, tf.transpose(dist_x)) + \
                        x_sq + tf.transpose(x_sq)  # (xi -xj)**2
                    tempxy = -2 * tf.matmul(dist_x, tf.transpose(dist_y)) + \
                        x_sq + tf.transpose(y_sq)  # (xi -yj)**2
                    tempyy = -2 * tf.matmul(dist_y, tf.transpose(dist_y)) + \
                        y_sq + tf.transpose(y_sq)  # (yi -yj)**2

                    for sigma in self.config.sigma_range:
                        kxx += tf.reduce_sum(tf.exp(-tempxx / 2 / sigma))
                        kxy += tf.reduce_sum(tf.exp(-tempxy / 2 / sigma))
                        kyy += tf.reduce_sum(tf.exp(-tempyy / 2 / sigma))

                    self.fake_obj = tf.sqrt(kxx + kyy - 2 * kxy)
                elif self.config.feature_match == 'mmd_h':
                    # too many nodes, use scan
                    kxx, kxy, kyy = 0, 0, 0

                    fake_mmd = capt_feat_compressed_fake
                    # fake_mmd = mlp_layer_tanh(d_params, fake_output1, prefix='dis_mmd')
                    # fake_mmd = middle_layer(d_params, tensor.tanh(capt_feat_fake), prefix='dis_d')

                    real_mmd = capt_feat_compressed_real
                    # real_mmd = mlp_layer_tanh(d_params, real_output, prefix='dis_mmd')
                    # real_mmd = middle_layer(d_params, tensor.tanh(capt_feat_real), prefix='dis_d')

                    dividend = 1
                    dist_x, dist_y = fake_mmd / dividend, real_mmd / dividend
                    x_sq = tf.reshape(
                        tf.reduce_sum(tf.square(dist_x), axis=1), [-1, 1])  # 64*1
                    y_sq = tf.reshape(
                        tf.reduce_sum(tf.square(dist_y), axis=1), [-1, 1])  # 64*1
                    tempxx = -2 * tf.matmul(dist_x, tf.transpose(dist_x)) \
                        + x_sq + tf.transpose(x_sq)  # (xi -xj)**2
                    tempxy = -2 * tf.matmul(dist_x, tf.transpose(dist_y)) \
                        + x_sq + tf.transpose(y_sq)  # (xi -yj)**2
                    tempyy = -2 * tf.matmul(dist_y, tf.transpose(dist_y)) \
                        + y_sq + tf.transpose(y_sq)  # (yi -yj)**2

                    for sigma in self.config.sigma_range:
                        kxx += tf.reduce_mean(tf.exp(-tempxx / 2 / tf.square(sigma)))
                        kxy += tf.reduce_mean(tf.exp(-tempxy / 2 / tf.square(sigma)))
                        kyy += tf.reduce_mean(tf.exp(-tempyy / 2 / tf.square(sigma)))

                    self.fake_obj = tf.sqrt(kxx + kyy - 2 * kxy)
                else:
                    self.fake_obj = - tf.reduce_sum(tf.log(D_logit_fake + 1e-6)) / n_z

                # self.s_params['acc_fake_xx'] = acc_fake_xx
                # self.s_params['acc_real_xx'] = acc_real_xx
                # self.s_params['acc_fake_mean'] = acc_fake_mean
                # self.s_params['acc_real_mean'] = acc_real_mean
                # self.s_params['seen_size'] = self.s_params['seen_size'] + self.config.batch_size

                tf.summary.scalar('mmd_loss', self.fake_obj)

            with tf.variable_scope('recons_loss'):
                r_t = recons_z_fake / 2.0 + .5
                z_t = self.image_embeddings / 2.0 + .5
                self.r_loss = tf.reduce_sum(
                    -z_t * tf.log(r_t + 0.0001) - (1. - z_t) * tf.log(1.0001 - r_t)) \
                    / (self.config.batch_size * n_z)

                tf.summary.scalar('r_loss', self.r_loss)

            with tf.variable_scope('infoGan_loss'):
                # D_proposal_fake = (D_proposal_fake + 1) / 2
                # D_proposal_fake = tf.log(D_proposal_fake)
                # z_code = tf.cast(z[:, 0], dtype='int32')
                # z_index = np.arange(n_z)
                # fake_logent = D_proposal_fake[z_index, z_code]
                # self.l_I = tf.reduce_sum(fake_logent)
                self.l_I = 0

                tf.summary.scalar('l_I', self.l_I)

            with tf.variable_scope('gan_loss'):
                if not self.config.wgan:
                    D_logit_fake = tf.nn.sigmoid(D_logit_fake) * \
                        (1 - 2 * self.config.label_smoothing) + self.config.label_smoothing

                    D_logit_real = tf.nn.sigmoid(D_logit_real) * \
                        (1 - 2 * self.config.label_smoothing) + self.config.label_smoothing

                # self._decay()
                if self.config.wgan:
                    # self.gan_loss_g = tf.reduce_mean(
                    #     tf.nn.sigmoid_cross_entropy_with_logits(
                    #         logits=D_logit_fake,
                    #         labels=tf.ones_like(D_logit_fake)
                    #     )
                    # )
                    self.gan_loss_g = -tf.reduce_sum(D_logit_fake) / n_z
                    self.gan_loss_d = tf.reduce_sum(D_logit_fake) / n_z - \
                        tf.reduce_sum(D_logit_real) / self.config.batch_size
                else:
                    self.gan_loss_g = self.fake_obj
                    self.gan_loss_d = - tf.reduce_sum(tf.log(1 - D_logit_fake + 1e-6)) / n_z - \
                        tf.reduce_sum(tf.log(D_logit_real + 1e-6)) / self.config.batch_size

                tf.summary.scalar('gan_loss_g', self.gan_loss_g)
                tf.summary.scalar('gan_loss_d', self.gan_loss_d)

            with tf.variable_scope('g_d_loss'):
                # self.g_loss = self.gan_loss_g + \
                #               self.config.lambda_recon * self.r_loss - \
                #               self.config.lambda_q * self.l_I / n_z
                self.g_loss = self.gan_loss_g - \
                              self.config.lambda_q * self.l_I / n_z
                tf.summary.scalar('g_loss', self.g_loss)

                self.d_loss = self.gan_loss_d - \
                    self.config.lambda_fm * self.fake_obj + \
                    (self.config.lambda_recon * self.r_loss +
                     self.config.lambda_q * self.l_I / n_z)
                tf.summary.scalar('d_loss', self.d_loss)

            with tf.variable_scope('optimizer'):
                self.g_solver = tf.train.AdamOptimizer(learning_rate=0.00005).\
                    minimize(loss=self.g_loss,
                             var_list=self.generator_variables)
                self.d_solver = tf.train.AdamOptimizer(learning_rate=0.0001).\
                    minimize(loss=self.d_loss,
                             var_list=self.discriminator_variables)

    def build_graph(self):
        """Build a whole graph for the model."""
        self.global_step = tf.contrib.framework.get_or_create_global_step()

        fake_caption_embeddings = self._build_G()

        D_logit_fake, capt_feat_mmd_fake, D_proposal_fake, recons_z_fake, capt_feat_fake, \
            capt_feat_compressed_fake = self._build_D(fake_caption_embeddings, False)

        D_logit_real, capt_feat_mmd_real, D_proposal_real, recons_z_real, capt_feat_real, \
            capt_feat_compressed_real = self._build_D(self.images_and_captions[1], True)

        if self.mode == 'train':
            self.build_train_op(D_logit_fake, capt_feat_mmd_fake, D_proposal_fake, recons_z_fake, capt_feat_fake,
                                capt_feat_compressed_fake,
                                D_logit_real, capt_feat_mmd_real, D_proposal_real, recons_z_real, capt_feat_real,
                                capt_feat_compressed_real)

            # return fake_caption_embeddings, g_solver, d_solver, g_loss, d_loss

        self.summaries = tf.summary.merge_all()
