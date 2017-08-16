
"""Image-to-text model and training configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """Sets the default model hyperparameters."""
        # File pattern of sharded TFRecord file containing SequenceExample protos.
        # Must be provided in training and evaluation modes.
        self.input_file_pattern = None

        # Approximate number of values per input shard. Used to ensure sufficient
        # mixing between shards in training.
        self.values_per_input_shard = 2500
        # Minimum number of shards to keep in the input queue.
        self.input_queue_capacity_factor = 2
        # Number of threads for prefetching SequenceExample protos.
        self.num_input_reader_threads = 1

        # Name of the SequenceExample context feature containing image data.
        self.image_feature_name = "image/data"
        # Name of the SequenceExample feature list containing integer captions.
        self.caption_feature_name = "image/caption_ids"

        # Number of unique words in the vocab (plus 1, for <UNK>).
        # The default value is larger than the expected actual vocab size to allow
        # for differences between tokenizer versions used in preprocessing. There is
        # no harm in using a value greater than the actual vocab size, but using a
        # value less than the actual vocab size will result in an error.
        self.vocab_size = 32200

        # Number of threads for image preprocessing. Should be a multiple of 2.
        self.num_preprocess_threads = 4

        # Batch size.
        self.batch_size = 256

        # File containing an Inception v3 checkpoint to initialize the variables
        # of the Inception model. Must be provided when starting training for the
        # first time.
        self.inception_checkpoint_file = None

        # Scale used to initialize model variables.
        self.initializer_scale = 0.08

        # LSTM input and output dimensionality, respectively.
        self.image_embedding_dim = 1024  # image embedding dimension
        self.embedding_size = 300  # word embedding dimension
        self.num_lstm_units = 512

        # If < 1.0, the dropout keep probability applied to LSTM variables.
        self.lstm_dropout_keep_prob = 0.7

        # hyper parameters for building train_op
        # self.mode = 'train'
        self.filter_sizes = [3, 4, 5]
        self.out_channels = 300

        self.wgan = False
        self.label_smoothing = 0.01
        self.feature_match = 'moment'
        self.diag = 0.1
        self.sigma_range = [20]
        self.dim_mmd = 32


class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""

    def __init__(self):
        """Sets the default training hyperparameters."""
        # Number of examples per epoch of training data.
        self.num_examples_per_epoch = 413915  # 586363

        # Optimizer for training the model.
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/optimizers.py#L40
        self.optimizer = 'RMSProp'  # 'Adam', 'Adagrad', 'SGD', "Momentum"

        # Learning rate for the initial phase of training.
        self.initial_learning_rate_g = 0.00005

        self.initial_learning_rate_d = 0.0001
        self.learning_rate_decay_factor = 0.5
        self.num_epochs_per_decay = 8.0

        self.weight_decay_rate = 0.5
        self.relu_leakiness = 0.1

        # Learning rate when fine tuning the Inception v3 parameters.
        # self.train_inception_learning_rate = 0.0005

        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 10

        #
        self.dispFreq = 10
        self.dg_ratio = 5
        self.saveFreq = 500

        # not use
        self.train_inception_learning_rate = 0.0
        self.debug = True
