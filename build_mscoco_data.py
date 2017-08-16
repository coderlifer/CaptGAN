
"""Converts MSCOCO data to TFRecord file format with SequenceExample protos.

The MSCOCO images are expected to reside in JPEG files located in the following
directory structure:

  train_image_dir/COCO_train2014_000000000151.jpg
  train_image_dir/COCO_train2014_000000000260.jpg
  ...

and

  val_image_dir/COCO_val2014_000000000042.jpg
  val_image_dir/COCO_val2014_000000000073.jpg
  ...

The MSCOCO annotations JSON files are expected to reside in train_captions_file
and val_captions_file respectively.

This script converts the combined MSCOCO data into sharded data files consisting
of 256, 4 and 8 TFRecord files, respectively:

  output_dir/train-00000-of-00256
  output_dir/train-00001-of-00256
  ...
  output_dir/train-00255-of-00256

and

  output_dir/val-00000-of-00004
  ...
  output_dir/val-00003-of-00004

and

  output_dir/test-00000-of-00008
  ...
  output_dir/test-00007-of-00008

Each TFRecord file contains ~2300 records. Each record within the TFRecord file
is a serialized SequenceExample proto consisting of precisely one image-caption
pair. Note that each image has multiple captions (usually 5) and therefore each
image is replicated multiple times in the TFRecord files.

The SequenceExample proto contains the following fields:

  context:
    image/image_id: integer MSCOCO image identifier
    image/data: string containing JPEG encoded image in RGB colorspace

  feature_lists:
    image/caption: list of strings containing the (tokenized) caption words
    image/caption_ids: list of integer ids corresponding to the caption words

The captions are tokenized using the NLTK (http://www.nltk.org/) word tokenizer.
The vocabulary of word identifiers is constructed from the sorted list (by
descending frequency) of word tokens in the training set. Only tokens appearing
at least 4 times are considered; all other words get the "unknown" word id.

NOTE: This script will consume around 100GB of disk space because each image
in the MSCOCO dataset is replicated ~5 times (once per caption) in the output.
This is done for two reasons:
  1. In order to better shuffle the training data.
  2. It makes it easier to perform asynchronous preprocessing of each image in
     TensorFlow.

Running this script using 16 threads may take around 1 hour on a HP Z420.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import namedtuple
from datetime import datetime
import os.path
import random
import sys
import threading

import nltk.tokenize
import numpy as np
import tensorflow as tf

tf.flags.DEFINE_string("train_image_embeddings_file",
                       "./input/train/train_imgs_embedding.txt",
                       "Training image embeddings directory.")
tf.flags.DEFINE_string("val_image_embeddings_file",
                       "./input/eval/val_imgs_embedding.txt",
                       "Validation image embeddings directory.")

tf.flags.DEFINE_string("train_captions_file",
                       "./input/train/coco_train_caps.txt",
                       "Training captions .txt file.")
tf.flags.DEFINE_string("val_captions_file",
                       "./input/eval/coco_dev_caps.txt",
                       "Validation captions .txt file.")

tf.flags.DEFINE_string("train_id_file",
                       "./input/train/coco_train.txt",
                       "Training id .txt file.")
tf.flags.DEFINE_string("val_id_file",
                       "./input/eval/coco_dev_ids.txt",
                       "Validation id .txt file.")

tf.flags.DEFINE_string("output_dir", "./input/tf_record/", "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 32,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 2,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 2,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_string("start_word", "<s>",  # <S>
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "<eos>",  # </S>
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer("min_word_count", 4,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file", "./tf_record/word_counts.txt",
                       "Output vocabulary file of word counts.")

tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "image_embedding", "captions"])


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, vocab, unk_id):
        """Initializes the vocabulary.

        Args:
          vocab: A dictionary of word to word_id.
          unk_id: Id of the special 'unknown' word.
        """
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        """Returns the integer id of a word string."""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _to_sequence_example(image, vocab):
    """Builds a SequenceExample proto for an image-caption pair.

    Args:
      image: An ImageMetadata object.
      vocab: A Vocabulary object.

    Returns:
      A SequenceExample proto.
    """

    context = tf.train.Features(feature={
        "image/image_id": _int64_feature(image.image_id),
        "image/data": _bytes_feature(image.image_embedding),
    })

    assert len(image.captions) == 1
    caption = image.captions[0]
    caption_ids = [vocab.word_to_id(word) for word in caption]
    feature_lists = tf.train.FeatureLists(feature_list={
        "image/caption": _bytes_feature_list(caption),
        "image/caption_ids": _int64_feature_list(caption_ids)
    })
    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    return sequence_example


def _process_image_files(thread_index, ranges, name, images, vocab, num_shards):
    """Processes and saves a subset of images as TFRecord files in one thread.

    Args:
      thread_index: Integer thread identifier within [0, len(ranges)].
      ranges: A list of pairs of integers specifying the ranges of the dataset to
        process in parallel.
      name: Unique identifier specifying the dataset.
      images: List of ImageMetadata.
      vocab: A Vocabulary object.
      num_shards: Integer number of shards for the output files.
    """
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in images_in_shard:
            image = images[i]

            sequence_example = _to_sequence_example(image, vocab)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_images_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        # shard_counter = 0
    print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def _process_dataset(name, images, vocab, num_shards):
    """Processes a complete data set and saves it as a TFRecord.

    Args:
      name: Unique identifier specifying the dataset.
      images: List of ImageMetadata.
      vocab: A Vocabulary object.
      num_shards: Integer number of shards for the output files.
    """
    # Break up each image into a separate entity for each caption.
    images = [ImageMetadata(image.image_id, image.image_embedding, [caption])
              for image in images for caption in image.captions]

    # Shuffle the ordering of images. Make the randomization repeatable.
    random.seed(12345)
    random.shuffle(images)

    # Break the images into num_threads batches. Batch i is defined as
    # images[ranges[i][0]:ranges[i][1]].
    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in xrange(len(ranges)):
        args = (thread_index, ranges, name, images, vocab, num_shards)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
          (datetime.now(), len(images), name))


def _create_vocab(captions):
    """Creates the vocabulary of word to word_id.

    The vocabulary is saved to disk in a text file of word counts. The id of each
    word in the file is its corresponding 0-based line number.

    Args:
      captions: A list of lists of strings.

      [['a', 'ball', 'cat'],
       ...,
       ['a', 'ball', 'cat']]

    Returns:
      A Vocabulary object.
    """
    print("Creating vocabulary.")
    counter = Counter()
    for c in captions:
        counter.update(c)
    print("Total words:", len(counter))

    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))

    # Write out the word counts file.
    with tf.gfile.FastGFile(FLAGS.word_counts_output_file, "w") as f:
        f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    print("Wrote vocabulary file:", FLAGS.word_counts_output_file)

    # Create the vocabulary dictionary.
    reverse_vocab = [x[0] for x in word_counts]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    vocab = Vocabulary(vocab_dict, unk_id)

    return vocab


def _load_vocab(vocab_file):
    """

    :param vocab_file:
    :return:
    """
    with tf.gfile.FastGFile(vocab_file, "r") as f:
        vocab = np.genfromtxt(f, dtype=None, comments=None)
        vocab = dict(vocab)

        unk_id = len(vocab)
        vocab = Vocabulary(vocab, unk_id)

        return vocab


def make_ids(img_name_file):
    coco_id = np.genfromtxt('/home/yang/Downloads/FILE/ml/GANCapt/input/coco_train.txt',
                            dtype=None,
                            comments=None)
    ids = []
    for x in coco_id:
        xs = x.split('_')[-1].split('.')[0]
        ids.append(int(xs))
    np.savetxt('/home/yang/Downloads/FILE/ml/GANCapt/input/coco_train_ids.txt', ids, fmt='%s')


def _process_caption(caption):
    """Processes a caption string into a list of tonenized words.

    Args:
      caption: A string caption.

    Returns:
      A list of strings; the tokenized caption.
    """
    tokenized_caption = [FLAGS.start_word]
    # tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
    tokenized_caption.extend(nltk.tokenize.word_tokenize(caption))
    tokenized_caption.append(FLAGS.end_word)
    return tokenized_caption


def _load_and_process_metadata(ids_file, image_embeddings_file, captions_file):
    """Loads image metadata from a JSON file and processes the captions.

    Args:
      ids_file: .txt file containing image_id.
      image_embeddings_file: .txt file containing the image embeddings.
      captions_file: .txt file containing caption annotations.

    Returns:
      A list of ImageMetadata.
    """
    with tf.gfile.FastGFile(ids_file, "r") as f:
        id_data = np.genfromtxt(f, dtype=np.int32)

    with tf.gfile.FastGFile(image_embeddings_file, "r") as f:
        image_embeddings_data = np.genfromtxt(f, dtype=np.float32, delimiter=' ')  #

    with tf.gfile.FastGFile(captions_file, "r") as f:
        content = f.readlines()
        caption_data = [x.strip() for x in content]

    # Extract the filenames.
    # id: 391895
    # file_name: COCO_val2014_000000391895.jpg
    # id_to_filename = [(x["id"], x["file_name"]) for x in caption_data["images"]]

    # Extract the captions. Each image_id is associated with multiple captions.
    id_to_captions = {}  # size = len(caption_data) / 5
    for i in xrange(len(caption_data)):
        image_id = id_data[i]  # image_id: 391895
        caption = caption_data[i]  # caption: 'a man with a red helmet ...'
        id_to_captions.setdefault(image_id, [])
        id_to_captions[image_id].append(caption)

    assert len(id_data) == len(id_to_captions) * 5
    assert set([x for x in id_data]) == set(id_to_captions.keys())
    print("Loaded caption metadata for %d images from %s" %
          (len(id_to_captions), captions_file))

    # Process the captions and combine the data into a list of ImageMetadata.
    print("Processing captions.")
    image_ids = id_to_captions.keys()
    image_metadata = []
    num_captions = 0
    for i in xrange(len(image_ids)):
        image_id = image_ids[i]
        captions = [_process_caption(c) for c in id_to_captions[image_id]]
        image_embedding = image_embeddings_data[i*5]
        image_metadata.append(ImageMetadata(image_id, image_embedding, captions))
        num_captions += len(captions)

    print("Finished processing %d captions for %d images in %s" %
          (num_captions, len(image_ids), captions_file))

    return image_metadata


def main(unused_argv):
    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with FLAGS.num_threads."""
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
    assert _is_valid_num_shards(FLAGS.val_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
    assert _is_valid_num_shards(FLAGS.test_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    # Load image metadata from caption files.
    # train_dataset = _load_and_process_metadata(FLAGS.train_id_file,
    #                                                   FLAGS.train_image_embeddings_file,
    #                                                   FLAGS.train_captions_file)

    val_dataset = _load_and_process_metadata(FLAGS.val_id_file,
                                             FLAGS.val_image_embeddings_file,
                                             FLAGS.val_captions_file)

    # test_dataset = _load_and_process_metadata(FLAGS.val_id_file,
    #                                           FLAGS.val_image_embeddings_file,
    #                                           FLAGS.val_captions_file)

    # Create vocabulary from the training captions.
    # train_captions = [c for image in train_dataset for c in image.captions]
    # vocab = _create_vocab(train_captions)
    vocab = _load_vocab('./input/word_to_id.txt')

    # _process_dataset("train", train_dataset, vocab, FLAGS.train_shards)
    _process_dataset("val", val_dataset, vocab, FLAGS.val_shards)
    # _process_dataset("test", test_dataset, vocab, FLAGS.test_shards)


if __name__ == "__main__":
    tf.app.run()
