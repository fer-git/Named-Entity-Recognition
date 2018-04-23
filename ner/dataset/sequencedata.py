# -*- coding: utf-8 -*-
from typing import List

import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from dataset.vocabbuilder import IndexedSentence


def _feature_int64(value):
    if isinstance(value, int):
        return _feature_int64([value])
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _feature_list(feature):
    return tf.train.FeatureList(feature=[feature])


class SequenceData(object):
    """
    Convert indexed sentences into tfrecords
    """

    def __init__(self):
        pass

    @staticmethod
    def _compression_opt(compression: str) -> tf.python_io.TFRecordOptions:
        if compression:
            compression = compression.lower()
        assert compression in [None, 'gzip', 'zlib'], \
            f'{compression} should be either None, gzip, or zlib'
        if compression == 'gzip':
            compression = tf.python_io.TFRecordCompressionType.GZIP
        elif compression == 'zlib':
            compression = tf.python_io.TFRecordCompressionType.ZLIB
        else:
            compression = tf.python_io.TFRecordCompressionType.NONE
        opt = tf.python_io.TFRecordOptions(compression)
        return opt

    @staticmethod
    def _len_element(features, label) -> tf.int32:
        return tf.size(features['words'])

    @staticmethod
    def _compression_type(compression: str) -> tf.string:
        if compression:
            compression = compression.upper()
        assert compression in [None, 'GZIP', 'ZLIB'], \
            f'{compression} should be either None, gzip, or zlib'
        return tf.convert_to_tensor(compression, tf.string)

    def write_data(self, index_sentences: List[IndexedSentence],
                   file_path: str, pad_idx: int = 0, compression: str = 'gzip'):
        """
        Convert then write indexed sentences into tfrecords
        Parameters
        ----------
        index_sentences: List of indexed sentences
        file_path: Tfrecord file location
        pad_idx: Padding index
        compression: Compression type, None, 'gzip', or 'zlib'

        Returns
        -------
        None
        """
        write_option = self._compression_opt(compression)
        writer = tf.python_io.TFRecordWriter(file_path, options=write_option)
        for idx_sentence in index_sentences:
            example = self.make_one_example(words=idx_sentence.words_idxes,
                                            chars=idx_sentence.chars_idxes,
                                            pos_tags=idx_sentence.pos_tags_idxes,
                                            named_entities=idx_sentence.named_entities_idxes,
                                            pad_idx=pad_idx)
            writer.write(example.SerializeToString())
        writer.close()

    def read_data(self, file_path: str, label: str,
                  bucket_boundaries: List[int], bucket_batch_sizes: List[int],
                  compression: str = 'gzip'):
        """
        Read tfrecords file. Bucket the data based on word length
        Parameters
        ----------
        file_path: Tfrecord file location
        label: Labels to be loaded, 'named_entities', 'pos_tags', or 'both'
        bucket_boundaries: List of word length boundaries
        bucket_batch_sizes: Batch size for each bucket
        compression: Compression type, 'gzip', 'zlib', or None

        Returns
        -------
        Tensorflow dataset
        """
        compression_type = self._compression_type(compression)
        datarecord = tf.data.TFRecordDataset(filenames=file_path,
                                             compression_type=compression_type)
        datarecord = datarecord.map(lambda ex: self.load_one_example(ex,
                                                                     label))
        datarecord = datarecord.apply(
            tf.contrib.data.bucket_by_sequence_length(
                element_length_func=self._len_element,
                bucket_boundaries=bucket_boundaries,
                bucket_batch_sizes=bucket_batch_sizes))
        return datarecord

    @staticmethod
    def load_one_example(serialized_example, label='named_entities'):
        """
        Parse a sequence example from dataset
        Parameters
        ----------
        serialized_example: A sequence example from dataset
        label: Labels to be loaded, 'named_entities', 'pos_tags', or 'both'

        Returns
        -------
        Tuple of dictionaries. First dictionary contains features of
        words, chars, word_length, and char_lengths. Second dictionary contains
        labels based on input
        """
        label = label.lower()
        assert label in ['named_entities', 'pos_tags', 'both'], \
            f'{label} should be either named_entities, pos_tags, or both'
        # Parse example
        sequence_features = {'num_chars': tf.VarLenFeature(tf.int64),
                             'words': tf.VarLenFeature(tf.int64),
                             'chars': tf.VarLenFeature(tf.int64)}
        if label == 'named_entities' or label == 'both':
            sequence_features['named_entities'] = tf.VarLenFeature(tf.int64)
        elif label == 'pos_tags' or label == 'both':
            sequence_features['pos_tags'] = tf.VarLenFeature(tf.int64)

        context, feature_lists = tf.parse_single_sequence_example(
            serialized_example,
            context_features={'num_words': tf.FixedLenFeature([], tf.int64),
                              'max_char_length': tf.FixedLenFeature([],
                                                                    tf.int64)},
            sequence_features=sequence_features
        )
        # Convert words and chars into dense tensor, reshape char into 2D
        dense_words = tf.squeeze(
            tf.sparse_tensor_to_dense(feature_lists['words']), axis=0)
        dense_chars = tf.squeeze(
            tf.sparse_tensor_to_dense(feature_lists['chars']), axis=0)
        reshaped_char = tf.reshape(dense_chars,
                                   [tf.cast(context['num_words'], tf.int32),
                                    tf.cast(context['max_char_length'],
                                            tf.int32)])

        dense_num_chars = tf.squeeze(
            tf.sparse_tensor_to_dense(feature_lists['num_chars']), axis=0)

        # Determine which labels to be included
        labels = dict()
        if label == 'named_entities' or label == 'both':
            dense_named_entities = tf.squeeze(
                tf.sparse_tensor_to_dense(feature_lists['named_entities']),
                axis=0)
            labels['named_entities'] = dense_named_entities
        elif label == 'pos_tags' or label == 'both':
            dense_pos_tags = tf.squeeze(
                tf.sparse_tensor_to_dense(feature_lists['pos_tags']), axis=0)
            labels['pos_tags'] = dense_pos_tags

        return {'words': dense_words,
                'chars': reshaped_char,
                'char_lengths': dense_num_chars,
                'word_length': context['num_words']}, labels

    @staticmethod
    def make_one_example(words: List[int], chars: List[List[int]],
                         pos_tags: List[int], named_entities: List[int],
                         pad_idx: int = 0):
        """
        Create an example from an indexed sentence
        Parameters
        ----------
        words: List of indexed word
        chars: List of List of indexed characters
        pos_tags: List of indexed pos tags
        named_entities: List of indexed named entities
        pad_idx: Pad index to ensure chars is a tensor

        Returns
        -------
        A sequence example
        """
        word_feature = _feature_int64(words)
        num_words = len(words)
        # num_words_feature = self._feature_int64([num_words])
        num_words_feature = _feature_int64(num_words)

        # Find the length of characters on each word
        char_lengths = [len(ch) for ch in chars]
        char_lengths_feature = _feature_int64(char_lengths)
        # Pad char to ensure the shape is 2D, num_words x max of char_lengths
        padded_char = pad_sequences(chars, dtype='int64', padding='post',
                                    value=pad_idx)
        # Get the max of char_lengths after padding
        max_char_length = len(padded_char[0])
        max_char_length_feature = _feature_int64([max_char_length])

        # Flatten to ensure the shape is 1D
        flattened_char = [item for sublist in padded_char for item in sublist]
        char_feature = _feature_int64(flattened_char)

        pos_tags_label = _feature_int64(pos_tags)
        named_entities_label = _feature_int64(named_entities)

        ex = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'num_words': num_words_feature,
                'max_char_length': max_char_length_feature}),
            feature_lists=tf.train.FeatureLists(feature_list={
                'num_chars': _feature_list(char_lengths_feature),
                'words': _feature_list(word_feature),
                'chars': _feature_list(char_feature),
                'pos_tags': _feature_list(pos_tags_label),
                'named_entities': _feature_list(named_entities_label)
            })
        )
        return ex
