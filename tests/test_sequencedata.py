# -*- coding: utf-8 -*-
import os

import tensorflow as tf

from ner.dataset.sequencedata import SequenceData


def test_datawriter_can_write(sample_idxed_sentences, tmpdir):
    data_writer = SequenceData()
    save_path = str(tmpdir.mkdir('data').join('sample.tfrecord'))
    data_writer.write_data(sample_idxed_sentences, save_path, 0, 'gzip')
    assert os.path.exists(save_path)


def test_datawriter_can_read(sample_idxed_sentences, tmpdir):
    data_writer = SequenceData()
    save_path = str(tmpdir.mkdir('data').join('sample.tfrecord'))
    data_writer.write_data(sample_idxed_sentences, save_path, 0, 'gzip')
    record = data_writer.read_data(save_path,
                                   label='named_entities',
                                   bucket_boundaries=[4, 7],
                                   bucket_batch_sizes=[3, 3, 3],
                                   compression='gzip')
    data_iter = record.make_one_shot_iterator()
    sess = tf.InteractiveSession()
    features, labels = sess.run(data_iter.get_next())
    assert features['words'].ndim == 2
    assert features['chars'].ndim == 3
    assert features['char_lengths'].ndim == 2
    assert features['word_length'].ndim == 1
    assert labels['named_entities'].ndim == 2
    sess.close()
