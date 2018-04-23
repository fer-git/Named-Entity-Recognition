# -*- coding: utf-8 -*-
import os

import tensorflow as tf

from ner.dataset.pretrainedwords import PretrainedWords
from ner.dataset.sentencepruner import SentencePruner
from ner.dataset.sequencedata import SequenceData
from ner.dataset.vocabbuilder import VocabBuilder


def test_entire_pipeline_on_sample(sample_ontonotes, pretrained_folder,
                                   tmpdir):
    pruned_samples = SentencePruner().transform(sample_ontonotes)
    vocab = VocabBuilder()
    vocab.fit(pruned_samples)
    idxed_sentences = vocab.transform(pruned_samples)
    pretrained = PretrainedWords(folder=pretrained_folder,
                                 dim=50,
                                 token2idx=vocab.word_vocab.token2idx)
    sequence_data = SequenceData()
    base_path = tmpdir.mkdir('data')
    sample_path = str(base_path.join('sample.tfrecord'))
    word_path = str(base_path.join('word_vocab.txt'))
    char_path = str(base_path.join('char_vocab.txt'))
    pos_path = str(base_path.join('pos_vocab.txt'))
    ne_path = str(base_path.join('ne_vocab.txt'))
    vocab.word_vocab.save(word_path)
    vocab.char_vocab.save(char_path)
    vocab.pos_tag_vocab.save(pos_path)
    vocab.ne_vocab.save(ne_path)

    assert os.path.exists(word_path)
    assert os.path.exists(char_path)
    assert os.path.exists(pos_path)
    assert os.path.exists(ne_path)

    sequence_data.write_data(idxed_sentences, sample_path)
    assert pretrained.embedding_matrix.shape == (len(pretrained.words), 50)
    record = sequence_data.read_data(sample_path,
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
