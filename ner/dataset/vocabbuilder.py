# -*- coding: utf-8 -*-
from typing import NamedTuple, List

from dataset.sentencepruner import Sentence


class Vocabulary(object):
    """
    Build vocabulary mapping from token to index
    """

    def __init__(self, pad_token: str = '<pad>', unk_token: str = '<unk>'):
        """
        Initialize vocabulary object

        Pad token always exists, while unknown token is optional
        Parameters
        ----------
        pad_token Pad token to be added as special token
        unk_token Unknown token to be added as special token
        """
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.token2idx = {pad_token: 0}
        if unk_token:
            self.token2idx[unk_token] = 1

    def __len__(self):
        return len(self.token2idx)

    def __getitem__(self, token):
        return self.token2idx[token]

    def __repr__(self):
        return f'Vocabulary contains {len(self)} tokens'

    def add_token(self, token: str):
        """
        Add new token to current vocabulary if it does not exist yet
        Parameters
        ----------
        token: New token

        Returns
        -------
        None
        """
        if token not in self.token2idx:
            self.token2idx[token] = len(self.token2idx)

    def add_tokens(self, tokens):
        """
        Add list of tokens to vocabulary
        Parameters
        ----------
        tokens: List of new token

        Returns
        -------
        None
        """
        for token in tokens:
            self.add_token(token)

    @property
    def idx2token(self):
        """
        Reverse the mapping of word and index
        Returns
        -------
        List where each index mapped to a word according to token2idx
        """
        return list(self.token2idx.values())

    def save(self, file_path):
        """
        Store vocabulary as a text file, one token per line, ordered by index
        Parameters
        ----------
        file_path: Vocabulary file location

        Returns
        -------
        None
        """
        with open(file_path, 'w') as f:
            for token in self.token2idx:
                f.write(token + '\n')

    def load(self, file_path):
        """
        Load vocabulary from a text file
        Parameters
        ----------
        file_path: Vocabulary file location

        Returns
        -------
        None
        """
        with open(file_path, 'r') as f:
            tokens = f.read().splitlines()
            self.token2idx = {token: index
                              for index, token in enumerate(tokens)}
            if self.pad_token:
                assert self.pad_token in self.token2idx, \
                    f'{self.pad_token} is missing'
            if self.unk_token:
                assert self.unk_token in self.token2idx, \
                    f'{self.unk_token} is missing'


class WordVocab(Vocabulary):
    def __init__(self, pad_token='<pad>', unk_token='<unk>'):
        super(WordVocab, self).__init__(pad_token, unk_token)


class CharVocab(Vocabulary):
    def __init__(self, pad_token='<pad>', unk_token='<unk>'):
        super(CharVocab, self).__init__(pad_token, unk_token)

    def add_tokens(self, tokens):
        if tokens and isinstance(tokens, list):
            tokens = [c for char in tokens for c in char]
        super().add_tokens(tokens)


class PosTagsVocab(Vocabulary):
    def __init__(self, pad_token='<pad>', unk_token=None):
        super(PosTagsVocab, self).__init__(pad_token, unk_token)


class NeVocab(Vocabulary):
    def __init__(self, pad_token='<pad>', unk_token=None):
        super(NeVocab, self).__init__(pad_token, unk_token)


class IndexedSentence(NamedTuple):
    words_idxes: List[int]
    chars_idxes: List[List[int]]
    pos_tags_idxes: List[int]
    named_entities_idxes: List[int]

    @property
    def num_words(self):
        return len(self.words_idxes)

    @property
    def num_chars(self):
        return len(self.chars_idxes)


class VocabBuilder(object):
    """
    Generate vocabularies and transform sentences
    """

    def __init__(self):
        self.word_vocab = WordVocab()
        self.char_vocab = CharVocab()
        self.pos_tag_vocab = PosTagsVocab()
        self.ne_vocab = NeVocab()

    def fit(self, sentences: List[Sentence]):
        """
        Build vocabularies for words, chars, pos tags, and named entities
        Parameters
        ----------
        sentences: List of sentence objects

        Returns
        -------
        None
        """
        for sentence in sentences:
            self.word_vocab.add_tokens(sentence.words)
            self.char_vocab.add_tokens(sentence.chars)
            self.pos_tag_vocab.add_tokens(sentence.pos_tags)
            self.ne_vocab.add_tokens(sentence.named_entities)

    def transform(self, sentences):
        """
        Convert words, chars, pos tags and named entities into indexes based
        on vocabulary
        Parameters
        ----------
        sentences: List of sentence objects

        Returns
        -------
        List of indexed sentence
        """
        results = []
        for sentence in sentences:
            words_idxes = [self.word_vocab[word] for word in sentence.words]
            chars_idxes = [[self.char_vocab[c] for c in char] for char in
                           sentence.chars]
            pos_tags_idxes = [self.pos_tag_vocab[pos_tag] for pos_tag in
                              sentence.pos_tags]
            ne_idxes = [self.ne_vocab[ne] for ne in sentence.named_entities]
            results.append(
                IndexedSentence(words_idxes, chars_idxes, pos_tags_idxes,
                                ne_idxes))
        return results
