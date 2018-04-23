# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict

import numpy as np


class PretrainedWords(object):
    def __init__(self, folder: str, dim: int, token2idx: Dict[str, int]):
        """
        Create PretrainedWords object

        Prune word that do not exist in token2idx. On the other hand,
        initialize token that is not in pretrained word vector with zero
        vector
        Parameters
        ----------
        folder: Pretrained word vector folder
        dim: Word vector dimension
        token2idx: Word vocabulary from fitting training sentences
        """
        vocab_size = len(token2idx)
        self.dim = dim
        word_vector_file = self._find_word_vector_file(folder, dim)
        self.embedding_matrix = np.zeros([vocab_size, dim], dtype=np.float32)
        with open(word_vector_file, 'r') as f:
            for value in f:
                splitted = value.split()
                word = splitted[0].strip()
                idx = token2idx.get(word, None)
                if idx:
                    word_vector = np.array(splitted[1:], np.float32)
                    self.embedding_matrix[idx] = word_vector
        self.words = self._sorted_words(token2idx)

    @staticmethod
    def _find_word_vector_file(folder, dim):
        folder_path = Path(folder)
        file = folder_path.glob(f'*{str(dim)}d*')
        return str(list(file)[0])

    @staticmethod
    def _sorted_words(token2idx):
        sorted_pairs = sorted(token2idx.items(), key=lambda pair: pair[1])
        return [word for word, _ in sorted_pairs]

    def save_embedding(self, file_path):
        np.save(file_path, self.embedding_matrix)

    def save_words(self, file_path):
        with open(file_path, 'w') as f:
            for word in self.words:
                f.write(word + '\n')
