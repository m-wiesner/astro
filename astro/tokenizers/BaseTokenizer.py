from lhotse import CutSet
import pickle
import bz2


class BaseTokenizer(object):
    def __call__(self, cuts: CutSet):
        raise NotImplementedError

    def tokens_to_syms(self, tokens):
        raise NotImplementedError

    def serialize(self, path):
        with bz2.BZ2File(path, 'w') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with bz2.BZ2File(path, 'r') as f:
            return pickle.load(f)
