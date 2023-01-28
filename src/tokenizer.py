#!/usr/bin/env python
__author__ = "Ricardo Jung"
__email__ = "s2458588@stud.uni-frankfurt.de"


# __copyright__ = ""
# __credits__ = [""]
# __license__ = ""
# __version__ = ""
# __maintainer__ = ""
# __status__ = ""


class SequenceTokenizer:
    """Generate a series of possible segmentations from a given vocabulary for a given target token."""
    def __init__(self, vocab, target):
        self.vocab = vocab
        self.segmentations = None
        self.subvocab = self.gen_subvocab(target)

    def segmenter(self, target, segments=None, start=0, stop=int):
        if segments is None:
            segments = []
        if start == stop:
            self.segmentations.append(segments)
        else:
            new_morpheme = [i for i in self.subvocab if target.startswith(i)]
            for m in new_morpheme:
                rest = target[len(m):]
                start += len(m)
                segments.append(m)
                self.segmenter(target=rest, segments=segments.copy(), start=start, stop=stop)
                segments = segments[:-1]
                start -= len(m)

    def gen_subvocab(self, target, vocab):
        return [i for i in self.vocab if i in target]
