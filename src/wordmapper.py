#!/usr/bin/env python
__author__ = "Ricardo Jung"
__email__ = "s2458588@stud.uni-frankfurt.de"

# __copyright__ = ""
# __credits__ = [""]
# __license__ = ""
# __version__ = ""
# __maintainer__ = ""
# __status__ = ""

import text_utilities as tu
import regex as rex
from scipy import stats

pc = tu.PosCorpus('../data/experiment/verbs')
pc.counted_corpus.keys()


class WordMapper:
    """Generates a Wordmap for a target token, comparing it to its POS-members in a dict sorted by syllables"""

    def __init__(self, target: str, tokenset: dict, syllable_threshold=2, clean=True, pattern='([^^1][0*1*]+[^$1])'):
        self.target = target
        self.tokenset = tokenset
        self.syllables = tu.count_syllables(target)
        self.st = syllable_threshold  # when to begin analyzing the left side of tokens
        self.maps = self.stack_maps(target, tokenset, self.syllables)
        self.pattern = pattern
        self.clean_maps = None
        if clean:
            self.filter_map_noise(self.maps, self.pattern)
            self.wordmap = self.sum_map_stack(self.clean_maps)
        else:
            self.wordmap = self.sum_map_stack(self.maps)

    def stack_maps(self, target, tokenset, syllables):
        lt = len(target)
        # cc1, cc2, cc3 = 0,0,0
        maps = []
        for k in tokenset:
            for v in tokenset[k]:
                pair = (v, self.target)
                case = tu.match_ends(v, target)
                shorter = min(pair, key=len)
                longer = max(pair, key=len)
                diff = len(longer) - len(shorter)

                if case.get("any"):
                    if diff:
                        if case.get("first") and syllables != self.st:
                            # cc1+=1
                            wm = tu.wordmap(longer=longer, shorter=shorter)
                            while len(wm) < lt:
                                wm.append(0)  # padding
                            maps.append(wm)

                        if case.get("last"):
                            wm = []
                            # cc2+=1
                            wm = tu.wordmap(longer=longer, shorter=shorter, start=diff)
                            while len(wm) < lt:
                                wm.insert(0, 0)  # padding
                            maps.append(wm)

                    else:
                        # cc3+=1
                        wm = tu.wordmap(longer=pair[0], shorter=pair[1])
                        maps.append(wm)
        # print("Cases:", cc1, cc2, cc3)
        return maps

    def filter_map_noise(self, maps, pattern):
        """Convert maps to strings and delete any consecutive '1' not at the start or end of the map"""
        str_maps = ["".join([str(c) for c in m]) for m in maps]  # cast to str
        recount_map = [rex.sub(pattern=pattern, repl=lambda m: len(m.group(1)) * "0", string=sm) for sm in
                       str_maps]  # regex sub
        rex_str = [list(i) for i in recount_map]  # into list form
        str2int = [[int(c) for c in m] for m in rex_str]  # cast back to int
        self.clean_maps = str2int

    def sum_map_stack(self, maps):
        return [sum(x) for x in zip(*maps)]


class MapToken:
    """Holds information about a single token. metrics must be text_utilities.PosCorpus metrics dict"""

    def __init__(self, token: str, wordmap: list):
        self.wordmap = wordmap
        # self.metrics = {c: metrics[c] for c in metrics if c in token}
        self.token = token
        self.zscores = stats.zscore(wordmap)
        self.bwm = [1 if z < 0 else 0 for z in self.zscores]  # boolean word map
        self.__morphemes = self.zip_wordmap(token, self.bwm)
        self.stem = self.__morphemes[0]
        self.affix = [x for x in self.__morphemes[1] if x != self.stem]

    def zip_wordmap(self, token: str, bwm: list):
        "t = target string, wm = boolean wordmap."
        stem = "".join([c for b, c in zip(self.bwm, token) if b])
        return stem, token.partition(stem)
