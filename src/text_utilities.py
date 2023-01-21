#!/usr/bin/env python
__author__ = "Ricardo Jung"
__email__ = "s2458588@stud.uni-frankfurt.de"

# __copyright__ = ""
# __credits__ = [""]
# __license__ = ""
# __version__ = ""
# __maintainer__ = ""
# __status__ = ""

import regex as re
import glob
import numpy as np
import collections as cl


def corpus_metrics(tokenset):
    all_chars = ""
    for sylnumber in tokenset:
        for v in tokenset[sylnumber]:
            all_chars += v

    n_maps = cl.Counter(list(all_chars)).most_common()
    char_array = np.array(n_maps)

    n_count = [i[1] for i in n_maps]
    total_count = sum(n_count)
    rel_freqs = [i / total_count for i in n_count]

    # ARRAY WITH CHARS, COUNTS and REL. FREQ
    full_array = np.column_stack((char_array, rel_freqs))
    return {j: (int(k), float(l)) for (j, k, l) in full_array}


class PosCorpus:
    """Corpus object generated from files in a folder"""

    def __init__(self, path):
        self.path = path
        self.types = from_path(path)
        self.counted_corpus = count_syllables(self.types)  # corpus sorted tokens into approx. syllable counts
        self.metrics = corpus_metrics(self.counted_corpus)  # contains character metrics; count, relative frequency


def from_path(path: str) -> set:
    """Returns a set of vocabulary from txt files in a directory (use wd)"""
    files = glob.glob(f'{path}/*.txt')
    words = []
    for file in files:
        with open(file, 'r', encoding='utf8') as f:
            words += f.read().split('\n')
    try:
        words.remove('')
    except ValueError:
        pass
    return set(words)


def count_syllables(text, pattern='[aeuioäöüAEIUOÄÖÜ][aeuioäöüAEIUOÄÖÜ]?'):
    """Counts chained vowels (max 2) from iterable word list and sorts them into a dict {vowelcount:{set of tokens}}"""

    if isinstance(text, set):
        count_dict = dict()
        for v in text:
            vowels = re.findall(pattern=pattern, string=v)
            if len(vowels) in count_dict:
                count_dict[len(vowels)].add(v)
            else:
                count_dict.update({len(vowels): {v}})
        try:
            del count_dict[0]
        except KeyError:
            pass
        return count_dict
    else:
        try:
            return len(re.findall(pattern=pattern, string=text))
        except TypeError:
            print(Exception)


def decode(target: str, m: str) -> (str, str):
    """Returns a subword from a target string and a map. Yet to implement maps with 1 on both ends and returning the
    remaining string """
    if m.startswith("1"):
        return target[:m.count("1")] + "##", target[-m.count("0"):]
    elif m.endswith("1"):
        return target[-m.count("0"):], "##" + target[-m.count("1"):]


def match_ends(s1: str, s2: str):
    """Compares characters of two strings at index and returns a case."""
    cases = {
        "first": s1[0] == s2[0],
        "last": s1[-1] == s2[-1],
        "any": (s1[0] == s2[0] or s1[-1] == s2[-1])
    }
    return cases


def wordmap(longer, shorter, start=0):
    """Compares every character for a pair of strings. Takes start index as optional argument. Returns wordmap"""
    return [int(c1 == c2) for c1, c2 in zip(list(longer)[start::], list(shorter))]

def derive_wordmap(wordmap):
    """Takes a wordmap from MapToken() and finds the lexeme with the wordmap's derivative."""
    x = np.ndarray(
        (len(wordmap),),
        dtype=int,
        buffer=np.array(mt1.wordmap)
    )
    dx = x[1] - x[0]
    f = np.sin(x)
    d_dx = FinDiff(0, dx)
    df_dx = d_dx(f)

    return df_dx