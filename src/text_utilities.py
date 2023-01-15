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



