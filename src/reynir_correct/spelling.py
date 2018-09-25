"""

    Reynir: Natural language processing for Icelandic

    Spelling correction module

    Copyright (C) 2018 Miðeind ehf.

       This program is free software: you can redistribute it and/or modify
       it under the terms of the GNU General Public License as published by
       the Free Software Foundation, either version 3 of the License, or
       (at your option) any later version.
       This program is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
       GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see http://www.gnu.org/licenses/.


    This module uses word frequency information extracted from the
    Reynir (greynir.is) database as a basis for guessing the correct
    spelling of words not found in BÍN and not recognized by the
    compound word algorithm.

"""

import os
import math
import re
import pickle
import time

from collections import defaultdict
from functools import lru_cache

from reynir import tokenize, correct_spaces, TOK

if __package__:
    from .settings import Settings
else:
    from settings import Settings


_PATH = os.path.dirname(__file__) or "."
_SPELLING_PICKLE = os.path.abspath(os.path.join(_PATH, "resources", "spelling.pickle"))

EDIT_0_FACTOR = math.log(1.0 / 1.0)
EDIT_S_FACTOR = math.log(1.0 / 8.0)
# Edit distance 1 is 25 times more unlikely than 0
EDIT_1_FACTOR = math.log(1.0 / 64.0)
# Edit distance 2 is 16 times more unlikely than 1
EDIT_2_FACTOR = math.log(1.0 / 2048.0)


class Dictionary:

    """ Container for a frequency dictionary of words """

    # Remove these words manually from the dictionary
    # since they occur often enough in the source text
    # to be taken seriously if not explicitly removed
    _REMOVE = [
        "ad",
        "tad",
        "tess",
        "thvi",
        "thad",
        "thess",
        "med",
        "eda",
        "tho",
        "ac",
        "ed",
        "da",
        "tm",
        "ted",
        "se",
        "de",
        "le",
        "sa",
        "ap",
        "am",
        "sg",
        "sr",
        "sþ",
        "afd",
        "av",
        "ba",
        "okkir",
        "tilhneygingu",
        "ff",
        "matarlist",
        "þu",
        "ut",
        "in",
        "min",
        "bin",
        "þin",
        "sb",
        "di",
        "fra",
        "þvi",
        "hb",
        "framundan",
    ] + [c for c in "abcdðeéfghijklmnopqrstuúvwxyýzþö"]

    def __init__(self):
        # All words in vocabulary, case significant
        self.words = set()
        self.bin_words = set()
        # Word count, case not significant
        self.counts = dict()
        # Total word count, case not significant
        self.total = 0
        # Logarithm of total word count
        self.log_total = 0.0
        # Storing probabilities for words already found
        self.probs = dict()
        self._load()

    def _load_pickle(self):
        """ Load the dictionary from a pickle file, if it exists """
        try:
            with open(_SPELLING_PICKLE, "rb") as f:
                self.counts = pickle.load(f)
                self.words = pickle.load(f)
        except (FileNotFoundError, EOFError):
            return False
        return True

    def _load(self):
        """ Load the dictionary from a pickle """
        # self._load_bin()
        self._load_pickle()
        # Calculate a sum total of the counts
        self.total = sum(self.counts.values())
        self.log_total = math.log(self.total)

    def _load_bin(self):
        with open("./resources/bin_words.txt", "r") as f:
            for line in f:
                self.bin_words.add(line)

    def __getitem__(self, wrd):
        """ Return the count of the given word, assumed to be lowercase """
        return self.counts[wrd]

    def get(self, wrd):
        """ Return the count of the given word, assumed to be lowercase """
        return self.counts[wrd]

    def __contains__(self, wrd):
        """ Return True if the word occurs in the dictionary, assumed to be lowercase """
        return wrd in self.counts

    def __len__(self):
        """ Get the number of distinct words in the dictionary, case significant """
        return len(self.words)

    def freq(self, wrd):
        """ Get the frequency of the given word as a ratio between 0.0 and 1.0 """
        return self.counts[wrd] / self.total

    def log_freq(self, wrd):
        """ Get the logarithm of the frequency of the given word """
        return math.log(self.counts[wrd]) - self.log_total

    def freq_1(self, wrd):
        """ Get the frequency of the given word as a ratio between 0.0 and 1.0,
            but give out-of-vocabulary words a count of 1 instead of 0 """
        return self.counts.get(wrd, self.bin(wrd)) / self.total

    def log_freq_1(self, wrd):
        """ Get the logarithm of the frequency of the given word,
            however assigning out-of-vocabulary words a count of 1 instead of 0 """
        return math.log(self.counts.get(wrd, self.bin(wrd))) - self.log_total

    def percentile_log_freq(self, percentile):
        """ Return the frequency of the word at the n'th percentile,
            where n is 0..100, n=0 is the most frequent word,
            n=100 the least frequent """
        cts = sorted(self.counts.values(), reverse=True)
        cnt = cts[(len(cts) - 1) * percentile // 100]
        return math.log(cnt) - self.log_total

    def pdist(self):
        """ Returns a probability distribution function """
        return lambda wrd: self.freq_1(wrd)

    def pdist_log(self):
        """ Returns a log probability distribution function """
        return lambda wrd: self.log_freq_1(wrd)

    def bin(self, word):
        if word in self.bin_words:
            self.probs[word] = 2
            return 2
        else:
            self.probs[word] = 1
            return 1


@lru_cache(maxsize=2048)
def _splits(word):
    """ Return a list of all possible (first, rest) pairs that comprise word. """
    return [(word[:i], word[i:]) for i in range(len(word) + 1)]


class Corrector:

    """ A spelling corrector class using a word frequency dictionary """

    # The characters used to form variants of words by insertion
    _ALPHABET = "aábcdðeéfghiíjklmnoópqrstuúvwxyýzþæö"

    # Translate wrongly accented characters before correcting
    _TRANSLATE = {
        "à": "á",
        "è": "é",
        "ì": "í",
        "ò": "ó",
        "ô": "ó",  # Possibly ö
        "ù": "ú",
        "ø": "ö",
    }
    _TRANSLATE_REGEX = "(" + "|".join(_TRANSLATE.keys()) + ")"

    _SUBSTITUTE_LIST = [
        # keyboard distance
        # Note: single character substitutions are already carried
        # out in the edit distance algorithm, so they do not need
        # to be repeated here.
        # ("a", ["q", "w", "s", "z"]),
        # ("s", ["w", "e", "d", "x", "z", "a"]),
        # ("d", ["e", "r", "f", "c", "x", "s"]),
        # ("f", ["r", "t", "g", "v", "c", "d"]),
        # ("g", ["t", "y", "h", "b", "v", "f"]),
        # ("h", ["y", "u", "j", "n", "b", "g"]),
        # ("j", ["u", "i", "k", "m", "n", "h"]),
        # ("k", ["i", "o", "l", "m", "j"]),
        # ("l", ["o", "p", "æ", "k"]),
        # ("æ", ["p", "ð", "þ", "l"]),
        # ("q", ["w", "a"]),
        # ("w", ["e", "s", "a", "q"]),
        # ("e", ["r", "d", "s", "w"]),
        # ("r", ["t", "f", "d", "e"]),
        # ("t", ["y", "g", "f", "r"]),
        # ("y", ["u", "h", "g", "t"]),
        # ("u", ["i", "j", "h", "y"]),
        # ("i", ["o", "k", "j", "u"]),
        # ("o", ["p", "l", "k", "i"]),
        # ("p", ["ö", "ð", "æ", "l", "o"]),
        # ("ð", ["ö", "-", "æ", "p"]),
        # ("z", ["a", "s", "x"]),
        # ("x", ["z", "s", "d", "c"]),
        # ("c", ["x", "d", "f", "v"]),
        # ("v", ["c", "f", "g", "b"]),
        # ("b", ["v", "g", "h", "n"]),
        # ("n", ["b", "h", "j", "m"]),
        # ("m", ["n", "j", "k"]),
        # ("þ", ["æ"]),
        # n/nk
        ("áng", ["ang"]),
        ("eing", ["eng"]),
        ("eyng", ["eng"]),
        ("úng", ["ung"]),
        ("íng", ["yng", "ing"]),
        ("ýng", ["yng", "ing"]),
        ("aung", ["öng"]),
        ("ánk", ["ank"]),
        ("eink", ["enk"]),
        ("eynk", ["enk"]),
        ("únk", ["unk"]),
        ("ínk", ["ynk", "ink"]),
        ("ýnk", ["ynk", "ink"]),
        ("aunk", ["önk"]),
        # sníkjuhljóð
        ("dl", ["ll", "rl"]),
        ("dn", ["nn", "rn"]),
        ("rdl", ["rl"]),
        ("rdn", ["rn"]),
        ("sdl", ["sl"]),
        ("sdn", ["sn"]),
        # /j/ø
        ("ýa", ["ýja"]),
        ("ýu", ["ýu"]),
        ("æu", ["æju"]),
        ("ji", ["i", "gi"]),
        ("j", ["gj"]),
        ("ægi", ["agi"]),
        ("eigi", ["egi"]),
        ("eygi", ["egi"]),
        ("ígi", ["igi"]),
        ("ýgi", ["igi"]),
        ("oji", ["ogi"]),
        ("uji", ["ugi"]),
        ("yji", ["ygi"]),
        ("augi", ["ögi"]),
        # ,/ø, f/ø í innstöðu
        ("á", ["ág", "áf"]),
        ("í", ["íg"]),
        ("æ", ["æg"]),
        ("ú", ["úg", "úf"]),
        ("ó", ["óg", "óf"]),
        # einfaldir/tvöfaldir samhljóðar
        ("g", ["gg"]),
        ("gg", ["g"]),
        ("k", ["kk"]),
        ("kk", ["k"]),
        ("l", ["ll"]),
        ("ll", ["l"]),
        ("m", ["mm"]),
        ("mm", ["m"]),
        ("n", ["nn"]),
        ("nn", ["n"]),
        ("p", ["pp"]),
        ("pp", ["p"]),
        ("r", ["rr"]),
        ("rr", ["r"]),
        ("s", ["ss"]),
        ("ss", ["s"]),
        ("t", ["tt"]),
        ("tt", ["t"]),
        ("gð", ["ggð"]),
        ("ggð", ["gð"]),
        ("gt", ["ggt"]),
        ("ggt", ["gt"]),
        ("gl", ["ggl"]),
        ("ggl", ["gl"]),
        ("gn", ["ggn"]),
        ("ggn", ["gn"]),
        ("kn", ["kkn"]),
        ("kkn", ["kn"]),
        ("kl", ["kkl"]),
        ("kkl", ["kl"]),
        ("kt", ["kkt"]),
        ("kkt", ["kt"]),
        ("pl", ["ppl"]),
        ("ppl", ["pl"]),
        ("pn", ["ppn"]),
        ("ppn", ["pn"]),
        ("pt", ["ppt"]),
        ("ppt", ["pt"]),
        ("tl", ["ttl"]),
        ("ttl", ["tl"]),
        ("tn", ["ttn"]),
        ("ttn", ["tn"]),
        # sérhljóðar
        # ("a", ["á"]),
        # ("e", ["é"]),
        ("ei", ["ey"]),
        ("ey", ["ei"]),
        # ("i", ["í", "y"]),
        # ("o", ["ó", "ö"]),
        # ("u", ["ú"]),
        # ("y", ["i", "ý"]),
        ("je", ["é"]),
        ("æ", ["aí"]),  # Tæland → Taíland
        # zeta og tengdir samhljóðaklasar
        ("z", ["ds", "ðs", "ts"]),  # "s"
        ("zt", ["st"]),
        ("zl", ["sl"]),
        ("nzk", ["nsk"]),
        ("tzt", ["st"]),
        ("ttzt", ["st"]),
        # einföldun, samhljóðaklasar
        ("md", ["fnd"]),
        ("mt", ["fnd"]),
        ("bl", ["fl"]),
        ("bbl", ["fl"]),
        ("bn", ["fn"]),
        ("bbn", ["fn"]),
        ("lgd", ["gld"]),
        ("gld", ["lgd"]),
        ("lgt", ["glt"]),
        ("glt", ["lgt"]),
        ("ngd", ["gnd"]),
        ("gnd", ["ngd"]),
        ("ngt", ["gnt"]),
        ("gnt", ["ngt"]),
        ("lfd", ["fld"]),
        ("fld", ["lfd"]),
        ("lft", ["flt"]),
        ("flt", ["lft"]),
        ("sn", ["stn"]),
        ("rn", ["rfn"]),
        ("rð", ["rgð"]),
        ("rgð", ["rð"]),
        ("ft", ["pt", "ppt"]),
        ("pt", ["ft"]),
        ("ppt", ["ft"]),
        ("nd", ["rnd"]),
        ("st", ["rst"]),
        ("ksk", ["sk"]),
        # annað
        ("kv", ["hv"]),
        ("hv", ["kv"]),
        ("gs", ["x"]),
        ("ks", ["x"]),
        ("x", ["gs", "ks"]),
        # ("v", ["f"]),
        # ("b", ["p"]),
        # ("g", ["k"]),
        # ("d", ["t"]),
        # erlend lyklaborð
        ("ae", ["æ"]),
        # ("t", ["þ"]),
        ("th", ["þ"]),
        # ("d", ["ð"]),
        # ljóslestur
        # ("c", ["æ", "é"]),
    ]

    _SUBSTITUTE = defaultdict(set)

    for _key, _subs in _SUBSTITUTE_LIST:
        _SUBSTITUTE[_key].update(_subs)

    # Sort the substitution keys in descending order by length
    _SUBSTITUTE_KEYS = sorted(_SUBSTITUTE.keys(), key=lambda x: len(x), reverse=True)
    # Create a regex to extract word fragments ending with substitution keys
    _SUBSTITUTE_REGEX = re.compile("(.*?(" + "|".join(_SUBSTITUTE_KEYS) + "))")

    # Minimum probability of a candidate other than the original
    # word in order for it to be returned
    _MIN_LOG_PROBABILITY = math.log(3.65e-9)

    def __init__(self, dictionary=None):
        self.d = dictionary or Dictionary()
        # Function for probability of word
        self.p_word = self.d.pdist()
        # Any word above the 40th percentile is probably correct
        self.accept_threshold = self.d.percentile_log_freq(40)

    def test_subs(self, word):
        """ Return all potential substitutions. This is used for
            testing purposes only and will probably be deleted
            eventually. """
        fragments = re.findall(self._SUBSTITUTE_REGEX, word)
        end = 0
        num_combs = 1
        combs = []
        for frag, sub in fragments:
            end += len(frag)
            subs = list(self._SUBSTITUTE[sub])
            combs.append(subs)
            num_combs *= 1 + len(subs)
        suffix = word[end:]
        for counter in range(1, num_combs):
            combo = counter
            result = []
            for (frag, sub), subs in zip(fragments, combs):
                ix = combo % (len(subs) + 1)
                if ix == 0:
                    result.append(frag)
                else:
                    result.append(frag[: -len(sub)] + subs[ix - 1])
                combo //= len(subs) + 1
            result.append(suffix)
            yield "".join(result)

    @lru_cache(maxsize=4096)
    def _correct(self, word):
        """ Find the best spelling correction for this word.
            Credits for this elegant code are due to Peter Norvig,
            cf. http://nbviewer.jupyter.org/url/norvig.com/ipython/
            How%20to%20Do%20Things%20with%20Words.ipynb """

        # Note: word is assumed to be in lowercase

        alphabet = self._ALPHABET

        def known(words):
            """ Return a generator of words that are actually in the dictionary. """
            return (w for w in words if w in self.d or w in self.d.bin_words)

        def edits0(word):
            """ Return all strings that are zero edits away from word (i.e., just word itself). """
            return {word}

        def edits1(pairs):
            """ Return all strings that are one edit away from this word. """
            # Deletes
            result = {a + b[1:]                 for (a, b) in pairs if b}
            # Transposes
            result |= {a + b[1] + b[0] + b[2:]  for (a, b) in pairs if len(b) > 1}
            # Replaces
            result |= {a + c + b[1:]            for (a, b) in pairs for c in alphabet if b}
            # Inserts
            result |= {a + c + b                for (a, b) in pairs for c in alphabet}
            return result

        def edits2(pairs):
            """ Return all strings that are two edits away from this word. """

            def sub_edits1(word):
                pairs = _splits(word)
                return edits1(pairs)

            return {e2 for e1 in edits1(pairs) for e2 in sub_edits1(e1)}

        def subs(word):
            """ Return all potential substitutions """
            fragments = re.findall(self._SUBSTITUTE_REGEX, word)
            end = 0
            num_combs = 1
            combs = []
            for frag, sub in fragments:
                end += len(frag)
                subs = list(self._SUBSTITUTE[sub])
                combs.append(subs)
                num_combs *= 1 + len(subs)
            suffix = word[end:]
            for counter in range(1, num_combs):
                combo = counter
                result = []
                for (frag, sub), subs in zip(fragments, combs):
                    ix = combo % (len(subs) + 1)
                    if ix == 0:
                        result.append(frag)
                    else:
                        result.append(frag[: -len(sub)] + subs[ix - 1])
                    combo //= len(subs) + 1
                result.append(suffix)
                yield "".join(result)

        def gen_candidates(word):
            """ Generate candidates in order of generally decreasing likelihood """
            P = self.d.pdist_log()
            e0 = edits0(word)
            for c in known(e0):
                if c not in self.d.probs:
                    self.d.probs[c] = P(c)
                yield (c, self.d.probs[c] + EDIT_0_FACTOR)
            for c in known(subs(word)):
                if c not in self.d.probs:
                    self.d.probs[c] = P(c)
                yield (c, self.d.probs[c] + EDIT_S_FACTOR)
            pairs = _splits(word)
            e1 = edits1(pairs) - e0
            for c in known(e1):
                if c not in self.d.probs:
                    self.d.probs[c] = P(c)
                yield (c, self.d.probs[c] + EDIT_1_FACTOR)
            e2 = edits2(pairs) - e1 - e0
            for c in known(e2):
                if c not in self.d.probs:
                    self.d.probs[c] = P(c)
                yield (c, self.d.probs[c] + EDIT_2_FACTOR)

        candidates = []
        acceptable = 0
        for c, log_prob in gen_candidates(word):
            if log_prob > self.accept_threshold:
                if c == word:
                    # print(f"The original word {word} is above the threshold, returning it")
                    return word
                # This candidate is likely enough: stop iterating and return it
                # print(f"Candidate {c} has log_prob {log_prob:.3f} > threshold")
                acceptable += 1
            # Otherwise, add to candidate list
            candidates.append((c, log_prob))
            if acceptable >= 10:
                # We already have ten candidates above the threshold:
                # that's enough
                break
        if not candidates:
            # No candidates beside the word itself: return it
            # print(f"Candidate {word} is only candidate, returning it")
            return word
        # Return the highest probability candidate
        # for i, (c, log_prob) in enumerate(sorted(candidates, key=lambda t:t[1], reverse=True)[0:5]):
        # print(f"Candidate {i+1} for {word} is {c} with log_prob {log_prob:.3f}")
        m = max(candidates, key=lambda t: t[1])
        if m[1] < self._MIN_LOG_PROBABILITY:
            # Best candidate is very unlikely: return the original word
            # print(f"Best candidate {m[0]} is highly unlikely, returning original {word}")
            return word
        # Return the most likely word
        return m[0]

    @staticmethod
    def _case_of(text):
        """ Return the case-function appropriate for text: upper, lower, title, or just str. """
        return str.upper if text.isupper() else str.title if text.istitle() else str

    def _cast(self, word):
        """ Cast the word to lowercase and correct accents """
        return re.sub(
            self._TRANSLATE_REGEX,
            lambda match: self._TRANSLATE[match.group()],
            word.lower(),
        )

    def correct(self, word):
        return self._case_of(word)(self._correct(self._cast(word)))

    def __getitem__(self, word):
        return self.correct(word)

    def correct_text(self, text):
        """ Correct all the words within a text, returning the corrected text. """

        def correct_match(match):
            """ Spell-correct word in match, and preserve proper upper/lower/title case. """
            return self.correct(match.group())

        result = []
        for token in tokenize(text):
            if token.kind == TOK.WORD:
                result.append(self.correct(token.txt))
            elif token.txt:
                result.append(token.txt)
        return correct_spaces(" ".join(result))

        # The regex finds all Unicode alphabetic letter sequences
        # (not including digits or underscores)
        # return re.sub(r"[^\W\d_]+", correct_match, text)


def test():

    c = Corrector()

    txts = [
        """
        FF er flokkur með rasisku ívafi og tilhneygjingu til að einkavinavæða alla fjölmiðla
        Íslands og færa þar með elítunni að geta ein haft áhrif á skoðanamyndandi áhri í
        fjölmiðlaheiminum, er ekki viðbúið að svona flokkur gamgi til samstarf við íhaldið
        eftir kosningar en ekki þessa vondu félagshyggjuflokka
        """,
        """
        fæ alveg hræðileg drauma vegna fyrri áfalla og það hjálpar mér að ná góðum svef og þar með
        betri andlegri lýðan og líka til að auka matarlist. Tek samt skýrt fram að ég hef bæði
        missnotað kannabis og ekki. Hef engan áhuga á að vera undir áhrifum kannabis alla dag.
        Mikil munur á að nota og missnota !
        """,
        """
        Bæði , lyf gegn áfengissyki (leiða) , mér hefur ekki leiðst mikið seinustu 30 ár. Gegn
        Taugaveiklun, konan hamrar á mér alla daga , skærur hennar eru langar og strangar. En ef ég fæ
        eina pípu og gríp gitarinn má hún tuða í mér klukkutímum saman.Ég er bæði rólegur og læri hratt
        á gítarinn, eftir 10 ára hjónaband er ég bara ótrúlega heill og stefni hátt. Ég og gitarinn erum
        orðnir samvaxnir. Auðvitað stefnum við á skilnað og þá mun ég sakna skalaæfinganna.
        """,
        """
        biddu nu hæg - var Kvennalistinn eins malefnis hreyfing. Hvað attu við - ef þu telur malefnið
        hafa verið eitt hvert var það? Kannski leikskola fyrir öll börn? Sömu laun fyrir sömu störf?
        Að borgarskipulag tæki mið af þörfum beggja kynja? Að kynjagleraugu væru notuð við gerð
        fjarlaga? Að þjoðfelagið opnaði augun fyrir kynferðsofbeldinu og sifjaspellum? (hvorutveggja
        sagt aðeins viðgangast i utlöndum). Þetta eru aðeins örfa dæmi um malefni sem brunnu a okkur
        og við börðumst fyrir. Ekki ertu i alvöru að tala framlag okkur niður. Tæplega
        telurðu það EITT malefni þo að i grunninn hafi baratta okkar sem stoðum að Kvennaframboðinu
        og -listanum gengið ut a að ,,betri,, helmingur þjoðarinnar öðlast - ekki bara i orði heldur
        einnig a borði - sömu rettindi og raðandi helmingurinn
        """,
        """
        Salvör ekki standa i að reyna að klora yfir mistök þin. Reynsluheimur kvenna visar að sjalsögðu
        til þess að helmingur mannkynsins - -konur - er olikur hinum helmingnum bæði sökum lffræðilegs munar og
        þess að þær eru gerðar að konum (sb de Beauvoir) þe fra frumbernsku er drengjum hrosað fyrir annað en
        stulkum og væntingar foreldra eru aðrar til dætra en sona og auk þess er ætlast til að dætur læri af mæðrum en synir af
        feðrum. Það er þetta sem gerir konur - helming mannkynsins - frabrugðna körlum sem hafa fra örofi alda verið
        ,,raðandi,, kynið. Það var gegn þvi orettlæti að reynsluheimur kvenna speglaðist ekki i politiskum akvörðunum sem við
        sem stofnaði Kvennafranboðið og - listann börðumst gegn - a öllum vigstöðvum. Að skilgreina barattu okkar
        Kvennalistans - fyrir rettindum halfrar þjoðarinnar til að skapa ,,rettlatara samfelag,, - sem eins mals flokk er
        fjarstæða.
        """,
    ]

    def linebreak(txt, margin=80, left_margin=0):
        """ Return a nicely column-formatted string representation of the given text,
            where each line is not longer than the given margin (if possible).
            A left margin can be optionally added, as a sequence of spaces.
            The lines are joined by newlines ('\n') but there is no trailing
            newline. """
        result = []
        line = []
        len_line = 0
        for wrd in txt.split():
            if len_line + 1 + len(wrd) > margin:
                result.append(" ".join(line))
                line = []
                len_line = 0
            line.append(wrd)
            len_line += 1 + len(wrd)
        if line:
            result.append(" ".join(line))
        return "\n".join(" " * left_margin + line for line in result)

    t0 = time.time()

    for t in txts:
        print("\nOriginal:\n")
        print(t)
        print("\nCorrected:\n")
        print(linebreak(c.correct_text(t), left_margin=8))

    t1 = time.time()
    print("Total time: {0:.2f} seconds".format(t1 - t0))


if __name__ == "__main__":

    test()
