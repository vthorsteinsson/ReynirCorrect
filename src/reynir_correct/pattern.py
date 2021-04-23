"""

    Greynir: Natural language processing for Icelandic

    Sentence tree pattern matching module

    Copyright (C) 2021 Miðeind ehf.

    This software is licensed under the MIT License:

        Permission is hereby granted, free of charge, to any person
        obtaining a copy of this software and associated documentation
        files (the "Software"), to deal in the Software without restriction,
        including without limitation the rights to use, copy, modify, merge,
        publish, distribute, sublicense, and/or sell copies of the Software,
        and to permit persons to whom the Software is furnished to do so,
        subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
        EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
        MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
        IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
        CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
        TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
        SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


    This module contains the PatternMatcher class, which implements
    functionality to look for questionable grammatical patterns in parse
    trees. These are typically not grammatical errors per se, but rather
    incorrect usage, e.g., attaching a wrong/uncommon preposition
    to a verb. The canonical example is "Ég leitaði af kettinum"
    ("I searched off the cat") which should very likely have been
    "Ég leitaði að kettinum" ("I searched for the cat"). The first form
    is grammatically correct but the meaning is borderline nonsensical.

"""

from typing import List, Tuple, Set, FrozenSet, Callable, Dict, Optional, Union, cast

from threading import Lock
from functools import partial
import os
import json

from reynir import Sentence, NounPhrase
from reynir.simpletree import SimpleTree
from reynir.verbframe import VerbErrors
from reynir.matcher import ContextDict

from .annotation import Annotation


# The types involved in pattern processing
AnnotationFunction = Callable[["PatternMatcher", SimpleTree], None]
PatternTuple = Tuple[
    Union[str, FrozenSet[str]], str, AnnotationFunction, Optional[ContextDict]
]


class IcelandicPlaces:

    """ Wraps a dictionary of Icelandic place names with their
        associated prepositions """

    # This is not strictly accurate as the correct prepositions
    # are based on convention, not rational rules. :/
    _SUFFIX2PREP = {
        "vík": "í",
        # "fjörður": "á",  # Skip this since 'í *firði' is also common
        "eyri": "á",
        "vogur": "í",
        "brekka": "í",
        "staðir": "á",
        # "höfn": "á",  # Skip this since 'í *höfn' is also common
        "eyjar": "í",
        "ey": "í",
        "nes": "á",
        "borg": "í",
    }
    _SUFFIXES = tuple(_SUFFIX2PREP.keys())

    ICELOC_PREP: Optional[Dict[str, str]] = None
    ICELOC_PREP_JSONPATH = os.path.join(
        os.path.dirname(__file__), "resources", "iceloc_prep.json"
    )

    @classmethod
    def _load_json(cls) -> None:
        """ Load the place name dictionary from a JSON file into memory """
        with open(cls.ICELOC_PREP_JSONPATH, encoding="utf-8") as f:
            cls.ICELOC_PREP = json.load(f)

    @classmethod
    def lookup_preposition(cls, place: str) -> Optional[str]:
        """ Look up the correct preposition to use with a placename,
            or None if the placename is not known """
        if cls.ICELOC_PREP is None:
            cls._load_json()
        assert cls.ICELOC_PREP is not None
        prep = cls.ICELOC_PREP.get(place)
        if prep is None and place.endswith(cls._SUFFIXES):
            # Not explicitly known: try to associate a preposition by suffix
            for suffix, prep in cls._SUFFIX2PREP.items():
                if place.endswith(suffix):
                    break
            else:
                # Should not get here
                assert False
                prep = None
        return prep

    @classmethod
    def includes(cls, place: str) -> bool:
        """ Return True if the given place is found in the dictionary """
        if cls.ICELOC_PREP is None:
            cls._load_json()
        assert cls.ICELOC_PREP is not None
        return place in cls.ICELOC_PREP  # pylint: disable=unsupported-membership-test


class PatternMatcher:

    """ Class to match parse trees with patterns to find probable usage errors """

    # The patterns to be matched are created when the
    # first class instance is initialized.

    # Each entry in the patterns list is a tuple of four values:

    # * Trigger lemma, which must be present in the sentence for the pattern
    #   to be applied. This is an optimization only, to save unnecessary matching.
    #   If the trigger is falsy (None, ""), it is not applied and the sentence will be
    #   checked regardless of its content.
    # * Match pattern expression, to be passed to match_pattern()
    # * Annotation function, called for each match
    # * Context dictionary to be passed to match_pattern()

    PATTERNS: List[PatternTuple] = []

    _LOCK = Lock()

    ctx_af: ContextDict = cast(ContextDict, None)
    ctx_að: ContextDict = cast(ContextDict, None)
    ctx_noun_af: ContextDict = cast(ContextDict, None)
    ctx_noun_af_obj: ContextDict = cast(ContextDict, None)
    ctx_verb_01: ContextDict = cast(ContextDict, None)
    ctx_verb_02: ContextDict = cast(ContextDict, None)
    ctx_noun_að: ContextDict = cast(ContextDict, None)
    ctx_place_names: ContextDict = cast(ContextDict, None)

    def __init__(self, ann: List[Annotation], sent: Sentence) -> None:
        # Annotation list
        self._ann = ann
        # The original sentence object
        self._sent = sent
        # Token list
        self._tokens = sent.tokens
        # Terminal node list
        self._terminal_nodes = sent.terminal_nodes
        # Avoid race conditions in multi-threaded scenarios
        with self._LOCK:
            if not self.PATTERNS:
                # First instance: create the class-wide pattern list
                self.create_patterns()

    def wrong_preposition_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verb phrase
        vp = match.first_match("VP > { %verb }", self.ctx_af)
        if vp is None:
            vp = match.first_match("VP >> { %verb }", self.ctx_af)
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        # Calculate the start and end token indices, spanning both phrases
        assert vp is not None
        assert pp is not None
        start, end = min(vp.span[0], pp.span[0]), max(vp.span[1], pp.span[1])
        text = "'{0} af' á sennilega að vera '{0} að'".format(vp.tidy_text)
        detail = (
            "Sögnin '{0}' tekur yfirleitt með sér "
            "forsetninguna 'að', ekki 'af'.".format(vp.tidy_text)
        )
        if match.tidy_text.count(" af ") == 1:
            # Only one way to substitute af -> að: do it
            suggest = match.tidy_text.replace(" af ", " að ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original="af",
                suggest=suggest,
            )
        )

    def wrong_preposition_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verb phrase
        vp = match.first_match("VP > { %verb }", self.ctx_að)
        if vp is None:
            vp = match.first_match("VP >> { %verb }", self.ctx_að)
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "að" }')
        # Calculate the start and end token indices, spanning both phrases
        assert vp is not None
        assert pp is not None
        start, end = min(vp.span[0], pp.span[0]), max(vp.span[1], pp.span[1])
        text = "'{0} að' á sennilega að vera '{0} af'".format(vp.tidy_text)
        detail = (
            "Sögnin '{0}' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'.".format(vp.tidy_text)
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def wrong_preposition_vitni_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending nominal phrase
        np = match.first_match(". > { 'vitni' }", self.ctx_af)
        if np is None:
            np = match.first_match(". >> { 'vitni' }", self.ctx_af)
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        if pp is None:
            pp = match.first_match('ADVP > { "af" }')
        assert np is not None
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(np.span[0], pp.span[0]), max(np.span[1], pp.span[1])
        text = "'verða vitni af' á sennilega að vera 'verða vitni að'"
        detail = (
            "Í samhenginu 'verða vitni að e-u' er notuð "
            "forsetningin 'að', ekki 'af'."
        )
        if match.tidy_text.count(" af ") == 1:
            # Only one way to substitute af -> að: do it
            suggest = match.tidy_text.replace(" af ", " að ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original="af",
                suggest=suggest,
            )
        )

    def wrong_preposition_grin_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal and nominal phrases
        vp = match.first_match("VP > { 'gera' }", self.ctx_af)
        np = match.first_match("NP > { 'grín' }", self.ctx_af)
        if np is None:
            np = match.first_match("NP >> { 'grín' }", self.ctx_af)
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        if pp is None:
            pp = match.first_match('ADVP > { "af" }')
        assert np is not None
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        if vp is None:
            start, end = min(np.span[0], pp.span[0]), max(np.span[1], pp.span[1])
        else:
            start, end = min(vp.span[0], np.span[0], pp.span[0]), max(vp.span[1], np.span[1], pp.span[1])
        text = "'gera grín af' á sennilega að vera 'gera grín að'"
        detail = (
            "Í samhenginu 'gera grín að e-u' er notuð "
            "forsetningin 'að', ekki 'af'."
        )
        if match.tidy_text.count(" af ") == 1:
            # Only one way to substitute af -> að: do it
            suggest = match.tidy_text.replace(" af ", " að ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original="af",
                suggest=suggest,
            )
        )

    def wrong_preposition_leida_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal and nominal phrases
        vp = match.first_match("VP > { 'leiða' }", self.ctx_af)
        np = match.first_match("NP > { ( 'líkur'|'rök'|'rak' ) }", self.ctx_af)
        if np is None:
            np = match.first_match("NP >> { ( 'líkur'|'rök'|'rak' ) }", self.ctx_af)
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        if pp is None:
            pp = match.first_match('ADVP > { "af" }')
        assert vp is not None
        assert np is not None
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(vp.span[0], np.span[0], pp.span[0]), max(vp.span[1], np.span[1], pp.span[1])
        text = "'leiða {0} af' á sennilega að vera 'leiða {0} að'".format(np.tidy_text)
        detail = (
            "Í samhenginu 'leiða {0} af e-u' er notuð "
            "forsetningin 'að', ekki 'af'.".format(np.tidy_text)
        )
        if match.tidy_text.count(" af ") == 1:
            # Only one way to substitute af -> að: do it
            suggest = match.tidy_text.replace(" af ", " að ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original="af",
                suggest=suggest,
            )
        )

    def wrong_preposition_marka_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal and nominal phrases
        vp = match.first_match("VP > { 'marka' }", self.ctx_af)
        if vp is None:
            vp = match.first_match("NP > { 'markaður' }", self.ctx_af)
        np = match.first_match("NP > { ( 'upphaf'|'upphafinn' ) }", self.ctx_af)
        if np is None:
            np = match.first_match("NP >> { ( 'upphaf'|'upphafinn' ) }", self.ctx_af)
        if np is None:
            np = match.first_match("VP > { 'upphefja' }", self.ctx_af)
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        if pp is None:
            pp = match.first_match('ADVP > { "af" }')
        assert vp is not None
        assert np is not None
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(vp.span[0], np.span[0], pp.span[0]), max(vp.span[1], np.span[1], pp.span[1])
        text = "'marka upphaf af' á sennilega að vera 'marka upphaf að'"
        detail = (
            "Í samhenginu 'marka upphaf að e-u' er notuð "
            "forsetningin 'að', ekki 'af'."
        )
        if match.tidy_text.count(" af ") == 1:
            # Only one way to substitute af -> að: do it
            suggest = match.tidy_text.replace(" af ", " að ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original="af",
                suggest=suggest,
            )
        )

    def wrong_preposition_leggja_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal phrase
        vp = match.first_match("VP > { 'leggja' }", self.ctx_af)
        if vp is None:
            vp = match.first_match("VP >> { 'leggja' }", self.ctx_af)
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        if pp is None:
            pp = match.first_match('ADVP > { "af" }')
        # Find the offending nominal phrase
        np = match.first_match("NP > { ( 'völlur'|'vell' ) }", self.ctx_af)
        if np is None:
            np = match.first_match("NP >> { ( 'völlur'|'vell' ) }", self.ctx_af)
        assert vp is not None
        assert pp is not None
        assert np is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(vp.span[0], pp.span[0], np.span[0]), max(vp.span[1], pp.span[1], np.span[1])
        text = "'leggja af velli' á sennilega að vera 'leggja að velli'"
        detail = (
            "Í samhenginu 'leggja einhvern að velli' er notuð "
            "forsetningin 'að', ekki 'af'."
        )
        if match.tidy_text.count(" af ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" af ", " að ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original="af",
                suggest=suggest,
            )
        )

    def wrong_preposition_utan_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending adverbial phrase
        advp = match.first_match("ADVP > { 'utan' }", self.ctx_af)
        if advp is None:
            advp = match.first_match("ADVP >> { 'utan' }", self.ctx_af)
        # Find the attached prepositional phrase
        pp = match.first_match('ADVP > { "af" }', self.ctx_af)
        assert advp is not None
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(advp.span[0], pp.span[0]), max(advp.span[1], pp.span[1])
        text = "'utan af' á sennilega að vera 'utan að'"
        detail = (
            "Í samhenginu 'kunna eitthvað utan að' er notuð "
            "forsetningin 'að', ekki 'af'."
        )
        if match.tidy_text.count(" af ") == 1:
            # Only one way to substitute af -> að: do it
            suggest = match.tidy_text.replace(" af ", " að ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original="af",
                suggest=suggest,
            )
        )

    def wrong_preposition_uppvis_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal phrase
        vp = match.first_match("VP > { 'verða' }", self.ctx_af)
        if vp is None:
            vp = match.first_match("VP >> { 'verða' }", self.ctx_af)
        # Find the attached nominal phrase
        np = match.first_match("NP > { 'uppvís' }", self.ctx_af)
        if np is None:
            np = match.first_match("NP >> { 'uppvís' }", self.ctx_af)
        # Find the attached prepositional phrase
        pp = match.first_match('PP > { "af" }', self.ctx_af)
        if pp is None:
            pp = match.first_match("ADVP > { 'af' }", self.ctx_af)
        assert vp is not None
        assert np is not None
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(vp.span[0], np.span[0], pp.span[0]), max(vp.span[1], np.span[1], pp.span[1])
        text = "'uppvís af' á sennilega að vera 'uppvís að'"
        detail = (
            "Í samhenginu 'verða uppvís að einhverju' er notuð "
            "forsetningin 'að', ekki 'af'."
        )
        if match.tidy_text.count(" af ") == 1:
            # Only one way to substitute af -> að: do it
            suggest = match.tidy_text.replace(" af ", " að ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,     
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original="af",
                suggest=suggest,
            )
        )

    def wrong_preposition_verða_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal phrase
        vp = match.first_match("VP > { 'verða' }", self.ctx_af)
        if vp is None:
            vp = match.first_match("VP >> { 'verða' }", self.ctx_af)
        # Find the attached prepositional phrase
        pp = match.first_match("P > 'af' ", self.ctx_af)
        if pp is None:
            pp = match.first_match("ADVP > 'af' ", self.ctx_af)
        # Find the attached nominal phrase
        np = match.first_match("NP > 'ósk' ", self.ctx_af)
        if np is None:
            np = match.first_match("NP >> 'ósk' ", self.ctx_af)
        assert vp is not None
        assert pp is not None
        assert np is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(vp.span[0], pp.span[0], np.span[0]), max(pp.span[1], pp.span[1], np.span[1])
        text = "'af ósk' á sennilega að vera 'að ósk'"
        detail = (
            "Í samhenginu 'að verða að ósk' er notuð " "forsetningin 'að', ekki 'af'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" af ", " að ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original="af",
                suggest=suggest,
            )
        )

    def wrong_preposition_ahyggja_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Calculate the start and end token indices, spanning both phrases
        start, end = match.span
        text = "'hafa áhyggjur að' á sennilega að vera 'hafa áhyggjur af'"
        detail = (
            "Í samhenginu 'hafa áhyggjur af e-u' er notuð "
            "forsetningin 'af', ekki 'að'."
        )
        if match.tidy_text.count(" af ") == 1:
            # Only one way to substitute af -> að: do it
            suggest = match.tidy_text.replace(" af ", " að ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="af",
                suggest=suggest,
            )
        )
    
    def wrong_preposition_hluti_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Calculate the start and end token indices, spanning both phrases
        start, end = match.span
        text = "'hluti að' á sennilega að vera 'hluti af'"
        detail = "Í samhenginu 'hluti af e-u' er notuð " "forsetningin 'af', ekki 'að'."
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def wrong_preposition_að_leiðandi(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending adverbial phrase
        advp = match.first_match("ADVP > { 'þar' }", self.ctx_að)
        if advp is None:
            advp = match.first_match("ADVP >> { 'þar' }", self.ctx_að)
        # Find the attached prepositional phrase
        vp = match.first_match('VP > { "leiðandi" }')
        assert advp is not None
        assert vp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(advp.span[0], vp.span[0]), max(advp.span[1], vp.span[1])
        text = "'þar að leiðandi' á sennilega að vera 'þar af leiðandi'"
        detail = (
            "Í samhenginu 'þar af leiðandi' er notuð " "forsetningin 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def wrong_preposition_að_mörkum(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending prepositional phrase
        pp = match.first_match("PP > { 'að' 'mark' }", self.ctx_að)
        if pp is None:
            pp = match.first_match("PP > { 'að' 'mörk' }", self.ctx_að)
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp.span[0], pp.span[1]
        text = "'að mörkum' á sennilega að vera 'af mörkum'"
        detail = (
            "Í samhenginu 'leggja e-ð af mörkum' er notuð "
            "forsetningin 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def wrong_preposition_að_leiða(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Calculate the start and end token indices, spanning both phrases
        start, end = match.span
        text = "'að leiða' á sennilega að vera 'af leiða'"
        detail = (
            "Í samhenginu 'láta gott af sér leiða' er notuð "
            "forsetningin 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def wrong_preposition_heiður_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending nominal phrase
        np = match.first_match("NP > { 'heiður' }", self.ctx_að)
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'að' }", self.ctx_að)
        assert np is not None
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(np.span[0], pp.span[0]), max(np.span[1], pp.span[1])
        text = "'heiðurinn að' á sennilega að vera 'heiðurinn af'"
        detail = (
            "Í samhenginu 'eiga heiðurinn af' er notuð " "forsetningin 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def wrong_preposition_eiga_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verb phrase
        vp = match.first_match("VP > { 'eiga' }", self.ctx_að)
        if vp is None:
            vp = match.first_match("VP >> { 'eiga' }", self.ctx_að)
        # Find the nominal object
        np = match.first_match("( NP|ADVP )", self.ctx_að)
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "að" }')
        assert vp is not None
        assert np is not None
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = (
            min(vp.span[0], np.span[0], pp.span[0]),
            max(vp.span[1], np.span[1], pp.span[1]),
        )
        text = "'{0} að' á sennilega að vera '{0} af'".format(np.tidy_text)
        detail = (
            "Orðasambandið 'að eiga {0}' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'.".format(np.tidy_text)
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def wrong_preposition_vera_til_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        start, end = match.span
        text = "'til að' á sennilega að vera 'til af'"
        detail = (
            "Orðasambandið 'að vera til af' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def wrong_preposition_gagn_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        start, end = match.span
        text = "'gagn að' á sennilega að vera 'gagn af'"
        detail = (
            "Orðasambandið 'að hafa gagn af e-u' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def wrong_preposition_að_sjalfu(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        start, end = match.span
        text = "'að sjálfu sér' á sennilega að vera 'af sjálfu sér'"
        detail = (
            "Orðasambandið 'af sjálfu sér' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def wrong_preposition_frettir_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending preposition
        pp = match.first_match("P > { 'að' }", self.ctx_að)
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp.span[0], pp.span[1]
        text = "'að' á sennilega að vera 'af'"
        detail = (
            "Orðasambandið 'fréttir berast af e-u' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def wrong_preposition_stafa_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal phrase
        vp = match.first_match("VP > { 'stafa' }", self.ctx_að)
        assert vp is not None
        start, end = match.span
        if " að " in vp.tidy_text:
            text = "'{0}' á sennilega að vera '{1}'".format(
                vp.tidy_text, vp.tidy_text.replace(" að ", " af ")
            )
        else:
            text = "'{0} að' á sennilega að vera '{0} af'".format(vp.tidy_text)
        detail = (
            "Orðasambandið 'að stafa af e-u' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def wrong_preposition_ólétt_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending nominal phrase
        np = match.first_match("NP > { 'óléttur' }", self.ctx_að)
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'að' }", self.ctx_að)
        assert np is not None
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(np.span[0], pp.span[0]), max(np.span[1], pp.span[1])
        text = "'{0} að' á sennilega að vera '{0} af'".format(np.tidy_text)
        detail = (
            "Orðasambandið 'að vera ólétt/ur af e-u' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def wrong_preposition_heyra_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal phrase
        vp = match.first_match("VP > { 'heyra' }", self.ctx_að)
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'að' }", self.ctx_að)
        assert vp is not None
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(vp.span[0], pp.span[0]), max(vp.span[1], pp.span[1])
        if " að " in vp.tidy_text:
            text = "'{0}' á sennilega að vera '{1}'".format(
                vp.tidy_text, vp.tidy_text.replace(" að ", " af ")
            )
        else:
            text = "'{0} að' á sennilega að vera '{0} af'".format(vp.tidy_text)
        detail = (
            "Orðasambandið 'að heyra af e-u' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def wrong_preposition_hafa_gaman_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending nominal phrase
        np = match.first_match("NP > { 'gaman' }", self.ctx_að)
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'að' }", self.ctx_að)
        assert np is not None
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(np.span[0], pp.span[0]), max(np.span[1], pp.span[1])
        text = "'gaman að' á sennilega að vera 'gaman af'"
        detail = (
            "Orðasambandið 'að hafa gaman af e-u' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def wrong_preposition_heillaður_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Calculate the start and end token indices, spanning both phrases
        start, end = match.span
        text = "'heillaður að' á sennilega að vera 'heillaður af'"
        detail = (
            "Í samhenginu 'heillaður af e-u' er notuð " "forsetningin 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )
    def wrong_preposition_valinn_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending nominal phrase
        np = match.first_match("NP > { 'velja' }", self.ctx_að)
        if np is None:
            np = match.first_match("NP > { 'valinn' }", self.ctx_að)
        assert np is not None
        start, end = match.span
        if " að " in np.tidy_text:
            text = "'{0}' á sennilega að vera '{0}'".format(np.tidy_text)
        else:
            text = "'{0} að' á sennilega að vera '{0} af'".format(np.tidy_text)
        detail = (
            "Orðasambandið 'að vera valin/n af e-m' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def wrong_að_use(self, match: SimpleTree, context: ContextDict) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending noun
        np = match.first_match(" %noun ", context)
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'að' }")
        assert np is not None
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(np.span[0], pp.span[0]), max(np.span[1], pp.span[1])
        text = "Hér á líklega að vera forsetningin 'af' í stað 'að'."
        detail = "Í samhenginu '{0}' er rétt að nota forsetninguna 'af' í stað 'að'.".format(
            match.tidy_text
        )
        if match.tidy_text.count(" af ") == 1:
            # Only one way to substitute af -> að: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original="að",
                suggest=suggest,
            )
        )

    def check_pp_with_place(self, match: SimpleTree) -> None:
        """ Check whether the correct preposition is being used with a place name """
        place = match.NP.lemma
        correct_preposition = IcelandicPlaces.lookup_preposition(place)
        if correct_preposition is None:
            # This is not a known or likely place name
            return
        preposition = match.P.lemma
        if correct_preposition == preposition:
            # Correct: return
            return
        start, end = match.span
        suggest = match.tidy_text.replace(preposition, correct_preposition, 1)
        text = "Rétt er að rita '{0}'".format(suggest)
        detail = (
            "Ýmist eru notaðar forsetningarnar 'í' eða 'á' með nöfnum "
            "staða, bæja og borga. Í tilviki {0:ef} er notuð forsetningin '{1}'.".format(
                NounPhrase(place), correct_preposition
            )
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PLACE_PP",
                text=text,
                detail=detail,
                original=preposition,
                suggest=suggest,
            )
        )

    def wrong_noun_with_verb(self, match: SimpleTree) -> None:
        """ Wrong noun used with a verb, for instance
            'bjóða e-m birginn' instead of 'byrginn' """
        # TODO: This code is provisional, intended as a placeholder for similar cases
        start, end = match.span
        text = "Mælt er með að rita 'bjóða e-m byrginn' í stað 'birginn'."
        detail = text
        tidy_text = match.tidy_text
        suggest = tidy_text.replace("birginn", "byrginn", 1)
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_NOUN_WITH_VERB",
                text=text,
                detail=detail,
                original="birginn",
                suggest=suggest,
            )
        )

    def wrong_verb_use(
        self, match: SimpleTree, correct_verb: str, context: ContextDict,
    ) -> None:
        """ Annotate wrong verbs being used with nouns,
            for instance 'byði hnekki' where the verb should
            be 'bíða' -> 'biði hnekki' instead of 'bjóða' """
        vp = match.first_match("VP > { %verb }", context)
        assert vp is not None
        verb = next(ch for ch in vp.children if ch.tcat == "so").own_lemma_mm
        np = match.first_match("NP >> { %noun }", context)
        assert np is not None
        start, end = min(vp.span[0], np.span[0]), max(vp.span[1], np.span[1])
        # noun = next(ch for ch in np.leaves if ch.tcat == "no").own_lemma
        text = "Hér á líklega að vera sögnin '{0}' í stað '{1}'.".format(
            correct_verb, verb
        )
        detail = "Í samhenginu '{0}' er rétt að nota sögnina '{1}' í stað '{2}'.".format(
            match.tidy_text, correct_verb, verb
        )
        suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_VERB_USE",
                text=text,
                detail=detail,
                original=verb,
                suggest=suggest,
            )
        )

    def wrong_af_use(self, match: SimpleTree, context: ContextDict) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending nominal phrase
        np = match.first_match(" %noun ", context)
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'af' }")
        assert np is not None
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(np.span[0], pp.span[0]), max(np.span[1], pp.span[1])
        text = "Hér á líklega að vera forsetningin 'að' í stað 'af'."
        detail = "Í samhenginu '{0}' er rétt að nota forsetninguna 'að' í stað 'af'.".format(
            match.tidy_text
        )
        if match.tidy_text.count(" af ") == 1:
            # Only one way to substitute af -> að: do it
            suggest = match.tidy_text.replace(" af ", " að ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,     
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original="af",
                suggest=suggest,
            )
        )

    def vera_að(self, match: SimpleTree) -> None:
        start, end = match.span
        text = "Mælt er með að sleppa 'vera að' og beygja frekar sögnina."
        detail = text
        # tidy_text = match.tidy_text
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_VeraAð",
                text=text,
                detail=detail,
                original="vera að",
            )
        )

    @classmethod
    def create_patterns(cls) -> None:
        """ Initialize the list of patterns and handling functions """
        p = cls.PATTERNS

        # Access the dictionary of verb+preposition attachment errors
        # from the settings (actually from the reynir settings),
        # read from config/Verbs.conf
        prep_errors = VerbErrors.PREPOSITIONS_ERRORS

        # Build a set of verbs with common af/að errors
        verbs_af: Set[str] = set()
        verbs_að: Set[str] = set()

        for verb, d in prep_errors.items():

            if "_" in verb:
                # At this point, we're not interested in composites ('birgja_sig' etc.)
                continue

            spec = d.get("af")
            if spec is not None:
                # Decipher the error specification from config/Verbs.conf
                a = spec.split(", ")
                if len(a) >= 2 and a[0] == "PP":
                    a = a[1].split()
                    if len(a) == 2 and a[0] == "/að" and a[1] == "þgf":
                        # Found a verb that has a correction from 'af' to 'að'
                        verbs_af.add(verb)
                continue

            spec = d.get("að")
            if spec is not None:
                # Decipher the error specification from config/Verbs.conf
                a = spec.split(", ")
                if len(a) >= 2 and a[0] == "PP":
                    a = a[1].split()
                    if len(a) == 2 and a[0] == "/af" and a[1] == "þgf":
                        # Found a verb that has a correction from 'að' to 'af'
                        verbs_að.add(verb)
                continue

        if verbs_af:
            # Create matching patterns with a context that catches the af/að verbs.

            # The following context dictionary defines a resolver function for the
            # '%verb' macro. This function returns True if the potentially matching
            # tree node refers to a lemma that is one of the að/af verbs, and that verb
            # does not have arguments (i.e. it doesn't have 1 or 2 arguments; it may
            # have none or 0). An example is a 'so_0_gm_fh_nt' terminal matching the
            # verb 'leita'.

            # Note that we use the own_lemma_mm property instead of own_lemma. This
            # means that the lambda condition matches middle voice stem forms,
            # such as 'dást' instead of 'dá'.
            cls.ctx_af = {
                "verb": lambda tree: (
                    tree.own_lemma_mm in verbs_af
                    and not (set(tree.variants) & {"1", "2"})
                )
            }
            # Catch sentences such as 'Jón leitaði af kettinum'
            p.append(
                (
                    "af",  # Trigger lemma for this pattern
                    'VP > { VP >> { %verb } PP >> { P > { "af" } } }',
                    cls.wrong_preposition_af,
                    cls.ctx_af,
                )
            )

            # Catch sentences such as 'Vissulega er hægt að brosa af þessu',
            # 'Friðgeir var leitandi af kettinum í allan dag'
            p.append(
                (
                    "af",  # Trigger lemma for this pattern
                    '. > { (NP-PRD | IP-INF) > { VP > { %verb } } PP >> { P > { "af" } } }',
                    cls.wrong_preposition_af,
                    cls.ctx_af,
                )
            )
            # Catch "Það sem Jón spurði ekki af...", "Jón spyr (ekki) af því."
            p.append(
                (
                    "spyrja",  # Trigger lemma for this pattern
                    "IP > { VP >> { 'spyrja' } ADVP > { 'af' } }",
                    cls.wrong_preposition_af,
                    cls.ctx_af,
                )
            )
            # Catch "Jón spyr af því."
            p.append(
                (
                    "spyrja",  # Trigger lemma for this pattern
                    "IP > { VP >> { 'spyrja' } PP > { 'af' } }",
                    cls.wrong_preposition_af,
                    cls.ctx_af,
                )
            )
            # Catch "...vegna þess að dýr leita af öðrum smærri dýrum."
            p.append(
                (
                    "leita",  # Trigger lemma for this pattern
                    "VP > { PP >> { %verb } PP > 'af' }",
                    cls.wrong_preposition_af,
                    cls.ctx_af,
                )
            )

            # Catch "Þetta er mesta vitleysa sem ég hef orðið vitni af"
            p.append(
                (
                    "vitni",  # Trigger lemma for this pattern
                    "VP > { VP >> [ .* ('verða' | 'vera') .* \"vitni\" ] "
                    'ADVP > "af" }',
                    cls.wrong_preposition_vitni_af,
                    None,
                )
            )
            # Catch "Hún varð vitni af því þegar kúturinn sprakk"
            p.append(
                (
                    "vitni",  # Trigger lemma for this pattern
                    "VP > { VP > [ .* ('verða' | 'vera') .* "
                    'NP-PRD > { "vitni" PP > { P > { "af" } } } ] } ',
                    cls.wrong_preposition_vitni_af,
                    None,
                )
            )
            # Catch ""
            p.append(
                (
                    "vitni",  # Trigger lemma for this pattern
                    "VP > { VP > 'verða' NP-PRD > { 'vitni' PP > { P > { 'af' } } } }",
                    cls.wrong_preposition_vitni_af,
                    None,
                )
            )
            # Catch "Hún gerði grín af því."
            p.append(
                (
                    "grín",  # Trigger lemma for this pattern
                    "VP > { VP > [ .* 'gera' .* "
                    'NP-OBJ > "grín" ] ( PP | ADVP ) > ( { P > { "af" } } | { "af" } ) } ',
                    cls.wrong_preposition_grin_af,
                    None,
                )
            )
            # Catch "Þetta er mesta vitleysa sem ég hef gert grín af."
            p.append(
                (
                    "grín",  # Trigger lemma for this pattern
                    "VP > { VP > { VP > { VP > { 'gera' } NP-OBJ > { 'grín' } } } ADVP > { 'af' } }",
                    cls.wrong_preposition_grin_af,
                    None,
                )
            )
            # Catch "...og gerir grín af sjálfum sér."
            p.append(
                (
                    "grín",  # Trigger lemma for this pattern
                    "VP > { VP > { VP > 'gera' NP-OBJ } PP > { 'af' } }",
                    cls.wrong_preposition_grin_af,
                    None,
                )
            )

            p.append(
                (
                    "grín",  # Trigger lemma for this pattern
                    "VP > { PP > { NP > { 'grín' } } PP > { 'af' } }",
                    cls.wrong_preposition_grin_af,
                    None,
                )
            )

            # Catch "Hann leiðir ekki líkur af því."
            p.append(
                (
                    "leiða",  # Trigger lemma for this pattern
                    "VP > { VP > { 'leiða' } NP-PRD > { ('líkur' | 'rök' | 'rak') } PP > { P > { 'af' } } } ",
                    cls.wrong_preposition_leida_af,
                    None,
                )
            )
            # Catch "Hann leiðir líkur af því.", "Hann leiðir rök af því.", "Hann hefur aldrei leitt líkur af því."
            p.append(
                (
                    "leiða",  # Trigger lemma for this pattern
                    "VP > { VP >> { VP > { 'leiða' } NP > { ('líkur' | 'rök' | 'rak') } } PP > { 'af' } } ",
                    cls.wrong_preposition_leida_af,
                    None,
                )
            )
            # Catch "Tíminn markar upphaf af því."
            p.append(
                (
                    "upphaf",  # Trigger lemma for this pattern
                    "VP > { VP > { 'marka' } NP-OBJ > { 'upphaf' PP > { 'af' } } }",
                    cls.wrong_preposition_marka_af,
                    None,
                )
            )
            # Catch "Það markar ekki upphaf af því."
            p.append(
                (
                    "upphafinn",  # Trigger lemma for this pattern
                    "VP > { VP > { 'marka' } NP > { 'upphafinn' } PP > { 'af' } }",
                    cls.wrong_preposition_marka_af,
                    None,
                )
            )
            # Catch "Það markar upphaf af því."
            p.append(
                (
                    "upphaf",  # Trigger lemma for this pattern
                    "VP > { VP > { VP > { 'marka' } NP-SUBJ > { 'upphaf' } } PP > { 'af' } }",
                    cls.wrong_preposition_marka_af,
                    None,
                )
            )
            # Catch "Það hefur ekki markað upphafið af því."
            p.append(
                (
                    "upphefja",  # Trigger lemma for this pattern
                    "VP > { VP > { NP > { 'markaður' } VP > { 'upphefja' } } PP > { 'af' } }",
                    cls.wrong_preposition_marka_af,
                    None,
                )
            )

            # Catch "Jón leggur hann (ekki) af velli."
            p.append(
                (
                    "leggja",  # Trigger lemma for this pattern
                    "VP > { VP > { VP > { 'leggja' } } PP > { P > { 'af' } NP > { ( 'völlur'|'vell' ) } } }",
                    cls.wrong_preposition_leggja_af,
                    None,
                )
            )
            # Catch "Jón hefur lagt hann af velli."
            p.append(
                (
                    "leggja",  # Trigger lemma for this pattern
                    "VP > { VP > { VP > { VP > { 'leggja' } } } PP > { P > 'af' NP > ( 'völlur'|'vell' ) } }",
                    cls.wrong_preposition_leggja_af,
                    None,
                )
            )

            # Catch "Jón kann það (ekki) utan af."
            p.append(
                (
                    "kunna",  # Trigger lemma for this pattern
                    "VP > { VP > { 'kunna' } ADVP > { 'utan' } ADVP > { 'af' } }",
                    cls.wrong_preposition_utan_af,
                    None,
                )
            )
            # Catch "Honum varð af ósk sinni."
            p.append(
                (
                    "ósk",  # Trigger lemma for this pattern
                    "(S-MAIN | IP) > { VP > { 'verða' } PP > { 'af' NP > { 'ósk' } } }",
                    cls.wrong_preposition_verða_af,
                    None,
                )
            )
            # Catch "...en varð ekki af ósk sinni."
            p.append(
                (
                    "ósk",  # Trigger lemma for this pattern
                    "IP > { VP > { VP > { 'verða' } PP > { P > { 'af' } NP > { 'ósk' } } } }",
                    cls.wrong_preposition_verða_af,
                    None,
                )
            )

            # Catch "Ég varð (ekki) uppvís af athæfinu."
            p.append(
                (
                    "uppvís",  # Trigger lemma for this pattern
                    "VP > { VP > { VP > { 'verða' } NP > { 'uppvís' } } PP > { 'af' } }",
                    cls.wrong_preposition_uppvis_af,
                    None,
                )
            )
            # Catch "Hann varð (ekki) uppvís af því.", "Hann hafði orðið uppvís af því."
            p.append(
                (
                    "uppvís",  # Trigger lemma for this pattern
                    "VP > { VP > { 'verða' } NP > { NP > { 'uppvís' } PP > { 'af' } } }",
                    cls.wrong_preposition_uppvis_af,
                    None,
                )
            )


        if verbs_að:
            # Create matching patterns with a context that catches the að/af verbs.
            cls.ctx_að = {
                "verb": lambda tree: (
                    tree.own_lemma_mm in verbs_að
                    and not (set(tree.variants) & {"1", "2"})
                )
            }
            # Catch sentences such as 'Jón heillaðist að kettinum'
            p.append(
                (
                    "að",  # Trigger lemma for this pattern
                    'VP > { VP >> { %verb } PP >> { P > { "að" } } }',
                    cls.wrong_preposition_að,
                    cls.ctx_að,
                )
            )
            # Catch sentences such as 'Vissulega er hægt að heillast að þessu'
            p.append(
                (
                    "að",  # Trigger lemma for this pattern
                    '. > { (NP-PRD | IP-INF) > { VP > { %verb } } PP >> { P > { "að" } } }',
                    cls.wrong_preposition_að,
                    cls.ctx_að,
                )
            )

            # Catch "Þetta er fallegasta kona sem ég hef orðið heillaður að"
            p.append(
                (
                    "heilla",  # Trigger lemma for this pattern
                    "VP > { VP > [ .* ('verða' | 'vera') ] NP-PRD > [ .* 'heilla' .* ADVP > { \"að\" } ] }",
                    cls.wrong_preposition_heillaður_að,
                    None,
                )
            )
            # Catch "Ég hef lengi verið heillaður að henni."
            p.append(
                (
                    "heilla",  # Trigger lemma for this pattern
                    "NP > { NP >> { 'heilla' } PP > { 'að' } }",
                    cls.wrong_preposition_heillaður_að,
                    None,
                )
            )

            # Catch "Ég er ekki hluti að heildinni."
            p.append(
                (
                    "hluti",  # Trigger lemma for this pattern
                    "VP > { VP > { 'vera' NP-PRD > { 'hluti' } } PP > { 'að' } }",
                    cls.wrong_preposition_hluti_að,
                    None,
                )
            )
            # Catch "Við höfum öll verið hluti að heildinni."
            p.append(
                (
                    "hluti",  # Trigger lemma for this pattern
                    "VP > { VP > { VP > { 'vera' 'hluti' } } PP > { 'að' } }",
                    cls.wrong_preposition_hluti_að,
                    None,
                )
            )
            # Catch "Þeir sögðu að ég hefði verið hluti að heildinni."
            p.append(
                (
                    "hluti",  # Trigger lemma for this pattern
                    "VP > { VP > { 'vera' } NP > { 'hluti' PP > { 'að' } } }",
                    cls.wrong_preposition_hluti_að,
                    None,
                )
            )

            # Catch "Þar að leiðandi virkar þetta.", "Þetta virkar þar að leiðandi."
            p.append(
                (
                    "leiða",  # Trigger lemma for this pattern
                    "(IP | VP) > { ADVP > { 'þar' } ADVP > { 'að' } VP > { 'leiða' } }",
                    cls.wrong_preposition_að_leiðandi,
                    None,
                )
            )

            # Catch "Ég hef (ekki) ekki áhyggjur að honum.", "Ég hef áhyggjur að því að honum líði illa."
            p.append(
                (
                    "áhyggja",  # Trigger lemma for this pattern
                    "VP > { VP >> { 'áhyggja' } PP > { 'að' } }",
                    cls.wrong_preposition_ahyggja_að,
                    None,
                )
            )

            # Catch "Ég lagði mikið að mörkum."
            p.append(
                (
                    "mörk",  # Trigger lemma for this pattern
                    "VP > { VP >> { 'leggja' } PP > { P > 'að' NP > { 'mörk' } } }",
                    cls.wrong_preposition_að_mörkum,
                    None,
                )
            )
            # Catch "Ég lagði ekki mikið að mörkum.", "Ég hafði (ekki) lagt mikið að mörkum."
            p.append(
                (
                    "mörk",  # Trigger lemma for this pattern
                    "VP > { VP >> { 'leggja' } PP > { P > 'að' NP > { 'mörk' } } }",
                    cls.wrong_preposition_að_mörkum,
                    None,
                )
            )

            # Catch "Ég lét (ekki) gott að mér leiða."
            p.append(
                (
                    "leiða",  # Trigger lemma for this pattern
                    "VP > { VP > { 'láta' } VP > { PP > { 'að' } VP > { 'leiða' } } }",
                    cls.wrong_preposition_að_leiða,
                    None,
                )
            )
            p.append(
                (
                    "leiður",  # Trigger lemma for this pattern
                    "VP > { VP > { 'láta' } PP > { P > { 'að' } 'leiður' } }",
                    cls.wrong_preposition_að_leiða,
                    None,
                )
            )
            # Catch "Ég lét (ekki) gott að mér leiða."
            p.append(
                (
                    "leiður",  # Trigger lemma for this pattern
                    "VP > { VP > { 'láta' } PP > { P > { 'að' } 'leiður' } }",
                    cls.wrong_preposition_að_leiða,
                    None,
                )
            )

            # Catch "Hún fær/hlýtur (ekki) heiðurinn að þessu.", "Hún hafði (ekki) fengið/hlotið heiðurinn að þessu."
            p.append(
                (
                    "heiður",  # Trigger lemma for this pattern
                    "VP > { VP > { ( 'fá'|'hljóta' ) } NP > { 'heiður' PP > { 'að' } } }",
                    cls.wrong_preposition_heiður_að,
                    None,
                )
            )

            # Catch "Hún á (ekki) mikið/fullt/helling/gommu... að börnum.", "Hún á (ekki) heiðurinn að þessu.", "Hún hafði (ekki) átt heiðurinn að þessu."
            p.append(
                (
                    "eiga",  # Trigger lemma for this pattern
                    "VP > { VP >> { 'eiga' NP } PP > { P > { 'að' } } }",
                    cls.wrong_preposition_eiga_að,
                    None,
                )
            )
            # Catch "Hún á (ekki) lítið að börnum."
            p.append(
                (
                    "eiga",  # Trigger lemma for this pattern
                    "VP > { VP > { 'eiga' } ADVP > { 'lítið' } PP > { 'að' } }",
                    cls.wrong_preposition_eiga_að,
                    None,
                )
            )

            # Catch "Það er (ekki) til mikið að þessu."
            p.append(
                (
                    "vera",  # Trigger lemma for this pattern
                    "VP > { VP > { 'vera' } NP > { NP >> { 'til' } PP > { 'að' } } }",
                    cls.wrong_preposition_vera_til_að,
                    None,
                )
            )
            # Catch "Mikið er til að þessu."
            p.append(
                (
                    "vera",  # Trigger lemma for this pattern
                    "( S|VP ) > { NP VP > { 'vera' } ADVP > { 'til' } PP > { 'að' } }",
                    cls.wrong_preposition_vera_til_að,
                    None,
                )
            )
            # Catch "Ekki er mikið til að þessu."
            p.append(
                (
                    "vera",  # Trigger lemma for this pattern
                    "VP > { VP > { 'vera' } ADVP > { 'til' } PP > { 'að' } }",
                    cls.wrong_preposition_vera_til_að,
                    None,
                )
            )

            # Catch "Hún hefur (ekki) gagn að þessu.", "Hún hefur (ekki) haft gagn að þessu."
            p.append(
                (
                    "gagn",  # Trigger lemma for this pattern
                    "VP > { VP >> { NP > { 'gagn' } } PP > { 'að' } }",
                    cls.wrong_preposition_gagn_að,
                    None,
                )
            )
            # Catch "Hvaða gagn hef ég að þessu?"
            p.append(
                (
                    "gagn",  # Trigger lemma for this pattern
                    "S > { NP > { 'gagn' } IP > { VP > { VP > { 'hafa' } PP > { 'að' } } } }",
                    cls.wrong_preposition_gagn_að,
                    None,
                )
            )

            # Catch "Þetta kom (ekki) að sjálfu sér.", "Þetta hafði (ekki) komið að sjálfu sér."
            p.append(
                (
                    "sjálfur",  # Trigger lemma for this pattern
                    "PP > { P > { 'að' } NP > { 'sjálfur' } }",
                    cls.wrong_preposition_að_sjalfu,
                    None,
                )
            )

            # Catch "Fréttir bárust (ekki) að slysinu."
            p.append(
                (
                    "frétt",  # Trigger lemma for this pattern
                    "( IP|VP ) > { NP > { 'frétt' } VP > { PP > { P > { 'að' } } } }",
                    cls.wrong_preposition_frettir_að,
                    None,
                )
            )
            # Catch "Það bárust (ekki) fréttir að slysinu."
            p.append(
                (
                    "frétt",  # Trigger lemma for this pattern
                    "NP > { 'frétt' PP > { P > { 'að' } } }",
                    cls.wrong_preposition_frettir_að,
                    None,
                )
            )

            # Catch "Hætta stafar (ekki) að þessu.", "Hætta hefur (ekki) stafað að þessu."
            p.append(
                (
                    "stafa",  # Trigger lemma for this pattern
                    "( VP|IP ) > { VP >> { 'stafa' } ( PP|ADVP ) > { 'að' } }",
                    cls.wrong_preposition_stafa_að,
                    None,
                )
            )

            # Catch "Hún er (ekki) ólétt að sínu þriðja barni.", "Hún hefur (ekki) verið ólétt að sínu þriðja barni."
            p.append(
                (
                    "óléttur",  # Trigger lemma for this pattern
                    "VP > { VP > { NP > { 'óléttur' } } PP > { 'að' } }",
                    cls.wrong_preposition_ólétt_að,
                    None,
                )
            )

            # Catch "Hún heyrði að lausa starfinu.", "Hún hefur (ekki) heyrt að lausa starfinu."
            p.append(
                (
                    "heyra",  # Trigger lemma for this pattern
                    "VP > { VP >> { 'heyra' } PP > { 'að' } }",
                    cls.wrong_preposition_heyra_að,
                    None,
                )
            )
            p.append(
                (
                    "heyra",  # Trigger lemma for this pattern
                    "VP > { PP >> { 'heyra' } PP > { 'að' } }",
                    cls.wrong_preposition_heyra_að,
                    None,
                )
            )

            # Catch "Ég hef (ekki) gaman að henni.", "Ég hef aldrei haft gaman að henni."
            p.append(
                (
                    "gaman",  # Trigger lemma for this pattern
                    "VP > { VP >> { VP > { 'hafa' } NP > { 'gaman' } } PP > { 'að' } }",
                    cls.wrong_preposition_hafa_gaman_að,
                    None,
                )
            )

            # Catch "Ég var valinn að henni.", "Ég hafði (ekki) verið valinn að henni."
            p.append(
                (
                    "velja",  # Trigger lemma for this pattern
                    "NP > { NP > { 'velja' } PP > { 'að' } }",
                    cls.wrong_preposition_valinn_að,
                    None,
                )
            )
            # Catch "Ég var ekki valinn að henni."
            p.append(
                (
                    "valinn",  # Trigger lemma for this pattern
                    "NP > { NP > { 'valinn' } PP > { 'að' } }",
                    cls.wrong_preposition_valinn_að,
                    None,
                )
            )
            p.append(
                (
                    "valinn",  # Trigger lemma for this pattern
                    "VP > { VP >> { 'valinn' } PP > { 'að' } }",
                    cls.wrong_preposition_valinn_að,
                    None,
                )
            )

        # Verbs used wrongly with particular nouns
        def wrong_noun(nouns: Set[str], tree: SimpleTree) -> bool:
            """ Context matching function for the %noun macro in combinations
                of verbs and their noun objects """
            lemma = tree.own_lemma
            if not lemma:
                # The passed-in tree node is probably not a terminal
                return False
            try:
                case = (set(tree.variants) & {"nf", "þf", "þgf", "ef"}).pop()
            except KeyError:
                return False
            return (lemma + "_" + case) in nouns

        NOUNS_01 = {
            "ósigur_þf",
            "hnekkir_þf",
            "álitshnekkir_þf",
            "afhroð_þf",
            "bani_þf",
            "færi_ef",
            "boð_ef",
            "átekt_ef",
        }
        # The macro %verb expands to "@'bjóða'" which matches terminals
        # whose corresponding token has a meaning with the 'bjóða' lemma.
        # The macro %noun is resolved by calling the function wrong_noun()
        # with the potentially matching tree node as an argument.
        cls.ctx_verb_01 = {"verb": "@'bjóða'", "noun": partial(wrong_noun, NOUNS_01)}
        p.append(
            (
                "bjóða",  # Trigger lemma for this pattern
                "VP > { VP > { %verb } NP-OBJ >> { %noun } }",
                lambda self, match: self.wrong_verb_use(
                    match, "bíða", cls.ctx_verb_01,
                ),
                cls.ctx_verb_01,
            )
        )

        NOUNS_02 = {"haus_þgf", "þvottur_þgf"}
        cls.ctx_verb_02 = {"verb": "@'hegna'", "noun": partial(wrong_noun, NOUNS_02)}
        p.append(
            (
                "hegna",  # Trigger lemma for this pattern
                "VP > { VP > { %verb } NP-OBJ >> { %noun } }",
                lambda self, match: self.wrong_verb_use(
                    match, "hengja", cls.ctx_verb_02,
                ),
                cls.ctx_verb_02,
            )
        )

        # 'af' incorrectly used with particular nouns
        def wrong_noun_af(nouns: Set[str], tree: SimpleTree) -> bool:
            """ Context matching function for the %noun macro in combination
                with 'af' """
            lemma = tree.own_lemma
            if not lemma:
                # The passed-in tree node is probably not a terminal
                return False
            try:
                case = (set(tree.variants) & {"nf", "þf", "þgf", "ef"}).pop()
            except KeyError:
                return False
            return (lemma + "_" + case) in nouns

        NOUNS_AF: Set[str] = {
            "beiðni_þgf",
            "siður_þgf",
            "tilefni_þgf",
            "fyrirmynd_þgf"
        }
        # The macro %noun is resolved by calling the function wrong_noun_af()
        # with the potentially matching tree node as an argument.
        cls.ctx_noun_af = {"noun": partial(wrong_noun_af, NOUNS_AF)}
        p.append(
            (
                "af",  # Trigger lemma for this pattern
                "PP > { P > { 'af' } NP > { %noun } }",
                lambda self, match: self.wrong_af_use(
                    match, cls.ctx_noun_af
                ),
                cls.ctx_noun_af,
            )
        )

        NOUNS_AF_OBJ: Set[str] = {
            "aðgangur_þf",
            "aðgangur_nf",
            "drag_nf",
            "drag_þf",
            "grunnur_nf",
            "grunnur_þf",
            "hugmynd_nf",
            "hugmynd_þf",
            "leit_nf",
            "leit_þf",
            "dauðaleit_nf",
            "dauðaleit_þf",
            "lykill_nf",
            "lykill_þf",
            "uppskrift_nf",
            "uppskrift_þf",
           # "grín_þf"
        }
        # The macro %noun is resolved by calling the function wrong_noun_af()
        # with the potentially matching tree node as an argument.
        cls.ctx_noun_af_obj = {"noun": partial(wrong_noun_af, NOUNS_AF_OBJ)}
        p.append(
            (
                "af",  # Trigger lemma for this pattern
                "NP > { %noun PP > { P > { 'af' } } }",
                lambda self, match: self.wrong_af_use(
                    match, cls.ctx_noun_af_obj
                ),
                cls.ctx_noun_af_obj,
            )
        )
        p.append(
            (
                "af",  # Trigger lemma for this pattern
                "VP > { VP >> { %noun } PP > { P > { 'af' } } }",
                lambda self, match: self.wrong_af_use(
                    match, cls.ctx_noun_af_obj
                ),
                cls.ctx_noun_af_obj,
            )
        )
        p.append(
            (
                "af",  # Trigger lemma for this pattern
                "VP > { PP > { NP > %noun } PP > { 'af' } }",
                lambda self, match: self.wrong_af_use(
                    match, cls.ctx_noun_af_obj
                ),
                cls.ctx_noun_af_obj,
            )
        )


        def wrong_noun_að(nouns: Set[str], tree: SimpleTree) -> bool:
            """ Context matching function for the %noun macro in combination
                with 'að' """
            lemma = tree.own_lemma
            if not lemma:
                # The passed-in tree node is probably not a terminal
                return False
            try:
                case = (set(tree.variants) & {"nf", "þf", "þgf", "ef"}).pop()
            except KeyError:
                return False
            return (lemma + "_" + case) in nouns

        NOUNS_AÐ: Set[str] = {
            "tag_þgf",
            "togi_þgf",
            "kraftur_þgf",
            "hálfa_þgf",
            "hálfur_þgf",
        }
        # The macro %noun is resolved by calling the function wrong_noun_að()
        # with the potentially matching tree node as an argument.
        cls.ctx_noun_að = {"noun": partial(wrong_noun_að, NOUNS_AÐ)}
        p.append(
            (
                "að",  # Trigger lemma for this pattern
                "PP > { P > { 'að' } NP > { %noun } }",
                lambda self, match: self.wrong_að_use(match, cls.ctx_noun_að),
                cls.ctx_noun_að,
            )
        )

        def maybe_place(tree: SimpleTree) -> bool:
            """ Context matching function for the %maybe_place macro.
                Returns True if the associated lemma is an uppercase
                word that might be a place name. """
            lemma = tree.lemma
            return lemma[0].isupper() if lemma else False

        # Check prepositions used with place names
        cls.ctx_place_names = {"maybe_place": maybe_place}
        p.append(
            (
                frozenset(("á", "í")),  # Trigger lemmas for this pattern
                "PP > { P > ('á' | 'í') NP > %maybe_place }",
                lambda self, match: self.check_pp_with_place(match),
                cls.ctx_place_names,
            )
        )

        # Check use of 'bjóða e-m birginn' instead of 'bjóða e-m byrginn'
        # !!! TODO: This is a provisional placeholder for similar cases
        p.append(
            (
                "birgir",  # Trigger lemma for this pattern
                "VP > [ VP > { 'bjóða' } .* NP-IOBJ .* NP-OBJ > { \"birginn\" } ]",
                cls.wrong_noun_with_verb,
                None,
            )
        )

        # Check use of "vera að" instead of a simple verb
        p.append(
            (
                "vera",  # Trigger lemma for this pattern
                "VP > [VP > { @'vera' } (ADVP|NP-SUBJ)? IP-INF > {TO > nhm}]",
                lambda self, match: self.vera_að(match),
                None,
            )
        )

    def run(self) -> None:
        """ Apply the patterns to the sentence """
        tree = None if self._sent is None else self._sent.tree
        if tree is None:
            # No tree: nothing to do
            return
        # Make a set of the lemmas in the sentence
        # (Note: these are ordinary lemmas, not middle voice lemmas, so be careful
        # not to use middle voice lemmas as trigger words)
        if not self._sent.lemmas:
            return
        lemmas = set(lemma.replace("-", "") for lemma in self._sent.lemmas)

        def lemma_match(trigger: Union[str, FrozenSet[str]]) -> bool:
            if not trigger:
                return True
            if isinstance(trigger, str):
                return trigger in lemmas
            return bool(lemmas & trigger)

        for trigger, pattern, func, context in self.PATTERNS:
            # We only do the expensive pattern matching if the trigger lemma
            # for a pattern rule (if given) is actually found in the sentence
            if lemma_match(trigger):
                for match in tree.all_matches(pattern, context):
                    # Call the annotation function for this match
                    func(self, match)

