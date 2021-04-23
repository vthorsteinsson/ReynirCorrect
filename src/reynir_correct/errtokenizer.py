"""

    Greynir: Natural language processing for Icelandic

    Error-correcting tokenization layer

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


    This module adds layers to the bintokenizer.py module in GreynirPackage.
    These layers add token-level error corrections and recommendation flags
    to the token stream.

"""

from typing import (
    Mapping,
    Sequence,
    TypeVar,
    cast,
    Any,
    Type,
    Union,
    Tuple,
    List,
    Dict,
    Iterable,
    Iterator,
    Optional,
)

import re
from abc import ABC, abstractmethod

from tokenizer.abbrev import Abbreviations
from tokenizer.definitions import (
    BIN_Tuple,
    BIN_TupleList,
    NumberTuple,
    PersonNameList,
    ValType,
)
from reynir import TOK, Tok
from reynir.bintokenizer import (
    DefaultPipeline,
    MatchingStream,
    TokenConstructor,
    load_token,
    GreynirBin,
    StringIterable,
    TokenIterator,
)
from reynir.binparser import BIN_Token, VariantHandler

from .settings import (
    AllowedMultiples,
    WrongCompounds,
    SplitCompounds,
    UniqueErrors,
    MultiwordErrors,
    CapitalizationErrors,
    TabooWords,
    CIDErrorForms,
    Morphemes,
    Settings,
)
from .spelling import Corrector


# Token constructor classes
TokenCtor = Type["Correct_TOK"]

# Words that contain any letter from the following set are assumed
# to be foreign and their spelling is not corrected, but suggestions are made
NON_ICELANDIC_LETTERS_SET = frozenset("cwqøâãäçĉčêëîïñôõûüÿßĳ")

# Month names, incorrectly capitalized
MONTH_NAMES_CAPITALIZED = (
    "Janúar",
    "Febrúar",
    # "Mars", # Can also be planet
    "Apríl",
    "Maí",
    "Júní",
    "Júlí",
    # "Ágúst",    # Also a name
    "September",
    "Október",
    "Nóvember",
    "Desember",
)

# Acronyms that should be all-uppercase
WRONG_ACRONYMS = frozenset(
    (
        # HÍ og HA ganga kannski ekki hér
        "Dv",
        "Rúv",
        "Byko",
        "Íbv",
        "Pga",
        "Em",
        "Ví",
        "Mr",
        "Mh",
        "Ms",
        "Hr",
        "Ísí",
        "Ksí",
        "Así",
        "Kr",
        "Fh",
        "Ía",
        "Ka",
        "Hk",
    )
)

# Word categories and their names
POS: Mapping[str, str] = {
    "lo": "lýsingarorð",
    "ao": "atviksorð",
    "kk": "nafnorð",
    "hk": "nafnorð",
    "kvk": "nafnorð",
    "so": "sagnorð",
}

# Dict of { (at sentence start, single letter token) : allowed correction }
SINGLE_LETTER_CORRECTIONS: Mapping[Tuple[bool, str], str] = {
    (False, "a"): "á",
    (False, "i"): "í",
    (True, "A"): "Á",
    (True, "I"): "Í",
}

# Correction of abbreviations that are not present in Abbreviations.WRONGDOTS
# !!! TODO: Move this to a config file
WRONG_ABBREVS: Mapping[str, str] = {
    "Amk.": "A.m.k.",
    "A.m.k": "A.m.k.",
    "Etv.": "E.t.v.",
    "Mtt.": "M.t.t.",
    "n.k.": "nk.",
    "T.d": "T.d.",
    "Uþb.": "U.þ.b.",
    "þ.á.m.": "þ. á m.",
    "Þ.á.m.": "Þ. á m.",
    "þeas.": "þ.e.a.s.",
    "Þeas.": "Þ.e.a.s.",
    "Þmt.": "Þ.m.t.",
    "ca": "ca.",
    "Ca": "Ca.",
}

# A dictionary of token error classes, used in serialization
ErrorType = Type["Error"]
ERROR_CLASS_REGISTRY: Dict[str, ErrorType] = dict()

_ErrorClass = TypeVar("_ErrorClass", bound=ErrorType)


def register_error_class(cls: _ErrorClass) -> _ErrorClass:
    """ A decorator that populates the registry of all error classes,
        to aid in serialization """
    global ERROR_CLASS_REGISTRY
    ERROR_CLASS_REGISTRY[cast(Any, cls).__name__] = cast(ErrorType, cls)
    return cls


def emulate_case(s: str, *, template: str) -> str:
    """ Return the string s but emulating the case of the template
        (lower/upper/capitalized) """
    if template.isupper():
        return s.upper()
    if template and template[0].isupper():
        return s.capitalize()
    return s


def is_cap(word: str) -> bool:
    """ Return True if the word is capitalized, i.e. starts with an
        uppercase character and is otherwise lowercase """
    return word[0].isupper() and (len(word) == 1 or word[1:].islower())


class CorrectToken(Tok):

    """ This class sneakily replaces the tokenizer.Tok tuple in the tokenization
        pipeline. When applying a CorrectionPipeline (instead of a DefaultPipeline,
        as defined in binparser.py in GreynirPackage), tokens get translated to
        instances of this class in the correct() phase. This works due to Python's
        duck typing, because a CorrectToken class instance is able to walk and quack
        - i.e. behave - like a tokenizer.Tok tuple. It adds an _err attribute to hold
        information about spelling and grammar errors, and some higher level functions
        to aid in error reporting and correction. """

    def __init__(self, kind: int, txt: str, val: ValType) -> None:
        super().__init__(kind, txt, val)
        # The following seems to be required for mypy
        self.val: ValType
        # Error annotation
        self._err: Union[None, Error, bool] = None
        # Capitalization state: indicates where this token appears in a sentence.
        # None or one of ("sentence_start", "after_ordinal", "in_sentence")
        self._cap: Optional[str] = None

    def __eq__(self, o: Any) -> bool:
        """ Comparison between two CorrectToken instances """
        if not isinstance(o, CorrectToken):
            return False
        return (
            self.kind == o.kind
            and self.txt == o.txt
            and self.val == o.val
            and self._err == o._err
        )

    def __ne__(self, o: Any) -> bool:
        """ Comparison between two CorrectToken instances """
        return not self.__eq__(o)

    @staticmethod
    def dump(tok: Tok) -> Tuple[Any, ...]:
        """ Returns a JSON-dumpable object corresponding to a CorrectToken """
        if not hasattr(tok, "_err"):
            # We assume this is a "plain" token, i.e. (kind, txt, val)
            assert isinstance(tok, Tok)
            return tok.as_tuple
        # This looks like a CorrectToken
        assert isinstance(tok, CorrectToken)
        t: Tuple[Any, ...] = (tok.kind, tok.txt, tok.val)
        err = tok._err
        if err is None or isinstance(err, bool):
            # Simple err field: return a 4-tuple
            return t + (err,)
        # This token has an associated error object:
        # return a 5-tuple with the error class name and instance dict
        # (which must be JSON serializable!)
        return t + (err.__class__.__name__, err.__dict__)

    @staticmethod
    def load(*args: Any) -> "CorrectToken":
        """ Loads a CorrectToken instance from a JSON dump """
        largs = len(args)
        assert largs > 3
        ct = CorrectToken(*load_token(*args))
        if largs == 4:
            # Simple err field: add it
            ct.set_error(args[3])
            return ct
        # 5-tuple: We have a proper error object
        error_class_name: str = args[3]
        error_dict = args[4]
        error_cls = ERROR_CLASS_REGISTRY[error_class_name]
        # Python hack to create a fresh, empty instance of the
        # error class, then update its dict directly from the JSON data.
        # Note that directly assigning __dict__ = error_dict causes
        # a segfault on PyPy, hence the .update() call - which anyway
        # feels cleaner.
        instance = error_cls.__new__(error_cls)
        instance.__dict__.update(error_dict)
        ct.set_error(instance)
        return ct

    @classmethod
    def from_token(cls, token: Tok) -> "CorrectToken":
        """ Wrap a raw token in a CorrectToken """
        return cls(token.kind, token.txt, token.val)

    @classmethod
    def word(cls, txt: str, val: Optional[BIN_TupleList] = None) -> "CorrectToken":
        """ Create a wrapped word token """
        return cls(TOK.WORD, txt, val)

    def __repr__(self) -> str:
        return "<CorrectToken(kind: {0}, txt: '{1}', val: {2})>".format(
            TOK.descr[self.kind], self.txt, self.val
        )

    __str__ = __repr__

    def set_capitalization(self, cap: str) -> None:
        """ Set the capitalization state for this token """
        self._cap = cap

    def copy_capitalization(
        self, other: Union["CorrectToken", Sequence["CorrectToken"]]
    ) -> None:
        """ Copy the capitalization state from another CorrectToken instance """
        if isinstance(other, CorrectToken):
            self._cap = other._cap
        else:
            self._cap = other[0]._cap

    @property
    def cap_sentence_start(self) -> bool:
        """ True if this token appears at sentence start """
        return self._cap == "sentence_start"

    @property
    def cap_after_ordinal(self) -> bool:
        """ True if this token appears after an ordinal at sentence start """
        return self._cap == "after_ordinal"

    @property
    def cap_in_sentence(self) -> bool:
        """ True if this token appears within a sentence """
        return self._cap == "in_sentence"

    def set_error(self, err: Union[None, "Error", bool]) -> None:
        """ Associate an Error class instance with this token """
        self._err = err

    def copy_error(
        self,
        other: Union[Sequence["CorrectToken"], "CorrectToken"],
        coalesce: bool = False,
    ) -> bool:
        """ Copy the error field from another CorrectToken instance """
        if isinstance(other, CorrectToken):
            self._err = other._err
            if coalesce and other.error_span > 1:
                # The original token had an associated error
                # spanning more than one token; now we're creating
                # a single token out of the span
                # ('fimm hundruð' -> number token), so we reset
                # the span to one token
                assert isinstance(self._err, Error)
                self._err.set_span(1)
        else:
            # We have a list of CorrectToken instances to copy from:
            # find the first error in the list, if any, and copy it
            for t in other:
                if self.copy_error(t, coalesce=True):
                    break
        return self._err is not None

    @property
    def error(self) -> Union[None, "Error", bool]:
        """ Return the error object associated with this token, if any """
        # Note that self._err may be a bool
        return self._err

    @property
    def error_description(self) -> str:
        """ Return the description of an error associated with this token, if any """
        return getattr(self._err, "description", "")

    @property
    def error_code(self) -> str:
        """ Return the code of an error associated with this token, if any """
        return getattr(self._err, "code", "")

    @property
    def error_original(self) -> str:
        """ Return the original text of the token """
        return getattr(self._err, "original", "")

    @property
    def error_suggest(self) -> str:
        """ Return the text of a suggested replacement of this token, if any """
        return getattr(self._err, "suggest", "")

    @property
    def error_span(self) -> int:
        """ Return the number of tokens affected by this error """
        return getattr(self._err, "span", 1)

    @property
    def error_detail(self) -> Optional[str]:
        """ Return the detailed description of this error, if any """
        return getattr(self._err, "detail", None)

    def add_corrected_meanings(self, m: Sequence[BIN_Tuple]) -> None:
        """ Add alternative BÍN meanings for this token, based on a
            suggested spelling correction """
        assert self.kind == TOK.WORD
        # We assume that the token has already been marked
        # with a SpellingSuggestion error
        assert isinstance(self._err, SpellingSuggestion)
        if self.val is None:
            self.val = list(m)
        else:
            assert isinstance(self.val, list)
            cast(List[BIN_Tuple], self.val).extend(m)

    def suggestion_does_not_match(
        self, terminal: VariantHandler, token: BIN_Token
    ) -> bool:
        """ Return True if this token has an associated spelling
            suggestion and that suggestion doesn't work grammatically
            within the sentence, as parsed """
        if not hasattr(self._err, "does_not_match"):
            return False
        return cast(Any, self._err).does_not_match(terminal, token)


class Error(ABC):

    """ Base class for spelling and grammar errors, warnings and recommendations.
        An Error has a code and can provide a description of itself.
        Note that Error instances (including subclass instances) are
        serialized to JSON and must therefore only contain serializable
        attributes, in a plain __dict__. """

    def __init__(
        self,
        code: str,
        *,
        original: Optional[str] = None,
        suggest: Optional[str] = None,
        is_warning: bool = False,
        span: int = 1,
    ) -> None:
        # Note that if is_warning is True, "/w" is appended to
        # the error code. This causes the Greynir UI to display
        # a warning annotation instead of an error annotation.
        self._code = code + ("/w" if is_warning else "")
        self._span = span
        self._original = original
        self._suggest = suggest

    def __eq__(self, o: Any) -> bool:
        return (
            isinstance(o, Error)
            and self._code == o._code
            and self._span == o._span
        )

    def __ne__(self, o: Any) -> bool:
        return not self.__eq__(o)

    @property
    def code(self) -> str:
        return self._code

    @property
    @abstractmethod
    def description(self) -> str:
        """ Should be overridden """
        ...

    def set_span(self, span: int) -> None:
        """ Set the number of tokens spanned by this error """
        self._span = span

    @property
    def span(self) -> int:
        """ Return the number of tokens spanned by this error """
        return self._span

    @property
    def original(self) -> Optional[str]:
        """ Return the original text to which the error applies """
        return self._original

    @property
    def suggest(self) -> Optional[str]:
        """ Return a suggestion for correction, if available """
        return self._suggest

    def __str__(self) -> str:
        return "{0}: {1} | {2}->{3}".format(
            self.code, self.description, self._original, self._suggest
        )

    def __repr__(self) -> str:
        return "<{3} {0}: {1} (span {2}) | '{4}'->'{5}'>".format(
            self.code,
            self.description,
            self.span,
            self.__class__.__name__,
            self.original or "",
            self.suggest or "",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "descr": self.description}


@register_error_class
class PunctuationError(Error):

    """ A PunctuationError is an error where punctuation is wrong """

    # N001: Wrong quotation marks
    # N002: Three periods should be an ellipsis
    # N003: Informal combination of punctuation marks (??!!)

    def __init__(
        self, code: str, txt: str, original: str, suggest: str, span: int = 1
    ) -> None:
        # Punctuation error codes start with "N"
        super().__init__("N" + code, span=span, original=original, suggest=suggest)
        self._txt = txt

    @property
    def description(self) -> str:
        return self._txt


@register_error_class
class CompoundError(Error):

    """ A CompoundError is an error where words are duplicated, split or not
        split correctly. """

    # C001: Duplicated word removed. Should be corrected.
    # C002: Wrongly compounded words split up. Should be corrected.
    # C003: Wrongly split compounds united. Should be corrected.
    # C004: Duplicated word marked as a possible error.
    #       Should be pointed out but not deleted.
    # C005/w: Possible split compound, depends on meaning/PoS chosen by parser.
    # C006: A part of a compound word is wrong.
    # C007: A multiword compound such as "skóla-og frístundasvið" correctly split up

    def __init__(
        self, code: str, txt: str, *, original: str, suggest: str, span: int = 1
    ) -> None:
        # Compound error codes start with "C"
        # We consider C004 to be a warning, not an error
        is_warning = code == "004"
        super().__init__(
            "C" + code,
            is_warning=is_warning,
            span=span,
            original=original,
            suggest=suggest,
        )
        self._txt = txt

    @property
    def description(self) -> str:
        return self._txt


@register_error_class
class UnknownWordError(Error):

    """ An UnknownWordError is an error where the given word form does not
        exist in BÍN or additional vocabularies, and cannot be explained as
        a compound word. """

    # U001: Unknown word. Nothing more is known. Cannot be corrected, only pointed out.

    def __init__(
        self, code: str, txt: str, original: str, suggest: str, is_warning: bool = False
    ) -> None:
        # Unknown word error codes start with "U"
        super().__init__(
            "U" + code, is_warning=is_warning, original=original, suggest=suggest
        )
        self._txt = txt

    @property
    def description(self) -> str:
        return self._txt


@register_error_class
class CapitalizationError(Error):

    """ A CapitalizationError is an error where a word is capitalized
        incorrectly, i.e. should be lower case but occurs in upper case
        except at the beginning of a sentence, or should be upper case
        but occurs in lower case. """

    # Z001: Word should begin with lowercase letter
    # Z002: Word should begin with uppercase letter
    # Z003: Month name should begin with lowercase letter
    # Z004: Numbers should be written in lowercase ('24 milljónir')
    # Z005: Amounts should be written in lowercase ('24 milljónir króna') (is_warning)
    # Z006: Acronyms should be written in uppercase ('RÚV')

    def __init__(
        self, code: str, txt: str, original: str, suggest: str, is_warning: bool = False
    ) -> None:
        # Capitalization error codes start with "Z"
        super().__init__(
            "Z" + code, original=original, suggest=suggest, is_warning=is_warning
        )
        self._txt = txt

    @property
    def description(self) -> str:
        return self._txt


@register_error_class
class AbbreviationError(Error):

    """ An AbbreviationError is an error where an abbreviation
        is not spelled out, punctuated or spaced correctly. """

    # A001: Abbreviation corrected
    # A002: Token found in Abbreviations.WRONGDOTS and no
    #       other meaning available; corrected as an acronym

    def __init__(self, code: str, txt: str, original: str, suggest: str) -> None:
        # Abbreviation error codes start with "A"
        super().__init__("A" + code, original=original, suggest=suggest)
        self._txt = txt

    @property
    def description(self) -> str:
        return self._txt


@register_error_class
class TabooWarning(Error):

    """ A TabooWarning marks a word that is vulgar or not appropriate
        in formal text. """

    # T001: Taboo word usage warning, with suggested replacement

    def __init__(
        self, code: str, txt: str, detail: Optional[str], original: str, suggest: str
    ) -> None:
        # Taboo word warnings start with "T"
        super().__init__(
            "T" + code, is_warning=True, original=original, suggest=suggest
        )
        self._txt = txt
        self._detail = detail

    @property
    def description(self) -> str:
        return self._txt

    @property
    def detail(self) -> Optional[str]:
        return self._detail

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["detail"] = self.detail
        return d


@register_error_class
class SpellingError(Error):

    """ A SpellingError is an erroneous word that was replaced
        by a much more likely word that exists in the dictionary. """

    # S001: Common errors picked up by unique_errors. Should be corrected
    # S002: Errors handled by spelling.py. Corrections should possibly
    #       only be suggested.
    # S003: Erroneously formed word forms picked up by ErrorForms.
    #       Should be corrected.
    # S004: Rare word, a more common one has been substituted.

    def __init__(self, code: str, txt: str, original: str, suggest: str) -> None:
        # Spelling error codes start with "S"
        super().__init__("S" + code, original=original, suggest=suggest)
        self._txt = txt

    @property
    def description(self) -> str:
        return self._txt


@register_error_class
class SpellingSuggestion(Error):

    """ A SpellingSuggestion is an annotation suggesting that
        a word might be misspelled. """

    # W001: Replacement suggested

    def __init__(self, code: str, txt: str, original: str, suggest: str) -> None:
        # Spelling suggestion codes start with "W"
        super().__init__(
            "W" + code, is_warning=True, original=original, suggest=suggest
        )
        self._txt = txt

    @property
    def description(self) -> str:
        return self._txt

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["suggest"] = self.suggest
        return d

    def does_not_match(self, terminal: VariantHandler, token: BIN_Token) -> bool:
        """ Return True if this suggestion would not work
            grammatically within the parsed sentence and should
            therefore be discarded """
        matches_original = False
        matches_suggestion = False
        # Go through the meaning list, which includes BIN_Tuple tuples
        # corresponding to both the original and the suggested word forms
        token_ref = token.lower
        assert self.suggest is not None
        suggestion_ref = self.suggest.lower()
        for m in cast(Iterable[BIN_Tuple], token.t2):
            if terminal.matches_token(token, m):
                # This meaning matches the terminal
                meaning_ref = m.ordmynd.lower()
                if meaning_ref == token_ref:
                    # This is the original text
                    matches_original = True
                elif meaning_ref == suggestion_ref:
                    # This is the suggestion
                    matches_suggestion = True
        # If the terminal matches the original word and not the
        # suggested one, return True to discard the suggestion
        return matches_original and not matches_suggestion


@register_error_class
class PhraseError(Error):

    """ A PhraseError is a wrong multiword phrase, where a word is out
        of place in its context. """

    # P_xxx: Phrase error codes

    def __init__(
        self,
        code: str,
        txt: str,
        original: str,
        suggest: str,
        span: int,
        is_warning: bool = False,
    ) -> None:
        # Phrase error codes start with "P", and are followed by
        # a string indicating the type of error, i.e. YI for y/i, etc.
        super().__init__(
            "P_" + code,
            is_warning=is_warning,
            span=span,
            original=original,
            suggest=suggest,
        )
        self._txt = txt

    @property
    def description(self) -> str:
        return self._txt


def parse_errors(
    token_stream: Iterator[Tok], db: GreynirBin, only_ci: bool
) -> Iterator[CorrectToken]:

    """ This tokenization phase is done before BÍN annotation
        and before static phrases are identified. It finds duplicated words,
        and words that have been incorrectly split or should be split. """

    def get() -> CorrectToken:
        """ Get the next token in the underlying stream and wrap it
            in a CorrectToken instance """
        return CorrectToken.from_token(next(token_stream))

    # def is_split_compound(token: Tok, next_token: Tok) -> bool:
    #     """ Check whether the combination of the given token and the next
    #         token forms a split compound. Note that the latter part of
    #         a split compound is specified as a stem (lemma), so we need
    #         to check whether the next_token word form has a corresponding
    #         lemma. Also, a single first part may have more than one (i.e.,
    #         a set of) subsequent latter part stems.
    #     """
    #     txt = token.txt
    #     next_txt = next_token.txt
    #     if txt is None or next_txt is None or is_cap(next_txt):
    #         # If the latter part is capitalized, we don't see it
    #         # as a part of split compound
    #         return False
    #     if next_txt.isupper() and not txt.isupper():
    #         # Don't allow a combination of an all-upper-case
    #         # latter part with anything but an all-upper-case former part
    #         return False
    #     if txt.isupper() and not next_txt.isupper():
    #         # ...and vice versa
    #         return False
    #     next_stems = SplitCompounds.DICT.get(txt.lower())
    #     if not next_stems:
    #         return False
    #     _, meanings = db.lookup_g(next_txt.lower(), at_sentence_start=False)
    #     if not meanings:
    #         return False
    #     # If any meaning of the following word has a stem (lemma)
    #     # that fits the second part of the split compound, we
    #     # have a match
    #     return any(m.stofn.replace("-", "") in next_stems for m in meanings)

    token = None
    at_sentence_start = False

    try:

        # Maintain a one-token lookahead
        token = get()

        while True:

            next_token = get()

            if token.kind == TOK.S_BEGIN:
                yield token
                token = next_token
                at_sentence_start = True
                continue

            # Make the lookahead checks we're interested in

            # Check wrong abbreviations
            if (
                not only_ci
                and token.kind == TOK.WORD
                and token.val
                and token.txt in WRONG_ABBREVS
                and len(token.txt) > 1
            ):
                original = token.txt
                corrected = WRONG_ABBREVS[original]
                token = CorrectToken.word(corrected, cast(BIN_TupleList, token.val))
                token.set_error(
                    AbbreviationError(
                        "001",
                        "Skammstöfunin '{0}' var leiðrétt í '{1}'".format(
                            original, corrected
                        ),
                        original=original,
                        suggest=corrected,
                    )
                )
                yield token
                token = next_token
                at_sentence_start = False
                continue

            # Check abbreviations with missing dots
            # If the missing dot leads to a word without periods that is
            # found in BÍN (token.val is truthy), it's not safe to assume
            # that it's an error.
            if (
                not token.val or "." in token.txt
            ) and token.txt in Abbreviations.WRONGDOTS:
                # Multiple periods in original, some subset missing here
                # We suggest the first alternative meaning here, out of
                # potentially multiple such meanings
                original = token.txt
                corrected = Abbreviations.WRONGDOTS[original][0]
                if (
                    corrected.endswith(".")
                    and not original.endswith(".")
                    and next_token.txt == "."
                ):
                    # If this is an abbreviation missing a period, and the
                    # correction is about adding a period to it, don't do
                    # it if the next token is a period
                    pass
                else:
                    _, token_m = db.lookup_g(original, at_sentence_start)
                    if not token_m and len(original) > 1:
                        # No meaning in BÍN: allow ourselves to correct it
                        # as an abbreviation
                        # !!! TODO: Amalgamate more than one potential correction
                        # !!! of the abbreviation (ma. -> 'meðal annars' or 'milljarðar')
                        am = Abbreviations.get_meaning(corrected)
                        if not am:
                            m = []
                        else:
                            m = list(map(BIN_Tuple._make, am))
                        token = CorrectToken.word(corrected, m)
                        token.set_error(
                            AbbreviationError(
                                "002",
                                "Skammstöfunin '{0}' var leiðrétt í '{1}'".format(
                                    original, corrected
                                ),
                                original=original,
                                suggest=corrected,
                            )
                        )
                        yield token
                        token = next_token
                        at_sentence_start = False
                        continue

            # Word duplication (note that word case must also match)
            # TODO STILLING - hér er bara samhengisháð leiðrétting
            if (
                not only_ci
                and token.txt
                and next_token.txt
                and token.txt == next_token.txt
                and token.kind == TOK.WORD
            ):
                # TODO STILLING - hér er bara uppástunga, skiptir ekki máli fyrir ósh. málrýni
                if token.txt.lower() in AllowedMultiples.SET:
                    next_token.set_error(
                        CompoundError(
                            "004/w",
                            "'{0}' er að öllum líkindum ofaukið".format(next_token.txt),
                            original=token.txt.lower(),
                            suggest="",
                        )
                    )
                    yield token
                else:
                    # Step to next token
                    next_token = CorrectToken.word(token.txt)
                    next_token.set_error(
                        CompoundError(
                            "001",
                            "Endurtekið orð ('{0}') var fellt burt".format(token.txt),
                            original=token.txt,
                            suggest="",
                        )
                    )
                token = next_token
                at_sentence_start = False
                continue

            # Word duplication with different cases
            # Only provide a suggestion
            # No need to check AllowedMultiples
            # TODO STILLING - hér er samhengisháð leiðrétting
            if (
                not only_ci
                and token.txt
                and next_token.txt
                and token.txt.lower() == next_token.txt.lower()
                and token.kind == TOK.WORD
            ):
                # Set possible error on next token
                next_token.set_error(
                    CompoundError(
                        "004",
                        "'{0}' er að öllum líkindum ofaukið".format(next_token.txt),
                        original=token.txt,
                        suggest="",
                    )
                )
                yield token
                token = next_token
                at_sentence_start = False
                continue

            if (
                token.txt
                and token.txt.endswith(("-og", "-eða"))
                and token.txt[0] != "-"
            ):
                # Coalesced word, such as 'fjármála-og'
                first, second = token.txt.rsplit("-", maxsplit=1)
                new_token = CorrectToken.word(first)
                new_token.set_error(
                    CompoundError(
                        "002",
                        "Orðinu '{0}' var skipt upp".format(token.txt),
                        original=token.txt,
                        suggest=f"{first} {second}",
                        span=2,  # TODO: Shouldn't this be 1?
                    )
                )
                yield new_token
                token = CorrectToken.word("-" + second)
                yield token
                token = next_token
                at_sentence_start = False
                continue

            # Splitting wrongly compounded words
            # TODO STILLING - hér er ósamhengisháð leiðrétting!
            if token.txt and token.txt.lower() in WrongCompounds.DICT:
                correct_phrase = list(WrongCompounds.DICT[token.txt.lower()])
                # Make the split phrase emulate the case of
                # the original token
                if token.txt.isupper():
                    # All upper case
                    for ix, p in enumerate(correct_phrase):
                        correct_phrase[ix] = p.upper()
                else:
                    # First word might be capitalized
                    correct_phrase[0] = emulate_case(
                        correct_phrase[0], template=token.txt
                    )
                for ix, phrase_part in enumerate(correct_phrase):
                    new_token = CorrectToken.word(phrase_part)
                    if ix == 0:
                        new_token.set_error(
                            CompoundError(
                                "002",
                                "Orðinu '{0}' var skipt upp".format(token.txt),
                                original=token.txt,
                                suggest=" ".join(correct_phrase),
                                span=len(correct_phrase),
                            )
                        )
                    yield new_token
                token = next_token
                at_sentence_start = False
                continue

            # TODO STILLING - hér er samhengisháð leiðrétting
            # TODO STILLING - ath. þó að e-ð af orðhlutunum í Morphemes.BOUND_DICT geta ekki staðið sjálfstæð
            # TODO STILLING - þá þarf að merkja þá orðhluta sem villu ef ósh. leiðrétting er valin.
            # Unite wrongly split compounds, or at least suggest uniting them
            if token.txt and (
                token.txt.lower() in SplitCompounds.DICT
                or token.txt.lower() in Morphemes.BOUND_DICT
            ):
                if only_ci:
                    if token.txt.lower() in SplitCompounds.DICT:
                        # Don't want to correct
                        yield token
                        token = next_token
                        at_sentence_start = False
                        continue
                    if token.txt.lower() in Morphemes.BOUND_DICT:
                        # Only want to mark as an error, can't fix in CI-mode.
                        token.set_error(
                            SpellingError(
                                "007",
                                "Orðhlutinn '{0}' á ekki að standa stakur".format(
                                    token.txt
                                ),
                                token.txt,
                                "",
                            )
                        )
                    yield token
                    token = next_token
                    at_sentence_start = False
                    continue
                if not next_token.txt or is_cap(next_token.txt):
                    # If the latter part is capitalized, we don't see it
                    # as a part of a split compound
                    yield token
                    token = next_token
                    at_sentence_start = False
                    continue
                if next_token.txt.isupper() and not token.txt.isupper():
                    # Don't allow a combination of an all-upper-case
                    # latter part with anything but an all-upper-case former part
                    yield token
                    token = next_token
                    at_sentence_start = False
                    continue
                if token.txt.isupper() and not next_token.txt.isupper():
                    # ...and vice versa
                    yield token
                    token = next_token
                    continue
                next_stems = SplitCompounds.DICT.get(token.txt.lower())
                if not next_stems:
                    yield token
                    token = next_token
                    at_sentence_start = False
                    continue
                _, meanings = db.lookup_g(
                    next_token.txt.lower(), at_sentence_start=False
                )
                if not meanings:
                    # The latter part is not in BÍN
                    yield token
                    token = next_token
                    at_sentence_start = False
                    continue
                if any(m.stofn.replace("-", "") in next_stems for m in meanings):
                    first_txt = token.txt
                    token = CorrectToken.word(token.txt + next_token.txt)
                    token.set_error(
                        CompoundError(
                            "003",
                            "Orðin '{0} {1}' voru sameinuð í eitt".format(
                                first_txt, next_token.txt
                            ),
                            original="{} {}".format(first_txt, next_token.txt),
                            suggest="{}{}".format(first_txt, next_token.txt),
                        )
                    )
                    yield token
                    token = get()
                    at_sentence_start = False
                    continue
                next_pos = Morphemes.BOUND_DICT.get(token.txt.lower())
                if not next_pos:
                    yield token
                    token = next_token
                    at_sentence_start = False
                    continue
                poses = set(m.ordfl for m in meanings if m.ordfl in next_pos)
                notposes = set(m.ordfl for m in meanings if m.ordfl not in next_pos)
                if not poses:
                    # Stop searching
                    yield token
                    token = next_token
                    at_sentence_start = False
                    continue
                if not notposes:
                    # No other PoS available, most likely a compound error
                    first_txt = token.txt
                    token = CorrectToken.word(token.txt + next_token.txt)
                    token.set_error(
                        CompoundError(
                            "003",
                            "Orðin '{0} {1}' voru sameinuð í eitt".format(
                                first_txt, next_token.txt
                            ),
                            original="{} {}".format(first_txt, next_token.txt),
                            suggest="{}{}".format(first_txt, next_token.txt),
                        )
                    )
                    yield token
                    token = get()
                    at_sentence_start = False
                    continue
                # TODO STILLING - Hér er bara uppástunga, skiptir ekki máli f. ósh. málrýni.
                # Erum búin að koma í veg fyrir að komast hingað ofar
                if poses:
                    transposes = list(set(POS[c] for c in poses))
                    if len(transposes) == 1:
                        tp = transposes[0]
                    else:
                        tp = ", ".join(transposes[:-1]) + " eða " + transposes[-1]
                    token.set_error(
                        CompoundError(
                            "005/w",
                            "Ef '{0}' er {1} á að sameina það '{2}'".format(
                                token.txt, tp, next_token.txt
                            ),
                            original="{} {}".format(token.txt, next_token.txt),
                            suggest="{}{}".format(token.txt, next_token.txt),
                        )
                    )
                    yield token
                    token = next_token
                    at_sentence_start = False
                    continue

            if (
                token.kind == TOK.PUNCTUATION
                and token.txt != cast(Tuple[str, str], token.val)[1]
            ):
                ntxt = cast(Tuple[str, str], token.val)[1]
                if ntxt in "„“":
                    # TODO Could add normalize_quotation_marks as a parameter,
                    # should normalize automatically if chosen
                    token.set_error(
                        PunctuationError(
                            "001",
                            "Gæsalappirnar {0} ættu að vera {1}".format(
                                token.txt, ntxt
                            ),
                            token.txt,
                            ntxt,
                        )
                    )
                elif ntxt == "…":
                    if token.txt == "..":
                        # Assume user meant to write a single period
                        # Doesn't happen with current tokenizer behaviour
                        token.set_error(
                            PunctuationError(
                                "002",
                                "Gæti átt að vera einn punktur (.)",
                                token.txt,
                                ".",
                            )
                        )
                    elif len(token.txt) > 3:
                        # Informal, should be standardized to an ellipsis
                        token.set_error(
                            PunctuationError(
                                "002",
                                "Óformlegt; gæti átt að vera þrípunktur (…)",
                                token.txt,
                                ntxt,
                            )
                        )
                    else:
                        # Three periods found, used as an ellipsis
                        # Not pointed out, allowed as-is
                        # TODO could add normalize_ellipsis as a parameter here
                        pass
                elif ntxt == "?!":
                    # Changed automatically, pointed out as informal
                    token.set_error(
                        PunctuationError(
                            "003",
                            "'{0}' er óformlegt, breytt í '{1}'".format(
                                token.txt, ntxt
                            ),
                            token.txt,
                            ntxt,
                        )
                    )

            # Yield the current token and advance to the lookahead
            yield token
            if token.kind != TOK.PUNCTUATION and token.kind != TOK.ORDINAL:
                at_sentence_start = False
            token = next_token

    except StopIteration:
        # Final token (previous lookahead)
        if token:
            yield token


class MultiwordErrorStream(MatchingStream):

    """ Class that filters a token stream looking for multi-word
        matches with the MultiwordErrors phrase dictionary,
        and inserting replacement phrases when matches are found """

    def __init__(self, db: GreynirBin, token_ctor: TokenCtor) -> None:
        super().__init__(MultiwordErrors.DICT)
        self._token_ctor = token_ctor
        self._db = db

    def length(self, ix: int) -> int:
        """ Return the length (word count) of the original phrase
            that is being replaced """
        return MultiwordErrors.get_phrase_length(ix)

    def match(self, tq: List[Tok], ix: int) -> Iterable[Tok]:
        """ This is a complete match of an error phrase;
            yield the replacement phrase """
        replacement = MultiwordErrors.get_replacement(ix)
        db = self._db
        token_ctor = self._token_ctor
        for i, replacement_word in enumerate(replacement):
            # !!! TODO: at_sentence_start
            _, m = db.lookup_g(replacement_word, False, False)
            if i == 0:
                # Fix capitalization of the first word
                # !!! TODO: handle all-uppercase
                if is_cap(tq[0].txt):
                    replacement_word = replacement_word.capitalize()
            ct = token_ctor.Word(replacement_word, m)
            if i == 0:
                ct.set_error(
                    PhraseError(
                        MultiwordErrors.get_code(ix),
                        "Orðasambandið '{0}' var leiðrétt í '{1}'".format(
                            " ".join(t.txt for t in tq), " ".join(replacement)
                        ),
                        span=len(replacement),
                        original=" ".join(t.txt for t in tq),
                        suggest=" ".join(replacement),
                    )
                )
            else:
                # Set the error field of multiword phrase
                # continuation tokens to True, thus avoiding
                # further meddling with them
                ct.set_error(True)
            yield ct


def handle_multiword_errors(
    token_stream: Iterator[CorrectToken], db: GreynirBin, token_ctor: TokenCtor
) -> Iterator[CorrectToken]:

    """ Parse a stream of tokens looking for multiword phrases
        containing errors.
        The algorithm implements N-token lookahead where N is the
        length of the longest phrase.
    """

    mwes = MultiwordErrorStream(db, token_ctor)
    tok_stream = cast(Iterator[Tok], token_stream)
    yield from cast(Iterator[CorrectToken], mwes.process(tok_stream))


# Compound word stuff

# Illegal prefixes that will be split off from the rest of the word
# Attn.: Make sure these errors are available as a prefix
NOT_FORMERS = frozenset(("allra", "alhliða", "fjölnota", "margnota", "ótal"))

# Tradition says these word parts should rather be used
# Using them results in a context-dependent error
# Attn.: Make sure these errors are available as a prefix
WRONG_FORMERS = {
    "akstur": "aksturs",
    "athugana": "athugunar",
    "ferminga": "fermingar",
    "fjárfestinga": "fjárfestingar",
    "forvarna": "forvarnar",
    "heyrna": "heyrnar",
    "kvartana": "kvörtunar",
    "loftlags": "loftslags",
    "næringa": "næringar",
    "pantana": "pöntunar",
    "ráðninga": "ráðningar",
    "skráninga": "skráningar",
    "Vestfjarðar": "Vestfjarða",
    "ábendinga": "ábendingar",
}

# Using these word parts results in a context-independent error
# Attn.: Make sure these errors are available as a prefix
WRONG_FORMERS_CI = {
    "akríl": "akrýl",
    "dísel": "dísil",
    "eyrnar": "eyrna",
    "feykna": "feikna",
    "fjarskiptar": "fjarskipta",
    "fyrna": "firna",
    "griðar": "griða",  # griðarstaður
    "kvenn": "kven",
    "Lundúnar": "Lundúna",
    "öldungar": "öldunga",
}


def fix_compound_words(
    token_stream: Iterable[CorrectToken],
    db: GreynirBin,
    token_ctor: TokenCtor,
    only_ci: bool,
) -> Iterator[CorrectToken]:

    """ Fix incorrectly compounded words """

    at_sentence_start = False

    for token in token_stream:
        if token.kind == TOK.S_BEGIN:
            yield token
            at_sentence_start = True
            continue
        if token.txt and token.txt.endswith("-og") and len(token.txt) > 3:
            prefix = token.txt[:-2]
            _, m = db.lookup_g(prefix, at_sentence_start)
            t1 = token_ctor.Word(prefix, m, token=token)
            t1.set_error(
                CompoundError(
                    "002",
                    "Orðinu '{0}' var skipt upp".format(token.txt),
                    original=token.txt,
                    suggest=prefix + " og",
                    span=2,  # TODO: Should this be 1?
                )
            )
            yield t1
            at_sentence_start = False
            suffix = "og"
            _, m = db.lookup_g(suffix, at_sentence_start)
            token = token_ctor.Word(suffix, m, token=token)

        if token.kind == TOK.PUNCTUATION or token.kind == TOK.ORDINAL:
            yield token
            # Don't modify at_sentence_start in this case
            continue

        if (
            token.kind != TOK.WORD
            or not token.has_meanings
            or "-" not in token.meanings[0].stofn
        ):
            # Not a compound word
            yield token
            at_sentence_start = False
            continue

        # Compound word
        cw = token.meanings[0].stofn.split("-")
        # Special case for the prefix "ótal" which the compounder
        # splits into ó-tal
        if len(cw) >= 3 and cw[0] == "ó" and cw[1] == "tal":
            cw = ["ótal"] + cw[2:]

        # TODO STILLING - hér er ósamhengisháð leiðrétting!
        if cw[0] in NOT_FORMERS:
            # Prefix is invalid as such; should be split
            # into two words
            prefix = emulate_case(cw[0], template=token.txt)
            suffix = token.txt[len(cw[0]) :]
            _, m = db.lookup_g(prefix, at_sentence_start)
            t1 = token_ctor.Word(prefix, m, token=token)
            t1.set_error(
                CompoundError(
                    "002",
                    "Orðinu '{0}' var skipt upp".format(token.txt),
                    original=token.txt,
                    suggest="{} {}".format(prefix, suffix),
                    span=2,
                )
            )
            yield t1
            at_sentence_start = False
            _, m = db.lookup_g(suffix, at_sentence_start)
            token = token_ctor.Word(suffix, m, token=token)

        # TODO STILLING - hér er ósamhengisháð leiðrétting!
        elif cw[0] in Morphemes.FREE_DICT:
            # Check which PoS, attachment depends on that
            at_sentence_start = False
            suffix = token.txt[len(cw[0]) :]
            prefix = emulate_case(cw[0], template=token.txt)
            freepos = Morphemes.FREE_DICT.get(cw[0])
            assert freepos is not None
            _, meanings2 = db.lookup_g(suffix, at_sentence_start)
            poses = set(m.ordfl for m in meanings2 if m.ordfl in freepos)
            if not poses:
                yield token
                continue
            notposes = set(m.ordfl for m in meanings2 if m.ordfl not in freepos)
            if not notposes:
                # No other PoS available, we found an error
                _, meanings1 = db.lookup_g(prefix, at_sentence_start)
                t1 = token_ctor.Word(prefix, meanings1, token=token)
                t1.set_error(
                    CompoundError(
                        "002",
                        "Orðinu '{0}' var skipt upp".format(token.txt),
                        original=token.txt,
                        suggest="{} {}".format(prefix, suffix),
                        span=2,
                    )
                )
                yield t1
                token = token_ctor.Word(suffix, meanings2, token=token)
            else:
                # TODO STILLING - hér er bara uppástunga.
                # Other possibilities but want to mark as a possible error
                # Often just weird forms in BÍN left
                if not only_ci:
                    transposes = list(set(POS[c] for c in poses))
                    if len(transposes) == 1:
                        tp = transposes[0]
                    else:
                        tp = ", ".join(transposes[:-1]) + " eða " + transposes[-1]
                    token.set_error(
                        CompoundError(
                            "005",
                            "Ef '{0}' er {1} á að skipta orðinu upp".format(
                                token.txt, tp
                            ),
                            original=token.txt,
                            suggest="{} {}".format(prefix, suffix),
                            span=2,
                        )
                    )

        # TODO STILLING - hér er ósamhengisháð leiðrétting, en það er spurning hvort allt hér teljist endilega villa.
        # TODO STILLING - viljum ekki endilega leiðrétta "byggingaregla", þó að venjan leyfi hitt frekar.
        # TODO STILLING - Þarf að fara í gegnum WRONG_FORMERS, mætti skipta upp í
        # TODO STILLING - ALWAYS_WRONG_FORMERS og MOSTLY_WRONG_FORMERS eða eitthvað þannig?
        # TODO STILLING - fyrra alltaf leiðrétt, en seinna bara ábending?

        # TODO STILLING - Athuga hvort hér ætti að hafa ólík villuskilaboð fyrir WRONG_FORMERS og WRONG_FORMERS_CI?
        elif cw[0] in WRONG_FORMERS_CI:
            correct_former = WRONG_FORMERS_CI[cw[0]]
            corrected = correct_former + token.txt[len(cw[0]) :]
            corrected = emulate_case(corrected, template=token.txt)
            _, m = db.lookup_g(corrected, at_sentence_start)
            t1 = token_ctor.Word(corrected, m, token=token)
            t1.set_error(
                CompoundError(
                    "006",
                    "Samsetta orðinu '{0}' var breytt í '{1}'".format(
                        token.txt, corrected
                    ),
                    original=token.txt,
                    suggest=corrected,
                )
            )
            token = t1

        elif not only_ci and cw[0] in WRONG_FORMERS:
            # Splice a correct front onto the word
            # ('feyknaglaður' -> 'feiknaglaður')
            correct_former = WRONG_FORMERS[cw[0]]
            corrected = correct_former + token.txt[len(cw[0]) :]
            corrected = emulate_case(corrected, template=token.txt)
            _, m = db.lookup_g(corrected, at_sentence_start)
            t1 = token_ctor.Word(corrected, m, token=token)
            t1.set_error(
                CompoundError(
                    "006",
                    "Samsetta orðinu '{0}' var breytt í '{1}'".format(
                        token.txt, corrected
                    ),
                    original=token.txt,
                    suggest=corrected,
                )
            )
            token = t1
        # TODO Bæta inn leiðréttingu út frá seinni orðhlutum?
        yield token
        at_sentence_start = False


def lookup_unknown_words(
    corrector: Corrector,
    token_ctor: TokenCtor,
    token_stream: Iterable[CorrectToken],
    only_ci: bool,
    apply_suggestions: bool,
) -> Iterator[CorrectToken]:

    """ Try to identify unknown words in the token stream, for instance
        as spelling errors (character juxtaposition, deletion, insertion...) """

    at_sentence_start = False
    context: Tuple[str, ...] = tuple()
    db = corrector.db
    # When entering parentheses, we push dict(closing=")", prefix=""),
    # where closing means the corresponding closing symbol (")", "]")
    # and prefix is the starting token within the parenthesis, if any,
    # such as "e." for "English"
    PARENS = {"(": ")", "[": "]", "{": "}"}
    parenthesis_stack: List[Dict[str, str]] = []

    def is_immune(token: CorrectToken) -> bool:
        """ Return True if the token should definitely not be
            corrected """
        if len(token.meanings) == 1 and token.meanings[0].beyging == "-":
            # This is probably an abbreviation, having a single meaning
            # and no declension information
            return True
        if token.txt.isupper():
            # Should not correct all uppercase words
            return True
        return False

    def replace_word(
        code: int, token: CorrectToken, corrected: str, corrected_display: Optional[str]
    ) -> CorrectToken:

        """ Return a token for a corrected version of token_txt,
            marked with a SpellingError if corrected_display is
            a string containing the corrected word to be displayed """

        _, m = db.lookup_g(corrected, at_sentence_start)
        ct = token_ctor.Word(corrected, m, token=token if corrected_display else None)
        if corrected_display:
            if "." in corrected_display:
                text = "Skammstöfunin '{0}' var leiðrétt í '{1}'".format(
                    token.txt, corrected_display
                )
            else:
                text = "Orðið '{0}' var leiðrétt í '{1}'".format(
                    token.txt, corrected_display
                )
            ct.set_error(
                SpellingError("{0:03}".format(code), text, token.txt, corrected_display)
            )
        else:
            # In a multi-word sequence, mark the replacement
            # tokens with a boolean value so that further
            # replacements will not be made
            ct.set_error(True)
        return ct

    def correct_word(
        code: int, token: CorrectToken, corrected: str, m: Sequence[BIN_Tuple]
    ) -> CorrectToken:
        """ Return a token for a corrected version of token_txt,
            marked with a SpellingError if corrected_display is
            a string containing the corrected word to be displayed """
        ct = token_ctor.Word(corrected, m, token=token)
        if "." in corrected:
            text = "Skammstöfunin '{0}' var leiðrétt í '{1}'".format(
                token.txt, corrected
            )
        else:
            text = "Orðið '{0}' var leiðrétt í '{1}'".format(token.txt, corrected)
        ct.set_error(SpellingError("{0:03}".format(code), text, token.txt, corrected))
        return ct

    def suggest_word(
        code: int, token: CorrectToken, corrected: str, m: Sequence[BIN_Tuple]
    ) -> CorrectToken:
        """ Mark the current token with an annotation but don't correct
            it, as we are not confident enough of the correction """
        text = "Orðið '{0}' gæti átt að vera '{1}'".format(token.txt, corrected)
        token.set_error(
            SpellingSuggestion("{0:03}".format(code), text, token.txt, corrected)
        )
        # Add the meanings of the potential correction to the token
        token.add_corrected_meanings(m)
        return token

    def only_suggest(token: CorrectToken, m: Sequence[BIN_Tuple]) -> bool:
        """ Return True if we don't have high confidence in the proposed
            correction, so it will be suggested instead of applied """
        if 2 <= len(token.txt) <= 4 and token.txt.isupper():
            # This is a 2 to 4-letter all-uppercase word:
            # it may be an unknown abbreviation, which we should not force-correct
            return True
        if set(token.txt.lower()) & NON_ICELANDIC_LETTERS_SET:
            # If the original word contains non-Icelandic letters, it is
            # probably a foreign word and in that case we only make a suggestion
            return True
        if not token.has_meanings:
            # The original word is not in BÍN, so we can
            # confidently apply the correction
            return False
        if "-" not in token.meanings[0].stofn:
            # The original word is in BÍN and not a compound word:
            # only suggest a correction
            return True
        # The word is a compound word: only suggest the correction if it is compound
        # too (or if no meaning is found for it in BÍN, which is an unlikely case)
        return (not m) or "-" in m[0].stofn

    for token in token_stream:
        if token.kind == TOK.S_BEGIN:
            yield token
            # A new sentence is starting
            at_sentence_start = True
            context = tuple()
            parenthesis_stack = []
            continue
        # Store the previous context in case we need to construct
        # a new current context (after token substitution)
        prev_context = context
        if token.txt:
            # Maintain a context trigram, ending with the current token
            context = (prev_context + tuple(token.txt.split()))[-3:]
        if token.kind == TOK.PUNCTUATION or token.kind == TOK.ORDINAL:
            # Manage the parenthesis stack
            if token.txt in PARENS:
                # Opening a new scope
                parenthesis_stack.append(dict(closing=PARENS[token.txt]))
            elif (
                bool(parenthesis_stack)
                and token.txt == parenthesis_stack[-1]["closing"]
            ):
                # Closing a scope
                parenthesis_stack.pop()
            # Don't modify at_sentence_start in this case
            yield token
            continue

        if token.kind != TOK.WORD or " " in token.txt:
            # No need to look at non-word tokens,
            # and we don't process multi-word composites
            yield token
            # We're now within a sentence
            at_sentence_start = False
            continue

        # The token is a word

        # Check unique errors - some of those may have
        # BÍN annotations via the compounder
        # Examples: 'kvenær' -> 'hvenær', 'starfssemi' -> 'starfsemi'
        # !!! TODO: Handle upper/lowercase
        # TODO STILLING - hér er ósamhengisháð leiðrétting!
        if token.txt in UniqueErrors.DICT:
            # Note: corrected is a tuple
            corrected = UniqueErrors.DICT[token.txt]
            assert isinstance(corrected, tuple)
            corrected_display = " ".join(corrected)
            for ix, corrected_word in enumerate(corrected):
                if ix == 0:
                    rtok = replace_word(1, token, corrected_word, corrected_display)
                    at_sentence_start = False
                else:
                    # In a multi-word sequence, we only mark the first
                    # token with a SpellingError
                    rtok = replace_word(1, token, corrected_word, None)
                yield rtok
                context = (prev_context + tuple(rtok.txt.split()))[-3:]
                prev_context = context
            continue

        if token.error_code:
            # This token already has an associated error and eventual correction:
            # let it be
            yield token
            # We're now within a sentence
            at_sentence_start = False
            continue

        # Check wrong word forms, i.e. those that do not exist in BÍN
        # !!! TODO: Some error forms are present in BÍN but in a different
        # !!! TODO: case (for instance, 'á' as a nominative of 'ær').
        # !!! TODO: We are not handling those here.
        # !!! TODO: Handle upper/lowercase
        # TODO STILLING - hér er ósamhengisháð leiðrétting!
        if not token.val and CIDErrorForms.contains(token.txt):
            corr_txt = CIDErrorForms.get_correct_form(token.txt)
            rtok = replace_word(2, token, corr_txt, corr_txt)
            at_sentence_start = False
            # Update the context with the replaced token
            context = (prev_context + tuple(rtok.txt.split()))[-3:]
            yield rtok
            continue

        if is_immune(token) or token.error:
            # Nothing more to do
            pass

        # Check rare (or nonexistent) words and see if we have a potential correction
        # TODO STILLING - hér er samhengisháð leiðrétting af því að við notum þrenndir!
        # TODO STILLING - og líka því skoðum líka sjaldgæf orð.
        elif not token.val or corrector.is_rare(token.txt):
            # Yes, this is a rare word that needs further attention
            if only_ci and token.txt[0].islower():
                # Don't want to correct
                # Don't want to tag capitalized words, mostly foreign entities
                token.set_error(
                    UnknownWordError(
                        code="001",
                        txt="Óþekkt orð: '{0}'".format(token.txt),
                        original=token.txt,
                        suggest="",
                    )
                )
                yield token
                at_sentence_start = False
                continue
            # TODO Consider limiting to words under 15 characters
            if Settings.DEBUG:
                print("Checking rare word '{0}'".format(token.txt))
            # We use context[-3:-1] since the current token is the last item
            # in the context tuple, and we want the bigram preceding it.
            corrected_txt = corrector.correct(
                token.txt, context=tuple(context[-3:-1]), at_sentence_start=at_sentence_start
            )
            if corrected_txt != token.txt:
                # We have a candidate correction: take a closer look at it
                _, m = db.lookup_g(corrected_txt, at_sentence_start=at_sentence_start)
                if (token.txt[0].lower() == "ó" and corrected_txt == token.txt[1:]) or (
                        corrected_txt[0].lower() == "ó" and token.txt == corrected_txt[1:]
                    ):
                    # The correction simply removed or added "ó" at the start of the
                    # word: probably not a good idea
                    pass
                elif token.txt[0] == "-" and corrected_txt == token.txt[1:]:
                    # The correction simply removed "-" from the start of the
                    # word: probably not a good idea
                    pass
                elif not m and token.txt[0].isupper():
                    # Don't correct uppercase words if the suggested correction
                    # is not in BÍN
                    pass
                elif "-" in token.txt and (
                    token.txt.lower() == corrected_txt.lower() or " " in token.txt
                ):
                    # Don't correct PCR-próf to Pcr-próf,
                    # or félags- og barnamálaráðherra to félags- og varnamálaráðherra
                    pass
                elif len(token.txt) == 1 and (
                    corrected_txt
                    != SINGLE_LETTER_CORRECTIONS.get((at_sentence_start, token.txt))
                ):
                    # Only allow single-letter corrections of a->á and i->í
                    pass
                # TODO STILLING - þetta er bara uppástunga
                elif not apply_suggestions and only_suggest(token, m):
                    # We have a candidate correction but the original word does
                    # exist in BÍN, so we're not super confident: yield a suggestion
                    if Settings.DEBUG:
                        print(
                            "Suggested '{1}' instead of '{0}'".format(
                                token.txt, corrected_txt
                            )
                        )
                    yield suggest_word(1, token, corrected_txt, m)
                    # We do not update the context in this case
                    at_sentence_start = False
                    continue
                else:
                    # We have a better candidate and are confident that
                    # it should replace the original word: yield it
                    if Settings.DEBUG:
                        print(
                            "Corrected '{0}' to '{1}'".format(token.txt, corrected_txt)
                        )
                    ctok = correct_word(4, token, corrected_txt, m)
                    yield ctok
                    # Update the context with the corrected token
                    context = (prev_context + tuple(ctok.txt.split()))[-3:]
                    at_sentence_start = False
                    continue

        # Check for completely unknown and uncorrectable words
        # TODO STILLING - hér er ósamhengisháð leiðrétting!
        if not token.val:
            # No annotation and not able to correct:
            # mark the token as an unknown word
            # (but only as a warning if it is an uppercase word or
            # if we're within parentheses)
            token.set_error(
                UnknownWordError(
                    code="001",
                    txt="Óþekkt orð: '{0}'".format(token.txt),
                    original=token.txt,
                    suggest="",
                    is_warning=token.txt[0].isupper() or bool(parenthesis_stack),
                )
            )

        if token.error is True:
            # Erase the boolean error continuation marker:
            # we no longer need it
            token.set_error(None)

        yield token
        at_sentence_start = False


def fix_capitalization(
    token_stream: Iterable[CorrectToken],
    db: GreynirBin,
    token_ctor: TokenCtor,
    only_ci: bool,
) -> Iterator[CorrectToken]:

    """ Annotate tokens with errors if they are capitalized incorrectly """

    stems = CapitalizationErrors.SET_REV
    wrong_stems = CapitalizationErrors.SET

    # TODO STILLING - hér er blanda. Orð sem eiga alltaf að vera hástafa en birtast lágstafa eru ósh.,
    # TODO STILLING - orð sem eiga alltaf að vera lágstafa nema í byrjun setningar eru sh. leiðrétting.

    # This variable must be defined before is_wrong() because
    # the function closes over it
    # The states are ("sentence_start", "after_ordinal", "in_sentence")
    state = "sentence_start"

    def is_wrong(token: CorrectToken) -> bool:
        """ Return True if the token's text is wrongly capitalized """
        word = token.txt
        lower = True
        if is_cap(word):
            if state != "in_sentence":
                # An uppercase word at the beginning of a sentence can't be wrong
                return False
            if word in WRONG_ACRONYMS:
                return True
            if token.val and any(
                is_cap(m.stofn) and "-" not in m.stofn
                for m in cast(Iterable[BIN_Tuple], token.val)
            ):
                # The token has a proper BÍN meaning as-is, i.e. upper case:
                # it's probably correct
                return False
            # Danskur -> danskur
            rev_word = word.lower()
            lower = False
        elif word.islower():
            if state == "sentence_start":
                # A lower case word at the beginning of a sentence is definitely wrong
                return True
            if len(word) >= 3 and word[1] == "-":
                if word[0] in "abcdefghijklmnopqrstuvwxyz":
                    # Something like 'b-deildin' or 'a-flokki' which should probably
                    # be 'B-deildin' or 'A-flokki'
                    return True
            if "-" in word:
                # norður-kórea -> Norður-Kórea
                rev_word = "-".join(part.capitalize() for part in word.split("-"))
            else:
                # íslendingur -> Íslendingur
                # finni -> Finni
                rev_word = word.capitalize()
        else:
            # All upper case or other strange/mixed capitalization:
            # don't bother
            return False
        rev_meanings = db.meanings(rev_word) or []
        # If this is a word without BÍN meanings ('ástralía') but
        # an reversed-case version is in BÍN (without being a compound),
        # consider that an error. Also, we don't correct to a
        # (fist) person name in this way; that would be too broad, and,
        # besides, the current person name logic in bintokenizer.py
        # would erase the error annotation on the token.
        if lower and any(
            "-" not in m.stofn and m.fl not in {"ism", "erm"} for m in rev_meanings
        ):
            # We find reversed-case meanings in BÍN: do a further check
            if all(
                m.stofn.islower() != lower or "-" in m.stofn
                for m in cast(Iterable[BIN_Tuple], token.val)
            ):
                # If the word has no non-composite meanings
                # in its original case, this is probably an error
                return True
        # If we find any of the 'wrong' capitalizations in the error set,
        # this is definitely an error
        if any(
            emulate_case(m.stofn, template=word) in wrong_stems for m in rev_meanings
        ):
            return True
        # If we don't find any of the stems of the "corrected"
        # meanings in the corrected error set (SET_REV),
        # the word was correctly capitalized
        if all(m.stofn not in stems for m in rev_meanings):
            return False
        # Potentially wrong, but check for a corner
        # case: the original word may exist in its
        # original case in a non-noun/adjective category,
        # such as "finni" and "finna" as a verb
        tval = cast(Iterable[BIN_Tuple], token.val)
        if lower and any(m.ordfl not in {"kk", "kvk", "hk", "lo"} for m in tval):
            # Not definitely wrong
            return False
        # Definitely wrong
        return True

    for token in token_stream:
        if token.kind == TOK.S_BEGIN or token.kind == TOK.P_BEGIN:
            token.set_capitalization(state)
            yield token
            state = "sentence_start"
            continue
        # !!! TODO: Consider whether to overwrite previous error,
        # !!! if token.error is not None
        if token.kind in {TOK.WORD, TOK.PERSON, TOK.ENTITY}:
            if is_wrong(token):
                if token.txt.islower():
                    # Token is lowercase but should be capitalized
                    original_txt = token.txt
                    # !!! TODO: Maybe the following should be just token.txt.capitalize()
                    if "-" in token.txt:
                        correct = "-".join(
                            part.capitalize() for part in token.txt.split("-")
                        )
                    elif " " in token.txt:
                        correct = token.txt.title()
                    else:
                        correct = token.txt.capitalize()
                    _, m = db.lookup_g(correct, True)
                    token = token_ctor.Word(correct, m, token=token)
                    token.set_error(
                        CapitalizationError(
                            "002",
                            "Orð á að byrja á hástaf: '{0}'".format(original_txt),
                            original=original_txt,
                            suggest=correct,
                        )
                    )
                elif token.txt in WRONG_ACRONYMS:
                    original_txt = token.txt
                    correct = token.txt.upper()
                    _, m = db.lookup_g(correct, False)
                    token = token_ctor.Word(correct, m, token=token)
                    token.set_error(
                        CapitalizationError(
                            "006",
                            "Hánefni á að samanstanda af hástöfum: '{0}'".format(
                                original_txt
                            ),
                            original=original_txt,
                            suggest=original_txt.upper(),
                        )
                    )
                else:
                    # Token is capitalized but should be lower case
                    original_txt = token.txt
                    correct = token.txt.lower()
                    _, m = db.lookup_g(correct, False)
                    token = token_ctor.Word(correct, m, token=token)
                    token.set_error(
                        CapitalizationError(
                            "001",
                            "Orð á að byrja á lágstaf: '{0}'".format(original_txt),
                            original=original_txt,
                            suggest=token.txt.lower(),
                        )
                    )
        elif token.kind in {TOK.DATEREL, TOK.DATEABS}:
            if any(m in token.txt for m in MONTH_NAMES_CAPITALIZED):
                # There is a capitalized month name within the token: suspicious
                if token.txt.upper() == token.txt:
                    # ALL-CAPS: don't worry about it
                    pass
                elif state == "sentence_start" and token.txt.startswith(
                    MONTH_NAMES_CAPITALIZED
                ):
                    # At the sentence start, it's OK to have a capitalized month name.
                    # Note that after an ordinal (state='after_ordinal'), a capitalized
                    # month name is not allowed.
                    pass
                else:
                    # Wrong capitalization of month name: replace it
                    if state == "sentence_start":
                        lower = token.txt.capitalize()
                    else:
                        lower = token.txt.lower()
                    original_txt = token.txt
                    tval = cast(Tuple[int, int, int], token.val)
                    if token.kind == TOK.DATEREL:
                        token = token_ctor.Daterel(lower, tval[0], tval[1], tval[2])
                    else:
                        assert token.kind == TOK.DATEABS
                        token = token_ctor.Dateabs(lower, tval[0], tval[1], tval[2])
                    token.set_error(
                        CapitalizationError(
                            "003",
                            "Í dagsetningunni '{0}' á mánaðarnafnið "
                            "að byrja á lágstaf".format(original_txt),
                            original=original_txt,
                            suggest=original_txt.lower(),
                        )
                    )

        token.set_capitalization(state)
        yield token
        if state == "sentence_start" and token.kind == TOK.ORDINAL:
            # Special state if we've only seen ordinals at the start
            # of a sentence. In this state, both upper and lower case
            # words are allowed.
            state = "after_ordinal"
        elif token.kind != TOK.PUNCTUATION:
            # Punctuation is not enough to change the state, but
            # all other tokens do change it to in_sentence
            state = "in_sentence"
        elif token.kind == TOK.PUNCTUATION and token.txt == ":":
            # Assume we're back at sentence start after a colon
            state = "sentence_start"


def late_fix_capitalization(
    token_stream: Iterable[CorrectToken],
    db: GreynirBin,
    token_ctor: TokenCtor,
    only_ci: bool,
) -> Iterator[CorrectToken]:

    """ Annotate final, coalesced tokens with errors
        if they are capitalized incorrectly """

    def number_error(
        token: CorrectToken, replace: str, code: str, instruction_txt: str
    ) -> CorrectToken:
        """ Mark a number token with a capitalization error """
        original_txt = token.txt
        num, cases, genders = cast(NumberTuple, token.val)
        token = token_ctor.Number(replace, num, cases, genders)
        token.set_error(
            CapitalizationError(
                code,
                "Töluna eða fjárhæðina '{0}' á að rita {1}".format(
                    original_txt, instruction_txt
                ),
                original=original_txt,
                suggest=replace,
            )
        )
        return token

    at_sentence_start = False
    stems = CapitalizationErrors.SET

    for token in token_stream:
        if token.kind == TOK.S_BEGIN:
            yield token
            at_sentence_start = True
            continue
        if token.kind == TOK.WORD:
            if token.cap_in_sentence and " " in token.txt:
                # Special check for compounds such as 'félags- og barnamálaráðherra'
                # that were not checked in fix_capitalization because compounds hadn't
                # been amalgamated at that point
                if any(m.stofn in stems for m in token.meanings):
                    if token.txt[0].isupper():
                        code = "001"
                        case = "lág"
                        correct = token.txt.lower()
                    else:
                        code = "002"
                        case = "há"
                        correct = token.txt.capitalize()
                    _, m = db.lookup_g(correct, True)
                    token = token_ctor.Word(correct, m, token=token)
                    token.set_error(
                        CapitalizationError(
                            code,
                            "Rita á '{0}' með {1}staf".format(token.txt, case),
                            original=token.txt,
                            suggest=correct,
                        )
                    )
        elif token.kind == TOK.NUMBER:
            if re.match(r"[0-9.,/-]+$", token.txt) or token.txt.isupper():
                # '1.234,56' or '3/4' or "-6" or '24 MILLJÓNIR' is always OK
                pass
            elif at_sentence_start:
                if token.txt[0].isnumeric() and not token.txt[1:].islower():
                    # '500 Milljónir' at sentence start
                    token = number_error(
                        token, token.txt.lower(), "004", "með lágstöfum"
                    )
                elif token.txt[0].isupper() and not token.txt[1:].islower():
                    # 'Fimm Hundruð milljónir' at sentence start
                    token = number_error(
                        token,
                        token.txt.capitalize(),
                        "005",
                        "með hástaf aðeins í upphafi",
                    )
                elif token.txt[0].islower():
                    # 'fimm hundruð milljónir' at sentence start
                    token = number_error(
                        token, token.txt.capitalize(), "006", "með hástaf"
                    )
            elif token.txt.islower():
                # All lower case: OK
                pass
            else:
                # Mixed case: something strange going on
                token = number_error(token, token.txt.lower(), "004", "með lágstöfum")
        elif token.kind == TOK.AMOUNT:
            if token.txt.islower() or token.txt.isupper():
                # All lower case or ALL-CAPS: don't worry about it
                pass
            else:
                # Mixed case: something strange going on
                original_txt = token.txt
                lower = token.txt.lower()
                # token.val tuple: (n, iso, cases, genders)
                tval2 = cast(Tuple[float, str, Any, Any], token.val)
                token = token_ctor.Amount(lower, tval2[1], tval2[0], tval2[2], tval2[3])
                token.set_error(
                    CapitalizationError(
                        "005",
                        "Fjárhæðina '{0}' á að rita "
                        "með lágstöfum".format(original_txt),
                        original=original_txt,
                        suggest=lower,
                        is_warning=True,
                    )
                )
        elif token.kind == TOK.MEASUREMENT:
            # !!! TODO
            pass
        yield token
        if token.kind != TOK.PUNCTUATION and token.kind != TOK.ORDINAL:
            # !!! TODO: This may need to be made more intelligent
            at_sentence_start = False


def check_taboo_words(token_stream: Iterable[CorrectToken]) -> Iterator[CorrectToken]:
    """ Annotate taboo words with warnings """

    tdict = TabooWords.DICT

    for token in token_stream:
        # TODO STILLING - hér er ósamhengisháð leiðrétting EN er bara uppástunga.
        # Check taboo words
        if token.has_meanings:
            # !!! TODO: This could be made more efficient if all
            # !!! TODO: taboo word forms could be generated ahead of time
            # !!! TODO: and checked via a set lookup
            for m in token.meanings:
                key = m.stofn.replace("-", "")
                # First, look up the lemma + _ + word category
                t = tdict.get(key + "_" + m.ordfl)
                if t is None:
                    # Then, look up the lemma only
                    t = tdict.get(key)
                if t is not None:
                    # Taboo word
                    replacement, detail = t
                    # There can be multiple suggested replacements,
                    # for instance 'þungunarrof_hk/meðgöngurof_hk'
                    sw = replacement.split("/")
                    suggestion = ""
                    if len(sw) == 1 and sw[0].split("_")[0] == key:
                        # We have a single suggested word, which is the same as the
                        # taboo word: there is no suggestion, only a notification
                        explanation = "Óheppilegt eða óviðurkvæmilegt orð"
                    else:
                        suggestion = ", ".join(f"'{w.split('_')[0]}'" for w in sw)
                        # Trick to replace the last ", " with " eða ":
                        # replace the first " ," with " aðe " in a reversed string,
                        # then re-reverse it
                        suggestion = suggestion[::-1].replace(" ,", " aðe ", 1)[::-1]
                        explanation = (
                            f"Óheppilegt eða óviðurkvæmilegt orð, "
                            f"skárra væri t.d. {suggestion}"
                        )
                    token.set_error(
                        TabooWarning(
                            "001",
                            explanation,
                            detail or None,
                            token.txt,
                            ", ".join(f"'{w.split('_')[0]}'" for w in sw),
                        )
                    )
                    # !!! TODO: Add correctly inflected suggestion here
                    break

        yield token


class Correct_TOK(TOK):

    """ A derived class to override token construction methods
        as required to generate CorrectToken instances instead of
        tokenizer.TOK instances """

    @staticmethod
    def Word(
        t: Union[Tok, str], m: Optional[BIN_TupleList] = None, token: Optional[CorrectToken] = None
    ) -> CorrectToken:
        """ Override the TOK.Word constructor to create a CorrectToken instance """
        assert isinstance(t, str)
        ct = CorrectToken.word(t, m)
        if token is not None:
            # This token is being constructed in reference to a previously
            # generated token, or a list of tokens, which might have had
            # an associated error: make sure that it is preserved
            ct.copy_error(token)
            ct.copy_capitalization(token)
        return ct

    @staticmethod
    def Number(
        t: Union[Tok, str],
        n: float,
        cases: Optional[List[str]] = None,
        genders: Optional[List[str]] = None,
        token: Optional[CorrectToken] = None,
    ) -> CorrectToken:
        """ Override the TOK.Number constructor to create a CorrectToken instance """
        assert isinstance(t, str)
        ct = CorrectToken(TOK.NUMBER, t, (n, cases, genders))
        if token is not None:
            # This token is being constructed in reference to a previously
            # generated token, or a list of tokens, which might have had
            # an associated error: make sure that it is preserved
            ct.copy_error(token, coalesce=True)
        return ct

    @staticmethod
    def Amount(
        t: Union[Tok, str],
        iso: str,
        n: float,
        cases: Optional[List[str]] = None,
        genders: Optional[List[str]] = None,
        token: Optional[CorrectToken] = None,
    ) -> CorrectToken:
        """ Override the TOK.Amount constructor to create a CorrectToken instance """
        assert isinstance(t, str)
        ct = CorrectToken(TOK.AMOUNT, t, (n, iso, cases, genders))
        if token is not None:
            # This token is being constructed in reference to a previously
            # generated token, or a list of tokens, which might have had
            # an associated error: make sure that it is preserved
            ct.copy_error(token, coalesce=True)
        return ct

    @staticmethod
    def Person(
        t: Union[Tok, str], m: Optional[PersonNameList] = None, token: Optional[CorrectToken] = None
    ) -> CorrectToken:
        """ Override the TOK.Person constructor to create a CorrectToken instance """
        assert isinstance(t, str)
        ct = CorrectToken(TOK.PERSON, t, m)
        if token is not None:
            # This token is being constructed in reference to a previously
            # generated token, or a list of tokens, which might have had
            # an associated error: make sure that it is preserved
            ct.copy_error(token)
        return ct

    @staticmethod
    def Entity(t: Union[Tok, str], token: Optional[CorrectToken] = None) -> CorrectToken:
        """ Override the TOK.Entity constructor to create a CorrectToken instance """
        assert isinstance(t, str)
        ct = CorrectToken(TOK.ENTITY, t, None)
        if token is not None:
            # This token is being constructed in reference to a previously
            # generated token, or a list of tokens, which might have had
            # an associated error: make sure that it is preserved
            ct.copy_error(token)
        return ct

    @staticmethod
    def Dateabs(
        t: Union[Tok, str], y: int, m: int, d: int, token: Optional[CorrectToken] = None
    ) -> CorrectToken:
        """ Override the TOK.Dateabs constructor to create a CorrectToken instance """
        assert isinstance(t, str)
        ct = CorrectToken(TOK.DATEABS, t, (y, m, d))
        if token is not None:
            # This token is being constructed in reference to a previously
            # generated token, or a list of tokens, which might have had
            # an associated error: make sure that it is preserved
            ct.copy_error(token)
        return ct

    @staticmethod
    def Daterel(
        t: Union[Tok, str], y: int, m: int, d: int, token: Optional[CorrectToken] = None
    ) -> CorrectToken:
        """ Override the TOK.Daterel constructor to create a CorrectToken instance """
        assert isinstance(t, str)
        ct = CorrectToken(TOK.DATEREL, t, (y, m, d))
        if token is not None:
            # This token is being constructed in reference to a previously
            # generated token, or a list of tokens, which might have had
            # an associated error: make sure that it is preserved
            ct.copy_error(token)
        return ct


class CorrectionPipeline(DefaultPipeline):

    """ Override the default tokenization pipeline defined in bintokenizer.py
        in GreynirPackage, adding a correction phase """

    # Use the Correct_TOK class to construct tokens, instead of
    # TOK (tokenizer.py) or Bin_TOK (bintokenizer.py)
    _token_ctor: TokenConstructor = cast(TokenConstructor, Correct_TOK)

    def __init__(self, text_or_gen: StringIterable, **options: Any) -> None:
        super().__init__(text_or_gen, **options)
        self._corrector: Optional[Corrector] = None
        # If only_ci is True, we only correct context-independent errors
        self._only_ci = options.pop("only_ci", False)
        # If apply_suggestions is True, we are aggressive in modifying
        # tokens with suggested corrections, i.e. not just suggesting them
        self._apply_suggestions = options.pop("apply_suggestions", False)

    def correct_tokens(self, stream: TokenIterator) -> TokenIterator:
        """ Add a correction pass just before BÍN annotation """
        assert self._db is not None
        return parse_errors(stream, self._db, self._only_ci)

    def check_spelling(self, stream: TokenIterator) -> TokenIterator:
        """ Attempt to resolve unknown words """
        # Create a Corrector on the first invocation
        assert self._db is not None
        if self._corrector is None:
            self._corrector = Corrector(self._db)
        only_ci = self._only_ci
        # Shenanigans to satisfy mypy
        token_ctor = cast(TokenCtor, self._token_ctor)
        ct_stream = cast(Iterator[CorrectToken], stream)
        # Fix compound words
        ct_stream = fix_compound_words(ct_stream, self._db, token_ctor, only_ci)
        # Fix multiword error phrases
        if not only_ci:
            ct_stream = handle_multiword_errors(ct_stream, self._db, token_ctor)
        # Fix capitalization
        ct_stream = fix_capitalization(ct_stream, self._db, token_ctor, only_ci)
        # Fix single-word errors
        ct_stream = lookup_unknown_words(
            self._corrector, token_ctor, ct_stream, only_ci, self._apply_suggestions
        )
        # Check taboo words
        if not only_ci:
            ct_stream = check_taboo_words(ct_stream)
        return ct_stream

    def final_correct(self, stream: TokenIterator) -> TokenIterator:
        """ Final correction pass """
        assert self._db is not None
        # Fix capitalization of final, coalesced tokens, such
        # as numbers ('24 Milljónir') and amounts ('3 Þúsund Dollarar')
        ct_stream = cast(Iterator[CorrectToken], stream)
        token_ctor = cast(TokenCtor, self._token_ctor)
        return cast(
            TokenIterator,
            late_fix_capitalization(
                ct_stream, self._db, token_ctor, self._only_ci
            ),
        )


def tokenize(text_or_gen: StringIterable, **options: Any) -> Iterator[CorrectToken]:
    """ Tokenize text using the correction pipeline,
        overriding a part of the default tokenization pipeline """
    pipeline = CorrectionPipeline(text_or_gen, **options)
    return cast(Iterator[CorrectToken], pipeline.tokenize())
