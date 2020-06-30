"""

    Greynir: Natural language processing for Icelandic

    Annotation class definition

    Copyright (C) 2020 Miðeind ehf.

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


    This module defines the Annotation class, containing information
    about an error or a warning that applies to a span of tokens within
    a source sentence.

"""


class Annotation:

    """ An annotation of a span of a token list for a sentence """

    def __init__(
        self, *, start, end, code, text, detail=None, suggest=None, is_warning=False
    ):
        assert isinstance(start, int)
        assert isinstance(end, int)
        self._start = start
        self._end = end
        if is_warning and not code.endswith("/w"):
            code += "/w"
        self._code = code
        # text is a short, straight-to-the-point human-readable description
        # of the error
        self._text = text
        # detail is a more detailed human-readable description of the error,
        # containing further explanations, eventually using grammatical terms,
        # and possibly links to further reference material (within <a>...</a> tags)
        self._detail = detail
        # If suggest is given, it is a suggested correction,
        # i.e. text that would replace the start..end token span.
        # The correction is in the form of token text joined by
        # " " spaces, so correct_spaces() should be applied to
        # it before displaying it.
        self._suggest = suggest

    def __str__(self):
        """ Return a string representation of this annotation """
        return "{0:03}-{1:03}: {2:6} {3}{4}".format(
            self._start, self._end, self._code, self._text,
            "" if self._suggest is None else " / [" + self._suggest + "]"
        )

    @property
    def start(self):
        """ The index of the first token to which the annotation applies """
        return self._start

    @property
    def end(self):
        """ The index of the last token to which the annotation applies """
        return self._end

    @property
    def code(self):
        """ A code for the annotation type, usually an error or warning code """
        # If the code ends with "/w", it is a warning
        return self._code

    @property
    def text(self):
        """ A description of the annotation """
        return self._text

    @property
    def detail(self):
        """ A detailed description of the annotation, possibly including
            links within <a>...</a> tags """
        return self._detail

    @property
    def suggest(self):
        """ A suggested correction for the token span, as a text string
            containing tokens delimited by spaces """
        return self._suggest