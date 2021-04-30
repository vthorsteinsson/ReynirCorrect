# type: ignore
"""

    test_patterns.py

    Tests for GreynirCorrect module

    Copyright (C) 2021 by Miðeind ehf.

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


    This module tests the grammatical pattern annotation functionality
    of GreynirCorrect.

"""

import pytest

import reynir_correct

from test_annotator import check_sentence  # type: ignore


@pytest.fixture(scope="module")
def rc():
    """ Provide a module-scoped GreynirCorrect instance as a test fixture """
    r = reynir_correct.GreynirCorrect()
    yield r
    # Do teardown here
    r.__class__.cleanup()


def test_verb_af(rc):
    s = "Ráðherrann dáðist af hugrekki stjórnarandstöðunnar."
    check_sentence(rc, s, [(1, 2, "P_WRONG_PREP_AF")])

    s = (
        "Mig langaði að leita af bílnum, en dáðist svo af hugrekki lögreglukonunnar "
        "að ég gerði það ekki."
    )
    check_sentence(rc, s, [(3, 4, "P_WRONG_PREP_AF"), (8, 10, "P_WRONG_PREP_AF")])

    s = "Við höfum leitað í allan dag af kettinum, en fundum hann ekki."
    check_sentence(rc, s, [(2, 4, "P_WRONG_PREP_AF")])

    s = "Allan daginn höfum við leitað af kettinum."
    check_sentence(rc, s, [(4, 5, "P_WRONG_PREP_AF")])

    s = "Páll brosti af töktunum í Gunnu."
    check_sentence(rc, s, [(1, 2, "P_WRONG_PREP_AF")], ignore_warnings=True)

    s = "Ég var leitandi af kettinum í allan dag."
    check_sentence(rc, s, [(2, 3, "P_WRONG_PREP_AF")])

    s = "Ég vildi leita af mér allan grun."
    check_sentence(rc, s, [])

    s = "Hver leitar af skrifstofuhúsnæði?"
    check_sentence(rc, s, [(1, 2, "P_WRONG_PREP_AF")])


def test_verb_að(rc):
    s = "Ég er ekki hluti að heildinni."
    check_sentence(rc, s, [(1, 5, "P_WRONG_PREP_AÐ")])
    s = "Við höfum öll verið hluti að heildinni."
    check_sentence(rc, s, [(1, 6, "P_WRONG_PREP_AÐ")])
    s = "Vissulega er hægt að vera hluti að heildinni."
    check_sentence(rc, s, [(1, 7, "P_VeraAð"), (4, 7, "P_WRONG_PREP_AÐ")])
#    s = "Þeir sögðu að ég hefði verið hluti að heildinni."
#    check_sentence(rc, s, [(6, 7, "P_WRONG_PREP_AÐ")])
    s = "Þar að leiðandi virkar þetta."
    check_sentence(rc, s, [(0, 2, "P_WRONG_PREP_AÐ")])
    s = "Þetta virkar þar að leiðandi."
    check_sentence(rc, s, [(1, 4, "P_WRONG_PREP_AÐ")])
    s = "Ég hef ekki áhyggjur að honum."
    check_sentence(rc, s, [(1, 5, "P_WRONG_PREP_AÐ")])
    s = "Ég hef áhyggjur að því að honum líði illa."
    check_sentence(rc, s, [(1, 8, "P_WRONG_PREP_AÐ")])
#    s = "Ég lagði ekki mikið að mörkum."
#    check_sentence(rc, s, [(4, 5, "P_WRONG_PREP_AÐ")])
#    s = "Ég hafði lagt mikið að mörkum."
#    check_sentence(rc, s, [(4, 5, "P_WRONG_PREP_AÐ")])
    s = "Sama hvað ég gerði lagði ég mikið að mörkum."
    check_sentence(rc, s, [(7, 8, "P_WRONG_PREP_AÐ")])
    s = "Ég heillast að þannig fólki."
    check_sentence(rc, s, [(1, 2, "P_WRONG_PREP_AÐ")])
#    s = "Ég lét gott að mér leiða."
#    check_sentence(rc, s, [(1, 5, "P_WRONG_PREP_AÐ")])  # Only works if sentence is parsed correctly
    s = "Hún á heiðurinn að þessu."
    check_sentence(rc, s, [(1, 3, "P_WRONG_PREP_AÐ")])
    s = "Hún fékk heiðurinn að þessu."
    check_sentence(rc, s, [(2, 4, "P_WRONG_PREP_AÐ")])
    s = "Hún hlaut heiðurinn að þessu."
    check_sentence(rc, s, [(2, 4, "P_WRONG_PREP_AÐ")])
#    s = "Hún á heilan helling að börnum."
#    check_sentence(rc, s, [(1, 4, "P_WRONG_PREP_AÐ")])
    s = "Hún hefur ekki haft gagn að þessu."
    check_sentence(rc, s, [(1, 6, "P_WRONG_PREP_AÐ")])
    s = "Þetta hafði ekki komið að sjálfu sér."
    check_sentence(rc, s, [(4, 6, "P_aðaf")])
#    s = "Fréttir bárust seint af slysinu."
#    check_sentence(rc, s, [(1, 3, "P_WRONG_PREP_AÐ")])
    s = "Þetta er afgreitt mál að minni hálfu."
    check_sentence(rc, s, [(4, 6, "P_WRONG_PREP_AÐ")])
    s = "Hætta hefur aldrei stafað að þessu."
    check_sentence(rc, s, [(1, 5, "P_WRONG_PREP_AÐ")])
#    s = "Hún er ólétt að sínu þriðja barni."
#    check_sentence(rc, s, [(2, 3, "P_WRONG_PREP_AÐ")])
    s = "Hann hefur ekki heyrt að lausa starfinu."
    check_sentence(rc, s, [(2, 4, "P_WRONG_PREP_AÐ")])
    s = "Ég hef aldrei haft gaman að henni."
    check_sentence(rc, s, [(4, 5, "P_WRONG_PREP_AÐ")])
    s = "Þau voru sérstaklega valin að stjórninni."
    check_sentence(rc, s, [(2, 5, "P_WRONG_PREP_AÐ")])
    s = "Það er til mjög lítið að mjólk."
    check_sentence(rc, s, [(1, 6, "P_WRONG_PREP_AÐ")])
    s = "Ekki er mikið til að mjólk."
    check_sentence(rc, s, [(1, 5, "P_WRONG_PREP_AÐ")])
    s = "Ég hef ekki unnið verkefni að þessu tagi."
    check_sentence(rc, s, [(5, 7, "P_WRONG_PREP_AÐ")])
    s = "Verkefni að þessum toga eru erfið."
    check_sentence(rc, s, [(1, 3, "P_WRONG_PREP_AÐ")])
    s = "Hann gerði það að sjálfsdáðum."
    check_sentence(rc, s, [(3, 4, "P_aðaf")])
    s = "Hún hefur ekki gert þetta að miklum krafti."
    check_sentence(rc, s, [(5, 7, "P_WRONG_PREP_AÐ")])
    

def test_placename_pp(rc):
    s = "Ég hef búið á Hafnarfirði alla mína tíð en flyt nú í Akureyri."
    check_sentence(rc, s, [(3, 4, "P_WRONG_PLACE_PP"), (9, 10, "P_WRONG_PLACE_PP")])
    s = "Ég hef veitt í Vopnafirði undanfarin ár en búið á Vopnafirði."
    check_sentence(rc, s, [])
    s = "Það eru mörg náttúruvætti á Reykjanesi en ekki í Húsavík."
    check_sentence(rc, s, [(8, 9, "P_WRONG_PLACE_PP")])


def test_verb_líst(rc):
    s = "Jóni veiðimanni lýst ekki á þetta mál."
    check_sentence(rc, s, [(2, 2, "P_WRONG_OP_FORM")])
    s = "Eins og fram hefur komið lýst mér vel á þetta."
    check_sentence(rc, s, [(5, 5, "P_WRONG_OP_FORM")])
    s = "Jón hefur lýst sinni afstöðu til málsins."
    check_sentence(rc, s, [])
    s = "Þegar leið á kvöldið var gangstéttin lýst með ljósum."
    check_sentence(rc, s, [], ignore_warnings=True)
    # TODO: The following gets no annotation:
    # 'Ég verð að segja að mér lýst ekkert á þetta.'

def test_dir_loc(rc):
    s = "Börnin voru út á túni allan daginn."
    check_sentence(rc, s, [(2, 4, "P_DIR_LOC")])
    s = "Börnin voru útá túni allan daginn."
    check_sentence(rc, s, [(2, 3, "P_DIR_LOC"), (2, 2, "W001/w")])
#    s = "Út í heimi er þetta öðruvísi."
#    check_sentence(rc, s, [(0, 2, "P_DIR_LOC")])
#    s = "Útí heimi er þetta öðruvísi."
#    check_sentence(rc, s, [(0, 1, "P_DIR_LOC")])
#    s = "Börnin voru inn á vellinum allan daginn."
#    check_sentence(rc, s, [(2, 4, "P_DIR_LOC")])
    s = "Börnin voru inná vellinum allan daginn."
    check_sentence(rc, s, [(2, 3, "P_DIR_LOC"), (2, 2, "W001/w")])
#    s = "Hann var oft upp á hestinum."
#    check_sentence(rc, s, [(3, 5, "P_DIR_LOC")])
#    s = "Málið liggur í augum upp."
#    check_sentence(rc, s, [(2, 4, "P_DIR_LOC")])
#    s = "Þau eru alltaf uppí bústað."
#    check_sentence(rc, s, [(1, 4, "P_DIR_LOC"), (3, 3, "W001/w")])     # Span is either 1,4 or 3,4, but always corrected.
#    s = "Hún var niður í bæ í gær."
#    check_sentence(rc, s, [(1, 5, "P_DIR_LOC")])
    s = "Ég varð mér út um smá mat."
    check_sentence(rc, s, [(3, 6, "P_DIR_LOC")])
    s = "Þegar upp er staðið erum við öll eins."
    check_sentence(rc, s, [(1, 3, "P_DIR_LOC")])
#    s = "Út í heimi er þetta öðruvísi."
#    check_sentence(rc, s, [(0, 2, "P_DIR_LOC")])
#    s = "Börnin safnast saman inn í búð."
#    check_sentence(rc, s, [(1, 5, "P_DIR_LOC")])
    s = "Ég keypti þetta út í búð."
    check_sentence(rc, s, [(3, 5, "P_DIR_LOC")])
#    s = "Illgresið er út um allt."
#    check_sentence(rc, s, [(2, 4, "P_DIR_LOC")])
#    s = "Hann læsti sig inn í gær."
#    check_sentence(rc, s, [(1, 4, "P_DIR_LOC")])
#    s = "Hún gaf það upp í fréttum."
#    check_sentence(rc, s, [])
    s = "Ég ólst upp í Breiðholtinu."
    check_sentence(rc, s, [])
