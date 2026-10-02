import pytest

from call_companion.util import clip, detect_lang, keywords, name_stem, names_match, stem, tokenize


@pytest.mark.parametrize(
    "forms",
    [
        ("иван", "ивана", "ивану", "иваном", "иване"),
        ("алексей", "алексея", "алексею", "алексеем", "алексее"),
        ("дмитрий", "дмитрия", "дмитрию", "дмитрием", "дмитрии"),
        ("анна", "анну", "анны", "анне", "анной"),
        ("мария", "марии", "марию", "марией"),
        ("олена", "олену", "олени", "олені", "оленою"),
        ("коваленко", "коваленка", "коваленком"),
        ("иванова", "ивановой", "иванову", "ивановым"),
    ],
)
def test_name_stem_unifies_case_forms(forms):
    assert len({name_stem(f) for f in forms}) == 1


def test_name_stem_keeps_latin_and_short_names():
    assert name_stem("weber") == "weber"
    assert name_stem("ян") == "ян"


def test_names_match_by_containment():
    assert names_match("Олена", "Олена Коваленко")
    assert names_match("Макса Вебера", "Max Weber") is False
    assert names_match("Олена Коваленко", "Олена Петрова") is False
    assert names_match("", "Олена") is False


def test_tokenize_handles_apostrophes_and_possessives():
    assert tokenize("Max's проект, п'ять!") == ["max", "проект", "пять"]


def test_stem_unifies_slavic_inflections():
    assert len({stem(w) for w in ("документа", "документом", "документы", "документ")}) == 1
    assert len({stem(w) for w in ("оплата", "оплату", "оплаты")}) == 1
    assert stem("счет") == stem("счета")


def test_stem_unifies_english_forms():
    assert len({stem(w) for w in ("schedule", "schedules", "scheduled", "scheduling")}) == 1
    assert stem("tests") == stem("test")


def test_stem_leaves_numbers_and_short_words():
    assert stem("2025") == "2025"
    assert stem("api") == "api"


def test_keywords_drop_stopwords():
    assert keywords("Что у нас по срокам и бюджету?") == ["срокам", "бюджету"]


def test_detect_lang():
    assert detect_lang("Расскажите, пожалуйста, про бюджет проекта") == "ru"
    assert detect_lang("Розкажіть, будь ласка, про бюджет проєкту") == "uk"
    assert detect_lang("Can you walk me through the timeline?") == "en"
    assert detect_lang("что это") == "ru"
    assert detect_lang("що це") == "uk"
    assert detect_lang("1234 ???") == "und"


def test_clip_cuts_on_word_boundary():
    out = clip("один два три четыре пять шесть", 14)
    assert out.endswith("…") and len(out) <= 15 and " " not in out[-2:]
    assert clip("коротко", 50) == "коротко"
