from eval.latency import stages
from eval.retrieval_eval import first_hit, summarise
from eval.run_eval import argument_correct, citation_correct, normalise, percentile, rate, rows_wanted, tool_correct


def test_normalise_strips_punctuation_and_dandas():
    assert normalise("प्रकाशसंश्लेषण म्हणजे काय?") == "प्रकाशसंश्लेषण म्हणजे काय"
    assert normalise("यह क्या है।  Photosynthesis!") == "यह क्या है photosynthesis"
    assert normalise("तत्\u200dत्\u200dवज्ञान") == normalise("तत्त्वज्ञान")       # invisible joiners are not a difference
    assert normalise("क\u093c") == normalise("\u0958")                          # nor is one letter stored two ways


def test_stage_times_separate_spoken_turns_from_typed_ones():
    rows = [(1000, 1500, 300, 4000, "main"), (1200, 2000, None, None, "main"), (None, 1700, 200, None, "backup")]
    times = stages(rows)
    assert times["listening"] == [1000, 1200]
    assert times["spoken question to written answer"] == [2800, 3200]
    assert times["spoken question to voice ready"] == [6800]            # only the turn whose voice was made
    assert times["typed question to written answer"] == [1900]


def test_tool_correct_treats_blank_as_none():
    assert tool_correct("", None)
    assert tool_correct("none", None)
    assert tool_correct("search_notes", "search_notes")
    assert not tool_correct("search_notes", None)


def test_argument_correct_only_scores_quiz_chapters():
    quiz = {"expected_tool": "quiz_me", "expected_query_or_chapter": "6"}
    search = {"expected_tool": "search_notes", "expected_query_or_chapter": "प्रकाशसंश्लेषण"}
    assert argument_correct(quiz, {"chapter": 6}) is True
    assert argument_correct(quiz, {"chapter": 7}) is False
    assert argument_correct(quiz, None) is False
    assert argument_correct(search, {"query": "anything"}) is None     # needs a human


def test_citation_needs_the_page_retrieved_and_spoken():
    sources = [{"source": "sci.pdf", "page": 12}]
    assert citation_correct("12", sources, "उत्तर. पान 12.") is True
    assert citation_correct("12", sources, "उत्तर. पान 112.") is False   # 112 is not 12
    assert citation_correct("12", [{"source": "sci.pdf", "page": 3}], "उत्तर. पान 12.") is False
    assert citation_correct("", sources, "उत्तर.") is None


def test_a_question_may_have_more_than_one_right_page():
    sources = [{"source": "sci.pdf", "page": 34}]
    assert citation_correct("35|34", sources, "Answer. Page 34.") is True
    assert citation_correct("35|34", sources, "Answer. Page 35.") is False   # named, but that page was not retrieved
    assert citation_correct(" | ", sources, "Answer.") is None


def test_a_slice_of_the_rows_counts_from_one():
    rows = [{"n": n} for n in range(1, 11)]
    assert [r["n"] for r in rows_wanted(rows, "3-5")] == [3, 4, 5]
    assert [r["n"] for r in rows_wanted(rows, "9-20")] == [9, 10]
    assert [r["n"] for r in rows_wanted(rows, "4")] == [4]
    assert rows_wanted(rows, "") == rows


def test_the_search_is_scored_by_where_the_right_page_first_appears():
    assert first_hit(["12", "35", "35"], {"35", "34"}) == 2
    assert first_hit(["12", "13"], {"35"}) is None
    assert first_hit([], {"35"}) is None
    scores = summarise([1, 2, 4, None])         # first, second, fourth, not found
    assert (scores["top 1"], scores["top 3"], scores["top 5"]) == (0.25, 0.5, 0.75)
    assert scores["MRR"] == (1 + 1 / 2 + 1 / 4) / 4
    assert summarise([])["top 3"] == 0


def test_percentile_and_rate():
    assert percentile([100, 200, 300, 400, 1000], 95) == 1000
    assert percentile([100, 200, 300], 50) == 200
    assert rate([True, False, None, True]) == "67% (2/3)"
    assert rate([None]) == "n/a"
