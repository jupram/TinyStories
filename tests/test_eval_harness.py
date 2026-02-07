import json

from src.eval_harness import answer_matches, build_summary, load_prompts, normalize_text


def test_normalize_text_basic():
    assert normalize_text("  Girl!!!  ") == "girl"
    assert normalize_text("On-the table.") == "on the table"


def test_answer_matches_contains_word_boundary():
    passed, matched = answer_matches("girl.", ["girl"], match="contains")
    assert passed is True
    assert matched == "girl"

    passed, _ = answer_matches("shelf", ["she"], match="contains")
    assert passed is False


def test_answer_matches_modes():
    assert answer_matches("Blue", ["blue"], match="exact")[0] is True
    assert answer_matches("blue sky", ["blue"], match="exact")[0] is False
    assert answer_matches("blue sky", ["blue"], match="prefix")[0] is True
    assert answer_matches("value=42", [r"value=\d+"], match="regex")[0] is True


def test_load_prompts_accepts_answer_and_answers(tmp_path):
    payload = {
        "prompts": [
            {
                "id": "p1",
                "category": "gender",
                "difficulty": 1,
                "score": 1,
                "prompt": "Tom is a boy. Lily is a",
                "answer": "girl",
            },
            {
                "id": "p2",
                "category": "counting",
                "difficulty": 2,
                "score": 2,
                "prompt": "1 2 3",
                "answers": ["4", "four"],
            },
        ]
    }
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    prompts = load_prompts(str(path))
    assert len(prompts) == 2
    assert prompts[0].answers == ["girl"]
    assert prompts[1].answers == ["4", "four"]


def test_build_summary_weighted_accuracy():
    rows = [
        {
            "run_name": "baseline",
            "model": "baseline",
            "checkpoint": "checkpoint_1000.pt",
            "passed": 1,
            "points_earned": 2.0,
            "points_possible": 2.0,
        },
        {
            "run_name": "baseline",
            "model": "baseline",
            "checkpoint": "checkpoint_1000.pt",
            "passed": 0,
            "points_earned": 0.0,
            "points_possible": 3.0,
        },
    ]
    summary = build_summary(rows)
    assert len(summary) == 1
    assert summary[0]["accuracy"] == 0.5
    assert summary[0]["weighted_accuracy"] == 0.4
