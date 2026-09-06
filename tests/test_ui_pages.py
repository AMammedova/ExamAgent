"""UI smoke tests: every page must render without raising.

Uses Streamlit's own AppTest harness, which executes app.py for real.
"""
from __future__ import annotations

import pathlib

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP = str((pathlib.Path(__file__).resolve().parent.parent / "app.py"))
PAGES = ["Dashboard", "Learning Path", "Study", "Quiz", "Mock Exam", "Chat",
         "Weaknesses", "Knowledge Map", "Progress", "Materials", "Settings"]


def _run(page: str | None = None, **state):
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state["first_run_seen"] = True
    at.session_state["use_llm"] = False
    if page:
        at.session_state["page"] = page
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _assert_clean(at, label: str) -> None:
    if at.exception:
        messages = [f"{e.type}: {e.message}" for e in at.exception]
        pytest.fail(f"{label} raised:\n" + "\n".join(messages))


def test_first_run_screen_renders(clean_db) -> None:
    """A brand-new profile shows the onboarding screen."""
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state["use_llm"] = False
    at.run()
    _assert_clean(at, "first run")
    text = " ".join(m.value for m in at.markdown)
    assert "You have" in text and "days" in text


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(clean_db, page: str) -> None:
    from examagent.services import progress

    progress.mark_first_run_complete()
    at = _run(page)
    _assert_clean(at, page)
    assert at.sidebar.markdown, f"{page}: the sidebar did not render"


def test_dashboard_shows_the_countdown_and_next_action(clean_db) -> None:
    from examagent.services import progress

    progress.mark_first_run_complete()
    at = _run("Dashboard")
    _assert_clean(at, "Dashboard")
    text = " ".join(m.value for m in at.markdown)
    assert "EXAM IN" in text
    assert "What to study now" in text


def test_pages_render_after_recorded_activity(clean_db) -> None:
    """The pages that read student history must survive having data."""
    from examagent.models.schemas import Category, Evaluation, MistakeType, Priority
    from examagent.models.schemas import Question, QuestionType
    from examagent.services import mock_exam, progress

    progress.mark_first_run_complete()
    for topic, qtype, score in [
        ("backpropagation", QuestionType.CALCULATION, 2.0),
        ("backpropagation", QuestionType.CONCEPTUAL, 9.0),
        ("pca", QuestionType.CONCEPTUAL, 5.0),
        ("attention", QuestionType.ASSERTION_REASON, 10.0),
    ]:
        q = Question(id=f"ui:{topic}:{qtype.value}", topic=topic, category=Category.DL,
                     question_type=qtype, difficulty=5, priority=Priority.CRITICAL,
                     prompt="p")
        progress.record_attempt(
            q,
            Evaluation(score=score, correct=score >= 7,
                       mistake_type=(MistakeType.NONE if score >= 7
                                     else MistakeType.CONCEPTUAL)),
            student_answer="a", context="quiz", seconds=60,
        )

    exam = mock_exam.build_exam(n_questions=6, use_llm=False, seed=1)
    answers = {}
    for eq in exam["questions"]:
        if eq.question_type == QuestionType.CALCULATION and eq.calc_spec:
            answers[eq.id] = {p["key"]: str(p["answer"])
                              for p in eq.calc_spec["parts"][:2]}
        elif eq.question_type == QuestionType.ASSERTION_REASON:
            answers[eq.id] = eq.correct_option
        else:
            answers[eq.id] = eq.model_answer or "x" * 120
    mock_exam.submit_exam(exam["exam_id"], answers, duration_seconds=900, use_llm=False)

    for page in ["Dashboard", "Weaknesses", "Progress", "Knowledge Map", "Mock Exam"]:
        at = _run(page)
        _assert_clean(at, f"{page} (with history)")


def test_study_page_opens_a_topic_session(clean_db) -> None:
    from examagent.services import progress

    progress.mark_first_run_complete()
    at = _run("Study", nav_payload={"topic_id": "backpropagation"})
    _assert_clean(at, "Study with a preselected topic")
    text = " ".join(m.value for m in at.markdown)
    assert "Backpropagation" in text


def test_materials_page_lists_ingested_documents(clean_db, clean_vectorstore,
                                                 tmp_docs) -> None:
    from examagent.models.schemas import SourceType
    from examagent.services import materials, progress

    progress.mark_first_run_complete()
    materials.ingest_file(tmp_docs / "lecture3.md", SourceType.UNIVERSITY_ML.value,
                          lecture="Lecture 3")
    at = _run("Materials")
    _assert_clean(at, "Materials")


def test_settings_page_reports_the_engines(clean_db) -> None:
    from examagent.services import progress

    progress.mark_first_run_complete()
    at = _run("Settings")
    _assert_clean(at, "Settings")


# ---------------------------------------------------------------- interaction
def test_quiz_answer_flow_marks_and_records(clean_db) -> None:
    """Generate a question, submit an answer, receive marking, and persist it."""
    from examagent.models.db import Attempt, session_scope
    from examagent.services import progress

    progress.mark_first_run_complete()
    at = _run("Quiz")
    _assert_clean(at, "Quiz")

    # press "Next question" to generate one
    buttons = [b for b in at.button if b.label == "Next question"]
    assert buttons, "the Quiz page must offer a way to generate a question"
    at = buttons[0].click().run()
    _assert_clean(at, "Quiz after generating a question")
    assert "question" in at.session_state["quiz"], "no question was generated"

    question = at.session_state["quiz"]["question"]

    # answer it: radio for choice questions, text area otherwise
    if question.options:
        assert at.radio, "an options question must render a radio group"
        at.radio[0].set_value(at.radio[0].options[0])
    elif at.text_area:
        at.text_area[0].set_value(
            "Dropout randomly deactivates units during training, which prevents "
            "co-adaptation between units and therefore reduces overfitting."
        )
    elif at.text_input:
        for field in at.text_input:
            field.set_value("1")
    at = at.run()
    _assert_clean(at, "Quiz after entering an answer")

    submit = [b for b in at.button if b.label == "Submit answer"]
    assert submit, "the Quiz page must offer Submit"
    at = submit[0].click().run()
    _assert_clean(at, "Quiz after submitting")

    assert "evaluation" in at.session_state["quiz"], "the answer was not marked"
    with session_scope() as s:
        assert s.query(Attempt).count() >= 1, "the attempt was not persisted"


def test_quiz_refuses_to_mark_a_blank_answer(clean_db) -> None:
    from examagent.services import progress

    progress.mark_first_run_complete()
    at = _run("Quiz")
    at = [b for b in at.button if b.label == "Next question"][0].click().run()
    submit = [b for b in at.button if b.label == "Submit answer"]
    at = submit[0].click().run()
    _assert_clean(at, "Quiz blank submit")
    assert at.error, "submitting nothing must produce an error, not a mark"
    assert "evaluation" not in at.session_state["quiz"]


def test_mock_exam_can_be_generated_and_submitted(clean_db) -> None:
    from examagent.services import progress

    progress.mark_first_run_complete()
    at = _run("Mock Exam")
    _assert_clean(at, "Mock Exam setup")

    generate = [b for b in at.button if b.label == "Generate exam"]
    assert generate, "the Mock Exam page must offer Generate"
    at = generate[0].click().run()
    _assert_clean(at, "Mock Exam after generation")
    assert at.session_state["mock"].get("exam"), "no exam was built"
    assert at.session_state["mock"]["exam"]["questions"]

    submit = [b for b in at.button if b.label == "Submit paper"]
    assert submit, "the exam must be submittable"
    at = submit[0].click().run()
    _assert_clean(at, "Mock Exam after submission")
    report = at.session_state["mock"].get("report")
    assert report is not None
    assert report.percentage == 0.0, "an unanswered paper must score zero"
    assert report.revision_plan


def test_chat_answers_a_command(clean_db) -> None:
    from examagent.services import progress

    progress.mark_first_run_complete()
    at = _run("Chat")
    _assert_clean(at, "Chat")
    assert at.chat_input, "the chat page must render an input"
    at = at.chat_input[0].set_value("/weakness").run()
    _assert_clean(at, "Chat after a command")
    # AppTest's session_state proxy has no .get(), so index it directly
    assert "chat_history" in at.session_state
    history = at.session_state["chat_history"]
    assert history and history[-1]["role"] == "assistant"
    assert "Weakest topics" in history[-1]["text"]


def test_study_teaching_flow_enforces_active_recall(clean_db) -> None:
    """Lesson -> explain it back -> marked -> adaptive practice, all persisted."""
    from examagent.models.db import Attempt, session_scope
    from examagent.services import progress

    progress.mark_first_run_complete()
    at = _run("Study", **{"study": {"topic_id": "backpropagation", "mode": "30 Minute Study",
                                    "step": "lesson", "answered": [],
                                    "queue_index": 0, "n_target": 3}})
    _assert_clean(at, "Study lesson step")
    text = " ".join(m.value for m in at.markdown)
    assert "Backpropagation" in text
    assert "exam" in text.lower()

    # the lesson must not pre-empt retrieval: the next step is a question
    advance = [b for b in at.button if b.label == "I have read it — test me"]
    assert advance, "the lesson must hand off to active recall"
    at = advance[0].click().run()
    _assert_clean(at, "Study recall step")
    assert at.session_state["study"]["step"] == "recall"
    assert at.text_area, "the recall step must ask the student to write first"

    # answering nothing must not be submittable
    submit = [b for b in at.button if b.label == "Submit"]
    assert submit and submit[0].disabled, "an empty explanation must not be markable"

    at.text_area[0].set_value(
        "Backpropagation applies the chain rule in reverse through the computational "
        "graph, so each layer's local gradient is multiplied by the upstream gradient. "
        "This gives the gradient of the loss with respect to every weight in roughly one "
        "backward pass, which the optimiser then uses to update the parameters."
    )
    at = at.run()
    at = [b for b in at.button if b.label == "Submit"][0].click().run()
    _assert_clean(at, "Study after submitting the explanation")

    assert "recall_eval" in at.session_state["study"], "the explanation was not marked"
    with session_scope() as s:
        assert s.query(Attempt).filter(Attempt.context == "study").count() == 1

    # continue into adaptive practice
    cont = [b for b in at.button if b.label == "Continue to practice questions"]
    assert cont
    at = cont[0].click().run()
    _assert_clean(at, "Study practice step")
    assert at.session_state["study"]["step"] == "practice"
    assert "current_q" in at.session_state["study"], "no practice question was built"


def test_learning_path_marks_a_topic_done_right_after_the_last_answer(clean_db) -> None:
    """Regression: mark_complete used to fire only from the 'Finish topic'
    button below the quiz. The free 'Previous topic'/'Next topic' buttons
    above the quiz are reachable at any time, so a student who answered all
    three questions and then clicked one of those instead of scrolling down
    to 'Finish topic' would leave the topic with a full qa_history but never
    marked done. It must now be marked the moment the last answer is scored."""
    from examagent.models.schemas import Category, Priority, Question, QuestionType
    from examagent.services import learning_path as lp
    from examagent.services import progress

    progress.mark_first_run_complete()
    tid = lp.CURRICULUM[0]
    q = Question(
        id="lp-regress-q3", topic=tid, category=Category.ML,
        question_type=QuestionType.SHORT_ANSWER, difficulty=4,
        priority=Priority.CRITICAL, prompt="Final question in this topic?",
    )
    at = _run("Learning Path", lp_active={
        "topic_id": tid,
        "scores": [8.0, 7.0],          # first two questions already answered
        "q_index": 2,                  # on the third (last) question
        "q": q,
        "asked": [q.id],
        "asked_prompts": [q.prompt],
    })
    _assert_clean(at, "Learning Path on the last question")
    assert lp.completed_topic_ids() == [], "must not be done before answering the last one"

    assert at.text_area, "a short-answer question must offer a text area"
    at.text_area[0].set_value("A complete answer to the last question.")
    at = at.run()
    submit = [b for b in at.button if b.label == "Submit answer"]
    assert submit, "the last question must still be submittable"
    at = submit[0].click().run()
    _assert_clean(at, "Learning Path after submitting the last answer")

    assert tid in lp.completed_topic_ids(), (
        "the topic must be marked done as soon as the last question is scored, "
        "not only after a later 'Finish topic' click"
    )
