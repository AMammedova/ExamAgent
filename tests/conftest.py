"""Test fixtures: every test runs against a throwaway data directory.

Isolation is enforced two ways, because getting it wrong destroys real work:

1. The suite writes its own .env into the temp directory and points
   EXAMAGENT_ENV_FILE at it. Environment variables alone are not enough -
   `reload_settings()` re-reads the .env with override=True, so a test that
   calls it (or any UI code path that does) would otherwise pull the
   developer's real DATA_DIR back into the running settings.
2. Every destructive fixture asserts it is operating inside the temp directory
   before it deletes anything, so a future isolation break fails the test run
   instead of wiping real uploads and progress.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Isolate the test run BEFORE examagent.config is imported anywhere.
_TMP = Path(tempfile.mkdtemp(prefix="examagent_tests_"))
_ENV_FILE = _TMP / "test.env"
_ENV_FILE.write_text(
    "\n".join([
        f"DATA_DIR={_TMP.as_posix()}",
        "LLM_PROVIDER=none",
        "ANTHROPIC_API_KEY=",
        "OPENAI_API_KEY=",
        "VECTOR_BACKEND=local",
        "EMBEDDING_BACKEND=local",
        "STUDY_DAYS=7",
        "EXAM_DATE=",
    ]) + "\n",
    encoding="utf-8",
)

os.environ["EXAMAGENT_ENV_FILE"] = str(_ENV_FILE)
os.environ["DATA_DIR"] = str(_TMP)
os.environ["LLM_PROVIDER"] = "none"
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["VECTOR_BACKEND"] = "local"
os.environ["STUDY_DAYS"] = "7"
os.environ["EXAM_DATE"] = ""


def assert_isolated() -> None:
    """Refuse to run anything destructive outside the throwaway directory."""
    from examagent.config import get_settings

    data_path = Path(get_settings().data_path).resolve()
    if _TMP.resolve() not in [data_path, *data_path.parents]:
        raise AssertionError(
            f"TEST ISOLATION BROKEN: settings point at {data_path}, not {_TMP}. "
            "Refusing to continue - a destructive fixture here would delete real "
            "uploads, progress and the vector store."
        )


@pytest.fixture(scope="session", autouse=True)
def _cleanup():
    yield
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture(autouse=True)
def _isolation_guard():
    """Checked around every test, so a leak is caught even when it is a test
    body (not a fixture) that calls reload_settings()."""
    assert_isolated()
    yield
    assert_isolated()


@pytest.fixture()
def clean_db():
    """A database seeded with topics and no student history."""
    assert_isolated()
    from examagent.models.db import (
        Attempt, Document, KeyValue, Mistake, MockExam, QuestionRecord,
        StudySession, Topic, session_scope,
    )
    from examagent.services.progress import ensure_topics

    with session_scope() as s:
        for model in (Attempt, Mistake, MockExam, StudySession, QuestionRecord,
                      Document, Topic, KeyValue):
            s.query(model).delete()
    with session_scope() as s:
        ensure_topics(s)
    yield


@pytest.fixture()
def clean_vectorstore():
    assert_isolated()
    from examagent.services.vectorstore import get_vector_store, reset_vector_store

    reset_vector_store()
    store = get_vector_store()
    store.reset()
    yield store
    assert_isolated()
    store.reset()
    reset_vector_store()


@pytest.fixture()
def tmp_docs(tmp_path: Path) -> Path:
    """A small sample course-material corpus."""
    (tmp_path / "lecture3.md").write_text(
        "# Model Validation\n\n"
        "Model validation separates the data into training, validation and test sets. "
        "The validation set is used to select hyperparameters, and the test set is "
        "touched exactly once at the end.\n\n"
        "## Cross Validation\n\n"
        "K-fold cross validation splits the data into k folds. Each fold serves as the "
        "validation set exactly once. This gives a lower variance estimate of "
        "generalisation performance than a single split. Preprocessing must be fitted "
        "inside each fold to avoid data leakage.\n",
        encoding="utf-8",
    )
    (tmp_path / "udemy_regression.txt").write_text(
        "Simple linear regression fits a straight line through the data using ordinary "
        "least squares. The slope is the covariance of x and y divided by the variance "
        "of x. Multiple linear regression extends this to several predictors. Watch out "
        "for the dummy variable trap when encoding categorical features.\n",
        encoding="utf-8",
    )
    (tmp_path / "exam_sample.md").write_text(
        "# Previous Midterm\n\n"
        "Question 1. A 2-2-1 MLP uses ReLU in the hidden layer and a sigmoid output with "
        "binary cross entropy loss. Compute the forward pass, the loss, the gradients, "
        "and perform one gradient descent update. Identify which hidden unit receives "
        "zero gradient and explain why.\n\n"
        "Question 2. Assertion: dropout reduces overfitting. Reason: dropout removes "
        "units at test time.\n",
        encoding="utf-8",
    )
    return tmp_path
