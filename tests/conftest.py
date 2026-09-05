"""Test fixtures: every test runs against a throwaway data directory."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Isolate the test run BEFORE examagent.config is imported anywhere.
_TMP = Path(tempfile.mkdtemp(prefix="examagent_tests_"))
os.environ["DATA_DIR"] = str(_TMP)
os.environ["LLM_PROVIDER"] = "none"
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["VECTOR_BACKEND"] = "local"
os.environ["STUDY_DAYS"] = "7"
os.environ["EXAM_DATE"] = ""


@pytest.fixture(scope="session", autouse=True)
def _cleanup():
    yield
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture()
def clean_db():
    """A database seeded with topics and no student history."""
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
    from examagent.services.vectorstore import get_vector_store, reset_vector_store

    reset_vector_store()
    store = get_vector_store()
    store.reset()
    yield store
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
