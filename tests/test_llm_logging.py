from __future__ import annotations

import logging

from ntech_agent.llm import log_llm_failure


def test_log_llm_failure_logs_exception_type_and_message(caplog):
    with caplog.at_level(logging.WARNING, logger="ntech_agent.llm"):
        log_llm_failure("nodes.py::_answer_multi_repo_question", RuntimeError("rate limited"))

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert "nodes.py::_answer_multi_repo_question" in record.message
    assert "RuntimeError" in record.message
    assert "rate limited" in record.message
