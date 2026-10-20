# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Rows that are not plain text become documents here.

A problem with its solution, a program with the tests it was run against and what happened, a commit with
its diff, a clinical question with its answer: each renderer turns one structured row into the text a
pretraining model reads. Column names vary between datasets, so every renderer takes the first column
that is present from a short list, and returns ``None`` when a row has nothing usable.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable

Renderer = Callable[[dict], "str | None"]


def _first(row: dict, *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else "\n".join(map(str, value))
        value = str(value).strip()
        if value:
            return value
    return None


def problem_solution(row: dict) -> str | None:
    """Math and reasoning sets: NuminaMath, OpenCodeReasoning, WebInstruct, medical-o1 ..."""
    problem = _first(row, "problem", "question", "Question", "input", "query", "instruction", "prompt")
    thinking = _first(row, "Complex_CoT", "reasoning", "thinking")
    solution = _first(row, "solution", "answer", "Response", "response", "output", "completion")
    if not problem or not solution:
        return None
    text = f"Problem:\n{problem}\n\n"
    if thinking:
        text += f"Reasoning:\n{thinking}\n\n"
    return text + f"Solution:\n{solution}\n"


def execution_trace(row: dict) -> str | None:
    """
    OpenCodeInstruct and friends: the task, the program, the unit tests it was run against, and the
    recorded result of running them -- a synthetic execution trace, the program *and* what it did.
    """
    task = _first(row, "input", "instruction", "prompt", "question")
    code = _first(row, "output", "solution", "response", "code")
    if not task or not code:
        return None
    tests = _first(row, "unit_tests", "tests", "test")
    status = _first(row, "tests_execution_status", "execution_status", "status")
    score = row.get("average_test_score")
    text = f"Task:\n{task}\n\nProgram:\n{code}\n"
    if tests:
        text += f"\nTests:\n{tests}\n"
    if status or score is not None:
        text += "\nExecution:\n"
        if status:
            text += f"{status}\n"
        if score is not None:
            with contextlib.suppress(TypeError, ValueError):
                text += f"tests passed: {float(score):.0%}\n"
    return text


def code_feedback(row: dict) -> str | None:
    """Conversations where code was run and its output fed back (CodeFeedback): kept as a transcript."""
    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            messages = None
    if messages:
        parts = []
        for m in messages:
            role = m.get("role") or m.get("from") or "user"
            content = m.get("content") or m.get("value") or ""
            parts.append(f"{role}: {content}")
        return "\n\n".join(parts) + "\n" if parts else None
    return problem_solution(row)


def commit(row: dict) -> str | None:
    """CommitPack: the message, then the change as before/after (or a diff when that is what is given)."""
    message = _first(row, "message", "subject", "commit_message")
    diff = _first(row, "diff", "patch")
    before, after = _first(row, "old_contents", "old_file"), _first(row, "new_contents", "new_file")
    path = _first(row, "new_file", "path", "filename") or ""
    lang = _first(row, "lang", "language") or ""
    if not message or not (diff or after):
        return None
    head = f"Commit: {message}\nFile: {path} ({lang})\n" if path else f"Commit: {message}\n"
    if diff:
        return head + f"\nDiff:\n{diff}\n"
    text = head
    if before:
        text += f"\nBefore:\n{before}\n"
    return text + f"\nAfter:\n{after}\n"


def github_issue(row: dict) -> str | None:
    title = _first(row, "title")
    events = row.get("events")
    body = _first(row, "body", "text", "content")
    if events and isinstance(events, list):
        thread = "\n\n".join(
            f"{e.get('author', 'user')}: {e.get('text') or e.get('body') or ''}" for e in events if isinstance(e, dict)
        )
        return (f"Issue: {title}\n\n" if title else "") + thread + "\n" if thread.strip() else None
    if not body:
        return None
    return (f"Issue: {title}\n\n" if title else "") + body + "\n"


def multiple_choice(row: dict) -> str | None:
    """MedMCQA-style: the question, the options, the answer, and the explanation when there is one."""
    question = _first(row, "question")
    options = [row.get(k) for k in ("opa", "opb", "opc", "opd") if row.get(k)]
    if not question or not options:
        return problem_solution(row)
    answer = row.get("cop")
    letters = "ABCD"
    text = f"Question: {question}\n" + "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options)) + "\n"
    if isinstance(answer, int) and 0 <= answer < len(options):
        text += f"Answer: {letters[answer]}. {options[answer]}\n"
    explanation = _first(row, "exp", "explanation")
    if explanation:
        text += f"Explanation: {explanation}\n"
    return text


def pubmed_qa(row: dict) -> str | None:
    question = _first(row, "question")
    context = row.get("context")
    if isinstance(context, dict):
        context = "\n".join(context.get("contexts") or [])
    answer = _first(row, "long_answer", "final_decision")
    if not question or not answer:
        return None
    text = f"Question: {question}\n"
    if context:
        text += f"\nContext:\n{context}\n"
    return text + f"\nAnswer: {answer}\n"


def guideline(row: dict) -> str | None:
    title, body = _first(row, "title"), _first(row, "clean_text", "text", "content")
    if not body:
        return None
    source = _first(row, "source")
    return (f"{title}\n" if title else "") + (f"Source: {source}\n\n" if source else "\n") + body + "\n"


def chat_transcript(row: dict) -> str | None:
    """Instruction sets with system/user/assistant columns (Finance-Instruct) or a messages list."""
    messages = row.get("messages") or row.get("conversations")
    if messages:
        return code_feedback(row)
    system, user, assistant = (
        _first(row, "system"),
        _first(row, "user", "inputs", "question"),
        _first(row, "assistant", "answer", "output"),
    )
    if not user or not assistant:
        return None
    return (f"system: {system}\n\n" if system else "") + f"user: {user}\n\nassistant: {assistant}\n"


RENDERERS: dict[str, Renderer] = {
    "problem_solution": problem_solution,
    "execution_trace": execution_trace,
    "code_feedback": code_feedback,
    "commit": commit,
    "github_issue": github_issue,
    "multiple_choice": multiple_choice,
    "pubmed_qa": pubmed_qa,
    "guideline": guideline,
    "chat_transcript": chat_transcript,
}
