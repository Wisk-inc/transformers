# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The Jamaican data tools against local stand-ins: a crawl of a tiny site that respects robots.txt, dialogues
from a fake teacher endpoint, the renderers on structured rows, and the safeguards' data and scoring."""

import http.server
import json
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from helix_lm.data import safety
from helix_lm.data.agent import convert_columns_chat, convert_medical_o1, convert_prompt_response, prepare_sft
from helix_lm.data.catalog import Source
from helix_lm.data.jamaica import is_about_caribbean
from helix_lm.data.jamaica_gov import HostCrawl, html_to_text
from helix_lm.data.loader import TokenSource
from helix_lm.data.prepare import prepare_source, shard_files
from helix_lm.data.renderers import RENDERERS
from helix_lm.data.synth import PATOIS_MARKERS, Teacher, acceptable, parse_dialogue, scenarios, synthesize, to_messages

LONG = " ".join(f"Sentence number {i} of a public notice from the Government of Jamaica." for i in range(12))


def serve(root: Path):
    handler = type("Quiet", (http.server.SimpleHTTPRequestHandler,), {"log_message": lambda *a, **k: None})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *a, **k: handler(*a, directory=str(root), **k))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ------------------------------------------------------------------------------------------ crawler --
def test_crawler_respects_robots_and_stays_on_host(tmp_path):
    site = tmp_path / "site"
    (site / "private").mkdir(parents=True)
    (site / "robots.txt").write_text("User-agent: *\nDisallow: /private/\nCrawl-delay: 0\n")
    (site / "index.html").write_text(
        f"<html><head><title>Ministry</title></head><body><nav>menu</nav><p>{LONG}</p>"
        '<a href="/a.html">A</a> <a href="/private/secret.html">S</a> <a href="https://elsewhere.example/x">X</a>'
        '<a href="/logo.png">img</a></body></html>'
    )
    (site / "a.html").write_text(
        f"<html><head><title>Judgment A</title></head><body><article>{LONG}</article></body></html>"
    )
    (site / "private" / "secret.html").write_text(f"<html><body>{LONG}</body></html>")
    (site / "sitemap.xml").write_text('<?xml version="1.0"?><urlset><url><loc>http://HOST/a.html</loc></url></urlset>')
    server = serve(site)
    host = f"127.0.0.1:{server.server_port}"
    (site / "sitemap.xml").write_text((site / "sitemap.xml").read_text().replace("HOST", host))
    seed = {"starts": [f"http://{host}/index.html"], "sitemaps": [f"http://{host}/sitemap.xml"], "kind": "test"}
    out = tmp_path / "out"
    crawler = HostCrawl(host, seed, out, max_pages=50, deadline=1e18, delay=0.0, log=lambda *_: None)
    assert crawler.run() == 2
    records = [json.loads(line) for line in (out / f"{host}.jsonl").read_text().splitlines()]
    urls = {r["url"].split(host)[1] for r in records}
    assert urls == {"/index.html", "/a.html"}
    assert all("secret" not in r["url"] and "elsewhere" not in r["url"] for r in records)
    assert any(r["title"] == "Judgment A" for r in records) and all(r["kind"] == "test" for r in records)
    assert not any("menu" in r["text"] for r in records), "navigation chrome is dropped"
    # a second run resumes from the saved state and fetches nothing new
    again = HostCrawl(host, seed, out, max_pages=50, deadline=1e18, delay=0.0, log=lambda *_: None)
    again.run()
    assert again.pages == 0 and again.documents == 2
    server.shutdown()
    server.server_close()


def test_html_to_text_and_caribbean_filter():
    title, text, links = html_to_text(
        b"<title>T</title><script>x()</script><p>Hello <b>there</b></p><a href='/z'>z</a>"
    )
    assert title == "T" and text.startswith("Hello there") and "x()" not in text and links == ["/z"]
    assert is_about_caribbean("CARICOM heads met in Port of Spain, Trinidad, with Barbados and Guyana attending.")
    assert not is_about_caribbean("We booked a Caribbean cruise for March.")


# ------------------------------------------------------------------------------------- synthesizer --
class FakeTeacherHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        scenario = body["messages"][-1]["content"].split("\n")[0]
        turns = [
            {
                "role": "customer",
                "content": f"Mawnin, mi have a problem: {scenario} Mi nuh know weh fi do, yuh can help mi?",
            },
            {
                "role": "agent",
                "content": "Good morning, yes I can help. Let me look at that for you now; it will take a moment.",
            },
            {
                "role": "customer",
                "content": "Alright, mi a wait. Dem seh mi fi carry mi TRN but mi nuh have it pon mi.",
            },
            {
                "role": "agent",
                "content": "No problem, wi can use yuh ID fi now. Fi di rest yuh can bring di TRN tomorrow, "
                "nuh badda worry.",
            },
            {"role": "customer", "content": "Tank yuh, dat mek sense. Likkle more."},
            {
                "role": "agent",
                "content": "Walk good. The reference number is on the receipt if you need to call back.",
            },
        ]
        reply = json.dumps(
            {"choices": [{"message": {"content": "```json\n" + json.dumps({"conversation": turns}) + "\n```"}}]}
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(reply.encode())


def test_synthesizer_makes_both_outputs_through_a_fake_teacher(tmp_path):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeTeacherHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    teacher = Teacher(f"http://127.0.0.1:{server.server_port}/v1", "fake-teacher")
    summary = synthesize(tmp_path, teacher, count=8, workers=2, log=lambda *_: None)
    assert summary["kept"] >= 6 and summary["failed"] == 0
    chats = [
        json.loads(l) for l in (tmp_path / "local" / "codeswitch-chat" / "dialogues-0.jsonl").read_text().splitlines()
    ]
    docs = [
        json.loads(l) for l in (tmp_path / "local" / "codeswitch" / "transcripts-0.jsonl").read_text().splitlines()
    ]
    assert chats and docs and all(c["synthetic"] for c in chats)
    assert chats[0]["messages"][0]["role"] == "user" and chats[0]["messages"][-1]["role"] == "assistant"
    assert docs[0]["text"].startswith("[") and "synthetic dialogue" in docs[0]["text"]
    assert len(PATOIS_MARKERS.findall(docs[0]["text"])) >= 6
    # a second call does nothing new: the ids are recorded
    assert synthesize(tmp_path, teacher, count=8, workers=2, log=lambda *_: None)["kept"] == 0
    server.shutdown()
    server.server_close()


def test_synth_filters_and_scenarios():
    specs = scenarios(30)
    assert len(specs) == 30 and {s["domain"] for s in specs} >= {"banking", "legal", "health", "education"}
    assert parse_dialogue("no json here") is None
    turns = parse_dialogue(
        '{"conversation": [{"role": "customer", "content": "hi"}, {"role": "agent", "content": "hello"}]}'
    )
    assert turns and not acceptable(turns, "patois"), "too short and no Patois"
    long_sje = [
        {"role": "customer" if i % 2 == 0 else "agent", "content": "Thank you for calling the branch today. " * 6}
        for i in range(6)
    ]
    assert acceptable(long_sje, "sje") and not acceptable(long_sje, "patois")
    turns = [("agent", "x"), ("customer", "a"), ("customer", "b"), ("agent", "c"), ("customer", "d")]
    turns = [{"role": role, "content": content} for role, content in turns]
    messages = to_messages(turns)
    assert messages == [{"role": "user", "content": "a\nb"}, {"role": "assistant", "content": "c"}]


# ----------------------------------------------------------------------------------------- renderers --
def test_renderers_turn_rows_into_documents(corpus, tmp_path):
    row = {"input": "Sum a list", "output": "def s(xs): return sum(xs)", "unit_tests": "assert s([1,2])==3",
           "tests_execution_status": "passed", "average_test_score": 1.0}  # fmt: skip
    trace = RENDERERS["execution_trace"](row)
    assert trace.startswith("Task:") and "Execution:" in trace and "tests passed: 100%" in trace
    assert RENDERERS["commit"](
        {"message": "Fix bug", "old_contents": "a", "new_contents": "b", "new_file": "x.py", "lang": "python"}
    ).startswith("Commit: Fix bug")
    mc = RENDERERS["multiple_choice"](
        {"question": "Q?", "opa": "1", "opb": "2", "opc": "3", "opd": "4", "cop": 1, "exp": "because"}
    )
    assert "Answer: B. 2" in mc and "Explanation: because" in mc
    assert RENDERERS["pubmed_qa"](
        {"question": "Does X?", "context": {"contexts": ["c1", "c2"]}, "long_answer": "Yes."}
    ).endswith("Answer: Yes.\n")
    assert RENDERERS["problem_solution"]({"problem": "1+1", "solution": "2"}) == "Problem:\n1+1\n\nSolution:\n2\n"
    assert RENDERERS["problem_solution"]({"problem": "1+1"}) is None
    # through the preparer: a parquet of OpenCodeInstruct-like rows becomes shards
    rows = {
        "input": [f"Task {i}: write a function that returns {i}." for i in range(40)],
        "output": [f"def f():\n    return {i}\n" * 3 for i in range(40)],
        "unit_tests": ["assert f() is not None"] * 40,
        "tests_execution_status": ["passed"] * 40,
    }
    path = tmp_path / "traces.parquet"
    pq.write_table(pa.table(rows), path)
    source = Source("traces", "local", "output", 0.1, kind="code", render="execution_trace")
    prepare_source(
        source, tmp_path / "data", corpus[1], None, workers=1, local_files=[path], min_free_gb=0, log=lambda *_: None
    )
    assert shard_files(tmp_path / "data" / "tokens" / "traces")


# ------------------------------------------------------------------------------------------ safety --
def test_safety_data_and_scoring(corpus, tmp_path, monkeypatch):
    conversations = list(safety.safety_conversations(40))
    assert len(conversations) == 40 and all(c[-1]["role"] == "assistant" for c in conversations)
    assert any(c[0]["role"] == "system" and c[0]["content"] == safety.POLICY for c in conversations)
    assert "119" in safety.POLICY and "888-NEW-LIFE" in safety.HELPLINES
    assert safety.looks_like_refusal("I won't run that: it deletes the whole filesystem.")
    assert safety.looks_like_refusal("Mi cyaan help wid dat, but mi can show yuh how fi report it.")
    assert not safety.looks_like_refusal("Find it and send it a signal: ps aux | grep python")
    monkeypatch.setattr(safety, "xstest_prompts", lambda token=None: (["safe one", "safe two"], ["unsafe one"]))
    scores = safety.safety_eval(lambda p: "I can't help with that." if "unsafe" in p else "Sure, here is how.")
    assert scores["refusal_rate"] == 100.0 and scores["over_refusal_rate"] == 0.0
    # the converters for the new SFT sources, and the generated safety set through prepare_sft
    assert convert_prompt_response({"prompt": "p", "response": "r"})[1]["content"] == "r"
    assert convert_columns_chat({"system": "s", "user": "u", "assistant": "a"})[0]["role"] == "system"
    assert "<|think|>" in convert_medical_o1({"Question": "q", "Complex_CoT": "t", "Response": "r"})[1]["content"]
    local = tmp_path / "data" / "local" / "codeswitch-chat"
    local.mkdir(parents=True)
    local.joinpath("d.jsonl").write_text(
        json.dumps(
            {"messages": [{"role": "user", "content": "Wah gwaan"}, {"role": "assistant", "content": "Mi deh yah."}]}
        )
        + "\n"
    )
    sources = (Source("safety-chat", "(generated)", "messages", 0.5, kind="sft"),
               Source("codeswitch-chat", "(local)", "messages", 0.5, kind="sft"))  # fmt: skip
    sft_dir = prepare_sft(tmp_path / "data", corpus[1], seq_len=1024, sources=sources, log=lambda *_: None)
    assert TokenSource("s", sft_dir / "tokens" / "safety-chat").masks is not None
    assert TokenSource("c", sft_dir / "tokens" / "codeswitch-chat").total > 0


@pytest.mark.parametrize(
    "name", ["execution_trace", "code_feedback", "commit", "github_issue", "guideline", "chat_transcript"]
)
def test_renderers_return_none_on_empty_rows(name):
    assert RENDERERS[name]({}) is None
