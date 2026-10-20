# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Code-switched Jamaican dialogues, made to order.

There is no public corpus of Jamaicans switching between Standard Jamaican English and Patois in a bank, a
courtroom, a clinic or a classroom. This module makes one: thousands of scenarios across those domains are
rendered into conversations by a teacher model behind an OpenAI-compatible endpoint (a local vLLM serving an
Apache-2.0 model, never a closed vendor's output), guided by a style guide that carries real Patois from the
corpus as examples. Every dialogue is checked -- enough turns, Patois actually present where it should be,
no duplicates -- then written twice: as conversations for the fine-tune and as transcripts for pretraining.

Synthetic data is labelled as such everywhere it is stored. A Jamaican linguist reviewing a sample before the
run is the right next step; this code makes the sample.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .jamaica import PARISHES, PATOIS_PHRASES, PLACES

DOMAINS: dict[str, list[str]] = {
    "banking": [
        "a customer's debit card was declined at a supermarket and they call the bank",
        "a customer disputes a charge they do not recognise on their statement",
        "someone wants to open a savings account and asks what documents and TRN they need",
        "a customer asks how to send money to family abroad and about the fees and limits",
        "a customer is behind on a car loan and asks what options exist before it goes to collections",
        "a customer got a call saying they won a lottery and must pay a fee first; the agent explains how to "
        "recognise the scam and report it",
        "a small business owner asks about a point-of-sale machine and settlement times",
        "a customer's mobile wallet top-up did not arrive and they want it traced",
        "an elderly customer asks the agent to explain an interest rate and a late fee on a credit card",
        "a customer asks why the bank blocked a transaction and how to lift the block",
    ],
    "legal": [
        "a tenant received a notice to quit and asks a legal aid clinic what it means and what the law requires",
        "a man got a traffic ticket and asks how the Parish Court process works and whether he must attend",
        "a woman asks how to file for a maintenance order for her child",
        "someone asks how to register a land title and what the National Land Agency needs",
        "a family asks about probate and letters of administration after a relative died without a will",
        "a witness is being prepared to give a statement and asks what will happen in court",
        "a worker asks whether they were dismissed fairly and what the Industrial Disputes Tribunal does",
        "a small claims matter: a contractor took a deposit and never finished the work",
        "a person asks about bail after a relative was arrested",
        "a vendor asks what a municipal corporation requires for a street-vending licence",
    ],
    "health": [
        "a mother at a health centre describes her child's fever and the nurse triages",
        "a patient asks how to get their diabetes medication through the National Health Fund card",
        "a pregnant woman books an antenatal clinic appointment and asks what to bring",
        "a caller to a helpline says they have not been sleeping and feel hopeless; the counsellor listens and "
        "connects them with support",
        "a patient asks whether their symptoms sound like dengue and what to do",
        "a pharmacist explains how to take a prescription: how many times a day and with food",
        "a man asks about blood pressure numbers the clinic gave him",
        "a patient asks what to expect at a public hospital emergency department and about wait times",
    ],
    "education": [
        "a student asks a teacher for help with fractions before a PEP exam",
        "a parent asks about PEP results and school placement",
        "a school leaver asks about HEART/NSTA Trust courses and how to apply",
        "a student asks how to apply to UWI Mona and about tuition and the Students' Loan Bureau",
        "a student asks for CSEC English essay feedback",
        "a student asks a tutor to explain a Python error message in their homework",
        "a teacher explains photosynthesis to a Grade 7 class",
    ],
    "customer service": [
        "a customer queries a high light bill and asks about a meter check",
        "a customer's water was disconnected and they ask how to reconnect and pay arrears",
        "a customer's phone data plan ran out early and they want an explanation",
        "a property owner asks how to pay property tax online and what happens if late",
        "someone asks how to renew a passport and what the appointment needs",
        "a customer asks a courier where their package is after a hurricane delayed deliveries",
    ],
    "coding": [
        "a junior developer asks a senior one, in Patois, why their Python script crashes with a KeyError",
        "two developers discuss, switching between Patois and English, how to fix a failing test in a Flask app",
        "a student asks for help understanding a JavaScript promise that never resolves",
    ],
}

REGISTERS = {
    "codeswitch": "The agent speaks Standard Jamaican English and the customer speaks Patois; both switch registers "
    "naturally mid-conversation the way Jamaicans do (the agent drops into Patois to reassure, the customer uses "
    "formal English for official details).",
    "patois": "Both speakers mostly use Jamaican Patois, with official terms (names of forms, laws, medications) in "
    "English, as happens in real life.",
    "sje": "Both speakers use Standard Jamaican English throughout, with Jamaican vocabulary and institutions.",
}

PATOIS_MARKERS = re.compile(
    r"\b(mi|yuh|unu|dem|deh|nuh|wah|fi|inna|weh|gwaan|likkle|cyaan|seh|bredda|pickney|mek|nyam|tank|ting|dat|dis|"
    r"wi|im|ar|bout|outta|a go|cho|yaad|bwoy|gyal|nuff|wha|waan)\b",
    re.I,
)

STYLE_GUIDE = """You write realistic Jamaican conversations for training a language model built in Jamaica.

Rules:
- Jamaican Patois (Patwa) is a language with its own grammar: "mi a go" (I am going), "im did sell" (he/she sold),
  "dem nuh know" (they don't know), "unu" (you, plural), "fi" (to/for), "deh" (to be at / is), "a" (is / progressive).
- Spell Patois the way Jamaicans actually write it in messages and news comments (e.g. "weh yuh deh", "mi cyaan",
  "likkle more", "wah gwaan", "nuh badda"), not as mocked or exaggerated dialect. Never caricature.
- Use real Jamaican institutions, places and terms: parishes ({parishes}); places like {places}; TRN, NIS, NHT, NHF,
  PATH, JPS, NWC, TAJ, PICA, HEART/NSTA Trust, UWI Mona, UTech, Parish Court, Supreme Court, Legal Aid Council,
  Bank of Jamaica, FSC, JMD and J$ amounts, CSEC, PEP.
- Be factual about procedures; when a detail could vary, have the agent say to confirm with the office.
- For legal, medical and financial matters the professional stays within their role: information and next steps,
  not a diagnosis or a verdict; emergencies go to 119 (police) or 110 (fire and ambulance); a caller in distress is
  given the mental health helpline 888-NEW-LIFE (888-639-5433).
- Keep people's dignity. No slurs, no demeaning stereotypes, no sexual content, nothing that helps a scam or a crime
  (when a scam comes up, the conversation is about recognising, refusing and reporting it).
- Examples of natural Patois with meanings:
{examples}

Return only JSON: {{"conversation": [{{"role": "customer" | "agent", "content": "..."}}, ...], "notes": "..."}}
with 6 to 14 turns, the agent's role named for the scenario (teller, clerk, nurse, teacher, officer, developer)."""


def style_guide() -> str:
    examples = "\n".join(f"  - {p}  ->  {e}" for p, e in PATOIS_PHRASES)
    return STYLE_GUIDE.format(parishes=", ".join(PARISHES), places=", ".join(PLACES[:12]), examples=examples)


def scenarios(count: int, seed: int = 0) -> list[dict]:
    """``count`` scenario specs, spread across domains and registers, each with a distinct seed."""
    rng = random.Random(seed)
    out = []
    domains = list(DOMAINS)
    customers = ["a young man", "an elderly woman", "a market vendor", "a nurse", "a taxi operator",
                 "a university student",
                 "a farmer", "a single mother", "a returning resident", "a small shop owner", "a construction worker",
                 "a teacher"]  # fmt: skip
    for i in range(count):
        domain = domains[i % len(domains)]
        out.append(
            {
                "id": i,
                "domain": domain,
                "scenario": rng.choice(DOMAINS[domain]),
                "register": rng.choices(list(REGISTERS), weights=[0.5, 0.3, 0.2])[0],
                "parish": rng.choice(PARISHES),
                "tone": rng.choice(["calm", "frustrated but polite", "worried", "cheerful", "in a hurry", "confused"]),
                "customer": rng.choice(customers),
            }
        )
    return out


def prompt_for(spec: dict) -> list[dict]:
    user = (
        f"Scenario: {spec['scenario']}.\nDomain: {spec['domain']}. Setting: {spec['parish']}, Jamaica.\n"
        f"The customer is {spec['customer']}, {spec['tone']}.\nRegister: {REGISTERS[spec['register']]}\n"
        "Write the conversation."
    )
    return [{"role": "system", "content": style_guide()}, {"role": "user", "content": user}]


class Teacher:
    """An OpenAI-compatible chat endpoint (vLLM's ``/v1/chat/completions``)."""

    def __init__(self, base_url: str, model: str, api_key: str = "none", temperature: float = 0.9, timeout: int = 180):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.temperature, self.timeout = temperature, timeout

    def complete(self, messages: list[dict], max_tokens: int = 1200) -> str:
        payload = json.dumps(
            {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": max_tokens}
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )  # fmt: skip
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read())
                return body["choices"][0]["message"]["content"]
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError):
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("unreachable")


def parse_dialogue(text: str) -> list[dict] | None:
    """The teacher's JSON, tolerant of fences and chatter around it."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    turns = data.get("conversation") if isinstance(data, dict) else None
    if not isinstance(turns, list):
        return None
    cleaned = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role, content = str(turn.get("role", "")).strip().lower(), str(turn.get("content", "")).strip()
        if content and role:
            cleaned.append({"role": role, "content": content})
    return cleaned or None


def acceptable(turns: list[dict], register: str) -> bool:
    if not 4 <= len(turns) <= 20:
        return False
    text = " ".join(t["content"] for t in turns)
    if len(text) < 300:
        return False
    markers = len(PATOIS_MARKERS.findall(text))
    if register in ("patois", "codeswitch") and markers < 6:
        return False
    return not (register == "sje" and markers > 25)


def to_messages(turns: list[dict]) -> list[dict]:
    """Customer -> user, the professional -> assistant, so the fine-tune learns the professional's side."""
    out = []
    for turn in turns:
        role = (
            "user" if turn["role"] in ("customer", "user", "patient", "student", "caller", "client") else "assistant"
        )
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n" + turn["content"]
        else:
            out.append({"role": role, "content": turn["content"]})
    while out and out[0]["role"] != "user":
        out.pop(0)
    while out and out[-1]["role"] != "assistant":
        out.pop()
    return out


def transcript(spec: dict, turns: list[dict]) -> str:
    head = f"[{spec['domain']} | {spec['parish']}, Jamaica | {spec['register']} | synthetic dialogue]\n"
    return head + "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in turns) + "\n"


def synthesize(
    data_dir: str | Path,
    teacher: Teacher,
    count: int = 100_000,
    workers: int = 32,
    seed: int = 0,
    log=print,
) -> dict:
    """Make ``count`` dialogues; resumable (finished ids are recorded); writes the two local sources."""
    root = Path(data_dir) / "local"
    chat_dir, doc_dir = root / "codeswitch-chat", root / "codeswitch"
    chat_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)
    done_path = root / "codeswitch.done.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    seen_hashes: set[str] = set()
    specs = [s for s in scenarios(count, seed) if s["id"] not in done]
    log(
        f"[synth] {len(specs)} dialogues to make ({len(done)} already done) with {teacher.model} at {teacher.base_url}"
    )
    kept = rejected = failed = 0
    started = time.time()

    def make(spec: dict) -> tuple[dict, list[dict] | None, str]:
        try:
            raw = teacher.complete(prompt_for(spec))
        except Exception as error:
            return spec, None, f"error: {type(error).__name__}"
        turns = parse_dialogue(raw)
        if turns is None or not acceptable(turns, spec["register"]):
            return spec, None, "rejected"
        return spec, turns, "ok"

    with (
        open(chat_dir / f"dialogues-{seed}.jsonl", "a", encoding="utf-8") as chats,
        open(doc_dir / f"transcripts-{seed}.jsonl", "a", encoding="utf-8") as docs,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        for i, (spec, turns, status) in enumerate(pool.map(make, specs), start=1):
            done.add(spec["id"])
            if turns is None:
                failed += status.startswith("error")
                rejected += status == "rejected"
            else:
                digest = hashlib.sha1(" ".join(t["content"] for t in turns).encode()).hexdigest()
                if digest in seen_hashes:
                    rejected += 1
                else:
                    seen_hashes.add(digest)
                    messages = to_messages(turns)
                    if len(messages) >= 2:
                        record = {"messages": messages, "domain": spec["domain"], "register": spec["register"],
                                  "synthetic": True}  # fmt: skip
                        chats.write(json.dumps(record, ensure_ascii=False) + "\n")
                    docs.write(
                        json.dumps({"text": transcript(spec, turns), "synthetic": True}, ensure_ascii=False) + "\n"
                    )
                    kept += 1
            if i % 200 == 0:
                chats.flush()
                docs.flush()
                done_path.write_text(json.dumps(sorted(done)))
                rate = i / max(time.time() - started, 1)
                log(
                    f"[synth] {i}/{len(specs)}: kept {kept}, rejected {rejected}, failed {failed}, {rate * 3600:.0f}/h"
                )
    done_path.write_text(json.dumps(sorted(done)))
    summary = {"kept": kept, "rejected": rejected, "failed": failed, "total_done": len(done)}
    log(f"[synth] done: {summary}")
    return summary
