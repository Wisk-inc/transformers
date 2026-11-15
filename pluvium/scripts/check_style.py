# Copyright 2026 Nathan. Apache-2.0.
"""Check the papers and the model card against Wikipedia's field guide "Signs of AI writing".

    python scripts/check_style.py [files ...]        (default: the papers, the model card, the readme)

The guide (https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing, read on 15 November 2026) lists habits of
machine-written text. It says itself that the signs are symptoms: the trouble underneath is prose that is generic
where it should be specific, and claims nobody can check. So this script is a second reader, not the goal. A page
that passes can still be vague; a page should pass because each sentence says one checkable thing in plain words.

What is checked, with the section of the guide it comes from:

``words``       the "AI vocabulary" lists, 2023 to 2026 (pivotal, underscore, leverage ...)
``puffery``     undue emphasis on significance (stands as, plays a key role, a testament to ...)
``trailing``    superficial analysis hung on the end of a sentence (", highlighting the ...")
``vague``       vague attribution and vague connection (experts say, associated with ...)
``formula``     outline-like endings (Despite its ..., In conclusion, Future outlook)
``copula``      avoiding "is" and "has" (serves as, boasts, features a ...)
``parallel``    negative parallelism (not just X but Y; it is not X, it is Y)
``chat``        text meant for the person who asked (I hope this helps, as of my last update)
``hedge``       disclaimers about sources (based on available information, should not be confused with)
``style``       title-case headings, a bold phrase opening a list item, emoji, curly quotes, em dashes

Reported but not failed: sentences that all sit between 8 and 20 words with nothing but full stops and commas (the
guide's newest sign), how often "rather than" is used, and lists of exactly three.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ["docs/papers/*.md", "hub/README.md", "README.md"]

WORDS = (r"additionally|align(?:s|ed|ing)? with|boasts?|bolster(?:s|ed|ing)?|crucial(?:ly)?|deep dive|delv(?:e|es|ed|ing)|"
         r"emphasi[sz](?:e|es|ed|ing)|enduring|enhanc(?:e|es|ed|ing|ement|ements)|foster(?:s|ed|ing)?|garner(?:s|ed|ing)?|"
         r"highlight(?:s|ed|ing)?|interplay|intricate|intricac(?:y|ies)|landscape|meticulous(?:ly)?|pivotal|"
         r"showcas(?:e|es|ed|ing)|tapestry|testament|underscor(?:e|es|ed|ing)|vibrant|leverag(?:e|es|ed|ing)|"
         r"multifaceted|prioriti[sz](?:e|es|ed|ing)|seamless(?:ly)?|groundbreaking|cutting-edge|state-of-the-art|"
         r"game-chang(?:er|ing)|holistic(?:ally)?|synerg(?:y|ies)|paradigm|realm|nestled|renowned|myriad|plethora|"
         r"utili[sz](?:e|es|ed|ing)|comprehensive(?:ly)?|notably|moreover|furthermore")
CHECKS = {
    "words": rf"\b(?:{WORDS})\b",
    "puffery": (r"\b(?:stands? as|is a reminder|(?:plays?|played) an? (?:\w+ )?role|key (?:role|moment|turning point)|"
                r"reflects? (?:a )?broader|setting the stage|marks? a (?:shift|turning)|represents? a (?:\w+ )?shift|"
                r"evolving|focal point|indelible|deeply rooted|turning point|cornerstone|at the forefront|"
                r"(?:rich|proud) (?:history|heritage|tradition)|valuable insights?|important to note|worth noting)\b"),
    "trailing": (r",\s+(?:highlighting|underscoring|emphasi[sz]ing|ensuring|reflecting|symboli[sz]ing|contributing to|"
                 r"cultivating|fostering|encompassing|enhancing|showcasing|demonstrating|illustrating|paving the way|"
                 r"making it|solidifying|cementing)\b"),
    "vague": (r"\b(?:experts? (?:argue|say|believe|agree|note)|observers|industry reports?|some critics|"
              r"(?:several|many|numerous) (?:sources|publications|studies|researchers|experts)|widely (?:regarded|"
              r"recogni[sz]ed|considered|seen)|is (?:often|generally) (?:regarded|considered)|(?:closely )?associated "
              r"with|in connection with|connected (?:to|with)|studies (?:show|suggest|have shown))\b"),
    "formula": (r"\b(?:despite (?:its|their|these|this)|faces? (?:several |many |significant )?challenges|future "
                r"(?:outlook|prospects)|challenges and|in (?:summary|conclusion)|to (?:summari[sz]e|conclude)|overall,|"
                r"in essence|ultimately,|at its core|when it comes to|it is clear that|looking ahead)\b"),
    "copula": r"\b(?:serv(?:es?|ed|ing) as|functions? as|operates? as|boasts?|features? an? |offers? an? |refers to)\b",
    "parallel": (r"\b(?:not (?:just|only|merely|simply)|isn't (?:just|only|merely)|more than just|"
                 r"is not (?:an? |the )?[\w -]{1,40}[,;] (?:it|but)|no [\w-]+, no [\w-]+)\b"),
    "chat": (r"(?:i hope this helps|let me know|would you like|certainly!|of course!|as an ai\b|as a large language|"
             r"my (?:last|latest) (?:training|knowledge)|knowledge cutoff|here is an?\b|here's an?\b|feel free to)"),
    "hedge": (r"\b(?:based on available information|not (?:extensively|widely|well) documented|should not be confused "
              r"with|does not establish|readily available sources|it is important to|it should be noted)\b"),
}
SOFT_WORDS = r"\b(?:robust|key|valuable|significant(?:ly)?|challenge|clearer|dependable|echoed|matters|practical|quietly|steady|universally)\b"
# words that may be capitalised inside a heading without making it title case
PROPER = {"Pluvium", "Blend", "Cyclone", "ECMWF", "NOAA", "AIFS", "GraphCast", "GraphCastGFS", "AIGFS", "GFS", "GEFS",
          "ENS", "HRES", "CMORPH", "MRMS", "Melissa", "Jamaica", "Caribbean", "Alps", "Andes", "Pacific", "Northwest",
          "Great", "Plains", "Maritime", "Continent", "Southern", "US", "East", "North", "Atlantic", "Hub", "GPU", "CPU",
          "Hugging", "Face", "Brier", "CRPS", "RMSE", "ACC", "ROC", "CSI", "I", "DeepMind", "Google", "PyPI", "Python",
          "Apache", "CC", "BY", "Hurricane", "NHC", "OFCL", "TVCN", "HCCA", "NatureV1", "iHELIX", "ERA5", "GDAS", "IFS",
          "December", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "Wikipedia", "Pickands", "Pareto", "Schaake", "Horvitz", "Thompson", "Bernstein", "DJF", "A", "B"}
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐✅❌]")


def strip_code(text: str) -> str:
    """The prose of a Markdown page: code blocks, inline code, link targets and the front matter are blanked, line
    numbers kept."""
    text = re.sub(r"\A---\n.*?\n---\n", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    text = re.sub(r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "``", text)
    text = re.sub(r"\]\([^)\s]*\)", "]()", text)
    return re.sub(r"<!--.*?-->", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)


def is_title_case(heading: str) -> bool:
    """Whether a heading capitalises its ordinary words (sentence case capitalises only the first and names)."""
    words = [w.strip("():,.'\"") for w in re.sub(r"^[#\s\d.]+", "", heading).split()]
    words = [w for w in words if w and w[0].isalpha()]
    ordinary = [w for w in words[1:] if w not in PROPER and not w.isupper() and not any(c.isdigit() for c in w)]
    return len(ordinary) >= 2 and sum(w[0].isupper() for w in ordinary) / len(ordinary) > 0.5


def findings(text: str) -> list:
    """``[(line, check, what)]`` for everything in ``text`` that the guide lists."""
    prose, out = strip_code(text), []
    for number, line in enumerate(prose.split("\n"), 1):
        low = line.lower()
        for name, pattern in CHECKS.items():
            out += [(number, name, m.group(0).strip()) for m in re.finditer(pattern, low)]
        if line.startswith("#") and is_title_case(line):
            out.append((number, "style", "title-case heading: " + line.strip()))
        if re.match(r"\s*(?:[-*]|\d+\.)\s+\*\*[^*]+\*\*\s*[:.–-]?", line) and not line.lstrip().startswith("- **Heavy rain (50 mm)."):
            out.append((number, "style", "bold phrase opening a list item: " + line.strip()[:50]))
        if EMOJI.search(line):
            out.append((number, "style", "emoji"))
        if re.search("[“”‘’]", line):
            out.append((number, "style", "curly quotation mark"))
        if "—" in line:
            out.append((number, "style", "em dash"))
    return out


def rhythm(text: str) -> dict:
    """How even the sentences are: the share with 8 to 20 words, and how often anything but a full stop or a comma
    is used. The guide's newest sign is prose in which every sentence is that length and that plain."""
    prose = strip_code(text)
    body = "\n".join(line for line in prose.split("\n") if line.strip() and not line.lstrip().startswith(("#", "|", "-", "*", ">")))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if len(s.split()) >= 3]
    lengths = [len(s.split()) for s in sentences]
    words = max(1, sum(lengths))
    return {"sentences": len(sentences),
            "share_8_to_20": sum(8 <= n <= 20 for n in lengths) / max(1, len(lengths)),
            "other_marks_per_1000": 1000 * len(re.findall(r"[;:()]", body)) / words,
            "rather_than_per_1000": 1000 * len(re.findall(r"\brather than\b", body.lower())) / words,
            "soft_words_per_1000": 1000 * len(re.findall(SOFT_WORDS, body.lower())) / words,
            "bold_per_1000": 1000 * len(re.findall(r"\*\*[^*]+\*\*", body)) / words,
            "threes": len(re.findall(r"\b\w+, \w+,? and \w+\b", body))}


def main(argv) -> int:
    paths = [Path(p) for p in argv] or sorted(p for pattern in DEFAULT for p in ROOT.glob(pattern))
    bad = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        found, r = findings(text), rhythm(text)
        bad += len(found)
        name = path.relative_to(ROOT) if path.is_absolute() and ROOT in path.parents else path
        print(f"{name}: {len(found)} to fix; {r['sentences']} sentences, {r['share_8_to_20']:.0%} of them 8 to 20 words, "
              f"{r['other_marks_per_1000']:.0f} colons, semicolons or brackets per 1000 words, "
              f"{r['threes']} lists of three, 'rather than' {r['rather_than_per_1000']:.1f} per 1000")
        for line, check, what in found:
            print(f"  {line:4d}  {check:9s} {what}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
