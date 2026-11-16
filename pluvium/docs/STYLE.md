# How the pages here are written

These are the rules for the readme, the card on the Hub and anything else people read first. I keep them here so every release is written the same way.

## The readme and the card

- They are the same text. `hub/README.md` is `README.md` with the Hub's header on top, a test checks that.
- Short. Three pieces of writing, what it is, is it standalone, can it tell the time, and tables for everything else.
- Each piece starts from a question I really asked or got asked, in the words it was asked in.
- Written the way I talk. One thought can run on with commas, that is fine. No selling it.
- It says what is not finished and where it loses. The numbers for the losses stay in.
- One code block only, how to install it and run it. Sample output and the long list of commands go in `GUIDE.md`.
- Nothing about data sources in a table there either, that list is in `GUIDE.md`.

## Everything else

- The long pages (`GUIDE.md`, the benchmarks, the papers) are plain and exact, every number comes from a result file.
- Comments in the code say why something is there or what went wrong before. They do not repeat the line under them.
- A release note says what was broken, in one or two sentences, then what changed.

## The check

```bash
python scripts/check_style.py                     # the pages
python scripts/check_style.py pluvium/*.py        # comments, docstrings and messages in code
```

It looks for the words and sentence shapes listed in Wikipedia's "Signs of AI writing" page, and the tests run it on every page and every Python file. If it finds something, reword the sentence, do not switch the check off.
