# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Jamaica in the training data.

Two things live here. The **filter** that recognizes a document about Jamaica while the English web and
Wikipedia stream past, so those documents can be written to their own source and read more than once. And
the **facts** -- parishes, places, institutions, history, language -- that the identity documents are
generated from, so what the model says about the country is at least not made up by a template.
"""

from __future__ import annotations

import re

PARISHES = (
    "Kingston", "St. Andrew", "St. Thomas", "Portland", "St. Mary", "St. Ann", "Trelawny", "St. James",
    "Hanover", "Westmoreland", "St. Elizabeth", "Manchester", "Clarendon", "St. Catherine",
)  # fmt: skip

COUNTIES = {"Surrey": PARISHES[:4], "Middlesex": PARISHES[4:7] + PARISHES[11:14], "Cornwall": PARISHES[7:11]}

PLACES = (
    "Kingston", "Montego Bay", "Spanish Town", "Portmore", "Ocho Rios", "Negril", "Mandeville", "May Pen",
    "Port Antonio", "Savanna-la-Mar", "Falmouth", "Black River", "Morant Bay", "Half Way Tree", "Port Royal",
    "Blue Mountains", "Blue Mountain Peak", "Cockpit Country", "Dunn's River Falls", "Rio Grande",
    "Black River Morass",
    "Treasure Beach", "Lucea", "Linstead", "Old Harbour", "Santa Cruz", "Christiana", "Brown's Town",
)  # fmt: skip

INSTITUTIONS = (
    "University of the West Indies, Mona", "University of Technology, Jamaica", "Northern Caribbean University",
    "Bank of Jamaica", "Jamaica Stock Exchange", "Jamaica Information Service", "Jamaica Gleaner", "Jamaica Observer",
    "Meteorological Service of Jamaica", "Office of Disaster Preparedness and Emergency Management",
    "Jamaica Defence Force", "Jamaica Constabulary Force", "Planning Institute of Jamaica", "CARICOM",
)  # fmt: skip

FACTS = (
    "Jamaica is an island country in the Caribbean Sea, the third-largest island of the Greater Antilles after Cuba "
    "and Hispaniola, about 235 km long and up to 82 km wide.",
    "The capital and largest city is Kingston, on the south-east coast; Montego Bay in St. James is the second city "
    "and the main tourist gateway.",
    "Jamaica has 14 parishes grouped into three counties: Cornwall in the west, Middlesex in the centre and Surrey in "
    "the east.",
    "Jamaica became independent from the United Kingdom on 6 August 1962 and is a member of CARICOM and the "
    "Commonwealth.",
    "The official language is English; most Jamaicans also speak Jamaican Patois (Patwa), an English-based creole "
    "with West African influences, written in the Cassidy-JLU orthography or in ad-hoc English spelling.",
    "Blue Mountain Peak, at 2,256 m, is the highest point; the Blue and John Crow Mountains are a UNESCO World "
    "Heritage Site, and the Blue Mountains grow one of the world's best-known coffees.",
    "The currency is the Jamaican dollar (JMD, J$); the Bank of Jamaica is the central bank.",
    "Jamaica's population is about 2.8 million; Kingston's metropolitan area holds roughly a fifth of it.",
    "The hurricane season runs from June to November. Gilbert (1988), Ivan (2004), Dean (2007), Sandy (2012), Beryl "
    "(2024) and Melissa (2025) are among the storms that struck or brushed the island.",
    "Hurricane Melissa made landfall in Westmoreland on 28 October 2025 as a Category 5 hurricane, the strongest "
    "landfall recorded in Jamaica.",
    "Reggae, ska, rocksteady and dancehall were born in Jamaica; Bob Marley, Peter Tosh, Jimmy Cliff and Burning "
    "Spear carried reggae to the world, and the music was inscribed by UNESCO as intangible cultural heritage in "
    "2018.",
    "Jamaica's athletes, among them Usain Bolt, Shelly-Ann Fraser-Pryce, Elaine Thompson-Herah, Veronica "
    "Campbell-Brown and Merlene Ottey, have made the country one of the great sprinting nations.",
    "The national dish is ackee and saltfish; jerk seasoning, curried goat, rice and peas, bammy and patties are "
    "staples of Jamaican cooking.",
    "The national motto is 'Out of Many, One People'; the flag is black, green and gold.",
    "The University of the West Indies has its oldest campus at Mona, in St. Andrew, founded in 1948.",
    "Port Royal, once the largest city in the Caribbean and a haven for privateers, was largely destroyed by the "
    "earthquake of 1692.",
    "The Maroons, descendants of Africans who escaped slavery, won treaties with the British in 1739 and keep "
    "communities at Accompong, Moore Town, Charles Town and Scott's Hall.",
    "Marcus Garvey, born in St. Ann's Bay in 1887, is Jamaica's first National Hero; the others are Paul Bogle, "
    "George William Gordon, Nanny of the Maroons, Sam Sharpe, Norman Manley and Alexander Bustamante.",
    "Kingston Harbour is the seventh-largest natural harbour in the world.",
    "Jamaica's time zone is UTC-5 all year; the country code is +1 876 (and +1 658).",
)  # fmt: skip

PATOIS_PHRASES = (
    ("Wah gwaan?", "What's going on? / How are you?"),
    ("Mi deh yah.", "I'm here. / I'm fine."),
    ("Everyting criss.", "Everything is fine."),
    ("Mi soon come.", "I'll be back soon."),
    ("Walk good.", "Travel safely. / Take care."),
    ("Nuh badda mi.", "Don't bother me."),
    ("Mi nuh know.", "I don't know."),
    ("Weh yuh a seh?", "What are you saying?"),
    ("Big up yuhself.", "Respect to you. / Well done."),
    ("Likkle more.", "See you later."),
    ("Mi a go a town.", "I'm going to Kingston."),
    ("Di pickney dem a play.", "The children are playing."),
    ("It a rain cat an dog.", "It is raining heavily."),
    ("Mi hungry bad.", "I'm very hungry."),
    ("Mek wi go.", "Let's go."),
)  # fmt: skip

# Terms that are strong evidence a document is about Jamaica. Case-insensitive; word-bounded where the
# term is a plain word. Weighted: the country itself and the demonym count double.
_STRONG = {
    "jamaica": 2, "jamaican": 2, "jamaicans": 2, "kingston jamaica": 3, "montego bay": 2, "ocho rios": 2,
    "patois": 1, "patwa": 2, "negril": 2, "spanish town": 1, "portmore": 2, "mandeville": 1, "port antonio": 2,
    "savanna-la-mar": 2, "morant bay": 2, "black river": 1, "blue mountain": 1, "cockpit country": 2,
    "uwi mona": 2, "university of the west indies": 1, "jamaica gleaner": 2, "jamaica observer": 2,
    "bank of jamaica": 2, "jamaican dollar": 2, "caricom": 1, "trelawny": 2, "westmoreland": 1, "st. elizabeth": 1,
    "st elizabeth": 1, "clarendon": 1, "st. catherine": 1, "st catherine": 1, "st. ann": 1, "st ann": 1,
    "manchester": 0,
    "reggae": 1, "dancehall": 1, "bob marley": 2, "usain bolt": 2, "marcus garvey": 2, "maroons": 1, "ackee": 2,
    "jerk chicken": 1, "nanny of the maroons": 3, "port royal": 1, "jamaica defence force": 3,
    "jamaica constabulary": 3, "odpem": 2, "jis.gov.jm": 3, ".gov.jm": 3, ".com.jm": 2, "876-": 1,
}  # fmt: skip
_PATTERN = re.compile(r"|".join(rf"(?<![a-z]){re.escape(t)}(?![a-z])" for t in sorted(_STRONG, key=len, reverse=True)))


def jamaica_score(text: str, max_chars: int = 20_000) -> int:
    """Weighted count of Jamaica terms in the first ``max_chars`` characters."""
    return sum(_STRONG[m.group(0).lower()] for m in _PATTERN.finditer(text[:max_chars].lower()))


def is_about_jamaica(text: str, threshold: int = 4) -> bool:
    """
    A document is about Jamaica when it carries several strong signals, not one stray mention: a travel
    piece on Negril, the Gleaner's reporting, a parish council minute, a reggae history. One ``Jamaica`` in
    a list of countries scores 2 and is left where it was.
    """
    return jamaica_score(text) >= threshold
