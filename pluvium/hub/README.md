---
license: apache-2.0
library_name: pluvium
pipeline_tag: time-series-forecasting
language:
- en
tags:
- weather
- weather-forecasting
- climate
- meteorology
- earth-science
- precipitation
- rainfall
- extreme-rainfall
- nowcasting
- probabilistic-forecasting
- tropical-cyclones
- hurricane
- multi-model-ensemble
- post-processing
- online-learning
- geospatial
- caribbean
- pytorch
- naturev1
- ihelix
- graphcast
- aifs
metrics:
- brier_score
- crps
---

# Pluvium

Pluvium is a rain model. It says when each parish might get rain, down to the exact town or city, and the time when rain is going to fall.

I made it for Jamaica first. Then I did not want it mainly focusing on Jamaica, other countries should be able to use it too, so now you give it a country code and it sets up that country by itself. Any country between 60 south and 60 north works.

You don't need a GPU for the forecast and you don't need an account. You copy one command, you run it, it should just work. If it has to download something it downloads it once and keeps it, it does not redownload the stuff every time because I find that completely useless.

```bash
pip install pluvium
pluvium now --country JM
```

## Is it a standalone model

Is Pluvium a standalone model, or did I use another model to power it? Both. The 14 days forecast is not standalone, that one uses the forecasts ECMWF and NOAA put out free every day, six of them, some are the AI weather models. Then it looks at one year of rain from the satellite for that country to see which of the six was right more times, and those it trusts more. That is the blend. No GPU for that part, any normal computer runs it.

The one that is standalone is Pluvium-500, a specific model that forecasts rain hour by hour. It uses the NatureV1 architecture and some extended features. It only works after it is trained on a GPU.

| The six forecasts | From | Kind |
|---|---|---|
| ENS, 51 runs | ECMWF | physics |
| HRES | ECMWF | physics |
| AIFS | ECMWF | AI |
| GEFS, 31 runs | NOAA | physics |
| GFS | NOAA | physics |
| GraphCast, called AIGFS since May 2026 | NOAA | AI |

## Can it tell the time when it is going to rain

Can it tell the time when it is going to rain? For the first three days yes, but in 6 hour parts and not the exact hour. In Jamaica the parts are 7 pm to 1 am, 1 am to 7 am, 7 am to 1 pm and 1 pm to 7 pm. The exact hour is what Pluvium-500 is for and that one needs the GPU training first.

If it says 34% for the afternoon then most times no rain is going to fall, and thats correct, not a mistake. Also places that are close get almost the same forecast, it works in squares of about 28 km.

## How good is it

| Test | Result |
|---|---|
| Rain or no rain, days 1 to 10, six regions, one full year | best of all the systems in 6 of 6 regions |
| Same, days 11 to 14 | best in 5 of 6 |
| Against each system on its own, days 1 to 7 | better in 72 of 72 comparisons, 95% confidence |
| 50 mm days, version 0.3, seven places | ahead of every other system in 15 of 28 cases |
| Winter, days 1 to 3 | ahead in only 2 of 6 regions |
| Hurricane Melissa in Jamaica, two days ahead | lost: said 66% for 50 mm, the raw systems said 83 to 100% |
| Part of the day, Jamaica, wet days | 7.8% better than the usual pattern |
| Part of the day, picking the wettest one | no better: 66.7% against 66.5% |
| Hurricane tracks, East Pacific 2025, 18 storms | lower error than ECMWF HRES and NOAA GFS at all 8 lead times |
| Hurricane tracks against the official NHC forecast | behind at every lead time, 125 against 100 nautical miles at five days |

## Is it finished

| Pluvium-500 | Status |
|---|---|
| Training cell | `notebooks/pluvium_everything.py`, needs a Hugging Face token and a big GPU |
| GPU it was made for | RTX 6000 Blackwell, 96 GB |
| GPU memory used | 95% of the card each step |
| Finished | no, first run stopped at 138 minutes |
| Hour by hour forecast | not published yet |

## What it can not do

| Can not do | Why |
|---|---|
| The exact hour | only 6 hour parts until Pluvium-500 is trained |
| Two places closer than 28 km | same forecast square |
| Rain gauge accuracy | checked against satellite rain only |
| Learn from last week | satellite rain is published 4.5 months late |
| Much past day 10 | only a little better than the usual weather |
| 50 mm days | hard for every forecast tested |

## More commands

| Command | What it gives |
|---|---|
| `pluvium now --country JM` | the biggest towns |
| `pluvium now --country JM --place "Port Antonio"` | one place by name |
| `pluvium now --country JM --when` | the 4 parts of each day, first 3 days |
| `pluvium now --country JM --parishes` | every parish |
| `pluvium now --country JM --parish "St Ann"` | the towns of one parish |
| `pluvium now --country DO,HT --storms` | two countries, and the hurricanes alive now |

## More

| Page | What is in it |
|---|---|
| [docs/GUIDE.md](docs/GUIDE.md) | the long manual |
| [docs/TRAINING.md](docs/TRAINING.md) | the training commands |
| [CHANGELOG.md](CHANGELOG.md) | every version |

Apache 2.0, CorX Labs.
