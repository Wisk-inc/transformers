# Module map

Every module of the package, what it is for, and its size. Generated from the modules' own docstrings.

| Module | Lines | Purpose |
|---|---|---|
| `pluvium/__init__.py` | 15 | Pluvium: hour-by-hour rain forecasts for every Caribbean town, on the NatureV1 / iHELIX engine. |
| `pluvium/advection.py` | 70 | Rain already falling, carried forward (experimental as an input to a neural forecaster of this kind). |
| `pluvium/area.py` | 195 | Where Pluvium forecasts: one country, several, or any box between 60 S and 60 N. |
| `pluvium/atmos/__init__.py` | 33 | Theoretical atmospheric physics computed from the driving fields and given to the network as inputs. |
| `pluvium/atmos/assemble.py` | 63 | Putting the atmospheric physics together: the extra coarse channels and the extra per-point inputs of version 3. |
| `pluvium/atmos/cyclones.py` | 96 | Tropical cyclones: too rare to learn from data, too important to miss. |
| `pluvium/atmos/diurnal.py` | 40 | The daily clock of tropical rain, written down instead of learned. |
| `pluvium/atmos/moisture.py` | 66 | The column's water, and the most robust law of tropical rain. |
| `pluvium/atmos/saharan.py` | 36 | The Saharan air layer (SAL): why a moist-looking summer forecast stays dry. |
| `pluvium/atmos/sea_breeze.py` | 50 | The sea-breeze front: where island afternoon storms are born. |
| `pluvium/atmos/thermo.py` | 128 | Moist thermodynamics, done properly. |
| `pluvium/atmos/upslope.py` | 56 | Wind meeting terrain, at every town and every hour. |
| `pluvium/atmos/waves.py` | 39 | Tropical (easterly) waves: the main source of Caribbean rain from June to November. |
| `pluvium/bench.py` | 254 | A fair fight between architectures on real, held-out Caribbean data. |
| `pluvium/benchmarks/__init__.py` | 3 | Benchmarks that need more than :mod:`pluvium.bench`: Pluvium against the operational forecasting systems on real forecasts (:mod:`pluvium.benchmarks.operational`). |
| `pluvium/benchmarks/fullyear.py` | 684 | A full year, 14 days ahead, around the world: Pluvium Blend against the operational systems. |
| `pluvium/benchmarks/operational.py` | 885 | Pluvium against the operational forecasting systems, on real forecasts only. |
| `pluvium/blend.py` | 181 | Pluvium Blend: one calibrated forecast from every operational system and Pluvium's own network. |
| `pluvium/blendlive.py` | 169 | Pluvium Blend for any country, every day. |
| `pluvium/calibrate.py` | 147 | Making the stated chances true: isotonic recalibration fitted on held-out forecasts. |
| `pluvium/card.py` | 146 | The model card written next to a trained Pluvium: what it is, how it was trained, and how well it did. |
| `pluvium/chain.py` | 256 | The intensity chain (experimental): rain at a place as a hidden Markov chain over *intensity states* -- dry, light, moderate, heavy -- whose hour-by-hour transition matrix the network writes for every place. |
| `pluvium/cli.py` | 233 | ``pluvium forecast / place / program / info``. |
| `pluvium/convection.py` | 90 | The convective environment: how ready the air is to make thunderstorms, from the driving model's temperature, humidity and wind at 850, 700, 500 and 200 hPa (the forecaster's classic instability and moisture indices). |
| `pluvium/dataset.py` | 424 | Training examples: one forecast issued at hour ``t0``, built from the staged caches. |
| `pluvium/domain.py` | 551 | The ground Pluvium forecasts for: the Caribbean as points the model thinks on, the cells it learns from, the towns it speaks about, and what the terrain looks like under each of them. |
| `pluvium/engine/__init__.py` | 28 | Neural components of the version 3 engine, each aimed at a weakness of the version 2 decoder: |
| `pluvium/engine/experts.py` | 38 | A mixture of experts for different kinds of rain. |
| `pluvium/engine/lead_film.py` | 53 | Lead time as a first-class input, and predictability that decays. |
| `pluvium/engine/mc_heads.py` | 31 | Knowing what it does not know. |
| `pluvium/engine/refiner.py` | 71 | Local attention among decoded points. |
| `pluvium/engine/temporal_mixer.py` | 34 | Attention along a point's own future. |
| `pluvium/engine/terrain_patch.py` | 55 | Reading the ground around every town. |
| `pluvium/features.py` | 304 | Diagnostics computed from the driving fields before the network sees them: where the fronts and shear lines are, when each one (and each band of rain the driving model shows) will reach every point, where island heating will pull the afternoon storms, where the sun is, and which kind of weather is making the rain. |
| `pluvium/flood.py` | 180 | Flood risk: rain is only half of it. |
| `pluvium/forecast.py` | 544 | The live forecast: for every town and village in the Caribbean, hour by hour for three days -- the chance of rain, whether it will be drizzle, showers, steady or heavy rain or thunderstorms, when it is most likely to start (in local time) and when it will be heaviest, how much may fall, the flash-flood risk, and why. |
| `pluvium/guard/__init__.py` | 24 | What can go wrong when the model is running, and what Pluvium does about it. |
| `pluvium/guard/consistency.py` | 71 | Outputs that contradict each other, found and repaired. |
| `pluvium/guard/drift.py` | 58 | Situations unlike anything in training. |
| `pluvium/guard/fallback.py` | 67 | A forecast when the network cannot run. |
| `pluvium/guard/freshness.py` | 47 | Old data posing as new. |
| `pluvium/guard/inputs.py` | 112 | Checking and repairing driving fields before the model sees them. |
| `pluvium/hydro/__init__.py` | 18 | From rain to flood, the hydrology the rule in :mod:`pluvium.flood` leaves out: |
| `pluvium/hydro/antecedent.py` | 39 | How wet the ground is before the storm. |
| `pluvium/hydro/hydrograph.py` | 56 | When the flood peaks. |
| `pluvium/hydro/runoff.py` | 47 | How much rain becomes runoff: the SCS curve-number method (USDA Soil Conservation Service, TR-55). |
| `pluvium/insight/__init__.py` | 16 | Understanding the model: why it forecast what it did, and where it goes wrong. |
| `pluvium/insight/attribution.py` | 68 | Why the forecast says what it says: occlusion attribution. |
| `pluvium/insight/cases.py` | 42 | Case replays: one day, forecast against truth, hour by hour. |
| `pluvium/insight/errors.py` | 50 | Where the model fails: verification split by situation. |
| `pluvium/losses.py` | 202 | What training maximises: the likelihood of what was observed, through what each observer is known to get wrong. |
| `pluvium/model.py` | 591 | Pluvium-500: rain, hour by hour, for every Caribbean town, on the NatureV1 / iHELIX engine. |
| `pluvium/narrative.py` | 98 | Forecasts in words: what each hour will be like at a place ("light showers", "thunderstorms", "heavy rain"), and a sentence per town and per parish ("Showers likely from about 2 pm (between 1 and 5 pm), heaviest around 3 pm with thunder possible; 5-20 mm, locally 35 mm. |
| `pluvium/online.py` | 242 | Learning from its own mistakes, live. |
| `pluvium/ops/__init__.py` | 14 | Running Pluvium as a service: |
| `pluvium/ops/api.py` | 93 | A small JSON API over the latest forecast (standard library only). |
| `pluvium/ops/loop.py` | 112 | The operational cycle, once an hour. |
| `pluvium/physics.py` | 177 | Physics the network is given rather than asked to discover. |
| `pluvium/prob/__init__.py` | 23 | Probability done carefully, where point-by-point forecasts fall short: |
| `pluvium/prob/conformal.py` | 52 | Prediction intervals that keep their promise. |
| `pluvium/prob/copula.py` | 74 | Spatially coherent samples. |
| `pluvium/prob/diagnostics.py` | 57 | Is the uncertainty right? |
| `pluvium/prob/extremes.py` | 91 | The tail: cloudbursts and their return periods. |
| `pluvium/prob/scores.py` | 88 | Scores that a single mean error hides. |
| `pluvium/products.py` | 192 | From the heads' raw outputs to what a person asks: will it rain here, when will it start, how hard, will there be thunder, how long will it last, how much will fall, and why. |
| `pluvium/program.py` | 620 | Pluvium-500 from nothing on one GPU, in stages that each finish or resume: |
| `pluvium/render/__init__.py` | 16 | Forecasts in the forms people and agencies use: |
| `pluvium/render/alerts.py` | 77 | Alerts in the Common Alerting Protocol (OASIS CAP 1.2). |
| `pluvium/render/bulletin.py` | 105 | A national rain bulletin. |
| `pluvium/render/maps.py` | 59 | Maps: one picture of where rain is likely. |
| `pluvium/schedule.py` | 67 | The forecast's clock: which hours each frame stands for, shared by training, live input and the products. |
| `pluvium/sources/__init__.py` | 3 | Where the data comes from: ERA5 (training predictors), CMORPH and airport reports (truth), GFS/ECMWF and GOES (live). |
| `pluvium/sources/archive.py` | 259 | Every operational system's rain, read once per run and cut to many regions at the same time. |
| `pluvium/sources/baselines.py` | 220 | The operational systems Pluvium is measured against, read exactly as their producers publish them. |
| `pluvium/sources/cmorph.py` | 235 | The truth Pluvium learns rain from: NOAA's CMORPH Climate Data Record, 8 km every 30 minutes since 1998 (``noaa-cdr-precip-cmorph-pds`` on AWS Open Data, about 1.6 MB an hour for the whole band 60 S - 60 N). |
| `pluvium/sources/cmorph_daily.py` | 152 | Daily rain truth for any region of the world between 60 S and 60 N: CMORPH, summed over each UTC day and averaged onto the 0.25-degree grid the operational systems publish. |
| `pluvium/sources/era5.py` | 218 | Training predictors: ERA5 reanalysis from the public ARCO and WeatherBench 2 stores on Google Cloud. |
| `pluvium/sources/glm.py` | 81 | Lightning now: flashes seen by GOES-East's Geostationary Lightning Mapper in the last minutes (``noaa-goes19``, ``GLM-L2-LCFA``, one file per 20 seconds), counted around every town. |
| `pluvium/sources/goes.py` | 162 | Rain already falling, live, from GOES (the forecast's "history" frames). |
| `pluvium/sources/metar.py` | 80 | Live airport reports: the last hours of METARs from every Caribbean airport, from the Aviation Weather Center's open API (``aviationweather.gov``), turned into the same per-hour codes the model was trained on (:mod:`pluvium.weather`). |
| `pluvium/sources/mrms.py` | 84 | Radar where the Caribbean has it: NOAA's Multi-Radar Multi-Sensor system, Caribbean domain (``noaa-mrms-pds``, ``CARIB/``), covering Puerto Rico, the Virgin Islands and the waters around them. |
| `pluvium/sources/nwp.py` | 382 | Live driving fields: GFS (NOAA, hourly to 5 days), ECMWF IFS and AIFS (ECMWF open data, CC-BY-4.0), read by byte range from their public buckets and averaged onto the exact grids the model was trained on (ERA5's 1 degree hourly and 1.5 degree upper-air), so live input looks like training input. |
| `pluvium/sources/stations.py` | 204 | Point truth: hourly weather reports from about 250 Caribbean airports and stations (NOAA's Integrated Surface Database, ``noaa-global-hourly-pds`` on AWS Open Data). |
| `pluvium/synthetic.py` | 206 | A synthetic Caribbean for tests and demonstrations: an island with a mountain spine, trade winds, afternoon storms, cold fronts that sweep south in the cool season, and rain made where the wind meets the slope -- with "reanalysis" fields and "CMORPH" truth produced by the same weather, so the whole pipeline can be trained and scored without a network. |
| `pluvium/train.py` | 571 | Training on one GPU: batches built on every CPU core, bf16 on the GPU, the fastest batch size found by measuring, a learning-rate schedule that ends when the time budget does, checkpoints that resume exactly, and (optionally) a mirror of every checkpoint on the Hugging Face Hub so a rented machine that stops loses nothing. |
| `pluvium/training/__init__.py` | 20 | How the network is taught, beyond the loss: |
| `pluvium/training/augment.py` | 53 | Perturbations with a physical meaning, applied to training examples. |
| `pluvium/training/curriculum.py` | 58 | Showing the network the rare events. |
| `pluvium/training/distill.py` | 38 | Distillation: a small Pluvium that forecasts like the large one. |
| `pluvium/training/ema.py` | 50 | Polyak averaging of the weights. |
| `pluvium/util.py` | 144 | Small shared pieces: logging, HTTP with retries and byte ranges, parallel fetching, and the hour clock. |
| `pluvium/verify.py` | 218 | Scoring forecasts against what fell, and against the forecasts a person could have used instead. |
| `pluvium/weather.py` | 283 | What the weather *is*, in the words people use: intensity classes, drizzle versus rain versus showers, thunder. |
| `pluvium/zarrlite.py` | 166 | A minimal reader for consolidated Zarr v2 stores over HTTPS (the public ERA5 stores on Google Cloud). |

94 modules, 13,900 lines.
