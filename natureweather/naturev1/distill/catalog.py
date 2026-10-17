# Copyright 2026 Nathan. Apache-2.0.
"""
Every teacher, described by how it sees the Earth -- the facts the alignment needs.

The teachers represent the atmosphere in incompatible ways: Pangu-Weather is a 3-D Earth-specific transformer
on a 0.25-degree latitude-longitude grid with 13 pressure levels; GraphCast and FGN (the model behind
WeatherNext 2) process an icosahedral multi-mesh but write 0.25-degree grids, GraphCast on 37 levels; Aurora
uses its own latent level encoding and writes 0.25 (or 0.1) degrees; ECMWF's AIFS runs on an N320 reduced
Gaussian grid (octahedral O96 for the processor); NVIDIA Earth-2 models (SFNO, FourCastNet) use spherical
harmonics on 0.25 degrees. NatureV1 encodes a 1.5-degree analysis onto a 4,096-point Fibonacci mesh with a
log-pressure vertical axis.

The catalog records, per teacher: where its forecasts can be read (WeatherBench 2 archives already regridded
conservatively to NatureV1's own 1.5-degree grid -- most of the spatial misalignment solved at the source --
and their 0.25-degree versions for the high-resolution decoder), which years, initial times, lead step and
horizon, its levels and variables, whether it is an ensemble, its native grid (for teachers read live, which
must be regridded here), naming quirks found in the archives, and the licence its outputs carry -- which binds
whatever is distilled from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

WB2 = "gs://weatherbench2/datasets/"


@dataclass(frozen=True)
class Teacher:
    name: str
    family: str
    native_grid: str
    native_vertical: str
    archive_1p5: dict = field(default_factory=dict)       # year -> store path (1.5 degrees, NatureV1's grid)
    archive_0p25: dict = field(default_factory=dict)      # year -> store path (0.25 degrees)
    init_hours: tuple = (0, 12)
    lead_step_h: int = 6
    max_lead_h: int = 240
    levels: tuple = ()
    variables: tuple = ()
    ensemble_dim: str | None = None
    source: str = "weatherbench2"                         # weatherbench2 | ecmwf-opendata | earth2studio
    licence: str = ""
    notes: str = ""

    def years(self) -> list[int]:
        return sorted(self.archive_1p5)


_SURFACE = ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure")
_UPPER = ("geopotential", "temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity")
_L13 = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
_L37 = (1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600,
        650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000)

TEACHERS: dict[str, Teacher] = {
    "pangu": Teacher(
        "pangu", "Huawei Pangu-Weather", "0.25-degree latitude-longitude, 3-D Earth-specific transformer",
        "13 pressure levels",
        archive_1p5={y: "pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr" for y in range(2018, 2023)},
        archive_0p25={y: "pangu/2018-2022_0012_0p25.zarr" for y in range(2018, 2023)},
        levels=_L13, variables=_SURFACE + _UPPER, licence="Pangu-Weather weights: CC BY-NC-SA 4.0 (non-commercial)",
        notes="initialised from ERA5; hierarchical 1/3/6/24 h models chained to 6-hourly output"),
    "graphcast": Teacher(
        "graphcast", "Google DeepMind GraphCast", "icosahedral multi-mesh (refinement 6), 0.25-degree output",
        "37 pressure levels",
        archive_1p5={2018: "graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr",
                     2020: "graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr"},
        archive_0p25={2018: "graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours.zarr",
                      2020: "graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours.zarr"},
        levels=_L37,
        variables=_SURFACE + _UPPER + ("vertical_velocity", "total_precipitation_6hr"),
        licence="GraphCast weights: CC BY-NC-SA 4.0; WeatherBench 2 archive: CC BY 4.0",
        notes="37 levels interpolated in log-pressure onto NatureV1's"),
    "hres": Teacher(
        "hres", "ECMWF IFS HRES (operational physics model)", "TCo1279 octahedral (~9 km)", "137 model levels -> 13",
        archive_1p5={y: "hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr" for y in range(2016, 2023)},
        archive_0p25={y: "hres/2016-2022-0012-1440x721.zarr" for y in range(2016, 2023)},
        levels=_L13, variables=_SURFACE + _UPPER + ("surface_pressure", "vertical_velocity", "total_precipitation_6hr"),
        licence="ECMWF data: CC BY 4.0 (WeatherBench 2 archive)",
        notes="starts at lead 0 (its own analysis), unlike the ERA5-initialised machine-learning teachers"),
    "aurora": Teacher(
        "aurora", "Microsoft Aurora", "0.25-degree latitude-longitude, Perceiver encoder to latent levels",
        "latent levels; archive has 3 pressure levels",
        archive_1p5={2022: "aurora/2022-240x121_equiangular_with_poles_conservative.zarr"},
        archive_0p25={2022: "aurora/2022-1440x721.zarr"},
        levels=(500, 700, 850), variables=_SURFACE + _UPPER,
        licence="Aurora: MIT licence",
        notes="archive variable names 'geopotential\\t' and 'specific_humidity\\t' carry a trailing tab"),
    "fgn": Teacher(
        "fgn", "Google DeepMind FGN (WeatherNext 2)", "icosahedral mesh graph transformer, 0.25-degree output",
        "archive has 3 pressure levels", archive_1p5={2022: "fgn/2022-240x121_equiangular_with_poles_conservative.zarr"},
        archive_0p25={2022: "fgn/2022-1440x721.zarr"}, levels=(500, 700, 850), max_lead_h=360,
        variables=_SURFACE + _UPPER + ("sea_surface_temperature", "vertical_velocity", "total_precipitation_6hr"),
        ensemble_dim="sample", licence="WeatherNext data: Google terms of use (check before distilling)",
        notes="64-member ensemble: its spread is a target variance, not just its mean"),
    "fuxi": Teacher(
        "fuxi", "Fudan FuXi", "0.25-degree latitude-longitude, cascaded U-transformer", "archive has 2 levels",
        archive_1p5={2020: "fuxi/2020-240x121_equiangular_with_poles_conservative.zarr"},
        archive_0p25={2020: "fuxi/2020-1440x721.zarr"}, levels=(500, 850),
        variables=("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure",
                   "geopotential", "temperature", "u_component_of_wind", "v_component_of_wind"),
        licence="FuXi: research licence"),
    "gencast": Teacher(
        "gencast", "Google DeepMind GenCast", "icosahedral mesh diffusion model, 0.25-degree output", "archive 3 levels",
        archive_1p5={2020: "gencast/2020-240x121_equiangular_with_poles_conservative.zarr"},
        archive_0p25={2020: "gencast/2020-1440x721.zarr"}, levels=(500, 700, 850), lead_step_h=12, max_lead_h=360,
        variables=_SURFACE + _UPPER + ("sea_surface_temperature",), ensemble_dim="sample",
        licence="GenCast weights: CC BY-NC-SA 4.0", notes="12-hourly steps: interpolated in time onto 6-hourly"),
    "neuralgcm": Teacher(
        "neuralgcm", "Google NeuralGCM (deterministic)", "spectral dynamical core T179 + learned physics",
        "37 pressure levels", archive_1p5={2020: "neuralgcm_deterministic/2020-240x121_equiangular_with_poles_conservative.zarr"},
        levels=_L37,
        variables=_UPPER, licence="NeuralGCM: CC BY-SA 4.0", notes="no surface fields in the archive"),
    "aifs": Teacher(
        "aifs", "ECMWF AIFS Single", "N320 reduced Gaussian grid (O96 processor)", "13 pressure levels",
        source="ecmwf-opendata", levels=_L13, variables=_SURFACE + _UPPER,
        licence="ECMWF open data: CC BY 4.0",
        notes="no WeatherBench 2 archive; read live from ECMWF open data (GRIB, recent days only) and regridded "
              "from the reduced Gaussian grid by naturev1.distill.align.points_to_grid"),
    "earth2": Teacher(
        "earth2", "NVIDIA Earth-2 (Earth2Studio: SFNO / FourCastNet v2)", "0.25-degree, spherical harmonic neural operator",
        "13 pressure levels", source="earth2studio", levels=_L13, variables=_SURFACE + _UPPER,
        licence="per model in Earth2Studio (NVIDIA Open Model Licence for most)",
        notes="no archive: run locally on the GPU from ERA5 initial conditions, then aligned like any 0.25-degree teacher"),
}

#: Teachers with a historical archive NatureV1 can be distilled from offline.
ARCHIVED = tuple(n for n, t in TEACHERS.items() if t.archive_1p5)


def teacher_years(names=ARCHIVED) -> dict[str, list[int]]:
    return {n: TEACHERS[n].years() for n in names}


__all__ = ["ARCHIVED", "TEACHERS", "Teacher", "WB2", "teacher_years"]
