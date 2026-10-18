# Copyright 2026 Nathan. Apache-2.0.
"""
Tropical-cyclone physics that needs no neural network -- tracks, intensity, genesis -- built to beat
extrapolation at every lead, and measured doing so on seasons it never saw.

Every module is one idea:

==========================  =================================================================================
geometry                    great-circle moves and storm-centred frames; NHC along/cross-track errors
fields                      analysis boxes with spherical calculus and pressure-weighted layer means
data                        hindcast cases from IBTrACS + WeatherBench 2 ERA5 boxes, chunk-aware and resumable
verify                      homogeneous verification with storm-block bootstrap confidence intervals
baselines                   every extrapolation (6 h, 12 h, great-circle) and the strongest of them per lead
kalman                      robust (Huber) Kalman motion estimate with damped acceleration
cliper                      CLIPER rebuilt: poleward frame, three motion windows, exact local displacements
analogs                     track-shape analogs rotated to the storm's heading, as departures from persistence
vortex                      the storm removed from the analysis by local Biot-Savart inversion
steering                    eight steering layers, intensity-dependent layer, case-adaptive optimal layer
beta                        beta drift from the vortex's outer strength and the absolute-vorticity gradient
trajectory                  RK4 on the sphere through flows sampled where the storm will be
bogus                       vortex bogus specified by outer circulation, for the barotropic model
barotropic                  limited-area barotropic vorticity model (Arakawa Jacobian, exact sine-transform
                            Poisson solver, sponge): beta gyres emerge, the environment evolves
vorticity_advection         motion from the vorticity budget, independent of the best track
climatology                 ERA5 1981-2010 flow climatology; steering anomalies damped with lead time
stall                       cols and stagnation points of the steering flow: Dorian's stall, detected
ridge                       subtropical ridge axis, strength, tendency, weakness and the upstream trough
shear                       vertical shear in seven layer pairs, generalised shear, vortex tilt
reformation                 competing centres and downshear re-formation toward the convection
ensemble                    physics ensemble with data-void-aware flow perturbations (the Bermuda High problem)
regimes                     stall / recurve / high / weak / straight classification
consensus                   along- and cross-track weights per lead, fitted on past seasons
members                     the member suite computed from one analysis
ocean                       cold wake under slow storms (the ocean fighting back)
land                        landfall timing, inland and terrain decay along the forecast track
intensity                   SHIFOR, SHIPS-style and LGEM-style intensity along the forecast track
structure                   outermost closed isobar, gale radius, size evolution
outflow                     200 hPa divergence, eddy flux convergence, outflow channels, trough interaction
field_tracker               GFDL-style multi-field tracker for any gridded forecast
teachers                    GraphCast, Pangu, HRES, Aurora, FGN, FuXi, GenCast tracks on the same storms
genesis                     disturbance detection, tracking and a 48-hour genesis probability
pouch                       the wave-relative marsupial pouch
synthetic                   Dorian-type and other hard scenarios with known truth, and balanced vortex implants
fit / coefficients          the constants, fitted once on disjoint data and shipped as code
benchmark                   the hindcast benchmark and its report
forecast                    :func:`physics_forecast`, everything in one call (``swarm=N`` adds
                            :mod:`naturev1.swarm`, which works alongside it)
==========================  =================================================================================
"""

from .forecast import physics_forecast

__all__ = ["physics_forecast"]
