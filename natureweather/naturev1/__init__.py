# Copyright 2026 Nathan. Apache-2.0.
"""NatureV1 - a probabilistic weather model over geolocated fields."""

from .besttrack import Track, parse_hurdat2, rapid_intensification, split_by_storm
from .checkpoint import CheckpointManager, TrainingState
from .corpora import CORPORA, augment_batch, catalogue, open_weatherbench, rotate_longitude
from .era5 import (
                   FIELD_DIMS,
                   CachedERA5,
                   ERA5Window,
                   GridCollate,
                   Normalizer,
                   corpus_scale,
                   equiangular_weights,
                   era5_loader,
                   era5_source_grid,
                   era5_splits,
                   expand_variables,
                   lead_offsets,
                   materialise,
                   normalization,
                   read_block,
)
from .fallback import (
                   Choice,
                   Plan,
                   ensure_packages,
                   install_packages,
                   resolve_all,
                   resolve_batch,
                   resolve_precision,
                   resolve_staging,
                   resolve_variables,
                   with_retry,
)
from .forecast import (
                   build_forecast,
                   decode_eyewall,
                   decode_landfall,
                   decode_point_forecast,
                   decode_rapid_intensification,
                   decode_track,
                   nearest_point,
)
from .guards import (
                   PHYSICAL,
                   Bounds,
                   PreflightError,
                   TrainingWatchdog,
                   check_device_agreement,
                   check_environment,
                   check_forecast,
                   check_grid_alignment,
                   check_normalization,
                   check_weights,
                   format_preflight,
                   preflight,
                   responds_to_input,
)
from .hub import from_pretrained, push_to_hub, save_for_hub
from .launch import follow, launch, status, stop
from .live import fetch_latest, latest_scene_keys, watch
from .losses import area_weights, focal_bce, masked_gaussian_nll, total_loss, track_mixture_nll
from .model import (
                   RI_THRESHOLDS_KT,
                   STORM_STATE_FEATURES,
                   SURFACE_FIELDS,
                   TRACKED_HEADS,
                   WEATHER_TYPES,
                   WIND_RADII_THRESHOLDS_KT,
                   EyewallHead,
                   NatureConfig,
                   NatureV1,
                   RapidIntensificationHead,
                   calendar_features,
)
from .planning import autotune_batch_size, benchmark_steps, device_report, format_plan, training_plan
from .rollout import (
                   RolloutSchedule,
                   advance_calendar,
                   drift_report,
                   field_to_channel,
                   reinject,
                   rollout_forecast,
                   rollout_loss,
)
from .satellite import SatelliteScene, fixed_grid_to_latlon, footprint_area_km2, normalize_channels, scene_from_netcdf
from .state import (
                   FORCING_VARIABLES,
                   HEADLINE_STATE_FIELDS,
                   STATIC_VARIABLES,
                   ChannelRoles,
                   StateStepper,
                   channel_loss_weights,
                   channel_roles,
                   conserve_mass,
                   fair_crps,
                   score_state,
                   solar_forcing,
                   spread_skill_ratio,
                   state_forecast,
                   state_rollout_loss,
                   tendency_std,
                   toa_accumulated,
                   toa_flux,
                   train_state_rollout,
)
from .storms import (
                   StormTargets,
                   StormWindow,
                   format_pairing,
                   lead_track_points,
                   pair_tracks_with_reanalysis,
                   storm_feature_bank,
                   storm_scorecard,
                   storm_state_vector,
                   train_storm_heads,
)
from .train import Trainer, TrainSettings, apply_ema, load_for_inference, next_state_targets
from .upper import (
                   HEADLINE_LEVELS,
                   PRESSURE_LEVELS,
                   SURFACE_EXTRAS,
                   UPPER_VARIABLES,
                   channel_names,
                   headline_channels,
                   open_weatherbench_levels,
                   read_levels,
                   upper_air_report,
                   volume_coords,
                   volume_grid,
)
from .wb2 import (
                   HEADLINE,
                   Score,
                   Scorecard,
                   ScoredField,
                   build_climatology,
                   climatology_forecast,
                   latitude_weights,
                   persistence_forecast,
                   score_model,
                   scoring_fields,
                   weighted_acc,
                   weighted_rmse,
)


__version__ = "0.8.4"
__author__ = "Nathan"
__all__ = [
    "CORPORA", "RI_THRESHOLDS_KT", "WIND_RADII_THRESHOLDS_KT", "EyewallHead", "RapidIntensificationHead",
    "Track", "apply_ema", "augment_batch", "autotune_batch_size", "benchmark_steps", "catalogue",
    "device_report", "focal_bce", "format_plan", "masked_gaussian_nll", "next_state_targets",
    "open_weatherbench", "parse_hurdat2", "rapid_intensification", "rotate_longitude", "split_by_storm",
    "training_plan", "CachedERA5", "ERA5Window", "corpus_scale", "era5_source_grid", "era5_splits",
    "materialise", "normalization", "Normalizer", "equiangular_weights", "era5_loader",
    "from_pretrained", "push_to_hub", "save_for_hub",
    "Plan", "Choice", "resolve_all", "resolve_variables", "resolve_staging",
    "resolve_precision", "resolve_batch", "with_retry", "ensure_packages", "install_packages",
    "ChannelRoles", "StateStepper", "channel_roles", "channel_loss_weights", "conserve_mass",
    "fair_crps", "solar_forcing", "spread_skill_ratio", "state_forecast", "state_rollout_loss",
    "tendency_std", "toa_accumulated", "toa_flux", "STATIC_VARIABLES", "FORCING_VARIABLES",
    "HEADLINE_STATE_FIELDS", "score_state", "train_state_rollout", "lead_track_points",
    "launch", "follow", "status", "stop",
    "area_weights", "TRACKED_HEADS", "STORM_STATE_FEATURES", "storm_feature_bank", "storm_scorecard",
    "storm_state_vector", "train_storm_heads",
    "preflight", "format_preflight", "PreflightError", "TrainingWatchdog", "check_forecast",
    "check_grid_alignment", "check_weights", "check_normalization", "check_device_agreement",
    "check_environment", "responds_to_input", "PHYSICAL", "Bounds",
    "RolloutSchedule", "rollout_loss", "rollout_forecast", "reinject", "advance_calendar",
    "field_to_channel", "drift_report",
    "PRESSURE_LEVELS", "UPPER_VARIABLES", "SURFACE_EXTRAS", "HEADLINE_LEVELS", "channel_names",
    "headline_channels", "open_weatherbench_levels", "read_levels", "upper_air_report",
    "volume_coords", "volume_grid",
    "HEADLINE", "Score", "Scorecard", "ScoredField", "scoring_fields", "score_model", "latitude_weights", "weighted_rmse",
    "weighted_acc", "persistence_forecast", "climatology_forecast", "build_climatology", "FIELD_DIMS", "read_block", "expand_variables", "lead_offsets", "GridCollate",
    "SURFACE_FIELDS", "WEATHER_TYPES", "CheckpointManager", "NatureConfig", "NatureV1", "SatelliteScene",
    "TrainSettings", "Trainer", "TrainingState", "build_forecast", "calendar_features", "decode_landfall",
    "decode_point_forecast", "decode_track", "decode_eyewall", "decode_rapid_intensification", "fetch_latest", "fixed_grid_to_latlon", "footprint_area_km2",
    "latest_scene_keys", "load_for_inference", "nearest_point", "normalize_channels", "scene_from_netcdf",
    "total_loss", "track_mixture_nll", "watch", "StormWindow", "StormTargets",
    "pair_tracks_with_reanalysis", "format_pairing",
]
