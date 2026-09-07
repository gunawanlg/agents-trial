"""Extensible configuration constants for smart metadata inference.

Everything an installation is likely to customise lives here as a plain module
constant so it can be edited or monkeypatched without touching logic.
"""

from typing import Dict, List, Optional, Tuple

#: Suffix that turns a target name into its observation-flag name.
OBS_SUFFIX = "Obs"

#: Fallback target / observation pair when nothing else can be inferred.
DEFAULT_TARGET = "TargetDefault"
DEFAULT_TARGET_OBS = "TargetDefaultObs"

#: portfolio value -> (target name, observation-flag name).
#: Extend this mapping to teach the resolver about a new portfolio.
PORTFOLIO_TARGET_MAP = {
    "PortfolioA": ("TargetA", "TargetAObs"),
    "PortfolioB": ("TargetB", "TargetBObs"),
    "PortfolioC": ("TargetC", "TargetCObs"),
}  # type: Dict[str, Tuple[str, str]]

#: Column-name candidates used when sniffing the table catalogue.
DATE_COLUMN_CANDIDATES = (
    "DATE_DECISION",
    "SCORE_DATE",
    "DTIME_DECISION",
    "APPLICATION_DATE",
    "score_date",
    "date_decision",
)

PORTFOLIO_COLUMN_CANDIDATES = ("PORTFOLIO", "NAME_PORTFOLIO", "portfolio")

SEGMENT_COLUMN_CANDIDATES = (
    "PRODUCT",
    "CHANNEL",
    "SEGMENT",
    "NAME_PRODUCT",
    "CODE_SEGMENT",
    "channel",
    "product",
    "segment",
)

#: Datatype names in the catalogue that count as "date-like".
DATE_TYPE_TOKENS = ("date", "timestamp", "datetime")

#: Maximum distinct values a catalogue column may have to be auto-adopted as a
#: segmentation column.
MAX_SEGMENT_CARDINALITY = 30

#: Default file name offered when persisting resolved settings.
DEFAULT_SETTINGS_FILENAME = "scorecard_eval_settings.json"

#: Default file name of the WoE grouping definition.
DEFAULT_GROUPING_FILENAME = "grouping.json"


def target_for_portfolio(portfolio):
    # type: (Optional[str]) -> Optional[Tuple[str, str]]
    """Return ``(target, target_obs)`` for a portfolio, or ``None``."""
    if portfolio is None:
        return None
    key = str(portfolio)
    if key in PORTFOLIO_TARGET_MAP:
        return PORTFOLIO_TARGET_MAP[key]
    lowered = {k.lower(): v for k, v in PORTFOLIO_TARGET_MAP.items()}
    return lowered.get(key.lower())


def target_from_obs(target_obs):
    # type: (Optional[str]) -> Optional[str]
    """``TargetAObs`` -> ``TargetA``."""
    if not target_obs:
        return None
    name = str(target_obs)
    if name.endswith(OBS_SUFFIX) and len(name) > len(OBS_SUFFIX):
        return name[: -len(OBS_SUFFIX)]
    for target, obs in PORTFOLIO_TARGET_MAP.values():
        if obs == name:
            return target
    return None


def obs_from_target(target):
    # type: (Optional[str]) -> Optional[str]
    """``TargetA`` -> ``TargetAObs``."""
    if not target:
        return None
    name = str(target)
    for known_target, obs in PORTFOLIO_TARGET_MAP.values():
        if known_target == name:
            return obs
    if name.endswith(OBS_SUFFIX):
        return name
    return name + OBS_SUFFIX


def known_targets():
    # type: () -> List[str]
    out = [DEFAULT_TARGET]
    for target, _obs in PORTFOLIO_TARGET_MAP.values():
        if target not in out:
            out.append(target)
    return out


def known_obs_flags():
    # type: () -> List[str]
    out = [DEFAULT_TARGET_OBS]
    for _target, obs in PORTFOLIO_TARGET_MAP.values():
        if obs not in out:
            out.append(obs)
    return out
