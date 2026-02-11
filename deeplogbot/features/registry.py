"""
Centralized Feature Registry for DeepLogBot.

This module provides a single source of truth for all features used in bot detection:
- Complete documentation for each feature
- Computation stage tracking
- Bot interpretation guidelines
- Feature dependencies

Usage:
    from deeplogbot.features.registry import FeatureRegistry, FeatureCategory

    # Get all features
    all_features = FeatureRegistry.get_enabled()

    # Get features by category
    behavioral = FeatureRegistry.get_by_category(FeatureCategory.BEHAVIORAL)

    # Print documentation
    docs = FeatureRegistry.print_documentation()
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum


class FeatureCategory(Enum):
    """Categories for grouping related features."""
    BASIC = "basic"                     # Basic location statistics
    TEMPORAL = "temporal"               # Time-of-day patterns
    YEARLY = "yearly"                   # Year-over-year patterns
    BEHAVIORAL = "behavioral"           # Behavioral signatures
    INTERACTION = "interaction"         # Bot interaction features
    SIGNATURE = "signature"             # Bot signature features
    DISCRIMINATIVE = "discriminative"   # Malicious vs legitimate
    TIMING = "timing"                   # Request timing precision (NEW)
    ACCESS_PATTERN = "access"           # Access pattern features (NEW)
    USER_DISTRIBUTION = "user"          # User distribution features (NEW)
    STATISTICAL = "statistical"         # Statistical anomaly features (NEW)
    SESSION = "session"                 # Session behavior features (NEW)
    COMPARATIVE = "comparative"         # Comparative/contextual features (NEW)
    # Time Series Categories (NEW)
    OUTBURST = "outburst"               # Spike/outburst detection
    PERIODICITY = "periodicity"         # Cyclical pattern detection
    TREND = "trend"                     # Long-term trend analysis
    RECENCY = "recency"                 # Recency-weighted features
    DISTRIBUTION = "distribution"       # Distribution shape analysis


class ComputationStage(Enum):
    """When in the pipeline each feature is computed."""
    LOCATION = "location"               # Basic aggregation from raw data
    POST_EXTRACTION = "post_extraction" # After feature extractors run
    POST_ISOLATION_FOREST = "post_if"   # After anomaly_score is available


@dataclass
class FeatureDefinition:
    """
    Complete definition of a feature including documentation and metadata.

    Attributes:
        name: Feature column name
        category: Category for grouping related features
        description: Human-readable description of what the feature measures
        formula: Mathematical formula or computation method
        bot_interpretation: What high/low values mean for bot detection
        value_range: Expected (min, max) values, None means unbounded
        dependencies: List of feature names this feature depends on
        stage: When in the pipeline this feature is computed
        enabled: Whether this feature is currently active
    """
    name: str
    category: FeatureCategory
    description: str
    formula: str
    bot_interpretation: str
    value_range: Tuple[Optional[float], Optional[float]] = (None, None)
    dependencies: List[str] = field(default_factory=list)
    stage: ComputationStage = ComputationStage.LOCATION
    enabled: bool = True


class FeatureRegistry:
    """
    Central registry for all feature definitions.

    Provides methods to:
    - Register new features
    - Query features by name, category, or stage
    - Get list of enabled features
    - Generate documentation
    """
    _features: Dict[str, FeatureDefinition] = {}

    @classmethod
    def register(cls, feature: FeatureDefinition) -> None:
        """Register a new feature definition."""
        cls._features[feature.name] = feature

    @classmethod
    def get(cls, name: str) -> Optional[FeatureDefinition]:
        """Get a feature definition by name."""
        return cls._features.get(name)

    @classmethod
    def get_all(cls) -> List[FeatureDefinition]:
        """Get all registered feature definitions."""
        return list(cls._features.values())

    @classmethod
    def get_by_category(cls, category: FeatureCategory) -> List[FeatureDefinition]:
        """Get all features in a specific category."""
        return [f for f in cls._features.values() if f.category == category]

    @classmethod
    def get_by_stage(cls, stage: ComputationStage) -> List[FeatureDefinition]:
        """Get all features computed at a specific stage."""
        return [f for f in cls._features.values() if f.stage == stage]

    @classmethod
    def get_enabled(cls) -> List[str]:
        """Get names of all enabled features."""
        return [f.name for f in cls._features.values() if f.enabled]

    @classmethod
    def get_enabled_definitions(cls) -> List[FeatureDefinition]:
        """Get definitions of all enabled features."""
        return [f for f in cls._features.values() if f.enabled]

    @classmethod
    def get_feature_names_by_category(cls, category: FeatureCategory) -> List[str]:
        """Get names of all features in a category."""
        return [f.name for f in cls._features.values() if f.category == category]

    @classmethod
    def validate_dependencies(cls) -> List[str]:
        """Check that all feature dependencies exist. Returns list of missing deps."""
        missing = []
        for feature in cls._features.values():
            for dep in feature.dependencies:
                if dep not in cls._features:
                    missing.append(f"{feature.name} depends on missing feature: {dep}")
        return missing

    @classmethod
    def print_documentation(cls) -> str:
        """Generate markdown documentation for all features."""
        lines = ["# DeepLogBot Feature Documentation\n"]
        lines.append(f"Total features: {len(cls._features)}\n")
        lines.append(f"Enabled features: {len(cls.get_enabled())}\n\n")

        # Group by category
        for category in FeatureCategory:
            features = cls.get_by_category(category)
            if not features:
                continue

            lines.append(f"## {category.value.title()} Features ({len(features)})\n")

            for f in sorted(features, key=lambda x: x.name):
                lines.append(f"### `{f.name}`\n")
                lines.append(f"**Description:** {f.description}\n")
                lines.append(f"**Formula:** `{f.formula}`\n")
                lines.append(f"**Bot Interpretation:** {f.bot_interpretation}\n")
                if f.value_range != (None, None):
                    lines.append(f"**Value Range:** {f.value_range}\n")
                if f.dependencies:
                    lines.append(f"**Dependencies:** {', '.join(f.dependencies)}\n")
                lines.append(f"**Stage:** {f.stage.value}\n")
                lines.append(f"**Enabled:** {f.enabled}\n\n")

        return "\n".join(lines)

    @classmethod
    def summary(cls) -> Dict[str, int]:
        """Get summary statistics about registered features."""
        return {
            "total": len(cls._features),
            "enabled": len(cls.get_enabled()),
            "by_category": {
                cat.value: len(cls.get_by_category(cat))
                for cat in FeatureCategory
            },
            "by_stage": {
                stage.value: len(cls.get_by_stage(stage))
                for stage in ComputationStage
            }
        }


# =============================================================================
# REGISTER ALL FEATURES
# =============================================================================

# -----------------------------------------------------------------------------
# BASIC FEATURES (11 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="unique_users",
    category=FeatureCategory.BASIC,
    description="Number of distinct users downloading from this location",
    formula="COUNT(DISTINCT user)",
    bot_interpretation="HIGH (>1000) suggests bot farm or coordinated attack; VERY LOW (<5) suggests mirror or individual",
    value_range=(1, None),
    stage=ComputationStage.LOCATION
))

FeatureRegistry.register(FeatureDefinition(
    name="downloads_per_user",
    category=FeatureCategory.BASIC,
    description="Average number of downloads per unique user",
    formula="total_downloads / unique_users",
    bot_interpretation="LOW (<20) with many users = bot farm; HIGH (>500) with few users = mirror/bulk downloader",
    value_range=(1, None),
    stage=ComputationStage.LOCATION
))

FeatureRegistry.register(FeatureDefinition(
    name="avg_users_per_hour",
    category=FeatureCategory.BASIC,
    description="Average number of concurrent users per active hour",
    formula="AVG(COUNT(DISTINCT user) per hour)",
    bot_interpretation="HIGH with stable pattern = coordinated bot; HIGH with spikes = burst attack",
    value_range=(0, None),
    stage=ComputationStage.LOCATION
))

FeatureRegistry.register(FeatureDefinition(
    name="max_users_per_hour",
    category=FeatureCategory.BASIC,
    description="Maximum concurrent users in any single hour",
    formula="MAX(COUNT(DISTINCT user) per hour)",
    bot_interpretation="Very HIGH relative to avg = sudden bot attack or viral event",
    value_range=(0, None),
    stage=ComputationStage.LOCATION
))

FeatureRegistry.register(FeatureDefinition(
    name="user_cv",
    category=FeatureCategory.BASIC,
    description="Coefficient of variation of hourly user counts",
    formula="STDDEV(users_per_hour) / MEAN(users_per_hour)",
    bot_interpretation="LOW = stable/mechanical pattern (bot); HIGH = organic variation",
    value_range=(0, None),
    stage=ComputationStage.LOCATION
))

FeatureRegistry.register(FeatureDefinition(
    name="users_per_active_hour",
    category=FeatureCategory.BASIC,
    description="User density: unique users divided by hours with activity",
    formula="unique_users / active_hours",
    bot_interpretation="HIGH = concentrated activity (suspicious); LOW = spread over time (normal)",
    value_range=(0, None),
    stage=ComputationStage.LOCATION
))

FeatureRegistry.register(FeatureDefinition(
    name="projects_per_user",
    category=FeatureCategory.BASIC,
    description="Average number of distinct projects accessed per user",
    formula="unique_projects / unique_users",
    bot_interpretation="HIGH = exploration (malicious bot); LOW = focused access (legitimate)",
    value_range=(0, None),
    stage=ComputationStage.LOCATION
))

FeatureRegistry.register(FeatureDefinition(
    name="total_downloads",
    category=FeatureCategory.BASIC,
    description="Total number of download events from this location",
    formula="COUNT(*)",
    bot_interpretation="Context-dependent; used with other features to determine pattern",
    value_range=(1, None),
    stage=ComputationStage.LOCATION
))

FeatureRegistry.register(FeatureDefinition(
    name="active_hours",
    category=FeatureCategory.BASIC,
    description="Number of distinct hours with download activity",
    formula="COUNT(DISTINCT DATE_TRUNC('hour', timestamp))",
    bot_interpretation="LOW relative to downloads = burst pattern; HIGH = sustained activity",
    value_range=(1, None),
    stage=ComputationStage.LOCATION
))

FeatureRegistry.register(FeatureDefinition(
    name="years_active",
    category=FeatureCategory.BASIC,
    description="Number of distinct years with download activity",
    formula="COUNT(DISTINCT YEAR(timestamp))",
    bot_interpretation="LOW (1) with high volume = new/suspicious; HIGH = established pattern",
    value_range=(1, None),
    stage=ComputationStage.LOCATION
))

FeatureRegistry.register(FeatureDefinition(
    name="unique_projects",
    category=FeatureCategory.BASIC,
    description="Number of distinct projects/accessions accessed",
    formula="COUNT(DISTINCT project)",
    bot_interpretation="HIGH with systematic access = crawler; LOW = focused legitimate use",
    value_range=(0, None),
    stage=ComputationStage.LOCATION
))

# -----------------------------------------------------------------------------
# TEMPORAL FEATURES (5 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="hourly_download_std",
    category=FeatureCategory.TEMPORAL,
    description="Standard deviation of downloads across 24 hours",
    formula="STDDEV(downloads per hour of day)",
    bot_interpretation="LOW = uniform/mechanical distribution; HIGH = peaked human pattern",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="peak_hour_concentration",
    category=FeatureCategory.TEMPORAL,
    description="Fraction of downloads in the peak hour",
    formula="MAX(hourly_fraction)",
    bot_interpretation="VERY HIGH (>0.5) = scheduled bot or brief attack; MODERATE = normal",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="working_hours_ratio",
    category=FeatureCategory.TEMPORAL,
    description="Fraction of downloads during working hours (9-17)",
    formula="SUM(downloads 9-17) / total_downloads",
    bot_interpretation="LOW (<0.3) = non-human timing; HIGH (>0.6) = human pattern",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="hourly_entropy",
    category=FeatureCategory.TEMPORAL,
    description="Shannon entropy of hourly download distribution",
    formula="entropy(hourly_proportions)",
    bot_interpretation="LOW = concentrated/scheduled (bot); HIGH = spread/random (human)",
    value_range=(0, 3.2),  # max for 24 hours = log2(24) ~ 4.6
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="night_activity_ratio",
    category=FeatureCategory.TEMPORAL,
    description="Fraction of downloads during night hours (22-6)",
    formula="SUM(downloads 22-6) / total_downloads",
    bot_interpretation="HIGH (>0.3) = nocturnal activity (bot); LOW = normal human",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

# -----------------------------------------------------------------------------
# YEARLY FEATURES (10 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="yearly_entropy",
    category=FeatureCategory.YEARLY,
    description="Shannon entropy of yearly download distribution",
    formula="entropy(yearly_proportions)",
    bot_interpretation="LOW = concentrated in few years (new bot); HIGH = established pattern",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="peak_year_concentration",
    category=FeatureCategory.YEARLY,
    description="Fraction of downloads in the peak year",
    formula="MAX(yearly_fraction)",
    bot_interpretation="VERY HIGH (>0.9) = single year spike (suspicious); MODERATE = normal",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="years_span",
    category=FeatureCategory.YEARLY,
    description="Number of years between first and last activity",
    formula="last_year - first_year + 1",
    bot_interpretation="LOW (1) with high volume = suspicious; HIGH = legitimate long-term use",
    value_range=(1, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="downloads_per_year",
    category=FeatureCategory.YEARLY,
    description="Average downloads per year of activity",
    formula="total_downloads / years_span",
    bot_interpretation="VERY HIGH for single year = attack; MODERATE stable = legitimate",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="year_over_year_cv",
    category=FeatureCategory.YEARLY,
    description="Coefficient of variation of yearly download counts",
    formula="STDDEV(yearly_downloads) / MEAN(yearly_downloads)",
    bot_interpretation="HIGH = erratic pattern; LOW = stable (could be either)",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="fraction_latest_year",
    category=FeatureCategory.YEARLY,
    description="Fraction of all downloads occurring in the latest year",
    formula="latest_year_downloads / total_downloads",
    bot_interpretation="VERY HIGH (>0.9) with history = sudden spike (investigate)",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="is_new_location",
    category=FeatureCategory.YEARLY,
    description="Binary: location first appeared in the latest year only",
    formula="1 if first_year == latest_year AND years_count == 1",
    bot_interpretation="1 = new location, higher risk; 0 = established",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="spike_ratio",
    category=FeatureCategory.YEARLY,
    description="Ratio of latest year downloads to average of previous years",
    formula="latest_year_downloads / AVG(previous_years_downloads)",
    bot_interpretation="HIGH (>3) = significant spike (investigate); ~1 = stable",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="years_before_latest",
    category=FeatureCategory.YEARLY,
    description="Number of years of activity before the latest year",
    formula="COUNT(years) - 1",
    bot_interpretation="0 = first year only (higher risk); HIGH = established pattern",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="latest_year_downloads",
    category=FeatureCategory.YEARLY,
    description="Total downloads in the most recent year",
    formula="SUM(downloads) WHERE year = MAX(year)",
    bot_interpretation="Used in conjunction with spike_ratio to detect anomalies",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

# -----------------------------------------------------------------------------
# ADVANCED BEHAVIORAL FEATURES (16 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="burst_pattern_score",
    category=FeatureCategory.BEHAVIORAL,
    description="Measures spike intensity: (max - mean) / std of hourly activity",
    formula="(max_hourly - mean_hourly) / std_hourly",
    bot_interpretation="HIGH = burst/attack pattern; LOW = steady activity",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="circadian_rhythm_deviation",
    category=FeatureCategory.BEHAVIORAL,
    description="Deviation from expected human activity pattern (10% night, 50% work, 25% evening, 15% morning)",
    formula="SUM(ABS(observed_ratio - expected_ratio))",
    bot_interpretation="HIGH = non-human timing pattern (bot); LOW = human-like",
    value_range=(0, 2),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="user_coordination_score",
    category=FeatureCategory.BEHAVIORAL,
    description="Measures synchronized multi-user activity",
    formula="avg_concurrent_users / std_concurrent_users",
    bot_interpretation="HIGH with many users = coordinated bot farm; LOW = independent users",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="hourly_cv_burst",
    category=FeatureCategory.BEHAVIORAL,
    description="Coefficient of variation of hourly activity",
    formula="std_hourly / mean_hourly",
    bot_interpretation="LOW = mechanical/uniform (bot); HIGH = variable (human)",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="spike_intensity",
    category=FeatureCategory.BEHAVIORAL,
    description="Ratio of max hourly activity to mean hourly activity",
    formula="max_hourly / mean_hourly",
    bot_interpretation="HIGH = concentrated spikes; LOW = uniform distribution",
    value_range=(1, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="user_peak_ratio",
    category=FeatureCategory.BEHAVIORAL,
    description="Ratio of max concurrent users to average concurrent users",
    formula="max_concurrent_users / avg_concurrent_users",
    bot_interpretation="HIGH = synchronized bot spikes; LOW = stable user distribution",
    value_range=(1, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="night_ratio_advanced",
    category=FeatureCategory.BEHAVIORAL,
    description="Fraction of activity during night hours (0-5)",
    formula="night_count / total_count",
    bot_interpretation="HIGH (>0.25) = nocturnal bot; LOW = human pattern",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="work_ratio_advanced",
    category=FeatureCategory.BEHAVIORAL,
    description="Fraction of activity during work hours (9-17)",
    formula="work_count / total_count",
    bot_interpretation="HIGH (~0.5) = human pattern; LOW = non-working hours (bot)",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="evening_ratio",
    category=FeatureCategory.BEHAVIORAL,
    description="Fraction of activity during evening hours (18-23)",
    formula="evening_count / total_count",
    bot_interpretation="Component of circadian rhythm analysis",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="morning_ratio",
    category=FeatureCategory.BEHAVIORAL,
    description="Fraction of activity during morning hours (6-8)",
    formula="morning_count / total_count",
    bot_interpretation="Component of circadian rhythm analysis",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="user_coordination_std",
    category=FeatureCategory.BEHAVIORAL,
    description="Standard deviation of concurrent users across hours",
    formula="STDDEV(concurrent_users)",
    bot_interpretation="LOW with high avg = coordinated (bot farm); HIGH = natural variation",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="avg_concurrent_users",
    category=FeatureCategory.BEHAVIORAL,
    description="Average number of concurrent users per hour",
    formula="AVG(concurrent_users_per_hour)",
    bot_interpretation="HIGH = active location; interpret with coordination score",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="max_concurrent_users",
    category=FeatureCategory.BEHAVIORAL,
    description="Maximum concurrent users in any hour",
    formula="MAX(concurrent_users_per_hour)",
    bot_interpretation="HIGH relative to avg = spike pattern",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="is_bursty_advanced",
    category=FeatureCategory.BEHAVIORAL,
    description="Binary: burst_pattern_score above 75th percentile",
    formula="1 if burst_pattern_score > Q75",
    bot_interpretation="1 = likely burst attack pattern",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="is_nocturnal",
    category=FeatureCategory.BEHAVIORAL,
    description="Binary: night_ratio_advanced > 0.25",
    formula="1 if night_ratio > 0.25",
    bot_interpretation="1 = significant night activity (suspicious)",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="is_coordinated",
    category=FeatureCategory.BEHAVIORAL,
    description="Binary: user_coordination_score above 75th percentile",
    formula="1 if coordination_score > Q75",
    bot_interpretation="1 = likely coordinated bot farm",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

# -----------------------------------------------------------------------------
# BOT INTERACTION FEATURES (6 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="dl_user_per_log_users",
    category=FeatureCategory.INTERACTION,
    description="Downloads per user normalized by log of user count",
    formula="downloads_per_user / log(unique_users + 2)",
    bot_interpretation="HIGH = abnormal download intensity",
    value_range=(0, None),
    dependencies=["downloads_per_user", "unique_users"],
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="user_scarcity_score",
    category=FeatureCategory.INTERACTION,
    description="Exponential decay based on user count for high DL/user locations",
    formula="exp(-unique_users/100) if downloads_per_user > 20 else 0",
    bot_interpretation="HIGH = few users with high activity (mirror or suspicious)",
    value_range=(0, 1),
    dependencies=["downloads_per_user", "unique_users"],
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="download_concentration",
    category=FeatureCategory.INTERACTION,
    description="Download intensity inversely weighted by users",
    formula="downloads_per_user * (1 / (unique_users + 1))",
    bot_interpretation="HIGH = concentrated download pattern",
    value_range=(0, None),
    dependencies=["downloads_per_user", "unique_users"],
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="temporal_irregularity",
    category=FeatureCategory.INTERACTION,
    description="Combines temporal regularity with download intensity",
    formula="(1 / (hourly_entropy + 0.1)) * log(downloads_per_user + 1)",
    bot_interpretation="HIGH = regular timing with high activity (scheduled bot)",
    value_range=(0, None),
    dependencies=["hourly_entropy", "downloads_per_user"],
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="bot_composite_score",
    category=FeatureCategory.INTERACTION,
    description="Weighted combination of bot indicators",
    formula="0.3*dl_user_log + 0.25*scarcity + 0.25*concentration + 0.2*anomaly",
    bot_interpretation="HIGH (>0.7) = strong bot signal; LOW (<0.3) = likely legitimate",
    value_range=(0, 1),
    dependencies=["dl_user_per_log_users", "user_scarcity_score", "download_concentration"],
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="anomaly_dl_interaction",
    category=FeatureCategory.INTERACTION,
    description="Anomaly score weighted by downloads per user",
    formula="(anomaly_score + 0.5) * downloads_per_user",
    bot_interpretation="HIGH = anomalous location with high download rate",
    value_range=(0, None),
    dependencies=["anomaly_score", "downloads_per_user"],
    stage=ComputationStage.POST_ISOLATION_FOREST
))

# -----------------------------------------------------------------------------
# BOT SIGNATURE FEATURES (8 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="request_velocity",
    category=FeatureCategory.SIGNATURE,
    description="Downloads per active hour",
    formula="total_downloads / (working_hours_ratio * 24 * 7)",
    bot_interpretation="HIGH = rapid download rate (scraper/bot)",
    value_range=(0, None),
    dependencies=["total_downloads", "working_hours_ratio"],
    stage=ComputationStage.POST_ISOLATION_FOREST
))

FeatureRegistry.register(FeatureDefinition(
    name="access_regularity",
    category=FeatureCategory.SIGNATURE,
    description="Inverse of hourly entropy - measures temporal consistency",
    formula="1 / (hourly_entropy + 0.1)",
    bot_interpretation="HIGH = very regular timing (scheduled bot)",
    value_range=(0, 10),
    dependencies=["hourly_entropy"],
    stage=ComputationStage.POST_ISOLATION_FOREST
))

FeatureRegistry.register(FeatureDefinition(
    name="ua_per_user",
    category=FeatureCategory.SIGNATURE,
    description="Proxy for user-agent diversity per user",
    formula="1 / (unique_users / (total_downloads + 1) + 0.01)",
    bot_interpretation="HIGH = many downloads per user (could be bot or power user)",
    value_range=(0, None),
    dependencies=["unique_users", "total_downloads"],
    stage=ComputationStage.POST_ISOLATION_FOREST
))

FeatureRegistry.register(FeatureDefinition(
    name="ip_concentration",
    category=FeatureCategory.SIGNATURE,
    description="Inverse of user distribution entropy (proxy for IP diversity)",
    formula="1 - log1p(unique_users) / log1p(max_unique_users)",
    bot_interpretation="HIGH = concentrated access (fewer distinct sources)",
    value_range=(0, 1),
    dependencies=["unique_users"],
    stage=ComputationStage.POST_ISOLATION_FOREST
))

FeatureRegistry.register(FeatureDefinition(
    name="session_anomaly",
    category=FeatureCategory.SIGNATURE,
    description="Deviation of downloads_per_user from median",
    formula="|downloads_per_user - median| / (median + 1)",
    bot_interpretation="HIGH = unusual download pattern for this location",
    value_range=(0, None),
    dependencies=["downloads_per_user"],
    stage=ComputationStage.POST_ISOLATION_FOREST
))

FeatureRegistry.register(FeatureDefinition(
    name="request_pattern_anomaly",
    category=FeatureCategory.SIGNATURE,
    description="Proxy for file request entropy - low diversity = suspicious",
    formula="1 / (downloads_per_user / (log1p(unique_users) + 1) + 0.1)",
    bot_interpretation="HIGH = likely requesting same files (bot/scraper)",
    value_range=(0, None),
    dependencies=["downloads_per_user", "unique_users"],
    stage=ComputationStage.POST_ISOLATION_FOREST
))

FeatureRegistry.register(FeatureDefinition(
    name="weekend_weekday_imbalance",
    category=FeatureCategory.SIGNATURE,
    description="Deviation from expected weekday/weekend ratio",
    formula="|working_hours_ratio - 0.5|",
    bot_interpretation="HIGH = atypical work pattern (bot or shift worker)",
    value_range=(0, 0.5),
    dependencies=["working_hours_ratio"],
    stage=ComputationStage.POST_ISOLATION_FOREST
))

FeatureRegistry.register(FeatureDefinition(
    name="is_high_velocity",
    category=FeatureCategory.SIGNATURE,
    description="Binary: request_velocity above threshold",
    formula="1 if request_velocity > Q90",
    bot_interpretation="1 = very high download rate",
    value_range=(0, 1),
    dependencies=["request_velocity"],
    stage=ComputationStage.POST_ISOLATION_FOREST
))

# -----------------------------------------------------------------------------
# DISCRIMINATIVE FEATURES (18 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="file_exploration_score",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Measures tendency to access many different files",
    formula="file_entropy * ln(unique_files + 1) if unique_files > 10",
    bot_interpretation="HIGH = exploratory access (malicious bot scanning)",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="file_mirroring_score",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Inverse of file entropy - measures focused file access",
    formula="1 / (file_entropy + 0.1)",
    bot_interpretation="HIGH = accessing same files repeatedly (legitimate mirror)",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="file_entropy",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Shannon entropy of file access distribution",
    formula="log2(total_downloads / unique_files)",
    bot_interpretation="LOW = diverse files (crawler); HIGH = repeated files (mirror)",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="bot_farm_score",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Detects many homogeneous users with similar behavior",
    formula="user_homogeneity * ln(unique_users + 1) if users > 100",
    bot_interpretation="HIGH = coordinated fake users (bot farm)",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="user_authenticity_score",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Measures genuine user activity patterns",
    formula="avg_active_days_per_user * avg_files_per_user",
    bot_interpretation="HIGH = diverse sustained activity (legitimate users)",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="user_homogeneity_score",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Measures uniformity of user behavior",
    formula="mean(downloads_per_user) / std(downloads_per_user)",
    bot_interpretation="HIGH = users behave identically (bot farm signature)",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="geographic_stability",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Inverse of IP concentration - stable locations are legitimate",
    formula="1 / (ip_concentration + 0.1)",
    bot_interpretation="HIGH = stable source (institutional/legitimate); LOW = rotating IPs (bot)",
    value_range=(0, 1),
    dependencies=["ip_concentration"],
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="version_concentration",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Measures targeting of specific versions",
    formula="1 / unique_versions",
    bot_interpretation="HIGH = targets few versions (vulnerability scanner); LOW = archives all",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="targets_latest_only",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Binary: only accesses latest versions",
    formula="1 if version_concentration > 0.5",
    bot_interpretation="1 = likely scanning for vulnerabilities in latest release",
    value_range=(0, 1),
    dependencies=["version_concentration"],
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="unique_versions",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Number of distinct versions accessed",
    formula="COUNT(DISTINCT version)",
    bot_interpretation="LOW = focused access; HIGH = archival behavior",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="lifespan_days",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Days between first and last download activity",
    formula="(last_timestamp - first_timestamp) / 86400",
    bot_interpretation="LOW = hit-and-run (malicious); HIGH = sustained use (legitimate)",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="activity_density",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Ratio of active days to lifespan",
    formula="active_days / lifespan_days",
    bot_interpretation="LOW = sporadic activity (suspicious); HIGH = consistent use",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="persistence_score",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Measures sustained activity over time",
    formula="active_weeks * active_days",
    bot_interpretation="HIGH = long-term legitimate use; LOW = temporary bot activity",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="malicious_bot_score",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Composite score for malicious bot likelihood",
    formula="0.25*exploration + 0.30*bot_farm + 0.20*(1-geo_stability) + 0.15*latest_only + 0.10*(1-density)",
    bot_interpretation="HIGH (>0.7) = likely malicious automation",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="legitimate_automation_score",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Composite score for legitimate automation likelihood",
    formula="0.30*mirroring + 0.25*authenticity + 0.20*geo_stability + 0.25*persistence",
    bot_interpretation="HIGH (>0.7) = likely legitimate automation (CI/CD, mirrors)",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="bot_vs_legitimate_score",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Difference between malicious and legitimate scores",
    formula="malicious_bot_score - legitimate_automation_score",
    bot_interpretation="POSITIVE = likely malicious; NEGATIVE = likely legitimate; ~0 = ambiguous",
    value_range=(-1, 1),
    dependencies=["malicious_bot_score", "legitimate_automation_score"],
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="is_likely_malicious",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Binary: bot_vs_legitimate_score > 0.3",
    formula="1 if bot_vs_legitimate_score > 0.3",
    bot_interpretation="1 = classified as likely malicious bot",
    value_range=(0, 1),
    dependencies=["bot_vs_legitimate_score"],
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="is_likely_legitimate_automation",
    category=FeatureCategory.DISCRIMINATIVE,
    description="Binary: bot_vs_legitimate_score < -0.3",
    formula="1 if bot_vs_legitimate_score < -0.3",
    bot_interpretation="1 = classified as likely legitimate automation",
    value_range=(0, 1),
    dependencies=["bot_vs_legitimate_score"],
    stage=ComputationStage.POST_EXTRACTION
))

# -----------------------------------------------------------------------------
# META/HELPER FEATURES (1 feature)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="time_series_features_present",
    category=FeatureCategory.BASIC,
    description="Placeholder flag indicating time series features are available",
    formula="1 if time_series_features exist else 0",
    bot_interpretation="Used by deep learning model to know time series data exists",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

# -----------------------------------------------------------------------------
# NEW TIMING PRECISION FEATURES (4 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="request_interval_mode",
    category=FeatureCategory.TIMING,
    description="Most common interval between consecutive requests (seconds)",
    formula="MODE(interval_seconds)",
    bot_interpretation="NARROW/ROUND (e.g., exactly 60s) = scheduled bot; VARIED = human",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="round_second_ratio",
    category=FeatureCategory.TIMING,
    description="Fraction of requests occurring at round seconds (:00, :15, :30, :45)",
    formula="COUNT(second IN [0,15,30,45]) / total_requests",
    bot_interpretation="HIGH (>0.5) = scheduled bot; ~0.25 = random human timing",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="millisecond_variance",
    category=FeatureCategory.TIMING,
    description="Variance of millisecond component of request timestamps",
    formula="VARIANCE(EXTRACT(MILLISECOND FROM timestamp))",
    bot_interpretation="ZERO/LOW = no sub-second precision (bot); HIGH = natural variation",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="interval_entropy",
    category=FeatureCategory.TIMING,
    description="Shannon entropy of request interval distribution",
    formula="entropy(binned_intervals)",
    bot_interpretation="LOW = mechanical regular intervals (bot); HIGH = varied timing",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

# -----------------------------------------------------------------------------
# NEW ACCESS PATTERN FEATURES (5 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="alphabetical_access_score",
    category=FeatureCategory.ACCESS_PATTERN,
    description="Spearman correlation of file access order with alphabetical order",
    formula="SPEARMAN_CORR(access_order, alphabetical_order)",
    bot_interpretation="HIGH (>0.7) = systematic crawler; LOW/NEGATIVE = random human access",
    value_range=(-1, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="sequential_file_ratio",
    category=FeatureCategory.ACCESS_PATTERN,
    description="Fraction of consecutive file accesses that are sequential (file_n, file_n+1)",
    formula="COUNT(sequential pairs) / total_consecutive_pairs",
    bot_interpretation="HIGH = systematic numbered file access (scraper)",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="directory_traversal_score",
    category=FeatureCategory.ACCESS_PATTERN,
    description="Measures if access follows directory tree structure (depth-first)",
    formula="correlation(access_order, depth_first_order)",
    bot_interpretation="HIGH = systematic crawler following directory structure",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="retry_ratio",
    category=FeatureCategory.ACCESS_PATTERN,
    description="Fraction of files accessed more than once",
    formula="COUNT(files with access_count > 1) / unique_files",
    bot_interpretation="HIGH = scraper retrying failures or systematic re-download",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="unique_file_ratio",
    category=FeatureCategory.ACCESS_PATTERN,
    description="Ratio of unique files to total downloads",
    formula="unique_files / total_downloads",
    bot_interpretation="LOW = repeated file access (mirror); HIGH = diverse exploration (crawler)",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

# -----------------------------------------------------------------------------
# NEW USER DISTRIBUTION FEATURES (4 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="user_entropy",
    category=FeatureCategory.USER_DISTRIBUTION,
    description="Shannon entropy of download distribution across users",
    formula="entropy(downloads_per_user_distribution)",
    bot_interpretation="LOW = concentrated in few users; HIGH = distributed across many",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="user_gini_coefficient",
    category=FeatureCategory.USER_DISTRIBUTION,
    description="Gini coefficient of download distribution (inequality measure)",
    formula="GINI(downloads_per_user)",
    bot_interpretation="0 = perfectly equal; 1 = one user does all; HIGH with many users = bot farm",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="single_download_user_ratio",
    category=FeatureCategory.USER_DISTRIBUTION,
    description="Fraction of users with exactly one download",
    formula="COUNT(users with 1 download) / unique_users",
    bot_interpretation="HIGH = bot farm creating fake single-use users",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="power_user_ratio",
    category=FeatureCategory.USER_DISTRIBUTION,
    description="Fraction of downloads from top 10% of users",
    formula="SUM(downloads from top 10% users) / total_downloads",
    bot_interpretation="VERY HIGH (>0.9) or VERY LOW (<0.1) = suspicious distribution",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

# -----------------------------------------------------------------------------
# NEW STATISTICAL ANOMALY FEATURES (3 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="benford_deviation",
    category=FeatureCategory.STATISTICAL,
    description="Chi-squared deviation from Benford's Law for download counts",
    formula="CHI_SQUARED(observed_first_digits, benford_expected)",
    bot_interpretation="HIGH = artificial/manufactured numbers (bot-generated traffic)",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="hourly_uniformity_score",
    category=FeatureCategory.STATISTICAL,
    description="How uniform the hourly distribution is (vs expected Poisson)",
    formula="1 - chi_squared_test(observed, uniform) / max_chi_squared",
    bot_interpretation="HIGH = uniform distribution (scheduled bot); LOW = peaked (human)",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="weekday_pattern_score",
    category=FeatureCategory.STATISTICAL,
    description="Deviation from expected 5/7 weekday activity ratio",
    formula="|weekday_ratio - 5/7| / (5/7)",
    bot_interpretation="HIGH = extreme weekday/weekend bias (automated scheduling)",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

# -----------------------------------------------------------------------------
# NEW SESSION BEHAVIOR FEATURES (4 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="session_duration_cv",
    category=FeatureCategory.SESSION,
    description="Coefficient of variation of session lengths",
    formula="STDDEV(session_duration) / MEAN(session_duration)",
    bot_interpretation="LOW = mechanical fixed-length sessions (bot); HIGH = varied human sessions",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="inter_session_regularity",
    category=FeatureCategory.SESSION,
    description="How regular the gaps between sessions are",
    formula="1 / (CV(inter_session_gaps) + 0.1)",
    bot_interpretation="HIGH = scheduled sessions (bot cron job); LOW = random timing",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="downloads_per_session_cv",
    category=FeatureCategory.SESSION,
    description="Coefficient of variation of downloads per session",
    formula="STDDEV(downloads_per_session) / MEAN(downloads_per_session)",
    bot_interpretation="LOW = consistent session size (bot); HIGH = varied (human)",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="session_start_hour_entropy",
    category=FeatureCategory.SESSION,
    description="Entropy of session start times across hours",
    formula="entropy(session_start_hour_distribution)",
    bot_interpretation="LOW = sessions always start same time (scheduled); HIGH = varied",
    value_range=(0, 3.2),
    stage=ComputationStage.POST_EXTRACTION
))

# -----------------------------------------------------------------------------
# NEW COMPARATIVE/CONTEXTUAL FEATURES (4 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="country_zscore",
    category=FeatureCategory.COMPARATIVE,
    description="Z-score of downloads_per_user vs country average",
    formula="(downloads_per_user - country_mean) / country_std",
    bot_interpretation="EXTREME (|z|>3) = significant outlier in country context",
    value_range=(None, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="temporal_trend_anomaly",
    category=FeatureCategory.COMPARATIVE,
    description="Deviation from location's historical trend",
    formula="|current_rate - predicted_rate| / predicted_rate",
    bot_interpretation="HIGH = sudden change from historical pattern (new bot activity)",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="peer_similarity",
    category=FeatureCategory.COMPARATIVE,
    description="Cosine similarity to locations with similar user counts",
    formula="COSINE_SIM(feature_vector, avg_peer_feature_vector)",
    bot_interpretation="LOW = behaves very differently from similar locations (anomalous)",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="global_rank_percentile",
    category=FeatureCategory.COMPARATIVE,
    description="Percentile rank by total downloads",
    formula="PERCENT_RANK() OVER (ORDER BY total_downloads)",
    bot_interpretation="Context for other features; extreme with suspicious pattern = investigate",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))


# -----------------------------------------------------------------------------
# TIME SERIES: OUTBURST DETECTION FEATURES (6 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="outburst_count",
    category=FeatureCategory.OUTBURST,
    description="Number of significant activity spikes (>2 std from mean)",
    formula="COUNT(days WHERE z-score > 2)",
    bot_interpretation="HIGH = frequent spikes (attack pattern); LOW = steady activity",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="outburst_intensity",
    category=FeatureCategory.OUTBURST,
    description="Average magnitude of outburst events (Z-score)",
    formula="AVG(z-score) WHERE z-score > 2",
    bot_interpretation="HIGH = severe spikes (aggressive bot); LOW = mild variations",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="max_outburst_zscore",
    category=FeatureCategory.OUTBURST,
    description="Highest Z-score across all time windows",
    formula="MAX(z-score)",
    bot_interpretation="VERY HIGH (>5) = extreme spike event (attack)",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="outburst_ratio",
    category=FeatureCategory.OUTBURST,
    description="Fraction of total downloads occurring during outbursts",
    formula="SUM(downloads during outbursts) / total_downloads",
    bot_interpretation="HIGH = concentrated attack; LOW = distributed activity",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="time_since_last_outburst",
    category=FeatureCategory.OUTBURST,
    description="Recency of last outburst (normalized by total timespan)",
    formula="days_since_last_outburst / total_days",
    bot_interpretation="LOW = recent spike (active attack); HIGH = historical only",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="longest_outburst_streak",
    category=FeatureCategory.OUTBURST,
    description="Maximum consecutive high-activity days",
    formula="MAX(consecutive days with z-score > 2)",
    bot_interpretation="HIGH = sustained attack; LOW = isolated spikes",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

# -----------------------------------------------------------------------------
# TIME SERIES: PERIODICITY DETECTION FEATURES (4 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="weekly_autocorr",
    category=FeatureCategory.PERIODICITY,
    description="Autocorrelation at 7-day lag (weekly cycle strength)",
    formula="CORR(downloads[t], downloads[t-7])",
    bot_interpretation="HIGH = strong weekly schedule (scheduled bot); LOW = random",
    value_range=(-1, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="dominant_period_days",
    category=FeatureCategory.PERIODICITY,
    description="Most significant period detected via FFT (in days)",
    formula="1 / argmax(FFT_power)",
    bot_interpretation="7 = weekly schedule; 30 = monthly; 0 = no clear period",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="periodicity_strength",
    category=FeatureCategory.PERIODICITY,
    description="Strength of dominant period (fraction of total power)",
    formula="max_FFT_power / total_FFT_power",
    bot_interpretation="HIGH = strict schedule (cron job/bot); LOW = irregular",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="period_regularity",
    category=FeatureCategory.PERIODICITY,
    description="How sharp/consistent the periodic peak is",
    formula="dominant_peak / avg(secondary_peaks)",
    bot_interpretation="HIGH = very regular cycle; LOW = noisy/varied",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

# -----------------------------------------------------------------------------
# TIME SERIES: TREND ANALYSIS FEATURES (5 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="trend_slope",
    category=FeatureCategory.TREND,
    description="Linear trend direction (normalized by mean)",
    formula="linear_regression_slope / mean_downloads",
    bot_interpretation="POSITIVE = growing; NEGATIVE = declining; ~0 = stable",
    value_range=(-10, 10),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="trend_strength",
    category=FeatureCategory.TREND,
    description="R² of linear fit (how linear is the trend)",
    formula="R²(linear_regression)",
    bot_interpretation="HIGH = clear trend; LOW = noisy/no trend",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="trend_acceleration",
    category=FeatureCategory.TREND,
    description="Second derivative of trend (speeding up or slowing down)",
    formula="d²/dt²(downloads) / mean",
    bot_interpretation="POSITIVE = accelerating growth; NEGATIVE = decelerating",
    value_range=(-10, 10),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="detrended_volatility",
    category=FeatureCategory.TREND,
    description="Volatility after removing linear trend",
    formula="std(residuals) / mean",
    bot_interpretation="HIGH = noisy around trend; LOW = smooth trend",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="trend_direction",
    category=FeatureCategory.TREND,
    description="Categorical trend direction (-1, 0, +1)",
    formula="+1 if slope > 0.01, -1 if slope < -0.01, else 0",
    bot_interpretation="+1 = growing, 0 = stable, -1 = declining",
    value_range=(-1, 1),
    stage=ComputationStage.POST_EXTRACTION
))

# -----------------------------------------------------------------------------
# TIME SERIES: RECENCY-WEIGHTED FEATURES (4 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="recent_activity_ratio",
    category=FeatureCategory.RECENCY,
    description="Recent (30 days) average vs historical average",
    formula="avg(recent_30_days) / avg(all_time)",
    bot_interpretation="HIGH = recent surge; LOW = declining; ~1 = stable",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="recent_volatility_ratio",
    category=FeatureCategory.RECENCY,
    description="Recent CV vs historical CV",
    formula="CV(recent) / CV(historical)",
    bot_interpretation="HIGH = recently more volatile; LOW = recently more stable",
    value_range=(0, None),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="recency_concentration",
    category=FeatureCategory.RECENCY,
    description="Fraction of all downloads in last 30 days",
    formula="downloads_last_30_days / total_downloads",
    bot_interpretation="VERY HIGH = new/sudden activity; LOW = established history",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="momentum_score",
    category=FeatureCategory.RECENCY,
    description="Exponentially-weighted recent trend direction",
    formula="(recent_avg - historical_avg) / historical_avg",
    bot_interpretation="POSITIVE = accelerating; NEGATIVE = decelerating",
    value_range=(-5, 5),
    stage=ComputationStage.POST_EXTRACTION
))

# -----------------------------------------------------------------------------
# TIME SERIES: DISTRIBUTION SHAPE FEATURES (4 features)
# -----------------------------------------------------------------------------

FeatureRegistry.register(FeatureDefinition(
    name="download_skewness",
    category=FeatureCategory.DISTRIBUTION,
    description="Skewness of daily download distribution",
    formula="skewness(daily_downloads)",
    bot_interpretation="RIGHT-SKEWED (+) = few high days; LEFT-SKEWED (-) = few low days",
    value_range=(-10, 10),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="download_kurtosis",
    category=FeatureCategory.DISTRIBUTION,
    description="Kurtosis of daily download distribution (tail heaviness)",
    formula="kurtosis(daily_downloads)",
    bot_interpretation="HIGH = heavy tails (extreme events); LOW = light tails",
    value_range=(-10, 50),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="tail_heaviness_ratio",
    category=FeatureCategory.DISTRIBUTION,
    description="99th percentile / median (extreme value ratio)",
    formula="P99(daily_downloads) / median(daily_downloads)",
    bot_interpretation="HIGH = extreme spikes relative to normal; LOW = uniform",
    value_range=(1, 100),
    stage=ComputationStage.POST_EXTRACTION
))

FeatureRegistry.register(FeatureDefinition(
    name="zero_day_ratio",
    category=FeatureCategory.DISTRIBUTION,
    description="Fraction of days with zero downloads",
    formula="count(days=0) / total_days",
    bot_interpretation="HIGH = sporadic activity; LOW = consistent presence",
    value_range=(0, 1),
    stage=ComputationStage.POST_EXTRACTION
))


# =============================================================================
# FEATURE LISTS BY CATEGORY (for backward compatibility)
# =============================================================================

# These lists can be imported directly for use in config.py and elsewhere

BASIC_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.BASIC)
TEMPORAL_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.TEMPORAL)
YEARLY_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.YEARLY)
BEHAVIORAL_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.BEHAVIORAL)
INTERACTION_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.INTERACTION)
SIGNATURE_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.SIGNATURE)
DISCRIMINATIVE_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.DISCRIMINATIVE)

# New feature categories
TIMING_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.TIMING)
ACCESS_PATTERN_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.ACCESS_PATTERN)
USER_DISTRIBUTION_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.USER_DISTRIBUTION)
STATISTICAL_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.STATISTICAL)
SESSION_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.SESSION)
COMPARATIVE_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.COMPARATIVE)

# Time series feature categories
OUTBURST_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.OUTBURST)
PERIODICITY_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.PERIODICITY)
TREND_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.TREND)
RECENCY_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.RECENCY)
DISTRIBUTION_FEATURES = FeatureRegistry.get_feature_names_by_category(FeatureCategory.DISTRIBUTION)

# All new features combined (from first batch)
NEW_FEATURES = (
    TIMING_FEATURES +
    ACCESS_PATTERN_FEATURES +
    USER_DISTRIBUTION_FEATURES +
    STATISTICAL_FEATURES +
    SESSION_FEATURES +
    COMPARATIVE_FEATURES
)

# Time series features combined
TIMESERIES_FEATURES = (
    OUTBURST_FEATURES +
    PERIODICITY_FEATURES +
    TREND_FEATURES +
    RECENCY_FEATURES +
    DISTRIBUTION_FEATURES
)

# Complete list of all features
ALL_FEATURES = FeatureRegistry.get_enabled()


def get_feature_documentation(feature_name: str) -> Optional[str]:
    """Get formatted documentation for a specific feature."""
    feature = FeatureRegistry.get(feature_name)
    if feature is None:
        return None

    return f"""
Feature: {feature.name}
Category: {feature.category.value}
Description: {feature.description}
Formula: {feature.formula}
Bot Interpretation: {feature.bot_interpretation}
Value Range: {feature.value_range}
Dependencies: {', '.join(feature.dependencies) if feature.dependencies else 'None'}
Stage: {feature.stage.value}
Enabled: {feature.enabled}
"""


def print_feature_summary() -> None:
    """Print summary of all registered features."""
    summary = FeatureRegistry.summary()
    print(f"\n{'='*60}")
    print("DEEPLOGBOT FEATURE REGISTRY SUMMARY")
    print(f"{'='*60}")
    print(f"Total Features: {summary['total']}")
    print(f"Enabled Features: {summary['enabled']}")
    print("\nBy Category:")
    for cat, count in summary['by_category'].items():
        if count > 0:
            print(f"  {cat}: {count}")
    print("\nBy Computation Stage:")
    for stage, count in summary['by_stage'].items():
        if count > 0:
            print(f"  {stage}: {count}")
    print(f"{'='*60}\n")
