"""Provider management for DeepLogBot.

This module manages provider-specific configurations, schemas, and classification rules.

Each provider represents a different log source (e.g., EBI/PRIDE, web server logs, FTP logs)
with its own:
- Schema: Field mappings for the log format
- Config: Classification thresholds and rules
- Feature Extractors: Custom feature extraction logic
- Taxonomy: Category definitions and subcategories

Usage:
    from deeplogbot.providers import get_provider, list_providers, register_provider

    # Get a specific provider
    provider = get_provider('ebi')
    config = provider.get_config()
    schema = provider.get_schema()

    # List available providers
    available = list_providers()

    # Register a custom provider
    register_provider('my_provider', MyProviderClass)
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass, field

import yaml

from ..utils import logger


# Base path for provider configurations
PROVIDERS_DIR = Path(__file__).parent
BASE_TAXONOMY_PATH = PROVIDERS_DIR / "base_taxonomy.yaml"


@dataclass
class ProviderConfig:
    """Configuration for a log provider.

    Attributes:
        name: Provider identifier (e.g., 'ebi', 'custom')
        display_name: Human-readable name
        config_path: Path to provider's config.yaml
        schema_module: Module path for schema class
        extractor_module: Module path for feature extractors
    """
    name: str
    display_name: str
    config_path: Path
    schema_module: Optional[str] = None
    extractor_module: Optional[str] = None
    _config_cache: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _taxonomy_cache: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def get_config(self) -> Dict[str, Any]:
        """Load and return the provider configuration.

        Returns:
            Configuration dictionary.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            yaml.YAMLError: If the config file is malformed.
        """
        if self._config_cache is None:
            try:
                with open(self.config_path, 'r') as f:
                    self._config_cache = yaml.safe_load(f) or {}
            except FileNotFoundError:
                logger.error(f"Configuration file not found: {self.config_path}")
                raise
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML configuration: {self.config_path}: {e}")
                raise
        return self._config_cache

    def get_taxonomy(self) -> Dict[str, Any]:
        """Get merged taxonomy (base + provider overrides)."""
        if self._taxonomy_cache is None:
            self._taxonomy_cache = _merge_taxonomy(self.get_config())
        return self._taxonomy_cache

    def get_behavior_type_rules(self) -> Dict[str, Any]:
        """Get behavior type classification rules."""
        taxonomy = self.get_taxonomy()
        return taxonomy.get('behavior_type', {})

    def get_automation_category_rules(self) -> Dict[str, Any]:
        """Get automation category classification rules."""
        taxonomy = self.get_taxonomy()
        return taxonomy.get('automation_category', {})

    def get_subcategory_rules(self) -> Dict[str, Any]:
        """Get subcategory classification rules."""
        taxonomy = self.get_taxonomy()
        return taxonomy.get('subcategories', {})

    def get_subcategories_by_parent(self, parent: str) -> Dict[str, Any]:
        """Get subcategories for a specific parent category."""
        subcategories = self.get_subcategory_rules()
        return {
            name: rules for name, rules in subcategories.items()
            if rules.get('parent') == parent
        }

    def get_rule_based_config(self) -> Dict[str, Any]:
        """Get rule-based classification config."""
        config = self.get_config()
        return config.get('rule_based', {})

    def get_deep_classification_config(self) -> Dict[str, Any]:
        """Get deep classification config."""
        config = self.get_config()
        return config.get('deep_classification', {})

    def get_schema_config(self) -> Dict[str, Any]:
        """Get schema configuration."""
        config = self.get_config()
        return config.get('schema', {})


def _load_base_taxonomy() -> Dict[str, Any]:
    """Load the base taxonomy file."""
    if BASE_TAXONOMY_PATH.exists():
        with open(BASE_TAXONOMY_PATH, 'r') as f:
            return yaml.safe_load(f)
    return {}


def _merge_taxonomy(provider_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge base taxonomy with provider-specific overrides.

    Provider config can override any part of the base taxonomy.
    Nested dictionaries are merged recursively.
    """
    base = _load_base_taxonomy()

    # Deep merge provider config into base
    def deep_merge(base_dict: Dict, override_dict: Dict) -> Dict:
        result = base_dict.copy()
        for key, value in override_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    # Merge each taxonomy section
    merged = base.copy()
    for section in ['taxonomy', 'behavior_type', 'automation_category', 'subcategories',
                    'required_features', 'derived_columns']:
        if section in provider_config:
            if section in merged:
                merged[section] = deep_merge(merged[section], provider_config[section])
            else:
                merged[section] = provider_config[section]

    return merged


# Provider registry
_PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {}


def register_provider(name: str, config: ProviderConfig) -> None:
    """Register a provider configuration.

    Args:
        name: Provider identifier
        config: ProviderConfig instance
    """
    _PROVIDER_REGISTRY[name] = config
    logger.debug(f"Registered provider: {name}")


def get_provider(name: str) -> ProviderConfig:
    """Get a provider by name.

    Args:
        name: Provider identifier

    Returns:
        ProviderConfig instance

    Raises:
        ValueError: If provider not found
    """
    if name not in _PROVIDER_REGISTRY:
        available = list(_PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown provider: {name}. Available providers: {available}")
    return _PROVIDER_REGISTRY[name]


def list_providers() -> List[str]:
    """List all registered providers."""
    return list(_PROVIDER_REGISTRY.keys())


def get_default_provider() -> str:
    """Get the default provider name."""
    return "ebi"


def _auto_discover_providers() -> None:
    """Auto-discover and register providers from the providers directory."""
    for item in PROVIDERS_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('_'):
            config_path = item / "config.yaml"
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)

                    provider_info = config_data.get('provider', {})
                    name = provider_info.get('name', item.name)
                    display_name = provider_info.get('display_name', name.upper())

                    provider = ProviderConfig(
                        name=name,
                        display_name=display_name,
                        config_path=config_path,
                        schema_module=f"deeplogbot.features.providers.{name}",
                        extractor_module=f"deeplogbot.features.providers.{name}",
                    )
                    register_provider(name, provider)
                except Exception as e:
                    logger.warning(f"Failed to load provider from {item}: {e}")


# Auto-discover providers on module load
_auto_discover_providers()


__all__ = [
    "ProviderConfig",
    "get_provider",
    "list_providers",
    "register_provider",
    "get_default_provider",
]
