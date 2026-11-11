"""Analytics service configuration.

SERVICE_MODE is the mode string that routes requests here. All LLM settings
come from the global Settings but are exposed here so future work can override
them per-service (e.g. switching this service to OpenRouter) without touching
shared infrastructure.
"""

from src.config import settings as _global_settings

SERVICE_MODE = "analytics"

# LLM provider for this service — mirrors global settings; override here when
# the analytics service needs a different provider than other services.
LLM_PROVIDER = _global_settings.llm_provider
MODEL_DATA_PROCESSING = _global_settings.gemini_model_data_processing
MODEL_ENRICHMENT = _global_settings.gemini_model_enrichment
