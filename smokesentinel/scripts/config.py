import os
import tomllib
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Optional, Dict, Any

class FIRMSSettings(BaseModel):
    product: str = "VIIRS_SNPP_NRT"
    resolution: int = 375  # m
    base_url: str = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

class HRRRSettings(BaseModel):
    bucket: str = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
    domain: str = "conus"
    product: str = "smoke10"  # surface PM₂.₅ µg m-3
    cycles: List[int] = [0, 6, 12, 18]
    max_fhour: int = 48  # inclusive

class OpenAQSettings(BaseModel):
    base_url: str = "https://api.openaq.org/v3"
    page_size: int = 10000

class PurpleAirSettings(BaseModel):
    base_url: str = "https://api.purpleair.com/v1"

class LoggingSettings(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

class Settings(BaseModel):
    # FIRMS settings
    firms: FIRMSSettings = Field(default_factory=FIRMSSettings)
    
    # HRRR settings
    hrrr: HRRRSettings = Field(default_factory=HRRRSettings)
    
    # OpenAQ settings
    openaq: OpenAQSettings = Field(default_factory=OpenAQSettings)
    
    # PurpleAir settings
    purpleair: PurpleAirSettings = Field(default_factory=PurpleAirSettings)
    
    # Logging settings
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    # API keys
    FIRMS_API_KEY: Optional[str] = None
    PURPLEAIR_API_KEY: Optional[str] = None

def load_settings() -> Settings:
    """Load settings from settings.toml file."""
    settings_path = Path(__file__).parent.parent / "settings.toml"
    
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found at {settings_path}")
    
    with open(settings_path, "rb") as f:
        data = tomllib.load(f)
    
    # Create Settings instance with the loaded data
    return Settings(**data)

# Create global settings instance
settings = load_settings() 