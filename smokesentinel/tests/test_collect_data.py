"""
Tests for the data collection script.
"""

import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, AsyncMock

import httpx
import pytest_httpx

from smokesentinel.scripts.collect_data import (
    _download_firms,
    _download_hrrr_smoke,
    _download_openaq,
    _download_purpleair
)
from smokesentinel.scripts.config import settings
from smokesentinel.scripts.utils import date_range_utc

@pytest.fixture
def mock_date():
    """Create a mock date for testing."""
    return datetime(2024, 1, 1, 12, 0)

@pytest.fixture
def mock_output_dir(tmp_path):
    """Create a temporary output directory."""
    return tmp_path / "data"

@pytest.fixture
def mock_progress():
    """Create a mock progress bar."""
    with patch("smokesentinel.scripts.collect_data.Progress") as mock:
        mock.return_value.__enter__.return_value = AsyncMock()
        yield mock

@pytest.fixture
def mock_dates():
    """Create a list of mock dates."""
    return [
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2023, 12, 31, tzinfo=timezone.utc)
    ]

@pytest.fixture
def mock_out_dir(tmp_path):
    """Create a temporary output directory."""
    return tmp_path / "data" / "raw"

@pytest.mark.asyncio
async def test_download_firms(httpx_mock, mock_dates, mock_out_dir, mock_progress):
    """Test FIRMS data download."""
    # Mock HTTP response
    httpx_mock.add_response(
        url="https://firms.modaps.eosdis.nasa.gov/api/area/csv/VIIRS_SNPP_NRT/2024-01-01",
        content=b"mock firms data"
    )
    httpx_mock.add_response(
        url="https://firms.modaps.eosdis.nasa.gov/api/area/csv/VIIRS_SNPP_NRT/2023-12-31",
        content=b"mock firms data"
    )
    
    # Create mock session
    async with httpx.AsyncClient() as session:
        await _download_firms(session, mock_dates, mock_out_dir, mock_progress)
    
    # Check if file was created
    assert (mock_out_dir / "firms" / "FIRMS_2024-01-01.tif").exists()
    assert (mock_out_dir / "firms" / "FIRMS_2023-12-31.tif").exists()

@pytest.mark.asyncio
async def test_download_hrrr_smoke(httpx_mock, mock_out_dir, mock_progress):
    """Test HRRR-Smoke data download."""
    # Mock HTTP responses for different cycles and forecast hours
    base_url = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240101/conus"
    for cycle in [0, 6, 12, 18]:
        for fhour in range(49):  # 0-48 inclusive
            fstr = f"{fhour:02d}"
            httpx_mock.add_response(
                url=f"{base_url}/hrrr.t{cycle:02d}z.smoke10.f{fstr}.grib2",
                content=b"mock hrrr data"
            )
    
    # Create mock session
    async with httpx.AsyncClient() as session:
        await _download_hrrr_smoke(
            session,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            mock_out_dir,
            mock_progress
        )
    
    # Check if files were created
    for cycle in [0, 6, 12, 18]:
        for fhour in range(49):
            fstr = f"{fhour:02d}"
            assert (
                mock_out_dir / "hrrr" / "20240101" /
                f"hrrr_{cycle:02d}z_f{fstr}.grib2"
            ).exists()

@pytest.mark.asyncio
async def test_download_openaq(httpx_mock, mock_out_dir, mock_progress):
    """Test OpenAQ v3 data download."""
    # Mock HTTP response
    httpx_mock.add_response(
        url="https://api.openaq.org/v3/measurements?parameter=pm25&date_from=2024-01-01T00%3A00%3A00%2B00%3A00&date_to=2024-01-02T00%3A00%3A00%2B00%3A00&limit=10000&country=US&page=1",
        method="GET",
        match_headers={"Accept": "*/*"},
        content=b'{"results": [], "meta": {"found": 0}}'
    )
    
    # Create mock session
    async with httpx.AsyncClient() as session:
        await _download_openaq(
            session,
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
            mock_out_dir,
            mock_progress
        )
    
    # Check if file was created
    assert (mock_out_dir / "openaq" / "openaq_20240101.json").exists()

@pytest.mark.asyncio
async def test_download_purpleair(httpx_mock, mock_out_dir, mock_progress):
    """Test PurpleAir data download."""
    # Mock HTTP response
    httpx_mock.add_response(
        url="https://api.purpleair.com/v1/sensors/42?fields=pm2.5_atm&read_key=",
        method="GET",
        match_headers={"X-API-Key": "test_key"},
        content=b'{"sensor": {"pm2.5_atm": 10.5}}'
    )
    
    # Set API key
    settings.PURPLEAIR_API_KEY = "test_key"

    # Create mock session
    async with httpx.AsyncClient() as session:
        await _download_purpleair(session, [42], mock_out_dir, mock_progress)
    
    # Check if file was created
    assert any((mock_out_dir / "purpleair").glob("42_*.json"))

@pytest.mark.asyncio
async def test_download_retry(httpx_mock, mock_dates, mock_out_dir, mock_progress):
    """Test download retry mechanism."""
    # Mock HTTP response with initial failure
    httpx_mock.add_response(
        url="https://firms.modaps.eosdis.nasa.gov/api/area/csv/VIIRS_SNPP_NRT/2024-01-01",
        status_code=500
    )
    httpx_mock.add_response(
        url="https://firms.modaps.eosdis.nasa.gov/api/area/csv/VIIRS_SNPP_NRT/2024-01-01",
        content=b"mock firms data"
    )
    httpx_mock.add_response(
        url="https://firms.modaps.eosdis.nasa.gov/api/area/csv/VIIRS_SNPP_NRT/2023-12-31",
        content=b"mock firms data"
    )
    
    # Create mock session
    async with httpx.AsyncClient() as session:
        await _download_firms(session, mock_dates, mock_out_dir, mock_progress)
    
    # Check if file was created after retry
    assert (mock_out_dir / "firms" / "FIRMS_2024-01-01.tif").exists()
    assert (mock_out_dir / "firms" / "FIRMS_2023-12-31.tif").exists() 