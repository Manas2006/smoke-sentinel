#!/usr/bin/env python3
"""
SmokeSentinel Data Collection Script

This script handles the collection of data from multiple sources:
- NASA FIRMS hotspot data
- NOAA HRRR-Smoke data
- PurpleAir and OpenAQ sensor data

Usage:
    python -m scripts.collect_data --days 1 --log-level DEBUG
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import httpx
from rich.logging import RichHandler
from rich.progress import Progress

from smokesentinel.scripts.config import settings
from smokesentinel.scripts.utils import (
    async_download,
    date_range_utc,
    ensure_dir,
    get_progress_bar,
    setup_logging,
    utc_now
)

logger = logging.getLogger(__name__)

async def _download_firms(
    session: httpx.AsyncClient,
    dates: List[datetime],
    out_dir: Path,
    progress: Progress
) -> None:
    """
    Download FIRMS hotspot data.
    
    Args:
        session: httpx AsyncClient
        dates: List of dates to download
        out_dir: Output directory
        progress: Rich progress bar
    """
    task = progress.add_task("[cyan]Downloading FIRMS data...", total=len(dates))
    
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        url = f"{settings.firms.base_url}/{settings.firms.product}/{date_str}"
        dest_path = out_dir / "firms" / f"FIRMS_{date_str}.tif"
        
        status = await async_download(url, dest_path, session)
        if status == 200:
            logger.info(f"Downloaded FIRMS data for {date_str}")
        
        progress.update(task, advance=1)

async def _download_hrrr_smoke(
    session: httpx.AsyncClient,
    run_date: datetime,
    out_dir: Path,
    progress: Progress
) -> None:
    """
    Download all f00–f48 smoke10 GRIB2 files for the four HRRR cycles.
    
    Args:
        session: httpx AsyncClient
        run_date: Date to download data for
        out_dir: Output directory
        progress: Rich progress bar
    """
    cfg = settings.hrrr
    total_files = len(cfg.cycles) * (cfg.max_fhour + 1)
    task = progress.add_task("[green]Downloading HRRR-Smoke data...", total=total_files)
    
    for cycle in cfg.cycles:
        run_dt = run_date.replace(hour=cycle, minute=0, second=0, microsecond=0)
        for fhour in range(cfg.max_fhour + 1):
            fstr = f"{fhour:02d}"
            url = (
                f"{cfg.bucket}/hrrr.{run_dt:%Y%m%d}/{cfg.domain}/"
                f"hrrr.t{cycle:02d}z.{cfg.product}.f{fstr}.grib2"
            )
            dest = (
                out_dir / "hrrr" / f"{run_dt:%Y%m%d}" /
                f"hrrr_{cycle:02d}z_f{fstr}.grib2"
            )
            
            if dest.exists():
                progress.update(task, advance=1)
                continue
                
            status = await async_download(url, dest, session)
            if status == 404:
                logger.debug("HRRR %s not present", url)
            
            progress.update(task, advance=1)

async def _download_openaq(
    session: httpx.AsyncClient,
    start: datetime,
    end: datetime,
    out_dir: Path,
    progress: Progress
) -> None:
    """
    Pull raw PM2.5 measurements in 1-day chunks (UTC).
    
    Args:
        session: httpx AsyncClient
        start: Start date
        end: End date
        out_dir: Output directory
        progress: Rich progress bar
    """
    task = progress.add_task("[yellow]Downloading OpenAQ data...", total=1)
    
    base = settings.openaq.base_url + "/measurements"
    params = {
        "parameter": "pm25",
        "date_from": start.isoformat(timespec="seconds"),
        "date_to": end.isoformat(timespec="seconds"),
        "limit": settings.openaq.page_size,
        "country": "US",
        "page": 1,
    }
    
    dest = out_dir / "openaq" / f"openaq_{start:%Y%m%d}.json"
    status = await async_download(base, dest, session, params=params)
    
    if status == 200:
        logger.info(f"Downloaded OpenAQ data for {start:%Y-%m-%d}")
    else:
        logger.warning(f"OpenAQ download failed with status {status}")
    
    progress.update(task, advance=1)

async def _download_purpleair(
    session: httpx.AsyncClient,
    sensor_ids: List[int],
    out_dir: Path,
    progress: Progress
) -> None:
    """
    Download PurpleAir sensor data.
    
    Args:
        session: httpx AsyncClient
        sensor_ids: List of sensor IDs to download
        out_dir: Output directory
        progress: Rich progress bar
    """
    if not settings.PURPLEAIR_API_KEY:
        logger.warning("PurpleAir API key not set, skipping download")
        return
        
    task = progress.add_task("[magenta]Downloading PurpleAir data...", total=len(sensor_ids))
    
    headers = {"X-API-Key": settings.PURPLEAIR_API_KEY}
    base = settings.purpleair.base_url + "/sensors"
    params = {"fields": "pm2.5_atm", "read_key": None}
    
    for sid in sensor_ids:
        url = f"{base}/{sid}"
        dest = out_dir / "purpleair" / f"{sid}_{utc_now():%Y%m%dT%H%M}.json"
        
        status = await async_download(url, dest, session, headers=headers, params=params)
        if status != 200:
            logger.warning(f"PurpleAir sensor {sid} → {status}")
        
        progress.update(task, advance=1)

async def main(days: int, out_dir: Path, log_level: str) -> None:
    """
    Main function to orchestrate data collection.
    
    Args:
        days: Number of days of historical data to collect
        out_dir: Output directory for downloaded files
        log_level: Logging level
    """
    setup_logging(log_level)
    ensure_dir(out_dir)
    
    end_date = datetime.now(timezone.utc)
    dates = date_range_utc(end_date, days)
    
    logger.info(f"Collecting data from {dates[-1]} to {dates[0]}")
    
    async with httpx.AsyncClient() as session:
        with get_progress_bar() as progress:
            await asyncio.gather(
                _download_firms(session, dates, out_dir, progress),
                _download_hrrr_smoke(session, dates[0], out_dir, progress),
                _download_openaq(session, dates[-1], dates[0], out_dir, progress),
                _download_purpleair(session, [42], out_dir, progress)  # Example sensor ID
            )
    
    logger.info("Data collection completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect data for SmokeSentinel")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days of historical data to collect"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for downloaded files"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    asyncio.run(main(args.days, args.out_dir, args.log_level)) 