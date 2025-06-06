"""
Utility functions for SmokeSentinel data collection.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import httpx
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

logger = logging.getLogger(__name__)

async def async_download(
    url: str,
    dest_path: Path,
    session: httpx.AsyncClient,
    headers: dict | None = None,
    params: dict | None = None,
    retries: int = 3,
    chunk_size: int = 8192
) -> int:
    """
    Download a file asynchronously with retries.
    
    Args:
        url: URL to download from
        dest_path: Path to save the file
        session: httpx AsyncClient
        headers: Optional HTTP headers
        params: Optional URL parameters
        retries: Number of retry attempts
        chunk_size: Size of chunks to download
    
    Returns:
        int: HTTP status code (200 for success, 404 for not found, etc.)
    """
    for attempt in range(retries):
        try:
            response = await session.get(url, headers=headers, params=params)
            
            if response.status_code in (200, 404):
                if response.status_code == 200:
                    # Create parent directories if they don't exist
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download file in chunks
                    with open(dest_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size):
                            f.write(chunk)
                
                return response.status_code
                
            if attempt == retries - 1:
                logger.error(f"Failed to download {url} after {retries} attempts: {response.status_code}")
                return response.status_code
                
            logger.warning(f"Download attempt {attempt + 1} failed for {url}: {response.status_code}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"Failed to download {url} after {retries} attempts: {e}")
                return 500
            logger.warning(f"Download attempt {attempt + 1} failed for {url}: {e}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    return 500

def ensure_dir(path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
    """
    path.mkdir(parents=True, exist_ok=True)

def date_range_utc(end: datetime, days: int = 1) -> List[datetime]:
    """
    Generate a list of UTC dates from end_date going back 'days' days.
    
    Args:
        end: End date (inclusive)
        days: Number of days to go back
    
    Returns:
        List of datetime objects in UTC
    """
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    
    return [
        end - timedelta(days=i)
        for i in range(days)
    ]

def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging with rich formatting.
    
    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

def get_progress_bar() -> Progress:
    """
    Create a rich progress bar.
    
    Returns:
        Progress bar instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    )

def utc_now():
    return datetime.now(timezone.utc) 