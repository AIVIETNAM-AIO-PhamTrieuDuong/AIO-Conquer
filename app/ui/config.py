import os
import httpx

API_BASE_URL = os.getenv("AIO_API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
EDA_POLL_INTERVAL_SECONDS = 2
EDA_MAX_WAIT_SECONDS = int(os.getenv("AIO_EDA_MAX_WAIT_SECONDS", "300"))
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}