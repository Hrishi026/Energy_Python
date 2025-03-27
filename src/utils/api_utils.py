import time
import logging
from typing import Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, base_url: str, api_key: str, rate_limit: int = 60, max_retries: int = 3):
        """
        Initialize API client with rate limiting and retry logic.
        
        Args:
            base_url: Base URL for the API
            api_key: API key for authentication
            rate_limit: Number of requests allowed per minute
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.requests_per_minute = 0
        self.last_request_time = 0
        
        # Set up session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })

    def _handle_rate_limit(self) -> None:
        """Handle rate limiting by waiting if necessary."""
        current_time = time.time()
        
        # Reset counter if a minute has passed
        if current_time - self.last_request_time >= 60:
            self.requests_per_minute = 0
            self.last_request_time = current_time
        
        # Wait if rate limit is reached
        if self.requests_per_minute >= self.rate_limit:
            wait_time = 60 - (current_time - self.last_request_time)
            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                self.requests_per_minute = 0
                self.last_request_time = time.time()
        
        self.requests_per_minute += 1

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GET request to the API with rate limiting and retries.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            API response as dictionary
        """
        try:
            self._handle_rate_limit()
            
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a POST request to the API with rate limiting and retries.
        
        Args:
            endpoint: API endpoint
            data: Request body data
            
        Returns:
            API response as dictionary
        """
        try:
            self._handle_rate_limit()
            
            url = f"{self.base_url}/{endpoint}"
            response = self.session.post(url, json=data, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise

    def close(self) -> None:
        """Close the API session."""
        self.session.close() 