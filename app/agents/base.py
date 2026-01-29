from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_not_exception_type,
)
from openai import APIConnectionError


class BaseAgent(ABC):
    """Abstract base class for all agents in the system.

    Design decisions:
    - Abstract interface ensures consistent agent behavior
    - Built-in retry logic for resilience
    - Standardized logging and error handling
    - Metrics collection hooks for monitoring
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"agents.{name}")
        self.metrics = {}

    @abstractmethod
    async def process(self, input_data: Any, **kwargs) -> Any:
        """Main processing method that each agent must implement."""
        pass

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_not_exception_type(APIConnectionError),
    )
    async def execute_with_retry(self, input_data: Any, **kwargs) -> Any:
        """Execute the agent with retry logic for resilience."""
        start_time = datetime.utcnow()
        try:
            self.logger.info(f"Starting {self.name} agent processing")
            result = await self.process(input_data, **kwargs)

            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.metrics["last_processing_time"] = processing_time
            self.metrics["last_success"] = datetime.utcnow()

            self.logger.info(f"{self.name} completed in {processing_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"{self.name} failed: {str(e)}", exc_info=True)
            self.metrics["last_error"] = str(e)
            self.metrics["last_failure"] = datetime.utcnow()
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Return agent metrics for monitoring."""
        return {"agent_name": self.name, "config": self.config, **self.metrics}
