"""
circuit_breaker.py — Circuit breaker pattern for API resilience.

Features:
  - Automatic failure detection and circuit opening
  - Half-open state for gradual recovery testing
  - Per-service circuit breakers (exchange APIs, price feeds, etc.)
  - Exponential backoff for retries

Usage:
  from circuit_breaker import CircuitBreaker, circuit_breaker_registry
  
  # Create a circuit breaker for an API
  cb = CircuitBreaker("hyperliquid_api", failure_threshold=3, recovery_timeout=60)
  
  # Use it to wrap API calls
  result = cb.call(exchange.get_account_state)
  
  # Or use as a decorator
  @circuit_breaker_registry.wrap("price_feed")
  def fetch_price(coin):
      return api.get_price(coin)
"""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict, TypeVar, Generic
from functools import wraps
from dataclasses import dataclass, field
from collections import defaultdict

from logger import get_logger

log = get_logger("circuit_breaker")

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject calls
    HALF_OPEN = "half_open" # Testing if recovered


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""
    failures: int = 0
    successes: int = 0
    rejects: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    opened_at: Optional[float] = None
    closed_at: Optional[float] = None
    consecutive_successes: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    def __init__(self, service_name: str, message: str = ""):
        self.service_name = service_name
        super().__init__(f"Circuit breaker OPEN for {service_name}: {message}")


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.
    
    States:
      CLOSED: Normal operation, calls pass through
      OPEN: Too many failures, calls are rejected immediately
      HALF_OPEN: After timeout, allow one test call to check recovery
    
    Args:
        name: Unique identifier for this circuit breaker
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before trying half-open
        half_open_max_calls: Max calls allowed in half-open state
        success_threshold: Consecutive successes to close circuit
        exception_types: Exception types to count as failures
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
        exception_types: tuple = (Exception,)
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.exception_types = exception_types
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0
        self._lock = threading.RLock()
        self._stats = CircuitBreakerStats()
        
        log.info(f"Circuit breaker '{name}' initialized (threshold={failure_threshold}, "
                f"timeout={recovery_timeout}s)")
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        with self._lock:
            return self._state
    
    @property
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        with self._lock:
            return CircuitBreakerStats(
                failures=self._stats.failures,
                successes=self._stats.successes,
                rejects=self._stats.rejects,
                last_failure_time=self._stats.last_failure_time,
                last_success_time=self._stats.last_success_time,
                opened_at=self._stats.opened_at,
                closed_at=self._stats.closed_at,
                consecutive_successes=self._stats.consecutive_successes
            )
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args, **kwargs: Arguments to pass to func
            
        Returns:
            Result from func
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception raised by func (if not a tracked failure)
        """
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    log.info(f"Circuit '{self.name}' entering HALF_OPEN state")
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                else:
                    self._stats.rejects += 1
                    raise CircuitBreakerError(
                        self.name, 
                        f"Circuit open, retry after {self._get_remaining_timeout():.0f}s"
                    )
            
            # In HALF_OPEN, limit concurrent test calls
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    self._stats.rejects += 1
                    raise CircuitBreakerError(
                        self.name, 
                        "Half-open call limit reached"
                    )
                self._half_open_calls += 1
        
        # Execute the call (outside lock to prevent blocking)
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.exception_types as e:
            self._on_failure()
            raise e
        finally:
            # Decrement half-open counter if applicable
            with self._lock:
                if self._state == CircuitState.HALF_OPEN:
                    self._half_open_calls = max(0, self._half_open_calls - 1)
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self._stats.successes += 1
            self._stats.last_success_time = time.time()
            self._stats.consecutive_successes += 1
            
            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.success_threshold:
                    log.info(f"Circuit '{self.name}' CLOSED (recovered)")
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._stats.closed_at = time.time()
                    self._stats.consecutive_successes = 0
            else:
                # In CLOSED state, reset failure count on success
                self._failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self._stats.failures += 1
            self._stats.last_failure_time = time.time()
            self._stats.consecutive_successes = 0
            
            if self._state == CircuitState.HALF_OPEN:
                # Failed in half-open, go back to open
                log.warning(f"Circuit '{self.name}' OPEN (failed in half-open)")
                self._state = CircuitState.OPEN
                self._stats.opened_at = time.time()
            else:
                # In CLOSED state, increment failure count
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    log.warning(f"Circuit '{self.name}' OPEN ({self._failure_count} failures)")
                    self._state = CircuitState.OPEN
                    self._stats.opened_at = time.time()
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try half-open."""
        if self._stats.opened_at is None:
            return True
        return (time.time() - self._stats.opened_at) >= self.recovery_timeout
    
    def _get_remaining_timeout(self) -> float:
        """Get remaining time before half-open attempt."""
        if self._stats.opened_at is None:
            return 0.0
        elapsed = time.time() - self._stats.opened_at
        return max(0.0, self.recovery_timeout - elapsed)
    
    def force_open(self):
        """Manually open the circuit (for maintenance, etc.)."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                log.info(f"Circuit '{self.name}' manually OPENED")
                self._state = CircuitState.OPEN
                self._stats.opened_at = time.time()
    
    def force_close(self):
        """Manually close the circuit (after fixing issue)."""
        with self._lock:
            if self._state != CircuitState.CLOSED:
                log.info(f"Circuit '{self.name}' manually CLOSED")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._stats.consecutive_successes = 0
                self._stats.closed_at = time.time()
    
    def reset(self):
        """Reset circuit to initial state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            self._stats = CircuitBreakerStats()
            log.info(f"Circuit '{self.name}' reset")
    
    def __repr__(self) -> str:
        return (f"CircuitBreaker(name='{self.name}', state={self._state.value}, "
                f"failures={self._failure_count}/{self.failure_threshold})")


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    
    Provides centralized management and monitoring of all circuits.
    """
    
    def __init__(self):
        self._circuits: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        **kwargs
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker.
        
        Args:
            name: Circuit breaker name
            failure_threshold: Failures before opening
            recovery_timeout: Seconds before half-open
            **kwargs: Additional CircuitBreaker parameters
            
        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name not in self._circuits:
                self._circuits[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    **kwargs
                )
            return self._circuits[name]
    
    def wrap(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        **kwargs
    ) -> Callable:
        """
        Decorator to wrap a function with circuit breaker.
        
        Usage:
            @registry.wrap("my_api")
            def call_api():
                return requests.get("...")
        """
        def decorator(func: Callable) -> Callable:
            cb = self.get(name, failure_threshold, recovery_timeout, **kwargs)
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                return cb.call(func, *args, **kwargs)
            
            # Attach circuit breaker to wrapper for access
            wrapper._circuit_breaker = cb
            return wrapper
        
        return decorator
    
    def get_stats(self) -> Dict[str, CircuitBreakerStats]:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return {name: cb.stats for name, cb in self._circuits.items()}
    
    def get_states(self) -> Dict[str, str]:
        """Get current states of all circuit breakers."""
        with self._lock:
            return {name: cb.state.value for name, cb in self._circuits.items()}
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for cb in self._circuits.values():
                cb.reset()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Get overall health status of all circuits.
        
        Returns:
            Dict with health information
        """
        states = self.get_states()
        total = len(states)
        open_circuits = sum(1 for s in states.values() if s == "open")
        half_open = sum(1 for s in states.values() if s == "half_open")
        
        return {
            "healthy": open_circuits == 0,
            "total_circuits": total,
            "open_circuits": open_circuits,
            "half_open_circuits": half_open,
            "closed_circuits": total - open_circuits - half_open,
            "circuit_states": states
        }


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


# Pre-configured circuit breakers for common services
def get_exchange_circuit(exchange_name: str) -> CircuitBreaker:
    """Get circuit breaker for exchange API (more lenient)."""
    return circuit_breaker_registry.get(
        name=f"exchange_{exchange_name}",
        failure_threshold=3,
        recovery_timeout=30.0,
        half_open_max_calls=2,
        success_threshold=2
    )


def get_price_feed_circuit(feed_name: str = "default") -> CircuitBreaker:
    """Get circuit breaker for price feed (strict)."""
    return circuit_breaker_registry.get(
        name=f"price_feed_{feed_name}",
        failure_threshold=5,
        recovery_timeout=60.0,
        half_open_max_calls=3,
        success_threshold=2
    )


def get_indicator_circuit(indicator_name: str = "default") -> CircuitBreaker:
    """Get circuit breaker for indicator calculations."""
    return circuit_breaker_registry.get(
        name=f"indicator_{indicator_name}",
        failure_threshold=3,
        recovery_timeout=30.0,
        half_open_max_calls=2,
        success_threshold=1
    )


# Convenience function for retry with exponential backoff
def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
    *args,
    **kwargs
) -> T:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Exceptions to catch and retry
        *args, **kwargs: Arguments for func
        
    Returns:
        Result from func
        
    Raises:
        Exception: Last exception after all retries exhausted
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                log.warning(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
                time.sleep(delay)
    
    raise last_exception
