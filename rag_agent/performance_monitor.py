"""
Performance monitoring module for the RAG agent system.
Provides detailed timing and performance metrics for key operations.
"""
import time
import logging
from typing import Dict, Callable, Any
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Performance metrics storage
performance_metrics = {
    'query_processing_times': [],
    'embedding_generation_times': [],
    'vector_search_times': [],
    'response_generation_times': [],
    'total_request_times': []
}


class PerformanceMonitor:
    """
    Class to monitor and track performance metrics for the RAG system.
    """

    def __init__(self):
        self.metrics = performance_metrics

    def record_metric(self, metric_name: str, value: float):
        """
        Record a performance metric.

        Args:
            metric_name: Name of the metric to record
            value: Value of the metric
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

        # Log the metric
        logger.info(f"Performance metric recorded - {metric_name}: {value:.4f}s")

    def get_average_metric(self, metric_name: str) -> float:
        """
        Get the average value for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Average value of the metric
        """
        if metric_name in self.metrics and self.metrics[metric_name]:
            values = self.metrics[metric_name]
            return sum(values) / len(values)
        return 0.0

    def get_performance_report(self) -> Dict[str, Dict[str, float]]:
        """
        Generate a comprehensive performance report.

        Returns:
            Dictionary with performance metrics summary
        """
        report = {}
        for metric_name, values in self.metrics.items():
            if values:
                report[metric_name] = {
                    'count': len(values),
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'total': sum(values)
                }
        return report

    def reset_metrics(self):
        """
        Reset all performance metrics.
        """
        for key in self.metrics:
            self.metrics[key].clear()
        logger.info("Performance metrics reset")


# Global performance monitor instance
monitor = PerformanceMonitor()


def performance_timer(metric_name: str):
    """
    Decorator to time function execution and record performance metrics.

    Args:
        metric_name: Name of the metric to record
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                monitor.record_metric(metric_name, duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                monitor.record_metric(metric_name, duration)

        # Return the appropriate wrapper based on function type
        if func.__name__.startswith('async_') or asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@contextmanager
def performance_context(metric_name: str):
    """
    Context manager to time code blocks and record performance metrics.

    Args:
        metric_name: Name of the metric to record
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        monitor.record_metric(metric_name, duration)


def log_performance_summary():
    """
    Log a summary of performance metrics.
    """
    report = monitor.get_performance_report()

    logger.info("=== PERFORMANCE METRICS SUMMARY ===")
    for metric_name, stats in report.items():
        logger.info(f"{metric_name}:")
        logger.info(f"  Count: {stats['count']}")
        logger.info(f"  Avg: {stats['average']:.4f}s")
        logger.info(f"  Min: {stats['min']:.4f}s")
        logger.info(f"  Max: {stats['max']:.4f}s")
        logger.info(f"  Total: {stats['total']:.4f}s")
    logger.info("==================================")