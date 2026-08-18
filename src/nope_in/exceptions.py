"""Custom exceptions for NOPE-IN data pipeline."""

from __future__ import annotations


class DataFetchError(Exception):
    """Raised when a data source fetch fails irrecoverably."""


class DataQualityError(Exception):
    """Raised when data fails critical validation checks."""
