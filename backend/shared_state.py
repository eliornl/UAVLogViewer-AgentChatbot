"""Shared state module for the UAV Log Viewer application.

This module contains shared state variables used across different modules
to prevent circular imports.
"""

import asyncio
from typing import Dict, Any

# In-memory session, agent, and lock storage
sessions = {}
agents = {}
per_session_locks = {}
