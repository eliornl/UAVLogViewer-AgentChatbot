"""Shared state module for the UAV Log Viewer application.

This module contains shared state variables used across different modules
to prevent circular imports. It provides global dictionaries for storing
session data, agent instances, and synchronization locks.

The shared state pattern is used here to avoid circular dependencies between
modules while still allowing them to access common state. This is particularly
important for the WebSocket implementation which needs access to session data
and agents from multiple modules.
"""

from typing import Dict, Any
from asyncio import Lock
from backend.models import Session

# In-memory session, agent, and lock storage
sessions: Dict[str, Session] = {}  # Maps session_id to Session objects
agents: Dict[str, Any] = {}  # Maps session_id to agent instances
per_session_locks: Dict[str, Lock] = {}  # Maps session_id to asyncio Lock objects
