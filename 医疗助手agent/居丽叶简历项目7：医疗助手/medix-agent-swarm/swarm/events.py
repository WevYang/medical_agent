"""
兼容层：保留旧的 `swarm.events` 导入路径。
"""

from .events_20260428_231035 import Event, EventType

__all__ = ["Event", "EventType"]
