"""Small in-memory session store for multi-turn context."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class Message:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SessionStore:
    def __init__(self, max_turns: int = 12):
        self.max_turns = max_turns
        self._sessions: dict[str, list[Message]] = {}

    def create_or_get(self, session_id: str | None = None) -> str:
        sid = session_id or str(uuid4())
        self._sessions.setdefault(sid, [])
        return sid

    def append(self, session_id: str, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        messages = self._sessions.setdefault(session_id, [])
        messages.append(Message(role=role, content=content, metadata=metadata or {}))
        max_messages = self.max_turns * 2
        if len(messages) > max_messages:
            del messages[:-max_messages]

    def get(self, session_id: str) -> list[Message]:
        return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
