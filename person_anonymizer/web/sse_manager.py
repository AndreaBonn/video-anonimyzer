"""
Gestione Server-Sent Events per il progresso della pipeline.
Supporta più client connessi allo stesso job.
"""

import threading
from queue import Queue


class SSEManager:
    """Gestisce la distribuzione di eventi SSE ai client sottoscritti."""

    def __init__(self):
        self._lock = threading.Lock()
        # job_id -> list[Queue]
        self._subscribers: dict[str, list[Queue]] = {}

    def subscribe(self, job_id: str) -> Queue:
        """Sottoscrive un client agli eventi di un job. Restituisce la coda."""
        q = Queue()
        with self._lock:
            if job_id not in self._subscribers:
                self._subscribers[job_id] = []
            self._subscribers[job_id].append(q)
        return q

    def unsubscribe(self, job_id: str, q: Queue):
        """Rimuove un client dalla sottoscrizione."""
        with self._lock:
            if job_id in self._subscribers:
                try:
                    self._subscribers[job_id].remove(q)
                except ValueError:
                    pass
                if not self._subscribers[job_id]:
                    del self._subscribers[job_id]

    def emit(self, job_id: str, event_type: str, data: dict):
        """Invia un evento a tutti i client sottoscritti a un job."""
        event = {"type": event_type, "data": data}
        with self._lock:
            for q in self._subscribers.get(job_id, []):
                q.put(event)

    def close(self, job_id: str):
        """Chiude tutte le code di un job (segnale di fine stream)."""
        with self._lock:
            for q in self._subscribers.get(job_id, []):
                q.put(None)
            self._subscribers.pop(job_id, None)
