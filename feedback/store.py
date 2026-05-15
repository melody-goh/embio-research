from storage.db import store_feedback


def record_feedback(source_id: str, source_type: str, signal: int, notes: str = "") -> None:
    store_feedback(source_id, source_type, signal, notes)
