import pytest

from app.logging.event_logger import EventLogger
from app.logging.store import Store


def make(tmp_path):
    return EventLogger(Store(tmp_path / "nested" / "db.sqlite"))


def test_structured_events_roundtrip(tmp_path):
    log = make(tmp_path)
    log.log("q1", "query_received", mode="smart")
    log.log("q1", "route_selected", selected_tier="cheap")
    events = log.read()
    assert [e["event_type"] for e in events] == ["query_received", "route_selected"]
    assert set(events[0]) == {"timestamp", "query_id", "event_type", "payload"}
    assert events[1]["payload"] == {"selected_tier": "cheap"}
    log.log("q2", "pipeline_completed")
    assert len(log.read()) == 3
    assert [e["query_id"] for e in log.read(query_id="q2")] == ["q2"]


def test_rejects_unknown_event_type(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path).log("q", "made_up")


def test_empty_store_reads_empty(tmp_path):
    assert make(tmp_path).read() == []


def test_provider_error_counts_aggregate_in_sql(tmp_path):
    log = make(tmp_path)
    for _ in range(2):
        log.log("q", "provider_error", provider="groq", error_type="RateLimitError")
    log.log("q", "provider_error", provider="gemini", error_type="ProviderTimeoutError")
    assert log.provider_error_counts() == {"groq:RateLimitError": 2, "gemini:ProviderTimeoutError": 1}


def test_storage_failure_is_counted_not_raised(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    log = EventLogger(store)
    store.close()  # every write now fails with sqlite3.ProgrammingError (a sqlite3.Error)
    log.log("q", "query_received")
    assert log.write_failures == 1


def test_corrupt_database_file_is_a_clear_config_error(tmp_path):
    from app.core.config import ConfigError

    path = tmp_path / "db.sqlite"
    path.write_bytes(b"not a database" * 100)
    with pytest.raises(ConfigError, match="Move it aside"):
        Store(path)
