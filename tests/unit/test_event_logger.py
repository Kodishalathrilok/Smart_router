import pytest

from app.logging.event_logger import EventLogger


def test_append_only_structured_events(tmp_path):
    log = EventLogger(tmp_path / "nested" / "events.jsonl")
    log.log("q1", "query_received", mode="smart")
    log.log("q1", "route_selected", selected_tier="cheap")
    events = list(log.read())
    assert [e["event_type"] for e in events] == ["query_received", "route_selected"]
    assert set(events[0]) == {"timestamp", "query_id", "event_type", "payload"}
    assert events[1]["payload"] == {"selected_tier": "cheap"}
    log.log("q2", "pipeline_completed")
    assert len(list(log.read())) == 3


def test_rejects_unknown_event_type(tmp_path):
    with pytest.raises(ValueError):
        EventLogger(tmp_path / "e.jsonl").log("q", "made_up")


def test_read_missing_file_is_empty(tmp_path):
    assert list(EventLogger(tmp_path / "none.jsonl").read()) == []
