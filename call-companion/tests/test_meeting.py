from datetime import datetime, timedelta, timezone

from call_companion.meeting.detector import pick_meeting, running_call_apps
from call_companion.models import Meeting

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


def meeting(id: str, start_min: float, end_min: float | None = None) -> Meeting:
    return Meeting(
        id=id,
        title=id,
        start=NOW + timedelta(minutes=start_min),
        end=NOW + timedelta(minutes=end_min) if end_min is not None else None,
    )


def test_call_apps_are_recognised_by_process_name():
    names = ["launchd", "zoom.us", "Microsoft Teams (work or school)", "Google Chrome", "Slack Helper", "zoom.us"]
    assert running_call_apps(names) == ["slack", "teams", "zoom"]
    assert running_call_apps(["Finder", "Safari"]) == []


def test_the_running_meeting_beats_the_next_one_and_finished_ones_are_skipped():
    running, upcoming, finished = meeting("running", -8, 22), meeting("next", 3, 33), meeting("done", -15, -2)
    assert pick_meeting([upcoming, running], NOW).id == "running"
    assert pick_meeting([finished, upcoming], NOW).id == "next"
    assert pick_meeting([finished], NOW) is None


def test_the_most_recently_started_meeting_wins_and_far_ones_are_ignored():
    assert pick_meeting([meeting("a", -15), meeting("b", -4)], NOW).id == "b"
    assert pick_meeting([meeting("later", 30), meeting("ancient", -90)], NOW) is None
    assert pick_meeting([meeting("soon", 4), meeting("sooner", 2)], NOW).id == "sooner"
    assert pick_meeting([], NOW) is None
    assert pick_meeting([Meeting(id="no-start")], NOW) is None
