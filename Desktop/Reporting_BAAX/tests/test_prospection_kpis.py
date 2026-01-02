import os
import sys
import pandas as pd
sys.path.insert(0, os.getcwd())

from src.ui.tabs.prospection import _channel_flags, _compute_channel_stats


def test_compute_channel_stats_calls_and_engagement():
    GUID = "f240bbac-87c9-4f6e-bf70-924b57d47db7"
    # activities: A: outbound + connected, B: outbound only, C: inbound connected only, D: outbound + connected
    acts = pd.DataFrame([
        {"contactId": "A", "activity_type": "CALL", "hs_call_direction": "OUTBOUND", "hs_call_disposition": GUID, "ts": "2025-06-01T10:00:00Z"},
        {"contactId": "B", "activity_type": "CALL", "hs_call_direction": "OUTBOUND", "hs_call_disposition": "", "ts": "2025-06-01T11:00:00Z"},
        {"contactId": "C", "activity_type": "CALL", "hs_call_direction": "INBOUND", "hs_call_disposition": GUID, "ts": "2025-06-01T12:00:00Z"},
        {"contactId": "D", "activity_type": "CALL", "hs_call_direction": "OUTBOUND", "hs_call_disposition": GUID, "ts": "2025-06-02T09:00:00Z"},
        # additional actions for engagement timing
        {"contactId": "A", "activity_type": "CALL", "hs_call_direction": "OUTBOUND", "hs_call_disposition": "", "ts": "2025-06-01T09:00:00Z"},
    ])
    acts["ts"] = pd.to_datetime(acts["ts"], utc=True)

    # contacts scope with engagement info: A engaged via Téléphone, B engaged via Email
    cs = pd.DataFrame([
        {"contactId": "A", "first_engaged_channel": "Téléphone", "first_engaged_at": "2025-06-01T10:05:00Z"},
        {"contactId": "B", "first_engaged_channel": "Email", "first_engaged_at": "2025-06-01T12:00:00Z"},
    ])

    flags = _channel_flags(acts)
    stats = _compute_channel_stats(
        acts,
        cs,
        channel="Téléphone",
        action_mask=flags["call_action"],
        response_mask=flags["call_response"],
        engaged_channel_value="Téléphone",
    )

    assert stats.actions == 4  # 4 outbound CALL rows
    assert stats.responses == 3  # rows with connected disposition (A, C, D)
    assert stats.prospects_contacted == 3  # unique contacts with outbound actions: A,B,D
    assert stats.prospects_responded == 2  # among contacted, A and D responded
    assert abs(stats.response_rate - (2.0 / 3.0)) < 1e-9

    # engagement metrics
    assert stats.engaged == 1  # only A engaged via Téléphone
    assert stats.engagement_rate == (1.0 / 3.0)
    assert stats.avg_actions_to_engage is not None
    assert stats.avg_delay_contact_to_engage_days is not None
