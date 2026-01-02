import os
import sys
import pandas as pd
sys.path.insert(0, os.getcwd())

from src.ui.tabs.prospection import _channel_flags, _compute_channel_stats


def test_phone_response_rate_computed_as_connected_disposition():
    GUID = "f240bbac-87c9-4f6e-bf70-924b57d47db7"
    acts = pd.DataFrame([
        # contact1: outbound action with connected disposition -> contacted + responded
        {"contactId": "1", "activity_type": "CALL", "hs_call_direction": "OUTBOUND", "hs_call_disposition": GUID, "hs_call_status": "" , "ts": "2025-12-01T10:00:00Z"},
        # contact2: outbound action with COMPLETED but no disposition -> contacted, NOT responded
        {"contactId": "2", "activity_type": "CALL", "hs_call_direction": "OUTBOUND", "hs_call_disposition": "", "hs_call_status": "COMPLETED", "ts": "2025-12-01T11:00:00Z"},
        # contact3: outbound action with other disposition -> contacted, NOT responded
        {"contactId": "3", "activity_type": "CALL", "hs_call_direction": "OUTBOUND", "hs_call_disposition": "other", "hs_call_status": "COMPLETED", "ts": "2025-12-01T12:00:00Z"},
        # contact4: inbound call with connected disposition -> responded but not contacted
        {"contactId": "4", "activity_type": "CALL", "hs_call_direction": "INBOUND", "hs_call_disposition": GUID, "hs_call_status": "", "ts": "2025-12-01T13:00:00Z"},
    ])

    # Ensure ts column is parsed as timezone-aware datetime (UTC)
    acts["ts"] = pd.to_datetime(acts["ts"], utc=True)

    flags = _channel_flags(acts)
    stats = _compute_channel_stats(acts, pd.DataFrame(), channel="Téléphone", action_mask=flags["call_action"], response_mask=flags["call_response"], engaged_channel_value="Téléphone")

    # contacted contacts = contacts with outbound attempts (1,2,3) -> 3
    assert stats.prospects_contacted == 3
    # responses = calls with connected disposition (1 and 4) -> 2
    assert stats.responses == 2
    # prospects_responded = intersection of responded & contacted -> only contact 1 -> 1
    assert stats.prospects_responded == 1
    # response_rate = 1/3
    assert abs(stats.response_rate - (1.0 / 3.0)) < 1e-9
