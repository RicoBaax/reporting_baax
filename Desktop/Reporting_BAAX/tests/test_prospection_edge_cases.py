import os
import sys
import pandas as pd
sys.path.insert(0, os.getcwd())

from src.ui.tabs.prospection import _channel_flags, _compute_channel_stats, _to_utc


def test_channel_flags_handles_missing_columns_and_auto_reply():
    # CALL row missing hs_call_direction and hs_call_disposition -> should not be action or response
    acts = pd.DataFrame([
        {"contactId": "1", "activity_type": "CALL", "ts": "2025-01-01T10:00:00Z"},
        {"contactId": "2", "activity_type": "EMAIL", "hs_email_direction": "INCOMING_EMAIL", "is_auto_reply": True, "ts": "2025-01-01T11:00:00Z"},
        {"contactId": "3", "activity_type": "EMAIL", "hs_email_direction": "INCOMING_EMAIL", "is_auto_reply": False, "ts": "2025-01-01T12:00:00Z"},
    ])
    acts["ts"] = pd.to_datetime(acts["ts"], utc=True)

    flags = _channel_flags(acts)

    # call masks should be present and False for the incomplete call row
    assert not bool(flags["call_action"].iloc[0])
    assert not bool(flags["call_response"].iloc[0])

    # incoming email with is_auto_reply True should be filtered by email_response_clean
    assert bool(flags["email_response"].iloc[1])
    assert not bool(flags["email_response_clean"].iloc[1])
    assert bool(flags["email_response_clean"].iloc[2])


def test_compute_channel_stats_handles_empty_activities():
    empty = pd.DataFrame()
    stats = _compute_channel_stats(
        empty,
        pd.DataFrame(),
        channel="Téléphone",
        action_mask=pd.Series([], dtype=bool),
        response_mask=pd.Series([], dtype=bool),
        engaged_channel_value="Téléphone",
    )

    assert stats.actions == 0
    assert stats.responses == 0
    assert stats.prospects_contacted == 0
    assert stats.prospects_responded == 0
    assert stats.response_rate == 0.0


def test_prepare_contacts_scope_defaults_when_contact_states_missing():
    from src.ui.tabs.prospection import _prepare_contacts_scope

    contacts = pd.DataFrame([{"contactId": "x"}])
    out = _prepare_contacts_scope(contacts, pd.DataFrame())
    # defaults present
    assert "first_contacted_at" in out.columns
    assert "first_engaged_at" in out.columns
    assert out.loc[0, "first_contacted_at"] == ""
