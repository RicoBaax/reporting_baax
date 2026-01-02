import os
import sys
import pandas as pd
sys.path.insert(0, os.getcwd())

from src.ui.tabs.prospection import _prepare_activities_prospection, _prepare_appointments_prospection, _to_utc
from src.processing.filters import SidebarFilters


def test_prepare_appointments_parsing_and_timestamps():
    appts = pd.DataFrame([
        {"contactId": "X", "hs_createdate": "2025-07-01T09:00:00Z", "hs_appointment_start_time": "2025-07-10T10:00:00Z"},
        {"contactId": "Y", "hs_createdate": "2025-07-02T09:00:00Z", "hs_appointment_start_time": ""},
    ])

    ap = _prepare_appointments_prospection(appts, pd.DataFrame([{"contactId": "X"}, {"contactId": "Y"}]), None)

    assert "created_at" in ap.columns and "start_at" in ap.columns
    assert ap.loc[ap["contactId"] == "X", "created_at"].iloc[0].tz is not None
    assert ap.loc[ap["contactId"] == "X", "start_at"].iloc[0].tz is not None


def test_prepare_activities_filters_scope_and_ts():
    acts = pd.DataFrame([
        {"contactId": "1", "ts": "2025-08-01T09:00:00Z", "activity_type": "EMAIL", "hs_email_direction": "EMAIL", "hs_email_status": "SENT"},
        {"contactId": "2", "ts": "2025-08-02T09:00:00Z", "activity_type": "CALL", "hs_call_direction": "OUTBOUND"},
    ])

    contacts_scope = pd.DataFrame([{"contactId": "1"}])

    acts_p = _prepare_activities_prospection(acts, contacts_scope, None, [])
    # only contact 1 should remain due to scope filter
    assert set(acts_p["contactId"]) == {"1"}
    assert "ts" in acts_p.columns
    assert acts_p["ts"].dtype.kind in ("M",)  # datetime dtype
