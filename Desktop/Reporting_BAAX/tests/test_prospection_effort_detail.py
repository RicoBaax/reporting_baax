import os
import sys
import pandas as pd
sys.path.insert(0, os.getcwd())

from src.ui.tabs.prospection import _channel_flags, _to_utc


def test_effort_before_engage_and_most_used_channel():
    # contacts scope with engaged prospects
    cs = pd.DataFrame([
        {"contactId": "A", "first_engaged_at": "2025-01-02T10:00:00Z", "first_contacted_at": "2025-01-01T09:00:00Z", "first_engaged_channel": "Téléphone", "reached_engaged": "1"},
        {"contactId": "B", "first_engaged_at": "2025-01-05T12:00:00Z", "first_contacted_at": "2025-01-03T08:00:00Z", "first_engaged_channel": "Email", "reached_engaged": "1"},
        {"contactId": "C", "first_engaged_at": "", "first_contacted_at": "", "first_engaged_channel": "", "reached_engaged": "0"},
    ])

    # activities (some before engage, some after)
    acts = pd.DataFrame([
        # A: 1 call, 1 email, 1 li before engage
        {"contactId": "A", "activity_type": "CALL", "hs_call_direction": "OUTBOUND", "hs_call_status": "", "ts": "2025-01-01T10:00:00Z"},
        {"contactId": "A", "activity_type": "EMAIL", "hs_email_direction": "EMAIL", "hs_email_status": "SENT", "ts": "2025-01-01T11:00:00Z"},
        {"contactId": "A", "activity_type": "COMMUNICATION", "hs_communication_channel_type": "LINKEDIN_MESSAGE", "hs_communication_status": "SENT", "ts": "2025-01-01T12:00:00Z"},
        # A: one action after engage (should not be counted)
        {"contactId": "A", "activity_type": "CALL", "hs_call_direction": "OUTBOUND", "hs_call_status": "", "ts": "2025-01-02T12:00:00Z"},
        # B: two emails before engage
        {"contactId": "B", "activity_type": "EMAIL", "hs_email_direction": "EMAIL", "hs_email_status": "SENT", "ts": "2025-01-03T09:00:00Z"},
        {"contactId": "B", "activity_type": "EMAIL", "hs_email_direction": "EMAIL", "hs_email_status": "SENT", "ts": "2025-01-04T09:00:00Z"},
        # C: some actions but not engaged
        {"contactId": "C", "activity_type": "CALL", "hs_call_direction": "OUTBOUND", "hs_call_status": "", "ts": "2025-01-02T10:00:00Z"},
    ])

    # normalize timestamps
    acts["ts"] = pd.to_datetime(acts["ts"], utc=True)

    # engaged table mimic
    engaged_tbl = cs[cs["reached_engaged"].astype(str).isin(["1", "True", "true", "yes", "Y"]) | (cs["first_engaged_at"].astype(str) != "")]

    assert not engaged_tbl.empty

    # effort counts before engage
    eng_ts = engaged_tbl[["contactId", "first_engaged_at"]].copy()
    eng_ts["first_engaged_at"] = _to_utc(eng_ts["first_engaged_at"])

    macts = acts.copy()
    macts = macts.merge(eng_ts, on="contactId", how="inner")
    macts = macts[(macts["ts"].notna()) & (macts["first_engaged_at"].notna()) & (macts["ts"] <= macts["first_engaged_at"])].copy()

    flags2 = _channel_flags(macts)
    macts["is_action_any"] = (
        flags2["call_action"].reindex(macts.index).fillna(False)
        | flags2["email_action"].reindex(macts.index).fillna(False)
        | flags2["li_action"].reindex(macts.index).fillna(False)
    )
    macts = macts[macts["is_action_any"]].copy()

    effort_counts = macts.groupby("contactId").size().rename("effort_before_engage")

    # A should have 3, B should have 2
    assert int(effort_counts.loc["A"]) == 3
    assert int(effort_counts.loc["B"]) == 2

    # most used channel
    mm = macts.copy()
    mm["call"] = flags2["call_action"].reindex(mm.index).fillna(False).astype(int)
    mm["email"] = flags2["email_action"].reindex(mm.index).fillna(False).astype(int)
    mm["li"] = flags2["li_action"].reindex(mm.index).fillna(False).astype(int)
    agg = mm.groupby("contactId")[ ["call", "email", "li"] ].sum()

    # contact A has 1 of each -> expect Téléphone as tie breaker (call first)
    def pick(r):
        if r.max() == 0:
            return ""
        return {"call": "Téléphone", "email": "Email", "li": "LinkedIn"}[r.idxmax()]

    most = agg.apply(pick, axis=1)
    assert most.loc["A"] == "Téléphone"
    assert most.loc["B"] == "Email"

    # delay contact -> engage for A = (2025-01-02T10:00:00Z - 2025-01-01T09:00:00Z) in days
    fca = pd.to_datetime(cs.loc[cs["contactId"] == "A", "first_contacted_at"]).iloc[0]
    if fca.tzinfo is None:
        fca = pd.to_datetime(fca, utc=True)
    fea = pd.to_datetime(cs.loc[cs["contactId"] == "A", "first_engaged_at"]).iloc[0]
    if fea.tzinfo is None:
        fea = pd.to_datetime(fea, utc=True)
    delta_days = (fea - fca).total_seconds() / 86400.0
    assert delta_days == ( (pd.Timestamp("2025-01-02T10:00:00Z", tz="UTC") - pd.Timestamp("2025-01-01T09:00:00Z", tz="UTC")).total_seconds() / 86400.0 )
