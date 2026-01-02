import os
import sys
import pandas as pd
import types
import builtins
sys.path.insert(0, os.getcwd())

from src.processing.filters import SidebarFilters, DateRange
import src.ui.tabs.prospection as prosp


def test_render_prospection_section2_table_capture(tmp_path):
    # capture dataframes passed to st.dataframe
    captured = []

    class StubST:
        def __init__(self):
            self._dataframes = []
            self._metrics = []

        def subheader(self, *args, **kwargs):
            pass

        def expander(self, *args, **kwargs):
            # context manager that yields self
            class Ctx:
                def __enter__(self_inner):
                    return self

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return Ctx()

        def write(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def metric(self, *args, **kwargs):
            # capture metric calls as (label, value)
            try:
                label = args[0] if len(args) > 0 else kwargs.get('label')
                value = args[1] if len(args) > 1 else kwargs.get('value')
                self._metrics.append((label, value))
            except Exception:
                self._metrics.append((None, None))


        def markdown(self, *args, **kwargs):
            pass

        def selectbox(self, *args, **kwargs):
            return "Tous"

        def columns(self, *args, **kwargs):
            # return list of stub columns where .metric is a no-op and context-manager for 'with'
            class Col:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def metric(self, *a, **k):
                    pass

            return [Col() for _ in range(args[0] if args else 2)]

        def dataframe(self, df, *args, **kwargs):
            # capture copy to avoid mutability issues
            try:
                self._dataframes.append(df.copy())
            except Exception:
                self._dataframes.append(df)

        def caption(self, *args, **kwargs):
            pass

        def download_button(self, *args, **kwargs):
            pass

    stub = StubST()

    # monkeypatch st in module
    orig_st = prosp.st
    prosp.st = stub

    try:
            # Create a minimal mart dataset in a temp account directory so we can force an engaged prospect
        account_dir = tmp_path / "acct"
        mart_dir = account_dir / "mart"
        mart_dir.mkdir(parents=True)

        # contacts_enriched
        contacts = pd.DataFrame([{"contactId": "A", "scenario": "s1"}])
        contacts.to_csv(mart_dir / "contacts_enriched.csv", index=False)

        # contact_states: A is engaged
        contact_states = pd.DataFrame([
            {"contactId": "A", "first_contacted_at": "2025-01-01T09:00:00Z", "first_engaged_at": "2025-01-02T10:00:00Z", "first_engaged_channel": "Téléphone", "reached_engaged": "1"}
        ])
        contact_states.to_csv(mart_dir / "contact_states.csv", index=False)

        # activities: A has actions before engage
        acts = pd.DataFrame([
            {"contactId": "A", "activity_type": "CALL", "hs_call_direction": "OUTBOUND", "hs_call_status": "", "ts": "2025-01-01T10:00:00Z"},
            {"contactId": "A", "activity_type": "EMAIL", "hs_email_direction": "EMAIL", "hs_email_status": "SENT", "ts": "2025-01-01T11:00:00Z"},
        ])
        acts.to_csv(mart_dir / "activities_by_contact.csv", index=False)

        # add a dummy appointment row so downstream computations have start_at
        appts = pd.DataFrame([
            {"contactId": "A", "hs_createdate": "2025-01-01T08:00:00Z", "hs_appointment_start_time": "2025-02-01T10:00:00Z", "hs_object_id": "appt-1", "hs_pipeline_stage": "done"}
        ])
        appts.to_csv(mart_dir / "appointments_by_contact.csv", index=False)
        pd.DataFrame(columns=["appointmentId", "hs_object_id"]).to_csv(mart_dir / "qualified_appointments.csv", index=False)

        base_dir = str(tmp_path)
        account_slug = "acct"

        # broad date range to include our sample
        dr = DateRange(start=pd.Timestamp("2020-01-01", tz="UTC"), end=pd.Timestamp("2030-01-01", tz="UTC"))
        filters = SidebarFilters(date_range=dr, owner_ids=None, scenarios=None)

        # call render_prospection which should use the temp mart under base_dir/account_slug
        prosp.render_prospection(base_dir=base_dir, account_slug=account_slug, filters=filters)

        # find dataframes captured that include 'effort_before_engage' column
        found = None
        for df in stub._dataframes:
            if isinstance(df, pd.DataFrame) and "effort_before_engage" in df.columns:
                found = df
                break

        assert found is not None, "Section 2 table with 'effort_before_engage' not captured"

        # basic sanity checks on columns
        assert "most_used_channel" in found.columns
        assert "engaged_channel" in found.columns
        assert "delay_contact_to_engage_j" in found.columns

        # ensure metrics captured include Volume de prospects engagés = 1
        mvals = {label: value for label, value in stub._metrics}
        assert "Volume de prospects engagés" in mvals
        assert int(mvals["Volume de prospects engagés"]) == 1

        # additional assertions for effort metrics
        assert "Volume d'actions menées" in mvals
        assert int(mvals["Volume d'actions menées"]) == 2

        assert "Volume de réponses" in mvals
        assert int(mvals["Volume de réponses"]) == 0

        assert "Taux de réponse" in mvals
        assert isinstance(mvals["Taux de réponse"], str)
        assert mvals["Taux de réponse"].startswith("0")

        assert "Délai moyen 1er contact → 1ère réponse (j)" in mvals
        # when no responses, delay metric is set to "-"
        assert mvals["Délai moyen 1er contact → 1ère réponse (j)"] == "-"

        # engagement rate: 1 engaged out of 2 contacted -> 50%
        assert "Taux d'engagement" in mvals
        assert isinstance(mvals["Taux d'engagement"], str)
        assert mvals["Taux d'engagement"].startswith("50")

    finally:
        prosp.st = orig_st
