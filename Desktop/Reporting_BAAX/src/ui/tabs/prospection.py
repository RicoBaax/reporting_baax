from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.lexique.lexique import load_lexique, get_email_auto_reply_keywords
from src.processing.filters import DateRange, SidebarFilters, filter_contacts
from src.settings.rules_store import RulesStore


# -----------------
# Small utilities
# -----------------

def _read_csv(path: Path) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        # fallback engine for some edge CSV
        try:
            df = pd.read_csv(path, dtype=str, engine="python", keep_default_na=False)
        except Exception:
            return pd.DataFrame()
    if df is None:
        return pd.DataFrame()
    return df


def _to_utc(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.to_datetime(series)
    return pd.to_datetime(series, utc=True, errors="coerce")


def _sanitize_account_slug(s: str) -> str:
    """Mirror DataManager slug sanitization so paths are stable."""
    s = str(s or "account").strip().lower()
    s = re.sub(r"[\\/:*?\"<>|]+", "-", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_.-]", "", s)
    return s or "account"


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def _pct(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return float(n) / float(d)


def _hours(delta: pd.Timedelta) -> Optional[float]:
    if delta is None or pd.isna(delta):
        return None
    try:
        return float(delta.total_seconds()) / 3600.0
    except Exception:
        return None


def _days(delta: pd.Timedelta) -> Optional[float]:
    if delta is None or pd.isna(delta):
        return None
    try:
        return float(delta.total_seconds()) / 86400.0
    except Exception:
        return None


def _contains_any_series(s: pd.Series, keywords: List[str]) -> pd.Series:
    if s is None:
        return pd.Series(dtype=bool)
    if len(s) == 0:
        return pd.Series(False, index=s.index, dtype=bool)
    if not keywords:
        return pd.Series(False, index=s.index)

    # vectorized "contains any" by building a regex union
    parts = [re.escape(str(k).strip()) for k in keywords if str(k).strip()]
    if not parts:
        return pd.Series(False, index=s.index)

    pat = "|".join(parts)
    return s.astype(str).str.lower().str.contains(pat.lower(), regex=True, na=False)


def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


@dataclass
class ChannelStats:
    channel: str
    actions: int
    prospects_contacted: int
    responses: int
    prospects_responded: int
    response_rate: float
    avg_delay_contact_to_response_days: Optional[float]
    engaged: int
    engagement_rate: float
    avg_actions_to_engage: Optional[float]
    avg_delay_contact_to_engage_days: Optional[float]


# -----------------
# Prospection scope logic
# -----------------


def _prepare_contacts_scope(
    contacts_scope: pd.DataFrame,
    contact_states_scope: pd.DataFrame,
) -> pd.DataFrame:
    """Join base contact info with contact_states fields used by the tab."""

    if contacts_scope is None or contacts_scope.empty:
        return pd.DataFrame()

    out = contacts_scope.copy()
    out["contactId"] = out.get("contactId", "").astype(str)

    # Join state fields if present
    if contact_states_scope is not None and not contact_states_scope.empty and "contactId" in contact_states_scope.columns:
        stc = contact_states_scope.copy()
        stc["contactId"] = stc["contactId"].astype(str)

        # keep only needed columns (tolerate missing)
        keep = [
            "contactId",
            "state",
            "first_contacted_at",
            "first_engaged_at",
            "first_engaged_channel",
            "first_met_at",
            "reached_contacted",
            "reached_engaged",
            "reached_met",
        ]
        keep = [c for c in keep if c in stc.columns]
        stc = stc[keep]

        out = out.merge(stc, on="contactId", how="left")

    # minimal defaults
    for col, default in [
        ("state", ""),
        ("first_contacted_at", ""),
        ("first_engaged_at", ""),
        ("first_engaged_channel", ""),
        ("first_met_at", ""),
        ("reached_contacted", ""),
        ("reached_engaged", ""),
        ("reached_met", ""),
    ]:
        if col not in out.columns:
            out[col] = default

    return out


def _prepare_activities_prospection(
    acts: pd.DataFrame,
    contacts_scope: pd.DataFrame,
    filters: SidebarFilters,
    auto_keywords: List[str],
) -> pd.DataFrame:
    """Filter activities to the prospecting window:

    - Only contacts in scope (after scenario/owner filters)
    - Only before the first RDV realized (first_met_at), if known
    - Apply date filter on activity timestamp

    Notes:
    - We include activities even if the contact is now "rencontré", as long as they are before first_met_at.
    """

    if acts is None or acts.empty:
        return pd.DataFrame()

    m = acts.copy()

    # normalize identifiers
    if "contactId" not in m.columns:
        return pd.DataFrame()
    m["contactId"] = m["contactId"].astype(str)

    # filter contact scope
    scope_ids = set(contacts_scope["contactId"].astype(str)) if contacts_scope is not None and not contacts_scope.empty else set()
    if scope_ids:
        m = m[m["contactId"].isin(scope_ids)].copy()

    # timestamps
    ts_col = _pick_first_existing(m, ["ts", "hs_timestamp", "hs_createdate", "createdate"]) or "ts"
    if ts_col not in m.columns:
        m[ts_col] = ""

    # primary parse: prefer explicit timestamp column (hs_timestamp), but fall back to hs_createdate/createdate
    m["ts"] = _to_utc(m[ts_col])
    if m["ts"].isna().all():
        # if primary column entirely empty, try hs_createdate / createdate
        if "hs_createdate" in m.columns:
            m["ts"] = _to_utc(m["hs_createdate"])
        elif "createdate" in m.columns:
            m["ts"] = _to_utc(m["createdate"])
    else:
        # fill row-wise missing timestamps from hs_createdate and createdate when available
        if "hs_createdate" in m.columns:
            m["ts"] = m["ts"].fillna(_to_utc(m["hs_createdate"]))
        if "createdate" in m.columns:
            m["ts"] = m["ts"].fillna(_to_utc(m["createdate"]))

    # apply date filter on activities
    if filters is not None and filters.date_range is not None and "ts" in m.columns:
        dr = filters.date_range
        if isinstance(dr, DateRange):
            start = getattr(dr, "start_utc", None) or getattr(dr, "start", None)
            end = getattr(dr, "end_utc", None) or getattr(dr, "end", None)
            if start is not None and end is not None:
                # inclusive end to match other filters
                m = m[(m["ts"] >= start) & (m["ts"] <= end)].copy()

    # join first_met_at (first RDV realized)
    first_met = None
    if contacts_scope is not None and not contacts_scope.empty and "first_met_at" in contacts_scope.columns:
        first_met = contacts_scope[["contactId", "first_met_at"]].copy()
        first_met["contactId"] = first_met["contactId"].astype(str)
        first_met["first_met_at"] = _to_utc(first_met["first_met_at"])

    if first_met is not None:
        m = m.merge(first_met, on="contactId", how="left")
        # keep activities strictly before first_met_at if first_met_at exists
        mask = m["first_met_at"].isna() | (m["ts"] < m["first_met_at"])
        m = m[mask].copy()

    # auto-replies flag for incoming emails (subject/text)
    subj_col = _pick_first_existing(m, ["hs_email_subject", "email_subject", "subject"]) or "hs_email_subject"
    body_col = _pick_first_existing(m, ["hs_email_text", "email_text", "body", "hs_body"]) or "hs_email_text"

    subj = m[subj_col] if subj_col in m.columns else pd.Series("", index=m.index)
    body = m[body_col] if body_col in m.columns else pd.Series("", index=m.index)
    is_auto = _contains_any_series(subj, auto_keywords) | _contains_any_series(body, auto_keywords)
    m["is_auto_reply"] = is_auto.astype(bool)

    # normalize types
    if "activity_type" not in m.columns and "type" in m.columns:
        m["activity_type"] = m["type"]
    if "activity_type" not in m.columns:
        m["activity_type"] = ""

    return m


def _prepare_appointments_prospection(
    appt_bc: pd.DataFrame,
    contacts_scope: pd.DataFrame,
    filters: SidebarFilters,
) -> pd.DataFrame:
    """Appointments: filter to contact scope and by date range.

    We use creation time for the "RDV pris" counts (hs_createdate).
    """

    if appt_bc is None or appt_bc.empty:
        return pd.DataFrame()

    m = appt_bc.copy()
    if "contactId" not in m.columns:
        return pd.DataFrame()
    m["contactId"] = m["contactId"].astype(str)

    scope_ids = set(contacts_scope["contactId"].astype(str)) if contacts_scope is not None and not contacts_scope.empty else set()
    if scope_ids:
        m = m[m["contactId"].isin(scope_ids)].copy()

    created_col = _pick_first_existing(m, ["hs_createdate", "createdate", "ts_created"]) or "hs_createdate"
    start_col = _pick_first_existing(m, ["hs_appointment_start_time", "start_time", "hs_start_time"]) or "hs_appointment_start_time"

    # ensure raw source columns exist so downstream code can rely on them
    if created_col not in m.columns:
        m[created_col] = ""
    if start_col not in m.columns:
        m[start_col] = ""

    # Normalize to standard timezone-aware timestamp columns used across the module
    m["created_at"] = pd.to_datetime(m[created_col], errors="coerce", utc=True)
    m["start_at"] = pd.to_datetime(m[start_col], errors="coerce", utc=True)

    # apply date filter on RDV pris timestamp (created_at)
    if filters is not None and getattr(filters, "date_range", None) is not None and "created_at" in m.columns:
        dr = filters.date_range

        if isinstance(dr, DateRange):
            start = getattr(dr, "start_utc", None) or getattr(dr, "start", None)
            end = getattr(dr, "end_utc", None) or getattr(dr, "end", None)

            if start is not None and end is not None:
                start_ts = pd.Timestamp(start)
                end_ts = pd.Timestamp(end)
                if start_ts.tz is None:
                    start_ts = start_ts.tz_localize("UTC")
                else:
                    start_ts = start_ts.tz_convert("UTC")
                if end_ts.tz is None:
                    end_ts = end_ts.tz_localize("UTC")
                else:
                    end_ts = end_ts.tz_convert("UTC")

                m = m[(m["created_at"].notna()) & (m["created_at"] >= start_ts) & (m["created_at"] < end_ts)].copy()

    return m


# -----------------
# KPI computations
# -----------------


def _channel_flags(acts: pd.DataFrame) -> Dict[str, pd.Series]:
    """Return boolean masks for action/response by channel."""

    if acts is None or acts.empty:
        return {
            "call_action": pd.Series([], dtype=bool),
            "call_response": pd.Series([], dtype=bool),
            "email_action": pd.Series([], dtype=bool),
            "email_response": pd.Series([], dtype=bool),
            "email_response_clean": pd.Series([], dtype=bool),
            "li_action": pd.Series([], dtype=bool),
            "li_response": pd.Series([], dtype=bool),
        }

    def _col(name: str, default):
        """
        Safe column accessor:
        - returns acts[name] if it exists
        - otherwise returns a Series filled with `default` (same index as acts)
        """
        if name in acts.columns:
            return acts[name]
        return pd.Series([default] * len(acts), index=acts.index)

    t = _col("activity_type", "").astype(str)

    # CALL
    is_call = t.eq("CALL")
    call_dir = _col("hs_call_direction", "").astype(str)
    call_status = _col("hs_call_status", "").astype(str)
    call_disp = _col("hs_call_disposition", "").astype(str)
    CONNECTED_DISPOSITION_ID = "f240bbac-87c9-4f6e-bf70-924b57d47db7"
    call_action = is_call & call_dir.eq("OUTBOUND")
    # Connected call response: only consider calls whose disposition matches the connected GUID.
    # This follows the rule: volume of responses = hs_call_disposition == CONNECTED_DISPOSITION_ID
    call_response = is_call & (call_disp.str.lower() == CONNECTED_DISPOSITION_ID.lower())

    # EMAIL
    is_email = t.eq("EMAIL")
    email_dir = _col("hs_email_direction", "").astype(str)
    email_status = _col("hs_email_status", "").astype(str)
    email_action = is_email & email_dir.eq("EMAIL") & email_status.eq("SENT")
    email_response = is_email & email_dir.eq("INCOMING_EMAIL")
    is_auto_reply = _col("is_auto_reply", False).astype(bool)
    email_response_clean = email_response & (~is_auto_reply)

    # LINKEDIN
    is_comm = t.eq("COMMUNICATION")
    comm_channel = _col("hs_communication_channel_type", "").astype(str)
    comm_status = _col("hs_communication_status", "").astype(str)
    is_li = is_comm & comm_channel.eq("LINKEDIN_MESSAGE")
    li_action = is_li & comm_status.eq("SENT")
    li_response = is_li & comm_status.eq("RECEIVED")

    return {
        "call_action": call_action,
        "call_response": call_response,
        "email_action": email_action,
        "email_response": email_response,
        "email_response_clean": email_response_clean,
        "li_action": li_action,
        "li_response": li_response,
    }


def _compute_channel_stats(
    acts: pd.DataFrame,
    contacts_scope: pd.DataFrame,
    *,
    channel: str,
    action_mask: pd.Series,
    response_mask: pd.Series,
    engaged_channel_value: str,
) -> ChannelStats:

    if acts is None or acts.empty:
        return ChannelStats(
            channel=channel,
            actions=0,
            prospects_contacted=0,
            responses=0,
            prospects_responded=0,
            response_rate=0.0,
            avg_delay_contact_to_response_days=None,
            engaged=0,
            engagement_rate=0.0,
            avg_actions_to_engage=None,
            avg_delay_contact_to_engage_days=None,
        )

    m = acts.copy()
    m["is_action"] = action_mask.reindex(m.index).fillna(False).astype(bool)
    m["is_response"] = response_mask.reindex(m.index).fillna(False).astype(bool)

    # totals
    actions = int(m["is_action"].sum())
    responses = int(m["is_response"].sum())

    # prospects contacted / responded
    contacted_ids = set(m.loc[m["is_action"], "contactId"].astype(str))
    responded_ids = set(m.loc[m["is_response"], "contactId"].astype(str))

    prospects_contacted = len(contacted_ids)
    prospects_responded = len(responded_ids & contacted_ids)  # responded among contacted

    response_rate = _pct(prospects_responded, prospects_contacted)

    # delays: first action -> first response per contact
    delays_resp: List[float] = []
    if prospects_responded > 0:
        # first action per contact
        first_action = (
            m.loc[m["is_action"], ["contactId", "ts"]]
            .dropna(subset=["ts"])
            .sort_values(["contactId", "ts"])
            .groupby("contactId", as_index=False)
            .first()
            .rename(columns={"ts": "first_action_ts"})
        )
        # first response per contact
        first_resp = (
            m.loc[m["is_response"], ["contactId", "ts"]]
            .dropna(subset=["ts"])
            .sort_values(["contactId", "ts"])
            .groupby("contactId", as_index=False)
            .first()
            .rename(columns={"ts": "first_resp_ts"})
        )
        tmp = first_action.merge(first_resp, on="contactId", how="inner")
        tmp["delta"] = tmp["first_resp_ts"] - tmp["first_action_ts"]
        for d in tmp["delta"].tolist():
            h = _days(d)
            if h is not None and h >= 0:
                delays_resp.append(h)

    # average delay in days
    avg_delay_contact_to_response_days = float(pd.Series(delays_resp).mean()) if delays_resp else None

    # engagement: use contact_states first_engaged_channel
    engaged = 0
    avg_actions_to_engage = None
    avg_delay_contact_to_engage_days = None

    if contacts_scope is not None and not contacts_scope.empty:
        cs = contacts_scope.copy()
        cs["contactId"] = cs["contactId"].astype(str)
        # engaged via channel
        engaged_ids = set(cs.loc[cs["first_engaged_channel"].astype(str).eq(engaged_channel_value), "contactId"].astype(str))
        engaged = len(engaged_ids)
        engagement_rate = _pct(engaged, prospects_contacted)

        # actions per prospect to engage (within this channel)
        if engaged > 0:
            # count channel actions before engage timestamp
            eng_ts = cs.loc[cs["contactId"].isin(engaged_ids), ["contactId", "first_engaged_at"]].copy()
            eng_ts["first_engaged_at"] = _to_utc(eng_ts["first_engaged_at"])

            mm = m[m["is_action"]].copy()
            mm = mm.merge(eng_ts, on="contactId", how="inner")
            mm = mm[(mm["first_engaged_at"].notna()) & (mm["ts"].notna()) & (mm["ts"] <= mm["first_engaged_at"])].copy()
            counts = mm.groupby("contactId").size()
            if not counts.empty:
                avg_actions_to_engage = float(counts.mean())

            # delay first action -> engaged
            first_action = (
                m.loc[m["is_action"], ["contactId", "ts"]]
                .dropna(subset=["ts"])
                .sort_values(["contactId", "ts"])
                .groupby("contactId", as_index=False)
                .first()
                .rename(columns={"ts": "first_action_ts"})
            )
            tmp = first_action.merge(eng_ts, on="contactId", how="inner")
            tmp["first_engaged_at"] = _to_utc(tmp["first_engaged_at"])
            tmp["delta"] = tmp["first_engaged_at"] - tmp["first_action_ts"]
            vals = []
            for d in tmp["delta"].tolist():
                h = _days(d)
                if h is not None and h >= 0:
                    vals.append(h)
            avg_delay_contact_to_engage_days = float(pd.Series(vals).mean()) if vals else None

    else:
        engagement_rate = 0.0

    return ChannelStats(
        channel=channel,
        actions=actions,
        prospects_contacted=prospects_contacted,
        responses=responses,
        prospects_responded=prospects_responded,
        response_rate=response_rate,
        avg_delay_contact_to_response_days=avg_delay_contact_to_response_days,
        engaged=engaged,
        engagement_rate=engagement_rate,
        avg_actions_to_engage=avg_actions_to_engage,
        avg_delay_contact_to_engage_days=avg_delay_contact_to_engage_days,
    )


# -----------------
# UI helpers
# -----------------


def _kpi_tooltip(text: str) -> None:
    # Streamlit doesn't have native tooltip on metric; we use caption under.
    st.caption(text)


def _download_csv_button(df: pd.DataFrame, *, label: str, filename: str) -> None:
    if df is None:
        df = pd.DataFrame()
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")


# -----------------
# Main tab
# -----------------


def render_prospection(
    *,
    base_dir: str,
    account_slug: str,
    filters: SidebarFilters,
) -> None:
    """Render Prospection tab."""

    account_slug = _sanitize_account_slug(account_slug)
    account_path = Path(base_dir) / account_slug
    mart_dir = account_path / "mart"

    st.subheader("Prospection")

    # --- Load data
    contacts_enriched = _read_csv(mart_dir / "contacts_enriched.csv")
    contact_states = _read_csv(mart_dir / "contact_states.csv")
    acts = _read_csv(mart_dir / "activities_by_contact.csv")
    appt_bc = _read_csv(mart_dir / "appointments_by_contact.csv")
    qualified = _read_csv(mart_dir / "qualified_appointments.csv")

    if contacts_enriched.empty:
        st.info("Aucune donnée dans mart/contacts_enriched.csv. Lance un Rebuild MART.")
        return

    if contact_states.empty:
        st.info("Aucun fichier mart/contact_states.csv (ou vide). Lance un Rebuild MART.")
        # we still can show partial KPIs without states, but most will be empty.

    # --- Apply sidebar filters on contacts
    contacts_scope = filter_contacts(
        contacts_enriched,
        owner_ids=filters.owner_ids,
        scenarios=filters.scenarios,
        scenario_column="scenario",
    )

    if contacts_scope.empty:
        st.warning("Aucun contact ne correspond aux filtres sélectionnés.")
        return

    # contact_states scope
    if not contact_states.empty and "contactId" in contact_states.columns:
        contact_states["contactId"] = contact_states["contactId"].astype(str)
        contact_states_scope = contact_states[
            contact_states["contactId"].isin(set(contacts_scope["contactId"].astype(str)))
        ].copy()
    else:
        contact_states_scope = pd.DataFrame(
            columns=[
                "contactId",
                "state",
                "first_contacted_at",
                "first_engaged_at",
                "first_engaged_channel",
                "first_met_at",
                "reached_contacted",
                "reached_engaged",
                "reached_met",
            ]
        )

    contacts_scope2 = _prepare_contacts_scope(contacts_scope, contact_states_scope)

    # lexique
    lex = load_lexique(base_dir=base_dir, account_slug=account_slug, init_if_missing=True)
    auto_keywords = get_email_auto_reply_keywords(lex)

    # rules store (RDV stage mapping)
    store = RulesStore(base_dir=base_dir, account_slug=account_slug)
    appt_map = store.load_appointments_stage_mapping()
    map_dict = appt_map.mapping if getattr(appt_map, "mapping", None) is not None else {}

    # prepare activities for prospection
    acts_prosp = _prepare_activities_prospection(
        acts,
        contacts_scope2,
        filters,
        auto_keywords,
    )

    # compute channel flags
    flags = _channel_flags(acts_prosp)

    # Debug: diagnostic email responses (affiché dans un expander, utile pour troubleshooting)
    with st.expander("Debug - Emails entrants et auto-réponses", expanded=False):
        if acts_prosp is None or acts_prosp.empty:
            st.write("Aucune activité à analyser pour la prospection.")
        else:
            # Masks
            email_resp_mask = flags["email_response"].reindex(acts_prosp.index).fillna(False).astype(bool)
            email_resp_clean_mask = flags["email_response_clean"].reindex(acts_prosp.index).fillna(False).astype(bool)

            total_incoming = int(email_resp_mask.sum())
            total_incoming_clean = int(email_resp_clean_mask.sum())
            total_auto = int((email_resp_mask & (~email_resp_clean_mask)).sum())

            # contacts
            contacts_incoming = set(acts_prosp.loc[email_resp_mask & acts_prosp.index.notna(), "contactId"].astype(str)) if "contactId" in acts_prosp.columns else set()
            contacts_incoming_clean = set(acts_prosp.loc[email_resp_clean_mask & acts_prosp.index.notna(), "contactId"].astype(str)) if "contactId" in acts_prosp.columns else set()

            st.write(f"Total entrantes (raw): {total_incoming}")
            st.write(f"Auto-réponses détectées: {total_auto}")
            st.write(f"Total entrantes (hors auto): {total_incoming_clean}")
            st.write(f"Contacts ayant répondu (hors auto): {len(contacts_incoming_clean)}")

            # sample rows (non-auto incoming)
            if total_incoming_clean > 0:
                sample_cols = [c for c in ["activity_type", "hs_email_direction", "hs_email_status", "hs_email_subject", "contactId", "ts"] if c in acts_prosp.columns]
                sample = acts_prosp[email_resp_clean_mask][sample_cols].head(10).copy()
                st.dataframe(sample, use_container_width=True)
                _download_csv_button(sample, label="Télécharger réponses email (ex.)", filename="email_responses_sample.csv")
            else:
                st.info("Aucune réponse email non-auto détectée pour les filtres actuels.")

            # show active filters for context
            dr = getattr(filters, "date_range", None)
            dr_str = f"{getattr(dr,'start',None)} → {getattr(dr,'end',None)}" if dr is not None else "Aucun"
            st.write(f"Plage de dates: {dr_str}")
            st.write(f"Owners: {getattr(filters, 'owner_ids', None)}")
            st.write(f"Scénarios: {getattr(filters, 'scenarios', None)}")

    # channel filter for section 1 only
    channel_choice = st.selectbox(
        "Canal (section Efforts)",
        options=["Tous", "Téléphone", "Email", "LinkedIn"],
        index=0,
        help="Ce filtre n'impacte que la section 'Analyse des efforts'.",
    )

    # --- Section 1: Analyse des efforts
    st.markdown("### 1) Analyse des efforts")

    # compute per channel stats
    call_stats = _compute_channel_stats(
        acts_prosp,
        contacts_scope2,
        channel="Téléphone",
        action_mask=flags["call_action"],
        response_mask=flags["call_response"],
        engaged_channel_value="Téléphone",
    )

    email_stats = _compute_channel_stats(
        acts_prosp,
        contacts_scope2,
        channel="Email",
        action_mask=flags["email_action"],
        response_mask=flags["email_response_clean"],
        engaged_channel_value="Email",
    )

    li_stats = _compute_channel_stats(
        acts_prosp,
        contacts_scope2,
        channel="LinkedIn",
        action_mask=flags["li_action"],
        response_mask=flags["li_response"],
        engaged_channel_value="LinkedIn",
    )

    all_stats = [call_stats, email_stats, li_stats]

    if channel_choice == "Téléphone":
        stats_list = [call_stats]
    elif channel_choice == "Email":
        stats_list = [email_stats]
    elif channel_choice == "LinkedIn":
        stats_list = [li_stats]
    else:
        stats_list = all_stats

    # display
    cols = st.columns(2)

    with cols[0]:
        st.markdown("#### Obtenir une réponse")
        total_actions = sum(s.actions for s in stats_list)
        total_responses = sum(s.responses for s in stats_list)
        total_contacted = sum(s.prospects_contacted for s in stats_list)
        total_responded = sum(s.prospects_responded for s in stats_list)

        st.metric("Volume d'actions menées", total_actions)
        _kpi_tooltip("Nombre total d'actions sortantes (email SENT, call OUTBOUND, LinkedIn SENT).")

        st.metric("Volume de réponses", total_responses)
        _kpi_tooltip("Nombre total de réponses (call connecté, email reçu hors auto, LinkedIn reçu).")

        st.metric("Taux de réponse", f"{_pct(total_responded, total_contacted):.1%}")
        _kpi_tooltip("Prospects ayant répondu / Prospects contactés (par canal).")

        # avg delay contact->response (simple mean across channels, weighted by responders)
        delays = []
        for s in stats_list:
            if s.avg_delay_contact_to_response_days is not None and s.prospects_responded > 0:
                delays.extend([s.avg_delay_contact_to_response_days] * s.prospects_responded)
        st.metric(
            "Délai moyen 1er contact → 1ère réponse (j)",
            "-" if not delays else f"{float(pd.Series(delays).mean()):.1f}",
        )
        _kpi_tooltip("Calculé par prospect: (1ère réponse sur le canal) - (1er contact sur le canal) en jours.")

    with cols[1]:
        st.markdown("#### Engager le contact")
        total_engaged = sum(s.engaged for s in stats_list)
        st.metric("Volume de prospects engagés", total_engaged)
        _kpi_tooltip("Nombre de prospects dont le premier canal d'engagement est ce canal.")

        st.metric("Taux d'engagement", f"{_pct(total_engaged, total_contacted):.1%}")
        _kpi_tooltip("Prospects engagés via ce canal / Prospects contactés via ce canal.")

        # actions to engage
        a2e = []
        for s in stats_list:
            if s.avg_actions_to_engage is not None and s.engaged > 0:
                a2e.extend([s.avg_actions_to_engage] * s.engaged)
        st.metric(
            "Actions moyennes par prospect pour engager",
            "-" if not a2e else f"{float(pd.Series(a2e).mean()):.1f}",
        )

        d2e = []
        for s in stats_list:
            if s.avg_delay_contact_to_engage_days is not None and s.engaged > 0:
                d2e.extend([s.avg_delay_contact_to_engage_days] * s.engaged)
        st.metric(
            "Délai moyen 1er contact → engagement (j)",
            "-" if not d2e else f"{float(pd.Series(d2e).mean()):.1f}",
        )

    # --- Section 2: Détail des efforts par canal
    st.markdown("### 2) Détail des efforts par canal")

    with st.expander("Voir le tableau des prospects engagés", expanded=False):
        cs = contacts_scope2.copy()
        cs["contactId"] = cs["contactId"].astype(str)

        engaged_tbl = cs[cs["reached_engaged"].astype(str).isin(["1", "True", "true", "yes", "Y"])] if "reached_engaged" in cs.columns else cs[cs["first_engaged_at"].astype(str) != ""]

        if engaged_tbl.empty:
            st.info("Aucun prospect engagé sur la période / filtres sélectionnés.")
        else:
            # effort volume before engage (all channels)
            engaged_ids = set(engaged_tbl["contactId"].astype(str))

            # count actions before engage
            eng_ts = engaged_tbl[["contactId", "first_engaged_at"]].copy()
            eng_ts["first_engaged_at"] = _to_utc(eng_ts["first_engaged_at"])

            macts = acts_prosp.copy()
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

            # most used channel
            def _most_used_channel(df: pd.DataFrame) -> pd.Series:
                if df.empty:
                    return pd.Series(dtype=str)
                mm = df.copy()
                mm["call"] = flags2["call_action"].reindex(mm.index).fillna(False).astype(int)
                mm["email"] = flags2["email_action"].reindex(mm.index).fillna(False).astype(int)
                mm["li"] = flags2["li_action"].reindex(mm.index).fillna(False).astype(int)
                agg = mm.groupby("contactId")[["call", "email", "li"]].sum()
                # argmax per row
                def pick(r):
                    if r.max() == 0:
                        return ""
                    return {"call": "Téléphone", "email": "Email", "li": "LinkedIn"}[r.idxmax()]
                return agg.apply(pick, axis=1)

            most_used = _most_used_channel(macts).rename("most_used_channel")

            out = engaged_tbl.copy()
            # effort_counts and most_used are Series indexed by contactId — merge on the series index
            out = out.merge(effort_counts.rename("effort_before_engage"), left_on="contactId", right_index=True, how="left")
            out = out.merge(most_used.rename("most_used_channel"), left_on="contactId", right_index=True, how="left")

            # delay first_contacted_at -> first_engaged_at
            out["first_contacted_at_dt"] = _to_utc(out.get("first_contacted_at", ""))
            out["first_engaged_at_dt"] = _to_utc(out.get("first_engaged_at", ""))
            out["delay_contact_to_engage_j"] = (out["first_engaged_at_dt"] - out["first_contacted_at_dt"]).apply(_days)

            scen = out[["contactId", "scenario"]].copy() if "scenario" in out.columns else out[["contactId"]].assign(scenario="")

            tbl = out[["contactId"]].copy()
            tbl = tbl.merge(scen, on="contactId", how="left")
            tbl["effort_before_engage"] = out.get("effort_before_engage", 0).fillna(0).map(_safe_int)
            tbl["most_used_channel"] = out.get("most_used_channel", "").fillna("")
            tbl["engaged_channel"] = out.get("first_engaged_channel", "").fillna("")
            tbl["delay_contact_to_engage_j"] = out.get("delay_contact_to_engage_j", pd.NA)

            st.dataframe(tbl, use_container_width=True)
            _download_csv_button(tbl, label="Télécharger (CSV)", filename="prospects_engages.csv")

    # --- Section 3: Analyse des résultats
    st.markdown("### 3) Analyse des résultats obtenus")

    appts = _prepare_appointments_prospection(appt_bc, contacts_scope2, filters)

    # classify appointment stage to to_do / done / not_done using mapping
    stage_col = _pick_first_existing(appts, ["hs_pipeline_stage", "hs_appointment_stage", "pipeline_stage"]) or "hs_pipeline_stage"
    if stage_col not in appts.columns:
        appts[stage_col] = ""

    def _stage_category(stage_id: str) -> str:
        s = str(stage_id or "")
        return str(map_dict.get(s, ""))  # expected: to_do / done / not_done

    appts["stage_category"] = appts[stage_col].map(_stage_category)

    rdv_pris = appts["hs_object_id"].nunique() if "hs_object_id" in appts.columns else len(appts)

    # NB: pd.Timestamp.utcnow() is tz-aware in recent pandas versions -> avoid tz_localize errors
    now_utc = pd.Timestamp.now(tz="UTC")
    rdv_a_venir = int(((appts["start_at"].notna()) & (appts["start_at"] >= now_utc)).sum())

    rdv_realises = int((appts["stage_category"] == "done").sum())
    rdv_non_realises = int((appts["stage_category"] == "not_done").sum())

    rdv_qualifies = 0
    if qualified is not None and not qualified.empty:
        # qualified file uses appointmentId column
        q_col = "appointmentId" if "appointmentId" in qualified.columns else "hs_object_id"
        rdv_qualifies = int(qualified[q_col].nunique()) if q_col in qualified.columns else len(qualified)

    # --- Layout: build the columns ONCE (your previous code created 5 separate rows → diagonal display)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RDV pris", rdv_pris)
    c2.metric("RDV à venir", rdv_a_venir)
    c3.metric("RDV réalisés", rdv_realises)
    c4.metric("RDV non réalisés", rdv_non_realises)
    c5.metric("RDV qualifiés", rdv_qualifies)

    r1, r2 = st.columns(2)
    r1.metric(
        "Taux de réalisation RDV",
        f"{_pct(rdv_realises, rdv_realises + rdv_non_realises):.1%}",
    )
    r2.metric(
        "Taux de qualification RDV (sur RDV réalisés)",
        f"{_pct(rdv_qualifies, rdv_realises):.1%}",
    )

    # Delays: first contact -> first rdv created / rdv created -> rdv start
    if not appts.empty:
        # first contact per contact
        fc = contacts_scope2[["contactId", "first_contacted_at"]].copy()
        fc["first_contacted_at"] = _to_utc(fc["first_contacted_at"])

        # first rdv created per contact
        fr = (
            appts.dropna(subset=["created_at"])
            .sort_values(["contactId", "created_at"])
            .groupby("contactId", as_index=False)
            .first()[["contactId", "created_at"]]
            .rename(columns={"created_at": "first_rdv_created_at"})
        )

        tmp = fc.merge(fr, on="contactId", how="inner")
        tmp["delta"] = tmp["first_rdv_created_at"] - tmp["first_contacted_at"]
        vals = [d for d in tmp["delta"].apply(_days).tolist() if d is not None and d >= 0]
        st.metric(
            "Délai moyen 1er contact → 1er RDV pris (j)",
            "-" if not vals else f"{float(pd.Series(vals).mean()):.1f}",
        )

        # created -> start
        tmp2 = appts.dropna(subset=["created_at", "start_at"]).copy()
        tmp2["delta"] = tmp2["start_at"] - tmp2["created_at"]
        vals2 = [d for d in tmp2["delta"].apply(_days).tolist() if d is not None and d >= 0]
        st.metric(
            "Délai moyen RDV pris → RDV réalisé (j)",
            "-" if not vals2 else f"{float(pd.Series(vals2).mean()):.1f}",
        )

    # --- Section 4: Détail des résultats
    st.markdown("### 4) Détail des résultats")

    with st.expander("Voir le tableau des RDV", expanded=False):
        if appts.empty:
            st.info("Aucun rendez-vous sur la période / filtres sélectionnés.")
        else:
            # effort before RDV creation: count actions before created_at (all channels), within prospection window
            if not acts_prosp.empty:
                # keep only actions
                flags_all = _channel_flags(acts_prosp)
                acts_prosp["is_action_any"] = (
                    flags_all["call_action"].reindex(acts_prosp.index).fillna(False)
                    | flags_all["email_action"].reindex(acts_prosp.index).fillna(False)
                    | flags_all["li_action"].reindex(acts_prosp.index).fillna(False)
                )
                acts_actions = acts_prosp[acts_prosp["is_action_any"]].copy()
            else:
                acts_actions = pd.DataFrame(columns=["contactId", "ts"])

            ap = appts.copy()

            # scenario (from contacts scope)
            if "scenario" in contacts_scope2.columns:
                ap = ap.merge(contacts_scope2[["contactId", "scenario"]], on="contactId", how="left")
            else:
                ap["scenario"] = ""

            # compute effort before rdv created per appointment contact
            effort = []
            for _, row in ap.iterrows():
                cid = str(row.get("contactId", ""))
                created_at = row.get("created_at")
                if cid and pd.notna(created_at) and not acts_actions.empty:
                    cacts = acts_actions[(acts_actions["contactId"].astype(str) == cid) & (acts_actions["ts"].notna()) & (acts_actions["ts"] <= created_at)]
                    effort.append(int(len(cacts)))
                else:
                    effort.append(0)
            ap["effort_before_rdv"] = effort

            # delay first contact -> created
            fc = contacts_scope2[["contactId", "first_contacted_at"]].copy()
            fc["first_contacted_at"] = _to_utc(fc["first_contacted_at"])
            ap = ap.merge(fc, on="contactId", how="left")
            ap["delay_contact_to_rdv_h"] = (ap["created_at"] - ap["first_contacted_at"]).apply(_hours)
            ap["delay_contact_to_rdv_start_h"] = (ap["start_at"] - ap["first_contacted_at"]).apply(_hours)

            # output
            out_cols = []
            if "hs_appointment_name" in ap.columns:
                out_cols.append("hs_appointment_name")
            if "scenario" in ap.columns:
                out_cols.append("scenario")
            out_cols += ["created_at", "start_at", "stage_category", "effort_before_rdv", "delay_contact_to_rdv_h", "delay_contact_to_rdv_start_h"]

            out = ap[out_cols].copy()
            out = out.rename(
                columns={
                    "hs_appointment_name": "rdv_name",
                    "created_at": "rdv_created_at",
                    "start_at": "rdv_start_at",
                    "stage_category": "rdv_stage",
                    "effort_before_rdv": "effort_before_rdv_actions",
                    "delay_contact_to_rdv_h": "delay_first_contact_to_rdv_created_h",
                    "delay_contact_to_rdv_start_h": "delay_first_contact_to_rdv_start_h",
                }
            )

            st.dataframe(out, use_container_width=True)
            _download_csv_button(out, label="Télécharger (CSV)", filename="rdv.csv")