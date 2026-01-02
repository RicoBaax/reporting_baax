Title: UI theme, timestamp fixes, days KPIs, tests, and archival

Summary:
- Apply brand fonts and theme (Poppins, Work Sans) and adjust CSS to match charter.
- Fix KeyError due to missing `start_at` by normalizing appointment timestamps.
- Fix incoming email counts by normalizing activity timestamps (`ts`) from `hs_timestamp` → `hs_createdate` → `createdate` and making date range end inclusive.
- Replace hour-based KPIs with day-based KPIs and update labels.
- Add Streamlit debug expander for incoming emails & auto-replies.
- Add unit tests for Prospection & Engaged table; all tests pass locally.
- Archive unused files: `onboarding/`, `tools/`, `src/ui/components.py` → moved to `archive/`.

Notes:
- I couldn't open a PR automatically since GitHub CLI (`gh`) isn't installed in this environment. You can create a PR from branch `ui-theme-archive` using your usual workflow.
- Suggested reviewers: @team, @frontend
- Related issues: # (none currently)
