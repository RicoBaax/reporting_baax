# Changelog

## [Unreleased]
- UI: Apply brand fonts and theme (Poppins, Work Sans, color tokens) ⚡
- Fix: Normalize appointment timestamps (`start_at`, `created_at`) to prevent KeyError in Prospection ✅
- Fix: Normalize activity timestamps (`ts`) with fallbacks (`hs_timestamp` → `hs_createdate` → `createdate`) and make date range end inclusive (<=) 🛠️
- Fix: Replace hour-based KPIs with day-based KPIs and update labels (h → j) 📈
- Fix: Use correct merge semantics when merging Series into DataFrames to avoid KeyError on `contactId` 🔧
- Add: Streamlit debug expander showing incoming emails / auto-replies for Prospection 🧪
- Add: Unit tests for Prospection and Engaged table (pytest) ✅
- Archival: Move onboarding, tools and unused components to `archive/` 📦

---

## [Previous]
- No prior releases documented.
