# Master Prompt: CoreDraft-BE Security Remediation & DPDPA-First Rebuild

Paste this whole document into Claude Code as your task prompt. Work through it phase by phase — do not skip ahead to Phase 1+ until Phase 0 is fully verified complete, since it's a live security exposure.

---

## Context

This is `CoreDraft-BE`, a FastAPI backend for an AI compliance copilot (RAG over compliance documents using ChromaDB + Gemini, Postgres via SQLAlchemy, JWT auth). It's a working prototype with real security issues that must be fixed before any further feature work, and an architecture that needs to evolve from "generic compliance chatbot" into a DPDPA-specific compliance product for Indian companies (with secondary SOC2/GDPR/ISO mapping for companies selling cross-border).

Repo structure you'll be working in:
```
main.py                     — FastAPI entrypoint, router registration
app/config.py                — pydantic Settings, loads .env
app/auth/auth.py             — JWT creation/validation, password hashing
app/auth/routes.py           — login/signup endpoints
app/database/models.py       — SQLAlchemy models (User, Document, Email)
app/database/schemas.py       — Pydantic schemas
app/routes/compliance.py     — document CRUD, AI generate/summarize/tag, /ask RAG endpoint
app/routes/admin.py          — user management (block/promote/reset password)
app/routes/users.py          — basic user read endpoints
app/chroma/vectorstore.py    — Chroma add/query/delete logic
app/chroma/embedder.py       — embedding generation (Gemini)
app/utils/ai_generator.py    — Gemini prompt logic incl. baseline-vs-user-doc gap comparison
app/utils/parser.py          — PDF/DOCX text extraction
app/utils/text_splitter.py   — chunking logic
.env                          — currently committed to git (see Phase 0)
chroma_db/                    — persisted vector store, currently committed to git
uploads/                      — uploaded files, currently committed to git (contains real PII)
venv/                         — committed to git, should never be
```

Work incrementally. Before making changes, read the relevant file in full. After each phase, summarize what changed and what you verified, and wait for my confirmation before starting the next phase unless I say otherwise.

---

## Phase 0 — Security remediation (do this first, treat as urgent)

**0.1 — Secret rotation (tell me what to do manually; you can't do this part yourself)**
- List every secret currently in `.env` and flag that I need to rotate each one externally: `GEMINI_API_KEY` (Google AI Studio), `SECRET_KEY` (generate a new random 64-byte value), database password.
- Do not generate new secret values yourself and put them in code — tell me to generate/rotate them externally and paste the new `.env` locally (untracked).

**0.2 — Remove secrets and unwanted files from git history**
- Add a proper `.gitignore` covering `.env`, `venv/`, `__pycache__/`, `*.pyc`, `chroma_db/`, `uploads/` (uploads should not ship in the repo at all — real user files don't belong in version control).
- Use `git filter-repo` (preferred) or BFG Repo-Cleaner to purge `.env`, `venv/`, `chroma_db/`, and `uploads/` from the *entire git history*, not just the current commit. Explain the exact commands you're running and why, since this rewrites history and requires a force-push.
- After cleaning, verify with `git log --all --full-history -- .env` (and the other paths) that they no longer appear in any commit.

**0.3 — Fix the hardcoded JWT secret**
- In `app/auth/auth.py`, remove the hardcoded `SECRET_KEY = "..."` literal entirely. Import it from `app.config.settings` instead (which already correctly loads it from `.env` via pydantic-settings) so there's a single source of truth.
- Confirm `ALGORITHM` is also sourced from config, not duplicated.

**0.4 — Enforce `blocked` status at auth time**
- In `get_current_user` (`app/auth/auth.py`), after loading the user from the DB, raise `403` if `user.blocked` is `True`. Blocking a user should immediately invalidate their access on the next request, not just at their next login.

**0.5 — Fix multi-tenant data isolation in the vector store (the most important fix)**
- Right now `query_similar_docs` / `retrieve_relevant_chunks` in `app/chroma/vectorstore.py` query the single shared `compliance_docs` Chroma collection with no tenant/user filter. Any authenticated user can retrieve any other user's document chunks via `/compliance/ask`.
- Add a `where` filter on every retrieval call scoped to the current user's `company_id` (see Phase 1 for adding that field) OR `user_id` if we're staying single-user-per-account for now, **except** for chunks tagged `is_default=True` (the shared baseline compliance docs like DPDPA text, which should remain queryable by everyone).
- Update every call site (`compliance.py` `/ask`, `/upload`, `ai_generator.py`) to pass and enforce this filter. Write a quick test that confirms user A cannot retrieve user B's uploaded document content through `/ask`.

**0.6 — Fix password reset**
- `admin.reset_password` currently sets `hashed_password = "hashed-temp"`, a literal string, not an actual bcrypt hash, so no one can ever log in with it.
- Generate a real random temporary password, hash it properly with the existing `get_password_hash`, store the hash, and return the plaintext temp password in the response **only to the admin who triggered it** (not logged, not emailed unless we build that separately) so it can be manually relayed.
- Flag this as a stopgap — proper email-based reset-token flow is Phase 4.

**0.7 — CORS cleanup**
- `main.py` has `"https://*.vercel.app"` in the CORS `origins` list — FastAPI's `CORSMiddleware` does not support wildcard subdomains as a literal string match, so this entry does nothing. Either list explicit deployed domains or use `allow_origin_regex` with a proper regex if wildcard subdomain support is actually needed.

**Acceptance criteria for Phase 0:** no secrets in git history, no hardcoded secrets in source, blocked users are rejected immediately, a cross-user retrieval test proves isolation, password reset actually produces a working credential, CORS config does what it claims to do.

---

## Phase 1 — Data model foundation for structured compliance (not just chat)

The current schema (`User`, `Document`, `Email`) can only support "upload a doc, chat about it." To support real compliance workflows we need persisted, queryable compliance objects instead of one-off LLM responses.

**1.1 — Add `company_id` tenant scoping**
- Add a `Company` model (id, name, plan tier).
- Add `company_id` FK to `User` and `Document`. Every query in every route must filter by the current user's `company_id`, not just `user_id`, so we're ready for multi-seat company accounts, not just single users.

**1.2 — Add structured compliance tables**
- `Framework` (id, name — e.g. "DPDPA", "SOC2", "GDPR")
- `Control` (id, framework_id FK, reference code, description) — seed this with a real DPDPA control set (consent requirements, breach notification, SDF obligations, data principal rights, cross-border transfer rules) as the first framework, since that's our wedge
- `Obligation` (id, control_id FK, company_id FK, status: not_started/in_progress/met/gap, evidence_doc_id FK nullable)
- `GapFinding` (id, obligation_id FK, description, severity, created_at, resolved_at nullable) — this should be *populated by* the existing Gemini baseline-vs-user-doc comparison logic in `ai_generator.py`, turning its output into persisted rows instead of a throwaway chat response
- `AuditLog` (id, company_id FK, actor_user_id FK, action, target_type, target_id, timestamp) — log every create/update/delete on Document, Obligation, GapFinding, and every admin action (block/promote/reset)

**1.3 — Wire the existing gap-analysis feature into these tables**
- Modify `process_question_and_docs` / `generate_answer_from_context` so that when it's used for a formal "run compliance check" action (as opposed to a casual chat question), it writes structured `GapFinding` rows tied to specific `Obligation`/`Control` records, with the LLM's cited source chunks stored as evidence — not just returned as a chat string that disappears after the response.

**Acceptance criteria for Phase 1:** a company's obligations, gaps, and audit trail persist and can be queried/filtered independent of any single chat exchange; multi-company data is provably isolated at the schema level, not just the query level.

---

## Phase 2 — DPDPA-specific workflow features (the actual differentiator)

Build these as real workflows with persisted state, not chatbot answers:

**2.1 — Significant Data Fiduciary (SDF) self-assessment**
- A short questionnaire (volume of data processed, categories like children's data, cross-border transfers, etc.) that produces a stored classification (`is_likely_sdf: bool`, with reasoning) on the `Company` record, surfaced on a dashboard.

**2.2 — Breach notification workflow**
- New `BreachIncident` model: reported_at, description, severity, `notification_deadline` (auto-calculated as reported_at + 72 hours — DPDPA requires notification for all breaches regardless of severity, unlike GDPR's risk threshold), `dpb_notified_at` nullable, `status`.
- On creation, auto-calculate and store the deadline; add an endpoint/dashboard view that shows time remaining and flags overdue incidents.
- Generate a draft DPB notification document (reuse the existing Gemini generation pattern) pre-filled with incident details.

**2.3 — Consent record data model (build now, integrate later)**
- `ConsentRecord` model: data_principal_reference, purpose, granted_at, withdrawn_at nullable, notice_version. Not wired to an external Consent Manager yet (that framework isn't live until Nov 2026), but structured so it's ready to integrate via API once it is.

**Acceptance criteria for Phase 2:** these are real database-backed workflows with deadlines, statuses, and dashboard visibility — not one-off prompts to Gemini.

---

## Phase 3 — Deliver on the README's existing claims, for real

**3.1 — Compliance checklists**
- Generate a checklist as persisted `Obligation`/`Control` rows scoped to a chosen framework (or combination — DPDPA + SOC2 + GDPR), assignable to a user, with a completion status — not a markdown list returned once by an LLM call and never stored.

**3.2 — Risk dashboard**
- A real aggregation endpoint: count of open `GapFinding`s by severity, overdue `BreachIncident`s, obligation completion percentage by framework, per company.

**3.3 — Integrations (lowest priority — build only after 1-3 above are solid)**
- Slack/Jira webhook on new `GapFinding` or overdue `BreachIncident` creation, since these are the two most time-sensitive events.

---

## Phase 4 — Productionization

- Replace hardcoded `MAX_USER_DOCS = 3` with a `Company.plan` field driving actual limits.
- Real email-based password reset flow (token + expiry, not admin-set temp password).
- Review whether a single-file Chroma `PersistentClient` per process is sufficient for expected load, or whether a dedicated vector DB service is needed as usage grows.
- Add basic rate limiting on `/compliance/ask` and `/upload` given they call paid LLM APIs.

---

## Working agreement

- Do not commit any `.env`, credentials, or generated secrets at any point.
- Write or update tests for anything touching auth, tenant isolation, or the breach-deadline calculation — these are the three places a bug is either a security incident or a compliance failure, not just a UX bug.
- After each phase, give me a short summary of exactly what changed, what you verified, and any open questions, before moving to the next phase.