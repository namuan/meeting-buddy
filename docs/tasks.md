# Improvement Tasks Checklist

A logically ordered, actionable checklist to improve Meeting Buddy across architecture, code quality, performance, testing, packaging, security, documentation, and developer experience. Each task is designed to be independently checkable and leads from foundational hygiene to advanced enhancements.

1. [ ] Establish supported Python versions consistently across the project (docs, pyproject requires-python, classifiers). Align on Python 3.10 or update to newer versions; remove conflicting classifiers (3.11–3.13) if not supported.
2. [ ] Fix Ruff configuration target-version to match supported Python version (e.g., set to "py310" if we stay on Python 3.10).
3. [ ] Correct packaging configuration: replace `py-modules = ["meeting_buddy"]` with proper package discovery (e.g., setuptools.find_packages) so the `meeting_buddy` package is installed correctly.
4. [ ] Review and correct dependency versions in pyproject: verify PyQt6, PyAudio, requests, and Whisper. Investigate the odd `openai-whisper>=20250625` spec; replace with a stable, known-good version or extras/optional dependency.
5. [ ] Add explicit development dependency management guidance (uv/venv) and update Makefile targets (install, lint, format, test, run) to reflect current tools.
6. [ ] Introduce a minimal CONTRIBUTING.md outlining environment setup, coding standards, and how to run/lint/test.
7. [ ] Create or update an architectural overview doc (docs/architecture.md) explaining MVP boundaries, key threads (recorder/transcriber/LLM), model responsibilities, and data flow.
8. [ ] Add CODEOWNERS or maintainers info to clarify review and ownership.
9. [ ] Normalize logging levels and patterns: define a logging policy (when to use info/debug/warn/error) and ensure structured logging redaction is enabled by default for sensitive data.
10. [ ] Implement a central exception hierarchy for domain errors (RecordingError, TranscriptionError, LLMServiceError, ConfigurationError) to improve error handling consistency.
11. [ ] Audit try/except blocks for broad exceptions; replace with specific exceptions and add actionable user messages in the View for recoverable errors.
12. [ ] Ensure all threads (AudioRecorderThread, AudioTranscriberThread, LLMThread) support clean start/stop semantics: idempotent stop, join with timeout, and resource cleanup in finally blocks.
13. [ ] Add cancellation/backpressure handling: verify queue size limits are respected and add drop/flush strategies with metrics when queues overflow.
14. [ ] Validate that Presenter always stops and joins threads on shutdown and on transitions (e.g., toggling recording). Add defensive guards to avoid double-starts and race conditions.
15. [ ] Break down MeetingBuddyPresenter (1500+ lines) into smaller coordinator/services:
    - [ ] RecordingCoordinator: handles device selection, recorder lifecycle, and audio chunk flow.
    - [ ] TranscriptionCoordinator: owns transcriber lifecycle and connects callbacks to the model/view.
    - [ ] LLMCoordinator: orchestrates LLMThread and LLM API requests/streaming.
    - [ ] TempFilesManager: encapsulates temp dirs/files creation/cleanup.
16. [ ] Extract Presenter-to-View update helpers into a separate UIAdapter with pure, testable methods that format UI texts and statuses.
17. [ ] Strengthen ConfigurationModel: validate values on update (timeouts, retry counts, model names); emit granular change events (per-field where useful) and debounce burst updates.
18. [ ] LLMApiService hardening: add retries with exponential backoff + jitter; distinguish client vs. server/network errors; timeouts per request; unify streaming and non-streaming error pathways.
19. [ ] Add health check on startup for LLM endpoint with clear UI indication and recovery guidance.
20. [ ] ModelDownloadService: handle missing Whisper install gracefully; display actionable install tips; support cancel reliably and verify partial downloads cleanup.
21. [ ] Transcription pipeline reliability: guard against empty or malformed audio buffers; standardize sampling rate/channels conversions; ensure Whisper model reload safety.
22. [ ] Resource management: centralize temporary directories under a predictable root; auto-clean stale temp directories on startup; add size limits and cleanup policies.
23. [ ] File path safety: use pathlib throughout; avoid relative paths where unsafe; ensure cross-platform compatibility (Windows/macOS/Linux) for audio and file operations.
24. [ ] Type hints: complete type annotations project-wide; add `from __future__ import annotations` where applicable to speed typing; ensure public APIs have precise types.
25. [ ] Docstrings: add/expand docstrings on public classes/methods; include usage examples for complex components (threads, services).
26. [ ] Introduce mypy static type checking; configure baseline with strictness gradually increased (e.g., disallow-incomplete-defs, warn-redundant-casts, warn-unused-ignores).
27. [ ] Lint and format: enforce Ruff + ruff format via pre-commit hooks; fix existing lint issues; align import ordering.
28. [ ] Unit tests foundation (pytest):
    - [ ] Models: ConfigurationModel, LLMModel, TranscriptionModel behaviors.
    - [ ] Services: LLMApiService (with requests-mock), ModelDownloadService (mock whisper, filesystem).
    - [ ] Utilities: logging_config structured formatting and redaction.
29. [ ] Concurrency tests: simulate audio/LLM queues with fakes to validate start/stop, backpressure, and cleanup semantics.
30. [ ] Integration tests (where possible): end-to-end flow using fakes/mocks for audio input and LLM endpoint; verify UI-independent logic paths (Presenter coordinators) produce expected model updates.
31. [ ] Coverage reporting: add pytest-cov, set a reasonable initial threshold, and track over time.
32. [ ] Continuous Integration: set up GitHub Actions to run lint, type-check, tests on pushes/PRs for supported OS (at least macOS and Linux for headless tests).
33. [ ] Dependency security: run `pip-audit` or `uv pip install --refresh --audit` (or GitHub Dependabot) and document process.
34. [ ] Secrets hygiene: ensure no secrets in logs; redact endpoints or tokens if introduced; provide env var-driven config with safe defaults.
35. [ ] Privacy and data handling: document how audio and transcriptions are stored/processed; provide a setting to auto-delete temp audio after processing.
36. [ ] Performance metrics: expand performance tracker coverage; log queue latencies, transcription times, LLM round-trip; add optional CSV/JSON export of metrics.
37. [ ] UI/UX improvements: status indicators for recorder/transcriber/LLM; progress bars for downloads; non-blocking error toasts; keyboard shortcuts for start/stop.
38. [ ] Accessibility: ensure color contrasts; add options for larger fonts; keyboard navigation for primary controls.
39. [ ] Internationalization readiness: define language config applicability beyond Whisper; isolate user-facing strings for future i18n.
40. [ ] Robust shutdown: handle SIGINT/SIGTERM to gracefully stop threads and save state; ensure QApplication quits cleanly.
41. [ ] Logging performance: measure overhead of structured logging; provide a flag to downgrade to simple formatter for low-resource environments.
42. [ ] Packaging/distribution: validate PyInstaller spec (main.spec); test building on CI; ensure runtime dependencies (libs, models) are handled or documented.
43. [ ] Sample data and examples: provide a small sample audio and example transcripts (if licensing permits); add example prompts for LLM analysis.
44. [ ] Documentation site refresh: link to architecture, troubleshooting, developer setup, and release notes; keep README concise and point to docs.
45. [ ] Release process: define versioning, changelog generation, and tagging; automate draft releases with artifacts (binary, wheel) in CI.
46. [ ] Error telemetry (optional/opt-in): capture anonymized error types and counts to help prioritize fixes; ensure strict opt-in and documentation.
47. [ ] Feature flags: introduce simple env or config-based flags to toggle LLM features, advanced logging, and experimental UI components.
48. [ ] Clean Makefile and tox/uv alignment: ensure commands use uv consistently and tox-uv matches CI steps; remove stale targets.
49. [ ] Ensure tests can run headless without GUI: refactor UI dependencies out of core logic, mock PyQt where necessary.
50. [ ] Add smoke test CLI entry to verify core stack without GUI (e.g., short audio simulation -> transcription -> LLM mock response).
