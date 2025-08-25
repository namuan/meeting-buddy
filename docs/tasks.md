# Improvement Tasks Checklist

A logically ordered, actionable checklist to guide architectural and code-level improvements for Meeting Buddy. Each item can be checked off as completed.

1. [ ] Establish project-wide engineering foundations

   - [ ] Add code style and quality tooling: ruff (lint), black (format), isort (imports), mypy (type checks); integrate with tox and pre-commit
   - [ ] Enforce minimum supported Python version in pyproject and type strictness (enable mypy strict optional where practical)
   - [ ] Configure logging defaults via utils/logging_config.setup_default_logging in app entry
   - [ ] Add .env-based configuration support (e.g., python-dotenv) and a centralized config module

2. [ ] Document architecture and developer workflow

   - [ ] Create docs/architecture.md explaining MVP layers (View, Presenter, Models, Utils threads), data flow, queues, callbacks, and lifecycles
   - [ ] Expand README with features, usage, troubleshooting (PyAudio, BlackHole), and platform notes (macOS/Linux/Windows)
   - [ ] Add CONTRIBUTING.md with dev setup, running tests, style guidelines, and commit conventions

3. [ ] Consolidate and clarify responsibility boundaries (architecture)

   - [ ] Define clear boundaries: Threads in utils perform I/O and heavy work; Models encapsulate state; Presenter orchestrates; View remains passive
   - [ ] Reduce duplication between TranscriptionModel and AudioTranscriberThread (single source of transcription pipeline logic)
   - [ ] Reduce duplication between LLMModel and LLMThread (centralize API call logic; thread only orchestrates)
   - [ ] Introduce simple interfaces/protocols for callbacks and models to improve testability and decoupling

4. [ ] Introduce configuration module

   - [ ] Create meeting_buddy/config.py with typed settings (e.g., Whisper model name, language, sample rates, LLM endpoint/model, timeouts)
   - [ ] Load order: env vars -> .env -> defaults; expose getters used across models/threads
   - [ ] Add validation for configuration (e.g., URL schemes, numeric ranges)

5. [ ] Improve concurrency and lifecycle management

   - [ ] Establish a consistent thread lifecycle pattern (start/stop/join, is_running flags, cleanup) across AudioRecorderThread, AudioTranscriberThread, LLMThread
   - [ ] Add graceful shutdown with timeouts and draining behavior for queues (backpressure, bounded sizes, clear semantics)
   - [ ] Ensure Presenter.stop\_\* waits for worker thread termination and handles late callbacks safely (weak refs or state guards)
   - [ ] Add retry/backoff for LLM requests and Whisper load/transcription operations with capped retries
   - [ ] Add cancellation support for in-flight operations when stopping

6. [ ] Strengthen error handling and resilience

   - [ ] Define a small hierarchy of custom exceptions (e.g., AudioInitError, TranscriptionError, LLMRequestError)
   - [ ] Normalize error reporting via callbacks and unified UI display methods in Presenter/View
   - [ ] Add timeouts for external operations (requests to LLM API, model loading)
   - [ ] Provide fallback behavior when Whisper or PyAudio are not available (disable features with clear messages)

7. [ ] Enhance typing, contracts, and docs in code

   - [ ] Complete type hints across public APIs; replace Optional[...] where defaults exist; prefer Protocols for callback types
   - [ ] Add docstrings with param/return/raises to public functions and classes
   - [ ] Use dataclasses for simple data containers (AudioChunk, TranscriptionResult, RecordingInfo, LLMRequest/Response)

8. [ ] Logging and observability

   - [ ] Standardize logger usage via LoggerMixin or get_logger; remove print statements
   - [ ] Add structured/contextual logging (chunk_id, request_id, durations)
   - [ ] Provide verbosity controls (trace/debug/info) and ensure sensitive data (API keys, prompts) are redacted if needed
   - [ ] Optionally add simple performance metrics counters and durations exported by get\_\*\_stats methods

9. [ ] Persistence and state improvements

   - [ ] Add optional persistence for settings (selected device, whisper model, window size) using a small settings file
   - [ ] Consider persistence of transcriptions and LLM responses per session with export/import helpers

10. [ ] Presenter and View UX improvements

    - [ ] Ensure UI remains responsive: avoid heavy work in UI thread; throttle UI updates for live transcription/LLM streaming
    - [ ] Implement status indicators (connected, recording, transcribing) with clear transitions
    - [ ] Validate and sanitize user inputs (prompt length, forbidden characters)
    - [ ] Add keyboard shortcuts for start/stop and clearing fields

11. [ ] Audio pipeline robustness

    - [ ] Validate sample rate/channel consistency between device, recorder thread, and Whisper expectations
    - [ ] Handle resampling if device sample rate mismatches Whisper
    - [ ] Add RMS/silence detection to drop empty or near-silent chunks to reduce load
    - [ ] Backpressure strategy when transcription falls behind (drop oldest, coalesce chunks, or pause recording)

12. [ ] Transcription pipeline improvements

    - [ ] Centralize Whisper preprocessing and language/model selection
    - [ ] Add dynamic batching or chunk coalescing to improve accuracy vs latency
    - [ ] Expose interim vs final result semantics; handle confidence scoring consistently
    - [ ] Add export formats beyond plain text (SRT/VTT with timestamps already partially present)

13. [ ] LLM integration improvements

    - [ ] Make endpoint/model configurable; add streaming and non-streaming modes abstraction
    - [ ] Implement request deduplication/debouncing to avoid spamming API on tiny updates
    - [ ] Add prompt templates and variable substitution; allow saving/loading prompts
    - [ ] Add rate limiting and concurrency caps for outgoing requests

14. [ ] Testing strategy and coverage

    - [ ] Add unit tests for models (RecordingModel, TranscriptionModel, LLMModel) with mocks for Whisper, PyAudio, and requests
    - [ ] Add unit tests for thread classes using fake queues and time controls; verify lifecycle, backpressure, and callbacks
    - [ ] Add presenter-level tests with a stub view to assert orchestration logic
    - [ ] Add basic integration test for end-to-end flow with fake audio and a mock LLM server
    - [ ] Configure CI (GitHub Actions) to run tox on pushes and PRs for multiple platforms (at least ubuntu-latest; optionally macos)

15. [ ] Packaging and distribution

    - [ ] Add console_script entry point in pyproject for meeting-buddy cli
    - [ ] Verify wheel build via tox and publish workflow (optional)
    - [ ] Ensure extras for UI vs headless (e.g., extra dependencies for PyQt6)

16. [ ] Performance and memory

    - [ ] Profile CPU hotspots during transcription; consider numpy vectorization where applicable
    - [ ] Add safeguards to trim retained histories (audio_chunks, transcription_results) or paginate in memory
    - [ ] Provide limits in RecordingModel to prevent unbounded growth

17. [ ] Security and privacy

    - [ ] Clearly document data flow and any external API calls; provide a local-only mode
    - [ ] Add an explicit user consent toggle before sending data to LLM endpoints
    - [ ] Redact or hash identifiers in logs; provide a privacy mode disabling file logging

18. [ ] Platform compatibility and installation

    - [ ] Add guidance and checks for PyAudio installation per platform
    - [ ] Provide fallback or instructions for Linux/Windows (in addition to macOS BlackHole)
    - [ ] Add Makefile targets for lint, type-check, test, format; update README

19. [ ] Housekeeping and cleanup

    - [ ] Ensure all cleanup/**del** methods are idempotent and safe to call multiple times
    - [ ] Use context managers where applicable (temporary dirs/files)
    - [ ] Review exception swallowing; re-raise where callers need to react; add actionable messages

20. [ ] Roadmap and future improvements
    - [ ] Multi-language detection and automatic language switching
    - [ ] Hot-swap Whisper models while running (with safe pause)
    - [ ] Plugin system for different LLM providers and speech-to-text engines
