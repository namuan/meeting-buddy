# MVP Architecture Boundaries for Meeting Buddy

This document defines clear boundaries and responsibilities for each component in the Meeting Buddy application's MVP (Model-View-Presenter) architecture.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      View       │◄──►│   Presenter     │◄──►│     Models      │
│   (Passive)     │    │ (Orchestrator)  │    │ (State & Data)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Utils/Threads  │
                       │ (I/O & Heavy    │
                       │    Work)        │
                       └─────────────────┘
```

## Component Boundaries

### 1. Models (State & Data Management)

**Responsibilities:**

- Encapsulate application state and data
- Provide data access methods
- Validate data integrity
- Notify of state changes through callbacks
- Maintain data consistency

**What Models SHOULD do:**

- Store and manage application data (recordings, transcriptions, device info, LLM requests/responses)
- Provide getter/setter methods for data access
- Validate data before storing
- Maintain internal data consistency
- Expose callbacks for state change notifications
- Handle data serialization/deserialization

**What Models SHOULD NOT do:**

- Perform I/O operations (file reading/writing, network calls)
- Execute heavy computational tasks
- Directly interact with hardware (audio devices, microphones)
- Manage threads or asynchronous operations
- Contain UI logic or know about UI components
- Process audio data or run ML models

**Current Violations to Fix:**

- `TranscriptionModel` loads and runs Whisper model (should be in utils)
- `LLMModel` makes HTTP requests directly (should be in utils)
- Models contain threading logic for processing

### 2. Views (Passive UI Components)

**Responsibilities:**

- Display data to the user
- Capture user input
- Delegate user actions to presenter via callbacks
- Remain completely passive

**What Views SHOULD do:**

- Render UI components and layouts
- Display data provided by presenter
- Capture user interactions (clicks, text input, selections)
- Call presenter callbacks when user actions occur
- Update display when presenter provides new data
- Handle UI-specific formatting and styling

**What Views SHOULD NOT do:**

- Contain business logic
- Directly access or modify models
- Perform data validation or processing
- Make network calls or file I/O
- Manage application state
- Know about other views or components
- Perform calculations or data transformations

**Current Implementation Status:**

- ✅ View is properly passive and uses callbacks
- ✅ No direct model access
- ✅ Proper separation of UI concerns

### 3. Presenters (Orchestration & Business Logic)

**Responsibilities:**

- Coordinate between models and views
- Handle business logic and application flow
- Manage component lifecycle
- Orchestrate utils/threads for I/O and heavy work

**What Presenters SHOULD do:**

- Initialize and coordinate models, views, and utils
- Handle user actions from view callbacks
- Update models based on business logic
- Update views with data from models
- Manage application state transitions
- Orchestrate background tasks via utils/threads
- Handle error scenarios and recovery
- Implement application-specific workflows

**What Presenters SHOULD NOT do:**

- Perform I/O operations directly
- Execute heavy computational tasks
- Contain UI rendering logic
- Directly manipulate UI components (beyond calling view methods)
- Perform low-level hardware operations

**Current Implementation Status:**

- ✅ Proper coordination between components
- ✅ Good use of callbacks and signals
- ⚠️ Some direct thread management (acceptable for orchestration)

### 4. Utils/Threads (I/O & Heavy Work)

**Responsibilities:**

- Perform I/O operations (file, network, hardware)
- Execute computationally heavy tasks
- Handle asynchronous operations
- Provide results back to presenter via callbacks

**What Utils/Threads SHOULD do:**

- Handle audio recording and hardware interaction
- Perform network requests (LLM API calls)
- Run ML models (Whisper transcription)
- Process audio data and signal processing
- Manage file I/O operations
- Execute long-running or blocking operations
- Provide progress updates and error handling
- Use queues for thread-safe communication

**What Utils/Threads SHOULD NOT do:**

- Directly update UI components
- Manage application state (beyond their specific task)
- Make decisions about business logic
- Know about other application components beyond their immediate needs
- Perform data validation (beyond input sanitization)

**Current Implementation Status:**

- ✅ Proper separation of I/O concerns
- ✅ Good use of callbacks for communication
- ✅ Thread-safe queue-based communication

## Required Refactoring

### 1. Move Heavy Processing from Models to Utils

**TranscriptionModel Refactoring:**

- Remove Whisper model loading and processing
- Keep only transcription data storage and management
- AudioTranscriberThread should handle all Whisper operations

**LLMModel Refactoring:**

- Remove direct HTTP request handling
- Keep only request/response data management
- LLMThread should handle all API communication

### 2. Ensure Clear Communication Patterns

**Data Flow:**

```
User Input → View → Presenter → Utils/Threads
                     ↓
Models ← Presenter ← Utils/Threads (via callbacks)
   ↓
View ← Presenter
```

**Callback Pattern:**

- Utils/Threads communicate with Presenter via callbacks
- Presenter updates Models with results
- Presenter updates Views with new data
- Views notify Presenter of user actions via callbacks

### 3. Logging and Error Handling

**Each component should:**

- Log actions appropriate to its responsibility level
- Handle errors within its boundary
- Propagate errors to presenter when necessary
- Use structured logging with component identification

## Implementation Guidelines

### Model Implementation

```python
class ExampleModel:
    def __init__(self):
        self._data = {}
        self._callbacks = []

    def set_data(self, key, value):
        """Store data and notify observers."""
        if self._validate_data(key, value):
            self._data[key] = value
            self._notify_callbacks(key, value)

    def get_data(self, key):
        """Retrieve data."""
        return self._data.get(key)

    def add_callback(self, callback):
        """Add observer callback."""
        self._callbacks.append(callback)
```

### Utils/Thread Implementation

```python
class ExampleThread(threading.Thread):
    def __init__(self, result_callback, error_callback):
        super().__init__(daemon=True)
        self._result_callback = result_callback
        self._error_callback = error_callback

    def run(self):
        """Perform heavy work and report results."""
        try:
            result = self._do_heavy_work()
            if self._result_callback:
                self._result_callback(result)
        except Exception as e:
            if self._error_callback:
                self._error_callback(e)
```

### Presenter Orchestration

```python
class ExamplePresenter:
    def __init__(self):
        self._model = ExampleModel()
        self._view = ExampleView()
        self._thread = None

        # Connect callbacks
        self._model.add_callback(self._on_model_changed)
        self._view.set_action_callback(self._on_user_action)

    def _on_user_action(self, action_data):
        """Handle user action by orchestrating components."""
        # Start background work
        self._thread = ExampleThread(
            result_callback=self._on_work_complete,
            error_callback=self._on_work_error
        )
        self._thread.start()

    def _on_work_complete(self, result):
        """Handle completion of background work."""
        # Update model with result
        self._model.set_data('result', result)

    def _on_model_changed(self, key, value):
        """Handle model changes by updating view."""
        self._view.update_display(key, value)
```

## Benefits of Clear Boundaries

1. **Testability**: Each component can be tested in isolation
2. **Maintainability**: Changes in one component don't affect others
3. **Reusability**: Components can be reused in different contexts
4. **Debugging**: Issues can be isolated to specific components
5. **Performance**: Heavy work is properly isolated in background threads
6. **Scalability**: New features can be added without disrupting existing code

## Validation Checklist

- [ ] Models only contain data and state management
- [ ] Views are completely passive and use callbacks
- [ ] Presenters orchestrate but don't perform I/O or heavy work
- [ ] Utils/Threads handle all I/O and computational tasks
- [ ] Communication flows through proper callback chains
- [ ] No component directly accesses components outside its boundary
- [ ] Error handling is appropriate to each component's responsibility
- [ ] Logging reflects each component's role and actions
