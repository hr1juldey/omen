## ADDED Requirements

### Requirement: LD_LIBRARY_PATH setup at addon startup
The addon SHALL set `LD_LIBRARY_PATH` to include `modular/lib/` directory at import time. This MUST happen before any ctypes calls to load Mojo .so files.

#### Scenario: Addon enable sets LD_LIBRARY_PATH
- **WHEN** the addon is enabled in Blender
- **THEN** LD_LIBRARY_PATH includes the modular/lib/ directory from the installed modular nightly package

### Requirement: Load pre-compiled Mojo .so kernels via ctypes
The system SHALL load Mojo-compiled .so files using Python's ctypes module. Each kernel SHALL expose C-callable functions that accept numpy array pointers and dimensions.

#### Scenario: Loading tile_fingerprint kernel
- **WHEN** the Mojo runtime initializes
- **THEN** omen_kernels.so is loaded via ctypes and tile_fingerprint_kernel function is available

#### Scenario: .so file missing
- **WHEN** omen_kernels.so is not found at expected path
- **THEN** a clear error message is logged indicating the missing file and how to compile it

### Requirement: Modular nightly version check
The system SHALL verify that modular nightly (version containing `.dev`) is installed. If stable modular is found, the system SHALL show a user-facing error with installation instructions.

#### Scenario: Modular nightly installed
- **WHEN** modular version contains `.dev` (e.g., 26.4.0.dev2026051206)
- **THEN** runtime initialization proceeds normally

#### Scenario: Stable modular installed
- **WHEN** modular version does not contain `.dev` (e.g., 26.2.0)
- **THEN** a clear error is shown: "Omen requires modular nightly. Run: pip install --pre modular --index https://whl.modular.com/nightly/simple/"
