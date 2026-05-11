# Spec: FFI Bridge

## ADDED Requirements

### Requirement: Directory structure setup
The system SHALL create `src/mojo/` and `src/c/` directories for future Mojo and C components.

#### Scenario: Directories created
- **WHEN** Project structure is initialized
- **THEN** `src/mojo/` and `src/c/` exist with placeholder files

### Requirement: C header placeholder
The system SHALL provide `src/c/omen_core.h` header file defining FFI interface structures.

#### Scenario: Header file exists
- **WHEN** C directory is inspected
- **THEN** `omen_core.h` contains type definitions for Python-Mojo bridge

### Requirement: Mojo module placeholder
The system SHALL create `src/mojo/__init__.mojo` as entry point for future Mojo kernels.

#### Scenario: Mojo module ready
- **WHEN** Mojo directory is inspected
- **THEN** `__init__.mojo` exists with minimal module structure

### Requirement: Forward compatibility
The system SHALL design FFI structures to support future scene data passing and kernel invocation.

#### Scenario: Extensible design
- **WHEN** FFI structures are defined
- **THEN** They accommodate future scene graph and JEPA model parameters

### Requirement: Absolute imports enforcement
The system SHALL use absolute imports only across all language boundaries.

#### Scenario: Import compliance
- **WHEN** Python imports C types or Mojo modules
- **THEN** All imports use absolute paths (no relative imports)
