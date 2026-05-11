# Spec: Blender Render Engine

## ADDED Requirements

### Requirement: Render engine registration
The system SHALL register Omen as a Blender render engine via `bpy.utils.register_class()`.

#### Scenario: Successful registration
- **WHEN** Blender loads the addon
- **THEN** Omen appears in Render Properties > Render Engine dropdown

### Requirement: Render engine identification
The system SHALL define unique engine identifier `bl_idname = "OMEN"` and display label `bl_label = "Omen"`.

#### Scenario: Engine properties
- **WHEN** RenderEngine class is defined
- **THEN** `bl_idname` is "OMEN" and `bl_label` is "Omen"

### Requirement: Preview rendering support
The system SHALL enable material/preview rendering via `bl_use_preview = True`.

#### Scenario: Preview render enabled
- **WHEN** User creates a material preview
- **THEN** Omen engine handles the preview render

### Requirement: Render callback invocation
The system SHALL implement `render(depsgraph)` method that Blender calls for final renders.

#### Scenario: Render method called
- **WHEN** User presses F12 or renders viewport
- **THEN** `render(depsgraph)` is invoked with current dependency graph

### Requirement: Test pattern generation
The system SHALL generate a visual test pattern (gradient or colored regions) for initial validation.

#### Scenario: Test pattern renders
- **WHEN** Render method executes
- **THEN** Output shows recognizable gradient/color pattern

### Requirement: Render result lifecycle
The system SHALL use `begin_result()`, write pixel data, and `end_result()` to display output.

#### Scenario: Render completes successfully
- **WHEN** Render finishes writing pixels
- **THEN** Blender displays the rendered image in Image Editor

### Requirement: Addon cleanup
The system SHALL provide `unregister()` function that removes the render engine from Blender.

#### Scenario: Clean unregister
- **WHEN** Addon is disabled
- **THEN** Omen is removed from render engine list

### Requirement: Source code organization
The system SHALL organize Python code in `src/python/` directory with absolute imports only.

#### Scenario: File structure compliance
- **WHEN** Source files are created
- **THEN** All imports use absolute paths from `src.python` module

### Requirement: File size limits
The system SHALL keep each Python file under 100 lines of executable code plus 50 lines overhead.

#### Scenario: File within limits
- **WHEN** Python modules are written
- **THEN** No file exceeds CLAUDE_POLICY.md size constraints
