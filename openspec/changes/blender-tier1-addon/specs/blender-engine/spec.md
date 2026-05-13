## ADDED Requirements

### Requirement: Omen appears as a render engine in Blender dropdown
The system SHALL register "Omen" as a Blender render engine via `bpy.types.RenderEngine`. The engine SHALL appear in the render engine dropdown menu alongside Cycles and Eevee.

#### Scenario: Engine registration
- **WHEN** the addon is enabled in Blender preferences
- **THEN** "Omen" appears in the render engine dropdown menu

#### Scenario: Engine unregistration
- **WHEN** the addon is disabled in Blender preferences
- **THEN** "Omen" is removed from the render engine dropdown menu

### Requirement: F12 render produces a final image
The system SHALL implement the `render()` callback on `OmenRenderEngine`. When the user presses F12 or triggers a render, the engine SHALL produce a final denoised image and write it to Blender's RenderResult via `begin_result()`/`end_result()`.

#### Scenario: F12 render with a simple scene
- **WHEN** user presses F12 with "Omen" selected as render engine and a scene containing a mesh, camera, and light
- **THEN** the render engine produces a denoised image displayed in Blender's image editor

#### Scenario: F12 render with empty scene
- **WHEN** user presses F12 with an empty scene (no objects)
- **THEN** the render engine produces a black image without crashing

### Requirement: Render settings available in properties panel
The system SHALL expose render settings (SPP, render mode, tile size) in Blender's render properties panel when Omen is the active engine.

#### Scenario: Settings display
- **WHEN** user selects "Omen" as render engine
- **THEN** Omen-specific settings appear in the render properties panel

### Requirement: Thin addon wrapper delegates to engine module
The addon (`src/omen_blender/`) SHALL be a thin wrapper that imports the engine (`src/omen_engine/`). The engine module SHALL be reloadable via Blender's "Reload Scripts" without reinstalling the addon.

#### Scenario: Engine code iteration
- **WHEN** developer modifies engine code and runs "Reload Scripts" in Blender
- **THEN** the updated engine code is active without restarting Blender or reinstalling the addon
