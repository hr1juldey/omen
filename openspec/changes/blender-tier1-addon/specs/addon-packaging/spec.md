## ADDED Requirements

### Requirement: ZIP install workflow
The addon SHALL be installable via Blender's standard "Install from ZIP" workflow (Edit > Preferences > Add-ons > Install). The user SHALL NOT need to open a terminal or run any commands.

#### Scenario: First-time install
- **WHEN** user downloads omen_blender.zip and installs it via Blender's addon installer
- **THEN** the addon appears in the addon list and can be enabled

#### Scenario: Enable after install
- **WHEN** user checks the addon checkbox to enable it
- **THEN** "Omen" appears in the render engine dropdown

### Requirement: Auto-installer for dependencies
The addon SHALL automatically install required pip packages (modular nightly, nabla-ml, mitsuba, numpy) on first enable using Blender's bundled Python. The install SHALL run silently with a progress indication.

#### Scenario: First enable triggers install
- **WHEN** the addon is enabled for the first time
- **THEN** dependencies are installed into Blender's Python site-packages

#### Scenario: Subsequent enables skip install
- **WHEN** the addon is enabled after dependencies are already installed
- **THEN** no installation occurs and the addon starts immediately

### Requirement: Build script produces distributable ZIP
The project SHALL include a build script that produces a distributable ZIP containing: addon code, pre-compiled Mojo .so kernels, and optionally bundled wheels for offline install.

#### Scenario: Build script execution
- **WHEN** the build script is run
- **THEN** omen_blender.zip is created containing all necessary files for end-user installation
