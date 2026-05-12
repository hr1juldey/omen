## Why

Omen render engine needs a production-ready rendering backend that integrates with Blender's existing Mitsuba-Blender addon. The current Blender RenderEngine skeleton is a placeholder that outputs test patterns. We need a real path tracing integrator that leverages Mitsuba 3's rendering architecture and provides a foundation for JEPA-accelerated rendering.

## What Changes

- Create Omen as a Mitsuba 3 Python plugin (custom integrator)
- Register the integrator with Mitsuba's plugin system
- Add Omen to Mitsuba-Blender's integrator list for Blender UI exposure
- Implement path tracing with standard sampling (JEPA integration comes later)
- Follow CLAUDE_POLICY.md compliance (absolute imports, 100-line file limits, SOLID principles)

## Capabilities

### New Capabilities

- `omen-integrator`: Core path tracing integrator for Mitsuba 3, with parameters for max depth, Russian roulette, JEPA model path (placeholder), and GPU enable flag

### Modified Capabilities

None (this is new functionality, not changing existing specs)

## Impact

- **Mitsuba-Blender addon**: Add `"omen"` entry to `integrators.json`
- **New directory**: `src/omen_integrator/` with Python modules
- **Dependencies**: Mitsuba 3.8.0 (already installed via pip), Dr.Jit (included with Mitsuba)
- **No Blender code changes**: Omen runs independently within Mitsuba's process
- **No breaking changes**: Existing integrators (path, direct, etc.) continue to work
