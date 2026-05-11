# Proposal: Omen Render Engine Skeleton

## Why

Omen needs a minimal integration point with Blender's render engine system to validate the entire pipeline before adding complexity. This skeleton establishes the "Hello World" foundation - a render engine that successfully registers, renders a test pattern, and displays results - enabling iterative development of JEPA denoising and Mojo GPU kernels.

## What Changes

- **Directory Structure**: Create `src/python/`, `src/mojo/`, `src/c/` for organized multi-language components
- **Render Engine**: Implement `OmenRenderEngine` class as `bpy.types.RenderEngine` subclass
- **Test Pattern Render**: Generate simple gradient/colored output to validate render pipeline
- **Addon Registration**: Set up `register()`/`unregister()` functions for Blender integration
- **FFI Foundation**: Create placeholder C header for future Python-Mojo bridge

## Capabilities

### New Capabilities

- `blender-render-engine`: Blender Python API render engine registration, render callback lifecycle, and result display pipeline
- `ffi-bridge`: Python-to-Mojo foreign function interface layer (foundation only)

### Modified Capabilities

- None (initial implementation)

## Impact

- **Code**: New files in `src/python/`, `src/mojo/`, `src/c/`
- **Dependencies**: Blender 5.1+ Python API (bpy)
- **Testing**: Manual verification in Blender UI
- **Future Work**: Scene extraction, Mojo kernels, JEPA integration build on this foundation
