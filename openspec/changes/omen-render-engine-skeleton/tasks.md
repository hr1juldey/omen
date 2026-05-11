# Tasks: Omen Render Engine Skeleton

## 1. Directory Structure

- [ ] 1.1 Create `src/python/` directory
- [ ] 1.2 Create `src/mojo/` directory
- [ ] 1.3 Create `src/c/` directory

## 2. Python Test Pattern Module

- [ ] 2.1 Create `src/python/test_pattern.py` with `generate_gradient()` function
- [ ] 2.2 Implement horizontal red-to-blue gradient generation
- [ ] 2.3 Return pixel array as list of [r, g, b, a] values

## 3. Render Engine Implementation

- [ ] 3.1 Create `src/python/render_engine.py` with `OmenRenderEngine` class
- [ ] 3.2 Define `bl_idname = "OMEN"` and `bl_label = "Omen"`
- [ ] 3.3 Set `bl_use_preview = True`
- [ ] 3.4 Implement `render(depsgraph)` method
- [ ] 3.5 Add `_get_dimensions(depsgraph)` helper method
- [ ] 3.6 Call `begin_result()`, write pixels, `end_result()`

## 4. Python Module Registration

- [ ] 4.1 Create `src/python/__init__.py`
- [ ] 4.2 Import `OmenRenderEngine` from `src.python.render_engine`
- [ ] 4.3 Define `register()` function with `bpy.utils.register_class()`
- [ ] 4.4 Define `unregister()` function with `bpy.utils.unregister_class()`

## 5. Blender Addon Entry Point

- [ ] 5.1 Create `omen/__init__.py` (Blender addon root)
- [ ] 5.2 Add `bl_info` dictionary (name, version, author, etc.)
- [ ] 5.3 Append `src/` to `sys.path` before importing
- [ ] 5.4 Import `register` and `unregister` from `src.python`
- [ ] 5.5 Define addon `register()` and `unregister()` functions

## 6. Mojo Placeholder

- [ ] 6.1 Create `src/mojo/__init__.mojo`
- [ ] 6.2 Add module docstring describing future kernel implementations
- [ ] 6.3 Leave empty body (no code yet)

## 7. C Header Placeholder

- [ ] 7.1 Create `src/c/omen_core.h`
- [ ] 7.2 Define `SceneData` struct (fields only, no implementation)
- [ ] 7.3 Define `MeshData` struct (fields only, no implementation)
- [ ] 7.4 Add `extern "C"` guards for C++ compatibility

## 8. Verification

- [ ] 8.1 Install addon in Blender (Preferences > Install)
- [ ] 8.2 Select "Omen" in Render Properties > Render Engine dropdown
- [ ] 8.3 Press F12 to render test scene
- [ ] 8.4 Verify gradient appears in Image Editor
- [ ] 8.5 Confirm addon unregisters cleanly on disable
