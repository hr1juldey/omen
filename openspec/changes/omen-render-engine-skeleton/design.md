# Design: Omen Render Engine Skeleton

## Context

Omen is a Blender 5.1+ render engine built with Mojo for GPU compute and JEPA-based scene analysis. The project currently has documentation and dependency configuration but no executable code structure. This design establishes the minimal foundation for Blender integration while following strict CLAUDE_POLICY.md constraints (absolute imports, file size limits, SOLID principles).

**Current State**: `omen/` has `pyproject.toml`, `ARCHITECTURE.md`, but no `src/` directory.

**Target State**: Working render engine visible in Blender UI, rendering test pattern.

## Goals / Non-Goals

**Goals:**
- Validate Blender render engine registration pipeline end-to-end
- Establish multi-language directory structure (`src/python/`, `src/mojo/`, `src/c/`)
- Create extensible foundation for scene extraction and JEPA integration
- Follow CLAUDE_POLICY.md constraints (absolute imports, <100 line files)

**Non-Goals:**
- Scene data extraction (next change)
- Mojo kernel invocation (requires FFI bridge completion)
- JEPA model integration (requires scene data + kernels)
- Viewport rendering (final render only initially)
- Production-quality rendering (test pattern sufficient for validation)

## Decisions

### 1. Directory Layout: `src/` prefix with language subdirectories

**Choice**: `src/python/`, `src/mojo/`, `src/c/` instead of flat `python/`, `mojo/`, `c/`.

**Rationale**:
- Clearly separates source code from project root (docs, config, build files)
- Matches multi-language project conventions
- Allows `src/` level operations (linting, testing, packaging)
- User explicitly requested `./src/` structure

**Alternatives Considered**:
- Flat structure: Rejects - mixes source with configuration
- `omen/` prefix: Redundant given project root is already `omen/`

### 2. Absolute Import Strategy: `from src.python.xxx import Yyy`

**Choice**: All imports use absolute `src.python` prefix.

**Rationale**:
- CLAUDE_POLICY.md Rule 1.1 forbids relative imports (`from .`)
- Unambiguous module boundaries aid refactoring
- Tooling (ruff, mypy) handles absolute imports better

**Example**:
```python
# src/python/__init__.py
from src.python.render_engine import OmenRenderEngine

# src/python/render_engine.py
import bpy
from src.python.test_pattern import generate_gradient
```

### 3. Test Pattern: Horizontal Gradient (Red → Blue)

**Choice**: Simple gradient rather than colored squares or noise.

**Rationale**:
- Validates pixel writing pipeline (color interpolation visible)
- Easy to verify correctness (left=red, right=blue)
- Minimal code footprint (<20 lines)

**Alternatives Considered**:
- Colored squares: More complex UV logic
- Noise pattern: Harder to validate correctness
- Single color: Doesn't test gradient interpolation

### 4. File Organization: Split `OmenRenderEngine` from test pattern logic

**Choice**: Separate `render_engine.py` (class) and `test_pattern.py` (function).

**Rationale**:
- Single Responsibility Principle (SOLID)
- Keeps each file under 100-line limit
- Test pattern can be replaced with scene extraction later

**File Structure**:
```
src/python/
├── __init__.py           (30 lines - registration)
├── render_engine.py      (60 lines - OmenRenderEngine class)
└── test_pattern.py       (20 lines - gradient generation)
```

### 5. Render Pipeline: Synchronous `begin_result()` → `write` → `end_result()`

**Choice**: Blocking render call, no threading.

**Rationale**:
- Matches Blender's RenderEngine examples
- Simplest viable implementation
- Threading complexity unnecessary for test pattern

**Flow**:
```python
def render(self, depsgraph):
    w, h = self._get_dimensions(depsgraph)
    pixels = generate_gradient(w, h)
    result = self.begin_result(0, 0, w, h)
    result.layers[0].passes["Combined"].rect = pixels
    self.end_result(result)
```

### 6. C Header: Minimal struct definitions for future scene data

**Choice**: Define `SceneData`, `MeshData` structs without implementations.

**Rationale**:
- Establishes FFI contract early
- Allows parallel Mojo/Python development
- No wasted effort implementing unused functions

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| **Absolute imports require `src/` on `PYTHONPATH`** | Add to Blender addon `__init__.py` before importing `src.python` |
| **100-line file limit may require aggressive splitting** | Design is already modular (render engine, test pattern separate) |
| **Blender API changes between 5.1 and future** | Pin to 5.1+ in docs, version-gate if breaking changes occur |
| **Mojo compilation not validated** | Out of scope for this change - FFI bridge is placeholder only |
| **Test pattern doesn't prove GPU kernels work** | Correct - this change validates Blender integration only, GPU work is separate |

## Migration Plan

1. **Create directories**: `src/python/`, `src/mojo/`, `src/c/`
2. **Implement Python modules** (order matters):
   - `test_pattern.py` (no dependencies)
   - `render_engine.py` (imports test_pattern)
   - `__init__.py` (imports render_engine, defines register/unregister)
3. **Create placeholder files**:
   - `src/mojo/__init__.mojo` (empty module with docstring)
   - `src/c/omen_core.h` (struct definitions only)
4. **Manual testing**:
   - Install addon in Blender
   - Select "Omen" in Render Properties
   - Press F12, verify gradient renders
   - Check Image Editor for output

**Rollback**: Delete `src/` directory - no existing code modified.

## Open Questions

1. **Should we use `bl_use_gpu_context = True` now?**
   - **Decision**: No - test pattern doesn't need GPU, add when implementing Mojo kernels
   - **Impact**: Can be added as single-line change later

2. **Should `test_pattern.py` be module-level function or class method?**
   - **Decision**: Module function (easier to test independently)
   - **Impact**: Affects how `render_engine.py` imports it

3. **Where do we put Blender addon `__init__.py`?**
   - **Decision**: Project root (`omen/__init__.py`) - standard Blender addon location
   - **Impact**: This file adds `src/` to path, then imports `src.python`
