## Context

Omen render engine currently has a Blender RenderEngine skeleton that outputs test gradients. The Mitsuba-Blender addon already exists and provides Blender→Mitsuba integration via scene export and the `MitsubaRenderEngine` class. We need to create a real integrator that plugs into this existing infrastructure.

**Current state:**
- Mitsuba 3.8.0 installed via pip in pixi environment
- Mitsuba-Blender addon at `/mitsuba-blender/` handles Blender integration
- Integrators are registered in `mitsuba-blender/mitsuba-blender/engine/integrators.json`
- Reference implementation: `mitsuba3/src/integrators/path.cpp` (383 lines C++)

**Constraints:**
- CLAUDE_POLICY.md: absolute imports, 100-line file limit, SOLID, Ruff compliance
- Must use Mitsuba Python plugin API (not C++)
- No Blender-specific code (Omen runs within Mitsuba process)
- JEPA/Mojo integration is future work (placeholder only)

## Goals / Non-Goals

**Goals:**
- Create Omen as a Mitsuba Python integrator plugin
- Implement standard path tracing with MIS
- Register with Mitsuba and expose in Blender UI
- Establish architecture for future JEPA/Mojo integration

**Non-Goals:**
- JEPA scene analysis (placeholder parameters only)
- Mojo GPU kernels (parameter storage only)
- Real-time viewport rendering
- Material node compilation (use Mitsuba's built-in BSDFs)

## Decisions

### Decision 1: Python plugin vs C++ implementation
**Choice:** Python plugin using `mi.Integrator` base class

**Rationale:**
- Faster development iteration
- Sufficient for initial path tracing
- Easier integration with future JEPA (Python ML libraries)
- Mitsuba tutorial demonstrates this pattern

**Alternatives considered:**
- C++ integrator: More performant but requires build toolchain, slower development
- Hybrid: Too complex for initial implementation

### Decision 2: File structure for 100-line limit
**Choice:** Split into 4 focused modules under 100 lines each

```
src/omen_integrator/
├── __init__.py       # Plugin registration, OmenIntegrator class (~70 lines)
├── core.py           # Path tracing render loop (~100 lines)
├── jepa.py           # JEPA integration (future, ~100 lines)
└── gpu.py            # Mojo kernel wrappers (future, ~100 lines)
```

**Rationale:**
- Meets CLAUDE_POLICY.md 100-line requirement
- Clear separation of concerns (SOLID)
- Future extensions have designated modules
- Each module is independently testable

### Decision 3: Parameter storage pattern
**Choice:** Store JEPA/GPU parameters as instance variables, use later

**Rationale:**
- Parameters are visible via `mi.traverse()` mechanism
- No impact on current rendering
- Easy to wire up when implementation ready

### Decision 4: Mitsuba-Blender integration
**Choice:** Add JSON entry only, no Blender Python code

**Rationale:**
- Mitsuba-Blender already handles integrator registration
- Omen runs inside Mitsuba, not Blender
- Minimal coupling between systems

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Python performance vs C++ | Start with Python, profile before optimizing |
| Mitsuba API changes | Pin to Mitsuba 3.8.x, document version requirement |
| File size limits cause fragmentation | Keep modules focused, use clear boundaries |
| JEPA placeholder becomes tech debt | Track in tasks.md, implement in follow-up change |

## Migration Plan

### Phase 1: Core integrator (this change)
1. Create `src/omen_integrator/` modules
2. Register plugin with Mitsuba
3. Add to Mitsuba-Blender integrators.json
4. Verify rendering produces output (not test gradient)

### Phase 2: JEPA integration (future change)
1. Implement `jepa.py` scene analysis
2. Generate adaptive sampling maps
3. Integrate with core render loop
4. Benchmark speedup

### Phase 3: Mojo GPU (future change)
1. Implement `gpu.py` kernel wrappers
2. Port path tracing to Mojo
3. Integrate with Mitsuba variant system
4. Performance validation

### Rollback strategy
- Remove `"omen"` from `integrators.json`
- Delete `src/omen_integrator/` directory
- Mitsuba-Blender continues with default integrators

## Open Questions

1. **Q:** Should we support participating media (volumes)?
   **A:** No, defer to future change. Path tracer in `path.cpp` doesn't handle volumes.

2. **Q:** Which Mitsuba variant should we target?
   **A:** `llvm_ad_rgb` for JIT compilation (recommended in docs)

3. **Q:** How to handle JEPA model loading failures?
   **A:** Graceful degradation to uniform sampling, log warning
