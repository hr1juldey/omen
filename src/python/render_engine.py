"""Omen render engine — Blender plugin via bpy.types.RenderEngine.

Registers as a Blender external render engine. Uses EEVEE for viewport
preview and Mitsuba + JEPA denoiser for final rendering.

Architecture (follows Cycles' BlenderSync pattern):
    .blend file → depsgraph → scene graph extraction → Mitsuba render → JEPA denoise → result

Blender shared node system (NTREE_SHADER):
    Material.node_tree (bNodeTree) → nodes (bNode) + links (bNodeLink)
    → Omen reads material types + params for scene graph conditioning
    (same bNodeTree that EEVEE and Cycles both read)

Source: source/blender/render/RE_engine.h — RenderEngineType callbacks
"""

import logging

logger = logging.getLogger("omen.render_engine")

try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

# Lazy imports — Mitsuba/Nabla may not be installed in Blender's Python
_mi = None
_nb = None


def _get_mitsuba():
    global _mi
    if _mi is None:
        import mitsuba as mi
        mi.set_variant("llvm_ad_rgb")
        _mi = mi
    return _mi


def _get_nabla():
    global _nb
    if _nb is None:
        import nabla as nb
        _nb = nb
    return _nb


if BLENDER_AVAILABLE:

    class OmenProperties(bpy.types.PropertyGroup):
        """Omen render settings stored in Scene."""
        spp: bpy.props.IntProperty(
            name="Samples Per Pixel",
            description="Number of light samples per pixel (low = faster, noisier)",
            default=4,
            min=1,
            max=4096,
        )
        spp_gt: bpy.props.IntProperty(
            name="GT Samples",
            description="Ground truth samples for training pair generation",
            default=256,
            min=1,
            max=65536,
        )
        use_denoiser: bpy.props.BoolProperty(
            name="Use JEPA Denoiser",
            description="Run JEPA neural denoiser on the noisy render",
            default=True,
        )
        model_tier: bpy.props.EnumProperty(
            name="Model Tier",
            description="Omen model quality/speed tradeoff",
            items=[
                ('FAST', "Fast (~4M)", "Real-time preview, 4M params"),
                ('MEDIUM', "Medium (~16M)", "Production quality, 16M params"),
                ('HIGH', "High (~64M)", "Film quality, 64M params"),
            ],
            default='MEDIUM',
        )
        model_path: bpy.props.StringProperty(
            name="Model Path",
            description="Path to Omen .omen checkpoint file",
            subtype='FILE_PATH',
            default="",
        )
        export_motion_vectors: bpy.props.BoolProperty(
            name="Motion Vectors",
            description="Enable motion vector AOV for temporal denoising",
            default=True,
        )
        export_cryptomatte: bpy.props.BoolProperty(
            name="Cryptomatte",
            description="Enable cryptomatte AOV for MoE expert routing",
            default=True,
        )

    class OmenRenderEngine(bpy.types.RenderEngine):
        """Omen render engine — JEPA-accelerated path tracing via Mitsuba 3.

        Blender RenderEngine API callbacks (from RE_engine.h):
            update_render_passes — declare AOV passes
            render              — final render (F12)
            view_update         — viewport scene sync
            view_draw           — viewport render
        """

        bl_idname = "OMEN_RENDER"
        bl_label = "Omen"
        bl_use_preview = True
        bl_use_eevee_viewport = True
        bl_use_postprocess = True
        bl_use_shading_nodes_custom = True

        # ------------------------------------------------------------------
        # Render pass registration
        # ------------------------------------------------------------------

        def update_render_passes(self, scene, view_layer):
            """Declare all AOV passes Omen produces.

            Called before render() to tell Blender what passes to expect.
            Maps to RE_engine_register_pass() in RE_engine.h.
            """
            # Standard
            self.register_pass(scene, view_layer, "Combined", 4, "RGBA", 'COLOR')
            self.register_pass(scene, view_layer, "Depth", 1, "Z", 'VALUE')

            # AOV buffers for denoiser input
            self.register_pass(scene, view_layer, "Diffuse Color", 3, "RGB", 'COLOR')
            self.register_pass(scene, view_layer, "Specular Color", 3, "RGB", 'COLOR')
            self.register_pass(scene, view_layer, "Normal", 3, "XYZ", 'VECTOR')

            # Motion vectors (for temporal reprojection + motion-aware MoE)
            if scene.omen_props.export_motion_vectors:
                self.register_pass(scene, view_layer, "Vector", 4, "XYZW", 'VECTOR')

            # Cryptomatte (for tile-based MoE routing via material histograms)
            if scene.omen_props.export_cryptomatte:
                self.register_pass(
                    scene, view_layer, "CryptoMaterial", 4, "RGBAAA", 'COLOR'
                )

        # ------------------------------------------------------------------
        # Final render (F12)
        # ------------------------------------------------------------------

        def render(self, depsgraph):
            """Final render callback — render via Mitsuba, denoise via JEPA.

            Pipeline:
                depsgraph → scene graph extraction → Mitsuba render → JEPA denoise → result

            Maps to RenderEngineType.render() in RE_engine.h.
            """
            scene = depsgraph.scene_eval
            width, height = self._get_dimensions(depsgraph)

            # Step 1: Extract scene graph from Blender depsgraph
            scene_graph = self._extract_scene_graph(depsgraph)

            # Step 2: Render noisy via Mitsuba
            props = scene.omen_props
            spp = props.spp

            # Step 3: Convert scene graph → Mitsuba scene → render
            noisy_pixels = self._render_mitsuba(scene_graph, width, height, spp)

            # Step 4: Denoise via JEPA model (if enabled)
            if props.use_denoiser and props.model_path:
                clean_pixels = self._denoise(
                    noisy_pixels, scene_graph, width, height, props
                )
            else:
                clean_pixels = noisy_pixels

            # Step 5: Return result to Blender
            result = self.begin_result(0, 0, width, height)
            result.layers[0].passes["Combined"].rect = clean_pixels
            self.end_result(result)

        # ------------------------------------------------------------------
        # Viewport (delegates to EEVEE)
        # ------------------------------------------------------------------

        def view_update(self, context, depsgraph):
            """Viewport update trigger — EEVEE handles actual drawing."""
            pass

        def view_draw(self, context, depsgraph):
            """Viewport draw — EEVEE handles actual drawing."""
            pass

        # ------------------------------------------------------------------
        # Scene graph extraction (from depsgraph)
        # ------------------------------------------------------------------

        def _extract_scene_graph(self, depsgraph):
            """Extract scene graph from Blender depsgraph.

            Reads the same shared bNodeTree (NTREE_SHADER) that both
            EEVEE and Cycles read for material evaluation.

            Returns:
                dict with keys: meshes, materials, lights, cameras
            """
            scene_graph = {
                "meshes": [],
                "materials": [],
                "lights": [],
                "cameras": [],
            }

            for obj in depsgraph.objects:
                if obj.type == 'MESH':
                    scene_graph["meshes"].append(
                        self._extract_mesh(obj)
                    )
                elif obj.type == 'LIGHT':
                    scene_graph["lights"].append(
                        self._extract_light(obj)
                    )
                elif obj.type == 'CAMERA':
                    scene_graph["cameras"].append(
                        self._extract_camera(obj)
                    )

            # Extract materials from shared node system
            seen_materials = set()
            for obj in depsgraph.objects:
                if hasattr(obj, 'material_slots'):
                    for slot in obj.material_slots:
                        mat = slot.material
                        if mat and mat.name not in seen_materials:
                            seen_materials.add(mat.name)
                            scene_graph["materials"].append(
                                self._extract_material(mat)
                            )

            logger.info(
                "Scene graph: %d meshes, %d materials, %d lights, %d cameras",
                len(scene_graph["meshes"]),
                len(scene_graph["materials"]),
                len(scene_graph["lights"]),
                len(scene_graph["cameras"]),
            )
            return scene_graph

        def _extract_mesh(self, obj):
            """Extract evaluated mesh geometry."""
            mesh = obj.to_mesh()
            if mesh is None:
                return {"name": obj.name, "vertices": [], "faces": []}

            data = {
                "name": obj.name,
                "transform": [list(row) for row in obj.matrix_world],
                "material_indices": (
                    [p.material_index for p in mesh.polygons]
                    if mesh.polygons else []
                ),
            }

            # Vertex positions
            data["vertices"] = [v.co[:] for v in mesh.vertices]

            # Face indices
            data["faces"] = [list(p.vertices) for p in mesh.polygons]

            # Normals
            if mesh.normals():
                data["normals"] = [n[:] for n in mesh.normals()]

            # UV coordinates
            if mesh.uv_layers.active:
                data["uvs"] = [d.uv[:] for d in mesh.uv_layers.active.data]

            obj.to_mesh_clear()
            return data

        def _extract_material(self, mat):
            """Extract material from Blender's shared node system (bNodeTree).

            Reads Material.node_tree → bNodeTree (NTREE_SHADER)
            Same structure that EEVEE (gpu_fn) and Cycles (shader.cpp) read.
            """
            material_data = {
                "name": mat.name,
                "nodes": [],
                "links": [],
            }

            if not mat.use_nodes or not mat.node_tree:
                # No node tree — extract basic properties
                material_data["diffuse_color"] = list(mat.diffuse_color)
                material_data["metallic"] = mat.metallic
                material_data["roughness"] = mat.roughness
                return material_data

            ntree = mat.node_tree

            # Read nodes
            for node in ntree.nodes:
                node_data = {
                    "type": node.bl_idname,
                    "name": node.name,
                    "location": list(node.location),
                    "inputs": {},
                }

                # Read input socket values (unlinked = default value)
                for socket in node.inputs:
                    if socket.is_linked:
                        node_data["inputs"][socket.name] = {
                            "linked": True,
                            "from_node": socket.links[0].from_node.name,
                            "from_socket": socket.links[0].from_socket.identifier,
                        }
                    else:
                        node_data["inputs"][socket.name] = {
                            "linked": False,
                            "value": self._get_socket_value(socket),
                        }

                material_data["nodes"].append(node_data)

            # Read links
            for link in ntree.links:
                material_data["links"].append({
                    "from_node": link.from_node.name,
                    "from_socket": link.from_socket.identifier,
                    "to_node": link.to_node.name,
                    "to_socket": link.to_socket.identifier,
                })

            return material_data

        def _extract_light(self, obj):
            """Extract light parameters."""
            light = obj.data
            return {
                "name": obj.name,
                "type": light.type,
                "energy": light.energy,
                "color": list(light.color),
                "transform": [list(row) for row in obj.matrix_world],
            }

        def _extract_camera(self, obj):
            """Extract camera parameters."""
            cam = obj.data
            return {
                "name": obj.name,
                "fov": cam.angle,
                "clip_start": cam.clip_start,
                "clip_end": cam.clip_end,
                "transform": [list(row) for row in obj.matrix_world],
            }

        @staticmethod
        def _get_socket_value(socket):
            """Extract default value from unlinked bNodeSocket."""
            if not hasattr(socket, 'default_value'):
                return None
            try:
                if socket.type == 'VALUE':
                    return float(socket.default_value)
                elif socket.type == 'RGBA':
                    return list(socket.default_value)[:4]
                elif socket.type == 'VECTOR':
                    return list(socket.default_value)[:3]
                elif socket.type == 'INT':
                    return int(socket.default_value)
                elif socket.type == 'BOOLEAN':
                    return bool(socket.default_value)
            except Exception:
                return None
            return None

        # ------------------------------------------------------------------
        # Rendering pipeline
        # ------------------------------------------------------------------

        def _render_mitsuba(self, scene_graph, width, height, spp):
            """Render scene via Mitsuba 3."""
            # TODO: Convert scene_graph → Mitsuba scene dict → render
            # For now return gradient placeholder
            return _generate_gradient(width, height)

        def _denoise(self, noisy_pixels, scene_graph, width, height, props):
            """Denoise render via JEPA model."""
            # TODO: Load model, run U-Net + MoE + scene conditioning
            # For now return noisy as-is
            return noisy_pixels

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def _get_dimensions(self, depsgraph):
            """Extract render dimensions from depsgraph."""
            scene = depsgraph.scene
            scale = scene.render.resolution_percentage / 100.0
            width = int(scene.render.resolution_x * scale)
            height = int(scene.render.resolution_y * scale)
            return width, height

    # ------------------------------------------------------------------
    # Blender addon registration
    # ------------------------------------------------------------------

    classes = (
        OmenProperties,
        OmenRenderEngine,
    )

    def register():
        for cls in classes:
            bpy.utils.register_class(cls)
        bpy.types.Scene.omen_props = bpy.props.PointerProperty(
            type=OmenProperties
        )

    def unregister():
        for cls in reversed(classes):
            bpy.utils.unregister_class(cls)
        del bpy.types.Scene.omen_props

else:

    def register():
        pass

    def unregister():
        pass


def _generate_gradient(width, height):
    """Generate placeholder gradient for testing."""
    pixels = []
    for y in range(height):
        for x in range(width):
            t = x / max(1, width - 1)
            pixels.append([1.0 - t, 0.0, t, 1.0])
    return pixels
