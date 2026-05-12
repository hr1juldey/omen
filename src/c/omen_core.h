/** Omen FFI Bridge - Python-Mojo interface definitions.
 *
 * This header defines data structures for passing scene data
 * between Python (Blender) and Mojo (GPU kernels).
 *
 * Placeholder for future implementation - structs only,
 * no function implementations yet.
 */

#ifndef OMEN_CORE_H
#define OMEN_CORE_H

#ifdef __cplusplus
extern "C" {
#endif


/* Scene data passed from Python to Mojo kernels. */
typedef struct SceneData {
    int width;
    int height;
    int num_objects;
    int num_lights;
    /* Future: camera matrix, world bounds, etc. */
} SceneData;


/* Mesh data for individual objects. */
typedef struct MeshData {
    int num_vertices;
    int num_faces;
    /* Future: vertex positions, normals, UVs, material ID */
} MeshData;


/* Future FFI function declarations:
 * int omen_render_scene(SceneData* scene, MeshData** meshes, float** output);
 * void omen_free_render_buffer(float* buffer);
 */


#ifdef __cplusplus
}
#endif

#endif /* OMEN_CORE_H */
