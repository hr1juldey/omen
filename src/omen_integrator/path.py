"""Single path sampling logic for Omen integrator."""

import mitsuba as mi
import drjit as dr


def trace_path(scene, ray, integrator, sampler, radiance):
    """Trace a single path through the scene.

    Args:
        scene: Mitsuba Scene object
        ray: Initial ray from camera
        integrator: OmenIntegrator instance
        sampler: Mitsuba Sampler for random numbers
        radiance: Accumulated radiance (modified in place)
    """
    throughput = mi.Spectrum(1.0)
    depth = 0
    active = mi.Bool(True)

    # Path tracing loop
    while active:
        # Find intersection
        si = scene.ray_intersect(ray)

        # No intersection - add environment emitter and terminate
        if not si.is_valid():
            if dr.any(active):
                # Sample environment emitters
                for emitter in scene.emitters():
                    if emitter.is_environment():
                        ds, emitter_val = emitter.sample_direction(
                            si, sampler.next_2d()
                        )
                        radiance += throughput * emitter_val
            break

        # Direct illumination sampling (next event estimation)
        if active:
            _sample_direct_lighting(scene, si, sampler, throughput, radiance)

        # BSDF sampling for indirect illumination
        bsdf_sample, bsdf_weight = si.bsdf().sample(
            si, sampler.next_1d(), sampler.next_2d()
        )

        # Update throughput
        throughput *= bsdf_weight

        # Update ray for next bounce
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
        depth += 1

        # Russian roulette termination
        if depth >= integrator.rr_depth:
            rr_prob = min(dr.mean(throughput) * 0.95, 1.0)
            if sampler.next_1d() >= rr_prob:
                break
            throughput /= rr_prob

        # Max depth termination
        if integrator.max_depth != -1 and depth >= integrator.max_depth:
            break


def _sample_direct_lighting(scene, si, sampler, throughput, radiance):
    """Sample direct illumination via next event estimation.

    Args:
        scene: Mitsuba Scene object
        si: Surface interaction point
        sampler: Mitsuba Sampler for random numbers
        throughput: Current path throughput weight
        radiance: Accumulated radiance (modified in place)
    """
    # Sample emitter randomly
    emitter_idx = min(
        len(scene.emitters()) - 1,
        int(sampler.next_1d() * len(scene.emitters())),
    )
    emitter = scene.emitters()[emitter_idx]

    # Sample direct lighting
    ds, emitter_val = emitter.sample_direction(si, sampler.next_2d())
    wo = si.to_local(ds.d)

    # BSDF evaluation
    bsdf = si.bsdf()
    bsdf_val = bsdf.eval(mi.BSDFContext(), si, wo)

    # Visibility test
    ray_shadow = si.spawn_ray_to(ds.p)
    shadow_si = scene.ray_intersect(ray_shadow)

    # Accumulate direct illumination if not occluded
    if not shadow_si.is_valid():
        radiance += throughput * emitter_val * bsdf_val
