"""Tests for Section 13: Checkpointing, scene caching, GPU budget.

Task 13.9: Save checkpoint, crash, resume from checkpoint.
Task 13.4: Topology hash consistency.
Task 13.8: GPU memory budget check.
"""

import json
import os
import tempfile

import numpy as np


def _fake_model():
    """Create a minimal model-like object with state_dict / named_parameters."""
    class FakeParam:
        def __init__(self, data):
            self.data = np.array(data, dtype=np.float32)
            self.grad = None
            self.ndim = data.ndim if hasattr(data, "ndim") else 1
        def to_numpy(self):
            return self.data
        @property
        def shape(self):
            return self.data.shape
        @property
        def size(self):
            return self.data.size

    class FakeModel:
        def __init__(self):
            self._params = {"w1": FakeParam(np.random.randn(4, 4)),
                            "w2": FakeParam(np.random.randn(4, 4))}
        def state_dict(self):
            return dict(self._params)
        def load_state_dict(self, sd):
            for k, v in sd.items():
                if k in self._params:
                    self._params[k].data = np.array(v)
        def named_parameters(self):
            return list(self._params.items())
        def parameters(self):
            return [p.data for p in self._params.values()]
        def eval(self):
            pass
        def train(self):
            pass

    return FakeModel()


def test_13_9_save_crash_resume():
    """Task 13.9: Save checkpoint, simulate crash, resume from checkpoint."""
    from omen.checkpoint import save_checkpoint, load_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt")
        model = _fake_model()
        original_w1 = model.state_dict()["w1"].to_numpy().copy()

        save_checkpoint(model, None, ckpt_path, tier="medium", iteration=42)

        assert os.path.exists(os.path.join(ckpt_path, "weights.npz"))
        assert os.path.exists(os.path.join(ckpt_path, "metadata.json"))

        with open(os.path.join(ckpt_path, "metadata.json")) as f:
            meta = json.load(f)
        assert meta["iteration"] == 42
        assert meta["tier"] == "medium"
        assert "arch_hash" in meta

        model2 = _fake_model()
        loaded_iter, loaded_meta = load_checkpoint(model2, ckpt_path, "medium")
        assert loaded_iter == 42
        restored_w1 = model2.state_dict()["w1"].to_numpy()
        np.testing.assert_allclose(restored_w1, original_w1, atol=1e-6)
        print("13.9 PASS: save -> crash -> resume verified")


def test_13_4_topology_hash():
    """Task 13.4: Topology-based scene hashing."""
    from omen.scene_cache import compute_topology_hash

    sg1 = {"geometry": np.zeros((100, 3)), "materials": np.zeros((5, 10)),
           "lights": np.zeros((3, 6)), "camera": np.zeros((7,))}
    sg2 = {"geometry": np.ones((100, 3)), "materials": np.ones((5, 10)),
           "lights": np.ones((3, 6)), "camera": np.ones((7,))}

    h1 = compute_topology_hash(sg1)
    h2 = compute_topology_hash(sg2)
    assert h1 == h2, f"Same topology should hash equal: {h1} != {h2}"

    sg3 = {"geometry": np.zeros((200, 3))}
    h3 = compute_topology_hash(sg3)
    assert h1 != h3, "Different topology should hash different"
    print(f"13.4 PASS: topology hash consistent ({h1})")


def test_13_8_gpu_budget():
    """Task 13.8: GPU memory budget check."""
    from omen.gpu_budget import check_memory_budget, estimate_frame_memory

    result = check_memory_budget("inference", "medium")
    assert "sufficient" in result
    assert "budget_mb" in result
    assert result["budget_mb"] > 0

    train = check_memory_budget("training", "medium")
    inf = check_memory_budget("inference", "medium")
    assert train["budget_mb"] > inf["budget_mb"]

    est = estimate_frame_memory(2160, 3840, "medium")
    assert 200 < est < 2000, f"Unexpected 4K estimate: {est}MB"
    print(f"13.8 PASS: budget inf={inf['budget_mb']}MB "
          f"train={train['budget_mb']}MB 4K={est}MB")


if __name__ == "__main__":
    test_13_4_topology_hash()
    test_13_8_gpu_budget()
    test_13_9_save_crash_resume()
    print("All Section 13 tests passed")
