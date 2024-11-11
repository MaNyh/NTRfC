import numpy as np

from ntrfc.meshquality.standards import classify_mesh_quality


def test_classify_mesh_quality():
    quality_name = "MeshExpansion"
    value = np.array([10, 15, 18, 12, 19])
    quality = classify_mesh_quality(quality_name, value)
    assert quality is True  # GOOD

    value = np.array([23, 12, 32, 12, 123])
    quality = classify_mesh_quality(quality_name, value)
    assert quality is False  # BAD

    value = np.array([23, 12, 32, 12, 30])
    quality = classify_mesh_quality(quality_name, value)
    assert quality is True  # OK

    value = np.array([23, 12, 32, 12, 50])
    quality = classify_mesh_quality(quality_name, value)
    assert quality is False  # BAD

    quality_name = "UndefinedQuality"
    value = np.array([23, 12, 32, 12, 23])
    quality = classify_mesh_quality(quality_name, value)
    assert quality == "Undefined Quality"
