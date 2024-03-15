import numpy as np

from ntrfc.meshquality.standards import classify_mesh_quality


def test_classify_mesh_quality():
    # Example usage:
    quality_name = "MeshExpansion"
    value = np.array([23, 12, 32, 12, 23])
    quality = classify_mesh_quality(quality_name, value)
    assert quality == True

    value = np.array([23, 12, 32, 12, 123])
    quality = classify_mesh_quality(quality_name, value)
    assert quality == False
