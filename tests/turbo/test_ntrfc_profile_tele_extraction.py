from ntrfc.turbo.profile_tele_extraction import get_centroids


def test_get_centroids():
    # Test for a simple triangle
    triangles = [((0, 0), (0, 1), (1, 1))]
    assert get_centroids(triangles) == [(1 / 3, 2 / 3)]

    # Test for multiple triangles
    triangles = [((0, 0), (0, 1), (1, 1)), ((0, 0), (1, 0), (1, 1))]
    assert get_centroids(triangles) == [(1 / 3, 2 / 3), (2 / 3, 1 / 3)]

    # Test for degenerate triangles (triangles with vertices collinear)
    triangles = [((0, 0), (0, 0), (0, 0)), ((0, 0), (0, 0), (0, 0))]
    assert get_centroids(triangles) == [(0, 0), (0, 0)]
