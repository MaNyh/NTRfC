import numpy as np

def calc_concavehull(x, y, alpha=None):

    if alpha is not None:
        import warnings
        warnings.warn("The `alpha` parameter is deprecated and will be removed in a future version.", DeprecationWarning)

    points = np.stack([x,y]).T

    # Step 1: Find the center of the shape
    center_x = points[:, 0].mean()
    center_y = points[:, 1].mean()

    # Step 2: Calculate the angle of each point relative to the center
    angles = np.arctan2(points[:, 1] - center_y, points[:, 0] - center_x)

    # Step 3: Sort the points based on their angles
    angles, points = angles.argsort(), points[angles.argsort()]

    return points[::,0],points[::,1]
