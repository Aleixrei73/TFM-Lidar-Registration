import numpy as np
from scipy.spatial import cKDTree

PointCloud = np.ndarray
Transform = np.ndarray

def x_to_world(pose: list[float]) -> Transform:
    """Compute the transformation matrix from x-coordinate system to carla world system.

    Parameters
    ----------
    pose : list[float]
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def x1_to_x2(x1: list[float], x2: list[float]) -> Transform:
    """Compute the transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list[float]
        The pose of x1 under world coordinates.
    x2 : list[float]
        The pose of x2 under world coordinates.

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.
    """
    x1_to_world = x_to_world(x1)
    x2_to_world = x_to_world(x2)
    world_to_x2 = np.linalg.inv(x2_to_world)

    transformation_matrix = np.dot(world_to_x2, x1_to_world)
    return transformation_matrix


def apply_transform(points: PointCloud, transform: Transform) -> PointCloud:
    """Apply a 4x4 homogeneous transform to a point cloud.

    Parameters
    ----------
    points : PointCloud
        Point cloud, shape (N, 3).
    transform : Transform
        4x4 homogeneous transform.

    Returns
    -------
    Transform
        The transformed point cloud, shape (N, 3).
    """
    # homogeneous coords so the 4x4 transform can be applied in one matmul
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    return (transform @ points_homogeneous.T).T[:, :3]


def compute_overlap_ratio(
    source: PointCloud,
    target: PointCloud,
    transformation: Transform,
    threshold: float,
) -> float:
    """Compute the one-sided overlap ratio between a source and target point cloud.

    The source point cloud is aligned to the target frame using the
    ground-truth transformation, then each aligned point is matched against
    its nearest neighbor in the target cloud. The ratio is one-sided: it
    measures how much of `source` is covered by `target`, not the reverse.

    Parameters
    ----------
    source : PointCloud
        Source point cloud, shape (N, 3).
    target : PointCloud
        Target point cloud, shape (M, 3).
    transformation : Transform
        Ground-truth 4x4 homogeneous transform mapping `source` into the
        `target` frame.
    threshold : float
        Maximum nearest-neighbor distance for a source point to be
        considered overlapping. Must be > 0 and dataset-scale dependent,
        so no default is provided.

    Returns
    -------
    float
        Fraction of source points with a target neighbor within
        `threshold`, in [0, 1].
    """
    if source.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("source and target point clouds must be non-empty.")
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"source must have shape (N, 3), got {source.shape}.")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(f"target must have shape (N, 3), got {target.shape}.")
    if transformation.shape != (4, 4):
        raise ValueError(
            f"transformation must have shape (4, 4), got {transformation.shape}."
        )
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}.")

    source_aligned = apply_transform(source, transformation)

    target_tree = cKDTree(target)
    distances, _ = target_tree.query(source_aligned)

    return float(np.mean(distances <= threshold))
