"""Open3D-based visualizers for point cloud registration results."""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d
import torch

logger = logging.getLogger(__name__)

_SRC_COLOR = (1.0, 0.706, 0.0)
_TGT_COLOR = (0.0, 0.651, 0.929)


def _to_numpy(array: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a numpy array or torch tensor to a numpy array.

    Parameters
    ----------
    array : np.ndarray or torch.Tensor
        Input array or tensor.

    Returns
    -------
    np.ndarray
        `array` as a numpy array, detached from any computation graph.
    """
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _to_point_cloud(
    points: np.ndarray | torch.Tensor, color: tuple[float, float, float]
) -> o3d.geometry.PointCloud:
    """Build a uniformly colored Open3D point cloud from an (N, 3) array.

    Parameters
    ----------
    points : np.ndarray or torch.Tensor
        Array of shape (N, 3) with point XYZ coordinates.
    color : tuple of float
        RGB color in [0, 1] applied to every point.

    Returns
    -------
    o3d.geometry.PointCloud
        The resulting point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_to_numpy(points).astype(np.float64))
    pcd.paint_uniform_color(color)
    return pcd


def visualize_registration(
    src_pts: np.ndarray | torch.Tensor,
    tgt_pts: np.ndarray | torch.Tensor,
    T_gt: np.ndarray | torch.Tensor,
    T_pred: np.ndarray | torch.Tensor,
    window_name: str = "Registration",
) -> None:
    """Visualize a registration result before and after alignment.

    Opens three Open3D windows in sequence:
    1. The raw, untransformed source and target point clouds overlaid,
       to show their initial misalignment.
    2. The source registered with `T_gt`, overlaid on the target.
    3. The source registered with `T_pred`, overlaid on the target.

    Parameters
    ----------
    src_pts : np.ndarray or torch.Tensor
        Array of shape (N, 3) with source point cloud coordinates.
    tgt_pts : np.ndarray or torch.Tensor
        Array of shape (M, 3) with target point cloud coordinates.
    T_gt : np.ndarray or torch.Tensor
        4x4 ground-truth transform mapping `src_pts` into the target frame.
    T_pred : np.ndarray or torch.Tensor
        4x4 predicted transform mapping `src_pts` into the target frame.
    window_name : str
        Base name used for all three visualization windows.

    Notes
    -----
    Each window blocks until closed, since Open3D's classic visualizer
    runs its own event loop per call to `draw_geometries`.
    """
    src = _to_point_cloud(src_pts, _SRC_COLOR)
    tgt = _to_point_cloud(tgt_pts, _TGT_COLOR)

    logger.info("Showing raw (unregistered) source and target point clouds.")
    o3d.visualization.draw_geometries([src, tgt], window_name=f"{window_name} - raw")

    src_gt = o3d.geometry.PointCloud(src)
    src_gt.transform(_to_numpy(T_gt).astype(np.float64))
    logger.info("Showing source registered with T_gt.")
    o3d.visualization.draw_geometries(
        [src_gt, tgt], window_name=f"{window_name} - T_gt"
    )

    src_pred = o3d.geometry.PointCloud(src)
    src_pred.transform(_to_numpy(T_pred).astype(np.float64))
    logger.info("Showing source registered with T_pred.")
    o3d.visualization.draw_geometries(
        [src_pred, tgt], window_name=f"{window_name} - T_pred"
    )
