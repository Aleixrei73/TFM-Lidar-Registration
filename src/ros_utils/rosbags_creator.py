"""
Creates a ROS1 (.bag) file from a V2X-Real dataset, containing:
  - /pointcloud_1  (sensor_msgs/PointCloud2)  -- infrastructure point cloud
  - /pointcloud_2  (sensor_msgs/PointCloud2)  -- vehicle point cloud
  - /tf            (tf2_msgs/TFMessage)       -- transform infra -> vehicle

Uses the `rosbags` pure-Python library -- no ROS1/ROS2 installation needed at all.

Install:
    pip install rosbags numpy scipy
"""

from __future__ import annotations

from typing import List, Literal, Union

import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from scipy.spatial.transform import Rotation

from datasets.v2x_real import V2XReal

# ROS1_NOETIC gives message definitions matching ROS1 conventions
# (e.g. Header still has 'seq', unlike ROS2's Header).
typestore = get_typestore(Stores.ROS1_NOETIC)

# rosbags' ROS1_NOETIC store does not ship tf2_msgs (only the ROS2 stores
# do), so it must be registered by hand from its .msg definition.
typestore.register(get_types_from_msg(
    'geometry_msgs/TransformStamped[] transforms', 'tf2_msgs/msg/TFMessage'
))

Header = typestore.types['std_msgs/msg/Header']
Time = typestore.types['builtin_interfaces/msg/Time']
PointCloud2 = typestore.types['sensor_msgs/msg/PointCloud2']
PointField = typestore.types['sensor_msgs/msg/PointField']
TransformStamped = typestore.types['geometry_msgs/msg/TransformStamped']
Transform = typestore.types['geometry_msgs/msg/Transform']
Vector3 = typestore.types['geometry_msgs/msg/Vector3']
Quaternion = typestore.types['geometry_msgs/msg/Quaternion']
TFMessage = typestore.types['tf2_msgs/msg/TFMessage']


def make_time(seconds: float):
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1e9))
    return Time(sec=sec, nanosec=nanosec)


def make_pointcloud2(points_xyz: np.ndarray, frame_id: str, seq: int, stamp):
    """points_xyz: Nx3 float32 numpy array."""
    n_points = points_xyz.shape[0]
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    header = Header(seq=seq, stamp=stamp, frame_id=frame_id)
    raw = points_xyz.astype(np.float32).tobytes()
    return PointCloud2(
        header=header,
        height=1,
        width=n_points,
        fields=fields,
        is_bigendian=False,
        point_step=12,  # 3 x float32
        row_step=12 * n_points,
        data=np.frombuffer(raw, dtype=np.uint8),
        is_dense=True,
    )


def make_transform_stamped(parent_frame, child_frame, translation, rotation_quat_xyzw, seq, stamp):
    tx, ty, tz = translation
    qx, qy, qz, qw = rotation_quat_xyzw
    header = Header(seq=seq, stamp=stamp, frame_id=parent_frame)
    return TransformStamped(
        header=header,
        child_frame_id=child_frame,
        transform=Transform(
            translation=Vector3(x=tx, y=ty, z=tz),
            rotation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        ),
    )


def create_rosbag(
    transforms: np.ndarray,
    indices: Union[List[int], Literal["all"]],
    dataset: V2XReal,
    bag_path: str,
    fps: float,
) -> None:
    """
    Write a ROS1 bag with infra/vehicle point clouds and their transform.

    Parameters
    ----------
    transforms : np.ndarray
        Array of shape (M, 4, 4) with homogeneous transformation matrices
        (infra -> vehicle), given in the same order as the resolved
        dataset indices.
    indices : list of int or "all"
        Dataset indices to include. If "all", every index in ``dataset``
        is used, in order.
    dataset : V2XReal
        Dataset to read point cloud pairs from.
    bag_path : str
        Output path for the written ``.bag`` file.
    fps : float
        Frame rate, in Hz, used to space consecutive messages in time.

    Raises
    ------
    ValueError
        If ``transforms`` does not have shape (M, 4, 4).
    """
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(
            f"transforms must have shape (M, 4, 4), got {transforms.shape}"
        )

    resolved_indices = range(len(dataset)) if indices == "all" else indices

    if indices == "all":
        assert len(transforms) >= len(dataset), (
            f"len(transforms)={len(transforms)} must be >= len(dataset)={len(dataset)}"
        )
    else:
        assert len(transforms) >= len(indices), (
            f"len(transforms)={len(transforms)} must be >= len(indices)={len(indices)}"
        )

    dt = 1.0 / fps

    with Writer(bag_path) as writer:
        conn_pc1 = writer.add_connection('/pointcloud_1', PointCloud2.__msgtype__, typestore=typestore)
        conn_pc2 = writer.add_connection('/pointcloud_2', PointCloud2.__msgtype__, typestore=typestore)
        conn_tf = writer.add_connection('/tf', TFMessage.__msgtype__, typestore=typestore)

        for i, idx in enumerate(resolved_indices):
            sample = dataset[idx]

            t_seconds = i * dt
            stamp = make_time(t_seconds)
            timestamp_ns = int(round(t_seconds * 1e9))  # rosbags timestamps are int nanoseconds

            infra_pts = sample["infra_pts"].numpy()
            vehicle_pts = sample["vehicle_pts"].numpy()

            pc1_msg = make_pointcloud2(infra_pts, 'infra', i, stamp)
            writer.write(conn_pc1, timestamp_ns, typestore.serialize_ros1(pc1_msg, PointCloud2.__msgtype__))

            pc2_msg = make_pointcloud2(vehicle_pts, 'vehicle', i, stamp)
            writer.write(conn_pc2, timestamp_ns, typestore.serialize_ros1(pc2_msg, PointCloud2.__msgtype__))

            T = transforms[i]
            translation = T[:3, 3]
            rotation_quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
            transform = make_transform_stamped(
                'vehicle', 'infra', translation, rotation_quat_xyzw, i, stamp
            )
            tf_msg = TFMessage(transforms=[transform])
            writer.write(conn_tf, timestamp_ns, typestore.serialize_ros1(tf_msg, TFMessage.__msgtype__))
