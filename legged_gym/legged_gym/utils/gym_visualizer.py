from isaacgym import gymtorch, gymapi, gymutil
import math
import numpy as np
from sympy import per


class WireframeArrowGeometry(gymutil.LineGeometry):

    def __init__(self, radius=0.02, height=0.1, num_segments=8, length=1.0, pose=None, color=None):
        if color is None:
            color = (1, 0, 0)

        num_lines = 2 * num_segments + 1

        verts = np.empty((num_lines, 2), gymapi.Vec3.dtype)
        colors = np.empty(num_lines, gymapi.Vec3.dtype)
        idx = 0

        # Apex of the cone
        apex = (0.0, 0.0, height+length)  # Convert to tuple
        step = 2 * math.pi / num_segments
        for i in range(num_segments):
            theta1 = i * step
            theta2 = (i + 1) % num_segments * step

            # First point on the base
            x1 = radius * math.cos(theta1)
            y1 = radius * math.sin(theta1)
            z1 = length

            # Second point on the base
            x2 = radius * math.cos(theta2)
            y2 = radius * math.sin(theta2)
            z2 = length

            # Line from the apex to the first point on the base
            verts[idx][0] = apex  # Apex as a tuple
            verts[idx][1] = (x1, y1, z1)
            colors[idx] = color

            idx += 1

            # Line between two consecutive points on the base
            verts[idx][0] = (x1, y1, z1)
            verts[idx][1] = (x2, y2, z2)
            colors[idx] = color

            idx += 1
        # Main line
        verts[idx][0] = apex
        verts[idx][1] = (0, 0, 0)
        colors[idx] = color

        if pose is None:
            self.verts = verts
        else:
            self.verts = pose.transform_points(verts)

        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors


class GymVisualizer:
    """Isaac Gym Visualizer
    NOTE: All geometries are drawn based on lines(meshes) in Isaac Gym.
    """

    def __init__(self, gym, sim, viewer, envs):
        self.gym = gym
        self.sim = sim
        self.viewer = viewer
        self.envs = envs
        self._init_geometries()

    def _init_geometries(self):
        """预定义可复用的几何模板"""
        # 箭头参数：长度0.1m，头部占比25%，红色
        self.arrow_geom = WireframeArrowGeometry(
            length=0.1,
            radius=0.01,
            # head_length_ratio=0.25,
            num_segments=8,
            color=(1, 0, 0)
        )

        # 点参数：半径0.02m，绿色
        # self.point_geom = gymutil.WireframeSphereGeometry(1, 4, 4, None, color=(0, 1, 0))

        # self.sphere_geom = gymutil.WireframeSphereGeometry(2, 4, 4, None, color=(0, 0, 1))

    def clear(self):
        """清除所有可视化元素"""
        self.gym.clear_lines(self.viewer)

    # Basic drawing functions
    def draw_line(self, env_idx, points, color=(0, 0, 1)):
        """
        Draw a polyline.
        :param env_idx: Environment index
        :param points: Sequence of consecutive points [[x1, y1, z1], [x2, y2, z2], ...]
        :param color: Line color
        """
        vertices = np.array(points, dtype=np.float32).flatten()
        self.gym.add_lines(
            self.viewer,
            self.envs[env_idx],
            len(points)-1,  # Number of line segments
            vertices,
            [color]*len(points)  # Vertex colors
        )

    def draw_boldline(self, env_idx, points, rad=0.02, resolution=8, color=(0, 0, 1)):
        """
        Draw a bold line.
        :param env_idx: Environment index
        :param points: Sequence of consecutive points [[x1, y1, z1], [x2, y2, z2], ...]
        :param rad: Line radius
        :param color: Line color
        """
        # vertices = np.array(points, dtype=np.float32).flatten()
        vertices_mat = np.array(points, dtype=np.float32)
        dir_mat = np.diff(vertices_mat, axis=0)
        dir_mat = np.concatenate([dir_mat, dir_mat[-1:]], axis=0)
        perp_dir1_mat = np.cross(dir_mat, np.random.rand(3))
        perp_dir1_mat = perp_dir1_mat / np.linalg.norm(perp_dir1_mat, axis=1)[:, None]
        perp_dir2_mat = np.cross(dir_mat, perp_dir1_mat)
        perp_dir2_mat = perp_dir2_mat / np.linalg.norm(perp_dir2_mat, axis=1)[:, None]
        for i in range(resolution):
            ver = (vertices_mat
                   + perp_dir1_mat * rad * np.cos(2 * np.pi * i / resolution)
                   + perp_dir2_mat * rad * np.sin(2 * np.pi * i / resolution))
            ver = np.array(ver, dtype=np.float32).flatten()

            self.gym.add_lines(
                self.viewer,
                self.envs[env_idx],
                len(points)-1,  # Number of line segments
                ver,
                [color]*len(points),  # Vertex colors
            )

    def draw_point(self, env_idx, position, color=None, size=0.05):
        """
        在指定位置绘制点状球体
        :param env_idx: 环境索引
        :param position: 坐标[x,y,z] 
        :param color: 可选颜色向量(RGB)
        """
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*position)

        geom = self.point_geom if color is None else \
            gymutil.WireframeSphereGeometry(size, 4, 4, None, color=color)

        gymutil.draw_lines(
            geom,
            self.gym,
            self.viewer,
            self.envs[env_idx],
            pose
        )

    def draw_points(self, env_idx, positions, color=None, size=0.05):
        """
        在指定位置绘制多个点状球体
        :param env_idx: 环境索引
        :param positions: 位置列表[[x1,y1,z1],[x2,y2,z2],...]
        :param color: 可选颜色向量(RGB)
        """
        for pos in positions:
            self.draw_point(env_idx, pos, color, size)

    def draw_arrow(self, env_idx, start, end, width=0.02, color=(0, 0, 1)):
        """
        在指定环境绘制方向箭头
        :param env_idx: 环境索引
        :param start: 起始点[x,y,z]
        :param end: 终点[x,y,z]
        :param color: 可选颜色向量(RGB)
        """
        self.draw_boldline(env_idx, [start, end], rad=width, color=color)

        len = np.linalg.norm(np.array(end) - np.array(start))
        dir_origin = gymapi.Vec3(0, 0, 1)
        direction = gymapi.Vec3(*(np.array(end) - np.array(start)))
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*start)
        axis = dir_origin.cross(direction)
        # Avoid division by zero
        angle = math.acos(
            dir_origin.dot(direction) / (dir_origin.length() * direction.length())
            ) if dir_origin.length() * direction.length() != 0 else 0
        pose.r = gymapi.Quat.from_axis_angle(axis, angle)
        geom = WireframeArrowGeometry(radius=width*3, length=len, color=color)
        gymutil.draw_lines(
            geom,
            self.gym,
            self.viewer,
            self.envs[env_idx],
            pose
        )

    def draw_frame_from_pose(self, env_idx, pose: gymapi.Transform, width=0.02, length=0.6):
        """
        在指定环境绘制坐标系
        :param env_idx: 环境索引
        :param pose: 坐标系位姿
        :param width: 坐标轴宽度
        """
        start = [pose.p.x, pose.p.y, pose.p.z]
        # 旋转到世界坐标系下的方向
        axes = [
            pose.r.rotate(gymapi.Vec3(1, 0, 0)),
            pose.r.rotate(gymapi.Vec3(0, 1, 0)),
            pose.r.rotate(gymapi.Vec3(0, 0, 1))
        ]
        ends = [[pose.p.x+axes[i].x*length,
                 pose.p.y+axes[i].y*length,
                 pose.p.z+axes[i].z*length,] for i in range(3)]
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        for i in range(3):
            self.draw_arrow(env_idx, start, ends[i], width, colors[i])

    def draw_frame_from_quat(self, env_idx, quat, position, width=0.02, length=0.6):
        """
        在指定环境绘制坐标系
        :param env_idx: 环境索引
        :param quat: 四元数[x,y,z,w]
        :param position: 位置[x,y,z]
        :param width: 坐标轴宽度
        """
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*position)
        pose.r = gymapi.Quat(*quat)
        self.draw_frame_from_pose(env_idx, pose, width, length)
