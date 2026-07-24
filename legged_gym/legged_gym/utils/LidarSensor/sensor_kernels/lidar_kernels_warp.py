import warp as wp

NO_HIT_RAY_VAL = wp.constant(1000.0)
NO_HIT_SEGMENTATION_VAL = wp.constant(wp.int32(-2))


class LidarWarpKernels:
    def __init__(self):
        pass

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_pointcloud(
        mesh_ids: wp.array(dtype=wp.uint64),
        lidar_pos_array: wp.array(dtype=wp.vec3, ndim=2),
        lidar_quat_array: wp.array(dtype=wp.quat, ndim=2),
        ray_vectors: wp.array2d(dtype=wp.vec3),
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3, ndim=4),
        local_dist: wp.array(dtype=wp.float32, ndim=4),
        pointcloud_in_world_frame: bool,
    ):

        env_id, cam_id, scan_line, point_index = wp.tid()
        mesh = mesh_ids[0]
        lidar_position = lidar_pos_array[env_id, cam_id]
        lidar_quaternion = lidar_quat_array[env_id, cam_id]
        ray_origin = lidar_position
        ray_dir = ray_vectors[scan_line, point_index]
        ray_dir = wp.normalize(ray_dir)
        ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))

        query = wp.mesh_query_ray(mesh, ray_origin, ray_direction_world, far_plane)
        if query.result:
            dist = query.t
            local_dist[env_id, cam_id, scan_line, point_index] = dist
            if pointcloud_in_world_frame:
                pixels[env_id, cam_id, scan_line, point_index] = ray_origin + dist * ray_direction_world
            else:
                pixels[env_id, cam_id, scan_line, point_index] = dist * ray_dir
        else:
            local_dist[env_id, cam_id, scan_line, point_index] = far_plane
            pixels[env_id, cam_id, scan_line, point_index] = far_plane * ray_dir

    @staticmethod
    @wp.kernel
    def draw_height_scanner_kernel(
        mesh_ids: wp.array(dtype=wp.uint64),
        lidar_pos_array: wp.array(dtype=wp.vec3, ndim=2),
        lidar_quat_array: wp.array(dtype=wp.quat, ndim=2),
        ray_origins: wp.array(dtype=wp.vec3, ndim=4),
        ray_directions: wp.array(dtype=wp.vec3, ndim=4),
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3, ndim=4),
        local_dist: wp.array(dtype=wp.float32, ndim=4),
        pointcloud_in_world_frame: bool,
    ):
        """Height scanner kernel with different ray origins"""
        env_id, cam_id, scan_line, point_index = wp.tid()
        mesh = mesh_ids[0]

        sensor_position = lidar_pos_array[env_id, cam_id]
        sensor_quaternion = lidar_quat_array[env_id, cam_id]

        ray_origin_local = ray_origins[env_id, cam_id, scan_line, point_index]
        ray_dir_local = ray_directions[env_id, cam_id, scan_line, point_index]

        qw = sensor_quaternion[3]
        qx = sensor_quaternion[0]
        qy = sensor_quaternion[1]
        qz = sensor_quaternion[2]

        yaw = wp.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        yaw_half = yaw / 2.0
        yaw_only_quat = wp.quat(0.0, 0.0, wp.sin(yaw_half), wp.cos(yaw_half))

        ray_origin_rotated = wp.quat_rotate(yaw_only_quat, ray_origin_local)
        ray_origin_world = ray_origin_rotated + sensor_position
        ray_direction_world = ray_dir_local

        query = wp.mesh_query_ray(mesh, ray_origin_world, ray_direction_world, far_plane)
        if query.result:
            dist = query.t
            local_dist[env_id, cam_id, scan_line, point_index] = dist
            if pointcloud_in_world_frame:
                pixels[env_id, cam_id, scan_line, point_index] = ray_origin_world + dist * ray_direction_world
            else:
                hit_point_world = ray_origin_world + dist * ray_direction_world
                pixels[env_id, cam_id, scan_line, point_index] = wp.quat_rotate(wp.quat_inverse(sensor_quaternion), hit_point_world - sensor_position)
        else:
            local_dist[env_id, cam_id, scan_line, point_index] = far_plane
            if pointcloud_in_world_frame:
                pixels[env_id, cam_id, scan_line, point_index] = ray_origin_world + far_plane * ray_direction_world
            else:
                far_point_world = ray_origin_world + far_plane * ray_direction_world
                pixels[env_id, cam_id, scan_line, point_index] = wp.quat_rotate(wp.quat_inverse(sensor_quaternion), far_point_world - sensor_position)
