import warp as wp


NO_HIT_RAY_VAL = wp.constant(1000.0)


class LidarWarpKernels:
    @staticmethod
    @wp.kernel
    def draw_pointcloud(
        mesh_ids: wp.array(dtype=wp.uint64),
        lidar_pos_array: wp.array(dtype=wp.vec3, ndim=2),
        lidar_quat_array: wp.array(dtype=wp.quat, ndim=2),
        ray_vectors: wp.array2d(dtype=wp.vec3),
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3, ndim=4),
        local_dist: wp.array(dtype=wp.float32, ndim=4),
        pointcloud_in_world_frame: bool,
    ):
        env_id, sensor_id, scan_line, point_index = wp.tid()

        mesh = mesh_ids[0]
        lidar_position = lidar_pos_array[env_id, sensor_id]
        lidar_quaternion = lidar_quat_array[env_id, sensor_id]
        ray_dir = wp.normalize(ray_vectors[scan_line, point_index])
        ray_direction_world = wp.normalize(wp.quat_rotate(lidar_quaternion, ray_dir))

        local_dist[env_id, sensor_id, scan_line, point_index] = NO_HIT_RAY_VAL
        pixels[env_id, sensor_id, scan_line, point_index] = wp.vec3(0.0, 0.0, 0.0)

        query = wp.mesh_query_ray(mesh, lidar_position, ray_direction_world, far_plane)
        if query.result:
            dist = query.t
            local_dist[env_id, sensor_id, scan_line, point_index] = dist
            if pointcloud_in_world_frame:
                pixels[env_id, sensor_id, scan_line, point_index] = (
                    lidar_position + dist * ray_direction_world
                )
            else:
                pixels[env_id, sensor_id, scan_line, point_index] = dist * ray_dir
