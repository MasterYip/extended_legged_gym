import argparse
import ast
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "legged_gym" / "envs" / "el_4090" / "spider_envelop" / "el4090_spider_config.py"

CONDITION_RANGE_NAMES = [
    "left_front",
    "right_front",
    "left_mid",
    "right_mid",
    "left_back",
    "right_back",
    "forward_limit",
    "backward_limit",
]


def parse_vec(text, default=(0.0, 0.0, 0.0)):
    if not text:
        return np.array(default, dtype=float)
    return np.array([float(x) for x in text.split()], dtype=float)


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def transform_matrix(xyz, rpy):
    transform = np.eye(4)
    transform[:3, :3] = rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def axis_angle_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(4)

    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    c1 = 1.0 - c
    rot = np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )
    transform = np.eye(4)
    transform[:3, :3] = rot
    return transform


def resolve_asset_path(asset_file):
    asset_path = asset_file.format(LEGGED_GYM_ROOT_DIR=PROJECT_ROOT)
    return Path(asset_path).expanduser().resolve()


def find_class(node, name):
    for child in node.body:
        if isinstance(child, ast.ClassDef) and child.name == name:
            return child
    raise KeyError(f"Class {name} not found")


def literal_assigns(class_node):
    values = {}
    for item in class_node.body:
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
            try:
                values[item.targets[0].id] = ast.literal_eval(item.value)
            except (ValueError, SyntaxError):
                pass
    return values


def load_el4090_config(config_path=CONFIG_PATH):
    tree = ast.parse(config_path.read_text(encoding="utf-8"))
    cfg_class = find_class(tree, "El4090EnvelopCfg")
    asset_class = find_class(cfg_class, "asset")
    init_state_class = find_class(cfg_class, "init_state")

    asset = literal_assigns(asset_class)
    init_state = literal_assigns(init_state_class)
    required = {
        "asset.file": asset.get("file"),
        "init_state.pos": init_state.get("pos"),
        "init_state.default_joint_angles": init_state.get("default_joint_angles"),
        "init_state.mammal_default_joint_angles": init_state.get("mammal_default_joint_angles"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing config values in {config_path}: {missing}")
    return asset, init_state


def resolve_mesh_path(urdf_path, filename):
    if filename.startswith("package://"):
        filename = filename.removeprefix("package://")
    path = Path(filename)
    if path.is_absolute():
        return path.resolve()
    return (urdf_path.parent / path).resolve()


def load_binary_stl_vertices(path):
    return load_binary_stl_triangles(path).reshape(-1, 3)


def load_binary_stl_triangles(path):
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path} is too small to be a binary STL")

    triangle_count = struct.unpack("<I", data[80:84])[0]
    expected_size = 84 + triangle_count * 50
    if expected_size != len(data):
        raise ValueError(f"{path} does not look like a binary STL")

    triangles = np.empty((triangle_count, 3, 3), dtype=np.float32)
    offset = 84
    for tri_idx in range(triangle_count):
        floats = struct.unpack_from("<12f", data, offset)
        triangles[tri_idx] = np.array(floats[3:12]).reshape(3, 3)
        offset += 50
    return triangles


def load_ascii_stl_vertices(path):
    return load_ascii_stl_triangles(path).reshape(-1, 3)


def load_ascii_stl_triangles(path):
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"{path} has no vertices")
    if len(vertices) % 3:
        raise ValueError(f"{path} has an incomplete STL triangle list")
    return np.array(vertices, dtype=np.float32).reshape(-1, 3, 3)


def load_stl_vertices(path):
    try:
        vertices = load_binary_stl_vertices(path)
    except ValueError:
        vertices = load_ascii_stl_vertices(path)
    return np.unique(vertices, axis=0)


def load_stl_triangles(path):
    try:
        return load_binary_stl_triangles(path)
    except ValueError:
        return load_ascii_stl_triangles(path)


def parse_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = {}
    joints = []

    for link in root.findall("link"):
        name = link.attrib["name"]
        visual = link.find("visual")
        mesh_path = None
        visual_tf = np.eye(4)
        if visual is not None:
            origin = visual.find("origin")
            xyz = parse_vec(origin.attrib.get("xyz") if origin is not None else None)
            rpy = parse_vec(origin.attrib.get("rpy") if origin is not None else None)
            visual_tf = transform_matrix(xyz, rpy)

            mesh = visual.find("geometry/mesh")
            if mesh is not None:
                mesh_path = resolve_mesh_path(urdf_path, mesh.attrib["filename"])

        links[name] = {"mesh_path": mesh_path, "visual_tf": visual_tf}

    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        joints.append(
            {
                "name": joint.attrib["name"],
                "type": joint.attrib["type"],
                "parent": joint.find("parent").attrib["link"],
                "child": joint.find("child").attrib["link"],
                "origin": transform_matrix(
                    parse_vec(origin.attrib.get("xyz") if origin is not None else None),
                    parse_vec(origin.attrib.get("rpy") if origin is not None else None),
                ),
                "axis": parse_vec(
                    axis.attrib.get("xyz") if axis is not None else None,
                    default=(0.0, 0.0, 1.0),
                ),
            }
        )

    return links, joints


def compute_link_transforms(joints, joint_angles, root_transform=None):
    children = defaultdict(list)
    child_links = set()
    for joint in joints:
        children[joint["parent"]].append(joint)
        child_links.add(joint["child"])

    root_links = sorted({joint["parent"] for joint in joints} - child_links)
    if not root_links:
        raise ValueError("Could not find URDF root link")

    transforms = {root_links[0]: np.eye(4) if root_transform is None else root_transform}
    queue = deque([root_links[0]])
    while queue:
        parent = queue.popleft()
        for joint in children[parent]:
            angle_tf = np.eye(4)
            if joint["type"] in {"revolute", "continuous"}:
                angle_tf = axis_angle_matrix(joint["axis"], joint_angles.get(joint["name"], 0.0))
            transforms[joint["child"]] = transforms[parent] @ joint["origin"] @ angle_tf
            queue.append(joint["child"])
    return transforms


def transformed_vertices(vertices, transform):
    homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
    return (homogeneous @ transform.T)[:, :3]


def transformed_triangles(triangles, transform):
    flat_vertices = triangles.reshape(-1, 3)
    return transformed_vertices(flat_vertices, transform).reshape(-1, 3, 3)


def downsample(vertices, max_vertices, rng):
    if len(vertices) <= max_vertices:
        return vertices
    idx = rng.choice(len(vertices), size=max_vertices, replace=False)
    return vertices[idx]


def downsample_triangles(triangles, max_triangles, rng):
    if len(triangles) <= max_triangles:
        return triangles
    idx = rng.choice(len(triangles), size=max_triangles, replace=False)
    return triangles[idx]


def build_joint_samples(start_angles, end_angles, sample_count, mode, rng):
    joint_names = sorted(start_angles)

    if mode == "path":
        for alpha in np.linspace(0.0, 1.0, sample_count):
            yield {
                name: (1.0 - alpha) * start_angles[name] + alpha * end_angles[name]
                for name in joint_names
            }
        return

    low = {name: min(start_angles[name], end_angles[name]) for name in joint_names}
    high = {name: max(start_angles[name], end_angles[name]) for name in joint_names}

    yield dict(start_angles)
    if sample_count > 1:
        yield dict(end_angles)
    for _ in range(max(0, sample_count - 2)):
        yield {name: rng.uniform(low[name], high[name]) for name in joint_names}


def set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max((maxs - mins).max() / 2.0, 0.05)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def make_plot(title):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.set_title(title)
    ax.view_init(elev=20, azim=-135)
    return fig, ax


def update_plot(ax, points, hull_collection, scatter, title, show_points=False, envelope_alpha=0.14):
    if scatter is not None:
        scatter.remove()
        scatter = None
    if show_points:
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.25, c="#D1495B", alpha=0.08)

    if hull_collection is not None:
        hull_collection.remove()
        hull_collection = None

    if len(points) >= 4:
        try:
            hull = ConvexHull(points)
            faces = [points[simplex] for simplex in hull.simplices]
            hull_collection = Poly3DCollection(
                faces,
                alpha=envelope_alpha,
                facecolor="#00A6A6",
                edgecolor="#064E5B",
                linewidth=0.10,
            )
            ax.add_collection3d(hull_collection)
        except QhullError:
            pass

    ax.set_title(title)
    set_axes_equal(ax, points)
    ax.figure.canvas.draw_idle()
    ax.figure.canvas.flush_events()
    return hull_collection, scatter


def get_region_lateral_ranges(points):
    x_min = float(points[:, 0].min())
    x_max = float(points[:, 0].max())
    x_span = x_max - x_min
    if x_span <= 0.0:
        raise ValueError("Envelope points have no x extent, cannot split front/mid/back regions.")

    back_end = x_min + x_span / 3.0
    front_start = x_min + 2.0 * x_span / 3.0
    regions = {
        "back": points[points[:, 0] <= back_end],
        "mid": points[(points[:, 0] > back_end) & (points[:, 0] < front_start)],
        "front": points[points[:, 0] >= front_start],
    }

    ranges = {}
    for region_name, region_points in regions.items():
        if len(region_points) == 0:
            raise ValueError(f"No envelope points found in {region_name} region.")
        y_min = float(region_points[:, 1].min())
        y_max = float(region_points[:, 1].max())
        ranges[f"left_{region_name}"] = (0.0, y_max)
        ranges[f"right_{region_name}"] = (y_min, 0.0)
    return ranges


def get_condition_ranges(points):
    x_min = float(points[:, 0].min())
    x_max = float(points[:, 0].max())
    ranges = get_region_lateral_ranges(points)
    ranges.update(
        {
        "forward_limit": (0.0, x_max),
        "backward_limit": (x_min, 0.0),
        }
    )
    return ranges


def print_condition_ranges(points):
    ranges = get_condition_ranges(points)
    print("Condition ranges:")
    for name in CONDITION_RANGE_NAMES:
        low, high = ranges[name]
        print(f"  {name}: [{low:.6f}, {high:.6f}]")


def load_mesh_vertices(links, max_vertices_per_link, rng):
    mesh_vertices = {}
    for link_name, link in links.items():
        mesh_path = link["mesh_path"]
        if mesh_path is None:
            continue
        vertices = load_stl_vertices(mesh_path)
        mesh_vertices[link_name] = downsample(vertices, max_vertices_per_link, rng)
    return mesh_vertices


def is_body_or_leg_link(link_name):
    return link_name == "BASE" or link_name.endswith(("_HIP", "_THIGH", "_SHANK", "_FOOT"))


def load_robot_triangles(links, max_triangles_per_link, rng):
    mesh_triangles = {}
    for link_name, link in links.items():
        mesh_path = link["mesh_path"]
        if mesh_path is None or not is_body_or_leg_link(link_name):
            continue
        triangles = load_stl_triangles(mesh_path)
        mesh_triangles[link_name] = downsample_triangles(triangles, max_triangles_per_link, rng)
    return mesh_triangles


def sample_robot_points(links, joints, mesh_vertices, joint_angles, root_tf):
    link_transforms = compute_link_transforms(joints, joint_angles, root_transform=root_tf)
    frame_points = []
    for link_name, vertices in mesh_vertices.items():
        if link_name not in link_transforms:
            continue
        visual_tf = links[link_name]["visual_tf"]
        frame_points.append(transformed_vertices(vertices, link_transforms[link_name] @ visual_tf))
    return np.vstack(frame_points)


def draw_mammal_robot(ax, links, joints, mesh_triangles, joint_angles, root_tf):
    link_transforms = compute_link_transforms(joints, joint_angles, root_transform=root_tf)
    collections = []
    for link_name in sorted(mesh_triangles):
        if link_name not in link_transforms:
            continue
        triangles = transformed_triangles(
            mesh_triangles[link_name],
            link_transforms[link_name] @ links[link_name]["visual_tf"],
        )
        is_base = link_name == "BASE"
        collection = Poly3DCollection(
            triangles,
            alpha=0.78 if is_base else 0.88,
            facecolor="#7A8288" if is_base else "#D98C2B",
            edgecolor="#40464A" if is_base else "#6C3D0E",
            linewidth=0.05,
        )
        ax.add_collection3d(collection)
        collections.append(collection)
    return collections


def main():
    parser = argparse.ArgumentParser(
        description="Show the el_4090 shape envelope between default and mammal poses."
    )
    parser.add_argument("--samples", type=int, default=120, help="Internal pose count used to build the envelope.")
    parser.add_argument(
        "--mode",
        choices=("box", "path"),
        default="box",
        help="'box': each joint independently ranges between the two configs; 'path': one shared interpolation alpha.",
    )
    parser.add_argument("--max-vertices-per-link", type=int, default=600, help="Downsample mesh vertices for speed.")
    parser.add_argument(
        "--max-triangles-per-link",
        type=int,
        default=700,
        help="Downsample rendered robot triangles for display speed.",
    )
    parser.add_argument("--seed", type=int, default=4090, help="Random seed used by --mode box.")
    parser.add_argument("--output", default=None, help="Optional PNG path. If omitted, only the live window is shown.")
    parser.add_argument("--no-show", action="store_true", help="Build the plot without opening the interactive window.")
    parser.add_argument("--no-robot", action="store_true", help="Only draw the envelope surface.")
    parser.add_argument("--envelope-alpha", type=float, default=0.14, help="Envelope surface transparency.")
    parser.add_argument("--show-points", action="store_true", help="Also draw the internal sampled mesh vertices.")
    parser.add_argument("--live", action="store_true", help="Refresh the envelope while it is being built.")
    parser.add_argument("--refresh-every", type=int, default=5, help="Live refresh interval in internal poses.")
    args = parser.parse_args()

    asset_cfg, init_state = load_el4090_config()
    start_angles = init_state["default_joint_angles"]
    end_angles = init_state["mammal_default_joint_angles"]
    missing = sorted(set(start_angles) ^ set(end_angles))
    if missing:
        raise ValueError(f"The two configs do not define the same joints: {missing}")

    urdf_path = resolve_asset_path(asset_cfg["file"])
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    rng = np.random.RandomState(args.seed)
    links, joints = parse_urdf(urdf_path)
    mesh_vertices = load_mesh_vertices(links, args.max_vertices_per_link, rng)
    robot_triangles = {}
    if not args.no_robot:
        robot_triangles = load_robot_triangles(links, args.max_triangles_per_link, rng)
    root_tf = transform_matrix(np.array(init_state["pos"], dtype=float), np.zeros(3))

    title = "el_4090 shape envelope"
    if args.live and not args.no_show:
        plt.ion()
    fig, ax = make_plot(title)

    all_points = []
    hull_collection = None
    scatter = None
    for sample_idx, joint_angles in enumerate(
        build_joint_samples(start_angles, end_angles, args.samples, args.mode, rng),
        start=1,
    ):
        all_points.append(sample_robot_points(links, joints, mesh_vertices, joint_angles, root_tf))
        if args.live and (
            sample_idx == 1 or sample_idx == args.samples or sample_idx % max(1, args.refresh_every) == 0
        ):
            points = np.vstack(all_points)
            live_title = f"{title}: building {sample_idx}/{args.samples}"
            hull_collection, scatter = update_plot(
                ax,
                points,
                hull_collection,
                scatter,
                live_title,
                show_points=args.show_points,
                envelope_alpha=args.envelope_alpha,
            )
            if not args.no_show:
                plt.pause(0.001)

    points = np.vstack(all_points)
    hull_collection, scatter = update_plot(
        ax,
        points,
        hull_collection,
        scatter,
        title,
        show_points=args.show_points,
        envelope_alpha=args.envelope_alpha,
    )
    if not args.no_robot:
        draw_mammal_robot(ax, links, joints, robot_triangles, end_angles, root_tf)
        ax.figure.canvas.draw_idle()
        ax.figure.canvas.flush_events()
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=220)
        print(f"Saved envelope image: {args.output}")

    print(f"URDF: {urdf_path}")
    print(f"Envelope points: {len(points):,}")
    print_condition_ranges(points)

    if not args.no_show:
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
