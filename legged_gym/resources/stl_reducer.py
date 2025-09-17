
import glob

import trimesh
import os
# Directory Management
try:
    # Run in Terminal
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except:
    # Run in ipykernel & interactive
    ROOT_DIR = os.getcwd()


def reduce_stl_file(input_path, output_path, percent=0.1, aggression=5):
    """
    Reduces the number of faces in an STL file using trimesh.

    Parameters:
        input_path (str): Path to the original STL file.
        output_path (str): Path where the reduced STL file will be saved.
        target_face_count_ratio (float): Ratio of the target face count relative to the original (default is 0.1).
    
    Returns:
        None
    """
    try:
        # Load the original STL file
        mesh = trimesh.load(input_path)
        # Check if the mesh has faces
        if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            raise ValueError(f"The loaded mesh from {input_path} does not contain any faces.")
        # Calculate the target number of faces
        # target_faces = int(len(mesh.faces) * target_face_count_ratio)
        # Decimate the mesh
        simplified_mesh = mesh.simplify_quadric_decimation(percent=percent, aggression=aggression)
        # Export the simplified mesh to a new STL file
        simplified_mesh.export(output_path)
        print(f"Simplified mesh successfully exported to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':

    reduction_factor = 0.02
    folder_dir = os.path.join(ROOT_DIR, 'robots')
    search_str = folder_dir + '/**/*.STL'
    search_str1 = folder_dir + '/**/base_link.STL'
    search_str2 = folder_dir + '/**/BASE.STL'
    # Glob through the robot folder to find all STL files
    stl_files = glob.glob(search_str1, recursive=True) + glob.glob(search_str2, recursive=True) + glob.glob(search_str, recursive=True)
    # Iterate over each STL file and reduce its vertices
    for stl_file in stl_files:
        # Reduce the vertices of the STL file
        reduce_stl_file(stl_file, stl_file, reduction_factor)
