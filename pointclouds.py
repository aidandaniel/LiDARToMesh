import numpy as np
import laspy as lp
import open3d as o3d

pcd_path = r"C:\Users\decke\Downloads\antartica.laz" 
pointcloud = lp.read(pcd_path)
points = np.asarray(pointcloud.xyz)

points -= np.min(points, axis=0)

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(points)


o3d.visualization.draw_geometries([pcd], window_name="Lidar Visualization Test")


downpcd = pcd.voxel_down_sample(voxel_size=1)
original_count = len(pcd.points) 
downsampled_count = len(downpcd.points)

compression_ratio = (1 - (downsampled_count / original_count)) * 100
print(f"Reduction: {compression_ratio:.2f}%")
center = downpcd.get_center()


downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.7,
                                  front=[-1,-1,1],
                                  lookat=center,
                                  up=[0, 0 ,1],
                                  point_show_normal=True)

downpcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, 1.0]))


#poisson surface
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(downpcd, depth=13)


mesh.compute_vertex_normals()
normal_vector = np.asarray(mesh.vertex_normals)


#Safe and not safe robotics map


up_vector = np.array([0,0,1])
#Both are unit vectors
cos_theta = np.dot(normal_vector, up_vector) 
#bounding and defining the angles for every vertix
angles = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * (180 / np.pi) 
#boolean check against the angles array
hazard_mask = angles > 20.0 #

#replaces the normal vector array with a new array of zeros
sim_colors = np.zeros_like(normal_vector) 
sim_colors[hazard_mask] = [1, 0, 0]   
sim_colors[~hazard_mask] = [0, 0.8, 0]
mesh.vertex_colors = o3d.utility.Vector3dVector(sim_colors)

o3d.visualization.draw_geometries([mesh], window_name="Robotic Traversability Simulation")





o3d.visualization.draw_geometries([mesh],
                                  zoom=0.7,
                                  front=[-1,-1,1],
                                  lookat=center,
                                  up=[0, 0 ,1]) 


