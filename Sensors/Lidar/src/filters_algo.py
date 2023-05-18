import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor, LinearRegression


class LidarFilter:
    def __init__(self, ground_model=RANSACRegressor(max_trials=200, residual_threshold=10),
                 cluster_model=DBSCAN(eps=20, min_samples=15)):
        self.ground_model = ground_model
        self.cluster_model = cluster_model

        self.filter_tests = [self.length_test, self.width_test, self.height_test,self.min_height_test,self.number_of_pts_test
                             ,self.cone_shape_egv_test]
        self.points = None
        self.ground_points = None
        self.clusters_list = None
        self.height_ground = None

    def length_test(self, points, min_val=5, max_val=60):
        x_coords = points[:, 0]
        if min_val < np.abs(np.max(x_coords) - np.min(x_coords)) < max_val:
            return True
        return False

    def width_test(self, points, min_val=5, max_val=70):
        y_coords = points[:, 1]
        if min_val < np.abs(np.max(y_coords) - np.min(y_coords)) < max_val:
            return True
        return False  # didn't pass the filter, then return False

    def height_test(self, points, min_val=10, max_val=80):
        z_coords = points[:, 2]
        if min_val < np.abs(np.max(z_coords) - np.min(z_coords)) < max_val:
            return True
        return False

    def min_height_test(self, points, min_diff=20):
        x_coords = points[:, 0]
        z_coords = points[:, 2]
        if np.min(z_coords) < (self.height_ground + min_diff) or \
                (np.min(x_coords) > 1250 and np.min(z_coords) < (self.height_ground+(min_diff*2))):
            return True
        return False

    def number_of_pts_test(self, points, min_pts=10,max_pts=1500):
        if min_pts < points.shape[0] < max_pts:
            return True
        return False

    def cone_shape_egv_test(self,points):
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        if eigenvalues[0] > eigenvalues[1] * 1.5 and eigenvalues[0] > eigenvalues[2] * 2.5:
            return True
        else:
            return False

    def filter_fov(self):
        # Filter the points that are out of the FOV
        # filter x axis (how far the car can see front and back)
        self.points = self.points[self.points[:, 0] > 200]
        self.points = self.points[self.points[:, 0] < 3500]

        # filter y axis (how far the car can see left and right)
        self.points = self.points[self.points[:, 1] > -2000]
        self.points = self.points[self.points[:, 1] < 2000]

        # filter z axis (how far the car can see up and down)
        self.points = self.points[self.points[:, 2] > -2000]
        self.points = self.points[self.points[:, 2] < 1]

    def filter_ground(self, pointcloud, height_threshold=15, model_option="RANSAC"):
        # Extract the x, y, and z coordinates of the points
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]

        self.ground_model.fit(np.column_stack((x, y)), z)
        self.height_ground = self.ground_model.estimator_.intercept_ + height_threshold

        # Calculate the height of each point relative to the ground plane
        z_pred = self.ground_model.predict(np.column_stack((x, y)))
        dz = z - z_pred

        # Filter out points that are below the height threshold
        ground_points = pointcloud[dz < np.min(dz) + height_threshold]
        non_ground_points = pointcloud[dz > np.min(dz) + height_threshold]
        self.points, self.ground_points = non_ground_points, ground_points

    def filter_clusters(self):
        self.points = np.empty((1,3)) # initialize the final points list.
        filtered_clusters = []
        vaild_cluster = True
        for cluster_num, cluster in enumerate(self.clusters_list):
            for filter_test in self.filter_tests:
                if not filter_test(cluster):  # if the cluster didn't pass the test then continue to next cluster
                    # print("cluster number ", cluster_num, " didn't pass the filter test ", filter_test)
                    vaild_cluster = False
                    break
            if vaild_cluster:
                # if the cluster passed all the filter tests then add it to the final points list and to the clusters
                self.points = np.concatenate((self.points, cluster))
                filtered_clusters.append(cluster)
            vaild_cluster=True
        # self.clusters_list = np.array(filtered_clusters)

    def points_to_clusters(self):
        cluster_labels = self.cluster_model.fit_predict(self.points)
        # Create a list to hold the points in each cluster
        unique_labels = np.unique(cluster_labels)
        cluster_point_lists = []
        # Iterate over the unique cluster labels
        for label in unique_labels:
            # Find the indices of the points in the current cluster
            cluster_indices = np.where(cluster_labels == label)[0]
            # Get the points in the current cluster
            cluster_points = self.points[cluster_indices]
            # Add the points to the list of cluster points
            cluster_point_lists.append(cluster_points)
        self.clusters_list = cluster_point_lists

    def run(self,points):
        self.points = points
        self.filter_fov()
        self.filter_ground(self.points)
        self.points_to_clusters()
        self.filter_clusters()


        # self.points are the final points after filtering
        # self.clusters_list are the final clusters after filtering (contains all the self.points)
        # self.ground_points are the ground points after filtering only the ground for debugging.
        return self.points, self.ground_points, self.clusters_list
