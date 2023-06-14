import argparse
import sys
import time

sys.path.append("../src")
from Sensors.Lidar.src.filters_algo import LidarFilter
from Sensors.Lidar.src.pcl_reader import PCL_Reader
from Sensors.Lidar.src.visualizers import Visualizer

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_path', type=str,
                        default='record_example/Recording_192.168.0.3_2022_11_27_02_47_07.invz',
                        help='path to the record relative to this ut folder')
    parser.add_argument('--visualizer', type=str, default="pptk", help="pptk or open3d")
    parser.add_argument('--frame_num', type=int,default=1650, help='the frame number to visualize')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    print(777)
    recording_path = opt.record_path
    print(777)
    reader = PCL_Reader(recording_path, start_frame_num=opt.frame_num)
    print(777)
    lidar_filter = LidarFilter()
    visualizer = Visualizer(visualizer=opt.visualizer)

    curr_original_points = reader.read_points_cloud()
    t1 = time.time()
    curr_points, curr_ground_points, curr_clusters_list = lidar_filter.run(curr_original_points)
    print("time: ", time.time() - t1)
    visualizer.show(curr_original_points)
    visualizer.show(curr_points)
    visualizer.show(curr_ground_points)
