# Copyright 2021 Innoviz Technologies
#
# Licensed under the Innoviz Open Dataset License Agreement (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/InnovizTechnologies/InnovizAPI/blob/main/LICENSE.md
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Innoviz API python record example #

from innopy.api import DeviceInterface

rec_length_sec = 20

def main():
    config_files_path = '../lidar_configuration_files'
    di = DeviceInterface(config_file_name=config_files_path+'/om_config.json', is_connect=False)

    # record all channels defined in lidar.json
    di.record(rec_length_sec, "pcl_recording")
    print(f"Finished recording {rec_length_sec} second")
    di.device_close()


if __name__ == '__main__':
    main()
