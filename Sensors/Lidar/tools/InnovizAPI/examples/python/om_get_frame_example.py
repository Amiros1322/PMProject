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

# Innoviz API Python get frame from device example #

from __future__ import print_function
from innopy.api import DeviceInterface, FrameDataAttributes, GrabType
import time


class NewFrameHandler:
    def __init__(self):
        self.attr = [FrameDataAttributes(GrabType.GRAB_TYPE_MEASURMENTS_REFLECTION0)]
        config_files_path = '../lidar_configuration_files'
        self.di = DeviceInterface(config_file_name=config_files_path+'/om_config.json', is_connect=False)

        for i in range(len(self.attr)):
            self.di.activate_buffer(self.attr[i], True)

    def callback(self, h):
        pixel_number = 20
        try:
            res = self.di.get_frame(self.attr)
            if res.success:
                mes = res.results['GrabType.GRAB_TYPE_MEASURMENTS_REFLECTION0'][pixel_number]
                print("frame_number:", res.frame_number,
                      "distance", mes['distance'],
                      "reflectivity", mes['reflectivity'])

        except:
            print('NewFrameHandler: failed executing frame')

    def print_data_frames(self):
        self.di.register_new_frame_callback(self.callback)
        time.sleep(10)
        self.di.unregister_new_frame_callback()

    def finish(self):
        self.di.device_close()

def main():

    fh = NewFrameHandler()
    fh.print_data_frames()
    print("The End:)")
    fh.finish()

if __name__ == '__main__':
    main()


