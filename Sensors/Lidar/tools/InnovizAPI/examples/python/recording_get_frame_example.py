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

# Innoviz API Python get frame from recording example #
from innopy.api import FileReader, FrameDataAttributes, GrabType


recording_path = '../example_recordings/Lidar_1.invz4_4'

pixel_number = 100

attr = [FrameDataAttributes(GrabType.GRAB_TYPE_MEASURMENTS_REFLECTION0)]

fr = FileReader(recording_path)
print("number of frames: ", fr.num_of_frames)

# data_attrs = fr.get_frame_data_attrs()

for i in range(20):
    res = fr.get_frame(i, attr)
    # res success is true when all required buffers are full
    if res.success is True:
        mes = res.results['GrabType.GRAB_TYPE_MEASURMENTS_REFLECTION0'][pixel_number]
        print("frame_number:", res.frame_number, "distance", mes['distance'], "reflectivity", mes['reflectivity'])

print("End of recording")