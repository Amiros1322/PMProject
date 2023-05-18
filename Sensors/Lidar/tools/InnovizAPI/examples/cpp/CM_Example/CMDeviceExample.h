// Copyright 2021 Innoviz Technologies
//
// Licensed under the Innoviz Open Dataset License Agreement (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/InnovizTechnologies/InnovizAPI/blob/main/LICENSE.md
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

///////////////////////////////////////////////////////////
//  CMDeviceExample.h
//  Implementation of the Class CMDeviceExample
//  Created on:      07-Oct-2020 1:13:29 AM
//  Original author: julia.sher
///////////////////////////////////////////////////////////

#ifndef _CM_DEVICE_DEMO_H
#define _CM_DEVICE_DEMO_H

#include "interface/DeviceApi.h"
#include <string>

class CMDeviceExample
{

public:
	explicit CMDeviceExample(std::string cm_device_config);
	virtual ~CMDeviceExample();

	virtual void RecordUDPStream();
	virtual void ReadObjects();

private:
	void initUserBuffers(size_t buffersCount, std::vector<invz::FrameDataUserBuffer>& userBuffers, std::vector<invz::FrameDataAttributes>& attributes);
	void frameCallback(uint32_t* id);

	std::unique_ptr<invz::IDevice> m_device;
	std::vector<invz::FrameDataUserBuffer> m_userBuffers;

};
#endif // _CM_DEVICE_DEMO_H
