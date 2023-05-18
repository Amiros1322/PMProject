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
//  CMDeviceExample.cpp
//  Implementation of the Class CMDeviceExample
//  Created on:      07-Oct-2020 1:13:29 AM
//  Original author: julia.sher
///////////////////////////////////////////////////////////

#include "CMDeviceExample.h"
#include "../common/CommonUtils.h"
#include "../common/ExampleTypes.h"
#include <iostream>


CMDeviceExample::CMDeviceExample(std::string config_path)
{
	m_device.reset(invz::DeviceInit(config_path));
}



CMDeviceExample::~CMDeviceExample()
{
	if (m_device)
		invz::DeviceClose(m_device.release());
}




void CMDeviceExample::initUserBuffers(size_t buffersCount, std::vector<invz::FrameDataUserBuffer>& userBuffers, std::vector<invz::FrameDataAttributes>& attributes)
{
	// Initialize User Buffers
	for (size_t i = 0; i < buffersCount; i++)
	{
		if (attributes[i].known_type == invz::GRAB_TYPE_DETECTIONS)
		{
			invz::FrameDataUserBuffer&& userBuffer(attributes[i]);
			userBuffers.push_back(userBuffer);
			m_device->ActivateBuffer(attributes[i], true);
		}
	}
}


void CMDeviceExample::frameCallback(uint32_t* id)
{
	uint32_t frameNumber;
	uint64_t timeStamp;
	// Grabbing Frame from device inteface
	auto result = m_device->GrabFrame(m_userBuffers.data(), (uint32_t)m_userBuffers.size(), frameNumber, timeStamp, *id);

	if (result.error_code == invz::ERROR_CODE_OK)
	{
		if (m_userBuffers[static_cast<size_t>(ECMDataType::Objects)].status == invz::USER_BUFFER_FULL)
		{
			std::cout << std::endl;
			std::cout << "\tFrame number: " << frameNumber << std::endl;
			CommonUtils::HandleObjectsData(m_userBuffers[static_cast<size_t>(ECMDataType::Objects)]);
		}
	}
}


void CMDeviceExample::RecordUDPStream()
{
	invz::Result status;
	std::string rec_file_name = "demoProgramDBPT";

	std::cout << "\nPlease enter the record length in seconds: ";

	double recordLen = 0.0;
	std::cin >> recordLen;

	std::cout << "Start recording..." << std::endl;
	status = m_device->Record(recordLen, rec_file_name);

	std::cout << "Finished recording " << rec_file_name << std::endl << std::endl;
}


void CMDeviceExample::ReadObjects()
{
	std::vector<invz::FrameDataAttributes> attributes;
	attributes.resize(DemoConfig::max_num_of_buffers);

	// buffersCount holds the actual number of existing buffers - returned by GetFrameDataAttributes
	size_t buffersCount = DemoConfig::max_num_of_buffers;


	invz::Result result;
	result.error_code = invz::ERROR_CODE_GENERAL;

	std::cout << "Connecting to device ..." << std::endl;
	while (result.error_code != invz::ERROR_CODE_OK)
	{
		// To receive OM packets - use wait_for_device=true to recieve meta data first
		result = m_device->GetFrameDataAttributes(attributes.data(), buffersCount, false);
	}

	// buffers to read the required data - 
	// GRAB_TYPE_DETECTIONS
	initUserBuffers(buffersCount, m_userBuffers, attributes);

	invz::Result status;
	auto callback = std::bind(&CMDeviceExample::frameCallback, this, std::placeholders::_1);
	status = m_device->RegisterFrameCallback(callback);
	std::cout << "Start Grabbing ... Press any key + Enter to stop" << std::endl;

	char user;
	std::cin >> user;
	m_device->UnregisterFrameCallback();
}

int main(void)
{
	std::string config_files_path = "../../lidar_configuration_files";
	std::string device_config_file = "cm_config.json";
	std::unique_ptr<CMDeviceExample> m_CMDeviceExample = std::make_unique<CMDeviceExample>(config_files_path + "/" + device_config_file);

	bool stop = false;
	while (!stop)
	{
		std::cout << "\n Please select the required demo: " << std::endl;
		std::cout << "\t1 - Read frames from CM (objects)" << std::endl;
		std::cout << "\t2 - Record CM stream" << std::endl;
		std::cout << "\tq - Quit" << std::endl;

		char user;
		std::cin >> user;
		switch (user)
		{
		case '1':
			m_CMDeviceExample->ReadObjects();
			break;
		case '2':
			m_CMDeviceExample->RecordUDPStream();
			break;
		case 'q':
			stop = true;
			break;
		default:
			std::cout << "Unsupported! Try again" << std::endl;
			break;
		}
	}

	return 0;
}