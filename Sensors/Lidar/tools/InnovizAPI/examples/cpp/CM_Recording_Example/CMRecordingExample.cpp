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
//  CMRecordingExample.cpp
//  Implementation of the Class CMRecordingExample
//  Created on:      07-Oct-2020 1:13:25 AM
//  Original author: julia.sher
///////////////////////////////////////////////////////////

#include "CMRecordingExample.h"
#include "../common/CommonUtils.h"
#include <iostream>


CMRecordingExample::CMRecordingExample(std::string cm_file_path)
{
	m_reader.reset(invz::FileReaderInit(cm_file_path));
}



CMRecordingExample::~CMRecordingExample()
{
	if (m_reader)
		FileReaderClose(m_reader.release());
}



void CMRecordingExample::initUserBuffers(size_t buffersCount, std::vector<invz::FrameDataUserBuffer>& userBuffers, std::vector<invz::FrameDataAttributes>& attributes)
{
	// Initialize User Buffers
	for (size_t i = 0; i < buffersCount; i++)
	{
		if (attributes[i].known_type == invz::GRAB_TYPE_DETECTIONS)
		{
		invz::FrameDataUserBuffer&& userBuffer(attributes[i]);
		userBuffers.push_back(userBuffer);
		}
	}
}


void CMRecordingExample::GetNumberOfFrames()
{
	// Read number of frames from file
	size_t frames;
	m_reader->GetNumOfFrames(frames);
	std::cout << std::endl;
	std::cout << "Number of frames: " << frames << std::endl;
}


void CMRecordingExample::ReadObjects()
{
	std::vector<invz::FrameDataAttributes> attributes;
	attributes.resize(DemoConfig::max_num_of_buffers);

	// buffersCount holds the actual number of existing buffers - returned by GetFrameDataAttributes
	size_t buffersCount = DemoConfig::max_num_of_buffers;
	m_reader->GetFrameDataAttributes(attributes.data(), buffersCount);


	initUserBuffers(buffersCount, m_userBuffers, attributes);

	// Initialize to default values - returned by GrabFrame
	uint32_t frameNumber = UINT32_MAX;
	uint64_t timeStamp = 0;

	// Read Frame required data into userBuffers initialized above 
	size_t frames;
	m_reader->GetNumOfFrames(frames);

	std::cout << std::endl;
	std::cout << "Reading " << frames << " frames ... " << std::endl;
	for (uint32_t frame_index = 0; frame_index < frames; ++frame_index)
	{
		auto result = m_reader->GrabFrame(m_userBuffers.data(), (uint32_t)m_userBuffers.size(), frameNumber, timeStamp, frame_index);
		std::cout << std::endl;
		std::cout << "\tFrame number: " << frameNumber << std::endl;
		if (result.error_code == invz::ERROR_CODE_OK)
		{
			if (m_userBuffers[static_cast<size_t>(ECMDataType::Objects)].status == invz::USER_BUFFER_FULL)
				CommonUtils::HandleObjectsData(m_userBuffers[static_cast<size_t>(ECMDataType::Objects)]);
		}
	}
	std::cout << std::endl;
	std::cout << "Finished reading frames." << std::endl;
}

int main(void)
{
	std::string recordings_path = "../../example_recordings";
	std::string recording = "pc_plus.dbpt";
	std::unique_ptr<CMRecordingExample> m_CMRecordingExample = std::make_unique<CMRecordingExample>(recordings_path + "/" + recording);

	bool stop = false;
	while (!stop)
	{
		std::cout << "\n Please select the required demo: " << std::endl;
		std::cout << "\t1 - Read objects from CM recording" << std::endl;
		std::cout << "\t2 - Get CM recording number of frames" << std::endl;
		std::cout << "\tq - Quit" << std::endl;

		char user;
		std::cin >> user;
		switch (user)
		{
		case '1':
			m_CMRecordingExample->ReadObjects();
			break;
		case '2':
			m_CMRecordingExample->GetNumberOfFrames();
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