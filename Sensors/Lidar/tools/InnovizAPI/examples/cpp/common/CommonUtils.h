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
//  CommonUtils.h
//  Implementation of class CommonUtils
//  Created on:      07-Oct-2020 1:13:25 AM
//  Original author: julia.sher
///////////////////////////////////////////////////////////

#ifndef _COMMON_UTILS_H
#define _COMMON_UTILS_H

#include "common_includes/invz_types.h"

class CommonUtils
{

public:
	CommonUtils();
	virtual ~CommonUtils();

	static void HandleReflection0Data(invz::FrameDataUserBuffer& ref_buffer);
	static void HandleObjectsData(invz::FrameDataUserBuffer& ref_buffer);

private:
	static bool objectExists(const invz::ObjectDetection& object);
	static invz::ClassesType objectClass(invz::ClassProbability probability_of_classtype);
	static std::string getObjectClassStr(invz::ClassProbability probability_of_classtype);

};

#endif // _COMMON_UTILS_H