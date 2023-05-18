cmake_minimum_required (VERSION 3.5.2)

project(cm_recording_example)

set(Boost_USE_STATIC_LIBS ON)
add_definitions(-DINVZ_API_IMPORT)
add_definitions(-DNDARRAY_IMPORT)


set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../../../include)

# set sources
add_executable(${PROJECT_NAME} 
    "CMRecordingExample.h"
    "CMRecordingExample.cpp"
    "../common/CommonUtils.h"
    "../common/CommonUtils.cpp"
    "../common/ExampleTypes.h"
)

target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})

if(WIN32)
    add_definitions(-DBOOST_USE_WINAPI_VERSION=0x0601)
    add_definitions(-D_WIN32_WINNT=0x0601)

    find_library(INVZ_API_LIB innovizApi PATHS ${PROJECT_SOURCE_DIR}/../../../lib/win32_x86/Release)
	find_library(INVZ_API_DEBUG_LIB innovizApi PATHS ${PROJECT_SOURCE_DIR}/../../../lib/win32_x86/Debug)
	target_link_libraries(${PROJECT_NAME} optimized ${INVZ_API_LIB})
	target_link_libraries(${PROJECT_NAME} debug ${INVZ_API_DEBUG_LIB})

    set_target_properties(${PROJECT_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif(WIN32)

if(UNIX)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")	

    find_library(INVZ_API_LIB innovizApi PATHS ${PROJECT_SOURCE_DIR}/../../../lib/linux_x86)
    target_link_libraries(${PROJECT_NAME} ${INVZ_API_LIB} stdc++fs pthread)
endif(UNIX)    