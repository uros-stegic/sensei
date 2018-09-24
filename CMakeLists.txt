cmake_minimum_required (VERSION 3.0.0)

# Project bio
project (sensei)
set (sensei_VERSION_MAJOR 0)
set (sensei_VERSION_MINOR 0)
set (sensei_VERSION_PATCH 1)
set (TARGET sensei)

# -std=c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# Configuration file
configure_file (
	"${PROJECT_SOURCE_DIR}/config.h.in"
	"${PROJECT_BINARY_DIR}/config.h"
)

# Debugging symbols
#set(CMAKE_BUILD_TYPE Debug)

# Armadillo Library
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Set up my class search paths
include_directories ("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories ("${CMAKE_CURRENT_BINARY_DIR}")
include_directories ("${PROJECT_SOURCE_DIR}/include")

file(GLOB_RECURSE SOURCES
	${PROJECT_SOURCE_DIR}/src/*.cpp
	#test/main.cpp
)
file(GLOB_RECURSE INCLUDES
	${PROJECT_BIN_DIR}/include/*.hpp
)

add_library(${TARGET} SHARED ${SOURCES})
#add_executable(${TARGET} ${SOURCES})

target_link_libraries(${TARGET} ${ARMADILLO_LIBRARIES})
target_include_directories(${TARGET} PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
)
file(COPY ${PROJECT_SOURCE_DIR}/include DESTINATION ${PROJECT_BINARY_DIR})

install(
	TARGETS ${TARGET}
	DESTINATION lib
)
install(
	DIRECTORY include/
	DESTINATION include/sensei
	PATTERN "*.hpp"
)

