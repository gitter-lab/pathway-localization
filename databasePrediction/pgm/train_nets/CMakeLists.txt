cmake_minimum_required (VERSION 3.1)

project (train_nets)

# OpenCV and DGM packages
find_package(OpenCV 4.0 REQUIRED core features2d highgui imgproc imgcodecs ml PATHS "~/research/pathwayLocalization/lib/opencv-4.4.0/build")
find_package(DGM 1.7 REQUIRED PATHS "~/research/pathwayLocalization/lib/DGM-v.1.7.0/build")

# Turn on the ability to create folders to organize projects (.vcproj)
# It creates "CMakePredefinedTargets" folder by default and adds CMake defined projects like INSTALL.vcproj and ZERO_CHECK.vcproj
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Sets
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD 17)
if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi /wd5040")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS} /DEBUG /INCREMENTAL:NO /OPT:REF /OPT:ICF")
endif(MSVC)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Properties -> C/C++ -> General -> Additional Include Directories
include_directories(${OpenCV_INCLUDE_DIRS} ${DGM_INCLUDE_DIRS})

#definitions
add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)

add_executable(train_nets "train_nets.cpp" "train_nets.h")

# Properties -> Linker -> Input -> Additional Dependencies
target_link_libraries(train_nets ${OpenCV_LIBS} ${DGM_LIBRARIES})
