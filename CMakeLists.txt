cmake_minimum_required(VERSION 3.5)

project(MyProject)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # Add include directories specific to x86_64, ARMv8, or arm64
    include_directories(openblas pthread gfortran)
endif()

add_executable(main main.cpp)

target_link_directories(main PRIVATE modules/)

# If you need to link libraries, you can add them with target_link_libraries()
# For example: target_link_libraries(main PRIVATE library_name)

# If you need additional compiler flags, you can set them with target_compile_options()
# For example: target_compile_options(main PRIVATE -Wall)
