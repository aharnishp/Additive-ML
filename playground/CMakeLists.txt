cmake_minimum_required(VERSION 3.5)

project(MyProject)

# if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
# Add include directories specific to x86_64, ARMv8, or arm64
# include_directories(openblas pthread gfortran)


add_executable(main main.cpp)

# find_package(OpenBLAS REQUIRED)

# target_link_directories(main PRIVATE modules/ ${OpenBLAS_INCLUDE_DIRS})
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    target_include_directories(main PRIVATE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_include_directories(main PRIVATE /usr/include/aarch64-linux-gnu/)
endif()

    # target_link_libraries(main PRIVATE /usr/lib/libopenblas.so)
    # target_link_libraries(main PRIVATE pthread gfortran)

# else()
#     add_executable(main main.cpp)
    
#     target_link_directories(main PRIVATE modules/)
# endif()


# If you need to link libraries, you can add them with target_link_libraries()
# For example: target_link_libraries(main PRIVATE library_name)

# If you need additional compiler flags, you can set them with target_compile_options()
# For example: target_compile_options(main PRIVATE -Wall)
