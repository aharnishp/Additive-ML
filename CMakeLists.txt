cmake_minimum_required(VERSION 3.0)

project(MyProject)

add_executable(main main.cpp)

target_link_directories(main PRIVATE modules/)

# If you need to link libraries, you can add them with target_link_libraries()
# For example: target_link_libraries(main PRIVATE library_name)

# If you need additional compiler flags, you can set them with target_compile_options()
# For example: target_compile_options(main PRIVATE -Wall)
