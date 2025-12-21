cmake_minimum_required(VERSION 3.22)

project(chopsticks)

add_executable(chopsticks main.cpp) 

target_link_libraries(chopsticks PRIVATE litert_runtime_c_api_shared_lib)

find_package(spdlog REQUIRED)
target_link_libraries(chopsticks PRIVATE spdlog::spdlog)