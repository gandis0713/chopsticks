cmake_minimum_required(VERSION 3.22)

project(chopsticks)

add_executable(chopsticks main.cpp) 

find_package(spdlog REQUIRED)
target_link_libraries(chopsticks PRIVATE spdlog::spdlog)