cmake_minimum_required(VERSION 3.22)

project(
  chopsticks
  VERSION 0.0.1
  DESCRIPTION "litert test in raspberry pi 5"
)

# Add subdirectories
# Add subdirectories
add_subdirectory(third_party/litert/litert)
add_subdirectory(chopsticks)

