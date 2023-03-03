cmake_minimum_required(VERSION 2.8.12)
project(examples)

# Fetch Dlib
include(FetchContent)
FetchContent_Declare(dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG        v19.18
)
FetchContent_MakeAvailable(dlib)

add_executable(assignment_learning_ex main.cxx)
target_link_libraries(assignment_learning_ex dlib::dlib)
