# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(guise)

# Set non-standard prefixes for dependencies
#set(CMAKE_PREFIX_PATH /home/user/libraries/foo /home/user/libraries/bar)

include(FetchContent)
FetchContent_Declare(dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG        v19.18
)
FetchContent_MakeAvailable(dlib)

# Add subdirectory
add_subdirectory(src)
