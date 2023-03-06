# Root CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(guise)

include(GNUInstallDirs)
include(FetchContent)

# Suppress -Wdeprecated error messages. This comes from DLib's use of std::iterator.
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
endif()

# Get DLib source, and make sure that it is being compiled with position independent code. 
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(DLIB dlib-1)
    #pkg_check_modules(FLEXIBLAS flexiblas)
endif()

if(NOT DLIB_FOUND)
    # Use FetchContent to pull in dlib as a dependency
    FetchContent_Declare(dlib
        GIT_REPOSITORY https://github.com/davisking/dlib.git
        GIT_TAG        v19.18
    )
    FetchContent_GetProperties(dlib)
    if(NOT dlib_POPULATED)
      FetchContent_Populate(dlib)
      add_subdirectory(${dlib_SOURCE_DIR} ${dlib_BINARY_DIR})
    endif()
    set_target_properties(dlib PROPERTIES POSITION_INDEPENDENT_CODE ON)
    FetchContent_MakeAvailable(dlib)
endif()

# Add subdirectory
add_subdirectory(src)
