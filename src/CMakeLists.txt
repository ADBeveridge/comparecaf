# src/CMakeLists.txt

set(SOURCES guise.cxx)

# Create shared library
add_library(guise SHARED ${SOURCES})

# Define a preprocessor variable to access the data file path from C++ code. This was defined back in the data subdir.
target_compile_definitions(guise PRIVATE DATA_FILE_PATH_RESNET="${DATA_FILE_PATH_RESNET}")
target_compile_definitions(guise PRIVATE DATA_FILE_PATH_LANDMK="${DATA_FILE_PATH_LANDMK}")

if(DLIB_FOUND)
    target_include_directories(guise PUBLIC ${DLIB_INCLUDE_DIRS})
    target_link_libraries(guise PUBLIC dlib)
else()
    target_link_libraries(guise PUBLIC dlib::dlib)
endif()

# Configure PkgConfig file.
configure_file(guise.pc.in guise.pc @ONLY)

# Install everything.
install(TARGETS guise LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES guise.hxx DESTINATION include/guise)
install(FILES ${CMAKE_BINARY_DIR}/src/guise.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
