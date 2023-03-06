# src/CMakeLists.txt

set(SOURCES guise.cxx)

# Create shared library
add_library(guise SHARED ${SOURCES} ${HEADERS})

if(DLIB_FOUND)
    target_include_directories(guise PUBLIC ${DLIB_INCLUDE_DIRS})
    target_link_libraries(guise PUBLIC dlib)
else()
    target_link_libraries(guise PUBLIC dlib::dlib)
endif()

# Configure PkgConfig file.
configure_file(guise.pc.in guise.pc @ONLY)

install(TARGETS guise LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES guise.hxx DESTINATION include/guise)
install(FILES ${CMAKE_BINARY_DIR}/src/guise.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
