# src/CMakeLists.txt
#set(SOURCES guise.cxx)
#set(HEADERS guise.hxx)

# Create shared library
#add_library(guise SHARED ${SOURCES} ${HEADERS})

# Set properties
#set_target_properties(guise PROPERTIES VERSION 1.0 SOVERSION 1 PUBLIC_HEADER guise.hxx)
#target_include_directories(guise PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${DEPS_INCLUDE_DIRS})
#target_link_libraries(guise dlib::dlib)

# Install library and headers
#install(TARGETS guise LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Configure and install pkg-config file
#configure_file(guise.pc.in guise.pc @ONLY)
#install(FILES ${CMAKE_BINARY_DIR}/guise.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)


add_executable(assignment_learning_ex guise.cxx)
target_link_libraries(assignment_learning_ex dlib::dlib)
