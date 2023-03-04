# src/CMakeLists.txt
set(SOURCES libfoo.cc)
set(HEADERS libfoo.h)

# Create shared library
add_library(mylib SHARED ${SOURCES} ${HEADERS})

# Set properties
set_target_properties(mylib PROPERTIES VERSION 1.0 SOVERSION 1 PUBLIC_HEADER libfoo.h)
target_include_directories(mylib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${DEPS_INCLUDE_DIRS})
target_compile_options(mylib PRIVATE -Wall -Wextra -pedantic -std=c++11)
target_link_libraries(mylib ${DEPS_LIBRARIES})

# Install library and headers
install(TARGETS mylib LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Configure and install pkg-config file
configure_file(mylib.pc.in mylib.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/mylib.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
