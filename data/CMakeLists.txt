# Set the path to the data files.
set(DATA_FILE_PATH_RESNET "${CMAKE_INSTALL_PREFIX}/share/guise/data/dlib_face_recognition_resnet_model_v1.dat")
set(DATA_FILE_PATH_LANDMK "${CMAKE_INSTALL_PREFIX}/share/guise/data/shape_predictor_5_face_landmarks.dat")

# Install the data file
install(FILES dlib_face_recognition_resnet_model_v1.dat DESTINATION share/guise/data)
install(FILES shape_predictor_5_face_landmarks.dat DESTINATION share/guise/data)
