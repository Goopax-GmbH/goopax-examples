find_package(HDF5)


add_sdl_main(cosmology glatter goopax_draw Boost::iostreams hdf5::hdf5)
if (TARGET cosmology)
  if (TARGET opencv AND NOT GOOPAX_DEBUG)
    target_link_libraries(cosmology opencv)
    target_compile_definitions(cosmology PUBLIC -DWITH_OPENCV=1)
  else()
    target_compile_definitions(cosmology PUBLIC -DWITH_OPENCV=0)
  endif()
  if (APPLE)
    SET_SOURCE_FILES_PROPERTIES(cosmology.cpp PROPERTIES LANGUAGE OBJCXX)
  endif()
endif()

