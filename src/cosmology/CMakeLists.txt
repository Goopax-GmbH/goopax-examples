
add_sdl_main(cosmology glatter goopax_draw)
if (TARGET cosmology)
  if (TARGET opencv)
    target_link_libraries(cosmology opencv)
    target_compile_definitions(cosmology PUBLIC -DWITH_OPENCV=1)
  else()
    target_compile_definitions(cosmology PUBLIC -DWITH_OPENCV=0)
  endif()
  if (APPLE)
    SET_SOURCE_FILES_PROPERTIES(cosmology.cpp PROPERTIES LANGUAGE OBJCXX)
  endif()
endif()

