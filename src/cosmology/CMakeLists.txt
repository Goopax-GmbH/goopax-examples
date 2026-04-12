include(CheckCXXSourceCompiles)

find_package(HDF5)


check_cxx_source_compiles([[
static_assert(__cpp_concepts >= 201907);
int main(){}
]] HAVE_CONCEPTS)

message("HAVE_CONCEPTS=${HAVE_CONCEPTS}")

if (HAVE_CONCEPTS)
  add_sdl_main(cosmology goopax_draw)
  message("building cosmology")
else()
  message("not building cosmology: requires concepts")
endif()

if (TARGET cosmology)
  if (TARGET opencv AND NOT GOOPAX_DEBUG)
    target_link_libraries(cosmology opencv)
    target_compile_definitions(cosmology PUBLIC -DWITH_OPENCV=1)
  else()
    target_compile_definitions(cosmology PUBLIC -DWITH_OPENCV=0)
  endif()
  if (TARGET hdf5::hdf5)
    target_link_libraries(cosmology hdf5::hdf5)
    target_compile_definitions(cosmology PUBLIC -DWITH_HDF5=1)
  else()
    target_compile_definitions(cosmology PUBLIC -DWITH_HDF5=0)
  endif()

  if (APPLE)
    SET_SOURCE_FILES_PROPERTIES(cosmology.cpp PROPERTIES LANGUAGE OBJCXX)
  endif()
endif()

