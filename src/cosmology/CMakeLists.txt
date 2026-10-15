include(CheckCXXSourceCompiles)

option(BUILD_MUSIC "build MUSIC (for cosmological initial conditions)")
find_package(HDF5)

ExternalProject_Add(
  build_music
  EXCLUDE_FROM_ALL 1
  GIT_REPOSITORY https://github.com/cosmo-sims/MUSIC2.git
  CMAKE_ARGS "${MY_CMAKE_ARGS}"
  "${MY_SDL_ARGS}"
  INSTALL_COMMAND ln -s ${CMAKE_CURRENT_BINARY_DIR}/build_music-prefix/src/build_music-build/MUSIC ${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_random.sh ${CMAKE_CURRENT_SOURCE_DIR}/scripts/goopax_logo_black.png ${CMAKE_CURRENT_SOURCE_DIR}/scripts/ics_goopax.conf.in ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run.sh ${CMAKE_CURRENT_BINARY_DIR}/
#  "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/music"
)


check_cxx_source_compiles([[
#include <ranges>
static_assert(__cpp_concepts >= 201907);
static_assert(__cpp_lib_ranges >= 201911);

int main(){}
]] HAVE_CONCEPTS)

message("HAVE_CONCEPTS=${HAVE_CONCEPTS}")

if (HAVE_CONCEPTS)
  add_sdl_main(cosmology goopax_draw)
  message("building cosmology")
else()
  message("not building cosmology: requires concepts and ranges")
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
  target_link_libraries(cosmology stdc++exp)
endif()

