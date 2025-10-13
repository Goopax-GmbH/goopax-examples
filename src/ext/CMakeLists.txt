add_custom_target(build_external_libraries)
include(ExternalProject)

set(MY_OPENCV_ARGS "-DOpenCV_STATIC=ON")
if (WIN32 OR ANDROID)
  set(MY_OPENCV_URL https://github.com/opencv/opencv/archive/refs/tags/4.12.0.zip)
  set(MY_OPENCV_URL_HASH SHA256=FA3FAF7581F1FA943C9E670CF57DD6BA1C5B4178F363A188A2C8BFF1EB28B7E4)
else ()
  set(MY_OPENCV_URL https://github.com/opencv/opencv/archive/refs/tags/4.10.0.zip)
  set(MY_OPENCV_URL_HASH SHA256=3810bca2b1d1c572912df0ac3888126341f3762dfd28e91068c805fb656d0e51)
endif()
if (ANDROID)
  set(MY_SDL_GIT_TAG release-3.2.18)
  set(MY_SDL_ARGS
    "-DSDL_SHARED=1"
    "-DSDL_STATIC=0")
else ()
  set(MY_SDL_GIT_TAG release-3.2.4)
  set(MY_SDL_ARGS
    "-DSDL_SHARED=0"
    "-DSDL_STATIC=1")
endif()

list(APPEND MY_CMAKE_ARGS -DBUILD_SHARED_LIBS=0)
list(APPEND MY_CMAKE_ARGS -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD})
list(APPEND MY_CMAKE_ARGS -DCMAKE_VERBOSE_MAKEFILE=1)

if (WIN32)
  list(APPEND MY_CMAKE_ARGS -DCMAKE_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY})
elseif (ANDROID OR IOS)
  list(APPEND MY_CMAKE_ARGS -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME})
  if (ANDROID)
    list(APPEND MY_CMAKE_ARGS
      -DCMAKE_ANDROID_STL_TYPE=${CMAKE_ANDROID_STL_TYPE}
      -DCMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION}
      -DCMAKE_ANDROID_ARCH_ABI=${CMAKE_ANDROID_ARCH_ABI}
      -DCMAKE_ANDROID_NDK=${CMAKE_ANDROID_NDK}
      -DCMAKE_POSITION_INDEPENDENT_CODE=${CMAKE_POSITION_INDEPENDENT_CODE})
    list(APPEND MY_OPENCV_ARGS
      -DANDROID_STL=${CMAKE_ANDROID_STL_TYPE}
      -DBUILD_JAVA=0
      -DBUILD_ANDROID_EXAMPLES=0
      -DBUILD_ANDROID_PROJECTS=0
      -DWITH_ANDROID_MEDIANDK=1)
  elseif (IOS)
    list(APPEND MY_OPENCV_ARGS
      -DWITH_PNG=0
      -DWITH_WEBP=0
      -DBUILD_WEBP=0
      -DWITH_JASPER=0
      -DBUILD_JASPER=0)
    list(APPEND MY_CMAKE_ARGS
      -G Xcode
      -DCMAKE_XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET=${CMAKE_XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET})
    set(MY_SDL_GIT_TAG 8f9a562969761ec626d298079c99341fd25b4644)
  endif()
endif()

list(APPEND MY_OPENCV_ARGS -DBUILD_opencv_gapi=0)
list(APPEND MY_OPENCV_ARGS -DWITH_ADE=0)
list(APPEND MY_OPENCV_ARGS -DWITH_OBSENSOR=0)


ExternalProject_Add(
  build_sdl3
  GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
  GIT_TAG ${MY_SDL_GIT_TAG}
  EXCLUDE_FROM_ALL 1
  CMAKE_ARGS "${MY_CMAKE_ARGS}"
  "${MY_SDL_ARGS}"
  "-DSDL_ASSEMBLY=0"
  "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/sdl3"
)
if (NOT TARGET SDL3::SDL3)
  add_dependencies(build_external_libraries build_sdl3)
endif()

ExternalProject_Add(
  build_opencv
  URL ${MY_OPENCV_URL}
  URL_HASH ${MY_OPENCV_URL_HASH}
  EXCLUDE_FROM_ALL 1
  CMAKE_ARGS "${MY_CMAKE_ARGS}"
  "${MY_OPENCV_ARGS}"
  "-DBUILD_ITT=0"
  "-DBUILD_PERF_TESTS=0"
  "-DBUILD_TESTS=0"
  "-DBUILD_opencv_apps=0"
  "-DWITH_PROTOBUF=0"
  "-DWITH_QUIRC=1"
  "-DWITH_LAPACK=0"
  "-DWITH_CUDA=0"
  "-DBUILD_opencv_dnn=0"
  "-DWITH_IPP=0"
  "-DWITH_OPENEXR=0"
  "-DWITH_JPEG=0"
  "-DWITH_OPENJPEG=0"
  "-DWITH_TIFF=0"
  "-DWITH_FFMPEG=0"
  "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/opencv"
)
if (NOT OpenCV_FOUND)
  add_dependencies(build_external_libraries build_opencv)
endif()

ExternalProject_Add(
  build_boost
  URL https://github.com/boostorg/multiprecision/archive/refs/tags/Boost_1_86_0.zip
  URL_HASH SHA256=855BD78A1A4EE18BFB385ACA54C153B55E4C97A137604BE951208B144DF7E3BD
  EXCLUDE_FROM_ALL 1
  CONFIGURE_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_COMMAND ""
)
if (NOT TARGET Boost::boost)
  add_dependencies(build_external_libraries build_boost)
endif()

ExternalProject_Add(
  build_eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  EXCLUDE_FROM_ALL 1
  CMAKE_ARGS "${MY_CMAKE_ARGS}"
  "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/eigen"
  #INSTALL_COMMAND ""
)
if (NOT TARGET Eigen3::Eigen)
  add_dependencies(build_external_libraries build_eigen)
endif()

ExternalProject_Add(
  build_glm
  URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.tar.gz
  URL_HASH SHA256=9f3174561fd26904b23f0db5e560971cbf9b3cbda0b280f04d5c379d03bf234c
  EXCLUDE_FROM_ALL 1
  CMAKE_ARGS "${MY_CMAKE_ARGS};-DGLM_BUILD_LIBRARY=0;-DGLM_BUILD_TESTS=0"
  "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/glm"
  #INSTALL_COMMAND ""
)
if (NOT TARGET glm::glm)
  add_dependencies(build_external_libraries build_glm)
endif()

ExternalProject_Add(
  build_glatter
  GIT_REPOSITORY https://github.com/imakris/glatter.git
  GIT_TAG 886f9765c6553ecb0934b5ec187db7d57ab0cfbc
  EXCLUDE_FROM_ALL 1
  CONFIGURE_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_COMMAND ""
)
if (NOT GLATTER_INCLUDE_DIR)
  add_dependencies(build_external_libraries build_glatter)
endif()
