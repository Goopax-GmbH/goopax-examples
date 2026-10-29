Building Example Programs
=========================

The following tools are required to build the example programs:
- cmake
- git
If they are not yet installed on your system, install them, and make sure the path to the executables is in the PATH variable.

We assume the following C++ compiler is installed, depending on your operating system:
- Linux: gcc or clang
- Windows: visual studio 2022
- MacOS/iOS: Xcode
- Android: android ndk

Run the script

    ./build-all.sh

or, on windows,

    build.bat

to fetch and build all required libraries (SDL3, OpenCV, boost, Eigen, glatter) that are not already installed on the system. Building the libraries may take a while. Then, the example programs are built.

If you prefer not to build external libraries, you can start cmake by hand:

    cmake -B build src
    cmake --build build

In this case, only those examples are build, for which all required libraries are found.

To run the cosmology example program, also read the comments at the top of file "cosmology/cosmology.cpp".


Cross-Compiling for iOS
-----------------------
To build for iOS, you also need to specify your developer team id (see https://developer.apple.com/help/account/manage-your-team/locate-your-team-id), and to pass some additional arguments. The following command will build for minimum iOS version 17.

    ./build-all.sh -G Xcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET=17 -DAPPLE_DEVELOPER_TEAM=<your_developer_team_id>

Then, open build/goopax_examples.xcodeproj and install the programs manually.


Cross-Compiling for iOS Simulator
---------------------------------
To build for the iOS simulator, follow the steps for iOS, but with the additional -DCMAKE_OSX_SYSROOT=iphonesimulator option.


Cross-Compiling for Android
---------------------------
This build is using the cmake toolchain supplied by Google:
https://developer.android.com/studio/projects/configure-cmake

Set the ANDROID_NDK environment variable to the location of the Android NDK.

Then run something like this:

    ./build-all.sh -G "Ninja" -DCMAKE_SYSTEM_NAME="Android" -DCMAKE_ANDROID_STL_TYPE="c++_static" -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM="android-35" -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" -DCMAKE_FIND_ROOT_PATH="$PWD/../;$PWD/build/ext/eigen;$PWD/build/ext/glm;$PWD/build/ext/sdl3;$PWD/build/ext/build_boost-prefix/src/build_boost/include;$PWD/build/ext/opencv/sdk/native/jni" -DGOOPAX_DRAW_WITH_OPENGL=0 -DGOOPAX_DRAW_WITH_METAL=0 -DGOOPAX_DRAW_WITH_VULKAN=1

For the graphical SDL-based example programs, shared libraries are created. They need to be further incorporated into an android app. For a simple test, (https://github.com/specpose/goopax-template-android) may be sufficient.
