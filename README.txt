Building Example Programs
=========================

Running the script

    ./build-all.sh

will fetch and build all missing libraries (SDL3, OpenCV, boost, Eigen, glatter), and then the example programs. Building the libraries may take a while. 

If you prefer not to build external libraries, you can start cmake by hand:

    cmake -B build src
    cmake --build build

In this case, only those examples are build, for which all required libraries are found.


Windows
-------
To build the example programs on windows, install the following applications:
- visual studio 2022
- cmake
- git
Then run

    build-all.bat


MacOS
-----
To build the example programs for MacOS, install the following applications:
- Xcode
- cmake
Open a terminal.
Set the PATH variable so that the cmake executable is found.
Then run

    ./build-all.sh


iOS
---
In addition to the steps required for MacOS, you also need to specify your developer team id (see https://developer.apple.com/help/account/manage-your-team/locate-your-team-id), and to pass some additional arguments. The following line will build for minimum iOS version 15.

    ./build-all.sh -G Xcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET=15 -DAPPLE_DEVELOPER_TEAM=<your_developer_team_id>

Then, open build/goopax_examples.xcodeproj and install the programs manually.


Android
-------
The following has been tested with requirements installed from the Android Studio SDK Installer for Linux:
- android ndk
- android sdk
- cmake, including ninja from the ndk.
Set the PATH variable so that both ninja and corresponding cmake are found.

    export PATH=$HOME/Android/Sdk/cmake/3.22.1/bin:$PATH

This represents the minimal system requirements to build the text-based examples.

    build_type="release" ABI="arm64-v8a" android_ndk="$HOME/Android/Sdk/ndk/27.0.12077973" bash -c './build-all.sh -G "Ninja" -DCMAKE_SYSTEM_NAME="Android" -DCMAKE_ANDROID_ARCH_ABI="$ABI" -DCMAKE_ANDROID_NDK="$android_ndk" -DCMAKE_FIND_ROOT_PATH="$PWD/../;$PWD/build/$build_type/$ABI/ext/boost;$PWD/build/$build_type/$ABI/ext/opencv/sdk/native/jni;$PWD/build/$build_type/$ABI/ext/eigen" -DGOOPAX_DRAW_WITH_OPENGL=0 -DGOOPAX_DRAW_WITH_METAL=0 -DGOOPAX_DRAW_WITH_VULKAN=1'
