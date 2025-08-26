@echo off
setlocal EnableDelayedExpansion

set CMAKE_BUILD_PARALLEL_LEVEL=%NUMBER_OF_PROCESSORS%

cmake %* %CMAKE_FLAGS% -DCMAKE_CONFIGURATION_TYPES="Debug;Release" -B build src
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

cmake --build build --config Release --target build_boost
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

cmake --build build --config Release --target build_opencv
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

cmake --build build --config Release --target build_sdl3
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

cmake --build build --config Release --target build_eigen
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

cmake --build build --config Release --target build_glatter
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

cmake build
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

cmake --build build --config Release
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

endlocal
