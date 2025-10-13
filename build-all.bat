@echo off
setlocal EnableDelayedExpansion

set CMAKE_BUILD_PARALLEL_LEVEL=%NUMBER_OF_PROCESSORS%

cmake %* %CMAKE_FLAGS% -B build src
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

cmake --build build --config Release --target build_external_libraries
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

cmake build
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

cmake --build build --config Release
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

endlocal
