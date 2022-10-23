# CMakeList.txt : CMake project for CVATParser, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.12)

include(FetchContent)
FetchContent_Declare(pugixml
  URL    http://github.com/zeux/pugixml/releases/download/v1.12/pugixml-1.12.tar.gz
)
FetchContent_MakeAvailable(pugixml)

find_package(OpenCV REQUIRED)


# Add source to this project's executable.
add_executable (CVATTools "CVATTools.cpp" "CVATTools.h")

target_link_libraries(CVATTools PRIVATE pugixml ${OpenCV_LIBS})
set_property(TARGET CVATTools PROPERTY CXX_STANDARD 20)



#FetchContent_Declare(
#  Catch2
#  URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.1.0.tar.gz
#)
#FetchContent_MakeAvailable(Catch2)
#
#add_executable(tests "tests/test.cpp")
#target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)


# TODO: Add tests and install targets if needed.
