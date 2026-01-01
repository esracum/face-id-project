cmake_minimum_required(VERSION 3.10)
project(faceid)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- OpenCV ----------
find_package(OpenCV REQUIRED)

# ---------- ONNX Runtime ----------
set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/onnxruntime)

if(NOT EXISTS ${ONNXRUNTIME_DIR})
    message(FATAL_ERROR "onnxruntime klasoru bulunamadi!")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

link_directories(
    ${ONNXRUNTIME_DIR}/lib
)

# ---------- Executable ----------
add_executable(faceid try_embed.cpp)

target_link_libraries(faceid
    ${OpenCV_LIBS}
    onnxruntime
)

# ---------- Runtime library path (Linux icin COK ONEMLI) ----------
set_target_properties(faceid PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
)
