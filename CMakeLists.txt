cmake_minimum_required(VERSION 2.8.0 FATAL_ERROR)
# Project Name
PROJECT(HW_OPENGL)

#########################################################
# FIND GLUT
#########################################################
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})
if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)
#########################################################
# FIND OPENGL
#########################################################
find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)
#########################################################
# FIND CUDA 
#########################################################
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
add_definitions(${CUDA_DEFINITIONS})
if(NOT CUDA_FOUND)
    message(ERROR " CUDA not found!")
endif(NOT CUDA_FOUND)
########################################################
# Linking & stuff
#########################################################
SET(CMAKE_CXX_FLAGS "-std=c++11 -Ofast")
SET(CUDA_NVCC_FLAGS "-std=c++11 -O0")

#########################################################
# Include Files
#########################################################
cuda_add_executable(life life.cu)
cuda_add_executable(mycuda mycuda.cu)

# create the program "test"
target_link_libraries(life ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${CUDA_LIBRARIES})
target_link_libraries(mycuda ${CUDA_LIBRARIES})
