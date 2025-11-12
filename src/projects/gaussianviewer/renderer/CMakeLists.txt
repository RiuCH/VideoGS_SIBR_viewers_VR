# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr

set(SIBR_PROJECT "gaussian")
# Add CUDA to the languages
project(sibr_${SIBR_PROJECT} LANGUAGES CXX CUDA)


#	GIT_REPOSITORY 	"https://github.com/r4dl/StopThePop-Rasterization.git"
#    GIT_TAG			"main"

sibr_gitlibrary(TARGET CudaRasterizer
    GIT_REPOSITORY 	"https://github.com/graphdeco-inria/diff-gaussian-rasterization.git"
    GIT_TAG			"255c532d5fd0c3d83afec0e8e49574b65ae381bd"
)

#sibr_gitlibrary(TARGET CudaRasterizer
#    GIT_REPOSITORY 	"https://github.com/graphdeco-inria/diff-gaussian-rasterization.git"
#    GIT_TAG			"9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0"
#)

find_package(CUDAToolkit REQUIRED)
# find curl
find_package(CURL REQUIRED)
find_package(TBB REQUIRED)

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp")
# Add a glob for .cu files
file(GLOB CUDA_SOURCES "*.cu" "*.hpp")
source_group("Source Files" FILES ${SOURCES})
# Add a source group for CUDA files
source_group("Source Files\\CUDA" FILES ${CUDA_SOURCES})

file(GLOB SHADERS "shaders/*.frag" "shaders/*.vert" "shaders/*.geom")
source_group("Source Files\\shaders" FILES ${SHADERS})

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp" "shaders/*.frag" "shaders/*.vert" "shaders/*.geom")

## Specify target rules
# Add CUDA_SOURCES to the library
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${CUDA_SOURCES})

include_directories(${Boost_INCLUDE_DIRS} .)
if (WIN32)
target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${ASSIMP_LIBRARIES}
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
	${OpenCV_LIBRARIES}
	${FFMPEG_LIBRARIES}
	TBB::tbb
	curl
	glfw3
	picojson
	sibr_system
	sibr_view
	sibr_assets
	sibr_renderer
	sibr_basic
	CUDA::cudart
	CUDA::nvml
	CudaRasterizer
)
else()
target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${ASSIMP_LIBRARIES}
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
	${OpenCV_LIBRARIES}
	${FFMPEG_LIBRARIES}
	TBB::tbb
	curl
	glfw
	picojson
	sibr_system
	sibr_view
	sibr_assets
	sibr_renderer
	sibr_basic
	CUDA::cudart
	CUDA::nvml
	CudaRasterizer
)
endif()

add_definitions( -DSIBR_EXP_ULR_EXPORTS -DBOOST_ALL_DYN_LINK  )

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "projects/${SIBR_PROJECT}/renderer")

## High level macro to install in an homogen way all our ibr targets
include(install_runtime)
ibr_install_target(${PROJECT_NAME}
    INSTALL_PDB                       
  ## mean install also MSVC IDE *.pdb file (DEST according to target type)
	SHADERS ${SHADERS}
	RSC_FOLDER ${SIBR_PROJECT}

    #STANDALONE  ${INSTALL_STANDALONE}   ## mean call install_runtime with bundle dependencies resolution
    COMPONENT   ${PROJECT_NAME}_install ## will create custom target to install only this project
)