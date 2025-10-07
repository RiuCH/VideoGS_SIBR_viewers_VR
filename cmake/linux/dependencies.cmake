# Fixed dependencies for Ubuntu system packages

include(Win3rdParty)
include(sibr_library)

Win3rdPartyGlobalCacheAction()

find_library(OPENGL_opengl_LIBRARY NAMES OpenGL PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    /sw/lib
)

find_library(OPENGL_glx_LIBRARY NAMES GLX PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    /sw/lib
    PATH_SUFFIXES libglvnd
)

find_package(OpenGL REQUIRED)
set(OpenGL_GL_PREFERENCE "GLVND")

############
## Find GLEW
############
find_package(EGL QUIET)

if(EGL_FOUND)
    add_definitions(-DGLEW_EGL)
    message("Activating EGL support for headless GLFW/GLEW")
else()
    message("EGL not found : EGL support for headless GLFW/GLEW is disabled")
endif()

# Use pkg-config to find GLEW
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(GLEW QUIET glew)
    if(GLEW_FOUND)
        set(GLEW_INCLUDE_DIR ${GLEW_INCLUDE_DIRS})
        set(GLEW_LIBRARIES ${GLEW_LIBRARIES})
        set(GLEW_LIBRARY ${GLEW_LIBRARIES})
        set(GLEW_SHARED_LIBRARY_RELEASE ${GLEW_LIBRARIES})
    endif()
endif()

if(NOT GLEW_FOUND)
    message("Using manual GLEW paths for Ubuntu")
    set(GLEW_INCLUDE_DIR "/usr/include")
    set(GLEW_SHARED_LIBRARY_RELEASE "/usr/lib/x86_64-linux-gnu/libGLEW.so")
    set(GLEW_LIBRARIES "/usr/lib/x86_64-linux-gnu/libGLEW.so")
    set(GLEW_FOUND TRUE)
endif()

include_directories(${GLEW_INCLUDE_DIR})

##############
## Find ASSIMP
##############
# Try pkg-config first
if(PKG_CONFIG_FOUND)
    pkg_check_modules(ASSIMP QUIET assimp)
endif()

if(NOT ASSIMP_FOUND)
    message("Using manual ASSIMP paths for Ubuntu")
    set(ASSIMP_INCLUDE_DIR "/usr/include/assimp")
    set(ASSIMP_LIBRARY "/usr/lib/x86_64-linux-gnu/libassimp.so")
    set(ASSIMP_LIBRARIES "/usr/lib/x86_64-linux-gnu/libassimp.so")
    set(ASSIMP_FOUND TRUE)
endif()

include_directories(${ASSIMP_INCLUDE_DIR})

################
## Find FFMPEG (make it optional)
################
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(FFMPEG QUIET libavcodec libavformat libavutil libswscale)
endif()
if(FFMPEG_FOUND)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    message("FFmpeg found")
else()
    message(WARNING "FFmpeg not found, some features may be disabled")
endif()

###################
## Find embree (make it optional)
###################
find_package(embree 3.0 QUIET)
if(NOT embree_FOUND)
    find_package(embree 4.0 QUIET)
endif()
if(embree_FOUND)
    message("Embree found")
else()
    message(WARNING "Embree not found, some features may be disabled")
endif()

###################
## Find eigen3
###################
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(EIGEN3 QUIET eigen3)
endif()
if(NOT EIGEN3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR Eigen/Core HINTS /usr/include/eigen3)
    if(EIGEN3_INCLUDE_DIR)
        set(EIGEN3_FOUND TRUE)
    endif()
endif()
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR} /usr/include/eigen3)
    add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_ZERO)
    message("Eigen3 found")
else()
    message(WARNING "Eigen3 not found")
endif()

#############
## Find Boost
#############
set(Boost_REQUIRED_COMPONENTS "system;chrono;filesystem;date_time" CACHE INTERNAL "Boost Required Components")

# Try modern CMake config first, then fallback to old module finder
set(Boost_DIR "/usr/lib/x86_64-linux-gnu/cmake/Boost-1.83.0")
find_package(Boost 1.65.0 QUIET CONFIG COMPONENTS system chrono filesystem date_time)

if(NOT Boost_FOUND)
    # Fallback to module finder with system paths
    set(Boost_NO_BOOST_CMAKE OFF)
    set(Boost_USE_SYSTEM_PATHS ON)
    find_package(Boost 1.65.0 REQUIRED COMPONENTS ${Boost_REQUIRED_COMPONENTS})
endif()

if(WIN32)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/EHsc>")
endif()

if(Boost_LIB_DIAGNOSTIC_DEFINITIONS)
    add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})
endif()

add_definitions(-DBOOST_ALL_DYN_LINK -DBOOST_ALL_NO_LIB)

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

##############
## Find OpenMP
##############
find_package(OpenMP)

##############
## Find OpenCV
##############
# Set OpenCV directory for Ubuntu
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV 4.0 REQUIRED)

# Set properties only if OpenCV_LIBS is not empty
if(OpenCV_LIBS)
    set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
endif()

add_definitions(-DOPENCV_TRAITS_ENABLE_DEPRECATED) 

if(OpenCV_INCLUDE_DIRS)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

###################
## Find GLFW
###################
# Set GLFW3 directory for Ubuntu
set(glfw3_DIR "/usr/lib/x86_64-linux-gnu/cmake/glfw3")
find_package(glfw3 REQUIRED)

###################
## Git libraries (keep as is)
###################
sibr_gitlibrary(TARGET openxr_loader
    GIT_REPOSITORY  "https://github.com/KhronosGroup/OpenXR-SDK.git"
    GIT_TAG         "release-1.0.29"
)

sibr_gitlibrary(TARGET imgui
    GIT_REPOSITORY  "https://gitlab.inria.fr/sibr/libs/imgui.git"
    GIT_TAG         "741fb3ab6c7e1f7cef23ad0501a06b7c2b354944"
)

sibr_gitlibrary(TARGET nativefiledialog
    GIT_REPOSITORY  "https://gitlab.inria.fr/sibr/libs/nativefiledialog.git"
    GIT_TAG         "ae2fab73cf44bebdc08d997e307c8df30bb9acec"
)

sibr_gitlibrary(TARGET mrf
    GIT_REPOSITORY  "https://gitlab.inria.fr/sibr/libs/mrf.git"
    GIT_TAG         "30c3c9494a00b6346d72a9e37761824c6f2b7207"
)

sibr_gitlibrary(TARGET nanoflann
    GIT_REPOSITORY  "https://gitlab.inria.fr/sibr/libs/nanoflann.git"
    GIT_TAG         "7a20a9ac0a1d34850fc3a9e398fc4a7618e8a69a"
)

sibr_gitlibrary(TARGET picojson
    GIT_REPOSITORY  "https://gitlab.inria.fr/sibr/libs/picojson.git"
    GIT_TAG         "7cf8feee93c8383dddbcb6b64cf40b04e007c49f"
)

sibr_gitlibrary(TARGET rapidxml
    GIT_REPOSITORY  "https://gitlab.inria.fr/sibr/libs/rapidxml.git"
    GIT_TAG         "069e87f5ec5ce1745253bd64d89644d6b894e516"
)

sibr_gitlibrary(TARGET xatlas
    GIT_REPOSITORY  "https://gitlab.inria.fr/sibr/libs/xatlas.git"
    GIT_TAG         "0fbe06a5368da13fcdc3ee48d4bdb2919ed2a249"
    INCLUDE_DIRS    "source/xatlas"
)

Win3rdPartyGlobalCacheAction()
