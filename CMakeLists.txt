cmake_minimum_required(VERSION 3.22.0)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#-------------------------------------------------------------------------------
# Fetch input/environment variables
#-------------------------------------------------------------------------------

if (NOT DEFINED($ENV{CMAKE_PYTHON_VERSION}))
    set(CMAKE_PYTHON_VERSION 3.9)
endif()
if (DEFINED($ENV{CMAKE_PREFIX_PATH}))
    set(CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
elseif (NOT DEFINED CMAKE_PREFIX_PATH)
    message(
        FATAL_ERROR
        "Must set CMAKE_PREFIX_PATH to the root of your Python environment.\n"
    )
endif()

message("-- Setup:")
message("   CMAKE_PYTHON_VERSION = ${CMAKE_PYTHON_VERSION}")
message("   CMAKE_PREFIX_PATH    = ${CMAKE_PREFIX_PATH}")
message("--")

#-------------------------------------------------------------------------------
# Define project metadata
#-------------------------------------------------------------------------------

project(
    "atmcirclib"
    LANGUAGES C Fortran
)

#-------------------------------------------------------------------------------
# Prepare cmake and scikit-build
#-------------------------------------------------------------------------------

# Safety net
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(
        FATAL_ERROR
        "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.\n"
    )
endif()

# Ensure scikit-build modules
if (NOT SKBUILD)
    # Kanged --> https://github.com/Kitware/torch_liberator/blob/master/CMakeLists.txt
    # If skbuild is not the driver; include its utilities in CMAKE_MODULE_PATH
    execute_process(
        COMMAND "${Python3_EXECUTABLE}"
        -c "import os, skbuild; print(os.path.dirname(skbuild.__file__))"
        OUTPUT_VARIABLE SKBLD_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(SKBLD_CMAKE_DIR "${SKBLD_DIR}/resources/cmake")
    list(APPEND CMAKE_MODULE_PATH ${SKBLD_CMAKE_DIR})
endif()

#-------------------------------------------------------------------------------
# Set up Python
#-------------------------------------------------------------------------------

# Set Python root (probably not even necessary)
set(Python_FIND_STRATEGY LOCATION)
set(Python_ROOT ${CMAKE_PREFIX_PATH})

# Manually set numpy include dir to ensure that `find_package(NumPy, ...)` below finds the correct
# installation; may otherwise, e.g., chose system installation over active conda environment (!)
set(NumPy_INCLUDE_DIR ${CMAKE_PREFIX_PATH}/lib/python${CMAKE_PYTHON_VERSION}/site-packages/numpy/core/include)
set(F2PY_INCLUDE_DIR ${CMAKE_PREFIX_PATH}/lib/python${CMAKE_PYTHON_VERSION}/site-packages/numpy/f2py/src)

# Locate Python and relevant packages/includes
find_package(Python3 ${CMAKE_PYTHON_VERSION} REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(PythonExtensions REQUIRED) # for ${PYTHON_EXTENSION_MODULE_SUFFIX}
find_package(PythonExtensions REQUIRED) # for ${PYTHON_EXTENSION_MODULE_SUFFIX}
find_package(NumPy REQUIRED MODULE) # for ${NumPy_INCLUDE_DIRS}
find_package(F2PY REQUIRED) # for ${F2PY_INCLUDE_DIRS}

message("-- Python paths:")
message("   Python_FIND_STRATEGY = ${Python_FIND_STRATEGY}")
message("   Python_ROOT          = ${Python_ROOT}")
message("   NumPy_INCLUDE_DIR    = ${NumPy_INCLUDE_DIR}")
message("   NumPy_INCLUDE_DIRS   = ${NumPy_INCLUDE_DIRS}")
message("   F2PY_INCLUDE_DIR     = ${F2PY_INCLUDE_DIR}")
message("   F2PY_INCLUDE_DIRS    = ${F2PY_INCLUDE_DIRS}")
message("--")

#-------------------------------------------------------------------------------
# Build Fortran extension modules
#-------------------------------------------------------------------------------

macro(f2py rpath_mod_f)
    set(apath_mod_f "${CMAKE_SOURCE_DIR}/${rpath_mod_f}")
    get_filename_component(mod "${rpath_mod_f}" NAME_WLE)
    get_filename_component(rpath "${rpath_mod_f}" DIRECTORY)
    string(REGEX REPLACE "/" "__" _pre "${rpath}")
    set(_pre_mod "${_pre}${mod}")
    set(mod_c "${mod}module.c")
    set(_pre_mod_c "${_pre_mod}module.c")
    set(rpath_pre_mod_c "${rpath}/${_pre_mod_c}")
    set(mod_so "${mod}${PYTHON_EXTENSION_MODULE_SUFFIX}")
    set(pre_mod_so "${_pre}${mod_so}")
    set(rpath_pre_mod_so "${rpath}/${pre_mod_so}")

    message("-- Extension module:")
    message("   rpath_mod_f      = ${rpath_mod_f}")
    message("   apath_mod_f      = ${apath_mod_f}")
    message("   mod              = ${mod}")
    message("   rpath            = ${rpath}")
    message("   _pre             = ${_pre}")
    message("   _pre_mod         = ${_pre_mod}")
    message("   mod_c            = ${mod_c}")
    message("   _pre_mod_c       = ${_pre_mod_c}")
    message("   rpath_pre_mod_c  = ${rpath_pre_mod_c}")
    message("   mod_so           = ${mod_so}")
    message("   pre_mod_so       = ${pre_mod_so}")
    message("   rpath_pre_mod_so = ${rpath_pre_mod_so}")
    message("--")

    # Use `f2py` to create `prefix_modmodule.c` from `mod.f90`
    add_custom_command(
        OUTPUT "${rpath_pre_mod_c}"
        COMMAND ${F2PY_EXECUTABLE} -m "${mod}" "${apath_mod_f}" --build-dir "${rpath}" --lower
        COMMAND mv -v "${rpath}/${mod_c}" "${rpath_pre_mod_c}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        DEPENDS "${apath_mod_f}"
    )

    # Create `pefix_mod.cpython-39.x86_64-linux-gnu.so` from `prefix_modmodule.c` and `mod.f90`
    add_library(
        "${pre_mod_so}" MODULE
        "${rpath_pre_mod_c}"
        "${F2PY_INCLUDE_DIR}/fortranobject.c"
        "${apath_mod_f}"
    )
    set_target_properties(
        "${pre_mod_so}" PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${rpath}"
        PREFIX ""
        SUFFIX ""
    )
    target_include_directories("${pre_mod_so}" PUBLIC ${F2PY_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

    # Set platform-specific linker flags
    if (UNIX)
        if (APPLE)
            set(mod_so_link_flags "-Wl,-dylib,-undefined,dynamic_lookup")
        else()
            set(mod_so_link_flags "-Wl,--allow-shlib-undefined")
        endif()
        set_target_properties("${pre_mod_so}" PROPERTIES LINK_FLAGS "${mod_so_link_flags}")
    endif()

    # Install `pefix_mod.cpython-39.x86_64-linux-gnu.so` as `mod.cpython-...-gnu.so` in source tree
    if (SKBUILD)
        set(install_dir_mod_so "${rpath}")
    else()
        set(install_dir_mod_so "${CMAKE_SOURCE_DIR}/${rpath}")
    endif()
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/${rpath_pre_mod_so}"
        DESTINATION "${install_dir_mod_so}"
        RENAME "${mod_so}"
    )
endmacro()

f2py("src/atmcirclib/deriv/ext/_f77/deriv.f")
f2py("src/atmcirclib/deriv/ext/_f90/deriv.f90")

#-------------------------------------------------------------------------------
# Done
#-------------------------------------------------------------------------------
