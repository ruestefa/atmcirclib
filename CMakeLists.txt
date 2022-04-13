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
    atmcirclib
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

message("-- Python paths:")
message("   Python_FIND_STRATEGY = ${Python_FIND_STRATEGY}")
message("   Python_ROOT          = ${Python_ROOT}")
message("   NumPy_INCLUDE_DIR    = ${NumPy_INCLUDE_DIR}")
message("--")

# Locate Python and relevant packages/includes
find_package(Python3 ${CMAKE_PYTHON_VERSION} REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(PythonExtensions REQUIRED) # for ${PYTHON_EXTENSION_MODULE_SUFFIX}
find_package(PythonExtensions REQUIRED) # for ${PYTHON_EXTENSION_MODULE_SUFFIX}
message(Python_INCLUDE_DIRS=${Python_INCLUDE_DIRS})
message(Python_SITELIB=${Python_SITELIB})
find_package(NumPy REQUIRED MODULE) # for ${NumPy_INCLUDE_DIRS}
find_package(F2PY REQUIRED) # for ${F2PY_INCLUDE_DIR}

#-------------------------------------------------------------------------------
# Build Fortran extension modules
#-------------------------------------------------------------------------------

set(
    f_src_files_rel
    "src/atmcirclib/deriv/ext/f77/_deriv_f77.f"
    "src/atmcirclib/deriv/ext/f90/_deriv_f90.f90"
)

foreach(f_src_file_rel ${f_src_files_rel})
    set(f_src_file_abs "${CMAKE_SOURCE_DIR}/${f_src_file_rel}")
    get_filename_component(f2py_mod_name ${f_src_file_rel} NAME_WLE)
    get_filename_component(f_src_dir ${f_src_file_rel} DIRECTORY)
    set(f2py_mod_dir "${f_src_dir}/${f2py_mod_name}")
    set(f2py_cmod_file "${f2py_mod_dir}/${f2py_mod_name}module.c")
    set(ext_mod_file "${f2py_mod_name}${PYTHON_EXTENSION_MODULE_SUFFIX}")

    message("-- Extension module:")
    message("   f_src_file_rel = ${f_src_file_rel}")
    message("   f_src_file_abs = ${f_src_file_abs}")
    message("   f_src_dir      = ${f_src_dir}")
    message("   f2py_mod_name  = ${f2py_mod_name}")
    message("   f2py_mod_dir   = ${f2py_mod_dir}")
    message("   f2py_cmod_file = ${f2py_cmod_file}")
    message("   ext_mod_file   = ${ext_mod_file}")
    message("--")

    # Target for enforcing dependencies
    add_custom_target(${f2py_mod_name} ALL DEPENDS "${f_src_file_abs}")

    # Custom command for generating .c
    add_custom_command(
        OUTPUT "${f2py_cmod_file}"
        COMMAND ${F2PY_EXECUTABLE}
            -m ${f2py_mod_name}
            ${f_src_file_abs}
            --build-dir ${f2py_mod_dir}
            --lower
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${f_src_file_abs}
    )

    add_library(
        ${ext_mod_file} MODULE
        "${f2py_cmod_file}"
        "${F2PY_INCLUDE_DIR}/fortranobject.c"
        "${f_src_file_abs}"
    )
    set_target_properties(
        ${ext_mod_file} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${f2py_mod_dir}"
    )

    target_include_directories(
        ${ext_mod_file} PUBLIC
        ${F2PY_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
    )
    set_target_properties(${ext_mod_file} PROPERTIES SUFFIX "")
    set_target_properties(${ext_mod_file} PROPERTIES PREFIX "")

    # Linker fixes
    if (UNIX)
        if (APPLE)
            set_target_properties(
                ${ext_mod_file} PROPERTIES
                LINK_FLAGS "-Wl,-dylib,-undefined,dynamic_lookup"
            )
        else()
            set_target_properties(
                ${ext_mod_file} PROPERTIES
                LINK_FLAGS "-Wl,--allow-shlib-undefined"
            )
        endif()
    endif()

    if (SKBUILD)
        install(TARGETS ${ext_mod_file} DESTINATION ${f_src_dir})
    else()
        install(TARGETS ${ext_mod_file} DESTINATION ${CMAKE_SOURCE_DIR}/${f_src_dir})
    endif()

endforeach()

#-------------------------------------------------------------------------------
# Done
#-------------------------------------------------------------------------------
