#!/bin/bash
# Install project in editable mode

# Show verbose output
VERBOSE=${VERBOSE:-true}

# Force installation even if no conda environment is active
FORCE_ACTIVE=${FORCE_ACTIVE:-true}

# Build extension modules
BUILD_EXT=${BUILD_EXT:-true}

# Also install projects that are linked in links/
INSTALL_LINKED=${INSTALL_LINKED:-false}

# Build extension modules of linked projects
BUILD_EXT_LINKED=${BUILD_EXT_LINKED:-${BUILD_EXT}}

if ${FORCE_ACTIVE} && [[ "${CONDA_PREFIX}" == "" ]]; then
    echo "error: no active conda environment found" >&2
    echo "(if you know what you do, you may set FORCE_ACTIVE=false)" >&2
    exit 1
fi

main()
{
    install_editable . ${BUILD_EXT} || return
    if ${INSTALL_LINKED}; then
        local linked=($(detect_linked || exit)) || return
        echo ${linked[@]}
        local path
        for path in "${linked[@]}"; do
            install_editable "${path}" ${BUILD_EXT_LINKED} || return
        done
    fi
}

install_editable()
{
    local path="${1}"
    local build_ext="${2}"
    local setup_py="${path}/setup.py"
    if ${build_ext} && [[ -f "${setup_py}" ]]; then
        # If setup.py is present, build extension modules (e.g., f2py, cython)
        # first as, e.g., skbuild doesn't do so during editable install
        # Note that no check is done whether there are extension modules as it is
        # not obvious how to reliably perform such a check
        (
            run cd "${path}" || exit
            run python setup.py build_ext --inplace || exit
        ) || return
    fi
    run python -m pip install --no-deps --ignore-installed -e "${path}" || return
}

run()
{
    local cmd=("${@}")
    ${VERBOSE} && echo "RUN ${cmd[@]^Q}"
    eval "${cmd[@]^Q}" || return
}

detect_linked()
{
    local links_dir="${1:-./links}"
    check_is_dir "${links_dir}" 2>/dev/null || return
    local link
    for link in "${links_dir}"/*; do
        check_is_dir_link "${link}" 2>/dev/null || continue
        \readlink -f "$(\readlink -f "${link}" || exit)/../.." || return
    done
}

# Check that path exists and is a directory
check_is_dir()
{
    local path="${1}"
    if [[ -d "${path}" ]]; then
        return 0
    elif [[ -f "${path}" ]]; then
        echo "error: path exists, but is a file instead of a directory: ${path}" >&2
        return 1
    else
        echo "error: directory does not exist: ${path}" >&2
        return 1
    fi
}

# Check that path exists and is a file
check_is_file()
{
    local path="${1}"
    if [[ -f "${path}" ]]; then
        return 0
    elif [[ -d "${path}" ]]; then
        echo "error: path exists, but is a directory instead of a file: ${path}" >&2
        return 1
    else
        echo "error: file does not exist: ${path}" >&2
        return 1
    fi
}

# Check that path exists and is a symlink
check_is_link()
{
    local path="${1}"
    if [[ -L "${path}" ]]; then
        return 0
    elif [[ -d "${path}" ]]; then
        echo "error: path exists, but is a directory instead of a symlink: ${path}" >&2
        return 1
    elif [[ -f "${path}" ]]; then
        echo "error: path exists, but is a file instead of a symlink: ${path}" >&2
        return 1
    else
        echo "error: symlink does not exist: ${path}" >&2
        return 1
    fi
}

# Check that path exists and is a symlink to a directory
check_is_dir_link()
{
    local path="${1}"
    check_is_link "${path}" || return
    check_is_dir "${path}" || return
    return 0
}

main "${@}"
