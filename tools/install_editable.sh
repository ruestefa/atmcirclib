#!/bin/bash
# Install project in editable mode

# Build extension modules
BUILD_EXT=${BUILD_EXT:-true}

# Also install projects that are linked in links/
INSTALL_LINKED=${INSTALL_LINKED:-false}

# Build extension modules of linked projects
BUILD_EXT_LINKED=${BUILD_EXT_LINKED:-false}

# Force installation even if no conda environment is active
FORCE_ACTIVE=${FORCE_ACTIVE:-true}

# Show verbose output
VERBOSE=${VERBOSE:-true}

if ${FORCE_ACTIVE} && [[ "${CONDA_PREFIX}" == "" ]]; then
    echo "error: no active conda environment found" >&2
    echo "(if you know what you do, you may set FORCE_ACTIVE=false)" >&2
    exit 1
fi

main()
{
    pip_install_editable . ${BUILD_EXT} || return
    if ${INSTALL_LINKED}; then
        local linked=($(detect_linked || exit)) || return
        echo "found ${#linked[@]} linked project(s)"
        local path
        for path in "${linked[@]}"; do
            echo "link project at ${path}"
            # If linked package is not uninstalled with conda, that installation
            # may have precedence over the editable local installation
            conda_uninstall_from_path "${path}" || return
            pip_install_editable "${path}" ${BUILD_EXT_LINKED} || return
        done
    fi
}

pip_install_editable()
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

conda_uninstall_from_path()
{
    local path="${1}"
    local output pkgs  # put local on separate line to retain return values of assignment commands
    output="$(python -m pip install --no-deps --ignore-installed -e "${path}" --dry-run)" || return
    pkgs=($(echo "${output}" | \grep '^ *Would install' | \sed "s/^ *Would install //")) || return
    local pkg
    for pkg in "${pkgs[@]}"; do
        rx='^\([a-zA-Z0-9_-]\+\)-\([0-9]\+\.[0-9]\+.*\)'
        local pkg_name="$(echo "${pkg}" | \sed "s/${rx}/\1/")" || return
        output="$(run conda uninstall --force "${pkg_name}" --yes 2>&1)"
        if [[ ${?} -ne 0 ]]; then
            if echo "${output}" | \grep -q "PackagesNotFoundError"; then
                continue
            else
                echo "output" >&2
                return 1
            fi
        fi
    done
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
