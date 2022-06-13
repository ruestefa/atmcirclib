#!/bin/bash
# Install project in editable mode

# Force installation even if no conda environment is active
FORCE=${FORCE:-false}

# Also install libs that are linked in links/
LINKED=${LINKED:-true}

if [[ "${CONDA_PREFIX}" == "" ]] && ! ${FORCE}; then
    echo "error: no active conda environment found; activate one or set FORCE=true" >&2
    exit 1
fi

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

install_editable()
{
    local path="${1}"
    cmd=(python -m pip install --no-deps -e "${path}")
    echo -e "\nRUN ${cmd[@]^Q}"
    eval "${cmd[@]}" || return
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

main()
{
    install_editable . || return
    if ${LINKED}; then
        local linked=($(detect_linked || exit)) || return
        echo ${linked[@]}
        local path
        for path in "${linked[@]}"; do
            install_editable "${path}" || return
        done
    fi
}

main "${@}"
