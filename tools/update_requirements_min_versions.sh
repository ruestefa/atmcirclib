#!/bin/bash

# Get path to this script (src: https://stackoverflow.com/a/53122736/4419816)
get_script_dir() { \cd "$(dirname "${BASH_SOURCE[0]}")" && \pwd; }


DBG=${DBG:-false}  # true or false
${DBG} && echo "DBG EXE $(get_script_dir)/$(basename ${0})"


print_usage()
{
    local exe="$(basename "${0}")"
    echo "usage: ${exe} infile [envfile]"
    echo ""
    echo "update minimum versions of requirements to minor version in environment or environment file"
    echo ""
    echo "examples:"
    echo "  ${exe} requirements.in"
    echo "  ${exe} dev-requirements.in dev-environment.yml"
}


get_env()
{
    local envfile="${1}"
    if [ -f "${envfile}" ]; then
        ${DBG} && echo "DBG use env file '${envfile}'" >&2
        \cat "${envfile}"
    else
        ${DBG} && echo "DBG env file '${envfile}' not found" >&2
        ${DBG} && echo "DBG obtain environment with 'conda env export --no-builds'" >&2
        conda env export --no-builds || {
            echo "error with command 'conda env export --no-builds'" >&2
            return 1
        }
    fi
}


check_file_exists()
{
    local name="${1}"
    local path="${2}"
    if [[ "${path}" == "" ]]; then
        return 1
    elif [ ! -f "${path}" ]; then
        echo "error: ${name} '${path}' not found" >&2
    fi
}


update_min_versions()
{
    local infile="${1}"
    local envfile="${2}"
    local env="$(get_env "${envfile}")" || return
    local i
    for i in $(cat "${infile}"); do
        ${DBG} && echo "DBG '${i}'" >&2
        i=$(echo $i | \sed -s 's/^\([a-zA-Z0-9_-]\+\).*/\1/g')
        ${DBG} && echo "DBG '${i}'" >&2
        ${DBG} && echo "DBG ${env}" | \grep -- "- $i=" >&2
        echo "${env}" | \grep -q -- "- $i="
        if [[ "${?}" -ne 0 ]]; then
            echo "${i}"
        else
            echo "${env}" | \grep -- "- $i=" | \sed 's/ *- \([^=]\+\)=\([0-9]\+\(\.[0-9]\+\)\?\).*/\1>=\2/'
        fi
    done
}


update_min_version_file()
{
    local infile="${1}"
    local envfile="${2}"
    check_file_exists "infile" "${infile}" || {
        echo >&2
        print_usage >&2
        return 1
    }
    local infile_tmp="${infile}.tmp"
    [ -f "${infile_tmp}" ] && {
        echo "error: tmp infile '${infile_tmp}' already exists" >&2
        return 1
    }
    \mv -v "${infile}" "${infile_tmp}"
    update_min_versions "${infile_tmp}" "${envfile}" > "${infile}" || return
    \rm -v "${infile_tmp}"
}


main()
{
    case ${#} in
        0) local prefixes=("" "dev-");;
        *) local prefixes=("${@}");;
    esac
    local prefix
    for prefix in "${prefixes}"; do
        local infile="${prefix}requirements.in"
        local envfile="${prefix}environment.yml"
        update_min_version_file "${infile}" "${envfile}"
    done
}


main "${@}"
