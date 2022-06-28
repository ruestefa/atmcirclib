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


process_requirements_line()
{
    local line="${1}"
    local env="${2}"
    ${DBG} && echo "DBG line='${line}'" >&2
    local comment="$(echo "${line}" | \sed 's/^\([^#]*[^ ]\)\?\( *#.*\)\?$/\2/')"
    local content="${line%${comment}}"
    local package="$(echo "${content}" | \sed 's/^ *\([a-zA-Z0-9_-]\+\).*/\1/')"
    ${DBG} && echo "DBG -> content='${content}'" >&2
    ${DBG} && echo "DBG -> package='${package}'" >&2
    ${DBG} && echo "DBG -> comment='${comment}'" >&2
    ${DBG} && echo "DBG -> grep: '$(echo "${env}" | \grep -- "- ${package}=")'" >&2
    echo "${env}" | \grep -qi -- "- ${package}="
    if [[ "${?}" -eq 0 ]]; then
        local rx='s/ *- \([^=]\+\)=\([0-9]\+\(\.[0-9]\+\)\?\).*/\1>=\2/'
        content="$(echo "${env}" | \grep -i -- "- ${package}=" | \sed "${rx}")"
    fi
    ${DBG} && echo "DBG -> content='${content}'" >&2
    ${DBG} && echo "DBG -> new_line='${content}${comment}'" >&2
    echo "${content}${comment}"
}


update_min_versions()
{
    local infile="${1}"
    local envfile="${2}"
    local env="$(get_env "${envfile}")" || return
    local line
    while read -r line; do
        process_requirements_line "${line}" "${env}" || return
    done < "${infile}"
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
    local vflag="$(${DBG} && echo "-v")"
    \mv ${vflag} "${infile}" "${infile_tmp}" >&2
    update_min_versions "${infile_tmp}" "${envfile}" > "${infile}" || {
        ${DBG} && echo "DBG restore infile '${infile}' from '${infile_tmp}'" >&2
        \mv -v "${infile_tmp}" "${infile}" >&2
        return 1
    }
    \rm ${vflag} "${infile_tmp}" >&2
}


main()
{
    case ${#} in
        0) local prefixes=("" "dev-");;
        *) local prefixes=("${@}");;
    esac
    local prefix
    for prefix in "${prefixes[@]}"; do
        ${DBG} && echo "DBG prefix='${prefix}'" >&2
        local infile="${prefix}requirements.in"
        local envfile="${prefix}environment.yml"
        echo "${infile} <- $([[ "${envfile}" != "" ]] && echo "${envfile}" || echo "conda")"
        update_min_version_file "${infile}" "${envfile}" || return
    done
}


main "${@}"
