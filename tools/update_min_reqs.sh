#!/bin/bash

# Get path to this script (src: https://stackoverflow.com/a/53122736/4419816)
get_script_dir() { \cd "$(dirname "${BASH_SOURCE[0]}")" && \pwd; }


DBG=${DBG:-false}  # true or false
${DBG} && echo "DBG EXE $(get_script_dir)/$(basename ${0})"


USAGE="Usage: $(basename "${0}") reqfile envfile

Update minimum versions of requirements to minor version in environment file

Example: $(basename "${0}") requirements.yml environment.yml
"


main()
{
    if [[ "${#}" -ne 2 ]]; then
        echo -e "Error: expected 2 args, got ${#}\n\n${USAGE}" >&2
        return 1
    fi
    local reqfile="${1}"
    local envfile="${2}"
    check_file_exists "reqfile" "${reqfile}" || { echo -e "\n${USAGE}" >&2; return 1; }
    check_file_exists "envfile" "${envfile}" || { echo -e "\n${USAGE}" >&2; return 1; }
    local pip_indent="$(get_pip_indent "${reqfile}")"
    ${DBG} && echo "DBG pip_indent='${pip_indent}'" >&2
    update_min_version_file "${reqfile}" "${envfile}" "${pip_indent}" || return
}


process_requirements_line()
{
    local line="${1}"
    local env="${2}"
    ${DBG} && echo "DBG line='${line}'" >&2
    local prefix="$(echo "${line}" | \sed 's/^\( \+- \+\)\?.*$/\1/')"
    local comment="$(echo "${line}" | \sed 's/^\([^#]*[^ ]\)\?\( *#.*\)\?$/\2/')"
    local content="$(echo "${line}" | \sed "s/${prefix}\(.*\)${comment}$/\1/")"
    local package="$(echo "${content}" | \sed 's/^ *\([a-zA-Z0-9_-]\+\)\(.*\)$/\1/')"
    local spec="$(echo "${content}" | \sed 's/^ *\([a-zA-Z0-9_-]\+\)\(.*\)$/\2/')"
    local indent="$(echo -n "${line}" | \sed 's/^\( *\)[^ ].*$/\1/' | \wc -c)"
    ${DBG} && echo "DBG -> prefix='${prefix}'" >&2
    ${DBG} && echo "DBG -> content='${content}'" >&2
    ${DBG} && echo "DBG -> comment='${comment}'" >&2
    ${DBG} && echo "DBG -> indent='${indent}'" >&2
    ${DBG} && echo "DBG -> package='${package}'" >&2
    ${DBG} && echo "DBG -> spec='${spec}'" >&2
    ${DBG} && echo "DBG -> grep: '$(echo "${env}" | \grep -- "- ${package}=")'" >&2
    if [[ "${spec}" == ":" ]]; then
        # e.g., '  - pip:'
        ${DBG} && echo "DBG -> spec is ':'" >&2
    elif $(echo "${spec}" | \grep -q '{{.*}}'); then
        # e.g., '  - python>={{ python_version }}'
        ${DBG} && echo "DBG -> spec contains jinja template" >&2
    elif [[ "${indent}" -gt "${pip_indent}" ]]; then
        # pip package
        ${DBG} && echo "DBG -> intent (${indent}) exceeds pip_indent (${pip_indent})" >&2
    else
        echo "${env}" | \grep -qi -- "- ${package}="
        if [[ "${?}" -eq 0 ]]; then
            local rx='s/ *- \([^=]\+\)=\([0-9]\+\(\.[0-9]\+\)\?\).*/\1>=\2/'
            content="$(echo "${env}" | \grep -i -- "- ${package}=" | \sed "${rx}")"
        fi
    fi
    ${DBG} && echo "DBG -> content='${content}'" >&2
    local new_line="${prefix}${content}${comment}"
    ${DBG} && echo "DBG -> new_line='${new_line}'" >&2
    echo "${new_line}"
}


update_min_version_file()
{
    local reqfile="${1}"
    local envfile="${2}"
    local pip_indent="${3}"
    local reqfile_tmp="${reqfile}.tmp"
    local vflag="$(${DBG} && echo "-v")"
    \mv ${vflag} "${reqfile}" "${reqfile_tmp}" >&2
    update_min_versions "${reqfile_tmp}" "${envfile}" > "${reqfile}" || {
        ${DBG} && echo "DBG restore reqfile '${reqfile}' from '${reqfile_tmp}'" >&2
        \mv -v "${reqfile_tmp}" "${reqfile}" >&2
        return 1
    }
    \rm ${vflag} "${reqfile_tmp}" >&2
}


update_min_versions()
{
    local infile="${1}"
    local envfile="${2}"
    local env="$(cat "${envfile}")"
    local line
    while IFS='' read -r line; do
        process_requirements_line "${line}" "${env}" || return
    done < "${infile}"
}


get_pip_indent()
{
    local reqfile="${1}"
    local pip_line="$(\grep '^ \+- \+pip *:' "${reqfile}")"
    if [[ "${pip_line}" == "" ]]; then
        # No `  - pip:` entry
        echo "-1"
        return
    fi
    echo -n "${pip_line}" | \sed 's/^\( \+\)-.*$/\1/' | \wc -c
}


check_file_exists()
{
    local name="${1}"
    local path="${2}"
    if [ ! -f "${path}" ]; then
        echo "error: missing ${name}: ${path}" >&2
        return 1
    fi
}


main "${@}"
