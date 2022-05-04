#!/bin/bash

get_script_dir()
{
    # src: https://stackoverflow.com/a/53122736/4419816
    \cd "$(dirname "${BASH_SOURCE[0]}")" && \pwd
}

update_min_versions()
{
    local infile="${1}"
    local envfile="${2}"
    local infile_tmp="${infile}.tmp"
    [ -f "${infile_tmp}" ] && {
        echo "error: tmp infile '${infile_tmp}' already exists" >&2
        return 1
    }
    \mv -v "${infile}" "${infile_tmp}"
    echo "update_min_versions.sh '${infile_tmp}' '${envfile}' > '${infile}'"
    bash $(get_script_dir)/update_min_versions.sh "${infile_tmp}" "${envfile}" > "${infile}" || return
    \rm -v "${infile_tmp}"
}

update_min_versions requirements.in environment.yml || exit
update_min_versions dev-requirements.in dev-environment.yml || exit
