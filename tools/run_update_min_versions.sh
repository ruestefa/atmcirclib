#!/bin/bash

get_script_dir()
{
    # src: https://stackoverflow.com/a/53122736/4419816
    \cd "$(dirname "${BASH_SOURCE[0]}")" && \pwd
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
        bash $(get_script_dir)/update_min_versions.sh "${infile}" "${envfile}" || exit
    done
}


main "${@}"
