#!/bin/bash

DBG=true  # true or false

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

infile="${1}"
envfile="${2}"

if [[ "${infile}" == "" ]]; then
    print_usage >&2
    exit 1
elif [ ! -f "${infile}" ]; then
    echo "error: infile '${infile}' not found" >&2
    echo >&2
    print_usage >&2
    exit 1
fi

if [ -f "${envfile}" ]; then
    ${DBG} && echo "DBG use env file '${envfile}'" >&2
    env="$(\cat "${envfile}")"
else
    ${DBG} && echo "DBG env file '${envfile}' not found" >&2
    ${DBG} && echo "DBG obtain environment with 'conda env export --no-builds'" >&2
    env="$(conda env export --no-builds)" || {
        echo "error with command 'conda env export --no-builds'" >&2
        exit 1
    }
fi

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
