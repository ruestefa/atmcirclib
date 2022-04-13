#!/bin/bash
# Create run and dev conda environments and export them.

PYTHON_VERSION=3.9
run_env_name="atmcirclib"
dev_env_name="atmcirclib-dev"
run_reqs_file="requirements.in"
dev_reqs_file="dev-requirements.in"
run_env_file="environment.yml"
dev_env_file="dev-environment.yml"

create_env()
{
    local env_name="${1}"
    local env_file="${2}"
    shift 2
    local reqs_files=("${@}")
    local reqs_file_flags=()
    local reqs_file

    if [[ "${CONDA_PREFIX}" != "" ]]; then
        echo "deactivate conda env '${CONDA_PREFIX}'"
        conda deactivate || return 1
    fi
    if $(conda info --env | \grep -q "^\<${env_name}\>"); then
        echo "remove conda env '${env_name}'"
        conda env remove -n "${env_name}" || return 1
    fi

    echo "create conda env '${env_name}' from ${reqs_files[@]}"
    for reqs_file in "${reqs_files[@]}"; do
        reqs_file_flags+=(--file="${reqs_file}")
    done
    conda create -n "${env_name}" python==${PYTHON_VERSION} "${reqs_file_flags[@]}" --yes || return 1

    echo "export conda env '${env_name}' to ${env_file}"
    conda env export -n "${env_name}" --no-builds > "${env_file}" || return 1

    return 0
}

create_env "${run_env_name}" "${run_env_file}" "${run_reqs_file}" || exit
create_env "${dev_env_name}" "${dev_env_file}" "${run_reqs_file}" "${dev_reqs_file}" || exit
