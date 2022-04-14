#!/bin/bash
# Create run and dev conda environments and export them.

if [[ "${CONDA_PREFIX}" != "" ]]; then
    echo "please deactivate conda env and retry (detected '${CONDA_PREFIX}')" >&2
    exit 1
fi

python -c 'import git' || { echo "Python module 'git' must be installed" >&2; exit 1; }

get_repo_name()
{
    python -c 'from pathlib import Path; from git import Repo; print(Path(Repo(".", search_parent_directories=True).working_tree_dir).name)' || return 1
}

PYTHON_VERSION=3.9
repo_name=$(get_repo_name) || { echo "could not determine repo name" >&2; exit 1; }
run_env_name="${repo_name}"
dev_env_name="${repo_name}-dev"
run_reqs_file="requirements.in"
dev_reqs_file="dev-requirements.in"
run_env_file="environment.yml"
dev_env_file="dev-environment.yml"

detect_conda()
{
    mamba --version 1>/dev/null 2>&1
    if [[ ${?} -eq 0 ]]; then
        echo "MAMBA_NO_BANNER=1 mamba"
        return 0
    fi
    conda --version 1>/dev/null 2>&1
    if [[ ${?} -eq 0 ]]; then
        echo conda
        return 0
    fi
    echo "error: neither mamba nor conda detectec" >&2
    return 1
}

CONDA=$(detect_conda) || exit
cmd=(${CONDA} --version)
echo "\$ ${cmd[@]^Q}"
eval "${cmd[@]}" || exit

create_env()
{
    local env_name="${1}"
    local env_file="${2}"
    shift 2
    local reqs_files=("${@}")
    local reqs_file_flags=()
    local reqs_file
    if $(eval ${CONDA} info --env | \grep -q "^\<${env_name}\>"); then
        echo "remove conda env '${env_name}'"
        cmd=(${CONDA} env remove -n "${env_name}")
        echo "\$ ${cmd[@]^Q}"
        eval "${cmd[@]}" || return 1
    fi

    echo "create conda env '${env_name}' from ${reqs_files[@]}"
    for reqs_file in "${reqs_files[@]}"; do
        reqs_file_flags+=(--file="${reqs_file}")
    done
    cmd=(${CONDA} create -n "${env_name}" python==${PYTHON_VERSION} "${reqs_file_flags[@]}" --yes )
    echo "\$ ${cmd[@]^Q}"
    eval "${cmd[@]}" || return 1

    echo "export conda env '${env_name}' to ${env_file}"
    cmd=(${CONDA} env export -n "${env_name}" --no-builds)
    echo "\$ ${cmd[@]^Q}"
    eval "${cmd[@]}" > "${env_file}" || return 1

    return 0
}

create_env "${run_env_name}" "${run_env_file}" "${run_reqs_file}" || exit
create_env "${dev_env_name}" "${dev_env_file}" "${run_reqs_file}" "${dev_reqs_file}" || exit
