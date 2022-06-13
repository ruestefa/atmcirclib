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

env_names=("${@}")
default_env_names=("${run_env_name}" "${dev_env_name}")
if [ ${#env_names[@]} -eq 0 ]; then
    env_names=("${default_env_names[@]}")
fi

UPDATE=${UPDATE:-true}
if ${UPDATE}; then
    echo "update environments from requirements"
else
    echo "recreate environments"
fi

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
    echo "error: neither mamba nor conda detected" >&2
    return 1
}

CONDA=$(detect_conda) || exit
cmd=(${CONDA} --version)
echo "\$ ${cmd[@]^Q}"
eval "${cmd[@]}" || exit

remove_existing_env()
{
    local env_name="${1}"
    if $(eval ${CONDA} info --env | \grep -q "^\<${env_name}\>"); then
        echo "remove conda env '${env_name}'"
        local cmd=(${CONDA} env remove -n "${env_name}")
        echo "\$ ${cmd[@]^Q}"
        eval "${cmd[@]}" || return 1
    fi
}

create_updated_env()
{
    local env_name="${1}"
    local env_file="${2}"
    shift 2
    local reqs_files=("${@}")
    echo "create up-to-date conda env '${env_name}' from ${reqs_files[@]}"
    local reqs_file_flags=()
    local reqs_file
    for reqs_file in "${reqs_files[@]}"; do
        reqs_file_flags+=(--file="${reqs_file}")
    done
    local cmd=(${CONDA} create --yes -n "${env_name}" python==${PYTHON_VERSION} "${reqs_file_flags[@]}")
    echo "\$ ${cmd[@]^Q}"
    eval "${cmd[@]}" || return 1
    echo "export conda env '${env_name}' to ${env_file}"
    local cmd=(${CONDA} env export -n "${env_name}" --no-builds)
    echo "\$ ${cmd[@]^Q}"
    eval "${cmd[@]}" > "${env_file}" || return 1
    return 0
}

recreate_env()
{
    local env_name="${1}"
    local env_file="${2}"
    echo "recreate conda env '${env_name}' from ${env_file}"
    local cmd=(${CONDA} env create -n "${env_name}" python==${PYTHON_VERSION} --file="${env_file}")
    echo "\$ ${cmd[@]^Q}"
    eval "${cmd[@]}" || return 1
    return 0
}

for env_name in "${env_names[@]}"; do
    remove_existing_env "${env_name}"
    case "${env_name}/${UPDATE}" in
        "${run_env_name}/false")
            recreate_env "${run_env_name}" "${run_env_file}" || exit
        ;;
        "${dev_env_name}/false")
            recreate_env "${dev_env_name}" "${dev_env_file}" || exit
        ;;
        "${run_env_name}/true")
            create_updated_env "${run_env_name}" "${run_env_file}" "${run_reqs_file}" || exit
        ;;
        "${dev_env_name}/true")
            create_updated_env "${dev_env_name}" "${dev_env_file}" "${run_reqs_file}" "${dev_reqs_file}" || exit
        ;;
        *)
            echo "error: invalid env_name '${env_name}'; choices: ${default_env_names[@]}" >&2
            exit 1
        ;;
    esac
done
