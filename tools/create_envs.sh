#!/bin/bash
# Create run and dev conda environments and export them.
# By Stefan Ruedisuehli, stefan.ruedisuehli@env.ethz.ch, 2021/2022

# Check that no conda env is activated
if [[ "${CONDA_PREFIX}" != "" ]]; then
    echo "please deactivate conda env and retry (detected '${CONDA_PREFIX}')" >&2
    exit 1
fi

# Check that python git module is installed
python -c 'import git' || { echo "Python module 'git' must be installed" >&2; exit 1; }

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

# Determine conda command; if available, prefer mamba over conda
CONDA=${CONDA:-$(detect_conda)} || exit
cmd=(${CONDA} --version)
echo "\$ ${cmd[@]^Q}"
eval "${cmd[@]}" || exit

# Specify a python version by setting ${PYTHON}, e.g., PYTHON=3.9
PYTHON=${PYTHON:-}


main()
{
    local env_names=("${@}")

    local repo_name
    repo_name=$(get_repo_name) || return
    local run_env_name="${repo_name}"
    local dev_env_name="${repo_name}-dev"

    local default_env_names=("${run_env_name}" "${dev_env_name}")
    if [ ${#env_names[@]} -eq 0 ]; then
        env_names=("${default_env_names[@]}")
    fi

    local run_reqs_file
    local dev_reqs_file
    run_reqs_file="$(select_first_existing_file "requirements.yml" "requirements.in")" || return
    dev_reqs_file="$(select_first_existing_file "dev-requirements.yml" "dev-requirements.in")" || return
    local run_env_file="environment.yml"
    local dev_env_file="dev-environment.yml"

    UPDATE=${UPDATE:-true}
    if ${UPDATE}; then
        echo "update environments from requirements"
    else
        echo "recreate environments"
    fi

    local env_name
    for env_name in "${env_names[@]}"; do
        remove_existing_env "${env_name}"
        case "${env_name}" in
            "${run_env_name}")
                create_new_env "${run_env_name}" "${run_env_file}" "${run_reqs_file}" || return
            ;;
            "${dev_env_name}")
                create_new_env "${dev_env_name}" "${dev_env_file}" "${run_reqs_file}" "${dev_reqs_file}" || return
            ;;
        esac
    done
}


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


create_new_env()
{
    local env_name="${1}"
    local env_file="${2}"
    shift 2
    local reqs_files=("${@}")
    if ${UPDATE}; then
        recreate_env "${env_name}" "${env_file}" || return
    else
        create_updated_env "${env_name}" "${env_file}" "${reqs_files[@]}" || return
    fi
}


create_updated_env()
{
    local env_name="${1}"
    local env_file="${2}"
    shift 2
    local reqs_files=("${@}")
    echo "create up-to-date conda env '${env_name}' from ${reqs_files[@]}"
    local reqs_files_in=()
    local reqs_files_yml=()
    sort_reqs_files "${reqs_files[@]}" || return
    create_empty_env "${env_name}" || return
    install_reqs_yml "${env_name}" "${reqs_files_yml[@]}" || return
    install_reqs_in "${env_name}" "${reqs_files_in[@]}" || return
    export_env "${env_name}" "${env_file}" || return
    check_python_version || return
}


sort_reqs_files()
{
    local reqs_files=("${@}")
    declare -g -a reqs_files_in
    declare -g -a reqs_files_yml
    local reqs_file
    for reqs_file in "${reqs_files[@]}"; do
        if [[ "${reqs_file: -3:3}" == ".in" ]]; then
            reqs_files_in+=("${reqs_file}")
        elif [[ "${reqs_file: -4:4}" == ".yml" ]]; then
            reqs_files_yml+=("${reqs_file}")
        else
            echo "error: unrecognized requirements file format: ${reqs_file}" >&2
            return 1
        fi
    done
}


create_empty_env()
{
    local env_name="${1}"
    # Install python if PYTHON is set
    case "${PYTHON}" in
        "") local pyflag="";;
        *) local pyflag=" python==${PYTHON}";;
    esac
    cmd=(${CONDA} create -n "${env_name}"${pyflag} --yes)
    echo "\$ ${cmd[@]^Q}"
    eval "${cmd[@]}" || return 1
}


install_reqs_yml()
{
    local env_name="${1}"
    shift 1
    local reqs_files_yml=("${@}")
    for reqs_file in "${reqs_files_yml[@]}"; do
        local cmd=(${CONDA} env update -n "${env_name}" --file="${reqs_file}")
        echo "\$ ${cmd[@]^Q}"
        eval "${cmd[@]}" || return 1
    done
}


install_reqs_in()
{
    local env_name="${1}"
    shift 1
    local reqs_files_in=("${@}")
    if [[ "${#reqs_files_in[@]}" -gt 0 ]]; then
        local reqs_file_flags=()
        local reqs_file
        for reqs_file in "${reqs_files_in[@]}"; do
            reqs_file_flags+=(--file="${reqs_file}")
        done
        local cmd=(${CONDA} install -n "${env_name}" "${reqs_file_flags[@]}" --yes)
        echo "\$ ${cmd[@]^Q}"
        eval "${cmd[@]}" || return 1
    fi
}


recreate_env()
{
    local env_name="${1}"
    local env_file="${2}"
    echo "recreate conda env '${env_name}' from ${env_file}"
    local cmd=(${CONDA} env create -n "${env_name}" python==${PYTHON} --file="${env_file}")
    echo "\$ ${cmd[@]^Q}"
    eval "${cmd[@]}" || return 1
    return 0
}


export_env()
{
    local env_name="${1}"
    local env_file="${2}"
    echo "export conda env '${env_name}' to ${env_file}"
    local cmd=(${CONDA} env export -n "${env_name}" --no-builds ">" "${env_file}")
    echo "\$ ${cmd[@]^Q}"
    eval "${cmd[@]}" && return 0
    echo "error exporting env '${env_name}' to ${env_file}" >&2
    return 1
}


check_python_version()
{
    if [[ "${PYTHON}" != "" ]]; then
        local installed_python="$(\grep -o '\<python=3.[0-9]\+.[0-9]\+' "${env_file}")"
        if [[ "${installed_python}" != "python=${PYTHON}"* ]]; then
            local msg="warning: installed python version (${installed_python}) differs from"
            msg+=" requested (PYTHON=${PYTHON}); overridden by requirements file?"
            echo "${msg}" >&2
        fi
    fi
}


get_repo_name()
{
    local cmd="from pathlib import Path"
    cmd+="; from git import Repo"
    cmd+="; print(Path(Repo('.', search_parent_directories=True).working_tree_dir).name)"
    python -c "${cmd}" && return 0
    echo "error getting repo name" >&2
    return 1
}


select_first_existing_file()
{
    local paths=("${@}")
    local path
    for path in "${paths[@]}"; do
        if [[ -f "${path}" ]]; then
            echo "${path}"
            return 0
        fi
    done
    echo "none of these files exist: ${files[@]^Q}" >&2
    return 1
}


main "${@}"
