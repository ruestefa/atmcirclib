#!/bin/bash
# Create run and dev conda environments and export them.
# By Stefan Ruedisuehli, stefan.ruedisuehli@env.ethz.ch, 2021--2022


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


# Default options
CONDA=""
DELETE=false
PYTHON_VERSION=""
UPDATE=false
PROJECT_NAME="atmcirclib"

USAGE="Usage: $(basename "${0}") [option[s]]

Options:
 -c CMD     Specify conda command instead of auto-detecting mamba or conda
 -d         Delete conda envs again in the end; may be useful with -u
 -h         Show this help message
 -n NAME    Name of the project used for the environments
 -p VER     Specify a python version, e.g., 3.11
 -u         Update package versions and export new environment files

"

# Eval CLI arguments
while getopts c:dn:p:hu flag; do
    case "${flag}" in
        c) CONDA="${OPTARG}";;
        d) DELETE=true;;
        n) PROJECT_NAME="${OPTARG}";;
        p) PYTHON_VERSION="${OPTARG}";;
        h)
            echo "${USAGE}"
            exit 0
        ;;
        u) UPDATE=true;;
        ?)
            echo -e "\n${USAGE}" >&2
            exit 1
        ;;
    esac
done

# Determine conda command; if available, prefer mamba over conda
[[ "${CONDA}" == "" ]] && { CONDA="$(detect_conda)" || exit; }
cmd=(${CONDA} --version)
echo "\$ ${cmd[*]^Q}"
eval "${cmd[*]^Q}" || exit


main()
{
    local run_env_name="${PROJECT_NAME}"
    local dev_env_name="${PROJECT_NAME}-dev"
    local env_names=("${run_env_name}" "${dev_env_name}")

    check_forbidden_active_conda_env "${env_names[@]}" || return

    local possible_run_reqs_files=(
        "requirements/requirements.yml"
        "requirements.yml"
        "requirements/requirements.in"
        "requirements.in"
    )
    local possible_dev_reqs_files=(
        "requirements/dev-requirements.yml"
        "dev-requirements.yml"
        "requirements/dev-requirements.in"
        "dev-requirements.in"
    )
    local run_reqs_file dev_reqs_file
    run_reqs_file="$(select_first_existing_file "${possible_run_reqs_files[@]}")" || return
    dev_reqs_file="$(select_first_existing_file "${possible_dev_reqs_files[@]}")" || return
    local run_env_file dev_env_file
    run_env_file="$(file_in_same_location "${run_reqs_file}" "environment.yml")" || return
    dev_env_file="$(file_in_same_location "${dev_reqs_file}" "dev-environment.yml")" || return

    if ${UPDATE}; then
        echo "update environments from requirements"
    else
        echo "recreate environments"
    fi

    local env_name
    for env_name in "${env_names[@]}"; do
        remove_existing_env "${env_name}" || return
        case "${env_name}" in
            "${run_env_name}")
                create_new_env "${run_env_name}" "${run_env_file}" "${run_reqs_file}" || return
            ;;
            "${dev_env_name}")
                create_new_env "${dev_env_name}" "${dev_env_file}" "${run_reqs_file}" "${dev_reqs_file}" || return
            ;;
        esac
        ${DELETE} && { remove_existing_env "${env_name}" || return; }
    done
}


remove_existing_env()
{
    local env_name="${1}"
    if $(eval ${CONDA} info --env | \grep -q "^\<${env_name}\>"); then
        echo "remove conda env '${env_name}'"
        local cmd=(${CONDA} env remove -n "${env_name}")
        echo "\$ ${cmd[*]^Q}"
        eval "${cmd[*]^Q}" || return 1
    fi
}


create_new_env()
{
    local env_name="${1}"
    local env_file="${2}"
    shift 2
    local reqs_files=("${@}")
    if ${UPDATE}; then
        create_updated_env "${env_name}" "${env_file}" "${reqs_files[@]}" || return
    else
        recreate_env "${env_name}" "${env_file}" || return
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
    # Install python if PYTHON_VERSION is set
    case "${PYTHON_VERSION}" in
        "") local pyflag="";;
        *) local pyflag=" python==${PYTHON_VERSION}";;
    esac
    cmd=(${CONDA} create -n "${env_name}"${pyflag} --yes)
    echo "\$ ${cmd[*]^Q}"
    eval "${cmd[*]^Q}" || return 1
}


install_reqs_yml()
{
    local env_name="${1}"
    shift 1
    local reqs_files_yml=("${@}")
    for reqs_file in "${reqs_files_yml[@]}"; do
        local cmd=(${CONDA} env update -n "${env_name}" --file="${reqs_file}")
        echo "\$ ${cmd[*]^Q}"
        eval "${cmd[*]^Q}" || return 1
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
        echo "\$ ${cmd[*]^Q}"
        eval "${cmd[*]^Q}" || return 1
    fi
}


recreate_env()
{
    local env_name="${1}"
    local env_file="${2}"
    echo "recreate conda env '${env_name}' from ${env_file}"
    local cmd=(${CONDA} env create -n "${env_name}" python==${PYTHON_VERSION} --file="${env_file}")
    echo "\$ ${cmd[*]^Q}"
    eval "${cmd[*]^Q}" || return 1
    return 0
}


export_env()
{
    local env_name="${1}"
    local env_file="${2}"
    echo "export conda env '${env_name}' to ${env_file}"
    local cmd=(${CONDA} env export -n "${env_name}" --no-builds "|" "\grep" -v '^prefix: ' ">" "${env_file}")
    echo "\$ ${cmd[*]^Q}"
    eval "${cmd[*]^Q}" && return 0
    echo "error exporting env '${env_name}' to ${env_file}" >&2
    return 1
}


check_python_version()
{
    if [[ "${PYTHON_VERSION}" != "" ]]; then
        local installed_python="$(\grep -o '\<python=3.[0-9]\+.[0-9]\+' "${env_file}")"
        if [[ "${installed_python}" != "python=${PYTHON_VERSION}"* ]]; then
            local msg="warning: installed python version (${installed_python}) differs from"
            msg+=" requested (PYTHON=${PYTHON_VERSION}); overridden by requirements file?"
            echo "${msg}" >&2
        fi
    fi
}


check_forbidden_active_conda_env()
{
    local forbidden_names=("${@}")
    local active_name="$(basename "${CONDA_PREFIX}")"
    if [[ "${active_name}" == "" ]]; then
        # No active conda env: All good!
        return 0
    fi
    # Active conda env found: Check its name
    for forbidden_name in "${forbidden_names[@]}"; do
        if [[ "${active_name}" == "${forbidden_name}" ]]; then
            echo "detected active conda env: ${active_name}" >&2
            echo "forbidden env names: ${forbidden_names[@]}" >&2
            echo "env has forbidden name, so please deactivate it!" >&2
            return 1
        fi
    done
    return 0
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


file_in_same_location()
{
    local ref_file_path="${1}"
    local file_name="${2}"
    local path
    path="$(dirname "${ref_file_path}")" || return
    case "${path}" in
        "") echo "${file_name}";;
        *) echo "${path}/${file_name}";;
    esac
}


main "${@}"
