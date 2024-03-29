# AtmCircLib: History

## v0.9.2 (2023-01-05)

- Lower min. typer version further to 0.4 to accommodate rstcheck (requires <0.5, not <=0.5)

## v0.9.1 (2023-01-04)

- Fix editable installs for extension modules (broken in template when stripping down setup.py)
- Switch copier template to new `atmcirc/python-blueprint-f2py` based on `mch-python-blueprint`
  - Adds docs and setups for github actions and jenkins
- Reduce min. typer version to 0.5 (otherwise incompatible with rstcheck)
- Merge `dev-requirements.yml` into `requirements.yml` and eliminate `requirements/dev*.yml`

## v0.9.0 (2022-12-19)

- In `atmcirclib.regular_grid.RegularGridPlot`, implement optional rasterization of topography
- Major copier update: Move toward the MCH Blueprint (not quite finished)

## v0.8.0 (2022-10-12)

- Add module `atmcirclib.plot_utils` with functions to manipulate colormaps (moved from `cosmo_natl.crclim_domains`).
- Add module `atmcirclib.regular_grid` with classes `RegularGrid`, `RegularGridPlotter` and `RegularGridPlot` (originally moved from `cosmo_natl.crclim_domains`)
  - Purpose: Create plot with multiple grid outlines on top of the model topography

## v0.7.1 (2022-10-05)

- Run copier updates
  - Replaces `{,dev}-requirements.in` by `~.yml`
  - Moves requirements and environments files into `requirements/`
- Update `gt4py` version to `v2022-09-29`
  - Fixes installation problem with `xxhash` (called `python-xxhash` in `conda-forge`)

## v0.7.0 (2022-09-14)

- Add module `atmcirclib.typer` with decorator `typer_wrap_pdb` (analogue to `atmcirclib.click.click_wrap_pdb`) and some additional conveniences
- Add subpackage `atmcirclib.fortran` with Fortran namelist parser with `parse_namelist` and `parse_namelist_file`
- Add module `atmcirclib.icon` with various utilities
  - Find path to full namelist by parsing master namelist file in ICON run directory with `find_namelist_path`
  - Insert ICON namelist params in format string templates with `format_icon_params`
  - Convert time steps from ICON format to regular datetime objects with `convert_icon_time_step` and `convert_icon_time`
  - Disaggregate total precip with `disaggr_tot_precip` and derive precipitation intensity units with `deriv_precip_intensity_units`
- Add module `atmcirclib.math` with function `step_ceil` with tests
- Fix `tools/create_envs.sh` (`${UPDATE}` implemented wrong)
  - Merged back into copier template `meta-python-project`
- Run copier update (template `python-project`)
  - `tools/create_envs.sh`: Fixed `${UPDATE}` bug; don't update by default
  - `tools/install_editable.sh`: Refactored; build extension modules by default; don't install linked by default

## v0.6.0 (2022-08-29)

- Add callback function `click_set_pdb_raise` to `atmcirclib.click`
- Add decorator function `click_add_option_pdb` to `atmcirclib.click` to add `--pdb` option with one line
- Add decorator function `click_wrap_pdb` to wrap click command function in `wrap_pdb` with option to call `click_add_option_pdb` internally, making the addition of `--pdb` a (potential) one-liner
- Update conda environments
- Update to copier template

## v0.5.0 (2022-06-28)

- Hook project back up to copier template python-project-f2py and run copier update
- Write simulation scanner to find simulation run directories
- Improve simulations code while working on simulations library in `cosmo-nawdex` repo
- Add module `plot.rgb` from `cosmo-nawdex` and adapt it

## v0.4.0 (2022-06-08)

- Improve functions used to drop into pdb in case of exception in `click` module, incl. docstrings
- Add module `simulations` (from `cosmo_nawdex` repo)
- Add new requirement `pandas` and update envs
- Remove conda build file archive; only retain latest as `recipe/meta.yaml`

## v0.3.0 (2022-05-31)

- Move `atmcirclib.tools.call_graph` and `atmcirclib.tools.create_recipe` to `atmcirctools v0.2.0` as commands `act call-graph` and `act create-recipe`
- Remove module `atmcirclib.tools`
- Merge `tools/{,run_}update_min_versions.sh` as `tools/update_min_reqs_versions.sh` and implement handling of comments

## v0.2.4 (2022-05-23)

- Add additional functions to module `click` (from `pyflexplot` via `cosmo_nawdex`)
  - Wrappers to drop into `ipdb` if exception/callback raises an exception
- Fix issue with `f2py` include path in `CMakeLists.txt`
- Write pair of bash scripts (`tools/{,run_}update_min_versions.sh`) to set min. versions in `requirements.in` to minor version of currently installed version
- Use `mamba` in `tox` to create envs
- Write script (`tools/call_graph.py`) to plot the call graph of a python module/package (based on `pycg` and `pygraphviz`)

## v0.2.3 (2022-04-25)

- Add module `click`

## v0.2.2 (2022-04-14)

- Add module `intp` (from repo `dpv_th`) with tests
- Add explorative `gt4py` tests (from repo `dpv_th`)
- Improve scripts to create recipe (now moved into atmcirclib) and environments

## v0.2.1 (2022-04-14)

- Switch build from (pure) `setuptools` to `scikit-build`
- Add Fortran extension modules as `atmcirclib.deriv.ext` (`deriv-f`, `deriv.f90`)

## v0.2.0 (2022-04-05)

- Implement `CriteriaFormatter` with string formatting of `*Criterion` and `Criteria` instances for titles ("human") and file names ("file"), with tests
- Rename `TrajDatasetMetadata` to `TrajTimeHandler`, extend it with some functionality required in `cosmo_nawdex.bin.plot_traj_density` and add tests for some crucial methods
- Restructure code and test packages
- Implement reading LAGRANTO output (NetCDF) and start files (four columns, no header)
- Implement creation of cosmo grid dataset from traj dataset
- Add model (`TrajModel`) and integration direction (`TrajDirection`) to `TrajDataset`
- Usual cleanup and refactoring along the way

### TODOs

- [ ] Add tests for untested `TrajTimeHandler` methods

## v0.1.0 (2022-03-29)

- Break up monolithic `TrajsDataset` (now `TrajsDataset`) class
- Move handling of COSMO grid file to new class `COSMOGridDataset`
- Move handling of start points to new class `TrajStartDataset`
- Move computation and formatting of times, durations etc. to new class `TrajDatasetMetadata`
- Implement selection criteria as classes derived from `Criteria`
- Write basic tests for `TrajDataset` based on trajs derived from a real output file
- Write extensive tests for `TrajDataset.count` based on idealized trajs, which also test the criteria and by extension traj selection
- Write tests for `COSMOGridDataset` based on idealized grid data modeled after a COSMO output file

### TODOs

- [ ] Implement reading trajs from text file (offline lagranto output)
- [ ] Simplify criteria interface with string-based shorthand arguments better suited for interactive sessions
- [ ] Write tests for additional functionality like trajs selection/removal
- [x] Put some thought into `TrajDatasetMetadata` and write some tests

## v0.0.1 (2022-03-25)

- Set up project for installation with pip or conda
- Copy module `atmcirclib.traj` from project `cosmo-nawdex` containing classes `TrajsDataset` and `BoundingBox`
