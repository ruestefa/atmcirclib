# Atmcirclib: Changelog

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
