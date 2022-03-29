# Atmcirclib: Changelog

## v0.2.0 (tbd)

- Implement string formatting of `*Criterion` and `Criteria` instances for titles ("human") and file names ("file"), with tests
- Extend `TrajDatasetMetadata` with some functionality required in `cosmo_nawdex.bin.plot_traj_density`

### TODOs

- [ ] Make interface of `TrajDatasetMetadata` cleaner and more self-consistent (names, return types)

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
- [ ] Put some thought into `TrajDatasetMetadata` and write some tests

## v0.0.1 (2022-03-25)

- Set up project for installation with pip or conda
- Copy module `atmcirclib.traj` from project `cosmo-nawdex` containing classes `TrajsDataset` and `BoundingBox`
