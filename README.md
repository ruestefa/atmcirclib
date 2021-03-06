# AtmCircLib: Python library of the Atmospheric Circulation Group

## Create new conda release

### Prepare environment

Prepare conda (only required once):

```bash
conda config --set anaconda_upload no
```

Install required packages:

```bash
conda env create -n <name> --file=dev-environment.yaml`
# or
conda install conda-build anaconda-client
```

### Create release

Save the old `build.yaml`:

```bash
git mv recipe/build.yaml recipe/build_v0.1.0.yaml
```

Generate a new `build.yaml` (rather than writing it by hand):

```bash
./tools/create_recipe.py
# -> create ./recipe/build.yaml
```

(Note that for repositories that reside on github, the tool `grayskull` can be used, but as of v1.1.2 it does not yet support gitlab repositories.)

Commit and push the new file, then create a new release on gitlab/github.

### Create conda package

Build conda package (may take a few minutes):

```bash
conda build ./recipe/
# -> creates <path/to/conda-bld>/noarch/atmcirclib-0.2.0-py_0.tar.bz2
```

Log in to anaconda (should only required once per machine):

```bash
anaconda login --username <username>
# -> prompts password
```

Upload the conda package to the anaconda channel `atmcirc`:

```bash
anaconda upload --user atmcirc <path/to/conda-bld>/noarch/atmcirclib-0.2.0-py_0.tar.bz2
# -> uploads to https://anaconda.org/atmcirc/atmcirclib
```

### Install the package

```bash
conda install -c atmcirc atmcirclib
# or
conda config --add channels atmcirc
conda install atmcirclib
```
