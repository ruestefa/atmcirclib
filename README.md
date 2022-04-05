# Atmcirclib: Python tools to analyze the atmospheric circulation

## Create new conda release

- Create new release in gitlab (or an equivalent reference like a tag)
- Prepare environment (or install `dev-environment.yaml` in `atmcirclib`):

  ```bash
  conda install conda-build anaconda-client
  conda config --set anaconda_upload no
  ```

- Generate `build.yaml` (or write it by hand):

  ```bash
  cd atmcirclib
  ./tools/create_recipe.py
  # -> creates ./recipe/build.yaml
  ```

- Build conda package (may take a few minutes):

  ```bash
  conda build ./recipe/
  # -> creates <path/to/conda-bld>/noarch/atmcirclib-0.0.1-py_0.tar.bz2
  ```

- Upload package to anaconda channel `atmcirc`:

  ```bash
  anaconda login --username <username>
  # -> prompts password
  anaconda upload --user atmcirc <path/to/conda-bld>/noarch/atmcirclib-0.0.1-py_0.tar.bz2
  # -> uploads to https://anaconda.org/atmcirc/atmcirclib
  ```

- Install package:

  ```bash
  conda install -c atmcirc atmcirclib
  # or
  conda config --add channels atmcirc
  conda install atmcirclib
  ```
