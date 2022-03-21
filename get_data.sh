#!/bin/bash

# Copy test data useful as reference to mock data files in tests
# (The data files are not required to run the unit tests)

loc="daint:/project/s1063/ruestefa/data/shared/atmcirclib/"

\rsync -au --progress "${loc}/data" .
