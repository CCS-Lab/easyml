#!/bin/bash

# Set environment to exit with any non-zero exit codes
set -e

if [ ${TASK} == "r_test" ]; then
    # Not sure what this export does
    export _R_CHECK_TIMINGS_=0
    
    # Install R for OSx
    wget https://cran.rstudio.com/bin/macosx/R-latest.pkg  -O /tmp/R-latest.pkg
    sudo installer -pkg "/tmp/R-latest.pkg" -target /
    
    # Install devtools and roxygen2
    Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
    Rscript -e "install.packages('roxygen2', repo = 'https://cran.rstudio.com')"
    
    # Install package dependencies
    cd R/
    Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
    
    # Build package
    cd ..
    R CMD INSTALL R
    
    # Run tests
    cd R/
    Rscript -e "devtools::test()" || exit -1
    
    # If successful this far, submit to test coverage and exit with exit 
    # code 0 (sucess).
    Rscript -e "library(covr); codecov()"
    exit 0
fi

if [ ${TASK} == "python_test" ]; then
    # Install Python for OSx
    brew update
    brew tap homebrew/science
    brew install python3
    
    # Install pytest
    python3 -m pip install --user pytest numpy
    
    # Install package
    pip3 install .
    
    # Install package dependencies
    pip3 install -r requirements.txt
    
    # Run tests
    cd Python/
    python3 -m pytest || exit -1
    
    # If successful this far, submit to test coverage and exit with exit 
    # code 0 (sucess).
    exit 0
fi
