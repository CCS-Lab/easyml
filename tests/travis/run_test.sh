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
    Rscript -e "install.packages(c('roxygen2', 'devtools'), repo = 'https://cran.rstudio.com')"
    
    # Install package dependencies
    cd R/
    Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
    
    # Build package
    cd ..
    R CMD INSTALL R
    
    # Run tests
    cd R/
    Rscript -e "devtools::test()" || exit -1
    
    # Check build
    Rscript -e "devtools::check()" || exit -1
    
    # If successful this far, submit to test coverage and exit with exit 
    # code 0 (sucess).
    Rscript -e "library(covr); codecov()"
    exit 0
fi

if [ ${TASK} == "python_test" ]; then
    # Update and upgrade brew
    brew update
    
    # Reinstall gcc (since it doesn't work for some reason)
    brew reinstall gcc
    
    # Install Python for OSx
    brew tap homebrew/science
    brew install python3
    
    # Install package
    cd Python/
    pip3 install .
    
    # Install package dependencies
    pip3 install numpy scipy
    pip3 install -r requirements.txt
    
    # Run tests
    pytest || exit -1
    
    # If successful this far, submit to test coverage and exit with exit 
    # code 0 (sucess).
    exit 0
fi
