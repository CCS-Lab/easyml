#!/bin/bash

if [ ${TASK} == "r_test" ]; then
    # Set environment to exit with any non-zero exit codes
    set -e
    export _R_CHECK_TIMINGS_=0
    
    # Install R for OSx
    wget https://cran.rstudio.com/bin/macosx/R-latest.pkg  -O /tmp/R-latest.pkg
    sudo installer -pkg "/tmp/R-latest.pkg" -target /
    
    # Install devtools
    Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
    
    # Install package dependencies
    cd R
    Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
    cd ..
    
    # Build package
    R CMD INSTALL R
    
    # Run tests
    cd R
    Rscript -e "devtools::test()" || exit -1
    Rscript tests/travis/r_vignettes.R
    
    # If successful this far, submit to test coverage and exit with exit 
    # code 0 (sucess).
    Rscript -e "library(covr); codecov()"
    exit 0
fi

if [ ${TASK} == "python_test" ]; then
    # make all || exit -1
    # # use cached dir for storing data
    # rm -rf ${PWD}/data
    # mkdir -p ${CACHE_PREFIX}/data
    # ln -s ${CACHE_PREFIX}/data ${PWD}/data
    # 
    # if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    #     python -m nose tests/python/unittest || exit -1
    #     python3 -m nose tests/python/unittest || exit -1
    #     make cython3
    #     # cython tests
    #     export MXNET_ENFORCE_CYTHON=1
    #     python3 -m nose tests/python/unittest || exit -1
    #     python3 -m nose tests/python/train || exit -1
    # else
    #     nosetests tests/python/unittest || exit -1
    #     nosetests3 tests/python/unittest || exit -1
    #     nosetests3 tests/python/train || exit -1
    # fi
    exit 0
fi
