#!/bin/bash

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    if [ ${TASK} == "r_test" ]; then
    
    # Install R for OSx
    wget https://cran.rstudio.com/bin/macosx/R-latest.pkg  -O /tmp/R-latest.pkg
    sudo installer -pkg "/tmp/R-latest.pkg" -target /
    
    fi
    
    if [ ${TASK} == "python_test" ]; then
    
    # Update and upgrade brew
    brew update
    
    # Reinstall gcc (since it doesn't work for some reason)
    brew reinstall gcc

    
    # Install Python for OSx
    brew tap homebrew/science
    brew install python3
    
    fi
fi
