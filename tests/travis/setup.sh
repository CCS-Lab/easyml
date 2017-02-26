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

if [ ${TRAVIS_OS_NAME} == "linux" ]; then
    # update apt-get
    sudo apt-get update
    
    if [ ${TASK} == "r_test" ]; then
    
    # Install R for linux
    sudo apt-get install r-base-core
    
    fi
    
    if [ ${TASK} == "python_test" ]; then
    
    # Install Python for linux
    sudo apt-get install python-pip python3-pip python3-dev python3-virtualenv python3-tk
    pip3 install --upgrade pip
    
    fi
fi
