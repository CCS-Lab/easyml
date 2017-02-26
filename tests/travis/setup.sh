#!/bin/bash

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    if [ ${TASK} == "r_test" ]; then
    
    # Install R for OSx
    wget https://cran.rstudio.com/bin/macosx/R-latest.pkg  -O /tmp/R-latest.pkg
    sudo installer -pkg "/tmp/R-latest.pkg" -target /
    
    fi
    
    if [ ${TASK} == "python_test" ]; then
    
    # Install Python for OSx
    brew update
    brew reinstall gcc
    brew tap homebrew/science
    brew install python3
    
    fi
fi

if [ ${TRAVIS_OS_NAME} == "linux" ]; then
    # update apt-get
    sudo apt-get update
    
    if [ ${TASK} == "r_test" ]; then
    
    # Install R for linux
    sudo sh -c 'echo "deb http://cran.rstudio.com/bin/linux/ubuntu trusty/" >> /etc/apt/sources.list'
    gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
    gpg -a --export E084DAB9 | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install r-base-core
    sudo apt-get install libssl-dev
    
    fi
    
    if [ ${TASK} == "python_test" ]; then
    
    # Install Python for linux
    sudo apt-get install gfortran
    sudo apt-get install python3-setuptools
    sudo easy_install3 pip
    pip3 install --upgrade pip
    sudo apt-get install python3-tk
    
    fi
fi
