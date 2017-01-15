#!/bin/bash

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    brew update
    brew tap homebrew/science
    brew install python3
    if [ ${TASK} == "python_test" ]; then
        python3 -m pip install --user nose numpy
    fi
fi
