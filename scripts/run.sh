#!/bin/sh

clear
SCRIPT_PATH="$(dirname "$(readlink -f "$0")")"
cd $SCRIPT_PATH/..

# rm -r data/*
python3 src/main.py
