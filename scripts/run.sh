#!/bin/sh

clear
SCRIPT_PATH="$(dirname "$(readlink -f "$0")")"
cd $SCRIPT_PATH/..

python3 example.py
