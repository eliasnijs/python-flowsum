#!/bin/sh

SCRIPT_PATH="$(dirname "$(readlink -f "$0")")"
cd $SCRIPT_PATH/..

pipreqs . --force
pre-commit run --all-files
