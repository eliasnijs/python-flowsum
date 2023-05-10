#!/bin/sh

pipreqs . --force
pre-commit run --all-files
