#!/bin/sh

clear
SCRIPT_PATH="$(dirname "$(readlink -f "$0")")"
cd $SCRIPT_PATH/..

#rm -r data/*
# wget https://dl01.irc.ugent.be/flow/FlowRepository_FR-FCM-ZZPH/Levine_13dim.fcs -O ./resources/Levine_13dim.fcs
python3 main.py
