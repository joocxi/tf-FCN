#!/usr/bin/env bash
DATA_DIR=data
mkdir -p $DATA_DIR

wget https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.zip -O $DATA_DIR/3Dircadb1.zip
unzip $DATA_DIR/3Dircadb1.zip -d $DATA_DIR

find ./$DATA_DIR -name '*.zip' -exec sh -c 'unzip -d `dirname {}` {}' ';'
