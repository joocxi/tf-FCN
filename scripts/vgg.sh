#!/usr/bin/env bash
LOG_DIR=log

wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz -O $LOG_DIR/vgg_16_2016_08_28.tar.gz
tar -xvf $LOG_DIR/vgg_16_2016_08_28.tar.gz -C $LOG_DIR
rm -rf $LOG_DIR/vgg_16_2016_08_28.tar.gz
