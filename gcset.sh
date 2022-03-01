#!/bin/bash

username='hudi'

current_sdk="poplar_sdk-ubuntu_18_04-2.5.0-EA.1+828-e76001ee57"
poplar="poplar-ubuntu_18_04-2.5.0+2258-bc8598cad8"
popart="popart-ubuntu_18_04-2.4.0+2258-bc8598cad8"

tmp_sdk_path=/localdata/hudi/sdk

export GCDA_MONITOR=1
export TF_CPP_VMODULE="poplar_compiler=1"
export TF_POPLAR_FLAGS="--max_compilation_threads=40 --executable_cache_path=/localdata/"$username"/cachedir"
export TMPDIR="/localdata/hudi/tmp"
#export POPLAR_LOG_LEVEL=INFO
export GM2=mk2

export PATH=$PATH:$tmp_sdk_path/$current_sdk/$poplar/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$tmp_sdk_path/$current_sdk/$poplar/lib/

echo $tmp_sdk_path/$current_sdk/$poplar/enable.sh
source $tmp_sdk_path/$current_sdk/$poplar/enable.sh

echo $tmp_sdk_path/$current_sdk/$popart/enable.sh
source $tmp_sdk_path/$current_sdk/$popart/enable.sh

export IPUOF_CONFIG_PATH=/localdata/zhiweit/.ipuof.conf.d/p64_ipuof.conf
