@echo off
git pull
set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--medvram --medvram-sdxl --xformers
set CUDA_VISIBLE_DEVICES=1
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.5,max_split_size_mb:256
call webui.bat
