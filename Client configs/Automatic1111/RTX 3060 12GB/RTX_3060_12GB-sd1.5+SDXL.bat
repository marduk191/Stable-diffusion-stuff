@echo off
git pull
set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--medvram --medvram-sdxl --opt-channelslast --api --deepdanbooru --gradio-img2img-tool color-sketch
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.5,max_split_size_mb:512
call webui.bat
