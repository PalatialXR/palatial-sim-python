#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python src/scene_graph/SceneDriver.py "$@" 