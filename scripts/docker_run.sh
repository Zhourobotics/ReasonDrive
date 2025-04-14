#!/bin/bash
docker run --gpus all \\
  -v "$(pwd)":/workspace/ \\
  -v /tmp/.X11-unix:/tmp/.X11-unix \\
  -e DISPLAY=$DISPLAY \\
  --ipc=host \\
  --network=host \\
  --rm \\
  --name reasondrive \\
  -it qwenllm/qwenvl:2.5-cu121 bash

