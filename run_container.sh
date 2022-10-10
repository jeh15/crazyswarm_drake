# XAuthority Docker Path:
XAUTH=/tmp/.docker.xauth

# Give local user authority of X display: (Not safe)
xhost +local:

docker run --rm -it \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --gpus=all \
    --volume="$PWD/scripts/:/scripts/" \
    jeh15/drone:latest