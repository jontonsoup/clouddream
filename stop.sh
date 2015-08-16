#!/bin/sh
#
# Start the docker container which will keep looking for images inside
# the inputs/ directory and spew out results into outputs/

docker stop deepdream-compute deepdream-enter
docker rm deepdream-compute deepdream-enter
