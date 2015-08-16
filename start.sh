#!/bin/sh
#
# Start the docker container which will keep looking for images inside
# the inputs/ directory and spew out results into outputs/

docker run --name deepdream-compute -v `pwd`/deepdream:/opt/deepdream -d visionai/clouddream /bin/bash -c "cd /opt/deepdream && ./process_images_once.sh 2>&1 > log.html"

