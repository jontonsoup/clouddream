#!/bin/sh
#A simple docker script to enter into the container with caffe and other shell scripts

if [ "`docker ps | grep deepdream-compute`" = "" ]; then
    echo "Making volume"
    docker run -t -i --rm --name deepdream-enter -v `pwd`/deepdream:/opt/deepdream friedmanj98/clouddream /bin/bash
else
    echo "Steaming volume from deepdream-compute"
    docker run -t -i --rm --name deepdream-enter --volumes-from=deepdream-compute friedmanj98/clouddream /bin/bash
fi
