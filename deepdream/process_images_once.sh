#!/bin/bash
# Simple script to process all of the images inside the inputs/ folder
# We will be running this script inside the visionai/clouddream Docker image
# Copyright vision.ai, 2015

#echo "Starting"
#cd /opt/deepdream
#chmod gou+r inputs/*
while [ true ];
do
  python deepdream.py >> log.file 2>&1
  sleep 1
done
#ERROR_CODE=$?
#echo "Error Code is" ${ERROR_CODE}


