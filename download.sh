#!/bin/bash

FILE=hico_20160224_det.tar.gz
EXTR=hico_20160224_det
ID=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk

if [ -d $EXTR ]; then
  echo "$EXTR already exists."
  exit 0
fi

echo "Connecting..."

if ! command -v gdown &> /dev/null; then
  pip install -q gdown
fi

gdown "https://drive.google.com/uc?id=$ID" -O $FILE || exit 1

echo "Extracting..."

tar zxf $FILE
rm $FILE

echo "Done."
