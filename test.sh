#!/bin/bash

./cuGEO --smem --verbose

echo -n "Test 1... "
returncode=1
while read -r line
do
if [ "$line" = "Done" ]
then
  returncode=0
fi
done < <(./cuGEO --config config.txt)

if [ "$returncode" = "0" ]
then
  echo Succeeded
else
  echo Failed
  exit $returncode
fi

echo -n "Test 2... "
returncode=1
while read -r line
do
if [ "$line" = "Done" ]
then
  returncode=0
fi
done < <(./cuGEO --measurements 1024 --verbose)

if [ "$returncode" = "0" ]
then
  echo Succeeded
else
  echo Failed
  exit $returncode
fi

echo -n "Test 3... "
returncode=1
while read -r line
do
if [ "$line" = "Done" ]
then
  returncode=0
fi
done < <(./cuGEO --iterations 1000 --measurements 1024 --blocks 32 --threads 1024)

if [ "$returncode" = "0" ]
then
  echo Succeeded
else
  echo Failed
  exit $returncode
fi

echo -n "Test 4... "
returncode=1
while read -r line
do
if [ "$line" = "Done" ]
then
  returncode=0
fi
done < <(./cuGEO --measurements 1024 --smem --verbose)

if [ "$returncode" = "0" ]
then
  echo Succeeded
else
  echo Failed
  exit $returncode
fi

exit
