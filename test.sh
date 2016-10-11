#!/bin/bash

./cuGEO --verbose

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
done < <(./cuGEO --measurements 1000)

if [ "$returncode" = "0" ]
then
  echo Succeeded
else
  echo Failed
  exit $returncode
fi

exit
