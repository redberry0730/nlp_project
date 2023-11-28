#!/bin/bash

messistring=""
while read i; do
	temp=`python parse.py $i`;
	messistring="$messistring $temp"
done < messi.txt > messitxt.txt

lebronstring=""
while read i; do
	temp=`python parse.py $i`;
	lebronstring="$lebronstring $temp"
done < lebron.txt > lebron.txt
