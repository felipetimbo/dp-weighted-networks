#!/bin/sh

for i in high-school-contacts copenhagen-interaction
do
   python3 dp-weighted-networks-global-POC-ps-final.py $i &
done