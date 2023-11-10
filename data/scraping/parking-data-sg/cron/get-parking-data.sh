#!/bin/bash
wget -O /root/sg-parking-time-series/data/sg-city-parking-$(($(date +%s%N)/1000000)).json https://daten.stadt.sg.ch/api/explore/v2.1/catalog/datasets/freie-parkplatze-in-der-stadt-stgallen-pls/records?limit=100
