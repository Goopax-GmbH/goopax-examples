#!/bin/bash

while (true); do
    rnd="$(shuf -i 1-1000000000 -n 1)"
    cat ics_goopax.conf.in | sed -re "s/RANDOM/$rnd/" > ics_goopax.conf
    
    build_dir/MUSIC2/MUSIC ics_goopax.conf
    mv -v ics_arepo.hdf5 ics_goopax.hdf5
    sleep 80
done
