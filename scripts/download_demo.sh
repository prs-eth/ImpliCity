#!/usr/bin/env bash


function download_demo_data() {
    if [ ! -d "data/" ]; then
        mkdir -p "data"
    fi

    cd data

    url="https://share.phys.ethz.ch/~pf/stuckercdata/implicity/"

    tar_file="ImpliCity_demo.tar"

    wget --no-check-certificate --show-progress "$url$tar_file"
    tar -xf "$tar_file"
    rm "$tar_file"
    cd ../
}


download_demo_data;
