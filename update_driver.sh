#!/bin/bash

sudo apt-get remove --purge '^nvidia-.*'
sudo apt autoremove

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

sudo apt install -Y nvidia-driver-555
