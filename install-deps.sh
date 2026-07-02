#!/usr/bin/env bash
# Install the apt packages listed in apt-packages.txt.
set -eu
cd "$(dirname "$0")"

sudo apt-get update
grep -v '^\s*#' apt-packages.txt | grep -v '^\s*$' | xargs sudo apt-get install -y
