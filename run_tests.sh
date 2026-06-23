#!/bin/bash

cmake -S . -B build-test -DBUILD_TESTING=ON
  cmake --build build-test --target bytetrack_tests -j4
  ctest --test-dir build-test --output-on-failure
