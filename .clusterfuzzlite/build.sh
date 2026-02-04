#!/bin/bash -eu
cd $SRC/*/fuzz
cargo +nightly fuzz build
for target in fuzz_targets/*; do
    target_name=$(basename ${target%.rs})
    cp target/x86_64-unknown-linux-gnu/release/$target_name $OUT/
done
