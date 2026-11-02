#! /bin/bash

datasets=(GDSC1 GDSC2 CTRPv2)

for dataset in "${datasets[@]}"; do
  uv run data-curation response fit-curves "${dataset}" &
done

wait
