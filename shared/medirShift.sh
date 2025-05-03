#!/usr/bin/env bash
out_file="offsets.csv"
echo "pc_ts,phone_ts,offset_ms" > "$out_file"

for i in $(seq 1 20); do
  pc_ts=$(date +%s%3N)
  phone_ts=$(adb shell date +%s%3N | tr -d '\r')
  offset=$((phone_ts - pc_ts))
  echo "$pc_ts,$phone_ts,$offset" >> "$out_file"
  sleep 0.5
done

echo "Captura terminada. Vê os primeiros valores em $out_file"
