#!/bin/bash
# Uso: ./scripts/demo_wterm.sh
set -u
BIN=./build/ddfacet_ompc_host
DAT=data/large

printf "  %-9s %-16s %-13s %-13s %s\n" "offset" "n0-1" "COM w" "SEM w" "dif"
echo "  ---------------------------------------------------------------------"

for off in 0 40 200 1000 4000; do
    OUT_A=$(DDF_OFFSET=$off "$BIN" "$DAT" 128 1 4 2>&1)
    OUT_B=$(DDF_OFFSET=$off DDF_NOW=1 "$BIN" "$DAT" 128 1 4 2>&1)

    N=$(printf '%s' "$OUT_A" | sed -n 's/.*n0-1=\([-0-9.e+]*\).*/\1/p' | head -1)
    A=$(printf '%s' "$OUT_A" | sed -n 's/.*checksum |dirty| *: *\([0-9.]*\).*/\1/p' | head -1)
    B=$(printf '%s' "$OUT_B" | sed -n 's/.*checksum |dirty| *: *\([0-9.]*\).*/\1/p' | head -1)

    awk -v o="$off" -v n="$N" -v a="$A" -v b="$B" 'BEGIN{
        d = (a>b ? a-b : b-a);
        printf "  %-9s %-16s %-13s %-13s %.4f%%\n", o, n, a, b, (a>0 ? 100*d/a : 0);
    }'
done
echo
