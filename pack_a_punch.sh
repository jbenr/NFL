#!/usr/bin/env bash
# pack_a_punch.sh: fetch the newest packet to this machine's Downloads folder.
#
# The Mac has a checkout but no packets -- they are generated on the box that
# runs weekly_packet.py. Run this from the repo to pull the newest one over
# ssh and open it:
#
#     ./pack_a_punch.sh
#
# It never names a results folder: it asks pack_a_punch.py --newest where the
# packet is, so a rename on the generating box (data/results/packet_shared ->
# model_2.0 -> packets/{season}/wk{week}, all of which have happened) reaches
# this machine through git instead of silently pulling a months-old file.
#
# Override any default for one run with an env var, e.g.
#     PACK_A_PUNCH_REMOTE= ./pack_a_punch.sh     # packets are local
#
#     PACK_A_PUNCH_REMOTE   ssh target; empty means the packets are local
#     PACK_A_PUNCH_REPO     repo path ON THE MACHINE THAT HAS THEM
#     PACK_A_PUNCH_DEST     where to put it
#     PACK_A_PUNCH_PYTHON   python to use there

remote="${PACK_A_PUNCH_REMOTE-wsl}"
repo="${PACK_A_PUNCH_REPO:-/home/jimbo/werk/NFL}"
dest="${PACK_A_PUNCH_DEST:-$HOME/Downloads}"
python="${PACK_A_PUNCH_PYTHON:-python3}"

if [ -n "$remote" ]; then
    pkt=$(ssh "$remote" "cd '$repo' && $python pack_a_punch.py --newest") || exit 1
    [ -n "$pkt" ] || { echo "pack_a_punch: no packet on $remote" >&2; exit 1; }
    echo "Pulling $remote:$pkt"
    scp -q "$remote:$pkt" "$dest/" || exit 1
else
    pkt=$(cd "$repo" && $python pack_a_punch.py --newest) || exit 1
    [ -n "$pkt" ] || { echo "pack_a_punch: no packet in $repo" >&2; exit 1; }
    cp "$pkt" "$dest/" || exit 1
fi

saved="$dest/$(basename "$pkt")"
echo "Saved: $saved"
# `open` only exists on the Mac.
if [ "$(uname)" = Darwin ]; then open "$saved"; fi
