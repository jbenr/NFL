# pack_a_punch: fetch the newest packet to this machine's Downloads folder.
#
# The Mac has a checkout but no packets -- they are generated on the box that
# runs weekly_packet.py. This pulls the newest one over ssh and opens it.
#
# Install once, in ~/.zshrc (or ~/.bashrc):
#
#     export PACK_A_PUNCH_REMOTE=jimbo@wsl-box     # your ssh target
#     source ~/PycharmProjects/NFL/pack_a_punch.sh
#
# Then `pack_a_punch` works from anywhere. It never names a results folder:
# it asks pack_a_punch.py --newest where the packet is, so a rename on the
# generating box (data/results/packet_shared -> model_2.0 -> packets/{season}/
# wk{week}, all of which have happened) reaches this machine through git
# instead of silently pulling a months-old file.
#
#     PACK_A_PUNCH_REMOTE   ssh target; unset means the packets are local
#     PACK_A_PUNCH_REPO     repo path ON THE MACHINE THAT HAS THEM (~/werk/NFL)
#     PACK_A_PUNCH_DEST     where to put it (~/Downloads)
#     PACK_A_PUNCH_PYTHON   python to use there (python)

pack_a_punch() {
    local repo="${PACK_A_PUNCH_REPO:-$HOME/werk/NFL}"
    local remote="${PACK_A_PUNCH_REMOTE:-}"
    local dest="${PACK_A_PUNCH_DEST:-$HOME/Downloads}"
    local python="${PACK_A_PUNCH_PYTHON:-python}"
    # NOT `local path` -- in zsh, `path` is a special array tied to $PATH;
    # declaring it local blanks $PATH for the rest of this function and
    # every external command below ("ssh", "scp", "cp") stops resolving.
    local pkt

    if [ -n "$remote" ]; then
        pkt=$(ssh "$remote" "cd '$repo' && $python pack_a_punch.py --newest") || return 1
        [ -n "$pkt" ] || { echo "pack_a_punch: no packet on $remote" >&2; return 1; }
        echo "Pulling $remote:$pkt"
        scp -q "$remote:$pkt" "$dest/" || return 1
    else
        pkt=$(cd "$repo" && $python pack_a_punch.py --newest) || return 1
        [ -n "$pkt" ] || { echo "pack_a_punch: no packet in $repo" >&2; return 1; }
        cp "$pkt" "$dest/" || return 1
    fi

    local saved="$dest/$(basename "$pkt")"
    echo "Saved: $saved"
    # `open` only exists on the Mac, and a plain `&&` here would make its
    # absence the function's exit status on every other machine.
    if [ "$(uname)" = Darwin ]; then open "$saved"; fi
    return 0
}
