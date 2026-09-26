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
    local path

    if [ -n "$remote" ]; then
        path=$(ssh "$remote" "cd '$repo' && $python pack_a_punch.py --newest") || return 1
        [ -n "$path" ] || { echo "pack_a_punch: no packet on $remote" >&2; return 1; }
        echo "Pulling $remote:$path"
        scp -q "$remote:$path" "$dest/" || return 1
    else
        path=$(cd "$repo" && $python pack_a_punch.py --newest) || return 1
        [ -n "$path" ] || { echo "pack_a_punch: no packet in $repo" >&2; return 1; }
        cp "$path" "$dest/" || return 1
    fi

    local saved="$dest/$(basename "$path")"
    echo "Saved: $saved"
    # `open` only exists on the Mac, and a plain `&&` here would make its
    # absence the function's exit status on every other machine.
    if [ "$(uname)" = Darwin ]; then open "$saved"; fi
    return 0
}

# Sourced, this just defines the function. Executed directly (./pack_a_punch.sh)
# it runs it, because that is what someone typing the filename meant. When a
# file is sourced $0 is the shell; when it is executed $0 is the file.
case "${0##*/}" in
    pack_a_punch.sh) pack_a_punch "$@" ;;
esac
