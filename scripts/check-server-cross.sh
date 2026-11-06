#!/bin/sh
# Try `cargo check` of the dedicated server for cross targets (NET-7). NOT run in CI.
#   scripts/check-server-cross.sh                       the default list below
#   scripts/check-server-cross.sh aarch64-unknown-linux-gnu
# A target that is not installed (`rustup target list --installed`) is reported as SKIP, never as a pass.
# `cargo check` proves the Rust compiles for the target; it does not link, so a missing cross
# linker or C toolchain (aarch64-linux-gnu-gcc, musl-gcc) is only found by a real build:
#   ZESTENGINE_SERVER_TARGET=<triple> just build-server <project>
# Exit status: 0 when every tried target passed or was skipped, 1 when one failed.
set -u
cd "$(dirname "$0")/.."
targets="${*:-x86_64-unknown-linux-gnu x86_64-unknown-linux-musl aarch64-unknown-linux-gnu aarch64-unknown-linux-musl}"
installed="$(rustup target list --installed 2>/dev/null)"
failed=0
for t in $targets; do
    if ! printf '%s\n' "$installed" | grep -qx "$t"; then
        echo "SKIP  $t (not installed: rustup target add $t)"
        continue
    fi
    if cargo check -p zestengine-server --target "$t" --locked >"target/cross-check-$t.log" 2>&1; then
        echo "PASS  $t (cargo check)"
    else
        echo "FAIL  $t (see target/cross-check-$t.log)"
        grep -m5 -E '^error' "target/cross-check-$t.log" | sed 's/^/        /'
        failed=1
    fi
done
exit $failed
