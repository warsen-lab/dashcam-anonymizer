#!/usr/bin/env bash
# Opens X-AnyLabeling on one video's frames, saving the review into data/review/<stem>/.
# Usage: scripts/review_ui.sh <stem>      (run `vc_label export-review` for that video first)
#
# The tool embeds Chromium and the link is metered, so every network path is pointed at a dead local
# proxy (port 9): nothing leaves the machine while labelling.
set -euo pipefail

usage() {
    echo "usage: scripts/review_ui.sh <stem>   (the directory name under data/frames/)" >&2
}

if [ "$#" -eq 0 ] || [ -z "$1" ]; then
    echo "review_ui.sh: missing <stem>" >&2
    usage
    exit 2
fi
if [ "$#" -gt 1 ]; then
    echo "review_ui.sh: expected one <stem>, got $#" >&2
    usage
    exit 2
fi
stem="$1"
case "$stem" in
    */* | . | ..)
        echo "review_ui.sh: <stem> is a directory name, not a path: $stem" >&2
        usage
        exit 2
        ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/.."

frames="data/frames/$stem"
review="data/review/$stem"
if [ ! -f "$frames/frames.csv" ] || [ ! -d "$review" ]; then
    echo "review_ui.sh: $frames/frames.csv or $review/ does not exist." >&2
    echo "Run vc_label export-review first (see py/README.md), e.g.:" >&2
    echo "  PYTHONPATH=py .venv/bin/python -m vc_label export-review --frames $frames --labels data/labels/$stem --review $review" >&2
    exit 2
fi

cache="$HOME/.cache/video-compliance"
tool="$cache/x-anylabeling-venv/bin/xanylabeling"
if [ ! -x "$tool" ]; then
    echo "review_ui.sh: X-AnyLabeling is not installed ($tool is missing)." >&2
    echo "See \"安装 X-AnyLabeling\" in docs/labeling-guide.md (it needs a ~320MB download, so ask first)." >&2
    exit 2
fi

proxy="http://127.0.0.1:9"
export HTTP_PROXY="$proxy" HTTPS_PROXY="$proxy" ALL_PROXY="$proxy"
export http_proxy="$proxy" https_proxy="$proxy" all_proxy="$proxy"
export NO_PROXY="localhost,127.0.0.1" no_proxy="localhost,127.0.0.1"
export QTWEBENGINE_CHROMIUM_FLAGS="--proxy-server=$proxy --disable-background-networking --disable-component-update"

mkdir -p "$cache/x-anylabeling-work"
exec "$tool" \
    --no-auto-update-check \
    --work-dir "$cache/x-anylabeling-work" \
    --filename "$frames" \
    --output "$review" \
    --nodata \
    --labels face,plate \
    --validatelabel exact \
    --flags reviewed,day,night,rain,tunnel,city,highway
