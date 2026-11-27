#!/usr/bin/env bash
# Glass in the shade (E7): on every run of the testbed's `glass-shade` scene in a results folder of `gfx-check run`
# (golden.ron runs it), the glass and the water sphere inside the shed must be no brighter than the opaque sphere
# beside them, as they are when they take the traced light. Before E7 the traced rungs lit them with the
# unshadowed sun: their brightest pixels were 2.6-2.7 times the opaque sphere's, their mean 1.25-1.40 times.
#
#   tools/gfx-check/glass-shade.sh <results folder> [gfx-check binary]
#
# Prints, per run, each sphere's mean luma (0..1) and brightest pixel (8-bit), and the shade/sun ratios of the
# means; exits 1 when a run fails, 2 when no run was found. Sphere squares are 90 px wide at 1920x1080 (the
# scene's fixed camera); other sizes are scaled.
set -uo pipefail
results=${1:?usage: glass-shade.sh <results folder> [gfx-check binary]}
gfx_check=${2:-target/release/gfx-check}
# to tune: how much brighter than the shaded opaque sphere the shaded glass and water may be.
max_mean=1.15
max_peak=1.30
# Sphere centres at 1920x1080: shade opaque, glass, water; sun water, glass, opaque.
centres=(258 485 710 1210 1435 1660)
names=(shade-opaque shade-glass shade-water sun-water sun-glass sun-opaque)

found=0
bad=0
while IFS= read -r png; do
    found=$((found + 1))
    run=${png%/a.png}
    size=$("$gfx_check" image stats "$png" | head -1 | awk '{print $1}')
    width=${size%x*}
    height=${size#*x}
    declare -A mean peak
    for i in "${!centres[@]}"; do
        x0=$(((centres[i] - 45) * width / 1920)); x1=$(((centres[i] + 45) * width / 1920))
        y0=$((535 * height / 1080)); y1=$((625 * height / 1080))
        line=$("$gfx_check" image stats "$png" --region "$x0,$y0,$x1,$y1" | head -1)
        mean[${names[i]}]=$(awk '{print $4}' <<<"$line")
        peak[${names[i]}]=$(awk '{print $6}' <<<"$line")
    done
    verdict=$(awk -v om="${mean[shade-opaque]}" -v op="${peak[shade-opaque]}" \
        -v gm="${mean[shade-glass]}" -v gp="${peak[shade-glass]}" \
        -v wm="${mean[shade-water]}" -v wp="${peak[shade-water]}" \
        -v mm="$max_mean" -v mp="$max_peak" 'BEGIN {
            ok = gm <= mm * om && wm <= mm * om && gp <= mp * op && wp <= mp * op
            printf "%s glass/opaque mean %.2f peak %.2f, water/opaque mean %.2f peak %.2f",
                ok ? "pass" : "FAIL", gm / om, gp / op, wm / om, wp / op
        }')
    ratios=$(awk -v so="${mean[shade-opaque]}" -v uo="${mean[sun-opaque]}" \
        -v sg="${mean[shade-glass]}" -v ug="${mean[sun-glass]}" \
        -v sw="${mean[shade-water]}" -v uw="${mean[sun-water]}" \
        'BEGIN { printf "shade/sun: opaque %.2f glass %.2f water %.2f", so / uo, sg / ug, sw / uw }')
    echo "$run: $verdict; $ratios"
    [[ $verdict == pass* ]] || bad=$((bad + 1))
    unset mean peak
done < <(find -L "$results" -path '*glass-shade*' -name a.png | sort)

if ((found == 0)); then
    echo "no glass-shade run in $results"
    exit 2
fi
echo "$found runs, $bad failed"
((bad == 0))
