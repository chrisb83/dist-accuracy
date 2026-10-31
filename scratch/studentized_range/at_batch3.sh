A=.claude/skills/dist-accuracy-review/scripts/accuracy.py
P=/c/Users/chbau/miniforge-pypy3/envs/ds/python.exe
for pt in "cdf 7 1000 10" "pdf 7 1000 10"; do
  timeout 900 $P -W ignore $A refs/studentized_range_ref.py --dps 30 --at $pt 2>&1 | grep -v "Warning\|warnings.warn\|^ *return"
done
