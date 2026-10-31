A=.claude/skills/dist-accuracy-review/scripts/accuracy.py
P=/c/Users/chbau/miniforge-pypy3/envs/ds/python.exe
for pt in "pdf 1e-6 3 10" "cdf 1e-4 3 10" "cdf 1e-6 3 10" "pdf 1e-4 3 10" "pdf 1e-9 3 10" "cdf 1e-8 3 10" "pdf 3.0 1.9 10" "cdf 5 100 10" "pdf 5 100 10"; do
  timeout 900 $P -W ignore $A refs/studentized_range_ref.py --dps 30 --at $pt 2>&1 | grep -v "Warning\|warnings.warn\|^ *return"
done
