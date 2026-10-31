#!/bin/bash
# usage: at_batch.sh REF OUTFILE < lines "method arg skew"
cd /c/Users/chbau/Documents/research/dist-accuracy
PY=/c/Users/chbau/miniforge-pypy3/envs/ds/python.exe
A=.claude/skills/dist-accuracy-review/scripts/accuracy.py
REF=$1; OUT=$2
while read m x k; do
  [ -z "$m" ] && continue
  s=$(date +%s)
  timeout 300 $PY $A $REF $EXTRA --at $m $x $k > /tmp/p3at_$$.txt 2>&1
  rc=$?
  ref=$(grep "reference" /tmp/p3at_$$.txt | head -1 | sed 's/.*(float \(.*\))/\1/')
  cnd=$(grep "cond (arg)" /tmp/p3at_$$.txt | awk '{print $3}')
  cs=$(grep "cond (shapes)" /tmp/p3at_$$.txt | awk '{print $3}')
  sc=$(grep "scipy " /tmp/p3at_$$.txt | sed 's/ *scipy [0-9.]* *//')
  echo "$m($x, $k) | $sc | ref $ref | cond $cnd | conds $cs | rc $rc | $(( $(date +%s)-s ))s" | tee -a $OUT
  [ $rc -ne 0 ] && tail -3 /tmp/p3at_$$.txt
done
rm -f /tmp/p3at_$$.txt
