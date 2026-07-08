#!/usr/bin/env python3
"""检查孤立 .raw 文件：验证它们是否安全可删"""
import os, sys
from collections import Counter

images_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/data/LUNA16/images"

raw_files = []
mhd_files = []

for root, dirs, files in os.walk(images_dir):
    for f in files:
        if f.endswith('.raw'):
            raw_files.append(f)
        elif f.endswith('.mhd'):
            mhd_files.append(f)

raw_names = {f.replace('.raw', '') for f in raw_files}
mhd_names = {f.replace('.mhd', '') for f in mhd_files}

# 1. 孤立 raw：没有同名 .mhd 的
orphans = raw_names - mhd_names
paired = raw_names & mhd_names
mhd_only = mhd_names - raw_names

print(f"{'='*60}")
print(f"  .raw/.mhd 文件完整性检查")
print(f"{'='*60}")
print(f"  .raw 总数:   {len(raw_files)}")
print(f"  .mhd 总数:   {len(mhd_files)}")
print(f"  配对 (raw+mhd): {len(paired)}")
print(f"  孤立 raw (缺mhd): {len(orphans)}")
print(f"  孤立 mhd (缺raw): {len(mhd_only)}")

# 2. 文件名去重检查
raw_counter = Counter(f.replace('.raw', '') for f in raw_files)
dups = {k: v for k, v in raw_counter.items() if v > 1}
if dups:
    print(f"\n  ⚠️  重复 .raw 文件名: {len(dups)} 个")
    for k, v in list(dups.items())[:5]:
        print(f"      {k}.raw ×{v}")
else:
    print(f"\n  ✅ 无重复 .raw 文件")

# 3. 结论
print(f"\n{'='*60}")
if len(mhd_only) == 0 and len(dups) == 0:
    print(f"  结论: {len(orphans)} 个孤立 .raw 可安全删除")
    print(f"  (它们没有对应的 .mhd, 无法被 simpleitk 读取)")
else:
    if mhd_only:
        print(f"  ⚠️  {len(mhd_only)} 个 .mhd 缺 raw — 无法加载")
    if dups:
        print(f"  ⚠️  有重复文件需手动处理")
print(f"{'='*60}")
