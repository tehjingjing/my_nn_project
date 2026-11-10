"""Read-only exact duplicate audit. No images are deleted."""
import argparse, csv, hashlib
from collections import defaultdict
from pathlib import Path

EXTS={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
PROJECT_DIR=Path(__file__).resolve().parents[1]
DEFAULT=PROJECT_DIR/'data'/'durian_leaf_dataset'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=DEFAULT)
    ap.add_argument('--report',type=Path,default=Path('duplicate_audit.csv'))
    a=ap.parse_args(); groups=defaultdict(list)
    for split in ('train','val','test'):
        root=a.data_dir/split
        if not root.is_dir(): raise FileNotFoundError(root)
        for cls in sorted(p for p in root.iterdir() if p.is_dir()):
            for p in sorted(cls.iterdir()):
                if p.is_file() and p.suffix.lower() in EXTS: groups[digest(p)].append((split,cls.name,p))
    rows=[]
    for idx,(h,items) in enumerate(groups.items(),1):
        if len(items)>1:
            conflict=len({c for _,c,_ in items})>1
            for split,cls,p in items:
                rows.append({'group':idx,'sha256':h,'label_conflict':conflict,'split':split,'class':cls,'path':str(p)})
    report=a.report.resolve(); report.parent.mkdir(parents=True,exist_ok=True)
    with report.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['group','sha256','label_conflict','split','class','path']);w.writeheader();w.writerows(rows)
    print(f'Exact duplicate groups: {len(set(r["group"] for r in rows))}; matching files: {len(rows)}')
    print(f'Read-only report saved: {report}')

if __name__=='__main__': main()
