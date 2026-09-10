"""Official public ReplicaCAD v1.6 and Fetch v2.0 assets; resume + SHA256 checks."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import re

SOURCES = [('ReplicaCAD_dataset','v1.6','replica_cad'), ('hab_fetch','v2.0','robots/hab_fetch')]


def get_tree(repo, revision):
    url = f'https://huggingface.co/api/datasets/ai-habitat/{repo}/tree/{revision}?recursive=true&limit=1000'
    items = []
    while url:
        with urllib.request.urlopen(url, timeout=90) as r:
            items.extend(json.load(r))
            match = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get('Link',''))
            url = match.group(1) if match else None
    return items


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True); a = p.parse_args()
    root = Path(a.data).resolve(); root.mkdir(parents=True, exist_ok=True)
    jobs, manifest = [], []
    for repo, rev, folder in SOURCES:
        tree = get_tree(repo, rev)
        for x in tree:
            if x['type'] != 'file':
                continue
            dest = (root/folder/x['path']).resolve()
            if not dest.is_relative_to(root):
                raise ValueError('Unsafe asset path')
            sha = x.get('lfs',{}).get('oid')
            item = dict(repo=repo, revision=rev, path=str(dest.relative_to(root)), size=x['size'], sha256=sha)
            manifest.append(item)
            if dest.exists() and dest.stat().st_size == x['size']:
                if sha is None or hashlib.sha256(dest.read_bytes()).hexdigest() == sha:
                    continue
            jobs.append((f'https://huggingface.co/datasets/ai-habitat/{repo}/resolve/{rev}/{x["path"]}',dest,x['size'],sha))
    def fetch(job):
        url,dest,size,sha = job
        for attempt in range(3):
            try:
                data = urllib.request.urlopen(url, timeout=90).read()
                if len(data) != size or (sha and hashlib.sha256(data).hexdigest()!=sha):
                    raise ValueError('Asset size/hash mismatch')
                dest.parent.mkdir(parents=True, exist_ok=True)
                temp = dest.with_suffix(dest.suffix+'.part'); temp.write_bytes(data); temp.replace(dest)
                return str(dest)
            except Exception:
                if attempt == 2:
                    raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for i, _ in enumerate(executor.map(fetch,jobs),1):
            if i % 25 == 0 or i == len(jobs):
                print(f'Assets {i}/{len(jobs)}', flush=True)
    (root/'asset_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'ASSETS READY: {len(manifest)} verified files, {sum(x["size"] for x in manifest)/1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
