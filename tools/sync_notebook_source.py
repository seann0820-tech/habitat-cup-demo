"""Refresh the Colab notebook's embedded source; clear stale execution output.

Run from any directory: python tools/sync_notebook_source.py
Only the clean root notebook is updated. results/recorded_run.ipynb is untouched.
"""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'Habitat_Cup_v4.ipynb'


def main():
    files = [ROOT / name for name in (
        'ENVIRONMENT_zh.md', 'LOCAL_VALIDATION_zh.md', 'METACOGNITION_zh.md',
        'README_zh.md', 'bootstrap.py', 'download_assets.py', 'record_video.py')]
    files += list((ROOT / 'cup_baseline').glob('*.py'))
    files += list((ROOT / 'tests').glob('test_*.py'))
    files = sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(),
                                   date_time=(2026, 9, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    payload = base64.b64encode(buffer.getvalue()).decode('ascii')
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    matches = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == 'PAYLOAD'
                for target in node.targets
            ):
                matches.append((cell, source, node))
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one top-level PAYLOAD assignment.')
    cell, source, node = matches[0]
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [f'PAYLOAD = {payload!r}\n']
    cell['source'] = lines
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
            for key in ('execution', 'executionInfo', 'outputId'):
                cell.get('metadata', {}).pop(key, None)
    notebook.get('metadata', {}).pop('widgets', None)
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n',
                        encoding='utf-8')
    print(f'Embedded {len(files)} files in {NOTEBOOK.name}; execution output cleared.')
    print('Payload ZIP SHA256:', hashlib.sha256(buffer.getvalue()).hexdigest())


if __name__ == '__main__':
    main()
