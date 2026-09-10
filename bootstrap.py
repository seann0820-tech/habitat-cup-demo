"""Run with the Colab kernel Python. All Habitat work uses an isolated Python 3.9."""
import argparse
import io
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request


def main():
    p = argparse.ArgumentParser(); p.add_argument('--prefix', required=True)
    a = p.parse_args()
    prefix = Path(a.prefix).resolve()
    bin_dir = prefix.parent/'tools'; bin_dir.mkdir(parents=True, exist_ok=True)
    mamba = bin_dir/'micromamba'
    if not mamba.exists():
        url = 'https://micro.mamba.pm/api/micromamba/linux-64/2.9.0'
        data = urllib.request.urlopen(url, timeout=120).read()
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:bz2') as archive:
            mamba.write_bytes(archive.extractfile('bin/micromamba').read())
        mamba.chmod(0o755)
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None); environment.pop('PYTHONHOME', None)
    environment['MPLBACKEND'] = 'Agg'
    environment['MAMBA_ROOT_PREFIX'] = str(prefix.parent/'mamba-cache')
    def run(cmd):
        print('Running:', ' '.join(map(str, cmd)), flush=True)
        subprocess.run(list(map(str,cmd)), check=True, env=environment)
    run([mamba,'create','-y','-p',prefix,'-c','conda-forge','-c','aihabitat',
         'python=3.9','habitat-sim=0.3.3','withbullet','headless','numpy=1.26.4','pip','libstdcxx-ng>=12'])
    py = prefix/'bin/python'
    # Small MLP ensemble runs on CPU; NVIDIA GPU is used only for optional EGL video.
    run([py,'-m','pip','install','torch==2.5.1','--index-url','https://download.pytorch.org/whl/cpu'])
    run([py,'-m','pip','install','opencv-python==4.10.0.84','gym==0.23.0',
         'hydra-core==1.3.2','imageio-ffmpeg==0.5.1',
         'https://github.com/facebookresearch/habitat-lab/archive/refs/tags/v0.3.3.zip#subdirectory=habitat-lab'])
    run([py,'-c','import habitat_sim,habitat,torch,numpy; print("READY",habitat_sim.__version__,torch.__version__,numpy.__version__)'])
    (prefix.parent/'environment-python.txt').write_text(str(py))


if __name__ == '__main__':
    main()
