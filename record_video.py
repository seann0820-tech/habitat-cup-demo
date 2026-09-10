"""Probe Habitat 0.3.3 EGL rendering, then record an existing checkpoint.

The notebook loads this module with runpy and calls repair_and_record with its
source, Python, asset, checkpoint-directory and workspace paths. No training,
package installation or system driver changes. Failed native probes are isolated
in subprocesses. A successful empty-scene probe does not validate scene assets.
"""
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


PROBE_CODE = r'''
import json, sys
import habitat_sim
c = habitat_sim.SimulatorConfiguration()
c.scene_id = "NONE"
c.gpu_device_id = int(sys.argv[1])
c.create_renderer = True
c.requires_textures = True
c.enable_physics = False
sensor = habitat_sim.CameraSensorSpec()
sensor.uuid = "rgb"
sensor.sensor_type = habitat_sim.SensorType.COLOR
sensor.resolution = [32, 32]
a = habitat_sim.agent.AgentConfiguration()
a.sensor_specifications = [sensor]
sim = habitat_sim.Simulator(habitat_sim.Configuration(c, [a]))
try:
    frame = sim.get_sensor_observations()["rgb"]
    assert tuple(frame.shape[:2]) == (32, 32), frame.shape
    info = dict(habitat_version=habitat_sim.__version__,
                gpu_device_id=c.gpu_device_id, frame_shape=list(frame.shape),
                vendor="unknown", renderer="unknown", opengl="unknown",
                renderer_info_source="unavailable")
finally:
    sim.close()
print("HABITAT_RENDER_OK=" + json.dumps(info), flush=True)
'''


def _renderer_info_from_log(info, output):
    """Optional metadata must never invalidate a successfully rendered frame.

    Habitat rendering may work while the separate Python Magnum binding has no
    current context. Read the native startup log instead of querying that binding.
    Unknown/missing metadata stays explicit; a selected vendor is not proof of GPU.
    """
    info = dict(info)
    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Renderer:"):
            value = line[len("Renderer:"):].strip()
            renderer, separator, vendor = value.rpartition(" by ")
            info["renderer"] = renderer if separator else value
            info["vendor"] = vendor if separator else "unknown"
            info["renderer_info_source"] = "native_startup_log"
        elif line.startswith("OpenGL version:"):
            info["opengl"] = line[len("OpenGL version:"):].strip()
    return info


def _capture(command, env, cwd, timeout):
    try:
        p = subprocess.run(command, env=env, cwd=str(cwd), text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return 124, output + "\n检查超过时限，已终止子进程。\n"
    except OSError as exc:
        return 127, str(exc)


def _patch_render_device(code):
    path = Path(code) / "cup_baseline/env.py"
    source = path.read_text(encoding="utf-8")
    replacement = ('cfg.gpu_device_id = int(os.environ.get('
                   '"HABITAT_RENDER_GPU_ID", "0")) if self.render_enabled else 0')
    lines = source.splitlines(keepends=True)
    matches = [node for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Assign)
               and len(node.targets) == 1
               and isinstance(node.targets[0], ast.Attribute)
               and isinstance(node.targets[0].value, ast.Name)
               and node.targets[0].value.id == "cfg"
               and node.targets[0].attr == "gpu_device_id"]
    if len(matches) != 1:
        raise RuntimeError("env.py 的设备配置与此版本不符；未修改源码，请提供该文件。")
    node = matches[0]
    if lines[node.lineno - 1].strip() == replacement:
        return
    try:
        old_value = ast.literal_eval(node.value)
    except (ValueError, TypeError):
        old_value = None
    if old_value not in (0, -1) or node.lineno != node.end_lineno:
        raise RuntimeError("env.py 已有其他设备配置；未覆盖，请提供该文件。")
    indent = lines[node.lineno - 1][:node.col_offset]
    lines[node.lineno - 1] = indent + replacement + "\n"
    patched = "".join(lines)
    # This project imports math at top level; keep its module docstring intact.
    if not any(isinstance(n, ast.Import) and any(a.name == "os" for a in n.names)
               for n in ast.parse(patched).body):
        patched = patched.replace("import math\n", "import math\nimport os\n", 1)
        if "\nimport os\n" not in patched:
            raise RuntimeError("未找到预期的 import math；未修改源码。")
    compile(patched, str(path), "exec")
    backup = path.with_name("env.py.before_egl_fix")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    path.write_text(patched, encoding="utf-8")
    print("录像设备配置已更新：", path, flush=True)


def repair_and_record(code, python, data, run_dir, work, task="pick_cup", domain=0):
    code, python, data, run_dir, work = map(Path, (code, python, data, run_dir, work))
    code, python, data, run_dir, work = [p.resolve() for p in (code, python, data, run_dir, work)]
    checkpoint = run_dir / "checkpoint.pt"
    for path in (python, code / "cup_baseline/env.py", checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {path}；请先准备运行环境和已训练的 checkpoint。")
    if not data.is_dir():
        raise FileNotFoundError(f"场景目录不存在：{data}；请先完成 Cell 3 的资产准备。")
    logs = work / "logs" / f"video_{task}_{domain}"
    logs.mkdir(parents=True, exist_ok=True)
    out = run_dir / "video" / f"{task}_{domain}"
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(MPLBACKEND="Agg", PYTHONUNBUFFERED="1", HABITAT_SIM_LOG="error",
               MAGNUM_LOG="default", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    print("使用 checkpoint：", checkpoint, flush=True)
    diagnostics = {"python": str(python), "checkpoint": str(checkpoint),
                   "environment": {k: env.get(k, "未设置") for k in (
                       "CUDA_VISIBLE_DEVICES", "EGL_PLATFORM", "LD_LIBRARY_PATH",
                       "__EGL_VENDOR_LIBRARY_FILENAMES", "__EGL_VENDOR_LIBRARY_DIRS")}}
    smi = shutil.which("nvidia-smi")
    if smi:
        rc, output = _capture([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                              env, code, 10)
        diagnostics["nvidia_smi"] = {"returncode": rc, "output": output.strip()}
    else:
        diagnostics["nvidia_smi"] = {"returncode": 127, "output": "当前运行时找不到 nvidia-smi"}
    print("nvidia-smi：", diagnostics["nvidia_smi"]["output"], flush=True)

    # Resolve an installed NVIDIA EGL library in the same Python environment.
    # A private GLVND descriptor affects only child processes, never /usr or /etc.
    rc, output = _capture([str(python), "-c",
        "import ctypes, ctypes.util; "
        "p=ctypes.util.find_library('EGL_nvidia') or 'libEGL_nvidia.so.0'; "
        "ctypes.CDLL(p); print(p)"], env, code, 10)
    diagnostics["nvidia_egl_library"] = {"returncode": rc, "output": output.strip()}
    candidates = [("cuda0", 0, {})]
    if rc == 0 and output.strip():
        vendor_json = logs / "habitat_nvidia_egl.json"
        vendor_json.write_text(json.dumps({"file_format_version": "1.0.0",
            "ICD": {"library_path": output.strip().splitlines()[-1]}}), encoding="utf-8")
        override = {"__EGL_VENDOR_LIBRARY_FILENAMES": str(vendor_json)}
        candidates += [("nvidia_cuda0", 0, override), ("nvidia_default_egl", -1, override)]
    else:
        print("未能在 Habitat 子环境加载 NVIDIA EGL 库；仍将检查默认 EGL。", flush=True)
    candidates.append(("default_egl", -1, {}))

    diagnostics["attempts"] = []
    selected = None
    selected_env = None
    for name, device, overrides in candidates:
        print(f"渲染检查：{name}，gpu_device_id={device}（最多 45 秒）", flush=True)
        candidate_env = dict(env, **overrides)
        candidate_env["HABITAT_RENDER_GPU_ID"] = str(device)
        rc, output = _capture([str(python), "-c", PROBE_CODE, str(device)],
                              candidate_env, code, 45)
        log_path = logs / f"render_probe_{name}.log"
        log_path.write_text(output, encoding="utf-8")
        info = None
        for line in output.splitlines():
            if line.startswith("HABITAT_RENDER_OK="):
                try:
                    info = _renderer_info_from_log(json.loads(line.split("=", 1)[1]), output)
                except json.JSONDecodeError:
                    pass
        success = rc == 0 and isinstance(info, dict) and "renderer" in info
        diagnostics["attempts"].append({"name": name, "device": device,
            "returncode": rc, "success": success, "log": str(log_path),
            "tail": "\n".join(output.splitlines()[-14:]), "renderer_info": info})
        if success:
            selected = dict(info, selection=name, env_overrides=overrides)
            selected_env = candidate_env
            break
        print("  未通过；日志末尾：\n" + "\n".join(output.splitlines()[-4:]), flush=True)
    diagnostics["selected"] = selected
    diagnostic_path = logs / "render_diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    if selected is None:
        print("诊断文件：", diagnostic_path, flush=True)
        raise RuntimeError("所有渲染检查均失败，未启动录像。请提供上方检查输出或 render_diagnostics.json；"
                           "checkpoint 和训练指标仍可使用。")

    renderer = (selected["vendor"] + " " + selected["renderer"]).lower()
    if any(x in renderer for x in ("llvmpipe", "softpipe", "swrast", "software rasterizer")):
        backend = "software"
    elif "nvidia" in renderer:
        backend = "nvidia"
    else:
        backend = "other_or_unknown"
    selected["backend_kind"] = backend
    (out / "render_backend.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    print("渲染检查通过：", selected["vendor"], "|", selected["renderer"], flush=True)
    if backend == "software":
        print("当前为软件渲染，录像可能较慢；录像耗时不能作为 GPU 能耗/性能结果。", flush=True)
    elif backend == "other_or_unknown":
        print("图像检查已通过，但日志未确认 NVIDIA/软件渲染器；不会据此宣称 GPU 性能。", flush=True)
    print("开始录制固定 episode；录像不写入训练/测试指标。", flush=True)
    _patch_render_device(code)
    command = [str(python), "-m", "cup_baseline.run", "record", "--data", str(data),
               "--out", str(out), "--checkpoint", str(checkpoint), "--episode-seed", "250000", "--task", task, "--domain", str(domain)]
    log_path = logs / "record_video.log"
    selected_env["MAGNUM_LOG"] = "quiet"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=str(code), env=selected_env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if not line.startswith("PluginManager::Manager: duplicate"):
                    print(line, end="", flush=True)
            rc = proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    if rc:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
        raise RuntimeError(f"渲染检查通过，但完整场景录像失败（退出码 {rc}）。\n{log_path}\n{tail}")
    video = out / "baseline_demo.mp4"
    if not video.is_file() or video.stat().st_size == 0:
        raise RuntimeError(f"录像进程结束，但没有生成有效视频文件；请提供 {log_path}")
    print("录像已生成：", video, flush=True)
    return video


if __name__ == "__main__":
    missing = [k for k in ("CODE", "PYTHON", "DATA", "RUN_DIR", "WORK") if k not in globals()]
    if missing:
        raise RuntimeError("请在原 Colab notebook 中执行此脚本，并恢复这些路径变量：" + ", ".join(missing))
    video_path = repair_and_record(CODE, PYTHON, DATA, RUN_DIR, WORK)
    from IPython.display import Video, display
    display(Video(str(video_path), embed=True))
