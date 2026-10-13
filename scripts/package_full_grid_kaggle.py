"""Package a private two-GPU full-grid continuation without uploading base models.

The existing completed owner kernel is attached read-only. The submitted code
contains only allowlisted project sources, the fixed study plan and a small
orchestrator. It contains no credentials, data CSVs, local notes or model bytes.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import inspect
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE_KERNEL = 'dongmyungpark/kd-gtsrb-track-split-20261012'
BASE_KERNEL_SOURCE_SHA = '5fdd504a08f3babdc9f2bee902034726655c534406460360ccc878c64b356eb6'


def checked_https_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
        or parsed.fragment or parsed.hostname in {'localhost', '127.0.0.1', '::1'}):
        raise ValueError('Public base artifacts require an anonymous HTTPS URL')
    return value


def validate_public_descriptor(value: dict, expected_inventory_sha: str) -> dict:
    if (value.get('schema_version') != 1 or value.get('kind') != 'full-grid-base-archive'
        or type(value.get('size_bytes')) is not int or value['size_bytes'] <= 0
        or not re.fullmatch('[0-9a-f]{64}', value.get('sha256', ''))):
        raise ValueError('Invalid public base archive descriptor')
    checked_https_url(value['url'])
    prefix = value.get('archive_prefix', '')
    if prefix and (str(PurePosixPath(prefix)) != prefix or PurePosixPath(prefix).is_absolute()
                   or '..' in PurePosixPath(prefix).parts or '\\' in prefix or ':' in prefix):
        raise ValueError('Unsafe public base archive prefix')
    inventory = value.get('inventory')
    if not isinstance(inventory, dict) or len(inventory) != 47:
        raise ValueError('Exactly forty-seven original base artifacts are required')
    folded = set()
    for name, spec in inventory.items():
        path = PurePosixPath(name)
        if (not name or str(path) != name or path.is_absolute() or '..' in path.parts
            or '\\' in name or ':' in name or name.casefold() in folded
            or set(spec) != {'bytes', 'sha256'} or type(spec['bytes']) is not int
            or spec['bytes'] <= 0 or not re.fullmatch('[0-9a-f]{64}', spec['sha256'])):
            raise ValueError('Unsafe or invalid original base artifact inventory')
        folded.add(name.casefold())
    calculated = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if calculated != expected_inventory_sha or value.get('inventory_sha256') != calculated:
        raise ValueError('Public base inventory differs from the registered original artifacts')
    return value


def download_public_base(descriptor: dict, target: Path) -> None:
    """Download anonymously, rejecting size/hash drift and insecure redirects."""
    class SecureRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, message, headers, new_url):
            checked_https_url(new_url)
            return super().redirect_request(request, fp, code, message, headers, new_url)

    opener = urllib.request.build_opener(SecureRedirect())
    request = urllib.request.Request(checked_https_url(descriptor['url']),
                                    headers={'Accept-Encoding': 'identity'})
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    digest = hashlib.sha256()
    with opener.open(request, timeout=60) as response, target.open('xb') as stream:
        if response.status != 200 or response.headers.get('Content-Encoding', 'identity') != 'identity':
            raise ValueError('Public base archive must return HTTP 200 without content encoding')
        declared = response.headers.get('Content-Length')
        if declared is not None and int(declared) != descriptor['size_bytes']:
            raise ValueError('Public base archive Content-Length differs')
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > descriptor['size_bytes']:
                raise ValueError('Public base archive exceeds its declared byte count')
            stream.write(block)
            digest.update(block)
    if count != descriptor['size_bytes'] or digest.hexdigest() != descriptor['sha256']:
        raise ValueError('Public base archive byte count or SHA256 differs')


def extract_public_base(archive_path: Path, descriptor: dict, target: Path) -> None:
    """Extract only the original exact inventory, never extra ZIP members."""
    target.mkdir(parents=True, exist_ok=False)
    prefix = descriptor.get('archive_prefix', '')
    expected = {(prefix+'/' if prefix else '')+name: (name, spec)
                for name, spec in descriptor['inventory'].items()}
    with zipfile.ZipFile(archive_path) as archive:
        indexed = {}
        folded = set()
        for info in archive.infolist():
            name = info.filename.rstrip('/') if info.is_dir() else info.filename
            path = PurePosixPath(name)
            mode = (info.external_attr >> 16) & 0o170000
            if (not name or str(path) != name or path.is_absolute() or '..' in path.parts
                or '\\' in name or ':' in name or mode == 0o120000
                or name.casefold() in folded):
                raise ValueError('Unsafe, duplicated or symlinked public ZIP member')
            folded.add(name.casefold())
            if not info.is_dir(): indexed[info.filename] = info
        if not set(expected) <= set(indexed):
            raise ValueError('Public archive is missing original base artifacts')
        for member, (relative, spec) in expected.items():
            info = indexed[member]
            if info.file_size != spec['bytes']:
                raise ValueError('Original base artifact byte count differs')
            destination = target.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            with archive.open(info) as source, destination.open('xb') as output:
                while block := source.read(1024 * 1024):
                    count += len(block)
                    if count > spec['bytes']: raise ValueError('Original base artifact exceeds declared size')
                    output.write(block)
                    digest.update(block)
            if count != spec['bytes'] or digest.hexdigest() != spec['sha256']:
                raise ValueError('Original base artifact SHA256 differs')


def restore_public_base(descriptor: dict, expected_inventory_sha: str) -> Path:
    validate_public_descriptor(descriptor, expected_inventory_sha)
    root = Path(tempfile.mkdtemp(prefix='kd-public-base-', dir='/tmp'))
    archive = root/'verified-base.zip'
    download_public_base(descriptor, archive)
    target = root/'original-base-run'
    extract_public_base(archive, descriptor, target)
    return target


def render_worker() -> str:
    header = 'import hashlib, json, re, tempfile, urllib.parse, urllib.request, zipfile\nfrom pathlib import Path, PurePosixPath\n'
    helpers = '\n\n'.join(inspect.getsource(function) for function in
                          (checked_https_url, validate_public_descriptor, download_public_base,
                           extract_public_base, restore_public_base))
    return header+helpers+'\n\n'+WORKER

WORKER = r'''
import csv, hashlib, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

source = Path(__file__).resolve().parent
sys.path.insert(0, str(source))
from knowledge_distillation.data import build_split_manifest, file_sha256, safe_dataset_path
from knowledge_distillation.experiments import source_inventory
from scripts.full_grid_study import inspect_base_run, load_plan

contract = json.loads((source / 'full-grid-contract.json').read_text())
plan_path = source / 'configs/full-grid-plan.json'
plan = load_plan(plan_path)
output = Path('/kaggle/working/full-grid-results')
runtime_path = Path('/tmp/kd-full-grid-runtime/kaggle-runtime.json')
start = time.perf_counter()
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

def write(name, value):
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def cli(stage, base, data, *extra):
    return [sys.executable, '-I', '-X', 'utf8', '-u', str(source / 'scripts/full_grid_study.py'),
            '--data', str(data), '--base-run', str(base), '--output', str(output),
            '--plan', str(plan_path), '--stage', stage, *extra]

try:
    if source_inventory(source) != contract['source_inventory']:
        raise RuntimeError('The immutable fourteen-file training core differs')
    candidates = []
    if contract.get('public_base_descriptor'):
        candidates.append(restore_public_base(contract['public_base_descriptor'],
                          plan['base_bindings']['artifact_inventory_sha256']))
    else:
        for marker in Path('/kaggle/input').rglob('kernel-completion.json'):
            parent = marker.parent
            if (parent / 'splits.json').is_file() and (parent / 'source-snapshot.json').is_file():
                if all((parent / name).is_file() and not (parent / name).is_symlink()
                       and (parent / name).stat().st_size == spec['bytes']
                       and file_sha256(parent / name) == spec['sha256']
                       for name, spec in contract['base_inventory'].items()):
                    candidates.append(parent)
    if len(candidates) != 1:
        raise RuntimeError('Expected exactly one checksum-bound original kernel output mount')
    base = candidates[0]
    if inspect_base_run(base, plan)['inventory'] != contract['base_inventory']:
        raise RuntimeError('The original mounted teacher, students or evidence differ')
    print('KD GRID BASE VERIFIED', json.dumps({'files': len(contract['base_inventory']),
          'bytes': sum(item['bytes'] for item in contract['base_inventory'].values())}), flush=True)

    mounts = [root for root in (Path('/kaggle/input/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign'),
                               Path('/kaggle/input/gtsrb-german-traffic-sign')) if (root / 'Train.csv').is_file()]
    if len(mounts) != 1 or file_sha256(mounts[0] / 'Train.csv') != contract['train_csv_sha256']:
        raise RuntimeError('Expected the exact original GTSRB version1 Train CSV')
    mounted = mounts[0]
    data = Path('/tmp/kd-full-grid-training-data')
    data.mkdir(exist_ok=False)
    shutil.copyfile(mounted / 'Train.csv', data / 'Train.csv')
    with (data / 'Train.csv').open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            original = safe_dataset_path(mounted, row['Path'])
            destination = safe_dataset_path(data, row['Path'])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
    manifest = build_split_manifest(data, seed=378, ratios=(.7,.15,.15))
    if manifest['manifest_sha256'] != contract['manifest_sha256']:
        raise RuntimeError('Train image bytes or track split differ from the original protocol')
    print('KD GRID DATA VERIFIED', json.dumps(manifest['counts']), flush=True)

    # Preparation requires an empty study directory, and imports no TensorFlow.
    subprocess.run(cli('prepare', base, data), cwd=source, check=True)
    metadata = json.loads(runtime_path.read_text())
    if metadata.get('gpu_count', 0) < 1 or any(item.get('name') != 'Tesla T4' for item in metadata['gpus']):
        raise RuntimeError('The actual managed runtime must expose Tesla T4 GPUs')
    count = 2 if metadata['gpu_count'] >= 2 else 1
    write('managed-runtime.json', metadata)
    write('wrapper-input-verification.json', {'base_kernel': contract['base_kernel'],
        'original_kernel_version_at_preparation': contract['base_kernel_version'],
        'original_kernel_source_sha256': contract['base_kernel_source_sha256'],
        'base_inventory': contract['base_inventory'], 'manifest_sha256': manifest['manifest_sha256'],
        'train_csv_sha256': contract['train_csv_sha256'], 'official_test_csv_read': False,
        'worker_count': count, 'preparation_wall_seconds': time.perf_counter()-start})
    print('KD GRID WORKERS START', json.dumps({'workers': count, 'new_grid_runs': 21,
        'diagnostic_runs': 3, 'gpu_assignment': list(range(count))}), flush=True)

    workers = []
    readers = []
    def relay(index, process):
        with (output / f'worker-{index}-stdout.log').open('x', encoding='utf-8') as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f'[worker {index}] ' + line.rstrip('\n'), flush=True)
    for index in range(count):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(index)
        process = subprocess.Popen(cli('train-worker', base, data, '--worker-index', str(index),
                                       '--worker-count', str(count)), cwd=source, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        workers.append(process)
        reader = threading.Thread(target=relay, args=(index, process), daemon=False)
        reader.start()
        readers.append(reader)
    codes = [process.wait() for process in workers]
    for reader in readers:
        reader.join()
    write('wrapper-worker-completion.json', {'worker_count': count, 'exit_codes': codes,
         'wall_seconds': time.perf_counter()-start, 'no_parent_tensorflow_import': 'tensorflow' not in sys.modules})
    if any(codes):
        raise RuntimeError('An assigned worker failed; no finalization, freeze or automatic retraining')
    for stage in ('finalize', 'freeze'):
        print('KD GRID STAGE START', stage, flush=True)
        subprocess.run(cli(stage, base, data), cwd=source, check=True)
        print('KD GRID STAGE COMPLETE', stage, flush=True)
    write('kernel-completion.json', {'status': 'completed', 'stage': 'full_grid_and_diagnostics_frozen',
         'additional_grid_runs': 21, 'additional_diagnostic_runs': 3,
         'worker_count': count, 'wall_seconds': time.perf_counter()-start,
         'locked_holdout_evaluated': False, 'official_test_evaluated': False,
         'teacher_retrained': False, 'original_base_run_modified': False})
    print('KD FULL GRID COMPLETE', flush=True)
except Exception as error:
    output.mkdir(parents=True, exist_ok=True)
    failure = output / 'kernel-failure.json'
    if not failure.exists():
        write('kernel-failure.json', {'status': 'failed', 'error_type': type(error).__name__,
              'message': str(error)[:1000], 'wall_seconds': time.perf_counter()-start,
              'automatic_retraining_or_reselection': False})
    print('KD FULL GRID FAILURE', type(error).__name__, str(error)[:1000], flush=True)
    raise
'''


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def archive_sources(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe source member')
            info = zipfile.ZipInfo(name, (2026, 10, 13, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return stream.getvalue()


def render_outer(raw: bytes, inventory: dict) -> str:
    encoded = base64.b64encode(raw).decode('ascii')
    return '''import base64, hashlib, io, json, subprocess, sys, zipfile
from pathlib import Path, PurePosixPath
PAYLOAD = __PAYLOAD__
ARCHIVE_SHA = __ARCHIVE_SHA__
INVENTORY = __INVENTORY__
source = Path('/tmp/kd-full-grid-source')
source.mkdir(exist_ok=False)
payload = base64.b64decode(PAYLOAD)
if hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA:
    raise RuntimeError('Embedded source ZIP checksum differs')
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    if len(archive.infolist()) != len(INVENTORY) or set(archive.namelist()) != set(INVENTORY):
        raise RuntimeError('Embedded source inventory differs')
    for info in archive.infolist():
        name = PurePosixPath(info.filename)
        if name.is_absolute() or '..' in name.parts or '\\\\' in info.filename or ':' in info.filename or info.is_dir():
            raise RuntimeError('Unsafe embedded source path')
        data = archive.read(info)
        expected = INVENTORY[info.filename]
        if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
            raise RuntimeError('Embedded source file differs')
        target = source.joinpath(*name.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
result = subprocess.run([sys.executable, str(source/'scripts/setup_kaggle_runtime.py'),
    '--project-root', str(source), '--runtime', '/tmp/kd-full-grid-runtime',
    '--metadata-file', '/tmp/kd-full-grid-runtime/kaggle-runtime.json', '--',
    str(source/'full-grid-worker.py')], cwd=source)
if result.returncode:
    target = Path('/kaggle/working/full-grid-results')
    target.mkdir(parents=True, exist_ok=True)
    failure = target/'outer-failure.json'
    if not failure.exists():
        failure.write_text(json.dumps({'status':'failed','bootstrap_or_worker_exit_code':result.returncode})+'\\n')
raise SystemExit(result.returncode)
'''.replace('__PAYLOAD__', repr(encoded)).replace('__ARCHIVE_SHA__', repr(digest(raw))).replace('__INVENTORY__', repr(inventory))


def package(base_run: Path | None, output: Path, dataset_root: Path, *, kernel_id='dongmyungpark/kd-gtsrb-full-grid',
            base_descriptor: Path | None = None) -> dict:
    from scripts.full_grid_study import inspect_base_run, load_plan
    from knowledge_distillation.experiments import source_inventory
    if output.exists():
        raise FileExistsError('Preserve an existing kernel package')
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[a-z0-9]+(?:-[a-z0-9]+)*', kernel_id):
        raise ValueError('Use OWNER/SLUG with a lowercase hyphenated kernel slug')
    plan_path = ROOT / 'configs/full-grid-plan.json'
    plan = load_plan(plan_path)
    public = None
    if base_descriptor is not None:
        public = validate_public_descriptor(json.loads(base_descriptor.read_text(encoding='utf-8')),
                                            plan['base_bindings']['artifact_inventory_sha256'])
    if base_run is not None:
        base = inspect_base_run(base_run, plan)
        if public and base['inventory'] != public['inventory']:
            raise ValueError('Local base artifacts differ from the public descriptor')
        inventory_base = base['inventory']
    elif public:
        inventory_base = public['inventory']
    else:
        raise ValueError('Provide --base-run or a checksum-bound --base-descriptor')
    core = source_inventory(ROOT)
    if hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != plan['base_bindings']['core_source_sha256']:
        raise ValueError('The registered immutable training core differs before packaging')
    files = {name: (ROOT/name).read_bytes() for name in core}
    for name in ('scripts/__init__.py', 'scripts/setup_kaggle_runtime.py', 'scripts/full_grid_study.py',
                 'configs/full-grid-plan.json'):
        files[name] = (ROOT/name).read_bytes()
    contract = {'schema_version': 1, 'source_inventory': core,
        'base_kernel': BASE_KERNEL, 'base_kernel_version': 1,
        'base_kernel_source_sha256': BASE_KERNEL_SOURCE_SHA,
        'base_inventory': inventory_base, 'manifest_sha256': plan['base_bindings']['manifest_sha256'],
        'train_csv_sha256': digest((dataset_root/'Train.csv').read_bytes())}
    if public: contract['public_base_descriptor'] = public
    files['full-grid-contract.json'] = (json.dumps(contract, indent=2)+'\n').encode()
    worker = render_worker()
    files['full-grid-worker.py'] = worker.encode()
    raw = archive_sources(files)
    inventory = {name: {'bytes': len(data), 'sha256': digest(data)} for name, data in sorted(files.items())}
    outer = render_outer(raw, inventory)
    compile(outer, 'run-full-grid.py', 'exec')
    compile(worker, 'full-grid-worker.py', 'exec')
    output.mkdir(parents=True)
    (output/'run-full-grid.py').write_text(outer, encoding='utf-8', newline='\n')
    metadata = {'id': kernel_id, 'title': kernel_id.split('/')[1].replace('-', ' '),
        'code_file': 'run-full-grid.py', 'language': 'python', 'kernel_type': 'script',
        'is_private': True, 'enable_gpu': True, 'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4',
        'dataset_sources': ['meowmeowmeowmeowmeow/gtsrb-german-traffic-sign'],
        'competition_sources': [], 'kernel_sources': [] if public else [BASE_KERNEL], 'model_sources': []}
    (output/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    provenance = {'schema_version': 1, 'kernel_id': kernel_id, 'private': True,
        'source_archive_sha256': digest(raw), 'source_archive_bytes': len(raw),
        'embedded_files': inventory, 'base_kernel': BASE_KERNEL, 'expected_base_kernel_version': 1,
        'expected_base_kernel_source_sha256': BASE_KERNEL_SOURCE_SHA,
        'base_mount_files': len(inventory_base), 'base_mount_bytes': sum(v['bytes'] for v in inventory_base.values()),
        'base_delivery': 'anonymous_checked_public_archive' if public else 'read_only_owner_kernel_output',
        'public_base_descriptor_sha256': digest(base_descriptor.read_bytes()) if base_descriptor else None,
        'core_source_inventory': core, 'code_file_sha256': digest(outer.encode()),
        'source_or_model_or_data_upload': ('Only allowlisted source code and a public download descriptor; no model or dataset upload'
            if public else 'Only allowlisted source code; models are attached read-only from the original owner kernel'),
        'parent_imports_tensorflow': False, 'two_gpu_parallel_workers_or_one_gpu_fallback': True,
        'additional_grid_runs': 21, 'diagnostic_runs': 3, 'kernel_or_dataset_created_by_packaging': False}
    (output/'package-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run', type=Path)
    parser.add_argument('--base-descriptor', type=Path)
    parser.add_argument('--kernel-id', default='dongmyungpark/kd-gtsrb-full-grid')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    args = parser.parse_args()
    value = package(args.base_run, args.output, args.dataset_root,
                    kernel_id=args.kernel_id, base_descriptor=args.base_descriptor)
    print(json.dumps({name: value[name] for name in ('kernel_id','source_archive_sha256','source_archive_bytes',
        'base_mount_files','base_mount_bytes','code_file_sha256')}))


if __name__ == '__main__':
    main()
