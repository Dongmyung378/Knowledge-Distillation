"""Create language derivatives of published SVGs without rerunning experiments.

Run from any directory with Python's standard library. Add --png to rasterize the
five Korean chart derivatives with a locally installed Chrome or Chromium.
English source charts and Korean portfolio originals are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
CHARTS = ['knowledge-distillation-release', 'four-candidate-release', 'full-grid-comparison', 'full-grid-release', 'teacher-student-analysis']
KO = {
 'Teacher':'교사', 'Baseline':'기준 학생', 'Distilled':'증류 학생', 'Accuracy (%)':'정확도 (%)',
 'Locked-holdout classification':'고정 홀드아웃 분류', 'Observed reference accuracy':'관측한 참조 정확도',
 'Training seed':'학습 시드', 'Student training seed':'학생 학습 시드',
 'p50 / p95 milliseconds (log scale)':'p50 / p95 밀리초 (로그 축)',
 'Teacher / Keras':'교사 / Keras', 'Baseline / Keras':'기준 학생 / Keras', 'Distilled / Keras':'증류 학생 / Keras',
 'Student / FP32 TFLite':'학생 / FP32 TFLite', 'Student / INT8 TFLite':'학생 / INT8 TFLite',
 'Warmed CPU inference':'워밍업 후 CPU 추론', 'Inference artifact MiB (log scale)':'추론 파일 MiB (로그 축)',
 'Actual model file sizes':'실제 모델 파일 크기', 'GTSRB class ID (all 43 classes)':'GTSRB 클래스 ID (전체 43개)',
 'Recall (%)':'재현율 (%)', 'Per-class recall on the locked holdout (float checkpoints)':'고정 홀드아웃의 클래스별 재현율 (실수형 체크포인트)',
 'Per-class recall on the observed reference set (float checkpoints)':'관측한 참조 집합의 클래스별 재현율 (실수형 체크포인트)',
 'Baseline, seed 378':'기준 학생, 시드 378', 'Selected distilled float checkpoint, seed 378':'선택한 증류 실수형 체크포인트, 시드 378',
 'Representative T5-a05 checkpoint, seed 378':'대표 T5-a05 체크포인트, 시드 378',
 'Traffic sign recognition: measured release evidence':'교통 표지판 인식: 측정한 배포 근거',
 'Traffic sign recognition: retained T5-a05 deployment':'교통 표지판 인식: 유지한 T5-a05 배포 모델',
 '5,850 locked-holdout images  |  34,406-parameter baseline  |  Windows AMD64, 1 CPU thread(s)':'고정 홀드아웃 5,850장  |  기준 학생 파라미터 34,406개  |  Windows AMD64, CPU 스레드 1개',
 '5,850 previously observed reference images  |  34,406-parameter baseline  |  Windows AMD64, 1 CPU thread(s)':'기존에 관측한 참조 이미지 5,850장  |  기준 학생 파라미터 34,406개  |  Windows AMD64, CPU 스레드 1개',
 'Quality bars: seed 378; F1 labels: macro F1. Points: 378, 379, 380; lines pair baseline and KD. No confidence intervals.':'성능 막대는 시드 378, F1 표시는 macro F1. 점은 378, 379, 380이며 선은 기준 학생과 KD의 쌍입니다. 신뢰구간이 아닙니다.',
 'CPU: batch 1, 128 identical validation images, 10 warmups, 3 repetitions; synchronized inference, excluding load and image decoding.':'CPU: 배치 1, 같은 검증 이미지 128장, 워밍업 10회, 반복 3회. 동기화한 추론이며 로딩과 이미지 디코딩은 제외합니다.',
 'Declared deployment: distilled / FP32. INT8 validation/CPU gate: rejected. Holdout excluded from current selection; the public data were historically exposed.':'선언한 배포: 증류 / FP32. INT8 검증과 CPU 기준은 탈락. 현재 선택에 홀드아웃을 쓰지 않았지만 공개 데이터는 과거 노출됐습니다.',
 'Retained representative: T5-a05 / seed 378 / FP32. Previously observed outcomes informed retention; the nine-condition T5-a01 study is preserved separately.':'유지한 대표: T5-a05 / 시드 378 / FP32. 관측된 결과를 고려한 유지 판단이며 9개 조건 T5-a01 실험은 별도로 보존합니다.',
 'alpha 0.1':'정답 비중 0.1', 'alpha 0.3':'정답 비중 0.3', 'alpha 0.5':'정답 비중 0.5',
 'Hard-label loss weight':'정답 손실 비중', 'Distillation temperature':'증류 온도',
 'Validation accuracy: selection data':'검증 정확도: 설정 선택 자료', 'Reference accuracy: previously reported holdout':'참조 정확도: 이미 보고한 홀드아웃',
 'Traffic sign distillation: complete temperature / alpha grid':'교통 표지판 증류: 온도와 정답 비중 전체 조합',
 '9 configurations | 3 matched development seeds each | identical 34,406-parameter students':'9개 설정 | 설정마다 대응하는 개발 시드 3개 | 파라미터 34,406개의 같은 학생 구조',
 'Purple border: validation-selected setting T5-a01. Every cell shows mean ± sample SD over seeds 378, 379 and 380.':'보라 테두리: 검증으로 선택한 T5-a01. 각 칸은 시드 378, 379, 380의 평균 ± 표본 표준편차입니다.',
 'All three seeds participate in development selection. Reference outcomes do not choose the configuration or deployment seed.':'세 시드는 모두 개발 단계의 선택에 참여합니다. 참조 결과로 설정이나 배포 시드를 선택하지 않습니다.',
 'The reference images had published results before this extension; these are follow-up comparisons, not an untouched test.':'참조 이미지의 결과는 이 실험 전에 공개됐습니다. 후속 비교이며 미관측 최종 평가가 아닙니다.',
 'Paired image count':'대응하는 이미지 수', 'Selected T5-a01: paired teacher errors':'선택한 T5-a01: 교사 오류의 대응 비교',
 'Teacher wrong, selected KD correct':'교사 오답, 선택 KD 정답', 'Teacher correct, selected KD wrong':'교사 정답, 선택 KD 오답',
 'Hard labels':'정답만 학습', 'Pure soft T5':'순수 증류 T5', 'Mixed T5':'혼합 증류 T5',
 'T5 objective controls: three student seeds':'T5 학습 목적 대조: 학생 시드 3개', 'Fixed teacher 83.54%':'고정 교사 83.54%',
 'Teacher and student: observed error differences':'교사와 학생: 관측한 오류 차이',
 '5,850 identical images | 195 physical-sign tracks | previously reported evaluation set':'동일 이미지 5,850장 | 실제 표지판 추적 집단 195개 | 이미 보고한 평가 집합',
 'The selected ID was supplied from validation selection; this figure does not rank models using final outcomes.':'선택 ID는 검증 단계의 선택 결과입니다. 이 그림은 최종 점수로 모델 순위를 정하지 않습니다.',
 'Means and error bars: three student seeds and sample SD. The teacher is one shared checkpoint, not three teacher-seed runs.':'평균과 오차 막대: 학생 시드 3개와 표본 표준편차. 교사는 공통 체크포인트 한 개이며 교사 시드 3회의 결과가 아닙니다.',
 'Pure-soft alpha=0 removes label CE from training; validation labels still select checkpoints and the learning-rate schedule.':'순수 증류 alpha=0은 학습의 정답 CE를 제거합니다. 검증 정답은 체크포인트와 학습률 일정 선택에 사용합니다.',
 'Alpha changes both CE and KL weights. Different realized LR paths/epochs limit causal attribution of these pipeline comparisons.':'정답 비중은 CE와 KL 가중치를 함께 바꿉니다. 실제 학습률 경로와 에포크가 달라 이 파이프라인 비교의 인과 해석은 제한됩니다.',
}
EN = {
 '지식 증류를 통한 모델 경량화':'Knowledge distillation for compact models',
 '큰 교사 모델의 확률 분포를 작은 학생에 전달합니다. 대표 학생의 참조 정확도는87.06%이며 파라미터가 약98.5% 적습니다.':'Teacher probability distributions supervise a compact student. The representative student has 87.06% reference accuracy and approximately 98.5% fewer parameters.',
 '교통 표지판 인식 모델 경량화':'Compact traffic-sign recognition', '교사 모델':'Teacher model', '대표 학생':'Representative student',
 '2,313,067개 파라미터':'2,313,067 parameters', '34,406개 파라미터':'34,406 parameters', '증류':'Distillation',
 '파라미터 약 98.5% 감소':'Approximately 98.5% fewer parameters', '대표 학생 T5-a05 / FP32':'Representative T5-a05 / FP32',
 '대표 배포와 보존한 실험 결과':'Representative deployment and retained study results',
 '같은5850개 참조 이미지에서 교사83.54%, 정답만 학습한 학생73.56%, 대표 증류 학생87.06%와 새 실험의 고정 학생85.69%를 비교합니다. 대표 배포와 연구 선택의 역할을 구분합니다.':'On the same 5850 reference images: teacher 83.54%, hard-label student 73.56%, representative distilled student 87.06%, and the study-selected student 85.69%. Deployment retention and research selection have distinct roles.',
 '대표 배포 모델과 확장 실험의 결과':'Representative deployment and extended study',
 '같은 참조 이미지 5,850장. 대표 배포는 T5-a05이며 9개 조건의 검증 선택은 T5-a01입니다.':'Same 5,850 reference images. Deployment: T5-a05. Validation selection among 9 conditions: T5-a01.',
 '정답만 학습한 학생':'Hard-label student', '같은 학생 구조 / 정답만 학습':'Same architecture / hard labels only',
 '대표 증류 학생':'Representative KD', 'T5-a05 / 대표 배포 유지':'T5-a05 / retained deployment',
 '새 실험의 고정 학생':'Fixed study student', 'T5-a01 / 실험 선택 보존':'T5-a01 / retained study selection', '정확도 (%)':'Accuracy (%)',
 '참조 결과는 이미 관측됐습니다. 기존 배포 유지는 후속 운영 결정이며 실험의 선택 기록은 바꾸지 않았습니다.':'Reference outcomes were already observed. Retention is an operational decision; study selection is unchanged.',
 '대표 모델의 구조와 파일 크기 및 CPU 비용':'Representative model capacity, file sizes and CPU costs',
 '기존 대표 T5-a05 모델의 파라미터, 파일 바이트와 동일CPU의 추론 지연 중앙값을 비교합니다. 새T5-a01의 측정값을 대신 사용하지 않습니다.':'Parameter counts, file bytes and same-CPU median inference latency of the retained T5-a05. Measurements of the new T5-a01 are not substituted.',
 '대표 모델의 경량화 효과와 CPU 실행 비용':'Representative model compression and CPU costs',
 '기존 대표 T5-a05의 측정. CPU 스레드 1개, 배치 1, 검증 이미지 128장과 반복 3회입니다.':'Retained T5-a05 measurements: 1 CPU thread, batch 1, 128 validation images, 3 repetitions.',
 '파라미터 수':'Parameters', '98.5% 감소':'98.5% fewer', '교사':'Teacher', '2,313,067 개':'2,313,067', '34,406 개':'34,406',
 '파일 크기':'File size', '98.1% 감소':'98.1% smaller', '교사 Keras':'Teacher Keras', '학생 Keras':'Student Keras',
 '학생 FP32 TFLite':'Student FP32 TFLite', 'CPU 추론 지연 중앙값':'Median CPU inference latency', 'Keras 3.68배':'Keras 3.68x',
 '파일 감소율은 같은 Keras 형식끼리 비교했습니다. FP32 TFLite는 별도 배포 형식입니다.':'File reduction compares the same Keras format. FP32 TFLite is a separate deployment format.',
 '한 Windows CPU 호스트의 순수 추론 측정이며 모델 로딩과 이미지 디코딩은 제외합니다. 다른 기기의 속도는 다릅니다.':'Pure inference on one Windows CPU host, excluding model loading and image decoding. Other devices differ.',
 '전체 9개 조건의 검증 및 참조 평가 반복 실험':'Repeated validation and reference evaluation of all 9 conditions',
 '온도 3, 5, 10과 정답 손실 비중 0.1, 0.3, 0.5를 각각 3개 시드로 학습했습니다. 검증 평균으로 T5-a01을 선택했으며 대표 배포는 기존 T5-a05를 유지합니다. 평균과 표본 표준편차는 독립 확인 신뢰구간이 아닙니다.':'Temperatures 3, 5, 10 and hard-label weights 0.1, 0.3, 0.5 each use 3 seeds. Validation selects T5-a01; deployment retains T5-a05. Means and sample deviations are not independent-confirmation intervals.',
 '9개 조건의 반복 실험 결과':'Repeated experiments across 9 conditions',
 '각 칸은 시드 378, 379, 380의 평균입니다. 아래의 ± 값은 표본 표준편차이며 신뢰구간이 아닙니다.':'Each cell averages seeds 378, 379, 380. The ± values are sample standard deviations, not confidence intervals.',
 '검증 정확도':'Validation accuracy', '평균 % / 표준편차 %p':'Mean % / sample SD pp', '참조 정확도':'Reference accuracy',
 'T5-a01: 검증 평균으로 선택':'T5-a01: validation-selected', 'T5-a05: 대표 배포 유지':'T5-a05: retained deployment',
 'T는 증류 온도이고 α는 정답 손실의 비중입니다. 조건 선택에는 참조 평가 결과를 사용하지 않았습니다.':'T is distillation temperature; α weights hard-label loss. Reference results did not select the setting.',
 '대표 배포 유지 결정은 이미 관측된 결과와 공개 모델의 연속성을 고려한 후속 결정입니다.':'Deployment retention is a later decision considering observed results and continuity of the public model.',
 '참조 평가: 같은 이미지 5,850장. 반복 개발 시드의 비교이며 독립적인 최종 확인 실험은 아닙니다.':'Reference set: the same 5,850 images. Repeated development seeds, not independent final confirmation.',
 '대표 학생의 교사 오류 수정과 순수 증류 대조':'Representative student error repairs and pure-distillation controls',
 '대표 T5-a05 학생은 교사 오답 473개를 수정하고 교사 정답 267개를 놓쳐 정답이 206개 늘었습니다. 순수 증류 대조도 교사를 넘었지만 각 원인의 인과 기여는 확정하지 않았습니다.':'Representative T5-a05 repairs 473 teacher errors and loses 267 teacher-correct predictions, a net gain of 206. Pure distillation also exceeds the teacher; causal contributions remain unconfirmed.',
 '대표 학생의 오류 수정과 학습 목적 비교':'Representative student errors and training objectives',
 '대표 T5-a05의 같은 참조 이미지 5,850장. 오류 수정과 새 오류를 함께 계산했습니다.':'Same 5,850 reference images for T5-a05. Repairs and new errors are counted together.',
 '교사 오답을 수정':'Teacher errors repaired', '교사 정답을 놓침':'Teacher correct lost', '정답의 순수 증가':'Net correct gain', '개':'cases',
 '정확도 차이: +3.52%p':'Accuracy difference: +3.52 pp', '같은 학생 구조의 학습 목적 비교':'Training objectives for the same student architecture',
 '각 목적의 3회 학습 평균과 표본 표준편차. 교사 정확도 83.54%는 고정 모델 한 개의 결과입니다.':'Mean and sample SD over 3 runs per objective. Teacher accuracy 83.54% comes from one fixed model.',
 '정답만 학습':'Hard labels only', '순수 증류':'Pure distillation', '대표 설정의 혼합 증류':'Representative mixed KD',
 '순수 증류도 3회 모두 교사를 넘었습니다. 정답 손실을 추가한 것만으로 현상을 설명할 수 없습니다.':'Pure distillation exceeds the teacher in all 3 runs; adding hard-label loss alone cannot explain this.',
 '교사의 부분 미세 조정과 고정 BN, 학생 전체 학습과 BN 갱신, 최적화 경로 차이는 설명 후보입니다.':'Possible explanations: partial teacher tuning with frozen BN, full student training with updated BN, and optimization.',
 '각 요인의 인과 기여는 확정하지 않았습니다. 압축이 정확도를 높이거나 학생이 항상 교사보다 낫다는 결론은 아닙니다.':'Causal contributions are unconfirmed. This does not show that compression raises accuracy or students always beat teachers.',
}

def translate(value, language):
    if language == 'en':
        if re.search('[가-힣]', value) and value not in EN:
            raise ValueError('Missing English translation: ' + value)
        return EN.get(value, value)
    if value in KO:
        return KO[value]
    value = re.sub(r'^Net ([+-]?\d+)$', r'순증 \1', value)
    value = re.sub(r'^SD (.+) pp$', r'표준편차 \1%p', value)
    value = re.sub(r' pp$', '%p', value)
    return value

def render(source, target, language):
    tree = ET.parse(source)
    root = tree.getroot()
    for element in root.iter():
        if element.tag.rsplit('}', 1)[-1] in {'text','title','desc'} and element.text:
            element.text = translate(element.text, language)
            if language == 'ko' and element.tag.rsplit('}', 1)[-1] == 'text' and re.search('[가-힣]', element.text):
                style = element.get('style', '')
                size = re.search(r'font-size:\s*([\d.]+)px', style)
                if size:
                    style = style[:size.start(1)] + str(float(size.group(1)) * .9) + style[size.end(1):]
                element.set('style', style + ";font-family:'Malgun Gothic','Noto Sans CJK KR',sans-serif")
            if language == 'ko' and element.tag.rsplit('}', 1)[-1] == 'text':
                # SVG font metrics differ from the original plotting backend.
                # Separate only stacked annotation baselines; never move data.
                value = element.text or ''
                offset = 0
                horizontal_offset = 0
                if source.stem.endswith('release'):
                    if re.fullmatch(r'[\d,]+/[\d,]+', value): offset = 3
                    if value.startswith('F1 '): offset = 4.5
                elif source.stem == 'full-grid-comparison' and value.startswith('± '):
                    offset = 7
                elif source.stem == 'teacher-student-analysis':
                    if re.fullmatch(r'\+[\d.]+%p', value) or value.startswith(('alpha=', '표준편차 ')): offset = 4
                    if value.startswith('동일 이미지 5,850장'): offset = 3
                    if value.startswith('순증 ') or re.fullmatch(r'\+[\d.]+%p', value): horizontal_offset = 20
                if offset or horizontal_offset:
                    element.set('transform', f'translate({horizontal_offset} {offset}) ' + element.get('transform', ''))
    style = ET.Element('{'+NS+'}style')
    style.text = "" if language == 'ko' else "text{font-family:'DejaVu Sans',Arial,sans-serif!important}"
    root.insert(0, style)
    root.set('lang', language)
    target.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target, encoding='utf-8', xml_declaration=True)
    # Normalize serialized indentation without changing SVG geometry or text.
    target.write_text(
        '\n'.join(line.rstrip() for line in target.read_text(encoding='utf-8').splitlines()) + '\n',
        encoding='utf-8', newline='\n')

def chrome_binary(explicit):
    candidates = [explicit, shutil.which('chromium'), shutil.which('google-chrome'),
      Path(os.environ.get('PROGRAMFILES', 'C:/Program Files'))/'Google/Chrome/Application/chrome.exe',
      Path(os.environ.get('LOCALAPPDATA', ''))/'Google/Chrome/Application/chrome.exe']
    return next((str(p) for p in candidates if p and Path(p).is_file()), None)

def rasterize(svg, png, size, chrome):
    with tempfile.TemporaryDirectory(prefix='kd-localized-') as folder:
        folder=Path(folder)
        page=folder/'figure.html'
        page.write_text('<!doctype html><style>html,body{margin:0;overflow:hidden}svg{display:block;width:100vw;height:100vh}</style>'+svg.read_text(encoding='utf-8'),encoding='utf-8')
        command=[chrome,'--headless=new','--disable-gpu','--hide-scrollbars','--no-first-run','--no-default-browser-check',
          '--disable-background-networking','--force-device-scale-factor=1',f'--user-data-dir={folder/"profile"}',
          f'--window-size={size[0]},{size[1]}','--virtual-time-budget=1500',f'--screenshot={png}',page.as_uri()]
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode or not png.is_file():
            raise RuntimeError('Chrome did not render '+str(png))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--png',action='store_true')
    parser.add_argument('--chrome')
    args=parser.parse_args()
    chrome=chrome_binary(args.chrome) if args.png else None
    if args.png and not chrome: raise RuntimeError('Install Chrome/Chromium or pass --chrome')
    records=[]
    pairs=[(ROOT/'assets/figures'/f'{name}.svg',ROOT/'assets/figures/ko'/f'{name}.svg','ko') for name in CHARTS]
    pairs += [(ROOT/'docs/assets/portfolio'/f'kd-{name}.svg',ROOT/'docs/assets/portfolio/en'/f'kd-{name}.svg','en') for name in ['thumbnail','results','costs','grid','errors']]
    for source,target,language in pairs:
        render(source,target,language)
        record={'source':source.relative_to(ROOT).as_posix(),'target':target.relative_to(ROOT).as_posix(),'language':language,
          'source_sha256':hashlib.sha256(source.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),'svg_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        if args.png and language=='ko':
            original=source.with_suffix('.png')
            size=struct.unpack('>II',original.read_bytes()[16:24])
            rasterize(target,target.with_suffix('.png'),size,chrome)
            record['png_sha256']=hashlib.sha256(target.with_suffix('.png').read_bytes()).hexdigest()
            record['png_size']=list(size)
        records.append(record)
    receipt={'scope':'Language-only derivatives of published figures; no experiment, model, data split or selection is changed. Original numeric shapes and palettes are retained.',
             'source_hash_policy':'SHA-256 of source SVG bytes with CRLF normalized to LF, matching published Git text. Source files are never rewritten.',
             'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':records}
    (ROOT/'docs/figure-localization.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'svg_derivatives':len(records),'png_derivatives':sum('png_sha256' in r for r in records)}))

if __name__=='__main__':main()
