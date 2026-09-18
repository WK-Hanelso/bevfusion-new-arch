# Confidentiality Audit

## Scope

검사 대상은 `portfolio/` 아래 Markdown, JSON, SVG, PNG, Python 생성 스크립트와 `agent/PORT_EXEC.md`다. 자산 재생성 후 2026-09-19에 재검사했다.

## Checks

| 검사 항목 | 검사 방법 | 결과 | 조치 |
|---|---|---|---|
| 사용자·workspace 절대 경로 | `/home/`, `/Users/`, `/workspace/`, `/mnt/`, `/data/` 패턴 검색 | 발견 없음 | 없음 |
| private/local IP | RFC1918 IPv4 패턴 검색 | 발견 없음 | 없음 |
| email, access key, private key, credential assignment | credential·email 정규식 검색 | 발견 없음 | 없음 |
| URL / endpoint | `https?://` 검색 | SVG 표준 namespace와 Matplotlib metadata의 공개 URL만 발견 | 공개 표준 URL로 확인 |
| PNG embedded strings | PNG `strings` 출력을 절대 경로·private IP·credential 패턴으로 재검색 | 발견 없음 | 없음 |
| 시각 자산 언어·민감 텍스트 | 생성 SVG와 생성 스크립트에서 한글 및 내부 endpoint/path 문자열 검색, 화면 수동 검토 | 모든 자산 본문은 영어; 내부 경로·서버명 없음 | 없음 |
| Source boundary | 발췌 전용 claim의 source·reproducible 필드와 로컬 primary source 유지 여부 대조 | C05–C07, C14–C16, C18은 지정된 author-published source / `false`; C03–C04, C08–C11, C17은 로컬 source 유지 | 없음 |
| 자산 크기·가독성 | PNG dimensions, SVG physical width, 렌더링 화면 검토 | PNG 1415–1800 px; 신규 SVG 18 in(1296 pt, 약 1728 CSS px); 흰 배경·읽을 수 있는 글자 | 없음 |
| JSON·생성·format 무결성 | `python3 -m json.tool`, claim/source invariant 검사, 자산 생성 스크립트 실행, Markdown·JSON·Python trailing-whitespace 검색 | 통과 | 없음 |

## Public-safety review

- 서버 hostname, GPU 서버 별칭, 사용자명, 이메일, IP, token, secret, private endpoint를 포함하지 않았다.
- 시각 자산에는 내부 코드 경로나 source artifact 경로를 넣지 않았다.
- `EVIDENCE.md`와 `evidence.json`의 source는 repo-relative 경로 또는 사용자가 지정한 author-published source 설명만 사용한다.
- 저자 공개 페이지에만 있는 수치는 raw training log와 공식 evaluation output이 이 머신에 없다는 제한을 함께 기록했고 모두 non-reproducible로 분류했다.
- 배포 수치는 legacy layout, random-init engine, synthetic input이라는 제한을 유지했다.
- 문서와 그림은 production, enterprise, large-scale 적용을 주장하지 않는다.

## Result

**PASS**
