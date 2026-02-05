import streamlit as st

# ✅ Streamlit은 set_page_config가 "가장 먼저" 실행되는 게 안전합니다.
st.set_page_config(page_title="농구 자동 팀 편성기", layout="wide")

import csv
import json
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
from datetime import datetime
from zoneinfo import ZoneInfo
from itertools import combinations
from io import StringIO
from urllib.parse import urlparse, parse_qs

import requests


# =========================
# 0) 기본 설정
# =========================
TIER_TO_SCORE = {
    "상위": 9,
    "중상": 8,
    "중위": 6,
    "중하": 5,
    "하위": 4,
}

HISTORY_FILE = "history.json"
KST = ZoneInfo("Asia/Seoul")


# =========================
# 1) 데이터 모델
# =========================
@dataclass
class Player:
    name: str
    height: int
    main_pos: str
    sub_pos: Optional[str]
    skill_tier: str
    skill: int

    def can_play(self, pos: str) -> bool:
        return self.main_pos == pos or (self.sub_pos == pos)

    def pos_tags(self) -> str:
        return f"{self.main_pos}/{self.sub_pos}" if self.sub_pos else self.main_pos


# =========================
# 2) 문자열 정리/정규화
# =========================
def safe_int(x: str) -> int:
    x = (x or "").strip()
    if not x:
        return 0
    try:
        return int(float(x))
    except:
        return 0

def norm_pos(x: str) -> str:
    x = (x or "").strip().upper()
    if x in ["C", "F", "G"]:
        return x
    if x in ["센터", "CENTER"]:
        return "C"
    if x in ["포워드", "FORWARD"]:
        return "F"
    if x in ["가드", "GUARD"]:
        return "G"
    if "/" in x:
        a = x.split("/")[0].strip().upper()
        if a in ["C", "F", "G"]:
            return a
    return x

def norm_tier(x: str) -> str:
    return (x or "").strip()

def try_fix_mojibake(s: str) -> str:
    if not s:
        return s
    if any(ch in s for ch in ["ì", "ë", "ê", "â", "Ã", "¤", "§"]):
        try:
            return s.encode("latin1").decode("utf-8")
        except:
            return s
    return s


# =========================
# 3) 링크 자동 변환 (edit 링크 → export csv 링크)
# =========================
def to_export_csv_url(sheet_url: str) -> str:
    sheet_url = (sheet_url or "").strip()
    if not sheet_url:
        return ""
    if "export?format=csv" in sheet_url:
        return sheet_url
    if "gviz/tq" in sheet_url and "out:csv" in sheet_url:
        return sheet_url

    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not m:
        return sheet_url

    sheet_id = m.group(1)

    gid = "0"
    if "#gid=" in sheet_url:
        gid = sheet_url.split("#gid=")[-1].split("&")[0].strip() or "0"
    else:
        parsed = urlparse(sheet_url)
        qs = parse_qs(parsed.query)
        if "gid" in qs and qs["gid"]:
            gid = qs["gid"][0]

    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


# =========================
# 4) Google Sheets CSV 로드 (헤더 영어/한글 모두 지원 + 인코딩 방지)
# =========================
HEADER_ALIASES = {
    "name": ["name", "이름", "성명", "닉네임"],
    "height": ["height", "키", "신장"],
    "main_pos": ["main_pos", "main", "주포", "주포지션", "정포지션", "포지션"],
    "sub_pos": ["sub_pos", "sub", "부포", "부포지션", "부포지", "서브포지션"],
    "skill_tier": ["skill_tier", "tier", "티어", "실력", "실력티어", "랭크"],
}

def unify_header(fieldnames: List[str]) -> List[str]:
    if not fieldnames:
        return fieldnames
    cleaned = [h.strip().replace("\ufeff", "") for h in fieldnames]
    mapping = {}
    for standard_key, aliases in HEADER_ALIASES.items():
        for h in cleaned:
            if h in aliases:
                mapping[h] = standard_key
    return [mapping.get(h, h) for h in cleaned]

def row_get(row: dict, standard_key: str) -> str:
    if standard_key in row and row.get(standard_key) is not None:
        return str(row.get(standard_key))
    for alias in HEADER_ALIASES.get(standard_key, []):
        if alias in row and row.get(alias) is not None:
            return str(row.get(alias))
    return ""

@st.cache_data(show_spinner=False, ttl=60)
def fetch_players_from_google_sheet(sheet_link_any: str) -> List[Player]:
    if not sheet_link_any or not sheet_link_any.strip():
        raise ValueError("Google Sheets 링크가 비어있습니다.")

    csv_url = to_export_csv_url(sheet_link_any.strip())
    r = requests.get(csv_url, timeout=15)
    if r.status_code != 200:
        raise ValueError(f"CSV 링크 요청 실패: HTTP {r.status_code}\n링크: {csv_url}")

    # ✅ HTML이 오면(권한/잘못된 링크) 즉시 에러로 안내
    content_type = (r.headers.get("Content-Type", "") or "").lower()
    if "text/html" in content_type:
        raise ValueError(
            "구글시트 CSV를 불러오지 못했습니다.\n"
            "원인: 시트가 비공개이거나 CSV 링크가 아닌 페이지로 연결됩니다.\n\n"
            "해결:\n"
            "1) 시트 공유 권한을 '링크가 있는 모든 사용자: 뷰어'로 변경\n"
            "2) 또는 파일 → 웹에 게시(Publish to web) → CSV 링크 사용"
        )

    text = r.content.decode("utf-8-sig", errors="replace")
    f = StringIO(text)
    reader = csv.DictReader(f)

    if reader.fieldnames:
        reader.fieldnames = unify_header(reader.fieldnames)

    required_cols = {"name", "height", "main_pos", "sub_pos", "skill_tier"}
    missing = required_cols - set(reader.fieldnames or [])
    if missing:
        raise ValueError(
            "시트 컬럼이 맞지 않습니다.\n"
            f"누락: {missing}\n\n"
            "허용 헤더 예시:\n"
            "- 영어: name,height,main_pos,sub_pos,skill_tier\n"
            "- 한글: 이름,키,주포지션,부포지션,티어"
        )

    players: List[Player] = []
    for row in reader:
        name = try_fix_mojibake((row_get(row, "name") or "").strip())
        if not name:
            continue

        height_raw = try_fix_mojibake(row_get(row, "height"))
        height = safe_int(height_raw)

        main_pos = norm_pos(try_fix_mojibake(row_get(row, "main_pos")))
        sub_pos_tmp = norm_pos(try_fix_mojibake(row_get(row, "sub_pos")))
        sub_pos = sub_pos_tmp if sub_pos_tmp in ["C", "F", "G"] else None

        tier = norm_tier(try_fix_mojibake(row_get(row, "skill_tier")))
        if tier not in TIER_TO_SCORE:
            raise ValueError(
                f"{name}의 skill_tier가 잘못되었습니다: '{tier}'\n"
                f"(허용: {list(TIER_TO_SCORE.keys())})"
            )

        players.append(
            Player(
                name=name,
                height=height,
                main_pos=main_pos,
                sub_pos=sub_pos,
                skill_tier=tier,
                skill=TIER_TO_SCORE[tier],
            )
        )

    if not players:
        raise ValueError("시트에서 선수 데이터를 읽지 못했습니다. (빈 시트/헤더 오류 가능)")

    # 동명이인 처리
    seen = {}
    for p in players:
        if p.name not in seen:
            seen[p.name] = 1
        else:
            seen[p.name] += 1
            p.name = f"{p.name}({seen[p.name]})"

    return players


# =========================
# 5) 히스토리 (세션 + 파일)
# =========================
def load_history_from_file() -> List[dict]:
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_history_to_file(history: List[dict]) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except:
        pass

def build_pairs_from_names(names: List[str]) -> Set[Tuple[str, str]]:
    names = sorted(names)
    return set(tuple(sorted(pair)) for pair in combinations(names, 2))

def history_pairs_from_last_n(history: List[dict], n: int) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for game in history[-n:]:
        teams = game.get("teams", [])
        for team_names in teams:
            pairs |= build_pairs_from_names(team_names)
    return pairs

def history_teamsets_from_last_n(history: List[dict], n: int) -> Set[frozenset]:
    s: Set[frozenset] = set()
    for game in history[-n:]:
        teams = game.get("teams", [])
        for team_names in teams:
            s.add(frozenset(team_names))
    return s


# =========================
# 6) 밸런스 평가
# =========================
def team_skill(team: List[Player]) -> int:
    return sum(p.skill for p in team)

def count_main_pos(team: List[Player], pos: str) -> int:
    return sum(1 for p in team if p.main_pos == pos)

def count_capable(team: List[Player], pos: str) -> int:
    return sum(1 for p in team if p.can_play(pos))

def avg_height(team: List[Player]) -> float:
    hs = [p.height for p in team if p.height > 0]
    return sum(hs) / len(hs) if hs else 0.0

def height_gap_score(teams: List[List[Player]]) -> float:
    avgs = [avg_height(t) for t in teams]
    known = [x for x in avgs if x > 0]
    if len(known) < 2:
        return 0.0
    return max(known) - min(known)

def all_pairs_now(teams: List[List[Player]]) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for t in teams:
        pairs |= build_pairs_from_names([p.name for p in t])
    return pairs

def repeat_pairs_count(teams: List[List[Player]], recent_pairs: Set[Tuple[str, str]]) -> int:
    return len(all_pairs_now(teams) & recent_pairs)

def teamset_repeat_count(teams: List[List[Player]], recent_teamsets: Set[frozenset]) -> int:
    cnt = 0
    for t in teams:
        if frozenset([p.name for p in t]) in recent_teamsets:
            cnt += 1
    return cnt

def overall_score(
    teams: List[List[Player]],
    recent_pairs: Optional[Set[Tuple[str, str]]],
    recent_teamsets: Optional[Set[frozenset]],
    repeat_weight: float,
    teamset_repeat_weight: float,
    center_min_weight: float,
    use_height_balance: bool,
    height_weight: float,
    variety_jitter: float,
) -> float:
    skills = [team_skill(t) for t in teams]
    skill_gap = max(skills) - min(skills)

    avgC = sum(count_main_pos(t, "C") for t in teams) / len(teams)
    avgF = sum(count_main_pos(t, "F") for t in teams) / len(teams)
    avgG = sum(count_main_pos(t, "G") for t in teams) / len(teams)

    pos_gap = 0.0
    for t in teams:
        pos_gap += abs(count_main_pos(t, "C") - avgC)
        pos_gap += abs(count_main_pos(t, "F") - avgF)
        pos_gap += abs(count_main_pos(t, "G") - avgG)

    center_short_cnt = sum(1 for t in teams if count_capable(t, "C") == 0)
    center_penalty = center_short_cnt * center_min_weight

    rep_pen = 0.0
    if recent_pairs:
        rep_pen = repeat_pairs_count(teams, recent_pairs) * repeat_weight

    teamset_pen = 0.0
    if recent_teamsets:
        teamset_pen = teamset_repeat_count(teams, recent_teamsets) * teamset_repeat_weight

    h_pen = 0.0
    if use_height_balance:
        h_pen = height_gap_score(teams) * height_weight

    base = (skill_gap * 3.0) + (pos_gap * 2.0) + center_penalty + rep_pen + teamset_pen + h_pen
    if variety_jitter > 0:
        base += random.random() * variety_jitter
    return base


# =========================
# 7) 팀 생성 알고리즘
# =========================
def snake_distribute(players_sorted: List[Player], team_count: int) -> List[List[Player]]:
    teams = [[] for _ in range(team_count)]
    direction = 1
    idx = 0
    for p in players_sorted:
        teams[idx].append(p)
        nxt = idx + direction
        if nxt >= team_count:
            direction = -1
            idx = team_count - 1
        elif nxt < 0:
            direction = 1
            idx = 0
        else:
            idx = nxt
    return teams

def shuffle_within_tiers(players_sorted: List[Player], tier_size: int = 2) -> List[Player]:
    result: List[Player] = []
    for i in range(0, len(players_sorted), tier_size):
        chunk = players_sorted[i:i + tier_size]
        random.shuffle(chunk)
        result.extend(chunk)
    return result

def improve_by_swaps_multi(
    teams: List[List[Player]],
    max_swaps: int,
    recent_pairs: Optional[Set[Tuple[str, str]]],
    recent_teamsets: Optional[Set[frozenset]],
    repeat_weight: float,
    teamset_repeat_weight: float,
    center_min_weight: float,
    use_height_balance: bool,
    height_weight: float,
    variety_jitter: float,
) -> List[List[Player]]:
    best = [t[:] for t in teams]
    best_score = overall_score(best, recent_pairs, recent_teamsets, repeat_weight, teamset_repeat_weight,
                               center_min_weight, use_height_balance, height_weight, variety_jitter)

    for _ in range(max_swaps):
        improved = False
        for i in range(len(best)):
            for j in range(i + 1, len(best)):
                for a_idx in range(len(best[i])):
                    for b_idx in range(len(best[j])):
                        new = [t[:] for t in best]
                        new[i][a_idx], new[j][b_idx] = new[j][b_idx], new[i][a_idx]
                        s = overall_score(new, recent_pairs, recent_teamsets, repeat_weight, teamset_repeat_weight,
                                          center_min_weight, use_height_balance, height_weight, variety_jitter)
                        if s < best_score:
                            best = new
                            best_score = s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best

def make_teams_once(
    players: List[Player],
    team_count: int,
    mode: str,
    seed: Optional[int],
    recent_pairs: Optional[Set[Tuple[str, str]]],
    recent_teamsets: Optional[Set[frozenset]],
    repeat_weight: float,
    teamset_repeat_weight: float,
    center_min_weight: float,
    use_height_balance: bool,
    height_weight: float,
    variety_jitter: float,
) -> List[List[Player]]:
    if seed is not None:
        random.seed(seed)

    players_sorted = sorted(players, key=lambda x: x.skill, reverse=True)

    if mode == "stable":
        teams = snake_distribute(players_sorted, team_count)
    elif mode == "variety":
        tiered = shuffle_within_tiers(players_sorted, tier_size=2)
        teams = snake_distribute(tiered, team_count)
    else:  # chaos
        pool = players[:]
        random.shuffle(pool)
        teams = [[] for _ in range(team_count)]
        for i, p in enumerate(pool):
            teams[i % team_count].append(p)

    teams = improve_by_swaps_multi(
        teams,
        max_swaps=90 if mode != "chaos" else 25,
        recent_pairs=recent_pairs,
        recent_teamsets=recent_teamsets,
        repeat_weight=repeat_weight,
        teamset_repeat_weight=teamset_repeat_weight,
        center_min_weight=center_min_weight,
        use_height_balance=use_height_balance,
        height_weight=height_weight,
        variety_jitter=variety_jitter,
    )
    return teams

def make_teams_search(
    players: List[Player],
    team_count: int,
    mode: str,
    fixed_seed: Optional[int],
    trials: int,
    top_k_pick: int,
    recent_pairs: Optional[Set[Tuple[str, str]]],
    recent_teamsets: Optional[Set[frozenset]],
    repeat_weight: float,
    teamset_repeat_weight: float,
    center_min_weight: float,
    use_height_balance: bool,
    height_weight: float,
    variety_jitter: float,
) -> Tuple[List[List[Player]], float]:
    candidates: List[Tuple[float, List[List[Player]]]] = []
    for t in range(trials):
        seed = fixed_seed if fixed_seed is not None else random.randrange(1, 10**9)
        seed = seed + t * 99991
        teams = make_teams_once(
            players=players,
            team_count=team_count,
            mode=mode,
            seed=seed,
            recent_pairs=recent_pairs,
            recent_teamsets=recent_teamsets,
            repeat_weight=repeat_weight,
            teamset_repeat_weight=teamset_repeat_weight,
            center_min_weight=center_min_weight,
            use_height_balance=use_height_balance,
            height_weight=height_weight,
            variety_jitter=variety_jitter,
        )
        s = overall_score(
            teams, recent_pairs, recent_teamsets, repeat_weight, teamset_repeat_weight,
            center_min_weight, use_height_balance, height_weight, variety_jitter
        )
        candidates.append((s, teams))

    candidates.sort(key=lambda x: x[0])
    top = candidates[: max(1, top_k_pick)]
    chosen = random.choice(top)
    return chosen[1], chosen[0]


# =========================
# 8) 역할 배정(센터 부족 시 부포 + 키 기반)
# =========================
def assign_roles(team: List[Player]) -> Dict[str, str]:
    roles = {p.name: p.main_pos for p in team}
    center_candidates = [p for p in team if p.can_play("C")]
    main_centers = [p for p in center_candidates if p.main_pos == "C"]

    if main_centers:
        chosen = sorted(main_centers, key=lambda x: (x.skill, x.height), reverse=True)[0]
        roles[chosen.name] = "C"
    elif center_candidates:
        chosen = sorted(center_candidates, key=lambda x: (x.height, x.skill), reverse=True)[0]
        roles[chosen.name] = "C"
    return roles


# =========================
# 9) 후보(벤치) 처리
# =========================
def choose_bench_players(players: List[Player], bench_count: int, seed: Optional[int]) -> List[Player]:
    if bench_count <= 0:
        return []
    if seed is not None:
        random.seed(seed)
    return random.sample(players, bench_count)


# =========================
# 10) 카톡 공유 텍스트
# =========================
def kakao_text_multi(
    teams: List[List[Player]],
    roles_by_team: List[Dict[str, str]],
    benches: List[Player],
    bench_assign: List[int],
) -> str:
    lines = []
    for idx, team in enumerate(teams):
        lines.append(f"[TEAM {idx+1}]")
        roles = roles_by_team[idx]
        parts = [f"{p.name}({roles.get(p.name, p.main_pos)})" for p in team]
        lines.append("  ".join(parts))
        lines.append("")
    if benches:
        lines.append("[후보]")
        for b, t_idx in zip(benches, bench_assign):
            lines.append(f"- TEAM {t_idx+1} 후보: {b.name}({b.main_pos})")
    return "\n".join(lines).strip()


# =========================
# 11) Streamlit UI
# =========================
st.markdown(
    """
    <style>
    div.stButton>button {
        padding: 0.9rem 1rem;
        font-size: 1.05rem;
        border-radius: 14px;
    }
    .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("농구 자동 팀 편성기")
st.caption("Google Sheets 연동 · 모바일 최적화 · 체크리스트 참석자 선택 · 후보 자동 처리 · 반복 방지")


# 11-1) 링크 입력
with st.expander("📌 선수 명단 불러오기 (Google Sheets 링크)", expanded=True):
    sheet_any_link = st.text_input(
        "구글시트 링크를 붙여넣으세요 (edit 링크도 OK)",
        placeholder="예: https://docs.google.com/spreadsheets/d/.../edit?usp=sharing",
        key="sheet_link",
    )
    st.caption("팁: edit 링크를 넣어도 자동으로 CSV(export) 링크로 변환됩니다.")

if not sheet_any_link.strip():
    st.warning("먼저 구글시트 링크를 입력하세요.")
    st.stop()

# 11-2) 선수 로드
try:
    all_players = fetch_players_from_google_sheet(sheet_any_link)
except Exception as e:
    st.error(f"선수 명단 불러오기 실패: {e}")
    st.stop()

# 히스토리 세션/파일
if "history" not in st.session_state:
    st.session_state.history = load_history_from_file()
history = st.session_state.history


# 11-3) 빠른 설정
st.subheader("빠른 설정")

c1, c2 = st.columns(2)
with c1:
    team_count = st.number_input("팀 개수", min_value=2, max_value=5, value=2, step=1)
    mode = st.selectbox("모드", ["stable", "variety", "chaos"], index=1)
with c2:
    use_height_balance = st.checkbox("키 밸런스", value=True)
    use_repeat_guard = st.checkbox("반복 방지", value=True)

with st.expander("고급 옵션(필요할 때만)"):
    trials = st.slider("탐색 횟수(다양성)", 5, 200, 50, 5)
    top_k_pick = st.slider("상위 후보 중 랜덤 선택(K)", 1, 20, 6, 1)

    n_history = st.slider("최근 몇 회 기록 참고", 1, 10, 1, 1)
    repeat_weight = st.slider("페어 반복 방지 강도", 0.0, 30.0, 10.0, 0.5)
    teamset_repeat_weight = st.slider("팀 전체 반복 방지 강도", 0.0, 100.0, 40.0, 1.0)

    center_min_weight = st.slider("센터 부족 패널티", 0.0, 40.0, 18.0, 1.0)
    height_weight = st.slider("키 밸런스 강도", 0.0, 2.0, 0.25, 0.05)
    variety_jitter = st.slider("variety 흔들림(jitter)", 0.0, 1.0, 0.15, 0.05)

    use_seed = st.checkbox("결과 고정(Seed)", value=False)
    seed = st.number_input("Seed 값", 0, 999999, 42, 1) if use_seed else None

# expander 안 열었을 때 기본
if "trials" not in locals():
    trials = 50
    top_k_pick = 6
    n_history = 1
    repeat_weight = 10.0
    teamset_repeat_weight = 40.0
    center_min_weight = 18.0
    height_weight = 0.25
    variety_jitter = 0.15
    seed = None


# 11-4) 선수 목록 표시(옵션)
with st.expander("선수 전체 목록 보기"):
    st.dataframe(
        [{
            "이름": p.name,
            "키": p.height if p.height > 0 else "",
            "주포": p.main_pos,
            "부포": p.sub_pos if p.sub_pos else "",
            "티어": p.skill_tier,
        } for p in all_players],
        use_container_width=True,
        hide_index=True
    )


# =========================
# ✅ 11-5) 참석자 선택 UI (체크리스트 방식으로 개선)
# =========================
st.divider()
st.subheader("오늘 참석자 체크")

# 최초 1회: attend_map 초기화 (모든 선수 True로 시작)
if "attend_map" not in st.session_state:
    st.session_state.attend_map = {p.name: True for p in all_players}

# 선수 명단이 변경되었을 때(구글시트 업데이트) 키 동기화
current_names = [p.name for p in all_players]
for n in current_names:
    if n not in st.session_state.attend_map:
        st.session_state.attend_map[n] = True
# 삭제된 선수는 맵에서 제거
for n in list(st.session_state.attend_map.keys()):
    if n not in current_names:
        del st.session_state.attend_map[n]

# 검색
filter_query = st.text_input("🔎 이름 검색(필터)", value="", key="attend_search").strip()

def is_visible(name: str) -> bool:
    return (filter_query in name) if filter_query else True

# 빠른 버튼
b1, b2, b3, b4 = st.columns(4)
with b1:
    if st.button("✅ 전체 선택", use_container_width=True):
        for n in current_names:
            st.session_state.attend_map[n] = True
with b2:
    if st.button("🧹 전체 해제", use_container_width=True):
        for n in current_names:
            st.session_state.attend_map[n] = False
with b3:
    if st.button("🔎 필터만 선택", use_container_width=True):
        for n in current_names:
            if is_visible(n):
                st.session_state.attend_map[n] = True
with b4:
    if st.button("🚫 필터만 해제", use_container_width=True):
        for n in current_names:
            if is_visible(n):
                st.session_state.attend_map[n] = False

# 체크리스트 렌더링 (모바일 고려: 2열)
col_left, col_right = st.columns(2)

visible_names = [n for n in current_names if is_visible(n)]
half = (len(visible_names) + 1) // 2
left_names = visible_names[:half]
right_names = visible_names[half:]

def render_checks(target_col, names_list: List[str]):
    with target_col:
        for n in names_list:
            st.checkbox(
                n,
                value=st.session_state.attend_map.get(n, False),
                key=f"chk_{n}",
                on_change=lambda name=n: st.session_state.attend_map.__setitem__(name, st.session_state[f"chk_{name}"]),
            )

render_checks(col_left, left_names)
render_checks(col_right, right_names)

# 오늘 참석자 리스트
today_names = [n for n, v in st.session_state.attend_map.items() if v]
today_players = [p for p in all_players if p.name in set(today_names)]
N = len(today_players)

st.info(f"현재 체크 인원: **{N}명**")
if N < 2:
    st.warning("팀을 만들려면 최소 2명 이상 필요합니다.")
    st.stop()

# 반복 방지 준비
recent_pairs = set()
recent_teamsets = set()
if use_repeat_guard and len(history) > 0:
    recent_pairs = history_pairs_from_last_n(history, n_history)
    recent_teamsets = history_teamsets_from_last_n(history, n_history)


# =========================
# 11-6) 팀 생성
# =========================
st.divider()
generate = st.button("🏀 팀 생성하기", type="primary", use_container_width=True)

if generate:
    team_count_int = int(team_count)

    # 남는 인원은 후보로 분리 (팀당 동일 인원)
    base_size = N // team_count_int
    target_total = base_size * team_count_int
    bench_count = N - target_total

    benches = []
    working_players = today_players[:]

    if bench_count > 0:
        benches = choose_bench_players(working_players, bench_count, seed=(int(seed) if seed is not None else None))
        bench_names = {b.name for b in benches}
        working_players = [p for p in working_players if p.name not in bench_names]

    teams, final_score = make_teams_search(
        players=working_players,
        team_count=team_count_int,
        mode=mode,
        fixed_seed=(int(seed) if seed is not None else None),
        trials=int(trials),
        top_k_pick=int(top_k_pick),
        recent_pairs=(recent_pairs if use_repeat_guard else None),
        recent_teamsets=(recent_teamsets if use_repeat_guard else None),
        repeat_weight=float(repeat_weight),
        teamset_repeat_weight=float(teamset_repeat_weight),
        center_min_weight=float(center_min_weight),
        use_height_balance=bool(use_height_balance),
        height_weight=float(height_weight),
        variety_jitter=(float(variety_jitter) if mode == "variety" else 0.0),
    )

    roles_by_team = [assign_roles(t) for t in teams]

    bench_assign: List[int] = []
    if benches:
        if seed is not None:
            random.seed(int(seed))
        for _ in benches:
            bench_assign.append(random.randrange(0, team_count_int))

    st.subheader("팀 편성 결과")
    st.write(f"- 팀 개수: **{team_count_int}팀**")
    st.write(f"- 팀당 인원: **{base_size}명**")
    if benches:
        st.write(f"- 후보(벤치): **{len(benches)}명**")
    st.write(f"- 점수(낮을수록 밸런스 좋음): **{final_score:.2f}**")

    cols = st.columns(team_count_int)

    def team_box(team: List[Player], title: str, roles: Dict[str, str]):
        st.markdown(f"### {title}")
        st.write(f"총 실력합: **{team_skill(team)}** · 센터 가능: **{count_capable(team,'C')}명**")
        if use_height_balance:
            ah = avg_height(team)
            st.write(f"평균 키: **{ah:.1f}cm**" if ah > 0 else "평균 키: (키 데이터 부족)")

        st.dataframe(
            [{
                "이름": p.name,
                "키": p.height if p.height > 0 else "",
                "주/부포": p.pos_tags(),
                "티어": p.skill_tier,
                "이번 역할": roles.get(p.name, p.main_pos),
            } for p in team],
            use_container_width=True,
            hide_index=True
        )

    for i in range(team_count_int):
        with cols[i]:
            team_box(teams[i], f"TEAM {i+1}", roles_by_team[i])

    if benches:
        st.divider()
        st.subheader("후보(벤치)")
        st.dataframe(
            [{
                "이름": b.name,
                "티어": b.skill_tier,
                "배정": f"TEAM {bench_assign[idx]+1} 후보",
            } for idx, b in enumerate(benches)],
            use_container_width=True,
            hide_index=True
        )

    st.divider()
    st.subheader("카톡 공유용 텍스트")
    share_text = kakao_text_multi(teams, roles_by_team, benches, bench_assign)
    st.text_area("복사해서 카톡에 붙여넣기", value=share_text, height=220)

    # 기록 저장 / 초기화
    st.divider()
    cA, cB = st.columns(2)

    with cA:
        if st.button("💾 이번 결과 기록 저장", use_container_width=True):
            now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
            history.append({
                "timestamp": now,
                "mode": mode,
                "team_count": team_count_int,
                "teams": [[p.name for p in t] for t in teams],
                "benches": [b.name for b in benches],
            })
            history = history[-50:]
            st.session_state.history = history
            save_history_to_file(history)
            st.success("저장 완료! 다음 생성부터 반복 방지에 반영됩니다.")

    with cB:
        if st.button("🗑️ 기록 초기화", use_container_width=True):
            st.session_state.history = []
            save_history_to_file([])
            st.success("기록을 초기화했습니다.")

