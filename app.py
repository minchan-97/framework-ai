"""
GasCore Framework — Streamlit App
Layer 1: 진화 인덱스 | Layer 2: CoreAI 가드레일 | Layer 3: XAI
"""
import streamlit as st
import pickle, io, os, time
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="GasCore", page_icon="⚡", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Noto+Sans+KR:wght@300;400;700&display=swap');
:root{--bg:#0a0e1a;--sf:#111827;--bd:#1e3050;--ac:#00e5ff;--gn:#00ff88;--yw:#ffd600;--rd:#ff4444;--tx:#e2e8f0;--mt:#64748b;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--tx)!important;font-family:'Noto Sans KR',sans-serif;}
[data-testid="stSidebar"]{background:var(--sf)!important;border-right:1px solid var(--bd);}
.lb{display:inline-block;padding:2px 10px;border-radius:4px;font-family:'Space Mono',monospace;font-size:0.62rem;font-weight:700;letter-spacing:1px;margin-bottom:0.4rem;}
.l1{background:#1a2a1a;border:1px solid #00ff88;color:#00ff88;}
.l2{background:#1a1a2a;border:1px solid #00e5ff;color:#00e5ff;}
.l3{background:#2a1a1a;border:1px solid #ffd600;color:#ffd600;}
.card{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:1rem;margin-bottom:0.6rem;}
.pass{color:#00ff88;font-family:'Space Mono',monospace;font-weight:700;}
.warn{color:#ffd600;font-family:'Space Mono',monospace;font-weight:700;}
.fatal{color:#ff4444;font-family:'Space Mono',monospace;font-weight:700;}
.tp{background:#0d1f0d;color:#00ff88;padding:2px 6px;border-radius:3px;font-family:monospace;font-size:0.8rem;margin:2px;}
.tw{background:#1f1a0d;color:#ffd600;padding:2px 6px;border-radius:3px;font-family:monospace;font-size:0.8rem;margin:2px;}
.tf{background:#1f0d0d;color:#ff4444;padding:2px 6px;border-radius:3px;font-family:monospace;font-size:0.8rem;margin:2px;}
[data-testid="stButton"] button{font-weight:700!important;border-radius:6px!important;border:none!important;}
hr{border-color:var(--bd)!important;}
</style>""", unsafe_allow_html=True)

try:
    from gascore_engine import GasCoreFramework
    ENGINE_OK = True
except Exception as e:
    ENGINE_OK = False
    st.error(f"엔진 오류: {e}")
    st.stop()

# ── 세션 초기화 ───────────────────────────────────────────────
for k, v in {
    "fw":           None,
    "initialized":  False,
    "qa_history":   [],
    "logp_thr":     -11.5,
    "ev_thr":       -13.0,
    "max_retry":    3,
    "api_key":      os.getenv("OPENAI_API_KEY",""),
    "model":        "gpt-4o-mini",
    "corpus_bytes": None,
    "corpus_name":  "",
    "pkl_bytes":    None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# fw 헬퍼
def get_fw() -> GasCoreFramework:
    return st.session_state.fw

def set_fw(fw):
    st.session_state.fw = fw


# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ GasCore")
    st.markdown("---")

    st.session_state.api_key = st.text_input(
        "OpenAI API Key", value=st.session_state.api_key, type="password")
    st.session_state.model = st.selectbox(
        "모델", ["gpt-4o-mini","gpt-4o"],
        index=["gpt-4o-mini","gpt-4o"].index(st.session_state.model))

    st.markdown("---")
    st.markdown("### 📚 코퍼스 초기화")

    uploaded = st.file_uploader("코퍼스 업로드", type=["txt","pdf","docx"], key="corpus_up")
    if uploaded:
        st.session_state.corpus_bytes = uploaded.read()
        st.session_state.corpus_name  = uploaded.name

    c1,c2 = st.columns(2)
    with c1: epochs   = st.select_slider("학습", [5,10,15,20], value=10)
    with c2: som_grid = st.select_slider("SOM", [4,5,6,7], value=6)

    st.session_state.logp_thr  = st.slider("가드레일 임계값", -15.0,-5.0, st.session_state.logp_thr, 0.5)
    st.session_state.ev_thr    = st.slider("진화 임계값",    -16.0,-8.0, st.session_state.ev_thr,   0.5)
    st.session_state.max_retry = st.slider("최대 재생성",    1, 5,       st.session_state.max_retry)

    if st.session_state.corpus_bytes:
        st.caption(f"📄 {st.session_state.corpus_name}")
    if st.button("🚀 초기화", use_container_width=True,
                 disabled=not st.session_state.corpus_bytes):
        with st.spinner("초기화 중..."):
            try:
                raw  = st.session_state.corpus_bytes
                name = st.session_state.corpus_name.lower()
                if name.endswith(".pdf"):
                    import pypdf
                    text = "\n".join(
                        p.extract_text() or ""
                        for p in pypdf.PdfReader(io.BytesIO(raw)).pages)
                elif name.endswith(".docx"):
                    import docx
                    text = "\n".join(
                        p.text for p in docx.Document(io.BytesIO(raw)).paragraphs
                        if p.text.strip())
                else:
                    text = raw.decode("utf-8", errors="ignore")

                fw = GasCoreFramework()
                prog = st.progress(0)
                fw.initialize(text, epochs=epochs, som_grid=som_grid,
                              on_progress=lambda p,m: prog.progress(p))
                set_fw(fw)
                st.session_state.initialized = True
                st.session_state.qa_history  = []
                nm_ok = fw.nm and fw.nm.is_trained
                st.success(f"✓ 완료 | NM {'✅' if nm_ok else '❌'} {fw.nm.total if nm_ok else 0}토큰")
            except Exception as e:
                st.error(f"실패: {e}")
                import traceback; st.code(traceback.format_exc())
            else:
                st.rerun()

    # pkl 로드
    st.markdown("---")
    st.markdown("### 💾 저장된 인덱스")
    pkl_up = st.file_uploader("인덱스 (.pkl)", type=["pkl"], key="pkl_up")
    if pkl_up:
        st.session_state.pkl_bytes = pkl_up.read()

    if st.session_state.pkl_bytes and not st.session_state.initialized:
        if st.button("📂 불러오기", use_container_width=True):
            try:
                with st.spinner("로드 중..."):
                    data = pickle.loads(st.session_state.pkl_bytes)
                    fw   = GasCoreFramework.from_dict(data)
                    set_fw(fw)
                    st.session_state.initialized = True
                    st.session_state.qa_history  = []
                ev = fw.evolving
                nm_n     = fw.nm.total if fw.nm and fw.nm.is_trained else 0
                empty_n  = len(ev.som.get_empty())   if ev and ev.som else 0
                filled_n = len(ev.som.neuron_sentences) if ev and ev.som else 0
                st.success(
                    f"✓ 로드 완료 ({fw.cycle}순환)\n"
                    f"NM:{nm_n}토큰 | SOM 빈:{empty_n} 채워짐:{filled_n}")
                st.rerun()
            except Exception as e:
                st.error(f"로드 실패: {e}")
                import traceback; st.code(traceback.format_exc())

    # 상태 + 저장
    if st.session_state.initialized:
        fw = get_fw()
        s  = fw.summary()
        st.markdown("---")
        st.markdown("**상태**")
        st.caption(f"순환: {s['cycle']} | 어휘: {s['nm_vocab']}")
        st.caption(f"총 문장: {s.get('ev_총 문장',0)}")
        st.caption(f"승인 생성: {s.get('ev_승인된 생성',0)}")
        st.caption(f"SOM 빈 자리: {s.get('ev_빈 자리',0)}")
        st.download_button(
            "💾 저장",
            data=pickle.dumps(fw.to_dict()),
            file_name=f"gascore_c{fw.cycle}.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )

    st.markdown("---")
    st.caption("⚡ GasCore | GPU 0 | CPU only")


# ── 메인 ─────────────────────────────────────────────────────
st.markdown("# ⚡ GasCore Framework")
st.caption("Layer 1: 진화 인덱스  |  Layer 2: CoreAI 가드레일  |  Layer 3: XAI")

if not st.session_state.initialized:
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="lb l1">LAYER 1</div>', unsafe_allow_html=True)
        st.markdown("**진화 인덱스**")
        st.caption("SOM → LLM 생성 → 사용자 검증 → 코퍼스 성장")
    with c2:
        st.markdown('<div class="lb l2">LAYER 2</div>', unsafe_allow_html=True)
        st.markdown("**CoreAI 가드레일**")
        st.caption("NeuralMarkov → 도메인 이탈 탐지 → 재생성")
    with c3:
        st.markdown('<div class="lb l3">LAYER 3</div>', unsafe_allow_html=True)
        st.markdown("**XAI 설명**")
        st.caption("토큰별 logP → 이탈 원인 설명")
    st.info("← 사이드바에서 코퍼스를 업로드하거나 저장된 인덱스를 불러오세요.")
    st.stop()

# ── 탭 ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 L1 진화 인덱스",
    "🛡️ L2 CoreAI 가드레일",
    "🔍 L3 XAI 설명",
    "🔄 순환 관리",
])


# ━━━ 탭 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    fw = get_fw()
    st.markdown('<div class="lb l1">LAYER 1 — 진화 인덱스</div>', unsafe_allow_html=True)

    cl, cr = st.columns([1,2])
    with cl:
        max_cands = st.slider("최대 후보", 3, 15, 5, key="l1_max")
        if st.button("⚡ 후보 생성", use_container_width=True):
            with st.spinner("생성 중..."):
                t0 = time.perf_counter()
                try:
                    if fw.evolving:
                        fw.evolving.pending = []  # 중복 방지 초기화
                    cands = fw.generate_candidates(
                        max_candidates=max_cands,
                        logp_thr=st.session_state.ev_thr,
                        api_key=st.session_state.api_key,
                        model=st.session_state.model,
                    )
                    ms = (time.perf_counter()-t0)*1000
                    if cands:
                        llm_n = sum(1 for c in cands if c.get("by_llm"))
                        st.success(f"✓ {len(cands)}개 ({ms:.0f}ms)"
                                   f"{' 🤖LLM '+str(llm_n) if llm_n else ''}")
                    else:
                        ev = fw.evolving
                        st.warning(
                            f"후보 없음 ({ms:.0f}ms)\n"
                            f"시도:{ev.stats['total_generated']} "
                            f"거부:{ev.stats['auto_rejected']}\n"
                            f"SOM 빈:{len(ev.som.get_empty()) if ev.som else 0} "
                            f"채워짐:{len(ev.som.neuron_sentences) if ev.som else 0}"
                        )
                    set_fw(fw)  # 변경사항 세션 반영
                except Exception as e:
                    st.error(f"오류: {e}")
                    import traceback; st.code(traceback.format_exc())

        st.markdown("---")
        if fw.evolving:
            ev_s = fw.evolving.summary()
            for k,v in ev_s.items():
                st.caption(f"{k}: {v}")

    with cr:
        pending = fw.evolving.pending if fw.evolving else []
        reasons = ["도메인 밖","사실 다름","표현 어색","중복","기타"]
        if not pending:
            st.info("← 후보를 생성하세요.")
        else:
            st.caption(f"대기: {len(pending)}개")
            for i, cand in enumerate(pending):
                status = cand.get("coreai_status","?")
                logp   = cand.get("logp",0)
                s_cls  = {"PASS":"pass","WARNING":"warn"}.get(status,"")
                s_icon = {"PASS":"🟢","WARNING":"🟡"}.get(status,"⬜")
                tag    = "🤖" if cand.get("by_llm") else "📐"
                vr     = f"vocab:{cand.get('vocab_ratio',0):.0%}" if "vocab_ratio" in cand else ""

                st.markdown(f"""
<div class="card">
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:var(--mt);">
    {tag} | <span class="{s_cls}">{s_icon}{status}</span>
    | logP:{logp:+.2f} | {cand.get('generation',0)}세대 {vr}
  </div>
  <div style="font-size:1rem;font-weight:700;margin:0.4rem 0;">
    {cand['sentence']}
  </div>
  <div style="font-size:0.74rem;color:var(--mt);">
    ← {cand['from_a'][:35]}<br>← {cand['from_b'][:35]}
  </div>
</div>""", unsafe_allow_html=True)

                ca, _, cb = st.columns([2,0.2,2])
                key_base = f"{i}_{cand['sentence'][:8]}"
                with ca:
                    if st.button("✅ 승인", key=f"ap_{key_base}",
                                 use_container_width=True):
                        fw.approve_candidate(cand)
                        set_fw(fw)
                        st.rerun()
                with cb:
                    r = st.selectbox("이유", reasons, key=f"rs_{key_base}",
                                     label_visibility="collapsed")
                    if st.button("❌ 거부", key=f"rj_{key_base}",
                                 use_container_width=True):
                        fw.reject_candidate(cand, r)
                        set_fw(fw)
                        st.rerun()
                st.markdown("---")

            ca2,cb2 = st.columns(2)
            with ca2:
                if st.button("✅ 전체 승인", use_container_width=True, key="ap_all"):
                    for c in list(fw.evolving.pending):
                        fw.approve_candidate(c)
                    set_fw(fw); st.rerun()
            with cb2:
                if st.button("❌ 전체 거부", use_container_width=True, key="rj_all"):
                    for c in list(fw.evolving.pending):
                        fw.reject_candidate(c, "일괄")
                    set_fw(fw); st.rerun()


# ━━━ 탭 2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    fw = get_fw()
    st.markdown('<div class="lb l2">LAYER 2 — CoreAI 가드레일</div>', unsafe_allow_html=True)

    if not st.session_state.api_key:
        st.warning("← 사이드바에서 API Key를 입력하세요.")
    
    question = st.text_area("질문", placeholder="질문을 입력하세요...",
                            height=80, label_visibility="collapsed", key="l2_q")
    run = st.button("▶ 실행", use_container_width=True,
                    disabled=not (st.session_state.api_key and question.strip()))

    if run and question.strip():
        def llm_fn(prompt):
            from openai import OpenAI
            client = OpenAI(api_key=st.session_state.api_key)
            msgs   = []
            hint   = get_fw().guideline_hint
            if hint:
                msgs.append({"role":"system","content":
                    f"가이드라인을 참고하여 답하세요.\n\n{hint[:800]}\n\n"
                    f"가이드라인을 최대한 활용하되 일반 지식으로 보완하세요."})
            msgs.append({"role":"user","content":prompt})
            resp = client.chat.completions.create(
                model=st.session_state.model, messages=msgs, max_tokens=500)
            return resp.choices[0].message.content.strip()

        with st.spinner("CoreAI 실행 중..."):
            try:
                result = fw.run_guardrail(
                    question=question,
                    llm_fn=llm_fn,
                    max_attempts=st.session_state.max_retry,
                    logp_thr=st.session_state.logp_thr,
                )
                st.session_state.qa_history.insert(0, {"question":question,"result":result})
                set_fw(fw)

                v     = result.status
                v_cls = {"PASS":"pass","WARNING":"warn","FATAL":"fatal"}.get(v,"")
                v_icon= {"PASS":"🟢","WARNING":"🟡","FATAL":"🔴"}.get(v,"⬜")
                st.markdown(f"""
<div class="card">
  <div class="{v_cls}" style="font-size:2rem;">{v_icon} {v}</div>
  <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:var(--mt);margin-top:0.3rem;">
    {result.attempts}회 | {result.total_ms:.0f}ms | logP {result.final_logp:+.2f}
  </div>
</div>""", unsafe_allow_html=True)

                st.info(result.answer)

                if result.xai:
                    st.markdown("---")
                    st.markdown('<div class="lb l3">XAI 자동 설명</div>',
                                unsafe_allow_html=True)
                    st.caption(result.xai.explanation)
                    if result.xai.token_scores:
                        html = ""
                        for tok,lp,_ in result.xai.token_scores:
                            cls = "tp" if lp>=-10 else "tw" if lp>=-14 else "tf"
                            html += f'<span class="{cls}">{tok}<sub>{lp:+.1f}</sub></span> '
                        st.markdown(html, unsafe_allow_html=True)
                        st.caption("🟢 정상 | 🟡 경계 | 🔴 이탈")
                    if result.xai.outlier_tokens:
                        st.error(f"이탈 토큰: {', '.join(result.xai.outlier_tokens)}")
            except Exception as e:
                st.error(f"오류: {e}")
                import traceback; st.code(traceback.format_exc())


# ━━━ 탭 3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    fw = get_fw()
    st.markdown('<div class="lb l3">LAYER 3 — XAI 설명</div>', unsafe_allow_html=True)

    xai_text = st.text_area("분석할 텍스트", height=100,
                            label_visibility="collapsed",
                            placeholder="도메인 이탈 여부를 분석할 텍스트...",
                            key="l3_text")
    if st.button("🔍 XAI 분석", use_container_width=True):
        if xai_text.strip():
            try:
                xai = fw.explain(xai_text, logp_thr=st.session_state.logp_thr)
                if xai:
                    v = xai.verdict
                    c1,c2,c3 = st.columns(3)
                    c1.metric("판정", f"{'🟢' if v=='PASS' else '🟡' if v=='WARNING' else '🔴'} {v}")
                    c2.metric("avg logP", f"{xai.avg_logp:+.2f}")
                    c3.metric("처리", f"{xai.ms:.2f}ms")
                    st.info(xai.explanation)
                    if xai.token_scores:
                        html = ""
                        for tok,lp,_ in xai.token_scores:
                            cls = "tp" if lp>=-10 else "tw" if lp>=-14 else "tf"
                            html += f'<span class="{cls}">{tok}<sub>{lp:+.1f}</sub></span> '
                        st.markdown(html, unsafe_allow_html=True)
                        st.caption("🟢 정상 | 🟡 경계 | 🔴 이탈")
                    if xai.cluster_hint:
                        st.caption(f"관련 코퍼스 개념: {xai.cluster_hint}")
                    if xai.outlier_tokens:
                        st.error(f"이탈 토큰: {', '.join(xai.outlier_tokens)}")
            except Exception as e:
                st.error(f"오류: {e}")
                import traceback; st.code(traceback.format_exc())


# ━━━ 탭 4 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    fw = get_fw()
    st.markdown("#### 🔄 순환 관리")
    s = fw.summary()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("순환",     s["cycle"])
    c2.metric("어휘",     s["nm_vocab"])
    c3.metric("총 문장",  s.get("ev_총 문장",0))
    c4.metric("승인 생성",s.get("ev_승인된 생성",0))

    st.markdown("---")
    has_gen = bool(fw.evolving and fw.evolving.generated)
    if st.button("🔄 순환 완료 — NeuralMarkov 재학습",
                 use_container_width=True, disabled=not has_gen):
        with st.spinner("재학습 중..."):
            fw.complete_cycle(epochs=5)
            set_fw(fw)
        st.success(f"✓ {fw.cycle}순환 완료 — NeuralMarkov 재학습됨")
        st.rerun()

    if has_gen:
        st.markdown("**이번 순환 승인 문장**")
        for g in fw.evolving.generated:
            icon = "🟢" if g["coreai_status"]=="PASS" else "🟡"
            st.caption(f"{icon} [{g['generation']}세대] {g['sentence']}")

    if st.session_state.qa_history:
        st.markdown("---")
        st.markdown("**Q&A 이력**")
        for item in st.session_state.qa_history[:5]:
            r    = item["result"]
            icon = {"PASS":"🟢","WARNING":"🟡","FATAL":"🔴"}.get(r.status,"⬜")
            with st.expander(f"{icon} {item['question'][:50]}"):
                st.caption(r.answer[:200])
                if r.xai:
                    st.caption(r.xai.explanation)
