"""
app.py — GasCore 통합 프레임워크 앱
====================================
Layer 1: 진화 인덱스 (SOM + LLM + 사용자 검증)
Layer 2: CoreAI v1 가드레일 (NeuralMarkov)
Layer 3: XAI (토큰별 이탈 설명)
"""
import streamlit as st
import pickle, io, os, time
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="GasCore Framework",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Noto+Sans+KR:wght@300;400;700&display=swap');
:root{
  --bg:#0a0e1a;--surface:#111827;--surface2:#1a2235;
  --border:#1e3050;--accent:#00e5ff;--green:#00ff88;
  --yellow:#ffd600;--red:#ff4444;--text:#e2e8f0;--muted:#64748b;
}
html,body,[data-testid="stAppViewContainer"]{
  background:var(--bg)!important;color:var(--text)!important;
  font-family:'Noto Sans KR',sans-serif;
}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
.title{font-family:'Space Mono',monospace;font-size:2rem;font-weight:700;color:var(--accent);}
.sub{font-family:'Space Mono',monospace;font-size:0.7rem;color:var(--muted);margin-bottom:1.5rem;}
.layer-badge{
  display:inline-block;padding:2px 10px;border-radius:4px;
  font-family:'Space Mono',monospace;font-size:0.65rem;font-weight:700;
  letter-spacing:1px;margin-bottom:0.5rem;
}
.l1{background:#1a2a1a;border:1px solid #00ff88;color:#00ff88;}
.l2{background:#1a1a2a;border:1px solid #00e5ff;color:#00e5ff;}
.l3{background:#2a1a1a;border:1px solid #ffd600;color:#ffd600;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1.2rem;margin-bottom:0.8rem;}
.pass{color:#00ff88;font-family:'Space Mono',monospace;font-weight:700;}
.warn{color:#ffd600;font-family:'Space Mono',monospace;font-weight:700;}
.fatal{color:#ff4444;font-family:'Space Mono',monospace;font-weight:700;}
.token-pass{background:#0d1f0d;color:#00ff88;padding:2px 6px;border-radius:3px;font-family:monospace;font-size:0.82rem;margin:2px;}
.token-warn{background:#1f1a0d;color:#ffd600;padding:2px 6px;border-radius:3px;font-family:monospace;font-size:0.82rem;margin:2px;}
.token-fatal{background:#1f0d0d;color:#ff4444;padding:2px 6px;border-radius:3px;font-family:monospace;font-size:0.82rem;margin:2px;}
[data-testid="stButton"] button{font-weight:700!important;border-radius:6px!important;border:none!important;}
hr{border-color:var(--border)!important;}
</style>
""", unsafe_allow_html=True)

try:
    from gascore_engine import GasCoreFramework
    ENGINE_OK = True
except Exception as e:
    ENGINE_OK = False
    st.error(f"엔진 로딩 실패: {e}")

# ── 세션 ─────────────────────────────────────────────────────
if "fw" not in st.session_state:
    st.session_state.fw = GasCoreFramework() if ENGINE_OK else None
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "logp_thr" not in st.session_state:
    st.session_state.logp_thr = -11.5
if "max_retry" not in st.session_state:
    st.session_state.max_retry = 3

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ GasCore")
    st.markdown("---")

    # API Key
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY",""),
        type="password",
    )
    model = st.selectbox("모델", ["gpt-4o-mini","gpt-4o"])

    st.markdown("---")
    st.markdown("### 📚 코퍼스")
    corpus_file = st.file_uploader(
        "코퍼스 업로드",
        type=["txt","pdf","docx"],
    )

    col1,col2 = st.columns(2)
    with col1: epochs  = st.select_slider("학습", [5,10,15,20], value=10)
    with col2: som_grid= st.select_slider("SOM", [4,5,6,7], value=6)

    logp_thr   = st.slider("가드레일 임계값", -15.0, -5.0, st.session_state.logp_thr, 0.5)
    ev_thr     = st.slider("진화 임계값", -16.0, -8.0, -13.0, 0.5)
    max_retry  = st.slider("최대 재생성", 1, 5, st.session_state.max_retry)
    st.session_state.logp_thr  = logp_thr
    st.session_state.max_retry = max_retry

    if corpus_file and st.button("🚀 초기화", use_container_width=True):
        with st.spinner("GasCore 초기화 중..."):
            try:
                name = corpus_file.name.lower()
                if name.endswith(".pdf"):
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(corpus_file.read()))
                    text = "\n".join(p.extract_text() or "" for p in reader.pages)
                elif name.endswith(".docx"):
                    import docx
                    doc = docx.Document(io.BytesIO(corpus_file.read()))
                    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                else:
                    text = corpus_file.read().decode("utf-8", errors="ignore")

                prog = st.progress(0)
                def cb(pct, msg):
                    prog.progress(pct)

                st.session_state.fw = GasCoreFramework()
                st.session_state.fw.initialize(
                    text, epochs=epochs,
                    som_grid=som_grid, on_progress=cb
                )
                # guideline_hint 세션 저장
                st.session_state.fw.guideline_hint = text[:1000]
                st.session_state.initialized = True
                st.session_state.qa_history = []
                # 학습 확인
                nm_ok = st.session_state.fw.nm and st.session_state.fw.nm.is_trained
                nm_total = st.session_state.fw.nm.total if nm_ok else 0
                st.success(f"✓ GasCore 초기화 완료 | NM {'✅' if nm_ok else '❌'} ({nm_total}토큰)")
            except Exception as e:
                st.error(f"실패: {e}")

    # pkl 로드
    st.markdown("---")
    pkl_file = st.file_uploader("저장된 프레임워크 (.pkl)", type=["pkl"])
    if pkl_file and not st.session_state.initialized:
        try:
            data = pickle.loads(pkl_file.read())
            st.session_state.fw = GasCoreFramework.from_dict(data)
            st.session_state.initialized = True
            st.success(f"✓ 로드 완료 ({st.session_state.fw.cycle}순환)")
            st.rerun()
        except Exception as e:
            st.error(f"로드 실패: {e}")

    # 상태 + 저장
    if st.session_state.initialized:
        fw = st.session_state.fw
        s = fw.summary()
        st.markdown("---")
        st.markdown("**상태**")
        st.caption(f"순환: {s['cycle']}회")
        st.caption(f"NM 어휘: {s['nm_vocab']}개")
        st.caption(f"총 문장: {s.get('ev_총 문장',0)}개")
        st.caption(f"승인 생성: {s.get('ev_승인된 생성',0)}개")
        st.caption(f"SOM 빈 자리: {s.get('ev_빈 자리',0)}개")

        data_bytes = pickle.dumps(fw.to_dict())
        st.download_button(
            "💾 프레임워크 저장",
            data=data_bytes,
            file_name=f"gascore_cycle{fw.cycle}.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )

    st.markdown("---")
    st.caption("⚡ GasCore Framework")
    st.caption("진화 인덱스 + CoreAI + XAI")
    st.caption("GPU 0 | CPU only")


# ── 메인 ─────────────────────────────────────────────────────
st.markdown('<div class="title">⚡ GasCore Framework</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">'
    'Layer 1: 진화 인덱스 &nbsp;|&nbsp; '
    'Layer 2: CoreAI 가드레일 &nbsp;|&nbsp; '
    'Layer 3: XAI 설명'
    '</div>',
    unsafe_allow_html=True
)

if not st.session_state.initialized:
    # 파이프라인 설명
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="layer-badge l1">LAYER 1</div>', unsafe_allow_html=True)
        st.markdown("**진화 인덱스**")
        st.caption("SOM 빈 자리 → LLM 의미 생성 → 사용자 검증 → 코퍼스 성장")
    with c2:
        st.markdown('<div class="layer-badge l2">LAYER 2</div>', unsafe_allow_html=True)
        st.markdown("**CoreAI 가드레일**")
        st.caption("NeuralMarkov → 도메인 이탈 탐지 → 재생성 루프")
    with c3:
        st.markdown('<div class="layer-badge l3">LAYER 3</div>', unsafe_allow_html=True)
        st.markdown("**XAI 설명**")
        st.caption("어떤 토큰에서 이탈? 어느 개념 클러스터? 왜 경고?")
    st.info("← 사이드바에서 코퍼스를 업로드하세요.")
    st.stop()

fw = st.session_state.fw
tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 L1 진화 인덱스",
    "🛡️ L2 CoreAI 가드레일",
    "🔍 L3 XAI 설명",
    "🔄 순환 관리",
])


# ── 탭 1: 진화 인덱스 ────────────────────────────────────────
with tab1:
    st.markdown('<div class="layer-badge l1">LAYER 1 — 진화 인덱스</div>',
                unsafe_allow_html=True)

    col_ctrl, col_cands = st.columns([1, 2])

    with col_ctrl:
        st.markdown("#### 후보 생성")
        max_cands = st.slider("최대 후보", 3, 15, 5, key="l1_max")

        if st.button("⚡ 생성 실행", use_container_width=True):
            with st.spinner("SOM + LLM + CoreAI 검증..."):
                t0 = time.perf_counter()
                cands = fw.generate_candidates(
                    max_candidates=max_cands,
                    logp_thr=ev_thr,
                    api_key=api_key,
                    model=model,
                )
                ms = (time.perf_counter()-t0)*1000
            llm_cnt = sum(1 for c in cands if c.get("by_llm"))
            st.success(f"✓ {len(cands)}개 생성 ({ms:.0f}ms)"
                      f"{' | 🤖 LLM '+str(llm_cnt)+'개' if llm_cnt else ''}")

        st.markdown("---")
        st.markdown("**거부 이유**")
        reasons = ["도메인 밖","사실 다름","표현 어색","중복","기타"]

        if fw.evolving:
            ev_s = fw.evolving.summary()
            st.metric("총 문장", ev_s["총 문장"])
            st.metric("승인 생성", ev_s["승인된 생성"])
            st.metric("SOM 빈 자리", ev_s["빈 자리"])

    with col_cands:
        st.markdown("#### 사용자 검증")
        pending = fw.evolving.pending if fw.evolving else []

        if not pending:
            st.info("후보를 생성하세요.")
        else:
            st.caption(f"대기: {len(pending)}개")
            for i, cand in enumerate(pending):
                status = cand.get("coreai_status","?")
                logp   = cand.get("logp",0)
                s_cls  = {"PASS":"pass","WARNING":"warn"}.get(status,"")
                s_icon = {"PASS":"🟢","WARNING":"🟡"}.get(status,"⬜")
                llm_tag = "🤖" if cand.get("by_llm") else "📐"

                st.markdown(f"""
<div class="card">
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:var(--muted);">
    {llm_tag} | CoreAI <span class="{s_cls}">{s_icon}{status}</span>
    | logP:{logp:+.2f} | {cand['generation']}세대
  </div>
  <div style="font-size:1rem;font-weight:700;margin:0.4rem 0;">
    {cand['sentence']}
  </div>
  <div style="font-size:0.75rem;color:var(--muted);">
    ← {cand['from_a'][:35]}<br>
    ← {cand['from_b'][:35]}
  </div>
</div>""", unsafe_allow_html=True)

                ca, cb, cc = st.columns([2,1,2])
                with ca:
                    if st.button("✅ 승인",
                                 key=f"ap_{i}_{cand['sentence'][:8]}",
                                 use_container_width=True):
                        fw.approve_candidate(cand)
                        st.rerun()
                with cc:
                    reason = st.selectbox(
                        "이유", reasons,
                        key=f"rs_{i}_{cand['sentence'][:8]}",
                        label_visibility="collapsed",
                    )
                    if st.button("❌ 거부",
                                 key=f"rj_{i}_{cand['sentence'][:8]}",
                                 use_container_width=True):
                        fw.reject_candidate(cand, reason)
                        st.rerun()
                st.markdown("---")

            ca2, cb2 = st.columns(2)
            with ca2:
                if st.button("✅ 전체 승인", use_container_width=True):
                    for c in list(fw.evolving.pending):
                        fw.approve_candidate(c)
                    st.rerun()
            with cb2:
                if st.button("❌ 전체 거부", use_container_width=True):
                    for c in list(fw.evolving.pending):
                        fw.reject_candidate(c, "일괄 거부")
                    st.rerun()


# ── 탭 2: CoreAI 가드레일 ────────────────────────────────────
with tab2:
    st.markdown('<div class="layer-badge l2">LAYER 2 — CoreAI 가드레일</div>',
                unsafe_allow_html=True)

    question = st.text_area(
        "질문",
        placeholder="질문을 입력하세요...",
        height=80,
        label_visibility="collapsed",
    )
    run = st.button("▶ 실행", use_container_width=True,
                    disabled=not api_key)

    if run and question.strip():
        def llm_fn(prompt):
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            msgs = []
            if fw.guideline_hint:
                msgs.append({"role":"system","content":
                    f"다음 가이드라인을 참고하여 답하세요.\n\n"
                    f"가이드라인:\n{fw.guideline_hint[:800]}\n\n"
                    f"가이드라인을 최대한 활용하되, 일반 지식으로 보완하세요."
                })
            msgs.append({"role":"user","content":prompt})
            resp = client.chat.completions.create(
                model=model, messages=msgs, max_tokens=500
            )
            return resp.choices[0].message.content.strip()

        with st.spinner("CoreAI 실행 중..."):
            result = fw.run_guardrail(
                question=question,
                llm_fn=llm_fn,
                max_attempts=st.session_state.max_retry,
                logp_thr=st.session_state.logp_thr,
            )

        st.session_state.qa_history.insert(0, {
            "question": question,
            "result": result,
        })

        # 판정 표시
        v = result.status
        v_cls = {"PASS":"pass","WARNING":"warn","FATAL":"fatal"}.get(v,"")
        v_icon = {"PASS":"🟢","WARNING":"🟡","FATAL":"🔴"}.get(v,"⬜")

        st.markdown(f"""
<div style="background:#111827;border:1px solid #1e3050;border-radius:8px;
padding:1.5rem;margin:1rem 0;">
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
  color:#64748b;letter-spacing:3px;margin-bottom:0.5rem;">VERDICT</div>
  <div class="{v_cls}" style="font-size:2rem;">{v_icon} {v}</div>
  <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
  color:#64748b;margin-top:0.4rem;">
    {result.attempts}회 시도 | {result.total_ms:.0f}ms | logP {result.final_logp:+.2f}
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("**답변**")
        st.info(result.answer)

        # XAI 자동 표시
        if result.xai:
            st.markdown("---")
            st.markdown('<div class="layer-badge l3">LAYER 3 — XAI</div>',
                       unsafe_allow_html=True)
            st.caption(result.xai.explanation)

            # 토큰별 시각화
            if result.xai.token_scores:
                token_html = ""
                for tok, lp, is_out in result.xai.token_scores:
                    if lp >= -10.0:     cls = "token-pass"
                    elif lp >= -14.0:   cls = "token-warn"
                    else:               cls = "token-fatal"
                    token_html += f'<span class="{cls}">{tok}<sub>{lp:+.1f}</sub></span> '
                st.markdown(f"**토큰별 logP**: {token_html}",
                           unsafe_allow_html=True)

            if result.xai.outlier_tokens:
                st.error(f"이탈 토큰: {', '.join(result.xai.outlier_tokens)}")


# ── 탭 3: XAI 독립 실행 ──────────────────────────────────────
with tab3:
    st.markdown('<div class="layer-badge l3">LAYER 3 — XAI 설명</div>',
                unsafe_allow_html=True)

    xai_text = st.text_area(
        "분석할 텍스트",
        placeholder="도메인 이탈 여부를 분석할 텍스트를 입력하세요...",
        height=100,
        label_visibility="collapsed",
    )
    if st.button("🔍 XAI 분석", use_container_width=True):
        if xai_text.strip():
            xai = fw.explain(xai_text, logp_thr=st.session_state.logp_thr)
            if xai:
                v = xai.verdict
                v_cls = {"PASS":"pass","WARNING":"warn","FATAL":"fatal"}.get(v,"")
                v_icon = {"PASS":"🟢","WARNING":"🟡","FATAL":"🔴"}.get(v,"⬜")

                c1,c2,c3 = st.columns(3)
                c1.metric("판정", f"{v_icon} {v}")
                c2.metric("avg_logP", f"{xai.avg_logp:+.2f}")
                c3.metric("처리", f"{xai.ms:.2f}ms")

                st.markdown("**설명**")
                st.info(xai.explanation)

                if xai.token_scores:
                    st.markdown("**토큰별 분석**")
                    token_html = ""
                    for tok, lp, is_out in xai.token_scores:
                        if lp >= -10.0:     cls = "token-pass"
                        elif lp >= -14.0:   cls = "token-warn"
                        else:               cls = "token-fatal"
                        token_html += f'<span class="{cls}">{tok}<sub>{lp:+.1f}</sub></span> '
                    st.markdown(token_html, unsafe_allow_html=True)
                    st.caption("🟢 정상 | 🟡 경계 | 🔴 이탈")

                if xai.cluster_hint:
                    st.markdown(f"**가장 가까운 코퍼스 개념**: {xai.cluster_hint}")

                if xai.outlier_tokens:
                    st.error(f"이탈 토큰: {', '.join(xai.outlier_tokens)}")


# ── 탭 4: 순환 관리 ──────────────────────────────────────────
with tab4:
    st.markdown("#### 🔄 순환 관리")
    st.caption("승인된 생성 문장으로 NeuralMarkov를 재학습해서 다음 순환을 시작해요")

    fw2 = st.session_state.fw
    s = fw2.summary()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("현재 순환", s["cycle"])
    c2.metric("NM 어휘", s["nm_vocab"])
    c3.metric("총 문장", s.get("ev_총 문장",0))
    c4.metric("승인 생성", s.get("ev_승인된 생성",0))

    st.markdown("---")
    if st.button("🔄 순환 완료 — NeuralMarkov 재학습",
                 use_container_width=True,
                 disabled=not (fw2.evolving and fw2.evolving.generated)):
        with st.spinner("재학습 중..."):
            fw2.complete_cycle(epochs=5)
        st.success(f"✓ {fw2.cycle}순환 완료 — NeuralMarkov 재학습됨")
        st.rerun()

    if fw2.evolving and fw2.evolving.generated:
        st.markdown("**이번 순환 승인 문장**")
        for g in fw2.evolving.generated:
            icon = "🟢" if g["coreai_status"]=="PASS" else "🟡"
            st.caption(f"{icon} [{g['generation']}세대] {g['sentence']}")

    st.markdown("---")
    if st.session_state.qa_history:
        st.markdown("**Q&A 이력**")
        for item in st.session_state.qa_history[:5]:
            r = item["result"]
            icon = {"PASS":"🟢","WARNING":"🟡","FATAL":"🔴"}.get(r.status,"⬜")
            with st.expander(f"{icon} {item['question'][:50]}"):
                st.caption(r.answer[:200])
                if r.xai:
                    st.caption(r.xai.explanation)
