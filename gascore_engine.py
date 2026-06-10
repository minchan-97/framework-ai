"""
gascore_engine.py — GasCore 통합 프레임워크
============================================
Layer 1: 진화 인덱스 (SOM + LLM + 사용자 검증)
Layer 2: CoreAI v1 가드레일 (NeuralMarkov + RAG)
Layer 3: XAI (토큰별 이탈 설명)

세 레이어가 순환하며 코퍼스가 점점 전문화됨
"""
from __future__ import annotations
import numpy as np
import time
from collections import Counter, defaultdict
from typing import Optional
from dataclasses import dataclass, field

try:
    from neural_markov_engine import NeuralMarkovEngine
    NM_OK = True
except Exception:
    NM_OK = False

try:
    from evolving_engine import EvolvingIndexEngine
    EV_OK = True
except Exception:
    EV_OK = False

_rng = np.random.default_rng(42)

# ── 토크나이저 ────────────────────────────────────────────────
_JOSA = ["에서","에게","으로","부터","까지","와","과","을","를",
         "은","는","이","가","의","도","만","에","로"]
_EOMI = ["했습니다","합니다","됩니다","있습니다","입니다",
         "했다","한다","이다","하고","해서","하여","되어",
         "이며","하는","된","한","이고","이에"]

def tokenize(text: str) -> list:
    tokens = []
    for word in text.replace("\n"," ").split():
        word = word.strip(".,!?()[]\"'：:；;~")
        stem = word
        for s in sorted(_JOSA+_EOMI, key=len, reverse=True):
            if word.endswith(s) and len(word)>len(s)+1:
                stem = word[:-len(s)]; break
        if stem and len(stem)>1:
            tokens.append(stem)
    return tokens


# ── XAI 결과 ─────────────────────────────────────────────────
@dataclass
class XAIResult:
    """XAI 설명 결과"""
    verdict: str                    # PASS/WARNING/FATAL
    avg_logp: float                 # 전체 평균 logP
    token_scores: list = field(default_factory=list)   # [(토큰, logP, 이상여부)]
    outlier_tokens: list = field(default_factory=list) # 이탈 토큰들
    cluster_hint: str = ""          # 가장 가까운 SOM 클러스터
    explanation: str = ""           # 한국어 설명
    ms: float = 0.0


# ── Layer 3: XAI ─────────────────────────────────────────────
class XAILayer:
    """
    토큰별 logP 추적 + SOM 클러스터 연결
    "왜 이 답변이 이탈로 판정됐는가"를 설명
    """
    def __init__(self, nm: NeuralMarkovEngine,
                 som_neurons: Optional[dict] = None):
        self.nm = nm
        self.som_neurons = som_neurons or {}  # 뉴런 → 문장 리스트

    def explain(self, text: str, logp_thr: float = -11.5) -> XAIResult:
        t0 = time.perf_counter()
        tokens = tokenize(text)
        if not tokens or not self.nm or not self.nm.is_trained:
            return XAIResult(verdict="SKIP", avg_logp=0.0)

        # 토큰별 logP 계산
        token_scores = []
        total_lp = 0.0
        scored = 0
        V = len(self.nm.uni)

        for i in range(len(tokens)):
            wc = tokens[i]
            p1 = (self.nm.uni[wc]+self.nm.alpha)/(self.nm.total+self.nm.alpha*V)
            p2 = p3 = 0.0
            if i>=1:
                wp = tokens[i-1]
                p2 = self.nm.bi[wp][wc]/self.nm.uni[wp] if self.nm.uni[wp]>0 else 0
            if i>=2:
                wpp = tokens[i-2]
                p3 = self.nm.tri[(wpp,wp)][wc]/self.nm.bi[wpp][wp] if self.nm.bi[wpp][wp]>0 else 0
            pjm = 0.6*p3+0.3*p2+0.1*p1
            lp = float(np.log(pjm+1e-12))
            is_outlier = lp < -12.0 and self.nm.bi.get(tokens[i-1] if i>0 else "", {}).get(wc,0)==0
            token_scores.append((wc, round(lp,2), is_outlier))
            total_lp += lp; scored += 1

        avg_logp = total_lp/max(scored,1)
        outliers = [t for t,lp,out in token_scores if out]

        # 판정
        if avg_logp >= -10.0:      verdict = "PASS"
        elif avg_logp >= logp_thr: verdict = "WARNING"
        else:                      verdict = "FATAL"

        # SOM 클러스터 힌트
        cluster_hint = ""
        if self.som_neurons:
            # 이탈 토큰과 가장 관련 있는 클러스터 찾기
            for neuron_idx, sents in self.som_neurons.items():
                for s in sents:
                    if any(ot in s for ot in outliers[:3]):
                        cluster_hint = s[:40]
                        break
                if cluster_hint: break

        # 한국어 설명 생성
        explanation = self._make_explanation(
            verdict, avg_logp, outliers, cluster_hint, logp_thr
        )

        ms = (time.perf_counter()-t0)*1000
        return XAIResult(
            verdict=verdict,
            avg_logp=avg_logp,
            token_scores=token_scores,
            outlier_tokens=outliers,
            cluster_hint=cluster_hint,
            explanation=explanation,
            ms=ms,
        )

    def _make_explanation(self, verdict, avg_logp,
                          outliers, cluster_hint, thr) -> str:
        if verdict == "PASS":
            return f"✅ 도메인 안 — 모든 토큰이 학습된 패턴 안에 있어요 (avg_logP: {avg_logp:+.2f})"
        elif verdict == "WARNING":
            out_str = f"'{', '.join(outliers[:3])}'" if outliers else "일부 표현"
            hint = f" → 가장 가까운 개념: '{cluster_hint}'" if cluster_hint else ""
            return (f"🟡 경계 수준 — {out_str}이 코퍼스 경계에 있어요 "
                    f"(avg_logP: {avg_logp:+.2f}, 임계값: {thr}){hint}")
        else:
            out_str = f"'{', '.join(outliers[:3])}'" if outliers else "다수 표현"
            hint = f" → 관련 코퍼스 개념: '{cluster_hint}'" if cluster_hint else ""
            return (f"🔴 도메인 이탈 — {out_str}이 학습된 패턴에서 크게 벗어났어요 "
                    f"(avg_logP: {avg_logp:+.2f}, 임계값: {thr}){hint}")


# ── Layer 2: CoreAI 가드레일 ──────────────────────────────────
@dataclass
class GuardrailResult:
    """가드레일 판정 결과"""
    answer: str
    status: str
    attempts: int
    final_logp: float
    xai: Optional[XAIResult] = None
    history: list = field(default_factory=list)
    total_ms: float = 0.0


class CoreAILayer:
    """
    NeuralMarkov 기반 가드레일
    이탈 시 재생성 루프 + XAI 설명
    """
    def __init__(self, nm: NeuralMarkovEngine,
                 xai: Optional[XAILayer] = None):
        self.nm = nm
        self.xai = xai

    def run(self, question: str, llm_fn,
            max_attempts: int = 3,
            logp_thr: float = -11.5,
            guideline_hint: str = "") -> GuardrailResult:
        t0 = time.perf_counter()
        history = []

        for attempt in range(1, max_attempts+1):
            # LLM 호출
            if attempt == 1:
                prompt = question
            else:
                prompt = (
                    f"이전 답변이 가이드라인에서 벗어났어요.\n"
                    f"가이드라인 참고: {guideline_hint[:300]}\n\n"
                    f"다시 답변해주세요: {question}"
                )

            try:
                answer = llm_fn(prompt)
            except Exception as e:
                answer = f"[LLM 오류: {e}]"
                break

            # 가드레일 평가
            if self.nm and self.nm.is_trained:
                result = self.nm.evaluate(answer, logp_thr=logp_thr)
                status  = result.get("status","FATAL")
                avg_logp = result.get("avg_logp",0.0)
            else:
                status = "PASS"; avg_logp = 0.0

            history.append({
                "attempt": attempt,
                "status":  status,
                "avg_logp": avg_logp,
                "answer_preview": answer[:80],
            })

            if status == "PASS":
                break

        # XAI 설명
        xai_result = None
        if self.xai:
            xai_result = self.xai.explain(answer, logp_thr=logp_thr)

        total_ms = (time.perf_counter()-t0)*1000
        return GuardrailResult(
            answer=answer,
            status=status,
            attempts=attempt,
            final_logp=avg_logp,
            xai=xai_result,
            history=history,
            total_ms=total_ms,
        )


# ── GasCore 통합 프레임워크 ───────────────────────────────────
class GasCoreFramework:
    """
    GasCore 통합 프레임워크
    세 레이어를 순환하며 코퍼스가 전문화됨

    Loop:
      1. 진화 인덱스 → 코퍼스 성장
      2. CoreAI → 성장한 코퍼스로 재학습
      3. XAI → 이탈 설명
      4. 사용자 피드백 → 1번으로
    """
    def __init__(self):
        # Layer 1
        self.evolving = EvolvingIndexEngine() if EV_OK else None
        # Layer 2
        self.nm = NeuralMarkovEngine() if NM_OK else None
        # Layer 3
        self.xai_layer: Optional[XAILayer] = None
        self.coreai: Optional[CoreAILayer] = None

        self.corpus_text: str = ""
        self.guideline_hint: str = ""
        self.is_initialized: bool = False
        self.cycle: int = 0  # 몇 번 순환했는지

    # ── 초기화 ───────────────────────────────────────────────
    def initialize(self, corpus_text: str,
                   epochs: int = 10,
                   som_grid: int = 6,
                   on_progress=None):
        """전체 프레임워크 초기화"""
        self.corpus_text = corpus_text
        self.guideline_hint = corpus_text[:1000]

        # Layer 1: 진화 인덱스
        if on_progress: on_progress(10, "진화 인덱스 초기화...")
        if self.evolving:
            self.evolving = EvolvingIndexEngine(grid=som_grid)
            self.evolving.initialize(corpus_text, epochs=epochs)

        # Layer 2: NeuralMarkov
        if on_progress: on_progress(70, "CoreAI 학습...")
        if self.nm:
            self.nm.train(corpus_text, embedding_dim=32, epochs=epochs)

        # Layer 3: XAI
        if on_progress: on_progress(90, "XAI 설정...")
        self._rebuild_layers()

        if on_progress: on_progress(100, "완료")
        self.is_initialized = True

    def _rebuild_layers(self):
        """nm 재학습 후 XAI + CoreAI 레이어 재구성"""
        som_neurons = {}
        if self.evolving and self.evolving.som:
            som_neurons = self.evolving.som.neuron_sentences

        if self.nm:
            self.xai_layer = XAILayer(self.nm, som_neurons)
            self.coreai    = CoreAILayer(self.nm, self.xai_layer)

    # ── Layer 1: 진화 실행 ────────────────────────────────────
    def generate_candidates(self, max_candidates: int = 10,
                            logp_thr: float = -13.0,
                            api_key: str = "",
                            model: str = "gpt-4o-mini") -> list:
        if not self.evolving: return []
        return self.evolving.generate_candidates(
            max_candidates=max_candidates,
            logp_thr=logp_thr,
            api_key=api_key,
            model=model,
        )

    def approve_candidate(self, candidate: dict):
        """승인 → 인덱스 추가 → NeuralMarkov 증분 학습 → XAI 재구성"""
        if not self.evolving: return
        self.evolving.user_approve(candidate)
        # NeuralMarkov 증분 학습
        if self.nm:
            sent = candidate["sentence"]
            toks = tokenize(sent)
            for i,t in enumerate(toks):
                self.nm.uni[t]   += 1
                self.nm.total    += 1
                if i>=1: self.nm.bi[toks[i-1]][t]              += 1
                if i>=2: self.nm.tri[(toks[i-2],toks[i-1])][t] += 1
            self._rebuild_layers()

    def reject_candidate(self, candidate: dict, reason: str = ""):
        if not self.evolving: return
        self.evolving.user_reject(candidate, reason)

    # ── Layer 2: CoreAI 가드레일 실행 ────────────────────────
    def run_guardrail(self, question: str, llm_fn,
                      max_attempts: int = 3,
                      logp_thr: float = -11.5) -> GuardrailResult:
        if not self.coreai:
            # 가드레일 없이 LLM만
            try:
                answer = llm_fn(question)
            except Exception as e:
                answer = str(e)
            return GuardrailResult(
                answer=answer, status="SKIP",
                attempts=1, final_logp=0.0,
            )
        return self.coreai.run(
            question=question,
            llm_fn=llm_fn,
            max_attempts=max_attempts,
            logp_thr=logp_thr,
            guideline_hint=self.guideline_hint,
        )

    # ── Layer 3: XAI 독립 실행 ───────────────────────────────
    def explain(self, text: str,
                logp_thr: float = -11.5) -> Optional[XAIResult]:
        if not self.xai_layer: return None
        return self.xai_layer.explain(text, logp_thr)

    # ── 순환 완료 (사용자 피드백 후 재학습) ──────────────────
    def complete_cycle(self, new_corpus_texts: list = None,
                       epochs: int = 5):
        """
        한 순환 완료:
        승인된 생성 문장들로 NeuralMarkov 재학습 → 다음 순환
        """
        if self.evolving and self.evolving.generated:
            # 생성된 문장 추가 학습
            new_sents = [g["sentence"] for g in self.evolving.generated]
            new_corpus = "\n".join(new_sents)
            if self.nm and new_corpus.strip():
                self.nm.train(new_corpus, embedding_dim=32, epochs=epochs)
        if new_corpus_texts:
            for text in new_corpus_texts:
                if self.nm:
                    self.nm.train(text, embedding_dim=32, epochs=epochs)
        self._rebuild_layers()
        self.cycle += 1

    # ── 저장/로드 ─────────────────────────────────────────────
    def to_dict(self) -> dict:
        data = {
            "corpus_text":    self.corpus_text,
            "guideline_hint": self.guideline_hint,
            "cycle":          self.cycle,
            "is_initialized": self.is_initialized,
        }
        if self.evolving:
            data["evolving"] = self.evolving.to_dict()
        if self.nm and self.nm.is_trained:
            data["nm"] = {
                "uni":   dict(self.nm.uni),
                "bi":    {k:dict(v) for k,v in self.nm.bi.items()},
                "tri":   {k:dict(v) for k,v in self.nm.tri.items()},
                "total": self.nm.total,
                "alpha": getattr(self.nm,"alpha",0.001),
            }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "GasCoreFramework":
        fw = cls()
        fw.corpus_text    = data.get("corpus_text","")
        fw.guideline_hint = data.get("guideline_hint","")
        fw.cycle          = data.get("cycle",0)
        fw.is_initialized = data.get("is_initialized",False)

        # evolving — SOM W 포함이므로 재학습 없이 즉시 복원
        if "evolving" in data and EV_OK:
            fw.evolving = EvolvingIndexEngine.from_dict(data["evolving"])

        # NeuralMarkov 복원
        if "nm" in data and NM_OK:
            nm = data["nm"]
            fw.nm = NeuralMarkovEngine()
            fw.nm.uni   = Counter(nm["uni"])
            fw.nm.bi    = defaultdict(Counter,
                            {k:Counter(v) for k,v in nm["bi"].items()})
            fw.nm.tri   = defaultdict(Counter,
                            {k:Counter(v) for k,v in nm["tri"].items()})
            fw.nm.total = nm["total"]
            fw.nm.alpha = nm.get("alpha",0.001)
            fw.nm.is_trained = True

        # XAI + CoreAI 레이어 구성 (SOM 재학습 없음)
        fw._rebuild_layers()
        return fw

    # ── 상태 요약 ─────────────────────────────────────────────
    def summary(self) -> dict:
        ev_sum = self.evolving.summary() if self.evolving else {}
        return {
            "cycle":         self.cycle,
            "nm_vocab":      len(self.nm.uni) if self.nm else 0,
            "nm_trained":    self.nm.is_trained if self.nm else False,
            **{f"ev_{k}":v for k,v in ev_sum.items()},
        }
