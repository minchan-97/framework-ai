"""
evolving_engine.py — 자기 진화 인덱스 엔진
==========================================
SOM 빈 자리 → 개념 조합 → CoreAI 1차 검증
→ 사용자 2차 검증 → 인덱스 누적
"""
from __future__ import annotations
import numpy as np
import pickle
import time
from collections import Counter, defaultdict
from typing import Optional

try:
    from neural_markov_engine import NeuralMarkovEngine
    NM_OK = True
except Exception:
    NM_OK = False

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


# ── SOM ──────────────────────────────────────────────────────
class SOM:
    def __init__(self, grid: int = 6, dim: int = 30):
        self.grid = grid
        self.dim = dim
        self.W = _rng.normal(0.5, 0.1, (grid*grid, dim))
        self.neuron_sentences: dict = {}

    def _bmu(self, x):
        d = np.linalg.norm(self.W - x, axis=1)
        return int(np.argmin(d)), float(np.min(d))

    def _h(self, i, bmu, sigma):
        gi,gj = i//self.grid, i%self.grid
        bi,bj = bmu//self.grid, bmu%self.grid
        return float(np.exp(-((gi-bi)**2+(gj-bj)**2)/(2*sigma**2)))

    def train(self, vecs, sentences, epochs=60):
        n = len(vecs)
        for ep in range(epochs):
            lr = 0.5*np.exp(-ep/epochs)
            sigma = max(0.5, self.grid/2*np.exp(-ep/epochs))
            for i in _rng.permutation(n):
                bmu,_ = self._bmu(vecs[i])
                for j in range(self.grid*self.grid):
                    self.W[j] += lr*self._h(j,bmu,sigma)*(vecs[i]-self.W[j])
        self.neuron_sentences = {}
        for v,s in zip(vecs, sentences):
            bmu,_ = self._bmu(v)
            if bmu not in self.neuron_sentences:
                self.neuron_sentences[bmu] = []
            self.neuron_sentences[bmu].append(s)

    def get_empty(self):
        return [i for i in range(self.grid*self.grid)
                if i not in self.neuron_sentences]

    def get_neighbors(self, idx, radius=2):
        gi,gj = idx//self.grid, idx%self.grid
        result = []
        for ni, sents in self.neuron_sentences.items():
            ngi,ngj = ni//self.grid, ni%self.grid
            dist = abs(gi-ngi)+abs(gj-ngj)
            if 0 < dist <= radius:
                result.append((dist, sents))
        return sorted(result)


# ── 자기 진화 인덱스 엔진 ─────────────────────────────────────
class EvolvingIndexEngine:
    """
    자기 진화 인덱스
    - SOM으로 개념 지도 형성
    - 빈 자리에서 새 의미 생성
    - CoreAI 1차 검증 + 사용자 2차 검증
    - 검증 통과 시 인덱스 누적
    """
    def __init__(self, grid: int = 6):
        self.grid = grid
        self.sentences: list = []
        self.generated: list = []         # 승인된 생성 문장
        self.rejected_auto: list = []     # CoreAI 자동 거부
        self.rejected_user: list = []     # 사용자 거부
        self.pending: list = []           # 사용자 검증 대기
        self.generation: int = 0
        self.vocab: dict = {}
        self.V: int = 0
        self.som: Optional[SOM] = None
        self.nm: Optional[NeuralMarkovEngine] = None
        self.is_trained: bool = False
        self.stats = {
            "total_generated": 0,
            "auto_passed": 0,
            "user_approved": 0,
            "user_rejected": 0,
            "auto_rejected": 0,
        }

    # ── 초기화 ───────────────────────────────────────────────
    def initialize(self, corpus_text: str, epochs: int = 10,
                   on_progress=None):
        self.sentences = [s.strip() for s in corpus_text.split("\n")
                         if s.strip() and len(s.strip()) > 5]
        if on_progress: on_progress(20, "어휘 구축 중...")
        self._build_vocab()
        if on_progress: on_progress(40, "NeuralMarkov 학습 중...")
        if NM_OK:
            self.nm = NeuralMarkovEngine()
            self.nm.train(corpus_text, embedding_dim=32, epochs=epochs)
        if on_progress: on_progress(80, "SOM 학습 중...")
        self._rebuild_som()
        if on_progress: on_progress(100, "완료")
        self.is_trained = True

    def _build_vocab(self):
        cnt = Counter(t for s in self.sentences for t in tokenize(s))
        self.vocab = {w:i for i,w in enumerate(
            w for w,c in cnt.most_common() if c >= 2)}
        self.V = len(self.vocab)

    def _vec(self, sentence):
        dim = min(self.V, 30) if self.V > 0 else 30
        v = np.zeros(dim)
        for t in tokenize(sentence):
            if t in self.vocab and self.vocab[t] < dim:
                v[self.vocab[t]] += 1.0
        norm = np.linalg.norm(v)
        return v/norm if norm > 1e-12 else v

    def _rebuild_som(self):
        if self.V == 0: return
        vecs = np.array([self._vec(s) for s in self.sentences])
        self.som = SOM(grid=self.grid, dim=vecs.shape[1])
        self.som.train(vecs, self.sentences, epochs=80)

    # ── 개념 조합 생성 (LLM) ─────────────────────────────────
    def _combine_llm(self, sent_a: str, sent_b: str,
                     api_key: str, model: str = "gpt-4o-mini") -> Optional[str]:
        """LLM으로 두 개념을 의미 있게 조합"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = (
                f"다음 두 교육 개념을 자연스럽게 연결한 한 문장을 만들어줘.\n"
                f"개념 A: {sent_a}\n"
                f"개념 B: {sent_b}\n\n"
                f"규칙:\n"
                f"- 한 문장으로만 답해\n"
                f"- 두 개념의 관계나 연결이 드러나야 해\n"
                f"- 교육학적으로 의미 있어야 해\n"
                f"- 문장 부호 없이 끝내\n"
                f"- 예: 'A를 활용한 B 중심 수업은 학생의 능동적 학습을 촉진한다'\n\n"
                f"문장:"
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                max_tokens=80,
                temperature=0.7,
            )
            result = resp.choices[0].message.content.strip()
            result = result.strip("\"'.")
            return result if len(result) > 10 else None
        except Exception:
            return self._combine(sent_a, sent_b)  # 실패 시 템플릿으로 폴백

    def _combine(self, sent_a: str, sent_b: str,
                 pattern_idx: int = 0) -> Optional[str]:
        """템플릿 기반 조합 (LLM 없을 때 폴백)"""
        ta = [t for t in tokenize(sent_a) if t in self.vocab and len(t) > 2]
        tb = [t for t in tokenize(sent_b) if t in self.vocab and len(t) > 2]
        if not ta or not tb: return None
        ka = ta[0]
        kb = next((t for t in tb if t != ka), tb[0])
        if ka == kb and len(ta) > 1: ka = ta[1]
        patterns = [
            f"{ka}과 {kb}의 관계는 교육 실천에서 중요하다",
            f"{kb}를 반영한 {ka} 중심 수업을 설계한다",
            f"{ka}의 원리를 적용하여 {kb}를 실현한다",
            f"{ka}과 {kb}를 통합한 교육과정을 운영한다",
            f"{kb} 관점에서 {ka}의 의미를 재해석한다",
        ]
        return patterns[pattern_idx % len(patterns)]

    # ── 1세대 생성 (CoreAI 1차 검증까지) ─────────────────────
    def generate_candidates(self, max_candidates: int = 10,
                            logp_thr: float = -13.0,
                            api_key: str = "",
                            model: str = "gpt-4o-mini") -> list:
        """
        SOM 빈 자리에서 후보 생성 → CoreAI 1차 검증
        api_key 있으면 LLM으로 생성, 없으면 템플릿 폴백
        """
        if not self.som or not self.is_trained:
            return []

        use_llm = bool(api_key)
        empty = self.som.get_empty()
        candidates = []
        seen = set(self.sentences + [p["sentence"] for p in self.pending])

        for neuron_idx in empty:
            if len(candidates) >= max_candidates:
                break
            neighbors = self.som.get_neighbors(neuron_idx, radius=2)
            if len(neighbors) < 2:
                continue

            sent_a = neighbors[0][1][0]
            sent_b = neighbors[min(1, len(neighbors)-1)][1][0]

            # 생성
            if use_llm:
                new_sent = self._combine_llm(sent_a, sent_b, api_key, model)
            else:
                new_sent = self._combine(sent_a, sent_b,
                                         self.generation % 5)

            if not new_sent or new_sent in seen:
                # LLM 실패 시 템플릿으로 재시도
                for pi in range(5):
                    new_sent = self._combine(sent_a, sent_b, pi)
                    if new_sent and new_sent not in seen:
                        break
                if not new_sent or new_sent in seen:
                    continue

            self.stats["total_generated"] += 1

            # CoreAI 1차 검증
            if self.nm and self.nm.is_trained:
                result = self.nm.evaluate(new_sent, logp_thr=logp_thr)
                status = result.get("status", "FATAL")
                logp   = result.get("avg_logp", -99)
            else:
                status = "PASS"
                logp   = 0.0

            if status in ("PASS", "WARNING"):
                self.stats["auto_passed"] += 1
                candidate = {
                    "sentence":      new_sent,
                    "from_a":        sent_a,
                    "from_b":        sent_b,
                    "coreai_status": status,
                    "logp":          logp,
                    "generation":    self.generation,
                    "neuron":        neuron_idx,
                    "by_llm":        use_llm,
                }
                candidates.append(candidate)
                seen.add(new_sent)
            else:
                self.stats["auto_rejected"] += 1
                self.rejected_auto.append(new_sent)

        self.pending.extend(candidates)
        return candidates

    # ── 사용자 2차 검증 ───────────────────────────────────────
    def user_approve(self, candidate: dict):
        """사용자 승인 → 인덱스 추가"""
        self.sentences.append(candidate["sentence"])
        self.generated.append({**candidate, "user_action": "approved"})
        self.stats["user_approved"] += 1
        # pending에서 제거
        self.pending = [p for p in self.pending
                       if p["sentence"] != candidate["sentence"]]
        # 증분 학습
        self._incremental_train(candidate["sentence"])
        self._rebuild_som()
        self.generation += 1

    def user_reject(self, candidate: dict, reason: str = ""):
        """사용자 거부 → 폐기"""
        self.rejected_user.append({
            **candidate,
            "reason": reason,
            "user_action": "rejected",
        })
        self.stats["user_rejected"] += 1
        self.pending = [p for p in self.pending
                       if p["sentence"] != candidate["sentence"]]

    def _incremental_train(self, sentence: str):
        """새 문장을 NeuralMarkov에 증분 학습"""
        if not self.nm: return
        toks = tokenize(sentence)
        for i,t in enumerate(toks):
            self.nm.uni[t]   += 1
            self.nm.total    += 1
            if i>=1: self.nm.bi[toks[i-1]][t]           += 1
            if i>=2: self.nm.tri[(toks[i-2],toks[i-1])][t] += 1
        self._build_vocab()

    # ── 저장/로드 ─────────────────────────────────────────────
    def to_dict(self) -> dict:
        data = {
            "sentences":      self.sentences,
            "generated":      self.generated,
            "rejected_auto":  self.rejected_auto,
            "rejected_user":  self.rejected_user,
            "pending":        self.pending,
            "generation":     self.generation,
            "stats":          self.stats,
            "grid":           self.grid,
        }
        if self.nm and self.nm.is_trained:
            data["nm"] = {
                "uni":   dict(self.nm.uni),
                "bi":    {k:dict(v) for k,v in self.nm.bi.items()},
                "tri":   {k:dict(v) for k,v in self.nm.tri.items()},
                "total": self.nm.total,
                "alpha": getattr(self.nm, "alpha", 0.001),
            }
        # SOM W 행렬 저장 → 로드 시 재학습 불필요
        if self.som is not None:
            data["som"] = {
                "W":                self.som.W,
                "neuron_sentences": {
                    str(k): v for k,v in self.som.neuron_sentences.items()
                },
                "grid": self.som.grid,
                "dim":  self.som.dim,
            }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "EvolvingIndexEngine":
        ei = cls(grid=data.get("grid", 6))
        ei.sentences     = data["sentences"]
        ei.generated     = data.get("generated", [])
        ei.rejected_auto = data.get("rejected_auto", [])
        ei.rejected_user = data.get("rejected_user", [])
        ei.pending       = data.get("pending", [])
        ei.generation    = data.get("generation", 0)
        ei.stats         = data.get("stats", ei.stats)

        if "nm" in data and NM_OK:
            nm = data["nm"]
            ei.nm = NeuralMarkovEngine()
            ei.nm.uni   = Counter(nm["uni"])
            ei.nm.bi    = defaultdict(Counter,
                            {k:Counter(v) for k,v in nm["bi"].items()})
            ei.nm.tri   = defaultdict(Counter,
                            {k:Counter(v) for k,v in nm["tri"].items()})
            ei.nm.total = nm["total"]
            ei.nm.alpha = nm.get("alpha", 0.001)
            ei.nm.is_trained = True

        ei._build_vocab()

        # SOM 복원 — W 행렬 저장됐으면 재학습 없이 즉시 복원
        if "som" in data:
            sd = data["som"]
            ei.som = SOM(grid=sd["grid"], dim=sd["dim"])
            ei.som.W = sd["W"]
            ei.som.neuron_sentences = {
                int(k): v for k,v in sd["neuron_sentences"].items()
            }
        else:
            # 구버전 pkl → 재학습
            ei._rebuild_som()

        ei.is_trained = True
        return ei

    # ── 통계 ─────────────────────────────────────────────────
    def summary(self) -> dict:
        return {
            "원본 문장": len(self.sentences) - len(self.generated),
            "승인된 생성": len(self.generated),
            "대기 중": len(self.pending),
            "자동 거부": self.stats["auto_rejected"],
            "사용자 거부": self.stats["user_rejected"],
            "총 문장": len(self.sentences),
            "진화 세대": self.generation,
            "빈 자리": len(self.som.get_empty()) if self.som else 0,
        }
