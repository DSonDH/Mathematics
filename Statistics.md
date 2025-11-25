# ⭐ 1. 정의 (Neyman–Fisher Factorization 기준)

확률모형 (f(x\mid \theta)) 에서
통계량 (T(X)) 가 **충분(sufficient)** 하다는 것은:

> **T(X)를 알면 X의 나머지 부분은 더 이상 θ에 관한 정보를 추가로 주지 않는다.**

즉,

[
f(x\mid \theta) = g(T(x),\theta),h(x)
]

꼴로 분해할 수 있을 때,
(T(X)) 는 θ에 대한 sufficient statistic이다.
(Fisher–Neyman factorization theorem)

# ⭐ 2. 조건부 확률 관점 정의 (더 직관적)

통계량 (T(X)) 가 충분하다는 것은:

[
P(X \mid T(X), \theta)
]

이 **θ에 의존하지 않는다**는 뜻이다.

즉,

> T(X)가 주어지면 원 데이터 X는 더는 θ에 대한 정보를 포함하지 않는다.

# ⭐ 3. 정보 관점 (Information-theoretic)

충분성은 다음과 같이 표현할 수도 있다:

[
I(\theta; X) = I(\theta; T(X)).
]

즉,
**X가 가진 θ에 관한 정보량을 T(X)가 100% 유지한다.**

# ⭐ 4. 왜 p(X)=P(T=1|X) 가 sufficient statistic인가?

인과추론에서는 파라미터 θ 는 “처치할당 모델”을 의미하고,
(T) 는 처치(binary treatment)이다.

처치 모델은 보통 다음과 같이 정의된다:

[
P(T=1\mid X) = p(X).
]

여기서 p(X)는 **X가 T에 영향을 주는 모든 정보를 압축한 1차원 요약값**이다.

따라서:

[
P(T\mid X) = P(T\mid p(X)).
]

이는 sufficient statistic 정의와 동일하다.

즉,

> **p(X)는 X가 T를 예측하는 데 필요한 모든 정보를 충분히 포함하며,
> p(X)를 알고 있으면 X는 T에 관한 추가 정보를 주지 않는다.**

# ⭐ 5. 정리로 표현

### 정리 (Rosenbaum & Rubin, 1983 — Balancing Score Theorem)

* 만약 처치할당 모델이
  [
  P(T=1\mid X)=p(X)
  ]
  으로 주어진다면,

[
T \perp X \mid p(X),
]

즉 **p(X)는 T에 대한 sufficient statistic이 된다.**

# ⭐ 6. 직관적 해석

* X를 알고 T의 확률을 계산한다 → p(X)
* 그런데 p(X)를 알고 있으면
  X의 “잔여 정보”는 T에 더 이상 영향을 미치지 못한다
* 따라서 p(X)는 X의 정보를 100% 보존하면서 1차원으로 요약한 sufficient statistic이다.

# ⭐ 7. 인과추론에서 sufficient statistic의 역할

p(X) 가 sufficient statistic이기 때문에:

1. **X 조건화와 p(X) 조건화가 인과적으로 동일한 효과를 낸다.**

[
(Y_0,Y_1)\perp T\mid X \quad\Longrightarrow\quad (Y_0,Y_1)\perp T\mid p(X)
]

2. 따라서 p(X)를 이용하면
   고차원 X를 직접 통제하지 않고도 균형(balance)을 달성한다.

3. 이것이 성향점수 기반 기법(IPW, matching)이
   고차원 문제를 1차원 문제로 축소할 수 있는 이유이다.

# ⭐ 요약

* sufficient statistic은 **파라미터에 관한 정보를 100% 보존하는 통계량**이다.
* 인과추론에서 p(X)=P(T=1|X) 는 T에 대한 **충분통계량(sufficient statistic)** 이다.
* 따라서 **X를 직접 통제하지 않고 p(X)에만 조건화해도 confounding이 제거**된다.
* 이것이 성향점수 방법의 수학적 근거다.
