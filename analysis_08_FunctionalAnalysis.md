# 함수해석학 *(Functional Analysis)*

# 1. 노름 공간 *(Normed Space)*

## Def. [노름] *(Norm)*
벡터 공간 $X$ over $\mathbb{R}$ (또는 $\mathbb{C}$) 위의 함수 $\|\cdot\| : X \to [0, \infty)$가 다음을 만족하면 **노름**이라 한다.

1. $\|x\| = 0 \iff x = 0$
2. $\|\alpha x\| = |\alpha| \|x\|$ &emsp; ($\forall \alpha \in \mathbb{R},\ x \in X$)
3. $\|x + y\| \leq \|x\| + \|y\|$ &emsp; (삼각부등식)

노름이 주어진 벡터 공간 $(X, \|\cdot\|)$을 **노름 공간** *(normed space)* 이라 한다.

### 유도 거리 *(Induced Metric)*
노름 공간에서 $d(x, y) := \|x - y\|$로 정의하면 거리 공간이 된다.

---

# 2. 바나흐 공간 *(Banach Space)*

## Def. [바나흐 공간] *(Banach Space)*
노름 공간 $(X, \|\cdot\|)$이 유도 거리에 대해 **완비** *(complete)* 이면, 즉

$$\{x_n\} \text{ 이 코시 수열} \implies \exists x \in X \text{ s.t. } x_n \to x$$

이면 $X$를 **바나흐 공간**이라 한다.

$$\boxed{\text{바나흐 공간} = \text{노름 공간} + \text{완비성}}$$

- 완비성이 없으면 $\mathbb{Q}$처럼 "구멍"이 생긴다.
- 실수의 완비성 공리(→ analysis_01)가 추상화된 것.

## 주요 예시

| 공간 | 노름 | 바나흐? |
|------|------|:-------:|
| $\mathbb{R}^n$ | $\|x\|_2 = \sqrt{\sum x_i^2}$ | ✅ |
| $\ell^p\ (1 \leq p \leq \infty)$ | $\|x\|_p = \left(\sum_{i=1}^\infty |x_i|^p\right)^{1/p}$ | ✅ |
| $L^p(\mu)\ (1 \leq p \leq \infty)$ | $\|f\|_p = \left(\int |f|^p\, d\mu\right)^{1/p}$ | ✅ |
| $C([a,b])$ | $\|f\|_\infty = \sup_{x} |f(x)|$ | ✅ |
| $C([a,b])$에 $L^2$ 노름 부여 | $\|f\|_2$ | ❌ (불완비) |

---

# 3. 힐베르트 공간 *(Hilbert Space)*

## Def. [내적] *(Inner Product)*
벡터 공간 $X$ 위의 함수 $\langle \cdot, \cdot \rangle : X \times X \to \mathbb{R}$이 다음을 만족하면 **내적**이라 한다.

1. $\langle x, x \rangle \geq 0$, &ensp; $\langle x, x \rangle = 0 \iff x = 0$
2. $\langle x, y \rangle = \langle y, x \rangle$
3. $\langle \alpha x + \beta y, z \rangle = \alpha\langle x, z \rangle + \beta\langle y, z \rangle$

내적은 자연스럽게 노름을 유도한다: $\|x\| := \sqrt{\langle x, x \rangle}$

## Def. [힐베르트 공간] *(Hilbert Space)*
내적 공간 $(X, \langle\cdot,\cdot\rangle)$이 유도 노름에 대해 완비이면 **힐베르트 공간**이라 한다.

## 공간들의 포함 관계

```
내적 공간
    ↓  + 완비성
힐베르트 공간  ⊂  바나흐 공간  ⊂  노름 공간  ⊂  거리 공간  ⊂  위상 공간
```

- 힐베르트 공간은 반드시 바나흐 공간이다.
- 역은 성립하지 않는다: $\ell^1$은 바나흐이지만 힐베르트가 아님.
- 노름이 내적에서 유도되는지 판별: **평행사변형 법칙** $\|x+y\|^2 + \|x-y\|^2 = 2(\|x\|^2 + \|y\|^2)$ 만족 $\iff$ 내적 유도 가능.

---

# 4. 핵심 정리들

## Thm. [한-바나흐 정리] *(Hahn-Banach Theorem)*
$X$가 노름 공간, $Y \subset X$ 부분 공간, $f: Y \to \mathbb{R}$ 유계 선형 범함수이면,
$\|F\| = \|f\|$를 만족하는 확장 $F: X \to \mathbb{R}$이 존재한다.

- 쌍대 공간 $X^*$가 충분히 풍부함을 보장한다.

## Thm. [균등 유계 원리] *(Uniform Boundedness Principle / Banach-Steinhaus)*
$X$가 바나흐 공간, $Y$가 노름 공간, $\{T_\alpha\} \subset \mathcal{B}(X, Y)$ 연속 선형 연산자族이면,

$$\sup_\alpha \|T_\alpha x\| < \infty\ (\forall x \in X) \implies \sup_\alpha \|T_\alpha\| < \infty$$

- 점별 유계이면 균등 유계.

## Thm. [열린 사상 정리] *(Open Mapping Theorem)*
$X, Y$가 바나흐 공간, $T: X \to Y$가 전사 연속 선형 연산자이면, $T$는 열린 사상이다.

**따름정리.** $T$가 전단사이면 $T^{-1}$도 연속, 즉 $T$는 위상동형사상.

## Thm. [닫힌 그래프 정리] *(Closed Graph Theorem)*
$X, Y$가 바나흐 공간, $T: X \to Y$가 선형 연산자이면,

$$\text{그래프 } \Gamma(T) = \{(x, Tx)\} \text{가 } X \times Y \text{에서 닫힌 집합} \iff T \text{가 연속}$$
