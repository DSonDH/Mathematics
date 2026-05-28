# Berief Introduction to Causal Inference
인과추론 (Causal Inference)은 treatment / policy / intervention 등의 효과를 추정하는 것 이다.

## 심슨의 역설 (Simpson's Paradox)
심슨의 역설이란, 전체 집단에서는 A 처치가 B 처치보다 효과적이지만, 각 하위 집단에서는 B 처치가 A 처치보다 효과적일 때 발생하는 통계적 현상이다.

## Correlation vs Causation (상관과 인과)
상관관계는 두 변수 간의 통계적 연관성(Association)을 나타내며, 인과관계는 한 변수의 변화가 다른 변수의 변화에 직접적인 영향을 미치는 관계를 나타낸다.  
상관관계는 인과관계를 내포할 수 있지만, 상관관계가 있다고 해서 반드시 인과관계가 존재하는 것은 아니다.  

예시: 아이스크림 판매량과 익사 사고 수 사이에는 양의 상관관계가 존재하지만, 이는 여름철에 두 변수가 동시에 증가하기 때문이지, 아이스크림 판매가 익사 사고를 유발하는 것은 아니다.

## What does imply causation? (인과가 의미하는 것은 무엇인가?)
Potential outcomes (잠재적 결과) 프레임워크에서, 인과 효과는 처치가 주어졌을 때의 결과와 처치가 주어지지 않았을 때의 결과 간의 차이로 정의된다.  
인과 효과를 추정하기 위해서는, 처치가 주어졌을 때의 결과와 처치가 주어지지 않았을 때의 결과를 모두 관찰할 수 있어야 한다. 그러나 현실에서는 한 개인에 대해 두 가지 결과를 동시에 관찰할 수 없기 때문에, 인과 효과를 추정하기 위해서는 추가적인 가정이 필요하다.  

예시: A 처치를 받은 환자와 B 처치를 받은 환자의 결과를 비교하여 인과 효과를 추정하려면, 두 그룹이 처치 외의 다른 요인에서 유사하다는 가정이 필요하다 (예: 연령, 성별, 건강 상태 등). 이러한 가정을 충족시키지 못하면, 관찰된 연관성이 실제 인과 효과를 반영하지 않을 수 있다.

인과 추론에서의 주요 도전 과제는 혼동(confounding) 문제이다. 혼동이란, 처치와 결과 사이에 제3의 변수(혼동변수)가 존재하여, 처치와 결과 간의 연관성을 왜곡하는 현상이다. 혼동을 통제하기 위해서는, 혼동변수를 조건부로 설정하거나, 무작위 대조 실험(randomized controlled trial)을 수행하는 등의 방법이 필요하다.

### Fundamental problem of causal inference (인과 추론의 근본적 문제)
인과 추론의 근본적 문제는, 한 개인에 대해 처치가 주어졌을 때의 결과와 처치가 주어지지 않았을 때의 결과를 동시에 관찰할 수 없다는 것이다.  
즉, 잠재적 결과 프레임워크에서, 인과 효과는 다음과 같이 정의된다:

$$\text{Causal Effect} = E[Y(t=1)] - E[Y(t=0)] \neq E[Y|T=1] - E[Y|T=0]$$

- $Y(t=1)$: 처치를 받았을 때의 잠재적 결과
- $Y(t=0)$: 처치를 받지 않았을 때의 잠재적 결과
- $E[Y|T=1]$: 처치를 받은 그룹의 관찰된 평균 결과
- $E[Y|T=0]$: 처치를 받지 않은 그룹의 관찰된 평균 결과
  - $E[Y|T=1]$와 $E[Y|T=0]$는 association difference지 인과 효과가 아니다. 혼동이 존재할 수 있기 때문이다.

Missing data interpretation 이라고도 불리는 문제 (예: 환자가 처치를 거부하거나, 처치 그룹에서 탈락하는 경우), 또는 윤리적 문제로 인해 무작위 대조 실험이 불가능한 경우 등에서 특히 중요하다.

**Ignorability assumption (무시 가능성 가정)**  
인과 효과를 추정하기 위해서는, 처치가 주어졌을 때의 결과와 처치가 주어지지 않았을 때의 결과가 조건부로 독립적이라는 가정이 필요하다.  
즉, 다음과 같은 가정이 필요하다:

$$Y(t) \perp T \mid X$$

- $Y(t)$: 잠재적 결과
- $T$: 처치 변수
- $X$: 혼동변수 집합

그러면 $E[Y(t=1)] - E[Y(t=0)] = E[Y|T=1, X] - E[Y|T=0, X]$가 성립하게 된다.

**Exchangeability assumption (교환 가능성 가정)**  
인과 효과를 추정하기 위해서는, 처치 그룹과 통제 그룹이 혼동변수에 대해 유사하다는 가정이 필요하다.
즉, 다음과 같은 가정이 필요하다:

$$P(X|T=1) = P(X|T=0)$$

- $P(X|T=1)$: 처치 그룹에서 혼동변수의 분포
- $P(X|T=0)$: 통제 그룹에서 혼동변수의 분포

**Identifiability assumption (식별 가능성 가정)**  
인과 효과를 추정하기 위해서는, 관찰된 데이터에서 인과적 효과를 식별할 수 있어야 한다. 즉, 다음과 같은 가정이 필요하다:

$$E[Y(t)] - E[Y(t')] = E[Y|T=t, X] - E[Y|T=t', X]$$

이는 **Causal quantity(인과량)** 이 **Statistical quantity(통계량)** 으로 표현 가능함을 의미한다.

- $E[Y(t)]$: 인과적 효과 (처치 $t$에서의 잠재적 결과의 기댓값) → 관찰 불가능
- $E[Y|T=t, X]$: 통계적 기댓값 (관찰 가능한 데이터에서 추정 가능)
- $X$: 혼동변수 집합

이 가정이 타당한지 확인하는 방법은 무작위 대조 실험(randomized controlled trial, RCT)을 수행하여, 처치 그룹과 통제 그룹이 혼동변수에 대해 유사한지 확인하는 것이다.

**Posiitivity assumption (양의 가정)**  
인과 효과를 추정하기 위해서는, 모든 처치 수준에서 혼동변수의 분포가 양의 확률을 가져야 한다. 즉, 다음과 같은 가정이 필요하다:

$$P(T=t|X) > 0 \quad \text{for all } t \text{ and } X$$

- $P(T=t|X)$: 혼동변수 $X$가 주어졌을 때 처치 $t$를 받을 확률

이는 모든 처치 수준에서 충분한 데이터가 존재해야 함을 의미한다. 예를 들어, 특정 혼동변수 조합에 대해 처치 $t$를 받은 사례가 전혀 없다면, 해당 조합에 대한 인과 효과를 추정할 수 없게 된다. 이때는 Extrapolation (외삽)을 통해 인과 효과를 추정해야 할 수도 있다.

**Consistency assumption (일관성 가정)**  
인과 효과를 추정하기 위해서는, 처치가 주어졌을 때의 결과가 실제로 관찰된 결과와 일치해야 한다. 즉, 다음과 같은 가정이 필요하다:

$$Y = Y(t) \quad \text{if } T=t$$

- $Y$: 실제로 관찰된 결과
- $Y(t)$: 처치 $t$가 주어졌을 때의 잠재적 결과

이는 처치가 주어졌을 때의 결과가 실제로 관찰된 결과와 일치해야 함을 의미한다. 예를 들어, 처치 $t$를 받은 환자의 실제 결과가 $Y$라면, $Y(t)$도 $Y$와 일치해야 한다. 이 가정이 타당하지 않은 경우, 인과 효과를 추정하는 것이 어려워질 수 있다.

### Randomized Controlled Trials (RCTs)와 Observational Studies (관찰 연구)
- RCTs: 처치를 무작위로 할당하여, 혼동을 통제할 수 있는 가장 강력한 방법. 그러나 비용이 많이 들고, 윤리적 문제로 인해 항상 가능하지는 않음.
  - Ethical reasons
  - Infeasibility 
  
- Observational Studies: 처치가 자연스럽게 발생하는 상황에서 데이터를 수집하여 분석하는 방법. 혼동을 통제하기 위해서는, 통계적 방법(예: 회귀 분석, 매칭, 인과 그래프 등)을 사용하여 혼동변수를 조정해야 함.
  - adjust/control for confounders
  - causal graphs (인과 그래프)

### Causation in observational studies (관찰 연구에서의 인과)

#### The Causal Estimand (인과 추정량)

**Causal Effect of Treatment T on Outcome Y**  
처치 T가 결과 Y에 미치는 인과적 영향을 정량화하려면, 잠재적 결과(potential outcomes)를 사용한다.

$$\text{Causal Effect} = E[Y(t=1)] - E[Y(t=0)]$$

- $Y(t=1)$: 처치를 받았을 때의 잠재적 결과
- $Y(t=0)$: 처치를 받지 않았을 때의 잠재적 결과

#### Confounding and Backdoor Paths (혼동과 백도어 경로)

**문제**: 인과 그래프에서 $C \leftarrow \text{Confounder} \rightarrow T \rightarrow Y$ 구조가 있으면, C는 혼동변수(confounder)로 작용한다.

**해결**: 혼동변수를 조건부로 설정하여 백도어 경로를 차단해야 한다.

**인과 그래프 구조**
```
   C (Confounder)
      / \
     ↓   ↓
     T → Y
```

**두 가지 경로 존재**:
1. **Frontdoor 경로 (인과적)**: $T \rightarrow Y$ (직접 인과)
2. **Backdoor 경로 (혼동)**: $T \leftarrow C \rightarrow Y$ (공통원인)

#### 조건부 기댓값을 통한 인과 추정
**공식**  

$$E[Y|\text{do}(T=t)] = \sum_c E[Y|T=t, C=c]P(c)$$

- $E[Y|T=t, C=c]$: 처치와 혼동변수를 모두 고정했을 때의 조건부 기댓값
- $P(c)$: 혼동변수의 주변분포

**테이블: 조건부 기댓값을 이용한 인과 추정**

| 조건 | Mild | Severe | Total | Causal |
|------|------|--------|-------|--------|
| **Treatment A** | $15\%$ (210/1400) | $30\%$ (30/100) | $16\%$ (240/1500) | $\frac{1450}{2050}(0.15) + \frac{600}{2050}(0.30) \approx 0.194$ |
| **Treatment B** | $10\%$ (5/50) | $20\%$ (100/500) | $19\%$ (105/550) | $\frac{1450}{2050}(0.10) + \frac{600}{2050}(0.20) \approx 0.129$ |
| | $E[Y\|t, C=0]$ | $E[Y\|t, C=1]$ | $E[Y\|t]$ | $E[Y\|\text{do}(t)]$ |

**예시 해석** (표의 "Causal" 열)
- Treatment A: $\frac{1450}{2050}(0.15) + \frac{600}{2050}(0.30) \approx 0.194$
  - 경증 조건에서 기댓값 × 확률 + 중증 조건에서 기댓값 × 확률
- Treatment B: $\frac{1450}{2050}(0.10) + \frac{600}{2050}(0.20) \approx 0.129$

**핵심**: 조건부로 설정하지 않으면(Total 열), 혼동 편향이 발생하여 관찰된 연관성이 진정한 인과 효과를 반영하지 못한다.


### 용어 정의

**Estimand (추정량)**
- 인과적 효과와 같이 추정하려는 실제 양
- 이론적 개념으로, 관찰된 데이터에서 직접적으로 계산할 수 없는 경우가 많음
- 예: $E[Y(1)] - E[Y(0)]$ (잠재적 결과에 기반한 인과 효과)

**Estimate (추정값)**
- 추정량을 데이터에서 계산하여 얻은 실제 수치
- 관찰된 데이터를 사용하여 계산한 결과값

**Estimation (추정 과정)**
- 추정량(Estimand)에서 추정값(Estimate)으로 가는 프로세스
- 데이터와 방법론을 통해 이론적 양을 실제 수치로 변환하는 과정

**Identification-Estimation Flowchart**  

```
Causal Estimand → Identification → Statistical Estimand → Estimation → Estimate
```

**단계별 설명:**

1. **Causal Estimand (인과 추정량)**
   - 우리가 추정하고자 하는 인과적 효과의 정의
   - 예: $E[Y(t=1)] - E[Y(t=0)]$

2. **Identification (식별)**
   - 인과 추정량을 관찰 가능한 통계량으로 변환 가능한지 확인
   - 필요한 가정들 검토 (Ignorability, Exchangeability, Positivity, Consistency)
   - 결과: 인과량과 통계량이 같음을 보이는 단계

3. **Statistical Estimand (통계 추정량)**
   - 관찰된 데이터로 계산 가능한 통계량
   - 예: $E[Y|T=t, X] - E[Y|T=t', X]$

4. **Estimation (추정)**
   - 통계 추정량을 계산하기 위한 구체적인 방법 선택
   - 회귀분석, 매칭, IPW (Inverse Probability Weighting) 등

5. **Estimate (최종 추정값)**
   - 실제 데이터에서 계산된 수치
   - 예: 0.15 (15% 효과)


# The Flow of Causation and Association in Graphs

## Graph terminology (그래프 용어)
- Node (노드): 확률변수 또는 개체를 나타냄.
- Edge (엣지): 노드 간의 관계를 나타냄. 방향이 있는 경우, 원인과 결과의 관계를 나타냄.
- Directed Acyclic Graph (DAG): 방향이 있는 엣지로 구성된 그래프이며, 사이클이 없는 구조. 인과 그래프에서 자주 사용됨.
- Path (경로): 노드 간의 연결된 엣지들의 연속.

## Bayesian networks and Causal Graphs (베이지안 네트워크와 인과 그래프)
베이지안 네트워크는 확률적 의존성을 모델링하는 그래프 모델로, 인과 그래프와 유사하지만 인과적 해석이 항상 가능한 것은 아니다.  
베이지안 네트워크에서는 노드 간의 방향성은 인과적 관계를 반드시 나타내는 것은 아니며, 단지 조건부 독립성을 표현하는 수단으로 사용될 수 있다.  
따라서, 베이지안 네트워크에서의 방향성은 인과적 해석이 아니라, 확률적 의존성의 표현으로 이해해야 한다.

### 결합분포의 단순 모델링 (Naive Modeling of Joint Distribution)

통계적 모델링(인과성 없음)에서 결합확률은 다음과 같이 분해된다:

$$P(x_1, x_2, \ldots, x_n) = \prod_i P(x_i \mid x_{1}, \ldots, x_1)$$

예시 (4변수):
$$P(x_1, x_2, x_3, x_4) = P(x_1)P(x_2 \mid x_1)P(x_3 \mid x_2, x_1)P(x_4 \mid x_3, x_2, x_1)$$

그래프 구조:
$$X_1 \rightarrow X_2, \quad X_1 \rightarrow X_3, \quad X_3 \rightarrow X_4, \quad X_2 \rightarrow X_4 \text{ (생략 가능)}$$

이진 변수 4개($x_1, x_2, x_3 \in \{0,1\}$)일 때, $P(x_4 \mid x_3, x_2, x_1)$을 표현하려면 $2^{n-1} = 8$개의 파라미터($\alpha_1, \ldots, \alpha_8$)가 필요하다.

> **핵심 문제**: 변수가 늘어날수록 필요한 파라미터 수가 지수적으로 증가한다 → $2^{n-1}$ parameters!  
> 이것이 인과 그래프(DAG)를 활용한 조건부 독립 가정이 필요한 이유다.

### Local Markov Assumption (지역 마르코프 가정)
각 노드는 자신의 부모 노드들이 주어졌을 때, 부모가 아닌 모든 비후손(non-descendant) 노드들과 조건부 독립이다.

$$X_i \perp \text{Non-Descendants}(X_i) \mid \text{Parents}(X_i)$$

이 가정 덕분에 결합분포를 다음과 같이 **부모 노드에 대한 조건부 확률의 곱**으로 단순화할 수 있다:

$$P(x_1, x_2, \ldots, x_n) = \prod_i P(x_i \mid \text{parents}(x_i))$$

예시 ($X_1 \to X_2,\ X_1 \to X_3,\ X_3 \to X_4$):

$$P(x_1, x_2, x_3, x_4) = P(x_1)P(x_2 \mid x_1)P(x_3 \mid x_1)P(x_4 \mid x_3)$$

| 변수 | 부모 | 필요한 파라미터 수 |
|------|------|------------------|
| $X_1$ | 없음 | 1 |
| $X_2$ | $X_1$ | 2 |
| $X_3$ | $X_1$ | 2 |
| $X_4$ | $X_3$ | 2 |

> **핵심 이점**: 조건부 독립 구조를 활용하면 파라미터 수가 지수적($2^{n-1}$)에서 **선형적**으로 줄어든다.

### Bayesian network factorization (베이지안 네트워크 인수분해)
베이지안 네트워크에서는 결합분포가 각 노드의 부모에 대한 조건부 확률의 곱으로 인수분해된다: 
$$P(x_1, x_2, \ldots, x_n) = \prod_i P(x_i \mid \text{parents}(x_i))$$

Local Markov Assumption과 동치인 이 인수분해는 베이지안 네트워크의 핵심 특징으로, 그래프 구조가 확률적 의존성을 어떻게 표현하는지를 보여준다.

**Minimality Assumption (최소성 가정)**:  
모델이 표현하고자 하는 실제 데이터의 조건부 독립(Conditional Independence) 관계를 가장 적은 수의 간선(Edge)을 사용하여 표현한다고 가정하는 원칙  
즉, 주어진 확률 분포 
를 표현하는 방향성 비순환 그래프(DAG) 
가 있을 때, 간선(Edge)을 하나라도 제거하면 더 이상 
의 모든 조건부 독립 관계를 만족할 수 없게 되는 가장 '희소한(Sparse)' 구조

1. 부모가 주어지면, 각 노드는 부모가 아닌 모든 노드들과 조건부 독립이다 (Local Markov Assumption).
2. 인접 노드들은 서로 독립이 아니다.

위 내용들은 statistical 내용이지, causal 내용은 아니다.  

Cause: 변수 X의 변화가 변수 Y의 변화에 직접적인 영향을 미치는 경우, 즉, X가 Y의 원인이라 한다.  
Directed graph에서는 모든 부모가 자식의 원인이라 가정한다.  

DAG + Local Markov Assumption + Minimality Assumption → Causal DAG (인과 DAG)

## The basic building blocks of graphs (그래프의 기본 구성 요소)
chain (사슬), fork (분기), collider (Immorality, 충돌부) 구조가 있음.

Association: 두 변수 간의 통계적 의존성.  
X1 → X2 → X3에서 X1과 X3는 연관이 있지만, 인과적 관계는 X1 → X2 → X3 뿐이다.  
fork 구조에서는 X1과 X3는 연관이 있지만, 인과적 관계는 X2 -> X1, X2 -> X3 뿐이다. 
X2를 조건부로 주면 X1과 X3는 독립이 된다. 이를 block된 경로라고도 하고, d-separation이라고도 한다. 조건부가 아닌 경우는 association이 flow하는 열린 경로라고도 한다.

collider 구조에서는 X1과 X3는 연관이 없지만, X2를 조건부로 주면 X1과 X3는 연관이 생긴다.  

**Proof of conditional independence in chains (사슬에서 조건부 독립성 증명)**  

$$T \to M \to Y$$

T와 Y는 연관이 있지만, M을 조건부로 주면 T와 Y는 독립이 된다.

$$T \perp Y \mid M$$

**증명:**  
베이지안 네트워크 인수분해에 의해:

$$P(T, M, Y) = P(T)P(M \mid T)P(Y \mid M)$$

M이 주어졌을 때 T와 Y의 조건부 확률:

$$P(T, Y \mid M) = \frac{P(T, M, Y)}{P(M)}$$

Bayesian network factorization에 의해:

$$=\frac{P(T)P(M \mid T)P(Y \mid M)}{P(M)}$$

$P(Y \mid M)$을 제외한 항으로 Bayes' rule을 적용하면:

$$= P(T \mid M) \cdot P(Y \mid M) \\
\therefore P(T, Y \mid M) = P(T \mid M) P(Y \mid M)$$

즉 $T \perp Y \mid M$ 이 성립한다.

**Proof of conditional independence in forks (분기에서 조건부 독립성 증명)**

$$
\begin{array}{ccccc}
 & & X & & \\
 & \swarrow & & \searrow & \\
 T & & & & Y
\end{array}
$$

T와 Y는 연관이 있지만, X를 조건부로 주면 T와 Y는 독립이 된다.

$$T \perp Y \mid X$$

**증명:**  
베이지안 네트워크 인수분해에 의해:

$$P(T, X, Y) = P(X)P(T \mid X)P(Y \mid X)$$

X가 주어졌을 때 T와 Y의 조건부 확률:

$$P(T, Y \mid X) = \frac{P(T, X, Y)}{P(X)} = \frac{P(X)P(T \mid X)P(Y \mid X)}{P(X)} = P(T \mid X)P(Y \mid X)$$

따라서 $T \perp Y \mid X$ 이 성립한다.

**Proof of conditional independence in colliders (충돌부에서 조건부 독립성 증명)**

$$
\begin{array}{ccccc}
 T & & & & Y \\
 & \searrow & & \swarrow & \\
 & & X & & \\
\end{array}
$$

T와 Y는 독립이지만, X를 조건부로 주면 T와 Y는 연관이 생긴다.

$$T \not\perp Y \mid X$$

**증명:**  
베이지안 네트워크 인수분해에 의해:

$$P(T, X, Y) = P(T)P(Y)P(X \mid T, Y)$$

X가 주어졌을 때 T와 Y의 조건부 확률:

$$P(T, Y \mid X) = \frac{P(T, X, Y)}{P(X)} = \frac{P(T)P(Y)P(X \mid T, Y)}{P(X)}$$

이 표현은 T와 Y가 X에 의해 연결되어 있음을 보여준다. 즉, T와 Y는 X를 조건부로 주면 독립이 아니다. (P(T | X)P(Y | X)식으로 분해할 수 없기 때문)

**Conditioning on descendants of colliders (충돌부의 자손에 조건부 설정)**

$$X_1 \not\!\perp X_3 \mid X_4$$

$$
\begin{array}{ccccc}
 X_1 & & & & X_3 \\
 & \searrow & & \swarrow & \\
 & & X_2 & & \\
 & & \downarrow & & \\
 & & X_4 & &
\end{array}
$$

충돌부(collider) $X_2$ 자체가 아니라, 그 자손인 $X_4$를 조건부로 설정해도 $X_1$과 $X_3$ 사이에 연관성이 생긴다.

즉, 충돌부의 **자손(descendant)** 에 조건부를 두는 것도 충돌부 자체에 조건부를 두는 것과 유사한 효과를 낸다.

**증명**  
베이지안 네트워크 인수분해에 의해:

$$P(X_1, X_2, X_3, X_4) = P(X_1)P(X_3)P(X_2 \mid X_1, X_3)P(X_4 \mid X_2)$$

$X_4$가 주어졌을 때 $X_1$과 $X_3$의 조건부 확률:

$$P(X_1, X_3 \mid X_4) = \frac{P(X_1, X_2, X_3, X_4)}{P(X_4)} = \frac{P(X_1)P(X_3)P(X_2 \mid X_1, X_3)P(X_4 \mid X_2)}{P(X_4)}$$

이 표현은 $X_1$과 $X_3$가 $X_4$에 의해 연결되어 있음을 보여준다. 즉, $X_1$과 $X_3$는 $X_4$를 조건부로 주면 독립이 아니다. (P(X_1 | X_4)P(X_3 | X_4)식으로 분해할 수 없기 때문)

## the flow of association and causation in graphs (그래프에서 연관성과 인과의 흐름)
그래프에서 연관성과 인과의 흐름을 이해하는 것은 인과 추론에서 매우 중요하다.

### Blocked path
정의: 그래프에서 특정 노드 집합에 조건부로 두었을 때, 두 변수 간의 경로가 차단되는 경우.

경로는 다음 조건 중 하나를 만족하면 **차단(blocked)** 된다:
1. 경로 위에 **non-collider(chain/fork)** 가 있고, 그 노드가 조건부 집합에 포함된 경우.
2. 경로 위에 **collider** 가 있고, 그 노드와 그 자손(descendant) 모두 조건부 집합에 포함되지 않은 경우.

반대로, 모든 경로가 차단되지 않으면 두 변수는 **연관(associated)** 되어 있다고 한다.

### d-separation
정의: 조건부 집합 $Z$가 주어졌을 때, 두 변수 집합 $X$와 $Y$ 사이의 모든 경로가 차단되면, $X$와 $Y$는 $Z$에 의해 **d-separated** 되었다고 한다.

### Theorem (d-separation implies conditional independence)
Given that P is Markov to G and P is faithful to G, then:

$$X \perp\!\!\!\perp_G Y \mid Z \implies X \perp\!\!\!\perp_P Y \mid Z$$

- G: 그래프에서 d-separation이 성립한다는 것을 의미한다.
- P: 데이터 분포에서의 조건부 독립이 성립한다는 것을 의미한다.
- 즉, 그래프에서 d-separation이 성립하면, 결합 분포에서도 조건부 독립이 성립한다 (faithfulness 가정 하에).

**Global Markov Property (글로벌 마르코프 성질)**  
그래프 G가 P에 마르코프적이면, 그래프에서 d-separation이 성립하는 모든 조건부 독립이 P에서도 성립한다.  

Markov assumption: local Markov property와 global Markov property는 서로 동치이다.

### 정리

**연관성(Association)의 흐름**
- 연관성은 **양방향**으로 흐름 (인과 방향과 무관)
- 위 그림에서 $T$와 $Y$ 사이에는 두 가지 경로로 연관성이 흐름:
  1. **직접 인과 경로**: $T \rightarrow Y$
  2. **공통원인(fork) 경로**: $T \leftarrow X \rightarrow Y$ (backdoor path)

**가정들의 연쇄**

$$\text{Markov Assumption} \xrightarrow{\text{implies}} \text{Statistical Independencies}$$

$$\xrightarrow{+\text{ Minimality Assumption}} \text{Statistical Dependencies}$$

$$\xrightarrow{+\text{ Causal Edges Assumption}} \text{Causal Dependencies}$$

| 가정 | 의미 |
|------|------|
| **Markov Assumption** | 부모가 주어지면 비후손과 조건부 독립 → 통계적 독립성 도출 |
| **Minimality Assumption** | 인접 노드는 독립이 아님 → 통계적 의존성 보존 |
| **Causal Edges Assumption** | 그래프의 방향성이 실제 인과관계를 나타냄 → 인과적 의존성 도출 |

> **핵심**: 세 가정이 모두 성립할 때, 그래프의 d-separation 구조가 실제 데이터의 인과적 독립성과 일치한다.

## 독립적 인과 메커니즘과 희소 메커니즘 시프트

### 독립적 인과 메커니즘 (Independent Causal Mechanism, ICM) 원칙

**구조적 인과 모델 (SCM)을 통한 형식적 정의**  
시스템이란, 변수 집합 $\{X_1, X_2, \ldots, X_n\}$과 이들 간의 인과적 관계로 구성된 연결 구조로 표현된다:

$$X_i = f_i(\text{PA}_i, U_i)$$

PA: parent variables (부모 변수)  
U: exogenous noise (외생적 잡음)  

혹은 분포 형태로:

$$P(X_1, \ldots, X_n) = \prod_i P(X_i \mid \text{PA}_i)$$

**ICM의 핵심 가정**  
각 조건부 분포 $P(X_i \mid \text{PA}_i)$는 다음을 만족하는 *자율적 메커니즘(autonomous mechanism)* 이다:  
메커니즘이란, 시스템의 특정 부분을 정의하는 함수적 또는 분포적 단위로, 다음과 같은 특성을 가진다:
- 다른 메커니즘과 독립적으로 정의됨
- 개입 시 다른 메커니즘과 독립적으로 변함

**중요한 구분: 모듈성 독립과 통계적 독립**  
모듈: 각 $P(X_i \mid \text{PA}_i)$는 시스템의 독립적인 단위로 작동하는 부분.

ICM의 "독립성"은 **통계적 독립성이 아니다**:
- ❌ 아님: $X_i \perp X_j$ (변수들의 통계적 독립)
- ❌ 아님: $P(X_i) \perp P(X_j)$ (주변분포의 독립)
- ✅ **맞음**: 메커니즘 수준의 모듈성 독립 (modular independence)
  - $P(X_i \mid \text{PA}_i)$를 수정해도 $i \neq j$인 $P(X_j \mid \text{PA}_j)$의 정의나 함수 형태는 변하지 않음

**ICM의 세 가지 구성 요소**

1. **모듈성 (Modularity)**: 각 인과 메커니즘은 독립적인 단위로 작동
   - 메커니즘 $i$의 변화는 메커니즘 $j$의 구조를 변화시키지 않음
   
2. **개별 개입 가능성 (Separate Intervenability)**: 각 변수를 독립적으로 개입 가능
   - $\text{do}(X_i = x_i)$ 수행 시, 오직 $P(X_i \mid \text{PA}_i)$만 대체됨
   - 다른 요소 $P(X_j \mid \text{PA}_j)$는 불변
   
3. **메커니즘 불변성 (Mechanism Invariance)**: 메커니즘은 문맥 간 안정적
   - $P(X_i \mid \text{PA}_i)$는 특정 종류의 개입(do-intervention)에 대해서만 불변
   - 모든 distribution shift에서 불변 ❌

### 희소 메커니즘 시프트 (Sparse Mechanism Shift, SMS)

**정의**  
데이터 생성 메커니즘이 서로 다른 문맥 간에 변할 때, 전체 시스템이 아닌 희소한 메커니즘 부분집합만 변한다는 가정

**수학적 표현**  
$\mathcal{M} = \{1, 2, \ldots, n\}$을 모든 메커니즘의 지표라 하면, SMS 하에서:
$$|\{i : P_{\text{문맥 1}}(X_i \mid \text{PA}_i) \neq P_{\text{문맥 2}}(X_i \mid \text{PA}_i)\}| \ll n$$

**중요성**  
1. **정보 추출**: 모든 메커니즘이 동시에 변하면, 개입 데이터는 인과 발견에 활용할 불변 구조가 남지 않음.

2. **국소화된 변화**: 실제 분포 변화는 특정 메커니즘에 국한됨 (예: 처치는 결과 메커니즘에만 영향, 혼동변수 메커니즘은 변하지 않음).

3. **알고리즘적 독립성**: 문맥 간 변하지 않는 메커니즘은 이론적으로 정보를 공유하지 않음:
   - 콜모고로프 복잡도(Kolmogorov complexity) 관점: 메커니즘은 시스템 엔트로피에 독립적으로 기여
   - $K(M_i) + K(M_j) \approx K(M_i, M_j)$ (메커니즘이 겹치지 않을 때)
     - 이론적으로는 맞지만 매우 강한 가정임

**ICM과의 연결**  
SMS는 ICM의 경험적 구현이다: 메커니즘이 모듈화되고 자율적이기 때문에, 실제 시스템의 변화는 메커니즘 경계를 존중하며—임의의 변수 부분집합이 아닌 특정 자율적 단위만 시프트된다.

