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