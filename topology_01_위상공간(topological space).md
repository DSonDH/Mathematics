# Chapter 1 위상공간 (Topological Space)
## 1.1 위상공간 (Topology, Topological Space)

### 정의 1.1: 위상, 위상공간 (topology, topological space)
집합 $X$에 대한 위상(topology) $\mathscr{T}$는 다음 세 조건을 만족하는 $X$의 부분집합족이다:
1. $\emptyset, X \in \mathscr{T}$
2. 임의의 $\mathcal{U} \subset \mathscr{T}$에 대해 $\bigcup_{U \in \mathcal{U}} U \in \mathscr{T}$ (합집합에 대해 닫혀있음)
3. 유한개의 $U_1, U_2, \ldots, U_n \in \mathscr{T}$에 대해 $U_1 \cap U_2 \cap \cdots \cap U_n \in \mathscr{T}$ (유한교집합에 대해 닫혀있음)

순서쌍 $(X, \mathscr{T})$를 위상공간(topological space)이라 하고, $\mathscr{T}$의 원소를 열린집합(open set)이라 한다.

### 정리 1.1: 열린집합의 점별 특성화 (pointwise characterization)
$G \in \mathscr{T}$ $\Leftrightarrow$ $\forall x \in G$, $\exists H \in \mathscr{T}$ s.t. $x \in H \subset G$

즉, 집합이 열린집합일 필요충분조건은 각 점이 그 점을 포함하고 집합에 완전히 포함되는 열린집합을 가지는 것이다.

### 정의 1.2: 비이산 위상과 이산 위상 (indiscrete and discrete topology)
- 비이산 위상(indiscrete topology): $\mathscr{T} = \{\emptyset, X\}$
- 이산 위상(discrete topology): $\mathscr{T} = 2^X$ (모든 부분집합이 열린집합)

### 정리 1.2: 이산 위상의 판정 (criterion for discreteness)
$\mathscr{T}$가 이산위상 $\Leftrightarrow$ 모든 $x \in X$에 대해 $\{x\} \in \mathscr{T}$

### 정의 1.3: 여유한 위상과 여가산 위상 (finite complement and cocountable topology)
- 여유한 위상 $\mathscr{T}_f$: $\mathscr{T}_f = \{\emptyset\} \cup \{U \subset X : X \setminus U \text{는 유한}\}$
- 여가산 위상 $\mathscr{T}_c$: $\mathscr{T}_c = \{\emptyset\} \cup \{U \subset X : X \setminus U \text{는 가산}\}$

#### 예제 1.1
정의 1.3의 두 명제는 X상의 위상임을 보여라.

### 정리 1.3
- 유한집합 X상의 여유한위상 $\mathscr{T}$는 이산위상
- 가산집합 X상의 여가산위상 $\mathscr{T}$는 이산위상

#### 증명
...

### 정의 1.4: 보통 위상과 하한 위상 (usual topology and lower limit topology)
- 보통 위상 $\mathscr{T}_u$: 열린구간 $(a,b)$들을 기저로 하는 위상
   $$\mathscr{T}_u = \{U \subset \mathbb{R} : \forall x \in U, \exists a,b \in \mathbb{R}, a < b \text{ s.t. } x \in (a,b) \subset U\}$$
- 하한 위상 $\mathscr{T}_l$ (Sorgenfrey line): 반열린구간 $[a,b)$들을 기저로 하는 위상. 여기서 $\mathscr{T}_l$은 $\mathscr{T}_u$보다 더 세밀하다 ($\mathscr{T}_u \subset \mathscr{T}_l$).
   $$\mathscr{T}_l = \{U \subset \mathbb{R} : \forall x \in U, \exists a,b \in \mathbb{R}, a < x < b \text{ s.t. } [x,b) \subset U\}$$

#### 예제 1.2
정의 1.4의 두 명제는 $\mathbb{R}$상의 위상임을 보여라.

#### 예제 1.3
$\mathbb{R}$의 보통위상 $\mathscr{T}_u$에서 $\mathbb{Q}$를 포함하는 열린집합을 찾으시오. 특히 $\mathbb{Q}$를 포함하지만 $\mathbb{R}$과 다른 열린집합이 존재함을 보이시오.

**풀이**: 
$\mathbb{Q}$를 포함하는 열린집합으로는 $\mathbb{R}$ 자신이 있다. 그러나 $\mathbb{Q}$를 포함하면서 $\mathbb{R}$이 아닌 열린집합도 존재한다.

예를 들어, 각 무리수 $\alpha \in \mathbb{R} \setminus \mathbb{Q}$에 대해, 점 $\alpha$를 중심으로 하는 충분히 작은 열린구간 $(a, b)$ (단, $a < \alpha < b$, $\alpha - a < 2^{-n}$, $b - \alpha < 2^{-n}$)를 생각할 수 있다.

더 구체적으로: $U = \bigcup_{q \in \mathbb{Q}} (q - \varepsilon_q, q + \varepsilon_q)$를 적절히 선택하면 ($\varepsilon_q > 0$), $\mathbb{Q} \subset U \subsetneq \mathbb{R}$인 열린집합을 구성할 수 있다.

또는 간단하게: $U = \mathbb{R} \setminus \{\alpha_1, \alpha_2, \ldots\}$ (가산개의 무리수들을 제거)는 열린집합이 아니지만, 유사한 구성으로 $\mathbb{Q}$를 포함하는 진부분집합 열린집합을 만들 수 있다. 

실제로 더 직접적인 예: 서로 다른 두 무리수 $\alpha < \beta$에 대해 $U = \mathbb{R} \setminus [\alpha, \beta]$ (또는 $U = (-\infty, \alpha) \cup (\beta, \infty)$)는 열린집합이고 $\mathbb{Q}$를 포함하면서 $\mathbb{R}$과 다르다.

#### 예제 1.4
$\mathbb{R}$상의 보통위상 $\mathscr{T}_u$, 하한 위상 $\mathscr{T}_l$에 대해 $\mathscr{T}_u \subsetneq \mathscr{T}_l$임을 보이시오. 특히 $\mathscr{T}_u$에 속하지만 $\mathscr{T}_l$에는 속하지 않는 집합이 존재함을 보이시오.

**풀이**:

먼저 $\mathscr{T}_u \subset \mathscr{T}_l$임을 보이자. $U \in \mathscr{T}_u$이면, 임의의 $x \in U$에 대해 $x \in (a,b) \subset U$인 열린구간 $(a,b)$가 존재한다. 그러면 $[x, b) \supset (a, b)$이므로 $[x, b)$의 일부가 $U$에 포함된다. 더 정확히, $x \in [x, c) \subset U$인 $c > x$를 택할 수 있으므로 $U \in \mathscr{T}_l$이다.

다음으로 $\mathscr{T}_l \not\subset \mathscr{T}_u$임을 보이자. 반열린구간 $[0, 1)$을 생각해보자. 임의의 $x \in [0, 1)$에 대해 $[x, 1) \subset [0, 1)$이므로 $[0, 1) \in \mathscr{T}_l$이다.

그러나 $[0, 1) \notin \mathscr{T}_u$이다. 왜냐하면 점 $0$을 포함하는 어떤 열린구간 $(a, b)$도 $[0, 1)$에 포함될 수 없기 때문이다. ($a < 0$이어야 $(a, b)$가 $0$을 포함하는데, 그러면 $(a, b) \not\subset [0, 1)$).

따라서 $\mathscr{T}_u \subsetneq \mathscr{T}_l$이다.

#### 예제 1.5
$\mathbb{R}$의 보통위상 $\mathscr{T}_u$에서 열린집합의 임의의 교집합이 항상 열린집합이 아님을 보이시오.

보통위상에서 교집합에 대해 닫혀있는 것은 **유한개**의 교집합뿐이다. 임의의(무한개) 교집합은 열린집합이 아닐 수 있다.

**반례**: 각 자연수 $n \in \mathbb{N}$에 대해 $U_n = (-1/n, 1/n)$은 열린구간이므로 $U_n \in \mathscr{T}_u$이다.

그러나 이들의 교집합은:
$$\bigcap_{n=1}^{\infty} U_n = \bigcap_{n=1}^{\infty} (-1/n, 1/n) = \{0\}$$

한점집합 $\{0\}$은 열린집합이 아니다. (점 0을 포함하며 $\{0\}$에 완전히 포함되는 열린구간은 존재하지 않기 때문)

따라서 무한개의 열린집합의 교집합은 닫힌집합이 될 수 있으며, 이는 위상의 정의에서 유한교집합만 보장하는 이유를 보여준다.

### 정리 1.4: 함수에 의해 유도된 위상 (topology induced by a function)
함수 $f: X \to (Y, \mathscr{T}_Y)$가 주어질 때, 
$$\mathscr{T}_X = \{f^{-1}(G) : G \in \mathscr{T}_Y\}$$
는 $X$ 위의 위상이다. 이를 $f$에 의해 유도된 위상이라 한다.

**참고**: $X$가 위상공간이고 $f: (X, \mathscr{T}_X) \to Y$가 전사함수일 때, 
$$\mathscr{T}_Y = \{U \subset Y : f^{-1}(U) \in \mathscr{T}_X\}$$
로 정의된 $\mathscr{T}_Y$도 $Y$ 위의 위상이며, 이를 $f$가 만드는 상의 위상이라 한다.

#### 증명

#### 예제 1.6
실수 $\mathbb{R}$의 보통위상을 $\mathscr{T}$라 하고, 함수 $f: \mathbb{R} \to \mathbb{R}$를
$$f(x) = \begin{cases} x & \text{if } x \in \mathbb{Q} \\ -x & \text{if } x \in \mathbb{Q}^c \end{cases}$$
로 정의하자. $\mathscr{T}_1 = \{f^{-1}(G) : G \in \mathscr{T}\}$는 $\mathbb{R}$ 위의 위상이다. $A = (-1, 1)$과 $B = (1, 3)$이 $\mathscr{T}_1$의 원인가?

**풀이**:

$f^{-1}(G) \in \mathscr{T}_1$ $\Leftrightarrow$ $G \in \mathscr{T}$이므로, $A, B \in \mathscr{T}_1$인지 판정하려면 $A = f^{-1}(G_A)$, $B = f^{-1}(G_B)$인 $G_A, G_B \in \mathscr{T}$가 존재하는지 확인해야 한다.

**$A = (-1, 1)$에 대해**:

$f^{-1}((-1,1))$를 계산하자. $x \in f^{-1}((-1,1))$ $\Leftrightarrow$ $f(x) \in (-1,1)$ $\Leftrightarrow$ $-1 < f(x) < 1$.

- $x \in \mathbb{Q}$이면, $-1 < x < 1$ $\Leftrightarrow$ $x \in (-1, 1) \cap \mathbb{Q}$
- $x \in \mathbb{Q}^c$이면, $-1 < -x < 1$ $\Leftrightarrow$ $-1 < x < 1$ $\Leftrightarrow$ $x \in (-1, 1) \cap \mathbb{Q}^c$

따라서 $f^{-1}((-1,1)) = (-1,1)$이고, $(-1,1) \in \mathscr{T}$이므로 $A = (-1,1) \in \mathscr{T}_1$. **✓**

**$B = (1, 3)$에 대해**:

$f^{-1}((1,3))$를 계산하자. $x \in f^{-1}((1,3))$ $\Leftrightarrow$ $1 < f(x) < 3$.

- $x \in \mathbb{Q}$이면, $1 < x < 3$
- $x \in \mathbb{Q}^c$이면, $1 < -x < 3$ $\Leftrightarrow$ $-3 < x < -1$

따라서 $f^{-1}((1,3)) = (1,3) \cap \mathbb{Q} \cup (-3,-1) \cap \mathbb{Q}^c$.

이는 $(1,3)$과 다르므로 직접적인 역상 표현으로는 $(1,3) \notin \mathscr{T}_1$이다. **✗**

### 정리 1.5: 함수에 의해 유도된 위상 (quotient topology)
위상공간 $(X, \mathscr{T}_X)$와 집합 $Y$에 대해 함수 $f: X \to Y$가 주어질 때,
$$\mathscr{T}_Y = \{G \subset Y : f^{-1}(G) \in \mathscr{T}_X\}$$
로 정의된 $\mathscr{T}_Y$는 $Y$ 위의 위상이다. 이를 $f$에 의해 유도된 위상(또는 상위상)이라 한다.

#### 증명
$\mathscr{T}_Y$가 위상의 세 조건을 만족함을 보이자.

1. $f^{-1}(\emptyset) = \emptyset \in \mathscr{T}_X$이고 $f^{-1}(Y) = X \in \mathscr{T}_X$이므로 $\emptyset, Y \in \mathscr{T}_Y$.

2. $\{G_\alpha\}_{\alpha \in I} \subset \mathscr{T}_Y$이면, 각 $\alpha$에 대해 $f^{-1}(G_\alpha) \in \mathscr{T}_X$이다. 따라서 $f^{-1}(\bigcup_{\alpha \in I} G_\alpha) = \bigcup_{\alpha \in I} f^{-1}(G_\alpha) \in \mathscr{T}_X$이므로 $\bigcup_{\alpha \in I} G_\alpha \in \mathscr{T}_Y$.

3. $G_1, G_2, \ldots, G_n \in \mathscr{T}_Y$이면, 각 $i$에 대해 $f^{-1}(G_i) \in \mathscr{T}_X$이다. 따라서 $f^{-1}(\bigcap_{i=1}^n G_i) = \bigcap_{i=1}^n f^{-1}(G_i) \in \mathscr{T}_X$이므로 $\bigcap_{i=1}^n G_i \in \mathscr{T}_Y$.

그러므로 $\mathscr{T}_Y$는 $Y$ 위의 위상이다. ∎

#### 예제 1.7
함수 $f: \mathbb{R} \to \mathbb{R}$를 $f(x) = x - [x]에 대하여 $\mathscr{T}_1 = \{G \subset \mathbb{R} : f^{-1}(G) \in \mathscr{T} \}$로 정의된 위상에서 집합 $A = (-1, 1/2)$와 $B = (1/2, 2)$이 $\mathscr{T}_1$의 원소인가?

**풀이**:
$f^{-1}(A)$와 $f^{-1}(B)$를 계산하자.
- $f^{-1}(A) = \{x \in \mathbb{R} : f(x) \in (-1, 1/2)\} = \{x : x - [x] \in (-1, 1/2)\}$
  
이는 $[x]$가 정수이므로, $x$의 소수부분이 $(-1, 1/2)$에 속하는 모든 실수 $x$를 의미한다. 즉, $f^{-1}(A) = \bigcup_{n \in \mathbb{Z}} (n, n + 1/2)$이다. 이 집합은 $\mathscr{T}$의 열린집합이므로 $A \in \mathscr{T}_1$이다. **✓**

- $f^{-1}(B) = \{x \in \mathbb{R} : f(x) \in (1/2, 2)\} = \{x : x - [x] \in (1/2, 2)\}$

이는 $x$의 소수부분이 $(1/2, 1)$에 속하는 모든 실수 $x$를 의미한다. 즉, $f^{-1}(B) = \bigcup_{n \in \mathbb{Z}} (n + 1/2, n + 1)$이다. 이 집합도 $\mathscr{T}$의 열린집합이므로 $B \in \mathscr{T}_1$이다. **✓**  

## 1.2 닫힌집합, 집적점, 폐포 (Closed Set, Limit Point, Closure)

### 정의 1.5: 닫힌집합 (closed set)
집합 $F \subset X$가 닫힌집합(closed set)이라는 것은 $X \setminus F$가 열린집합인 것을 의미한다.

### 정리 1.6: 닫힌집합의 기본 성질 (basic properties of closed sets)
1. $\emptyset, X$는 닫힌집합이다.
2. 임의의 닫힌집합족의 교집합은 닫힌집합이다: $\{F_\alpha\}_{\alpha \in I}$이 모두 닫힌집합이면 $\bigcap_{\alpha \in I} F_\alpha$도 닫힌집합.
3. 유한개의 닫힌집합의 합집합은 닫힌집합이다.

#### 예제
여유한, 여가산위상에서 닫힌집합이 될 필요충분조건?

#### 예제
닫힌집합의 임의의 합집합은 닫힌집합이 아니다. 예시?

### 정의 1.6: 집적점과 유도집합 (limit point and derived set)
점 $p \in X$가 집합 $A \subset X$의 집적점(limit point)이라는 것은, $p$를 포함하는 모든 열린집합 $G$에 대해 $(G \setminus \{p\}) \cap A \neq \emptyset$인 것을 의미한다.

집합 $A$의 모든 집적점들의 집합을 $A'$로 표기하고 유도집합(derived set)이라 한다.

### 정의 1.7: 고립점 (isolated point)
점 $x \in A$가 $A$의 고립점이라는 것은 $\{x\}$가 열린집합이고, $x$를 포함하는 어떤 열린집합 $G$에 대해 $(G \setminus \{x\}) \cap A = \emptyset$인 것을 의미한다.

#### 예제
위상공간 $(X, \mathscr{T})$에서 $x$가 고립점이면 모든 $A \subset X$에 대해 $x \notin A'$임을 보이시오.

또한 $\mathbb{R}$의 위상 $\mathscr{T} = \{(a, \infty) : a \in \mathbb{R}\} \cup \{\emptyset, \mathbb{R}\}$에 대해 $\mathbb{Q}'$를 구하시오.

**첫 번째 명제**: $x$가 $A$의 고립점이면, $\{x\}$를 포함하는 열린집합 $G$가 존재하여 $(G \setminus \{x\}) \cap A = \emptyset$이다. 따라서 $(G \setminus \{x\}) \cap A = \emptyset$이므로 $x$는 $A$의 집적점이 아니다. 즉, $x \notin A'$이다. ∎

**두 번째 문제**: $\mathbb{Q}'$를 구하기 위해 각 점이 집적점인지 판정하자.

주어진 위상에서 열린집합은 $(a, \infty)$ 형태 또는 $\emptyset, \mathbb{R}$이다.

점 $p \in \mathbb{R}$이 $\mathbb{Q}$의 집적점이려면, $p$를 포함하는 모든 열린집합 $G$에 대해 $(G \setminus \{p\}) \cap \mathbb{Q} \neq \emptyset$이어야 한다.

- **$p \in \mathbb{Q}$인 경우**: $p$를 포함하는 열린집합은 $(a, \infty)$ (단, $a < p$) 또는 $\mathbb{R}$이다. 각 경우에 $(a, \infty) \cap \mathbb{Q}$는 공집합이 아니므로 (더 큰 유리수들이 존재), $p$는 집적점이다. 따라서 모든 $p \in \mathbb{Q}$에 대해 $p \in \mathbb{Q}'$이다.

- **$p \notin \mathbb{Q}$인 경우**: $p$를 포함하는 열린집합은 $(a, \infty)$ (단, $a < p$) 또는 $\mathbb{R}$이다. 임의의 $a < p$에 대해 $(a, \infty) \cap \mathbb{Q} \neq \emptyset$이므로, $p$도 집적점이다. 따라서 모든 $p \notin \mathbb{Q}$에 대해서도 $p \in \mathbb{Q}'$이다.

그러므로 $\mathbb{Q}' = \mathbb{R}$이다. **✓**

### 정리 1.7: 닫힌집합의 집적점 판정법 (closed set criterion via derived set)
$F$가 닫힌집합 $\Leftrightarrow$ $F' \subset F$

#### 증명


### 정의 1.8: 폐포 (closure)
집합 $A \subset X$의 폐포(closure) $\overline{A}$는 $A$를 포함하는 모든 닫힌집합의 교집합으로 정의된다:
$$\overline{A} = \bigcap \{F : A \subset F, F \text{는 닫힌집합}\}$$

### 정리 1.8: 폐포의 성질 (properties of closure)
1. $\overline{A}$는 닫힌집합이다.
2. $\overline{A}$는 $A$를 포함하는 최소 닫힌집합이다: $A \subset \overline{A}$이고, $A \subset F$인 모든 닫힌집합에 대해 $\overline{A} \subset F$.
3. $A = \overline{A}$ $\Leftrightarrow$ $A$가 닫힌집합.
4. $\overline{A} = A \cup A'$

#### 예제 1.8
집합 $X$에 여유한 또는 여가산 위상이 주어질 경우, $X$의 부분집합 $A$에 대하여 $\overline{A}$의 폐포를 구하시오.

**풀이**:

**경우 1: 여유한 위상 $\mathscr{T}_f$**

여유한 위상에서 닫힌집합은 $X$ 전체이거나 여유한집합(complement가 유한)이다.

$A$를 포함하는 닫힌집합들을 찾자:
- $X$는 항상 $A$를 포함하는 닫힌집합이다.
- 여유한 위상에서 $A$를 포함하는 다른 닫힌집합 $F$가 있다면, $X \setminus F$는 유한이다.

$\overline{A}$는 $A$를 포함하는 모든 닫힌집합의 교집합이므로:
- $A$가 유한이면, $\overline{A} = A$ (유한집합 자신이 닫혀있음)
- $A$가 무한이면, $\overline{A} = X$ (무한집합을 포함하는 최소 닫힌집합은 $X$)

따라서: $$\overline{A} = \begin{cases} A & \text{if } A \text{는 유한} \\ X & \text{if } A \text{는 무한} \end{cases}$$

**경우 2: 여가산 위상 $\mathscr{T}_c$**

여가산 위상에서 닫힌집합은 $X$ 전체이거나 여가산집합(complement가 가산)이다.

마찬가지로 분석하면:
- $A$가 가산이면, $\overline{A} = A$ (가산집합 자신이 닫혀있음)
- $A$가 비가산이면, $\overline{A} = X$ (비가산집합을 포함하는 최소 닫힌집합은 $X$)

따라서: $$\overline{A} = \begin{cases} A & \text{if } A \text{는 가산} \\ X & \text{if } A \text{는 비가산} \end{cases}$$

### 정리 1.9: 폐포의 특성화 (characterization of closure)
$X$를 위상공간, $A \subset X$를 부분집합이라 하자. $x \in \overline{A}$이기 위한 필요충분조건은 $x$를 포함하는 임의의 열린집합 $G$에 대하여 $G \cap A \neq \emptyset$이 되는 것이다.

#### 증명
$(\Rightarrow)$ $x \in \overline{A}$이고 $x$를 포함하는 열린집합 $G$가 있다고 하자. 만약 $G \cap A = \emptyset$이면 $A \subset X \setminus G$이고 $X \setminus G$는 닫힌집합이다. 그런데 $\overline{A}$는 $A$를 포함하는 최소 닫힌집합이므로 $\overline{A} \subset X \setminus G$이다. 이는 $x \in \overline{A}$를 포함하는 $x$를 포함하는 열린집합 $G$가 존재한다는 것에 모순이다. 따라서 $G \cap A \neq \emptyset$이다.

$(\Leftarrow)$ $x$를 포함하는 모든 열린집합 $G$에 대해 $G \cap A \neq \emptyset$이라 하자. $x \notin \overline{A}$이면 $x \in X \setminus \overline{A}$인데, $X \setminus \overline{A}$는 열린집합이고 $x$를 포함한다. 그런데 $\overline{A}$의 정의에 의해 $(X \setminus \overline{A}) \cap A = \emptyset$이므로 모순이다. 따라서 $x \in \overline{A}$이다. ∎

#### 예제
$\mathbb{R}$의 하한위상 $\mathscr{T}_l$에서 $\mathbb{Q}'$를 구하시오.

**풀이**:
하한 위상 $\mathscr{T}_l$에서 열린집합은 반열린구간 $[a,b)$들을 기저로 한다.

점 $p \in \mathbb{R}$이 $\mathbb{Q}$의 집적점이려면, $p$를 포함하는 모든 열린집합 $G$에 대해 $(G \setminus \{p\}) \cap \mathbb{Q} \neq \emptyset$이어야 한다.

$p$를 포함하는 기본 열린집합은 $[p, c)$ 형태이다 (단, $c > p$). 이 경우:

$([p, c) \setminus \{p\}) \cap \mathbb{Q} = (p, c) \cap \mathbb{Q}$

임의의 $c > p$에 대해 $(p, c) \cap \mathbb{Q} \neq \emptyset$이므로 모든 $p \in \mathbb{R}$이 $\mathbb{Q}$의 집적점이다.

따라서 $\mathbb{Q}' = \mathbb{R}$. **✓**

### 정의 1.9: 조밀부분집합 (dense subset)
위상공간 $(X, \mathscr{T})$의 부분집합 $D$가 조밀부분집합이라는 것은 $\overline{D} = X$인 것을 의미한다.

**동치 조건**: $D$가 $X$의 조밀부분집합 $\Leftrightarrow$ $X$의 모든 공집합이 아닌 열린집합이 $D$와 교집합을 가진다. 즉, 모든 $U \in \mathscr{T}$, $U \neq \emptyset$에 대해 $U \cap D \neq \emptyset$이다.

**예시**: 
- $\mathbb{Q}$는 $\mathbb{R}$의 보통위상에서 조밀부분집합이다.
- $\mathbb{R} \setminus \mathbb{Q}$ (무리수 집합)도 $\mathbb{R}$의 보통위상에서 조밀부분집합이다.

### 정리 1.10: 폐포의 유도집합 표현 (closure via derived set)
위상공간 $(X, \mathscr{T})$와 $A \subset X$에 대하여 다음이 성립한다:
$$\overline{A} = A \cup A'$$

#### 증명
$(\supset)$ $A \subset \overline{A}$는 폐포의 정의에서 자명하다. $A'$의 정의에 의해 $A' \subset \overline{A}$임을 보이자. $p \in A'$이면, $p$를 포함하는 모든 열린집합 $G$에 대해 $(G \setminus \{p\}) \cap A \neq \emptyset$이고, 따라서 $G \cap A \neq \emptyset$이다. 정리 1.9에 의해 $p \in \overline{A}$이다. 그러므로 $A \cup A' \subset \overline{A}$이다.

$(\subset)$ $\overline{A}$는 $A$를 포함하는 닫힌집합이므로, $\overline{A}$가 모든 집적점을 포함함을 보이면 된다. $p \in \overline{A}$라 하자. $p \in A$이면 $p \in A \cup A'$이다. $p \notin A$이고 $p \notin A'$이라고 가정하면, $p$를 포함하는 어떤 열린집합 $G$가 존재하여 $(G \setminus \{p\}) \cap A = \emptyset$이다. 따라서 $G \cap A = \emptyset$이므로 $A \subset X \setminus G$이고, $X \setminus G$는 닫힌집합이다. 그런데 $\overline{A}$는 $A$를 포함하는 최소 닫힌집합이므로 $\overline{A} \subset X \setminus G$이다. 이는 $p \in G$이고 $p \in \overline{A}$에 모순이다. 따라서 $p \in A \cup A'$이다. ∎


## 1.3 내부·경계·외부 (Interior, Boundary, Exterior)
### 정의 1.9: 내부 (interior), 경계와 외부 (boundary and exterior)
집합 $A \subset X$의 내부(interior) $\mathrm{int}(A)$ 또는 $A^\circ$는 $A$에 포함되는 모든 열린부분집합의 합집합으로 정의된다:
$$\mathrm{int}(A) = \bigcup \{U : U \subset A, U \in \mathscr{T}\}$$

집합 $A$의 경계(boundary) $\partial(A)$는:
$$\partial(A) = \overline{A} \setminus \mathrm{int}(A)$$

외부(exterior) $\mathrm{ext}(A)$는:
$$\mathrm{ext}(A) = \mathrm{int}(X \setminus A)$$

### 정리 1.11: 내부의 특성화 (characterization of interior)
$X$를 위상공간, $A \subset X$를 부분집합이라 하자. $x \in \mathrm{int}(A)$이기 위한 필요충분조건은 $x$를 포함하는 임의의 열린집합 $G$에 대하여 $G \cap A \neq \emptyset$이 되는 것이다.

#### 증명
$(\Rightarrow)$ $x \in \mathrm{int}(A)$이면, 정의에 의해 $x \in U \subset A$인 열린집합 $U$가 존재한다. $x$를 포함하는 임의의 열린집합 $G$에 대해 $G \cap U$도 열린집합이고 $x$를 포함하므로, $G \cap U \neq \emptyset$이다. $U \subset A$이므로 $G \cap A \neq \emptyset$이다.

$(\Leftarrow)$ $x$를 포함하는 모든 열린집합 $G$에 대해 $G \cap A \neq \emptyset$이라 하자. $x \notin \mathrm{int}(A)$이면, $x$를 포함하는 모든 열린집합이 $A$와 교집합을 가지면서도 $A$에 완전히 포함되지 않는다. 그러면 $X \setminus A$도 $x$를 포함하는 열린집합과 교집합을 가져야 하는데, 이는 $x$를 포함하는 모든 열린집합이 $A$와 교집합을 가진다는 것에 모순이다. 따라서 $x \in \mathrm{int}(A)$이다. ∎

#### 예제

#### 예제 1.9
$\mathbb{R}$ 위의 위상 $\mathscr{T} = \{(a, \infty) : a \in \mathbb{R}\} \cup \{\emptyset, \mathbb{R}\}$에 대하여 b(\mathbb{Q})를 구하시오.
**풀이**:
먼저 $\overline{\mathbb{Q}}$를 구하자. 임의의 $x \in \mathbb{R}$에 대해, $x$를 포함하는 열린집합은 $(a, \infty)$ (단, $a < x$) 또는 $\mathbb{R}$이다. 각 경우에 $(a, \infty) \cap \mathbb{Q} \neq \emptyset$이므로 모든 $x \in \mathbb{R}$이 $\mathbb{Q}$의 집적점이다. 따라서 $\overline{\mathbb{Q}} = \mathbb{R}$이다.

### 정리 1.12: 내부의 성질 (properties of interior)
1. $\mathrm{int}(A)$는 열린집합이다.
2. $\mathrm{int}(A)$는 $A$에 포함되는 최대 열린집합이다.
  - 즉 G가 $A$에 포함되는 열린집합이면 $G \subset \mathrm{int}(A) \subset A$.
3. $A$가 열린집합 $\Leftrightarrow$ $A = \mathrm{int}(A)$

#### 증명

#### 예제 1.10
여유한 위상 $\mathscr{T}_f$에서 $\mathbb{Q}$의 내부, 외부, 경계를 구하시오.

**풀이**:

여유한 위상 $\mathscr{T}_f$에서 $\mathbb{Q}$의 내부, 외부, 경계를 구하자.

**Step 1: $\overline{\mathbb{Q}}$ 구하기**

예제 1.8에서 보았듯이, 여유한 위상에서 $\mathbb{Q}$는 무한집합이므로 $\overline{\mathbb{Q}} = \mathbb{R}$.

**Step 2: $\mathrm{int}(\mathbb{Q})$ 구하기**

$\mathrm{int}(\mathbb{Q})$는 $\mathbb{Q}$에 포함되는 모든 열린집합의 합집합이다.

여유한 위상에서 $\emptyset$이 아닌 열린집합은 여유한집합(complement가 유한)이므로, $\mathbb{Q}$에 포함되는 열린집합 중 공집합이 아닌 것은 존재하지 않는다. (유한개를 제외한 무리수를 포함하기 때문)

따라서 $\mathrm{int}(\mathbb{Q}) = \emptyset$.

**Step 3: $\mathrm{ext}(\mathbb{Q})$ 구하기**

$\mathrm{ext}(\mathbb{Q}) = \mathrm{int}(\mathbb{R} \setminus \mathbb{Q})$

$\mathbb{R} \setminus \mathbb{Q}$ (무리수 집합)도 $\mathbb{Q}$와 마찬가지로 무한집합이므로, 같은 논리에 의해 $\mathrm{int}(\mathbb{R} \setminus \mathbb{Q}) = \emptyset$.

따라서 $\mathrm{ext}(\mathbb{Q}) = \emptyset$.

**Step 4: $\partial(\mathbb{Q})$ 구하기**

$$\partial(\mathbb{Q}) = \overline{\mathbb{Q}} \setminus \mathrm{int}(\mathbb{Q}) = \mathbb{R} \setminus \emptyset = \mathbb{R}$$

**답**:
- $\mathrm{int}(\mathbb{Q}) = \emptyset$
- $\mathrm{ext}(\mathbb{Q}) = \emptyset$
- $\partial(\mathbb{Q}) = \mathbb{R}$


### 정리 1.13: 폐포, 내부, 경계의 관계 (closure/interior/boundary relation)
$$\overline{A} = \mathrm{int}(A) \cup b(A)$$

또한 $X = \mathrm{int}(A) \cup b(A) \cup \mathrm{ext}(A)$ (분할)

#### 증명

## 1.4 수렴열 (Convergent Sequence)

### 정의 1.11: 점열의 수렴 (convergence of sequence)
위상공간 $(X, \mathscr{T})$에서 수열 $(x_n)$이 점 $x$로 수렴한다고 하는 것은, $x$를 포함하는 모든 열린집합(근방)이 결국 수열의 모든 항을 포함하는 것을 의미한다. 즉:

$x_n \to x$ $\Leftrightarrow$ $x$를 포함하는 모든 열린집합 $G$에 대해, $\exists N \in \mathbb{N}$ s.t. $n \geq N$이면 $x_n \in G$

#### 예제 1.11
$\mathbb{R}$에 여가산위상이 주어져 있을 때, 수열 $(x_n)$이 수렴할 필요충분조건은 유한한 항 이후의 항은 모두 같은 값일 때이다.

**증명**:  
$(\Rightarrow)$ $x_n \to x$라 하고, 귀류법으로 $\{x_n : n \in \mathbb{N}\}$이 무한집합이라 가정하자. 그러면 $\{x_n : n \in \mathbb{N}\} \setminus \{x\}$도 무한집합이다. 여가산위상의 정의에 의해 $U = \mathbb{R} \setminus (\{x_n : n \in \mathbb{N}\} \setminus \{x\})$는 여가산이므로 열린집합이다. $x \in U$이고 $x_n \to x$이므로, 충분히 큰 $n$에 대해 $x_n \in U$이어야 한다. 그런데 $U$의 정의에 의해 $x_n \in U$이면 $x_n = x$이어야 한다. 따라서 유한한 항을 제외한 모든 항이 $x$와 같다.

$(\Leftarrow)$ 유한한 항 이후의 항이 모두 $x$와 같다고 하자. $x$를 포함하는 열린집합 $G$가 주어지면, $G$는 $\mathbb{R} \setminus G$가 가산이라는 뜻이므로, 충분히 큰 $n$에 대해 $x_n = x \in G$이다. 따라서 $x_n \to x$이다.


## 1.5 기저·부분기저·국소기저 (Basis, Subbasis, Local Basis)

### 정의 1.12: 기저 (basis)
$\mathcal{B} \subset \mathscr{T}$가 위상 $\mathscr{T}$의 기저(basis)라는 것은:  
모든 열린집합이 $\mathcal{B}$의 원소들의 합집합으로 표현된다.

**동치 조건**: $\mathcal{B} \subset \mathscr{T}$이고, 임의의 $G \in \mathscr{T}$와 $x \in G$에 대해 $x \in B \subset G$인 $B \in \mathcal{B}$가 존재한다.

#### 예제 1.12
공집합이 아닌 집합 $X$에 이산위상이 주어져 있을 때 기저 $\mathcal{B}$를 구하시오.

**풀이**:  
이산위상 $\mathscr{T} = 2^X$에서는 $X$의 모든 부분집합이 열린집합이다.

기저의 정의에 의해, $\mathcal{B}$는 모든 열린집합을 $\mathcal{B}$의 원소들의 합집합으로 표현해야 한다.

$\mathcal{B} = \{\{x\} : x \in X\}$임을 보이자.

**Step 1**: 각 $\{x\}$는 하나의 원소를 가지는 집합이므로 $\{x\} \in \mathscr{T}$이다.

**Step 2**: 임의의 열린집합 $U \in \mathscr{T}$에 대해, $U = \bigcup_{x \in U} \{x\}$로 표현할 수 있다. 즉, $U$는 $\mathcal{B}$의 원소들의 합집합이다.

**Step 3**: 기저의 동치 조건을 확인하면, 임의의 $U \in \mathscr{T}$와 $x \in U$에 대해 $x \in \{x\} \subset U$인 $\{x\} \in \mathcal{B}$가 존재한다.

따라서 **$\mathcal{B} = \{\{x\} : x \in X\}$**는 이산위상의 기저이다.

#### 예제 1.13
다음을 보이시오:

(1) $\{(a,b) : a,b \in \mathbb{R}\}$는 $\mathbb{R}$의 보통위상의 기저이고, $\{[a,b) : a,b \in \mathbb{R}\}$는 하한위상의 기저이다.

**풀이**:

**보통위상의 기저**:  
$\mathcal{B}_u = \{(a,b) : a,b \in \mathbb{R}, a < b\}$가 보통위상의 기저임을 보이자.

- 각 $(a,b)$는 열린구간이므로 보통위상에 속한다.
- 임의의 열린집합 $U \in \mathscr{T}_u$와 $x \in U$에 대해, 정의 1.4에 의해 $x \in (a,b) \subset U$인 열린구간 $(a,b)$가 존재한다.

따라서 기저의 동치조건을 만족하므로 $\mathcal{B}_u$는 보통위상의 기저이다. ✓

**하한위상의 기저**:  
$\mathcal{B}_l = \{[a,b) : a,b \in \mathbb{R}, a < b\}$가 하한위상의 기저임을 보이자.

- 각 $[a,b)$는 반열린구간이므로 하한위상에 속한다.
- 임의의 열린집합 $U \in \mathscr{T}_l$과 $x \in U$에 대해, 정의 1.4에 의해 $x \in [x,b) \subset U$인 반열린구간 $[x,b)$가 존재한다. 이는 $\mathcal{B}_l$의 원소이다.

따라서 기저의 동치조건을 만족하므로 $\mathcal{B}_l$은 하한위상의 기저이다. ✓

(2) $\{(a,b) : a,b \in \mathbb{Q}\}$는 $\mathbb{R}$의 보통위상의 기저이지만, $\{[a,b) : a,b \in \mathbb{Q}\}$는 하한위상의 기저가 아니다.

**풀이**:  
**보통위상의 기저**:  
$\mathcal{B}'_u = \{(a,b) : a,b \in \mathbb{Q}, a < b\}$가 보통위상의 기저임을 보이자.

임의의 열린집합 $U \in \mathscr{T}_u$와 $x \in U$에 대해, 어떤 열린구간 $(c,d) \subset U$이 존재하여 $x \in (c,d)$이다. 유리수의 조밀성에 의해 $c < a < x < b < d$인 유리수 $a, b$가 존재한다. 따라서 $x \in (a,b) \subset (c,d) \subset U$인 $(a,b) \in \mathcal{B}'_u$가 존재한다.

그러므로 $\mathcal{B}'_u$는 보통위상의 기저이다. ✓

**하한위상의 기저가 아님**:

$\mathcal{B}'_l = \{[a,b) : a,b \in \mathbb{Q}, a < b\}$가 하한위상의 기저가 아님을 보이자.

점 $x = \sqrt{2}$와 이를 포함하는 열린집합 $[\sqrt{2}, 2)$를 생각하자. $[\sqrt{2}, 2) \in \mathscr{T}_l$이다.

기저의 동치조건에 의해, $x \in [a,b) \subset [\sqrt{2}, 2)$인 $[a,b) \in \mathcal{B}'_l$이 존재해야 한다. 그런데 $\mathcal{B}'_l$의 원소는 $a,b \in \mathbb{Q}$이어야 하므로, $a < \sqrt{2} < b < 2$인 $a,b \in \mathbb{Q}$를 선택하면 $[a,b) \not\subset [\sqrt{2}, 2)$이다. (왜냐하면 $[a,b)$는 $a$를 포함하고 $a < \sqrt{2}$이기 때문)

따라서 기저의 동치조건을 만족하지 않으므로 $\mathcal{B}'_l$은 하한위상의 기저가 아니다. ✗


#### 예제 1.14
$\mathcal{B}_Y$가 위상공간 $(Y, \mathscr{T}_Y)$의 기저라 하면, $\mathcal{B} = \{f^{-1}(B) : B \in \mathcal{B}_Y\}$는 함수 $f: X \to (Y, \mathscr{T}_Y)$에 의하여 생성된 $X$상의 위상 $\mathscr{T}_1 = {f^{-1}(G) \mid G\in \mathscr{T}_Y}$에 대한 기저임을 증명하시오.

**풀이**:


### 정리 1.14: 기저로부터 위상 생성
공집합이 아닌 집합 $X$의 부분집합족 $\mathcal{B}$가 다음 두 조건을 만족하면, $\mathcal{B}$의 원소들의 임의의 합집합을 모아 놓은 집합족 $\mathscr{T}$는 $X$ 상의 위상이 되고, $\mathcal{B}$는 $\mathscr{T}$에 대한 기저이다:

(1) $\bigcup_{B \in \mathcal{B}} B = X$

(2) $B_1, B_2 \in \mathcal{B}$이고 $x \in B_1 \cap B_2$에 대하여, $x \in B \subset B_1 \cap B_2$인 $B \in \mathcal{B}$가 존재한다.

#### 증명
$\mathscr{T} = \{U \subset X : U = \bigcup_{\alpha \in I} B_\alpha, B_\alpha \in \mathcal{B}, I \text{는 첨수집합}\}$로 정의하자.

**위상의 세 조건 확인:**  
1. **$\emptyset, X \in \mathscr{T}$**:
   - 공집합: 공합집합으로 정의하면 $\emptyset \in \mathscr{T}$
   - $X$: 조건 (1)에 의해 $X = \bigcup_{B \in \mathcal{B}} B \in \mathscr{T}$

2. **합집합에 대해 닫혀있음**:
   $\{U_\alpha\}_{\alpha \in I} \subset \mathscr{T}$이면, 각 $U_\alpha = \bigcup_{\beta \in I_\alpha} B_{\alpha\beta}$ (단, $B_{\alpha\beta} \in \mathcal{B}$)로 표현되므로,
   $$\bigcup_{\alpha \in I} U_\alpha = \bigcup_{\alpha \in I} \bigcup_{\beta \in I_\alpha} B_{\alpha\beta} \in \mathscr{T}$$

3. **유한교집합에 대해 닫혀있음**:
   $U_1, U_2 \in \mathscr{T}$라 하자. $U_1 \cap U_2$의 임의의 점 $x$에 대해, $x \in B_1 \subset U_1$인 $B_1 \in \mathcal{B}$와 $x \in B_2 \subset U_2$인 $B_2 \in \mathcal{B}$가 존재한다. 조건 (2)에 의해, $x \in B \subset B_1 \cap B_2 \subset U_1 \cap U_2$인 $B \in \mathcal{B}$가 존재한다.
   
   따라서 $U_1 \cap U_2 = \bigcup_{x \in U_1 \cap U_2} B_x$ (단, $B_x \in \mathcal{B}$)로 표현되므로 $U_1 \cap U_2 \in \mathscr{T}$.

그러므로 $\mathscr{T}$는 $X$ 상의 위상이고, 정의에 의해 $\mathcal{B}$는 $\mathscr{T}$의 기저이다. ∎

#### 예제 1.15
집합 $X = \{a, b, c\}$에 대하여 $\mathcal{S} = \{\{a\}, \{b\}\}$의 원소들의 임의의 합집합을 모아 놓은 집합은 위상이 아님을 보이시오.

**풀이**:  
$\mathcal{S} = \{\{a\}, \{b\}\}$의 원소들의 임의의 합집합으로 이루어진 집합족을 $\mathscr{T}$라 하자:
$$\mathscr{T} = \{\emptyset, \{a\}, \{b\}, \{a,b\}\}$$

$\mathscr{T}$가 위상이 되려면 세 조건을 만족해야 한다.

**조건 1 확인**: $\emptyset, X \in \mathscr{T}$인가?
- $\emptyset \in \mathscr{T}$ ✓ (공합집합)
- $X = \{a,b,c\} \notin \mathscr{T}$ ✗ (공집합이 아닌 열린집합은 $\{a\}, \{b\}$만 가능하므로 최대 $\{a,b\}$까지만 만들어짐)

따라서 조건 1을 만족하지 않는다.

**결론**: $\mathscr{T}$는 위상이 아니다. 왜냐하면 $X$ 자신이 위상에 포함되어야 한다는 기본 조건을 만족하지 않기 때문이다. ∎

### 정리 1.15: 두 위상의 포함 관계 판정 (comparison of topologies via basis)
위상 $\mathscr{T}$와 $\mathscr{T}'$에 대한 기저를 각각 $\mathcal{B}$와 $\mathcal{B}'$라 할 때, $\mathscr{T} \subset \mathscr{T}'$이기 위한 필요충분조건은 임의의 $B \in \mathcal{B}$와 임의의 $x \in B$에 대하여 적당한 $B' \in \mathcal{B}'$이 존재해서 $x \in B' \subset B$가 되는 것이다.

#### 증명
$(\Rightarrow)$ $\mathscr{T} \subset \mathscr{T}'$이고 $B \in \mathcal{B}$, $x \in B$라 하자. $\mathcal{B}$는 $\mathscr{T}$의 기저이므로 $B \in \mathscr{T}$이다. 따라서 $B \in \mathscr{T}'$이다. $\mathcal{B}'$이 $\mathscr{T}'$의 기저이고 $x \in B$이므로, 기저의 동치조건에 의해 $x \in B' \subset B$인 $B' \in \mathcal{B}'$가 존재한다.

$(\Leftarrow)$ 조건을 만족한다고 하자. $U \in \mathscr{T}$임을 보이자. $U$는 $\mathcal{B}$의 원소들의 합집합이므로 $U = \bigcup_{\alpha \in I} B_\alpha$ (단, $B_\alpha \in \mathcal{B}$)로 표현된다. 임의의 $x \in U$에 대해, $x \in B_\alpha \subset U$인 $B_\alpha$가 존재한다. 조건에 의해 $x \in B'_x \subset B_\alpha \subset U$인 $B'_x \in \mathcal{B}'$가 존재한다. 따라서 $U = \bigcup_{x \in U} B'_x$로 표현되므로 $U \in \mathscr{T}'$이다. 그러므로 $\mathscr{T} \subset \mathscr{T}'$이다. ∎

#### 예제 1.11
$\mathbb{R}$ 상에 $\mathcal{B}_K = \{[a, b) : a, b \in \mathbb{R}\} \cup {(a,b)-K \mid a, b \in R}$를 기저로 갖는 위상을 $K$-위상이라 할 때, 상한위상과 $K$-위상의 포함관계를 설명하시오. (단, $K \in \{1/n \mid n \in \mathbb{N}\}$이다.)


### 정의 1.13: 부분기저 (subbasis)
$S \subset \mathscr{T}$가 부분기저(subbasis)라는 것은, $S$의 원소들의 유한교집합 전체가 위상의 기저를 이루는 것을 의미한다.

#### 예제
- X = {a, b, c, d, e}의 위상 $\mathscr{T} = \{\emptyset, X, \{a, b\}, \{a, c, d, e\} \}$에 대하여 S = { {a, b}, {a, c, d, e} }가 부분기저임을 보이시오.

- S = { (a, \infty), (-\infty, b) : a, b \in \mathbb{R} }가 \mathbb{R}의 보통위상의 부분기저임을 보이시오.

### 정의 1.14: 부분집합족으로부터 생성된 위상 (topology generated by a family)
집합 $X$의 부분집합족 $\mathcal{A}$를 포함하는 가장 작은 위상을 $\mathcal{A}$에 의해 생성된 위상이라 하고 $\langle\mathcal{A}\rangle$으로 표기한다.

**동치 표현**: $\langle\mathcal{A}\rangle$은 $\mathcal{A}$의 원소들의 유한교집합을 기저로 하는 위상이다.

즉, $\langle\mathcal{A}\rangle$의 열린집합은 $\mathcal{A}$의 원소들의 유한교집합들의 임의의 합집합이다:
$$\langle\mathcal{A}\rangle = \left\{\bigcup_{\alpha \in I} \bigcap_{i=1}^{n_\alpha} A_{\alpha,i} : A_{\alpha,i} \in \mathcal{A}, n_\alpha \in \mathbb{N} \cup \{0\}, I \text{는 첨수집합}\right\}$$

### 정리 1.16: 부분집합족으로부터 생성된 위상 (topology generated by a family)
공집합이 아닌 집합 $X$의 부분집합족 $\mathcal{A}$에 대하여 $\mathcal{A}$는 $\langle\mathcal{A}\rangle$의 부분기저가 된다.

#### 증명
$\langle\mathcal{A}\rangle$은 $\mathcal{A}$를 포함하는 가장 작은 위상이므로, $\mathcal{A}$의 원소들의 유한교집합 전체가 $\langle\mathcal{A}\rangle$의 기저를 이룬다. 부분기저의 정의에 의해, $\mathcal{A}$의 원소들의 유한교집합이 기저를 이루면 $\mathcal{A}$는 $\langle\mathcal{A}\rangle$의 부분기저이다. ∎

#### 예제 1.16
$X = \mathbb{Z}$이고 $\mathcal{A} = \{\{a, a+1\} : a \in \mathbb{Z}\}$일 때, $\langle\mathcal{A}\rangle$을 구하시오.


### 정의 1.15: 국소기저 (local basis)
점 $p \in X$에서의 국소기저(local basis)는 $p$를 포함하는 열린집합들의 모임 $\mathcal{B}_p$로서, 다음을 만족한다:

$p$를 포함하는 임의의 열린집합 $G$에 대해, $p \in B \subset G$인 $B \in \mathcal{B}_p$가 존재한다.

#### 예제 1.17
$\mathbb{R}$ 위에 보통위상, 하한위상, 이산위상, 비이산위상이 주어져 있을 때 각각의 위상에 대하여 점 $p \in \mathbb{R}$에 대한 국소기저를 구하시오.

**풀이**:  
**1. 보통위상 $\mathscr{T}_u$에서 $p$의 국소기저**

$p$를 포함하는 열린집합은 열린구간들의 합집합이다. $p$를 포함하는 기본 열린집합은 $(a, b)$ (단, $a < p < b$)이다.

따라서 국소기저는:
$$\mathcal{B}_p = \{(p - \varepsilon, p + \varepsilon) : \varepsilon > 0\}$$

또는 더 구체적으로:
$$\mathcal{B}_p = \{(p - 1/n, p + 1/n) : n \in \mathbb{N}\}$$

**2. 하한위상 $\mathscr{T}_l$에서 $p$의 국소기저**

$p$를 포함하는 기본 열린집합은 $[p, b)$ (단, $b > p$)이다.

따라서 국소기저는:
$$\mathcal{B}_p = \{[p, p + \varepsilon) : \varepsilon > 0\}$$

또는 더 구체적으로:
$$\mathcal{B}_p = \{[p, p + 1/n) : n \in \mathbb{N}\}$$

**3. 이산위상 $\mathscr{T}_d = 2^\mathbb{R}$에서 $p$의 국소기저**

이산위상에서 $\{p\}$는 열린집합이다. $p$를 포함하는 임의의 열린집합 $G$에 대해 $p \in \{p\} \subset G$이므로:

$$\mathcal{B}_p = \{\{p\}\}$$

**4. 비이산위상 $\mathscr{T} = \{\emptyset, \mathbb{R}\}$에서 $p$의 국소기저**

$p$를 포함하는 유일한 열린집합은 $\mathbb{R}$이다.

$$\mathcal{B}_p = \{\mathbb{R}\}$$

### 정리 1.17: 국소기저를 이용한 위상적 성질의 특성화 (characterization via local basis)
위상공간 $(X, \mathscr{T})$에 대하여 $\mathcal{B}_p$는 $p$의 국소기저라 하자. 그러면 다음이 성립한다.

(1) $p \in A'$ $\Leftrightarrow$ 임의의 $B \in \mathcal{B}_p$에 대하여 $(B \setminus \{p\}) \cap A \neq \emptyset$.

(2) $p \in \overline{A}$ $\Leftrightarrow$ 임의의 $B \in \mathcal{B}_p$에 대하여 $B \cap A \neq \emptyset$.

(3) $p \in \mathrm{int}(A)$ $\Leftrightarrow$ 임의의 $B \in \mathcal{B}_p$에 대하여 $B \cap A \neq \emptyset$, $B \cap (X \setminus A) \neq \emptyset$.

(4) $x_n \to p$ $\Leftrightarrow$ 임의의 $B \in \mathcal{B}_p$에 대하여 $\exists N \in \mathbb{N}$ s.t. $n \geq N$ $\Rightarrow$ $x_n \in B$.

#### 증명
각 명제는 국소기저의 정의를 이용하여 원래의 위상적 정의와 동치임을 보일 수 있다.

**(1)** 국소기저의 정의에 의해, 점 $p$를 포함하는 모든 열린집합을 국소기저 원소들의 합집합으로 나타낼 수 있으므로, 집적점의 정의를 국소기저로 재정의할 수 있다.

**(2)** 폐포의 특성화(정리 1.9)에서 열린집합을 국소기저 원소로 대체하면 된다.

**(3)** 내부의 특성화(정리 1.11)를 국소기저로 나타낸 것이다.

**(4)** 수렴의 정의에서 열린집합을 국소기저 원소로 제한해도 동치성이 유지된다. ∎

#### 예제 1.12
$\mathbb{R}$의 하한위상 $\mathscr{T}_l$에서 수열 $-(1/n)$의 수렴값을 모두 구하시오.

### 따름정리 1.18: 국소기저를 이용한 기본 성질들
위상공간 $(X, \mathscr{T})$에서 점 $p$의 국소기저를 $\mathcal{B}_p$라 하면, 다음이 성립한다:

**(1)** $p \in A'$ $\Leftrightarrow$ $\mathcal{B}_p \in \mathscr{T}$인 임의의 $B \in \mathcal{B}_p$에 대하여 $(B \setminus \{p\}) \cap A \neq \emptyset$

**(2)** $p \in \overline{A}$ $\Leftrightarrow$ $\mathcal{B}_p \in \mathscr{T}$인 임의의 $B \in \mathcal{B}_p$에 대하여 $B \cap A \neq \emptyset$

**(3)** $p \in \mathrm{int}(A)$ $\Leftrightarrow$ $\mathcal{B}_p \in \mathscr{T}$인 임의의 $B \in \mathcal{B}_p$에 대하여 $B \cap A \neq \emptyset, B \cap A^c \neq \emptyset$

**(4)** $x_n \to p$ $\Leftrightarrow$ $\mathcal{B}_p \in \mathscr{T}$인 임의의 $B \in \mathcal{B}_p$에 대하여 $\exists N \in \mathbb{N}$ s.t. $n \geq N$ $\Rightarrow$ $x_n \in B$


## 1.6 적공간 (Product Space)

### 정의 1.16: 적위상 (product topology)
$X, Y$가 위상공간일 때, $X \times Y$의 적위상(product topology)은 다음을 기저로 하는 위상이다:
$$\mathcal{B} = \{U \times V : U \in \mathscr{T}_X, V \in \mathscr{T}_Y\}$$

  - 정의 1.16에서의 $\mathcal{B}$는 정리 1.14의 1, 2를 만족한다.

#### 예제 1.18
두 위상공간 $(X, \mathscr{T}_X)$와 $(Y, \mathscr{T}_Y)$에 대해 $\{U \times V : U \in \mathscr{T}_X, V \in \mathscr{T}_Y\}$는 일반적으로 $X \times Y$ 위에서의 위상이 아님을 보이시오.

**풀이**:

$\mathcal{S} = \{U \times V : U \in \mathscr{T}_X, V \in \mathscr{T}_Y\}$가 위상의 조건을 만족하지 않음을 보이기 위해 반례를 구성하자.

**반례**: $X = Y = \mathbb{R}$ (보통위상)

$U_1 = (0, 1) \in \mathscr{T}_X$, $V_1 = (0, 1) \in \mathscr{T}_Y$이므로 $U_1 \times V_1 = (0,1) \times (0,1) \in \mathcal{S}$.

$U_2 = (0, 2) \in \mathscr{T}_X$, $V_2 = (0, 2) \in \mathscr{T}_Y$이므로 $U_2 \times V_2 = (0,2) \times (0,2) \in \mathcal{S}$.

그러나 $(U_1 \times V_1) \cap (U_2 \times V_2) = (0,1) \times (0,1)$는 $\mathcal{S}$에 속한다.

더 결정적인 예: $U_1 = (0, 1), V_1 = (0, 2)$와 $U_2 = (0, 2), V_2 = (0, 1)$를 생각하자.

$A = U_1 \times V_2 = (0,1) \times (0,1)$와 $B = U_2 \times V_1 = (0,2) \times (0,2)$라 하면, $A, B \in \mathcal{S}$이다.

그러나 $A \cup B = [(0,1) \times (0,1)] \cup [(0,2) \times (0,2)]$는 $\mathcal{S}$의 원소가 될 수 없다.

왜냐하면 만약 $A \cup B = U \times V$ (단, $U \in \mathscr{T}_X, V \in \mathscr{T}_Y$)라면, 점 $(1/2, 1/2) \in A \cup B$이므로 $(1/2, 1/2) \in U \times V$, 즉 $1/2 \in U$이고 $1/2 \in V$이어야 한다.

또한 점 $(3/2, 3/2) \in A \cup B$이므로 $3/2 \in U$이고 $3/2 \in V$이어야 한다.

따라서 $U \supset [1/2, 3/2]$이고 $V \supset [1/2, 3/2]$이어야 한다. 그러면 $U \times V \supset [1/2, 3/2] \times [1/2, 3/2]$인데, 이는 $(1.2, 1.2) \in U \times V$를 의미하고, 이는 $A \cup B$에 포함되지 않는다.

따라서 $\mathcal{S}$는 합집합에 대해 닫혀있지 않으므로 위상이 아니다. ✗

**결론**: 적위상(정의 1.16)에서 기저는 $U \times V$ 형태의 집합들이지만, 임의의 $U \times V$들의 합집합이 모두 $U' \times V'$ 형태(단, $U' \in \mathscr{T}_X, V' \in \mathscr{T}_Y$)가 되는 것은 아니다. 이것이 기저 개념이 필요한 이유이다.

### 정리 1.19: 곱공간의 기저 (basis of product space)
$\mathcal{B}_X$가 $(X, \mathscr{T}_X)$에 대한 기저이고 $\mathcal{B}_Y$가 $(Y, \mathscr{T}_Y)$에 대한 기저라고 할 때,
$$\mathcal{B} = \{B_X \times B_Y : B_X \in \mathcal{B}_X, B_Y \in \mathcal{B}_Y\}$$
는 $X \times Y$ 상의 적공간에 대한 기저가 된다.

#### 증명
정리 1.14의 두 조건을 확인하자.

**(1) $\bigcup_{B \in \mathcal{B}} B = X \times Y$인가?**

$\mathcal{B}_X$와 $\mathcal{B}_Y$는 각각의 위상의 기저이므로:
$$\bigcup_{B_X \in \mathcal{B}_X} B_X = X, \quad \bigcup_{B_Y \in \mathcal{B}_Y} B_Y = Y$$

따라서:
$$\bigcup_{B \in \mathcal{B}} B = \bigcup_{B_X \in \mathcal{B}_X, B_Y \in \mathcal{B}_Y} (B_X \times B_Y) = \left(\bigcup_{B_X \in \mathcal{B}_X} B_X\right) \times \left(\bigcup_{B_Y \in \mathcal{B}_Y} B_Y\right) = X \times Y$$

**(2) $B_1, B_2 \in \mathcal{B}$이고 $(x,y) \in B_1 \cap B_2$일 때, $(x,y) \in B \subset B_1 \cap B_2$인 $B \in \mathcal{B}$가 존재하는가?**

$B_1 = B_X^{(1)} \times B_Y^{(1)}$, $B_2 = B_X^{(2)} \times B_Y^{(2)}$라 하자. 단, $B_X^{(i)}, B_X^{(2)} \in \mathcal{B}_X$, $B_Y^{(1)}, B_Y^{(2)} \in \mathcal{B}_Y$.

$(x,y) \in B_1 \cap B_2 = (B_X^{(1)} \times B_Y^{(1)}) \cap (B_X^{(2)} \times B_Y^{(2)}) = (B_X^{(1)} \cap B_X^{(2)}) \times (B_Y^{(1)} \cap B_Y^{(2)})$

따라서 $x \in B_X^{(1)} \cap B_X^{(2)}$이고 $y \in B_Y^{(1)} \cap B_Y^{(2)}$이다.

$\mathcal{B}_X$가 $\mathscr{T}_X$의 기저이므로, 기저의 조건에 의해 $x \in B_X^{(0)} \subset B_X^{(1)} \cap B_X^{(2)}$인 $B_X^{(0)} \in \mathcal{B}_X$가 존재한다.

$\mathcal{B}_Y$가 $\mathscr{T}_Y$의 기저이므로, 기저의 조건에 의해 $y \in B_Y^{(0)} \subset B_Y^{(1)} \cap B_Y^{(2)}$인 $B_Y^{(0)} \in \mathcal{B}_Y$가 존재한다.

따라서 $B = B_X^{(0)} \times B_Y^{(0)} \in \mathcal{B}$이고:
$$(x,y) \in B \subset B_1 \cap B_2$$

그러므로 정리 1.14의 두 조건을 만족하므로 $\mathcal{B}$는 $X \times Y$의 적공간에 대한 기저이다. ∎

### 정리 1.20: 곱집합의 폐포 (closure in product)
$$\overline{A \times B} = \overline{A} \times \overline{B}$$
따라서 C가 $X \times Y$의 닫힌집합이라면, $C\timesD$는 $X \times Y$의 닫힌집합이다.

#### 증명

### 정리 1.21: 곱집합의 내부 (interior in product)
$$\mathrm{int}(A \times B) = \mathrm{int}(A) \times \mathrm{int}(B)$$

**주의**: 외부, 유도집합, 경계는 이러한 성질을 만족하지 않는다. 예를 들어, 곱공간에서 경계는 다음과 같이 더 복잡하다:
$$\partial(A \times B) \neq \partial(A) \times \partial(B)$$

#### 예제 1.19
하한위상과 이산위상의 적공간 $\mathbb{R} \times \mathbb{R}$의 부분집합 $A = [1, 3) \times (2, 4]$에 대하여 내부, 외부, 경계, 유도집합, 폐포를 구하시오.

**풀이**:  
첫 번째 좌표는 하한위상 $\mathscr{T}_l$, 두 번째 좌표는 이산위상 $\mathscr{T}_d = 2^\mathbb{R}$을 갖는 적공간을 생각한다.

**Step 1: $\overline{A}$ 구하기**

정리 1.20에 의해:
$$\overline{A} = \overline{[1, 3)} \times \overline{(2, 4]}$$

- 하한위상에서: $\overline{[1, 3)} = [1, 3]$ (반열린구간 $[1, 3)$의 폐포는 $[1, 3]$)
- 이산위상에서: $\overline{(2, 4]} = (2, 4]$ (모든 집합이 닫혀있음)

따라서:
$$\boxed{\overline{A} = [1, 3] \times (2, 4]}$$

**Step 2: $\mathrm{int}(A)$ 구하기**

정리 1.21에 의해:
$$\mathrm{int}(A) = \mathrm{int}([1, 3)) \times \mathrm{int}((2, 4])$$

- 하한위상에서: 반열린구간 $[1, 3)$는 열린집합이므로 $\mathrm{int}([1, 3)) = [1, 3)$
- 이산위상에서: 한 점 집합은 모두 닫혀있으므로 $\mathrm{int}((2, 4]) = (2, 4]$ (이산위상에서 모든 집합이 열려있기도 하고 닫혀있음)

따라서:
$$\boxed{\mathrm{int}(A) = [1, 3) \times (2, 4]}$$

**Step 3: $\partial(A)$ 구하기**

$$\partial(A) = \overline{A} \setminus \mathrm{int}(A) = ([1, 3] \times (2, 4]) \setminus ([1, 3) \times (2, 4])$$

$$= \{3\} \times (2, 4]$$

따라서:
$$\boxed{\partial(A) = \{3\} \times (2, 4]}$$

**Step 4: $\mathrm{ext}(A)$ 구하기**

$$\mathrm{ext}(A) = \mathrm{int}((\mathbb{R} \times \mathbb{R}) \setminus A) = \mathrm{int}((\mathbb{R} \setminus [1, 3)) \times \mathbb{R} \cup [1, 3) \times (\mathbb{R} \setminus (2, 4]))$$

$\mathbb{R} \setminus [1, 3) = (-\infty, 1)$ (하한위상에서, $[1, 3)$의 여집합)

이산위상에서는 모든 집합의 내부를 구하기가 다르다. 이산위상에서는 임의의 비어있지 않은 집합이 열려있으므로, 실제로 내부를 구하려면 더 신중해야 한다.

대신, $\overline{A} = [1, 3] \times (2, 4]$이고 $\mathrm{int}(A) = [1, 3) \times (2, 4]$이며 $\partial(A) = \{3\} \times (2, 4]$이므로:

$$X = \mathrm{int}(A) \cup \partial(A) \cup \mathrm{ext}(A)$$

분할 조건을 사용하면:
$$\mathrm{ext}(A) = X \setminus \overline{A} = (\mathbb{R} \times \mathbb{R}) \setminus ([1, 3] \times (2, 4])$$

$$= ((-\infty, 1) \cup [3, \infty)) \times \mathbb{R} \cup [1, 3] \times ((-\infty, 2] \cup [4, \infty))$$

따라서:
$$\boxed{\mathrm{ext}(A) = ((-\infty, 1) \cup [3, \infty)) \times \mathbb{R} \cup [1, 3] \times ((-\infty, 2] \cup [4, \infty))}$$

**Step 5: $A'$ (유도집합) 구하기**

점 $(x, y) \in \mathbb{R} \times \mathbb{R}$이 $A$의 집적점이 되려면, $(x, y)$를 포함하는 모든 열린집합 $G$에 대해 $(G \setminus \{(x,y)\}) \cap A \neq \emptyset$이어야 한다.

적공간의 기저는 $\{[a, b) \times V : a, b \in \mathbb{R}, V \in 2^\mathbb{R}\}$이다. (하한위상의 반열린구간과 이산위상의 임의의 집합)

- **$(x, y) \in [1, 3] \times (2, 4]$인 경우**:
   점 $(x, y)$를 포함하는 기본 열린집합 $[x, c) \times \{y\}$ (단, $c > x$)를 생각하자.
   
   $1 \leq x < 3$이고 $2 < y \leq 4$이면, $([x, c) \setminus \{(x,y)\}) \cap A$가 공집합일 수 있다. 특히 $x = 3$인 경우나 $y$의 경계에서.
   
   더 정확히, 이산위상에서 $\{y\}$만 열린집합이므로, $(x, y)$가 집적점이 되려면 $(x, y)$ 근처의 다른 점들이 $A$에 있어야 한다.
   
   - $x \in [1, 3)$이고 $y \in (2, 4]$이면, $(x + \delta, y) \in A$ (충분히 작은 $\delta > 0$)이므로 $(x, y)$는 집적점.
   - $x = 3$이면, $3$을 포함하는 하한위상의 기본 열린집합은 $[3, c)$인데, 이는 $A$와 교집합을 가지지 않음. 따라서 $3$은 집적점이 아님.
   - $y \notin (2, 4]$이면 집적점이 아님.

따라서:
$$\boxed{A' = [1, 3) \times (2, 4]}$$

**최종 답**:
- $\overline{A} = [1, 3] \times (2, 4]$
- $\mathrm{int}(A) = [1, 3) \times (2, 4]$
- $\partial(A) = \{3\} \times (2, 4]$
- $\mathrm{ext}(A) = ((-\infty, 1) \cup [3, \infty)) \times \mathbb{R} \cup [1, 3] \times ((-\infty, 2] \cup [4, \infty))$
- $A' = [1, 3) \times (2, 4]$


## 1.7 부분공간 (Subspace)

### 정의 1.17: 상대위상과 부분공간 (relative topology and subspace)
$(X, \mathscr{T})$가 위상공간이고 $A \subset X$일 때, 상대위상(relative topology) 또는 부분공간 위상은:
$$\mathscr{T}_A = \{A \cap U : U \in \mathscr{T}\}$$

순서쌍 $(A, \mathscr{T}_A)$를 부분공간(subspace)이라 한다.
- $\mathscr{T}_A = \{A \cap U : U \in \mathscr{T}\}$는 $A$ 위의 위상이다.

#### 예제 1.20
보통위상공간 $\mathbb{R}$의 부분집합 $A = \{1/n : n \in \mathbb{N}\}$의 상대위상을 구하시오.

**풀이**:

상대위상 $\mathscr{T}_A = \{A \cap U : U \in \mathscr{T}_u\}$의 열린집합을 구하자.

$\mathbb{R}$의 보통위상에서 열린집합은 열린구간들의 합집합이다.

**Step 1: $A$의 구조**

$A = \{1, 1/2, 1/3, 1/4, \ldots\}$이고, $\lim_{n \to \infty} 1/n = 0$이다.

**Step 2: 상대위상의 열린집합 특성화**

$V \in \mathscr{T}_A$ $\Leftrightarrow$ $V = A \cap U$인 열린집합 $U \in \mathscr{T}_u$가 존재.

$U$는 열린구간들의 합집합이므로, $V$는 $A$의 부분집합 중에서 $U$의 열린구간들과의 교집합으로 나타나는 집합이다.

**Step 3: 구체적인 열린집합들**

- $\emptyset$: $A \cap \emptyset = \emptyset \in \mathscr{T}_A$ ✓
- $A$: $A \cap \mathbb{R} = A \in \mathscr{T}_A$ ✓
- $\{1/n\}$ (단일원소): $\{1/n\} = A \cap (1/n - \varepsilon, 1/n + \varepsilon)$인 충분히 작은 $\varepsilon$를 선택하면 ($1/n$과 다른 점들을 분리하는 $\varepsilon$), $\{1/n\} \in \mathscr{T}_A$ ✓
- 일반적으로, $A$의 임의의 부분집합 $B$에 대해:
   $$B = A \cap \left(\bigcup_{1/n \in B} (1/n - \varepsilon_n, 1/n + \varepsilon_n)\right)$$
   로 표현할 수 있으므로 $B \in \mathscr{T}_A$.

**Step 4: 결론**

각 점 $1/n$에 대해, 충분히 작은 $\varepsilon$를 선택하면 $(1/n - \varepsilon, 1/n + \varepsilon)$는 $A$의 다른 점들을 포함하지 않는다. (왜냐하면 $A$의 점들이 서로 떨어져 있고, $0$에만 수렴하기 때문)

따라서 $\{1/n\}$은 상대위상에서 열린집합이다. 이는 $A$가 이산위상을 가진다는 의미이다.

**답**:
$$\boxed{\mathscr{T}_A = 2^A \text{ (이산위상)}}$$

즉, $A$의 모든 부분집합이 상대위상에서 열린집합이다.


#### 예제 1.21
$\mathbb{R}$에 여유한 위상 $\mathscr{T}_f$가 주어져 있을 때, $A$의 유한 부분집합 $B$에 대하여 $B$에서의 상대위상을 구하시오.

**풀이**:  
상대위상 $\mathscr{T}_B = \{B \cap U : U \in \mathscr{T}_f\}$를 구하자.

여유한 위상의 열린집합은 $\emptyset$이거나 여유한집합(complement가 유한)이다.

**Step 1: $B$의 구조**

$B$는 유한집합이므로 $B = \{b_1, b_2, \ldots, b_n\}$ (단, $n \in \mathbb{N}$).

**Step 2: 상대위상의 열린집합 특성화**

$V \in \mathscr{T}_B$ $\Leftrightarrow$ $V = B \cap U$인 $U \in \mathscr{T}_f$가 존재.

$U \in \mathscr{T}_f$이면 $U = \emptyset$이거나 $\mathbb{R} \setminus U$가 유한이다.

**Step 3: 경우 분석**

- $U = \emptyset$인 경우: $B \cap \emptyset = \emptyset \in \mathscr{T}_B$

- $U \neq \emptyset$인 경우 (즉, $\mathbb{R} \setminus U$가 유한): 
  
  $\mathbb{R} \setminus U$가 유한이므로, $B$의 점 중 $\mathbb{R} \setminus U$에 속하는 점의 개수는 최대 $|B| = n$개이고 실제로는 그보다 적을 수 있다.
  
  따라서:
  $$B \cap U = B \setminus (B \cap (\mathbb{R} \setminus U))$$
  
  $B \cap (\mathbb{R} \setminus U)$는 $B$의 부분집합이고, $B$가 유한이므로 이는 $B$의 임의의 부분집합이 될 수 있다.

**Step 4: 결론**

$U$가 여유한 위상의 열린집합(즉, $\mathbb{R} \setminus U$가 유한)이면, $B \cap U$는 $B$의 어떤 부분집합이다.

역으로, $B$의 임의의 부분집합 $V'$에 대해, $U = V' \cup (\mathbb{R} \setminus B)$로 정의하면:
- $\mathbb{R} \setminus U = (\mathbb{R} \setminus (V' \cup (\mathbb{R} \setminus B))) = B \setminus V'$
- $B \setminus V'$는 $B$의 부분집합이므로 유한
- 따라서 $U \in \mathscr{T}_f$

그리고 $B \cap U = B \cap (V' \cup (\mathbb{R} \setminus B)) = V'$

**답**:

$$\boxed{\mathscr{T}_B = 2^B \text{ (이산위상)}}$$

즉, $B$의 모든 부분집합이 상대위상에서 열린집합이다.

**일반화**: 여가산 위상에서도 유사한 논리에 의해, $\mathbb{R}$의 가산 부분집합 $C$에 대한 상대위상은 $\mathscr{T}_C = 2^C$ (이산위상)이다.

### 정리 1.22: 부분공간에서의 폐집합 판정법 (closed set criterion in subspace)
위상공간 $(X, \mathscr{T})$에 대하여 $A \subset X$이고 $B \subset A$라 하자. $B$가 $A$에서 폐집합일 필요충분조건은 $X$-폐집합인 $F$가 존재하여 $B = A \cap F$인 것이다.

#### 증명
$(\Rightarrow)$ $B$가 $A$의 상대위상에서 폐집합이라 하자. 그러면 정의에 의해 $A \setminus B$는 $A$의 상대위상에서 열린집합이다. 따라서 $A \setminus B = A \cap U$인 $U \in \mathscr{T}$가 존재한다.

$F = X \setminus U$로 정의하면 $F$는 $X$에서 폐집합이다.

$A \cap F = A \cap (X \setminus U) = A \setminus U = A \setminus (A \cap U) = A \setminus (A \setminus B) = B$

따라서 $B = A \cap F$이다.

$(\Leftarrow)$ $F$가 $X$에서 폐집합이고 $B = A \cap F$라 하자. $U = X \setminus F$로 정의하면 $U \in \mathscr{T}$이다.

$A \setminus B = A \setminus (A \cap F) = A \cap (X \setminus F) = A \cap U$

따라서 $A \setminus B$는 $A$의 상대위상에서 열린집합이므로, $B$는 $A$에서 폐집합이다. ∎

### 정리 1.23: 부분공간에서의 기저 (basis in subspace)
$\mathcal{B}$가 위상공간 $(X, \mathscr{T})$에 대한 기저이고 $A \subset X$일 때,
$$\mathcal{B}_A = \{B \cap A : B \in \mathcal{B}\}$$
는 부분공간 $(A, \mathscr{T}_A)$에 대한 기저이다.

#### 증명
$\mathcal{B}_A$가 부분공간 $A$의 기저임을 보이자.

**(1) 기저의 첫 번째 조건: $\bigcup_{B' \in \mathcal{B}_A} B' = A$**

$$\bigcup_{B' \in \mathcal{B}_A} B' = \bigcup_{B \in \mathcal{B}} (B \cap A) = \left(\bigcup_{B \in \mathcal{B}} B\right) \cap A = X \cap A = A$$

(여기서 $\mathcal{B}$가 $X$의 기저이므로 $\bigcup_{B \in \mathcal{B}} B = X$)

**(2) 기저의 두 번째 조건**

$B_1', B_2' \in \mathcal{B}_A$이고 $x \in B_1' \cap B_2'$라 하자. 그러면 $B_1' = B_1 \cap A$, $B_2' = B_2 \cap A$인 $B_1, B_2 \in \mathcal{B}$가 존재한다.

$x \in B_1' \cap B_2' = (B_1 \cap A) \cap (B_2 \cap A) = (B_1 \cap B_2) \cap A$

따라서 $x \in B_1 \cap B_2$이고, $\mathcal{B}$가 $X$의 기저이므로 $x \in B \subset B_1 \cap B_2$인 $B \in \mathcal{B}$가 존재한다.

$B' = B \cap A$로 정의하면 $B' \in \mathcal{B}_A$이고:
$$x \in B' = B \cap A \subset (B_1 \cap B_2) \cap A = B_1' \cap B_2'$$

따라서 정리 1.14의 두 조건을 만족하므로 $\mathcal{B}_A$는 $A$의 상대위상에 대한 기저이다. ∎

### 정리 1.24: 부분공간의 상대위상 추이성 (transitivity of relative topology)
위상공간 $(X, \mathscr{T})$에 대하여 $A \subset B \subset X$라 하자. $B$의 부분공간 $A$에 대해, $V$가 $B$에서 열린집합이고 $U$가 $A$에서 열린집합이면 $U$는 $X$에서 열린집합이다.

#### 증명
$U$가 $A$의 상대위상에서 열린집합이면, 정의에 의해 $U = A \cap W$인 $W \in \mathscr{T}_B$가 존재한다.

$V$가 $B$의 상대위상에서 열린집합이면, 정의에 의해 $V = B \cap U'$인 $U' \in \mathscr{T}$가 존재한다.

$\mathscr{T}_B$는 상대위상이므로 $W = B \cap U''$인 $U'' \in \mathscr{T}$가 존재한다.

따라서:
$$U = A \cap W = A \cap (B \cap U'') = (A \cap B) \cap U'' = A \cap U''$$

$U'' \in \mathscr{T}$이므로 $U$는 $X$의 상대위상에서 $A$에 대한 열린집합이다. ∎

### 정리 1.25: 부분공간의 상대위상 추이성 (transitivity of relative topology for closed sets)
위상공간 $(X, \mathscr{T})$에 대하여 $A \subset B \subset X$라 하자. $F$가 $B$에서 닫힌집합이고 $G$가 $A$에서 닫힌집합이면, $G$는 $X$에서 닫힌집합이다.

#### 증명
$F$가 $B$에서 닫힌집합이고 $G$가 $A$에서 닫힌집합이라 하자.

정리 1.22에 의해, $F$가 $B$에서 닫힌집합이면 $B \setminus F$는 $B$에서 열린집합이다. 따라서 $B \setminus F = B \cap U$인 $U \in \mathscr{T}$가 존재한다. 그러므로 $F = B \cap (X \setminus U)$이고, $X \setminus U \in \mathscr{T}$의 여집합이므로 $X \setminus U$는 $X$에서 폐집합이다.

마찬가지로, $G$가 $A$에서 닫힌집합이면 정리 1.22에 의해 $G = A \cap V$인 $V \in \mathscr{T}$가 존재한다. (여기서 $V$는 실제로는 어떤 $X$-폐집합의 교집합)

따라서:
$$G = A \cap V$$

$V$가 $B$에서 닫힌집합인 $F$에 포함되면, $G = A \cap F$로 표현되고, $A \cap B = A$이므로:
$$G = (A \cap B) \cap (X \setminus 어떤 \text{ 열린집합}) = A \cap (X \setminus 어떤 \text{ 열린집합})$$

정리 1.22를 다시 적용하면, $G$는 $X$에서 어떤 닫힌집합과 $A$의 교집합이다. $A \subset B \subset X$이고 $G$가 각 부분공간에서 닫힌조건을 만족하므로, $G$는 $X$에서 닫힌집합이다. ∎

