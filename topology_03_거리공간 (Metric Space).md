# Chapter 3 거리공간 (Metric Space)

## 3.1 거리공간 (Metric Space)

### 정의 3.1: 거리함수 (metric / distance function)
집합 $X$에서 거리함수(또는 거리, metric) $d: X \times X \to \mathbb{R}$는 다음 네 조건을 만족하는 함수이다:

1. **비음성(non-negativity)**: $d(x,y) \geq 0$ for all $x,y \in X$
2. **동일성(identity of indiscernibles)**: $d(x,y) = 0 \Leftrightarrow x = y$
3. **대칭성(symmetry)**: $d(x,y) = d(y,x)$ for all $x,y \in X$
4. **삼각부등식(triangle inequality)**: $d(x,z) \leq d(x,y) + d(y,z)$ for all $x,y,z \in X$

순서쌍 $(X, d)$를 거리공간(metric space)이라 한다.

### 정의 3.2: 열린공(open ball)
거리공간 $(X, d)$에서 점 $x \in X$와 $r > 0$에 대해, 열린공(open ball)을 다음과 같이 정의한다:
$$B_d(x, r) = \{y \in X : d(x,y) < r\}$$

### 정의 3.3: 거리위상 (metric topology)
거리공간 $(X, d)$에서 거리위상 $\mathscr{T}_d$는 열린공들을 기저로 하는 위상이다:
$$\mathscr{T}_d = \{U \subset X : \forall x \in U, \exists r > 0 \text{ s.t. } B_d(x,r) \subset U\}$$

## 3.2 거리공간의 기본 성질 (Basic Properties)

### 정리 3.5: 가산국소기저와 폐포의 거리표현
거리공간에서 다음이 성립한다:

1. 각 점 $x \in X$에 대해 $\{B_d(x, 1/n) : n \in \mathbb{N}\}$은 $x$의 가산국소기저이다.

2. 부분집합 $A \subset X$의 폐포는 다음과 같이 표현된다:
$$\overline{A} = \{x \in X : d(x, A) = 0\}$$
여기서 $d(x, A) = \inf\{d(x,a) : a \in A\}$는 점 $x$에서 집합 $A$까지의 거리이다.

3. 유한부분집합은 항상 닫힌집합이다.

### 정리 3.6: 서로소 폐집합의 분리
거리공간에서 $A, B$가 공집합이 아닌 서로소 닫힌집합이면, 다음을 만족하는 서로소 열린집합 $U, V$가 존재한다:
$$A \subset U, \quad B \subset V, \quad U \cap V = \emptyset$$

즉, 거리공간은 정규공간이다.

## 3.3 분리 성질과 거리화가능성

### 정의 3.4: 거리화가능 (metrizable)
위상공간 $(X, \mathscr{T})$가 거리화가능(metrizable)이라는 것은, 어떤 거리 $d$가 존재하여 $\mathscr{T} = \mathscr{T}_d$인 것을 의미한다.

### 정리 3.7: 거리공간은 $T_4$-공간
모든 거리공간은 $T_4$-공간(즉, 정규 $T_1$-공간)이다.

