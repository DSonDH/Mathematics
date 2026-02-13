# Chapter 2 연속함수 (Continuous Maps)
## 2.1 연속성과 위상동형 (Continuity, Homeomorphism)

### 정의 2.1: 연속함수 (continuous function)
함수 $f: (X, \mathscr{T}_X) \to (Y, \mathscr{T}_Y)$가 연속(continuous)이라는 것은:

$\forall G \in \mathscr{T}_Y$에 대해 $f^{-1}(G) \in \mathscr{T}_X$

즉, $Y$의 모든 열린집합의 역상이 $X$의 열린집합이다.

### 정리 2.1: 기저와 부분기저에 의한 연속성 판정 (basis/subbasis criterion)
다음은 모두 동치이다:
1. $f$는 연속이다.
2. $Y$의 기저 $\mathcal{B}_Y$에 대해, $\forall B \in \mathcal{B}_Y$일 때 $f^{-1}(B) \in \mathscr{T}_X$이다.
3. $Y$의 부분기저 $S_Y$에 대해, $\forall S \in S_Y$일 때 $f^{-1}(S) \in \mathscr{T}_X$이다.

### 정리 2.2: 연속함수의 기본 성질 (basic properties)
1. 상수함수는 연속이다.
2. 두 연속함수의 합성은 연속이다: $f: X \to Y$와 $g: Y \to Z$가 연속이면 $g \circ f: X \to Z$도 연속.
3. 연속함수의 제한함수도 연속이다: $f: X \to Y$가 연속이고 $A \subset X$이면, $f|_A: A \to Y$도 연속.

### 정리 2.5: 점별 연속과 연속의 동치성 (pointwise continuity)
$f: X \to Y$가 연속 $\Leftrightarrow$ 모든 $x \in X$에서 점별 연속이다.

(여기서 $f$가 점 $x$에서 점별 연속이라는 것은, $f(x)$를 포함하는 모든 열린집합 $G$에 대해 $f(x) \in f(x) \in G$를 포함하는 $x$의 열린집합 $H$가 존재하여 $f(H) \subset G$인 것.)

### 정의 2.3: 위상동형 (homeomorphism)
함수 $f: X \to Y$가 위상동형사상(homeomorphism)이라는 것은:
1. $f$가 전단사이다.
2. $f$와 $f^{-1}$ 모두 연속이다.

이 경우 $X$와 $Y$를 위상동형(homeomorphic)이라 하고, $X \cong Y$로 표기한다.

### 정의 2.4: 위상적 성질 (topological property)
어떤 성질이 위상적 성질(topological property)이라는 것은, 위상동형에 의해 보존되는 성질을 의미한다.

**예시**: 컴팩트성, 연결성, 분리공리($T_0, T_1, T_2, T_3, T_4$), 가산공리(가산기저, 가산국소기저), 거리화가능성 등이 위상적 성질이다.

## 2.2 상공간 (Quotient Space)

### 정의 2.6: 상공간과 상위상 (quotient space and quotient topology)
$f: X \to Y$가 전사함수일 때, $Y$의 상위상(quotient topology)은:
$$\mathscr{T}_Y = \{U \subset Y : f^{-1}(U) \in \mathscr{T}_X\}$$

순서쌍 $(Y, \mathscr{T}_Y)$를 상공간(quotient space)이라 하고, $f$를 상사상(quotient map)이라 한다.

### 정리 2.7: 상위상의 성질
상사상 $f: X \to Y$에 대해:
1. $\mathscr{T}_Y$는 $Y$ 위의 위상이다.
2. $f$는 연속이다.
3. $f$는 열린사상이거나 닫힌사상이다.