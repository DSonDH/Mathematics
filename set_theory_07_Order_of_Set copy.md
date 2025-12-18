# 📘 집합의 순서 (Order in Set Theory)
## 1. 부분순서집합 (Partially Ordered Set)
### (1) 정의
#### ① 부분순서관계
* 성질: **반사적**, **반대칭적**, **추이적**
- 반사적 (Reflexive): $aRa$
- 반대칭적 (Antisymmetric): $aRb \land bRa \Rightarrow a = b$
- 추이적 (Transitive): $aRb \land bRc \Rightarrow aRc$

* 예시
  1. 두 집합 $A, B$에 대해 $A \subseteq B$
  2. 두 수 $x, y$에 대해 $x \leq y$
  3. 자연수 $n, m$에 대해 “$n$이 $m$의 배수” 관계

#### ② 부분순서집합

집합 $A$ 위에 부분순서관계 $\le$가 주어져 있을 때, 이를 **부분순서집합(poset)** 이라 하며 $(A, \le)$로 나타낸다.
- 모든 원소들이 순서관계를 가져야 하는 것은 아님! 일부 원소끼리 만족해도 됨.

#### ③ 극대원소(Maximal element)와 극소원소(Minimal element)

부분순서집합 $A$에서 $a$ in A는 **극대원소**: 
$$
\forall x \in A,\ a \le x \Rightarrow x = a
$$

$a$ in A는 **극소원소**:
$$
\forall x \in A,\ x \le a \Rightarrow x = a
$$
극대, 극소 원소는 유일하지는 않다.  

예: 멱집합 $\mathcal{P}(X)$에서 $\emptyset, X$

#### ④ 최대원소(Greatest element)와 최소원소(Least element)

부분순서집합 $A$,  
 a in A에서 **최대원소** :
$$
\forall x \in A,\ x \le a
$$

a in A에서 **최소원소** :
$$
\forall x \in A,\ a \le x
$$

- 극대/극소 원소는 존재가능하지만, 최대/최소 원소는 없을 수 있다: 모든 원소끼리 비교가능하지 않으면 그럼.  
- 개구간 (0, 1)에서 극대, 극소를 알 수 없음.  
- 이런 무한 집합에서 극대, 극소 개념이 필요해서 상한, 하한이 도입됨.

### (3) 상한과 하한
극대, 극소, 최대원소, 최소원소와 달리, 상한은 $A$ 안에 부분집합 $B$ 에 대해 정의됨  

#### ① 상계(upper bound)와 하계(lower bound)

부분순서집합 $A$의 부분집합 $B$에 대하여  
$a$ in A가 **상계** : 
$$
\forall x \in B,\ x \le a
$$

$a$ in A가 **하계** :
$$
\forall x \in B,\ a \le x
$$

a가 B의 원소일수도, 아닐수도 있음  
상계, 하계가 없을수도 있음  
상계가 존재하는 A를 위로유계이다 라고 한다.  
위로 유계, 아래로 유계인 집합을 유계집합이라 한다.  
#### ② 상한(supremum)과 하한(infimum)
상계 중 최소인 원소를 **상한**,
하계 중 최대인 원소를 **하한**이라 한다.
$$\sup B, \quad \inf B$$

예: $A = [0,1)$이면 $\sup A = 1,; \inf A = 0$

### (4) 절편과 절단
#### ① 절편 (section 또는 segment)
$$
S_a = \{x \in A \mid x < a\}
$$

예:
* $\mathbb{R}$의 절편 $S_0 = (-\infty, 0)$
* $\mathbb{N}$의 절편 $S_3 = \{1,2\}$

#### ② 절단 (cut)
$$
B \cap C = \emptyset,\ B \cup C = A,\\
x \in B,\ y \leq x \Rightarrow y \in B \\
x \in C,\ x \leq y \Rightarrow y \in C
$$
를 만족하는 공집합 아닌 부분집합들의 쌍 $(B, C)$를 **절단(cut)** 이라 한다.  

어떤 것 기준으로 작은 것들, 어떤 것 기준으로 큰 것을이 나뉘고 그 둘이 전체를 구성한다.  

예: $\mathbb{R}$에서 $M = (-\infty, 0)$, $N = [0, \infty)$.

### (5) 순서동형 (Order Isomorphism)
#### ① 순서보존함수 (Order-preserving map)
부분순서집합 A, B에 대해 함수 f  
$$
f: A \to B,\quad \forall x, y \in A, \quad x \le y \Rightarrow f(x) \le f(y)
$$
이면 **순서보존함수**라 한다.  
입력값 순서가 출력값 순서에도 동일하게 유지되는 것  
#### ② 순서동형사상 (Order isomorphism)
$$
f: A \to B \text{ 전단사이며 } x \le y \Rightarrow f(x) \le f(y)
$$
이면 **순서동형사상(order isomorphism)** 이라 한다.
이때 $A, B$가 순서동형이면
$$ A \cong B $$
로 나타낸다.

예: 항등함수 $I_A : A \to A$

## 2. 전순서집합 (Totally Ordered Set)
### (1) 정의
#### ① 비교가능성
$$
\forall x, y \in A,\ (x \le y) \lor (y \le x)
$$
이면 $A$는 **전순서집합(totally ordered set)** 이라 한다.

#### ② 전순서집합

모든 원소가 비교 가능한 경우
$(A, \le)$를 **전순서집합** 이라 한다.

### (2) 쇄(chain)

부분순서집합 $A$의 전순서 부분집합 $B$를
$A$에서의 **쇄(chain)** 라 한다.

### (3) 정렬집합 (Well-ordered Set)

$$
A \neq \emptyset \Rightarrow \exists a \in A,\ a \text{는 최소원소}
$$
모든 비공집합이 최소원소를 가지면 $A$는 **정렬집합(well-ordered set)** 이라 한다.  
정렬의 기준이 되는 최소원소가 있어야 한다는 말.  
정렬집합이면 전순서집합. 전순서집합이라고 정렬집합은 아님.  

## 3. 서수 (Ordinal Number, Ord)
### (1) 서수의 개념
서수는 집합의 **순서형태(길이)** 를 나타낸다.
(기수는 단순 크기. 서수는 수학적 구조(정렬집합)가 추가됨)
* 모든 정렬집합 $A$에 대해 서수 $\alpha$가 존재.
* 모든 순서수 $\alpha$에 대해 $o(A) = \alpha$인 정렬집합 A가 존재한다. 
  - o 또는 ord는 서수를 나타내는 기호
  $$ A \cong \alpha $$
1) $A \cong B \iff o(A) = o(B)$

2) $A = \emptyset \iff o(A) = 0$

3) $A \cong \{1, 2, \dots, k\} \iff o(A) = k$
* $\mathbb{N}$의 정렬형은 $\omega$ (최초의 초한서수).

예: $A={1,2,3}$이면 $o(A)=3$

### (2) 서수의 종류
* **유한서수**: 유한 정렬집합의 서수
* **초한서수**: 무한 정렬집합의 서수

대표적 초한서수:
$\omega = o(\mathbb{N})$

### (3) 서수의 순서
정렬집합 $A,B$에 대해
$$
o(A)=\alpha,\ o(B)=\beta
$$
일 때,
$A$가 $B$의 절편과 순서동형이면 $\alpha$는 $\beta$보다 작거나 같다고 하며, $\alpha \preceq \beta$라 한다. $\alpha \neq \beta$이면 $\alpha \prec \beta$

예:
$A={1}, B={3,4,5}$ → $o(A)=1, o(B)=3$, $\alpha\prec\beta$

### (4) 서수의 연산
#### ① 서수 합
서로소인 두 집합 A, B의 서수가 alpha, beta. 
$$
\alpha + \beta = o(A \cup B)
$$

#### ② 서수 곱
집합 A, B의 서수가 alpha, beta
$$
\alpha \times \beta = o(B \times A)
$$ 
AxB가 아님!! BxA로 정의 한것임.  
서수 합은 서로소여야하고, 서수 곱은 서로소 아니어도 됨.  

### (5) 연산 법칙

임의의 서수 $\alpha, \beta, \gamma$에 대하여

1. 결합법칙
   $$
   (\alpha + \beta) + \gamma = \alpha + (\beta + \gamma)
   $$
   $$
   (\alpha \times \beta) \times \gamma = \alpha \times (\beta \times \gamma)
   $$

2. 분배법칙
   $$
   \alpha(\beta + \gamma) = \alpha\beta + \alpha\gamma
   $$
   
   $$
   (\beta + \gamma)\alpha \neq \alpha\gamma + \beta\gamma
   $$

단, **서수의 덧셈과 곱셈은 교환법칙이 성립하지 않는다.**  
덧셈의 경우:  
$1+ \omega$는 1과 무한자연수의 순서구조.  
$\omega + 1$은 무한자연수가 이어지다가 한 상수로 막힌 구조.  
위 아래는 그래서 다른 개념임.  

곱셈의 경우:  
$2 \times \omega = o(\{0,1\}\times\mathbb{N})$

순서쌍의 나열:  
$(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),\dots$

설명: 각 자연수마다 두 원소가 짝을 이뤄 번갈아 나오므로 전체 열은 연속된 $\omega$개의 블록으로 이루어지고, 그 길이는 자연수열의 길이와 같다. 따라서
$2\times\omega \cong \omega$.

$\omega \times 2$
$\omega \times 2 = o(\mathbb{N}\times\{0,1\})$

순서쌍의 나열:
$(0,0),(1,0),(2,0),\dots,(0,1),(1,1),(2,1),\dots$

설명: 먼저 $(\_,0)$에 해당하는 모든 항목이 모두 나열된 뒤에 $(\_,1)$이 이어지므로 두 개의 $\omega$열이 이어진 꼴이다. 따라서
$\omega\times 2 \cong \omega + \omega$.

## 🧩 연습문제

1. 실수집합 $\{1/n \mid n \in \mathbb{N}\}$의 최대·최소·상한·하한 구하기.

2. 전순서집합 $([0,1], \le)$이 정렬집합이 아님을 설명.

3. 서수 $\alpha, \beta, \gamma$에 대해

   * $\beta + \alpha = \gamma + \alpha \to \beta = \gamma$  성립 여부
   * $\alpha \gamma = \beta \gamma \Rightarrow \alpha = \beta$ 성립 여부.

4. 네 명제 중 거짓인 것 찾고 이유 제시:
   $$
   \alpha < \beta \Rightarrow \gamma + \alpha < \gamma + \beta
   $$
   $$
   \alpha < \beta \Rightarrow \alpha + \gamma < \beta + \gamma
   $$
   $$
   \alpha < \beta \Rightarrow \gamma\alpha < \gamma\beta
   $$
   $$
   \alpha < \beta \Rightarrow \alpha\gamma < \beta\gamma
   $$