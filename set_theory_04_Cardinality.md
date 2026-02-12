# 1. 집합의 분류

## (1) 유한집합과 무한집합

### ① 동등(equivalence)

두 집합 $X, Y$에 대하여 전단사함수
$$
f: X \to Y
$$
가 존재하면 $X$와 $Y$는 **동등(equinumerous)** 하다고 한다.
즉,
$$
X \approx Y \quad \text{또는} \quad f: X \leftrightarrow Y
$$

### ② 유한집합, 무한집합
(진부분집합이란 부분집합 중, 자기 자신을 제외한 부분집합)  
* 집합 $X$의 적당한 **진부분집합 $Y$** 가
  $X$와 동등하면 $X$는 **무한집합**이라 한다.
* 그렇지 않으면 $X$는 **유한집합**이라 한다.

예시:
$$
(0,1) \approx \mathbb{R} \Rightarrow \mathbb{R} \text{은 무한집합이다.}
$$

### ③ 여러 가지 정리
1. 공집합은 유한집합이다.
2. 무한집합을 포함하는 집합은 무한이다.
3. 유한집합의 부분집합은 유한이다.
4. 전단사함수 $f: X \to Y$에 대하여
   $X$가 무한집합이면 $Y$도 무한집합이고,
   $Y$가 유한집합이면 $X$도 유한집합이다.
5. 무한집합 $X$의 부분집합 $Y$가 유한이면
   $X - Y$는 무한집합이다.

## (2) 가부번, 비가부번집합
가:가능 부:붙이다 번:번호  
### ① 가부번집합(Denumerable set)

집합 $X$가 $X \approx \mathbb{N}$일 때,
$X$를 **가부번집합(Denumerable set)** 이라 한다.

### ② 가산집합(Countable set)
가부번 집합 확장개념:  
유한집합이나 가부번집합을 **가산집합**이라 한다.
무한집합만 대상으로 함.  

### ③ 여러 가지 정리
1. 가산집합의 부분집합은 가산집합이다.
2. 가부번집합들의 합집합은 가부번집합이다.
3. $\mathbb{N} \times \mathbb{N}$은 가부번집합이다.
4. $\mathbb{Q}$는 가부번집합이다.
5. $\mathbb{R}$의 부분집합 $(0,1)$은 비가부번집합이다.
  - 귀류법으로 증명
  $$ \text{가정: } \exists f:\mathbb{N}\to(0,1),; f \text{ 전단사.} \\
  f(n)=0.a_{n1}a_{n2}a_{n3}\dots, \quad ; a_{nm}\in{0,1,\dots,9}.
  $$
  $$
  \text{Let z} \in (0, 1), z=0.z_1z_2z_3\dots,\quad z_k\neq a_{kk};(\forall k\in\mathbb{N})
  $$
  $$
  \Rightarrow z\neq f(k);(\forall k\in\mathbb{N})
  \Rightarrow z\notin f(\mathbb{N})
  $$
  $$
  \therefore f \text{ 전단사 아님 } \Rightarrow (0,1)\text{ 비가산.}
  $$

  - 이렇게 하나씩 원소가 어긋남을 비교하는 방법을 칸토어의 대각법이라고 함

6. 모든 무리수의 집합 $\mathbb{I}$는 비가부번집합이다.
7. 복소수 집합 $\mathbb{C}$는 비가부번집합이다.
8. 임의의 무한집합은 가산무한집합을 부분집합으로 갖는다.
9. 가산집합의 가산합집합은 가산집합이다.


# 2. 기수(Cardinal Number)
## (1) 기수의 개념
### ① 정의
집합의 크기를 나타내는 수로,
$$
\text{card } A \text{ 또는 } \#A
$$

* 각 집합에 대해 $\#A$는 유일하다
* #A에 해당하는 집합 A는 항상 존재한다
* 유한집합의 경우 $\#A = k, (k \in \mathbb{N})$
* $A \approx B \iff \#A = \#B$

### ② 유한기수와 초한기수
* **유한기수**: 유한집합의 기수
* **초한기수**: 무한집합의 기수

대표적 초한기수:
$$
\#\mathbb{N} = \aleph_0, \quad \#\mathbb{R} = \mathfrak{c}
$$

### ③ $\#A \lt \#B$
: A는 B의 한 부분집합과 동등이고, B는 A의 어떠한 부분집합과도 동등이지 않다.  

1. $\#A \leq \#B \leftrightarrow $ $A$에서 $B$로의 단사함수가 존재
2. 칸토어 번슈타인 정리: A가 B의 부분집합과 동등이고, B도 A의 부분집합과 동등이면 $A \approx B$ ($\#A = \#B$)
3. $\#A \leq \#B, \#B \leq \#C \Rightarrow \#A \leq \#C$

## (2) 기수의 연산

### ① 기수 합
서로소인 두 집합 $A, B$의 기수를 각각 $a, b$라 할 때,
$$
a + b = \#(A \cup B)
$$

### ② 기수 곱

집합 $A, B$의 기수를 각각 $a, b$라 할 때,
$$
a \times b = \#(A \times B)
$$

### ③ 연산 법칙

기수 $x, y, z$에 대해

* 교환법칙: $x + y = y + x$, $xy = yx$
* 결합법칙: $(x + y) + z = x + (y + z)$, $(xy)z = x(yz)$
* 분배법칙: $x(y + z) = xy + xz$

### ④ 여러 가지 정리

1. $\aleph_0 + \aleph_0 = \aleph_0$
  - $\aleph_0 + 1= \aleph_0$
2. $\mathfrak{c} + \mathfrak{c} = \mathfrak{c}$
3. $\aleph_0 + \mathfrak{c} = \mathfrak{c}$
4. $\aleph_0 \aleph_0 = \aleph_0$
5. $\mathfrak{c}\mathfrak{c} = \mathfrak{c}$
6. $\aleph_0{\mathfrak{c}} = \mathfrak{c}$

## (3) 기수의 지수

### ① 정의

집합 $A, B$에 대하여
$$
\#A = m, \#B = n \text{일 때 } \#(B^A) = n^m
$$
단, $B^A = \{ f \mid f : A \to B \}$

예시:
$B = {0,1}$일 때,
$$
B^A = {0,1}^A = 2^{A}
$$

### ② 여러 가지 정리

1. 집합 $X$에 대하여 $\#X = x$일 때,
   $$
   \#\mathcal{P}(X) = 2^x
   $$
2. 기수 $x, y, z$에 대하여
   $$
   (x^y)(x^z) = x^{y+z}, \quad (x^{y})^z = x^{yz}, \quad (xy)^z = x^z y^z
   $$
3. 중요: $\mathfrak{c} = {\aleph_0}^{\aleph_0} = \mathfrak{c}^{\aleph_0}$
4. 중요: $2^{\mathfrak{c}} = {\aleph_0}^{\mathfrak{c}} = \mathfrak{c}^{\mathfrak{c}}$

* 기수를 지수로 올리면 더 큰 집합이, 더 큰 기수가 된다.
* 연속체 가설과 연결됨! 다음에 나옴

# 연습문제
1. $A={x,y,z}, B={a,b}$일 때
   $\#A, \#B, \#(A\times B), \#(B^A)$를 구하시오.

2. $\mathbb{Z}, \mathbb{N}, \mathbb{Q}, \mathbb{R}$의 유한/무한 여부를 판정하시오.

3. $\mathbb{Z} \times \mathbb{N}, \mathbb{Z} \times \mathbb{Q}, \mathbb{Q} \times \mathbb{Q}$가 가부번집합임을 증명하시오.

4. $\#(\mathbb{R}-\mathbb{Q}) = \mathfrak{c}$임을 증명하시오.
