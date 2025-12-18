페아노 공리계가 아닌, 집합으로 새로 정의하는 버전임.  

# 1. 유리수체계
## (1) 자연수의 구성
### Def 1. (자연수의 구성적 정의)
자연수집합 $N$의 원소는 다음과 같이 정의한다.
* $1 = {\varnothing}$
* $2 = 1 \cup \{1\} = \{\varnothing, \{\varnothing\}\}$
* $3 = 2 \cup \{2\}$
* $n' = n \cup \{n\}$ 

폰노이만 방식임. 러셀의 방식 등등 다른 방법들 많음.  

### Def 2. (자연수의 순서)
임의의 집합 $n,m\in N$에 대하여

* $n \subset m \Rightarrow n < k$
* $n \subseteq m \ \land \ n \supseteq m \Rightarrow n = m$

### Def 3. (자연수의 연산)
임의의 $n,m\in N$에 대하여

1. $n+1 = n'$이고 $n + m' = (n + m)'$
3. $(n+m)-n = m$  
  - (큰 자연수에서 작은 자연수를 빼야함)
4. $n\times 1 = n$이고 $n\times m' = n\times m + n$

**Thm.** $N$은 전순서집합이다.

## (2) 정수의 구성
### Def 1. (정수의 구성적 정의)
$N\times N$의 동치관계

$$
E : (m,n) E (m',n') \iff m+n' = m'+n
$$
의 동치류 
$$
[(m,n)] =
\begin{cases}
n - m, & m < n, \\
0, & m = n, \\
-(m - n), & m > n.
\end{cases}
$$

를 정수라 하며, 이들의 집합을 $\mathbb{Z}$로 표현한다.

- [(m,n)]라고 하면, (m,n)연산 결과와 같은 모든 (?, ?)들을 일컫는말임. 그래서 []로 감싸서 동치류 라고 함. set theory 강좌에도 있는 기호임.
- 즉 정수는 자연수 순서쌍의 동치류.

### Def 2. (정수의 연산)
두 정수 $a=[(a_1,a_2)], b=[(b_1,b_2)]$ 에 대해

* $a + b = [(a_1+b_1,\ a_2+b_2)]$
* $a - b = [(a_1+b_2,\ a_2+b_1)]$
* $a \times b = [(a_1b_2 + a_2b_1,\ a_1b_1 + a_2b_2)]$

**Thm.** $\mathbb{Z}$는 환(ring)이다.

## (3) 유리수의 구성
### Def 1. (유리수의 구성적 정의)
$\mathbb{Z}\times(\mathbb{Z}-\{0\})$의 동치관계
$$
E : (a,b)E(a',b') \iff a b' = a' b
$$
의 동치류 $[(a,b)]$를 유리수라 하며, 이들의 집합을 $\mathbb{Q}$ 로 표현한다.

### Def 2. (유리수의 연산)
두 유리수 $a=[(a_1,a_2)],\ b=[(b_1,b_2)]$ 에 대해

* $a + b = [(a_1 b_2 + a_2 b_1,\ a_2 b_2)]$
* $a \cdot b = [(a_1 b_1,\ a_2 b_2)]$
* $a \div b = \frac ab = [(a_1 b_2,\ a_2 b_1)]$  (단, $b\ne 0$)

**Thm 1.** $\mathbb{Q}$는 체(field)이다.

### Def 2. (ordered field(순서체)의 정의)

다음 성질을 만족하는 체 $F$를 순서체(ordered field)라 한다.

* $x,y,z\in F,\ y<x \Rightarrow y+z < x+z$
* $x,y\in F,\ 0 < x\land \ 0 < y \Rightarrow 0 < xy$

**Thm 2.** $\mathbb{Q}$는 순서체이다.

# 2. 실수체계
## (1) 실수의 구성
### Def 1. (실수의 구성적 정의 — 데데킨트 절단)
"다음 성질을 만족하는 $\mathbb{Q}$의 부분집합 $C$ (cut)를 실수라 하고, $C$들의 집합을 $\mathbb{R}$로 표현한다."

1. $C\ne \varnothing$
2. $s\in C \land t < s \Rightarrow t\in C$
  - 폐구간 정의 (상계 포함)
3. $s\in C\ \Rightarrow \exists u\in C\ \text{s.t. } s<u$
  - 개구간 정의 (상계를 포함하지 않음)
4. $\forall c\in C,\ \exists x\in \mathbb{Q}\ \text{s.t. } c<x$
  - 상계가 존재한다 (위로 유계)

(참고)
* $s\notin C \Rightarrow \not \exists t\in C$ s.t. $t > s$
* $r \in C \land s \not\in C \Rightarrow r < s$

### Def 2. (실수의 순서)
실수 집합 $C, D \in \mathbb{R}$에 대하여
* $C \subset D$ 이면 $C < D$
* $C \subseteq D \ \land \ C \supseteq D$ 이면 $C = D$

## (2) 실수의 덧셈
$C, D \in \mathbb{R}$에 대하여
1. $C + D = \{c+d \mid c\in C,\ d\in D\}$
2. 실수 $0' = \{x\in \mathbb{Q} \mid x < 0\}$ 로 정의한다.
3. $C\in \mathbb{R}$ 에 대해
   $-C = \{d\in \mathbb{Q} \mid \forall c\in C, \exists d \ \ \text{s.t.}\ \ d < d' \ \ \text{with}\ \ c + d' < 0\}$

## (3) 실수의 곱셈
$C, D \in \mathbb{R}$에 대해 $CD$ 혹은 $C \times D$를 다음과 같이 정의한다 (단, $c\in C,d\in D$)
1. $C>0',\ D>0'$ 일 때
   $CD = \{ q \in \mathbb{Q} \mid q < cd \}$
2. $C>0',\ D<0'$ 일 때
   $CD = -\bigl( C(-D) \bigr)$
3. $C<0',\ D>0'$ 일 때
   $CD = -\bigl( (-C)D \bigr)$
4. $C<0',\ D<0'$ 일 때
   $CD = (-C)(-D)$
5. $C = 0'$ 또는 $D = 0'$ 일 때
   $CD = 0'$

* 실수 $1'$은
   $$
   1' = \{ x \in \mathbb{Q} \mid x < 1 \}
   $$
   로 정의한다.

* $C \in \mathbb{R}$에 대해
   $$
   \frac{1}{C}
   = \{ d \in \mathbb{Q} \mid \forall c \in C,\ \exists d \text{ s.t. } d < d' \ \text{with}\ cd' < 1 \}
   $$
   로 정의한다.

# 3. 실수체계의 성질
### Thm 1.
$\mathbb{R}$은 순서체이다.
- 증명은 동영상 강의 참고. 별로 안 김

### Thm 2. (실수의 완비성, completeness of the real numbers)
$\mathbb{R}$의 공집합이 아닌 부분집합이 위로 유계이면 그 부분집합은 상한을 갖는다.
- 증명은 동영상 강의 참고. 별로 안 김

### Thm 3. (실수의 조밀성, density of the real numbers)
$\forall A,B\in\mathbb{R},\ A<B \Rightarrow \exists C\in\mathbb{R}\ \text{s.t.}\ A<C<B.$
- 증명은 동영상 강의 참고. 별로 안 김
