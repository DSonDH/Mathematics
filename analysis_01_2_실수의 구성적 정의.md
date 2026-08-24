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

### Def 2. 순서(ordering),  순서체(ordered field)의 정의

순서(ordering)은 $\leq$로 나타내고, 다음 세가지 성질을 만족한다
- 임의의 $x, y \in F$, $x \leq y$ 또는 $y \leq x$ 중 적어도 하나는 참이다.
- $x \leq y$ 이고 $y \leq x$ 이면 $x = y$ 이다.
- $x \leq y$ 이고 $y \leq z$ 이면 $x \leq z$ 이다.


다음 성질을 만족하는 체 $F$를 순서체(ordered field)라 한다.

* $x,y,z\in F,\ y \leq x \Rightarrow y+z \leq x+z$
* $x,y\in F,\ 0 \leq x\land \ 0 \leq y \Rightarrow 0 \leq xy$

**Thm 2.** $\mathbb{Q}$는 순서체이다.

# 2. 실수체계
## (1) 실수의 구성
### Def 1. (실수의 구성적 정의 — 데데킨트 절단)
"다음 성질을 만족하는 $\mathbb{Q}$의 부분집합 $C$ (cut, 절단)를 실수라 하고, $C$들의 집합을 $\mathbb{R}$로 표현한다."

다음 세 조건을 만족하는 집합 $C\subseteq\mathbb{Q}$를 데데킨트 절단이라고 한다.

1. $C$는 공집합도 아니고 $\mathbb{Q}$ 전체도 아니다.
   $C\ne\varnothing,\quad C\ne\mathbb{Q}.$ 
2. $C$는 아래쪽으로 닫혀 있다.
   $s\in C, t \in \mathbb Q, \ t<s\implies t\in C.$ 
3. $C$에는 최댓값이 없다.
   $s\in C\implies\exists u\in C\ \text{such that}\ s<u.$ 

### 예제
절단의 정의는 다음 세 조건이다.

$$
\begin{aligned}
&\text{(c1)}\quad A\neq\varnothing,\qquad A\neq\mathbb Q,\\
&\text{(c2)}\quad r\in A,\ q<r\Longrightarrow q\in A,\\
&\text{(c3)}\quad A\text{는 최댓값을 갖지 않는다.}
\end{aligned}
$$

(a) $C_r=\{t\in\mathbb Q:t<r\}$ 가 절단임을 보여라. 여기서 $r\in\mathbb Q$를 고정한다.

(c1) 공집합도 아니고 $\mathbb Q$ 전체도 아님

$r-1<r$ 이므로 $r-1\in C_r$이다. 따라서 $C_r\neq\varnothing$이다. 한편 $r<r$ 는 거짓이므로 $r\notin C_r.$ 따라서 $C_r\neq\mathbb Q$이다.

(c2) 아래쪽으로 닫혀 있음

$t\in C_r$이고 $q<t$라고 하자. $t\in C_r$ 이므로 $t<r$ 이다. 따라서 $q<t<r$ 이므로 $q<r$, 즉 $q\in C_r$이다.

(c3) 최댓값이 없음

임의의 $t\in C_r$를 택한다. 그러면 $t<r$이다. 다음 유리수를 생각하자. $s=\frac{t+r}{2}.$ $t,r\in\mathbb Q$이므로 $s\in\mathbb Q$이다. 또한 $t<r$이므로 $t<\frac{t+r}{2}<r.$

따라서 $s\in C_r$이면서 $t<s$이다. 즉, $C_r$ 의 어떤 원소를 택하더라도 그보다 큰 $C_r$의 원소가 존재한다. 그러므로 $C_r$에는 최댓값이 없다.

이상으로 $C_r$는 절단이다.

---

(b) $S=\{t\in\mathbb Q:t\le 2\}$

절단이 아니다.

실제로 $2\in S$이고, 모든 $t\in S$에 대하여 $t\le 2$이다. 따라서 $2$가 $S$의 최댓값이다. 즉, $S$는 조건 (c3)을 만족하지 않는다.

참고로 $S$는 (c1), (c2)는 만족하지만 (c3)만 만족하지 않는다. 절단 $C_2$는 $C_2=\{t\in\mathbb Q:t<2\}$ 이며, $2$를 포함하지 않는다는 점이 중요하다.

---

(c) $T=\{t\in\mathbb Q:t^2<2\text{ 또는 }t<0\}$

$T$는 절단이다.

이 집합은 직관적으로 유리수 중에서 $\sqrt2$보다 작은 수들을 모은 집합이다. 다만 아직 실수 $\sqrt2$를 구성하기 전이므로 정의에는 $\sqrt2$를 직접 사용하지 않는다.

(c1) 공집합도 아니고 $\mathbb Q$ 전체도 아님

$(0^2=0<2)$이므로 $0\in T.$ 따라서 $T\neq\varnothing$이다. 한편 $2>0$이고 $2^2=4>2$이므로 $2\notin T.$ 따라서 $T\neq\mathbb Q$이다.

(c2) 아래쪽으로 닫혀 있음

$t\in T$이고 $q<t$ 라 하자. 두 경우로 나눈다. 

- 경우 1: $t<0$.  
그러면 $q<t<0$ 이므로 $q<0$이다. 따라서 $q\in T$이다.

- 경우 2: $t\ge0$  
$t\in T$이고 $t<0$은 아니므로 $t^2<2$이다.

   * $q<0$이면 정의에 의해 $q\in T$이다.
   * $q\ge0$이면 $0\le q<t$이므로 $q^2<t^2<2$

따라서 역시 $q\in T$이다. 그러므로 $T$는 (c2)를 만족한다.

(c3) 최댓값이 없음

임의의 $t\in T$를 택한다.

- 경우 1: $t<0$ 

$s=\frac t2$ 라고 두면 $t<0$이므로 $ t<\frac t2<0.$ 따라서 $s\in T$이고 $t<s$ 이다.

- 경우 2: $t\ge0$

이 경우 $t^2<2$ 이다. 다음과 같이 양의 유리수 $\varepsilon$ 을 잡는다.

$$ 0<\varepsilon < \min \{1,\frac{2-t^2}{2t+1} \}.$$

$t\in\mathbb Q$ 이고 $t^2<2$ 이므로 이러한 유리수 $\varepsilon$을 잡을 수 있다. $s=t+\varepsilon$ 이라고 하면 $s>t$ 이고 $s^2 =(t+\varepsilon)^2 =t^2+2t\varepsilon+\varepsilon^2.$ $\varepsilon<1$ 이므로 $\varepsilon^2<\varepsilon$ 이고, 따라서

$$ s^2<t^2+(2t+1)\varepsilon<2.$$

그러므로 $s\in T$이고 $t<s$ 이다.

모든 $t\in T$보다 큰 T의 원소가 존재하므로 T에는 최댓값이 없다. 따라서

$$ \boxed{T\text{는 절단이다.}}$$

이 절단은 이후 실수 체계에서 $\sqrt2$를 나타내는 실수가 된다.

---

(d) $U={t\in\mathbb Q:t^2\le2\text{ 또는 }t<0}$

결론적으로 U도 절단이다. 핵심은 유리수 중에는 $t^2=2$를 만족하는 수가 없다는 사실이다.

이를 확인하자. 유리수 $t$가 $t^2=2$를 만족한다고 가정하고, 서로소인 정수 $p,q$에 대하여

$$ t=\frac pq,\qquad q\neq0$$

라고 쓰자. 그러면

$$ \frac{p^2}{q^2}=2
\quad\Longrightarrow\quad
p^2=2q^2.$$

따라서 $p^2$가 짝수이므로 $p$가 짝수이다. $p=2k$라고 놓으면

$$ 4k^2=2q^2
\quad\Longrightarrow\quad
q^2=2k^2.$$

따라서 $q$도 짝수이다. 이는 $p,q$가 서로소라는 것과 모순이다. 그러므로

$$ t\in\mathbb Q\Longrightarrow t^2\neq2.$$

따라서 유리수 $t$에 대해서는

$$ t^2\le2\quad\Longleftrightarrow\quad t^2<2$$

이다. 그러므로

$$ \begin{aligned}
U
&={t\in\mathbb Q:t^2\le2\text{ 또는 }t<0}\
&={t\in\mathbb Q:t^2<2\text{ 또는 }t<0}\
&=T.
\end{aligned}$$

$c$에서 $T$가 절단임을 보였으므로

$$ \boxed{U\text{도 절단이다.}}$$


(참고)
* $s\notin C \Rightarrow \not \exists t\in C$ s.t. $t > s$
* $r \in C \land s \not\in C \Rightarrow r < s$

### Def 2. (실수의 순서)
실수 집합 $C, D \in \mathbb{R}$에 대하여
* $C \subset D$ 이면 $C < D$
* $C \subseteq D \ \land \ C \supseteq D$ 이면 $C = D$


> **$A \leq B$ 는 $A \subseteq B$를 의미한다**
>
> 이 정의가 순서(ordering)의 세가지 성질을 만족하는지 확인해보자.
>

#### 예제.

$\mathbb{R}$에서 순서를 다음과 같이 정의한다.

$$
A\le B\quad\Longleftrightarrow\quad A\subseteq B.
$$

이 정의가 순서의 성질 (o1), (o2), (o3)을 만족함을 보이자.

**(o1) 비교 가능성**

정의에 따라 이는 $A\subseteq B\quad\text{또는}\quad B\subseteq A$ 임을 보이는 것과 같다.

반대로, 둘 다 성립하지 않는다고 가정하자. 그러면 다음을 만족하는 유리수 $a,b$가 존재한다. $a\in A\setminus B, b\in B\setminus A.$ 즉, $a\in A,\quad a\notin B,$ 이고 $b\in B,\quad b\notin A$ 이다.

앞선 예제의 결과를 절단 $A$에 적용하면 $a\in A,\quad b\notin A \Rightarrow a < b$ 이다. $B$에 적용하면 $b\in B,\quad a\notin B \Rightarrow b<a$ 이다.

따라서 $a<b \land b<a$ 가 성립해야 하는데, 이는 모순이다. 그러므로 처음 가정이 잘못되었으며, $A\subseteq B\quad\text{또는}\quad B\subseteq A$ 이다. 

**(o2) 반대칭성**

$A\le B$이고 $B\le A$라고 하자. 순서의 정의에 의해 $A\subseteq B,\ B\subseteq A$ 이다. 집합의 상호 포함 관계에 의해 $A=B$ 이다.

**(o3) 추이성**

$A\le B$이고 $B\le C$라고 하자.

순서의 정의에 의해 $A\subseteq B,\ B\subseteq C$ 이다. 집합의 포함 관계는 추이적이므로 $A\subseteq C$ 이다. 따라서 순서의 정의에 의해 $A\le C$ 이다. 


결론적으로 $A\le B$를 $A\subseteq B$로 정의하면, 이 관계는 $\mathbb{R}$에서 (o1), (o2), (o3)을 모두 만족한다.


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
- 여기선 공리로 보지 않는다.
- 증명은 동영상 강의 참고. 별로 안 김

### Thm 3. (실수의 조밀성, density of the real numbers)
$\forall A,B\in\mathbb{R},\ A<B \Rightarrow \exists C\in\mathbb{R}\ \text{s.t.}\ A<C<B.$
- 증명은 동영상 강의 참고. 별로 안 김

### Thm. 실수의 존재성
공집합이 아니고, 위로 유계인 집합은 반드시 상한을 가지는 순서체가 존재한다. 또한 이 체는 $\mathbb Q$를 부분체로 포함한다.

- 즉, 실수가 존재한다.