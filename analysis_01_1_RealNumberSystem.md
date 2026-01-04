# 1. 자연수
## (1) 페아노 공리계 (Peano Axioms)
자연수는 다음의 다섯 공리로 이루어진 페아노 공리계를 만족하는 수체계이다.

1. $1 \in \mathbb N$
2. $n \in \mathbb N \Rightarrow n' \in \mathbb N$  ( $n'$는 그 다음 수 )
3. $\forall n \in \mathbb N, n' \ne 1$
4. $\forall n,m \in \mathbb N, n' = m' \Rightarrow n = m$
  - 2 -> 3 -> 4 -> 2 같은 순환되는 구조를 방지하기 위함
5. $1 \in S \land (\forall n \in S, n' \in S) \Rightarrow \mathbb N \subseteq S$
  - 귀납접 공리
   (1과 그 다음 수는 무정의 용어(primitive term)이다)

### Theorem(줄여서 Thm.) 수학적 귀납법 (Mathematical Induction)

$n'=n+1$이라 정의할 때, 명제 $P(n)$에 대하여

1. $P(1)$이 참
2. $P(n)$이 참이면 $P(n+1)$이 참

이 성립하면 $P(n)$은 모든 자연수 $n$에 대하여 참이다.

#### 증명
페아노 공리 5번을 이용한다.

$S = \{ n \in \mathbb{N} \mid P(n) \text{이 참} \}$ 라고 하자.

조건 1에 의해 $P(1)$이 참이므로 $1 \in S$.

조건 2에 의해 $P(n)$이 참이면 $P(n+1)$이 참이므로, 
$n \in S \Rightarrow n' = n+1 \in S$.

따라서 페아노 공리 5번에 의해 $\mathbb{N} \subseteq S$.

그런데 $S \subseteq \mathbb{N}$ (정의에 의해) 이므로 $S = \mathbb{N}$.

즉, 모든 자연수 $n$에 대하여 $P(n)$이 참이다. $\blacksquare$


## (2) 자연수의 성질
**페아노 공리 5가지를 만족하는 무었이던 자연수라고 정의한다.**  

1. 정렬성 (well-ordering principle)
   자연수의 부분집합은 항상 최소원소를 가진다.

2. 자연수집합 $\mathbb N$은 위로 유계가 아니다.
  - 즉, 상계가 없다. 자연수를 벗어난 더 큰 값이 없다

3. 아르키메데스 성질 (Archimedean property)
   $$
   \forall \varepsilon > 0, \exists n \in \mathbb N \text{ such that } \frac{1}{n} < \varepsilon.
   $$
   - 증명: 위에 유계 성질, 귀류법 활용해서 쉽게 증명 가능

# 2. 유리수와 무리수
## (1) 집합의 구성
* 정수집합: $\mathbb Z = (-\mathbb N)\cup{0}\cup \mathbb N$
* 유리수집합:
$$\mathbb Q = \left\{ \frac{m}{n} \middle| m,n \in \mathbb Z, n\ne 0 \right\} $$
* 무리수집합: $\mathbb I = \mathbb R - \mathbb Q$
  * 실수집합 갑툭튀?: 자연수 집합은 페아노 공리계로. 실수집합은 또 다른 공리계로 설명가능함. 그래서 실수집합은 따로 정의하게 되서 갑툭튀임.
  * (자연수 집합으로 실수집합을 만들 수는 있는데 나중에 나올거임)

## (2) 조밀성 (Density)
### Thm 1. 유리수의 조밀성
$$
\forall a,b \in \mathbb R, a<b \Rightarrow \exists q \in \mathbb Q \text{ such that } a<q<b.
$$
어떻게든 작은 간격을 잡아도, 그 사이에 또 다른 유리수가 존재함.  

#### 증명
**case 1) $0<a<b$**  
$a<b \Longleftrightarrow 0<b-a$  
$\Rightarrow \exists n\in\mathbb N\ \text{s.t.}\ \dfrac{1}{n} < b-a$  (아르키메데스 성질)

Let $S=\{ m\in\mathbb N \mid m>na \}$

Then $na>0$ 이므로 $S\neq\varnothing$ (자연수는 위로 유계가 아님).  
정렬성에 의해 $S$는 최소원소를 가진다: $\min S = m$.

$m$이 $S$의 최소원소이므로 $m-1 \notin S$ 이고, 따라서 $m-1 \le na$  
$\Leftrightarrow \dfrac{m-1}{n} \le a$.

이제
$$
a < \frac{m}{n} \le a+\frac{1}{n} < b
$$
가 된다. (마지막 부등식은 $\frac{1}{n} < b-a$ 사용)

따라서
$$
\exists q=\frac{m}{n}\in\mathbb{Q}\ \text{s.t.}\ a<q<b.
$$

**case 2) $a<0<b$**  
$0$은 유리수이므로 $a<0<b$ 에서 바로 $q=0\in\mathbb Q$ 가
$a<q<b$ 를 만족한다.

**case 3) $a<b<0$**  
부호를 바꾸면 $0<-b<-a$이므로 $-b$와 $-a$는 양수이고 $-b < -a$ 이다.

따라서 case 1을 $(-b,-a)$에 적용하면
$$
\exists r\in\mathbb Q \ \text{s.t.}\ -b<r<-a.
$$

유리수는 음수에 대해서도 닫혀 있으므로 $q=-r \in \mathbb Q$ 이고,
부호를 다시 바꾸면
$$
a< -r < b.
$$

즉
$$
\exists q\in\mathbb Q\ \text{s.t.}\ a<q<b.
$$

세 경우를 모두 합쳐서
$\forall a,b\in\mathbb R,\ a<b \Rightarrow \exists q\in\mathbb Q$ with $a<q<b$ 가 성립한다.
따라서 유리수의 조밀성이 증명된다.
### Thm 2. 무리수의 조밀성
$$
\forall a,b \in \mathbb R, a<b \Rightarrow \exists \alpha \in \mathbb I \text{ such that } a<\alpha<b.
$$
어떻게든 작은 간격을 잡아도, 그 사이에 또 다른 무리수가 존재함.  

#### 증명  
$a<b \Longleftrightarrow a+\sqrt{2} < b+\sqrt{2}$  
유리수의 조밀성에 의해
$$
\exists r\in\mathbb Q\ \text{s.t.}\ a+\sqrt{2} < r < b+\sqrt{2}.
$$

Let $s = r - \sqrt{2}$  
: $r\in\mathbb Q$ 이고 $\sqrt{2}\notin\mathbb Q$ 이므로
$s = r - \sqrt{2} \in \mathbb I.$

부등식에 $-\sqrt{2}$ 를 더하면
$$
a < r-\sqrt{2} < b
$$

즉,
$$
a < s < b.
$$

따라서 $a$와 $b$ 사이에 무리수 $s$가 존재한다 $\blacksquare$

**Note:** $\sqrt{2}$는 임의의 무리수 상수를 대표하는 예시이다.  
다른 무리수(예: $\pi$, $e$, $\sqrt{3}$ 등)를 사용해도 동일하게 증명된다.
$$
\forall a,b \in \mathbb R, a<b \Rightarrow \exists \alpha \in \mathbb I \text{ such that } a<\alpha<b.
$$

# 3. 실수
## (1) 체 공리 (Field Axioms)
집합 $S$ 위의 두 이항연산 $+$, $\cdot$이 다음 9개의 공리를 만족하면, 대수구조 $(S,+,\cdot)$를 체라고 한다.  
(덧셈, 곱셈은 닫혀있다 가정)  

덧셈의 교환법칙, 결합법칙, 항등원, 역원  
1. $x,y \in S \Rightarrow x+y = y + x$
2. $x,y,z \in S \Rightarrow x+(y+z) = (x+y)+z$
3. $\forall x \in S, \exists 0 \in S$ s.t. $0+x = x$
4. $\forall x \in S, \exists (-x) \in S$ s.t. $x+(-x)=0$ 

<br>  
곱셈의 교환법칙, 결합법칙, 항등원, 역원  

5. $x,y \in S \Rightarrow x\cdot y = y\cdot x$
6. $x,y,z \in S \Rightarrow x(yz) = (xy)z$
7. $\exists 1 \in S, 1\ne 0$ s.t. $1\cdot x = x$
8. $\forall x\ne 0, \exists x^{-1} \in S$ s.t. $xx^{-1} = 1$

<br>  
곱셈, 덧셈의 분배법칙  

9. $x,y,z \in S  \Rightarrow x(y+z) = xy + xz$

근데, 대수적으로 유리수, 실수는 동일한 구조이다...  
$(\mathbb Q,+,\cdot)$, $(\mathbb R,+,\cdot)$은 모두 체이다.  
유리수체, 실수체라 부름.  

## (2) 순서 공리 (Order Axioms)
### 1) 순서 공리
$\mathbb R$에는 다음 두 조건을 만족하는 공집합이 아닌 부분집합 $P$가 존재한다.

1. $\forall x,y \in P, x+y \in P \land xy \in P$
2. 임의의 실수 $x$에 대하여 다음 중 하나만 성립한다.  
   i) $x \in P$  
   ii) $x=0$  
   iii) $-x \in P$  

이 P는 양의실수집합을 정의하고 있음.  

### 2) 삼분성질 (Trichotomy)
Def. [부등식의 정의]  
임의의 $a,b \in \mathbb R$에 대하여
1. $a-b \in P \Rightarrow a>b \ \lor \ b <a$
2. $a-b \in P \cup \{0\} \Rightarrow b\geq a \ \lor \ b \leq a$

Thm. [삼분성질]  
임의의 $a,b \in \mathbb R$에 대하여 아래 중 정확히 하나가 성립한다.  
i) $a>b$  
ii) $a=b$  
iii) $a<b$  
(순서공리 2번째로 증명 가능.)  

## (3) 완비성 공리 (Axiom of Completeness)
완비성: 모든 상한을 갖는 성질  
(조밀성: 두 수 사이에 항상 다른 수가 존재하는 성질)  

조밀해도 완비가 아닐 수 있다: 유리수  
완비여도 조밀하지 않을 수 있다: 정수  
### 1) 완비성 공리 (다른말로, 상한 공리)
$\mathbb R$의 공집합이 아닌 부분집합이 위로 유계이면 그 부분집합은 상한을 갖는다.  

Def. [상한]  
부분순서집합 A의 부분집합 B의 상계들의 집합이 최소원소를 가질때,  
그 최소원소를 B의 상한이라하고, $supB$로 나타낸다. (supremum)  

(빽빽함 (완전 들어참)을 정의하는 여러 버전 중 하나임.)  
유리수집합은 이 공리를 만족하지 않음을 아래서 보임.  

### 2) 주요 정리
Thm 1. 상한은 유일하다.  

---
Thm 2. $s\in\mathbb{R}$가 집합 $S$의 상계일 때, 다음 세 명제는 동치이다.  
1. $s = \sup S$
2. $\forall \varepsilon > 0,\ \exists x\in S \text{ such that } s-\varepsilon < x \le s$
3. $\forall \varepsilon > 0,\ S\cap (s-\varepsilon]  \neq \varnothing$

(Thm2 증명)  
**(1) ⇒ (2)**  
2번 식의 부정을 가정하면
$\forall x\in S, x \le s-\varepsilon.$  
즉, $s-\varepsilon$ 은 $S$의 상계가 된다.  
그러나 $s-\varepsilon < s = \sup S$ 이므로 모순.  
따라서
$\exists x\in S \text{ such that } s-\varepsilon < x \le s.$

**(2) ⇒ (1)**  
Let $s \ne \sup S \Rightarrow \sup S < s.$  
이제
$\varepsilon = \dfrac{s - \sup S}{2} > 0$
로 둔다.

가정 (2)에 의해
$\exists x\in S \text{ such that } x > s-\varepsilon.$

그런데 
$$
s-\varepsilon
= s - \frac{s-\sup S}{2}
= \frac{s+\sup S}{2}
> \sup S.
$$

따라서 $x > s-\varepsilon > \sup S.$

이는 상계의 정의 $\forall y\in S, y \le \sup S$ 와 모순.

따라서 $s = \sup S.$

---
Thm 3. $\mathbb{Q}는 완비성을 갖지 않는다$  

#### 증명
반례를 제시한다.  
집합 $S = \{ x \in \mathbb{Q} \mid x > 0 \land x^2 < 2 \}$ 를 생각하자.

**Step 1. $S$는 위로 유계이다.**  
$\forall x \in S, x^2 < 2 < 4 = 2^2$ 이므로 $x < 2$.  
따라서 $2$는 $S$의 상계이다.

**Step 2. $S$가 $\mathbb{Q}$에서 상한을 가진다고 가정하자.**  
즉, $\sup_{\mathbb{Q}} S = r \in \mathbb{Q}$ 라 하자.

**Step 3. $r^2 = 2$임을 보인다.**  
- $r^2 < 2$ 라 가정하면, 충분히 작은 $\varepsilon > 0$에 대해
   $(r+\varepsilon)^2 < 2$ 가 되도록 할 수 있다.  
   그러면 $r+\varepsilon \in S$ 이므로 $r < r+\varepsilon$ 인 원소가 $S$에 존재한다.  
   이는 $r$이 상계라는 가정에 모순.

- $r^2 > 2$ 라 가정하면, 충분히 작은 $\varepsilon > 0$에 대해
   $(r-\varepsilon)^2 > 2$ 가 되도록 할 수 있다.
   
   그러면 $\forall x \in S, x^2 < 2 < (r-\varepsilon)^2$ 이므로 $x < r-\varepsilon$.
   
   즉, $r-\varepsilon$이 $r$보다 작은 상계가 되어 모순.
따라서 $r^2 = 2$.

**Step 4. 모순 도출**  
그런데 연습문제 2에서 제곱하여 2가 되는 유리수는 존재하지 않는다.  
따라서 $r \notin \mathbb{Q}$ 이므로 모순.

**결론**  
$S$는 $\mathbb{Q}$에서 위로 유계이지만 상한을 갖지 않는다.  
따라서 $\mathbb{Q}$는 완비성 공리를 만족하지 않는다. $\blacksquare$

## 3) 완비성의 예 – 무한소수
위로 유계인 임의의 무한소수 부분집합 $A$를 생각한다.
$$
a_0 = \max \{ x_0 \mid x_0.x_1x_2x_3\ldots \in A \} \\ 
a_1 = \max \{ x_1 \mid a_0.x_1 x_2 x_3 \cdots \in A \} \\
\dots \\
a_k = \max \{ x_k \mid a_0.a_1 \dots a_{k-1} x_{k} x_{k+1} \cdots \in A \} \\
$$

이를 계속하면 무한소수 $a_0.a_1a_2\ldots$는 집합 $A$의 상한이 된다.
즉, 무한소수의 집합은 완비성 공리를 만족한다.

이는 직관적인 정의이고, 수학적으로 더 엄밀한 정의는 따로 있긴 함.  

# 연습 문제
1. (베르누이의 부등식) 다음 명제를 증명하시오.  
   $\forall n \in \mathbb N, h>-1 \Rightarrow (1+h)^n \ge 1+nh$

2. 제곱하여 2가 되는 유리수는 존재하지 않음을 증명하시오.

3. $0<1$임을 증명하시오.

4. 개구간 $(0,1)$의 상한이 1임을 증명하시오.

5. $x \in \mathbb R$일 때 다음 명제를 증명하시오.  
   $\forall \varepsilon >0, 0\leq x<\varepsilon \Rightarrow x=0$
