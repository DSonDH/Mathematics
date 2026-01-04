# 확률분포 (Probability Distributions)

## 1. 확률의 뜻과 성질 (Meaning and Axioms of Probability)

### 1.1 표본공간과 사건 (Sample Space and Events)
과학적 실험이나 자연현상 또는 사회현상을 관측할 때,
가능한 모든 관측 결과들의 집합을 **표본공간**
(sample space)이라 하고 이를
$$S \quad (\text{또는 } \Omega)$$
로 나타낸다.

표본공간 $S$의 부분집합인 특정 결과들의 집합을 **사건** (event)이라 한다.  

각 사건 $A$의 가능성을 수량화한 값을 **확률**
(probability)이라 하며, 이를 $P(A)$로 나타낸다.

확률이 엄밀하게 정의되기 전에는 상대도수(relative frequency)의 극한으로 이해되었으나,
러시아의 수학자 **Kolmogorov (1903–1987)** 는 이를 공리적으로 정식화하였다.

### 1.2 사건의 정의 (Definition of Events, σ-field)
표본공간 $S$의 부분집합들 중 다음 조건을 만족하는 집합들의 모임을
**사건의 집합체** (sigma field of events) 또는
$\sigma$-대수라 하며 $\mathcal E$로 나타낸다.

1. **전체사건 (Sure Event)**
   $$S\in\mathcal E$$
  - 표본공간 $S$는 사건이다
2. **여사건 (Complement)**
   $$A\in\mathcal E \Rightarrow A^c\in\mathcal E$$

3. **가산합사건 (Countable Union)**
   $$A_1,A_2,\dots\in\mathcal E
   \Rightarrow
   \bigcup_{n=1}^\infty A_n\in\mathcal E$$
  - 사건들의 합집합도 사건이다

### 1.3 확률의 정의 (Kolmogorov Axioms)
확률은 다음과 같은 함수로 정의된다.

$$P : \mathcal E \to [0,1]$$

여기서 $\mathcal E$는 사건들의 집합이며, 확률은 다음의 공리를 만족한다고 가정한다.

#### 확률의 공리 (Axioms of Probability)
확률의 공리에서 전제하는 것: 사건이란 확률을 정해줄 수 있는 부분집합.

1. **확률의 범위 (Non-negativity)**
   $$\forall A\in\mathcal E, P(A)\ge 0 \quad$$

2. **전체의 확률 (Normalization)**
   $$P(S)=1$$

3. **가산가법성 (Countable Additivity)**  
   서로소 사건 $A_1, A_2, \dots$ 에 대하여
   $$A_i\cap A_j=\varnothing \;(i\neq j)$$
   이면
   $$P(A_1\cup A_2\cup\cdots)
   =P(A_1)+P(A_2)+\cdots$$

### 1.3 확률의 기본 성질 (Basic Properties of Probability)
아래 정리들은 확률의 공리로 유도 가능하다.

#### 정리 1.1.1 (Fundamental Properties of Probability)
(a) 각 사건 $A$에 대하여
$$0\le P(A)\le 1,\qquad P(\varnothing)=0$$

(b) **여사건의 확률 (Complement Rule)**
$$P(A^c)=1-P(A)$$

(c) **단조성 (Monotonicity)**
$$A\subseteq B \;\Rightarrow\; P(A)\le P(B)$$

#### 증명 (Proof)
다음의 집합 관계를 이용한다.
$$A\cap A^c=\varnothing,\quad
A\cup A^c=S,\quad
\varnothing=S^c$$

또한 $A\subset B$이면
$$A\cap(B-A)=\varnothing,\quad
A\cup(B-A)=B$$

가산가법성(countable additivity)과
전체 확률 공리를 적용하면 (a), (b), (c)가 모두 성립한다. ∎

#### 예: 베르누이 시행 (Bernoulli Trial)
(a) 공정한 동전을 던져 앞면 $H$, 뒷면 $T$를 관측할 때

$$S=\{H,T\},\qquad
P(\{H\})=\frac12,\quad
P(\{T\})=\frac12$$

(b) 관측 결과가 두 가지뿐인 실험을
**베르누이 시행 (Bernoulli trial)** 이라 한다.

성공(success)을 $s$, 실패(failure)를 $f$라 하면

$$S=\{s,f\},\qquad
P(\{s\})=p,\quad
P(\{f\})=1-p,\quad 0\le p\le1$$

### 정리 1.1.2 (합사건의 확률 (Probability of Unions))
(a)
$$P(A_1\cup A_2)
= P(A_1)+P(A_2)-P(A_1\cap A_2)$$

(b)
$$\begin{aligned}
P(A_1\cup A_2\cup A_3)
&=P(A_1)+P(A_2)+P(A_3)\\
&\quad -P(A_1\cap A_2)-P(A_2\cap A_3)-P(A_3\cap A_1)\\
&\quad +P(A_1\cap A_2\cap A_3)
\end{aligned}$$

(c) 일반적으로
$$P\Big(\bigcup_{i=1}^n A_i\Big)
=\sum_i P(A_i)
-\sum_{i<j}P(A_i\cap A_j)
+\cdots
+(-1)^{n-1}P\Big(\bigcap_{i=1}^n A_i\Big)$$

(d) **가산반가법성 (Countable Subadditivity)**
$$P(A_1\cup A_2\cup\cdots)
\le
P(A_1)+P(A_2)+\cdots$$

#### 증명 (Proof)
$$A_1\cup A_2
=(A_1-A_2)\cup(A_2-A_1)\cup(A_1\cap A_2)$$
세 집합은 서로소이므로 가산가법성을 적용하면 (a)가 성립한다.  

(b), (c)는 수학적 귀납법(mathematical induction)을 이용하여 얻는다.

(d)는
$B_1=A_1, B_2=A_2-A_1,\ldots, B_n = A_n - (A_1 \cup A_2 \cup \dots \cup A_{n-1})$
로 정의하면
$B_i\cap B_j=\varnothing$ ($i \neq j$) 이고
$\bigcup B_i=\bigcup A_i$ 이므로 $P(A_1 \cup A_2 \cup \dots) = P(B_1 \cup B_2 \cup \dots)=P(B_1)+P(B_1)+\dots$ 이다.  
이때 $B_n \subseteq A_n$ 이므로 단조성(monoticity)을 적용하면 얻어진다. ∎

#### 예: 짝 짓기 확률
n쌍의 부부가 남편은 남편끼리, 부인은 부인끼리 두 줄로 랜덤하게 늘어 설 때, 적어도 한 쌍의 부부가 서로마주보고 서게 될 확률?

**풀이 (Solution)**  
$A_i$를 "$i$번째 부부가 마주보는 사건"이라 하자.

구하는 확률은
$$P(A_1\cup A_2\cup\cdots\cup A_n)$$

포함-배제 원리(inclusion-exclusion principle)를 적용하면

$$P\Big(\bigcup_{i=1}^n A_i\Big)
=\sum_i P(A_i)
-\sum_{i<j}P(A_i\cap A_j)
+\cdots
+(-1)^{n-1}P(A_1\cap\cdots\cap A_n)$$

각 항을 계산하면:
- $P(A_i)=\frac{1}{n}$ (남편 위치가 정해지면 부인이 마주볼 확률)
- $P(A_i\cap A_j)=\frac{1}{n(n-1)}$ (두 쌍이 모두 마주볼 확률)
- 일반적으로 $P(A_{i_1}\cap\cdots\cap A_{i_k})=\frac{1}{n(n-1)\cdots(n-k+1)}$

따라서
$$\begin{aligned}
P\Big(\bigcup_{i=1}^n A_i\Big)
&=\binom{n}{1}\frac{1}{n}
-\binom{n}{2}\frac{1}{n(n-1)}
+\binom{n}{3}\frac{1}{n(n-1)(n-2)}
-\cdots\\[5pt]
&=1-\frac{1}{2!}+\frac{1}{3!}-\cdots+(-1)^{n-1}\frac{1}{n!}\\[5pt]
&=1-\sum_{k=2}^{n}\frac{(-1)^k}{k!}
\end{aligned}$$

$n\to\infty$일 때 이 확률은 $1-e^{-1}\approx 0.632$로 수렴한다.
#### 다른 풀이: 여사건 이용

아무도 마주보지 않을 확률을 먼저 구한 뒤, 여사건을 취할 수도 있다.

$n$쌍의 부부가 모두 마주보지 않을 확률은 **완전순열(derangement)** 문제와 동일하다.

부인들의 순열 중 어떤 부인도 자신의 남편과 마주보지 않는 경우의 수를 $D_n$이라 하면

$$P(\text{아무도 안 마주봄})=\frac{D_n}{n!}$$

완전순열의 개수는
$$D_n=n!\sum_{k=0}^n\frac{(-1)^k}{k!}$$

따라서
$$P(\text{적어도 한 쌍 마주봄})
=1-\frac{D_n}{n!}
=1-\sum_{k=0}^n\frac{(-1)^k}{k!}$$

$n\to\infty$일 때
$$\sum_{k=0}^\infty\frac{(-1)^k}{k!}=e^{-1}$$
이므로 확률은 $1-e^{-1}\approx 0.632$로 수렴한다.

이는 포함-배제 원리로 구한 결과와 일치한다.

### 정리 1.1.3 (확률측도의 연속성 (Continuity of Probability Measure))
(a) 증가열(increasing sequence)
$$A_1\subseteq A_2\subseteq\cdots
\Rightarrow
P\Big(\bigcup_{n=1}^\infty A_n\Big)
=\lim_{n\to\infty}P(A_n)$$
- 증가열 $C_1 \subseteq C_2 \subseteq \cdots$에 대해 극한집합은 다음과 같이 정의된다:
   $$\lim_{k\to\infty} C_k = \bigcup_{k=1}^\infty C_k = \{\omega : \exists k_0 \text{ s.t. } \forall k \ge k_0, \omega \in C_k\}$$
   즉, 어떤 $k_0$부터는 항상 포함되는 원소들의 집합이다.
- $k=\infin$는 자연수가 아니다. 끝점은 포함시키지 않는다.

(b) 감소열(decreasing sequence)
$$B_1\supseteq B_2\supseteq\cdots
\Rightarrow
P\Big(\bigcap_{n=1}^\infty B_n\Big)
=\lim_{n\to\infty}P(B_n)$$
- 감소열 $D_1 \supseteq D_2 \supseteq \cdots$에 대해 극한집합은 다음과 같이 정의된다:
   $$\lim_{k\to\infty} D_k = \bigcap_{k=1}^\infty D_k = \{\omega : \omega \in D_k \text{ for all } k\}$$
   즉, 모든 $k$에 대해 항상 포함되는 원소들의 집합이다.
- $k=\infin$는 자연수가 아니다. 끝점은 포함시키지 않는다.

#### 증명 (Proof)
**(a) 증가열의 경우**  
$C_1=A_1$, $C_2=A_2-A_1$, $C_3=A_3-A_2$, ..., $C_n=A_n-A_{n-1}$ 로 정의하자.

그러면 각 $C_i$들은 서로소이고
$$\bigcup_{n=1}^\infty C_n = \bigcup_{n=1}^\infty A_n$$

가산가법성에 의해
$$P\Big(\bigcup_{n=1}^\infty A_n\Big)
=P\Big(\bigcup_{n=1}^\infty C_n\Big)
=\sum_{n=1}^\infty P(C_n)$$

한편 $A_n = \bigcup_{i=1}^n C_i$ 이므로
$$P(A_n)=\sum_{i=1}^n P(C_i)$$

따라서
$$\lim_{n\to\infty}P(A_n)
=\lim_{n\to\infty}\sum_{i=1}^n P(C_i)
=\sum_{i=1}^\infty P(C_i)
=P\Big(\bigcup_{n=1}^\infty A_n\Big)$$

**(b) 감소열의 경우**  
$B_1\supseteq B_2\supseteq\cdots$ 이면 여사건들은 증가열을 이룬다:
$$B_1^c\subseteq B_2^c\subseteq\cdots$$

(a)를 적용하면
$$P\Big(\bigcup_{n=1}^\infty B_n^c\Big)
=\lim_{n\to\infty}P(B_n^c)$$

드모르간 법칙(De Morgan's law)에 의해
$$\bigcup_{n=1}^\infty B_n^c
=\Big(\bigcap_{n=1}^\infty B_n\Big)^c$$

따라서
$$P\Big(\Big(\bigcap_{n=1}^\infty B_n\Big)^c\Big)
=\lim_{n\to\infty}P(B_n^c)
=\lim_{n\to\infty}(1-P(B_n))
=1-\lim_{n\to\infty}P(B_n)$$

양변에 여사건을 취하면
$$P\Big(\bigcap_{n=1}^\infty B_n\Big)
=\lim_{n\to\infty}P(B_n)$$

#### 확률측도의 연속성 의미 (Interpretation of Continuity)
성질 (a)와 (b)를 다음과 같이 나타낸다
$$P(\lim_{n\to\infty}{A_n})=\lim_{n\to\infty}P(A_n),\ P(\lim_{n\to\infty}{B_n})=\lim_{n\to\infty}P(B_n)$$
즉, 포함관계가 커지거나 작아지는 사건들에 대해, 연속인 실수함수와 같이 확률계산이 가능하다는 의미로, probability measure의 연속성이라 함.  
증가열이나 감소열이 아니면 성립 안함!!!  

확률측도의 연속성은 **집합의 극한과 확률의 극한이 교환 가능함**을 보여준다.  
- **증가열**: 사건들이 점점 커질 때, 극한 사건의 확률은 각 사건 확률의 극한과 같다
- **감소열**: 사건들이 점점 작아질 때도 마찬가지로 성립한다

이는 확률이 **연속적인 측도(continuous measure)** 임을 의미하며,
유한가법성 만으로는 충분하지 않고 가산가법성이 필요한 이유를 보여준다.
  - 유한가법성 (finite additivity)만으로는 이러한 연속성을 보장할 수 없다.
    - **유한가법성 (Finite Additivity)**  
      유한 개의 서로소 사건 $A_1, A_2, \ldots, A_n$에 대해서만
      $$P(A_1 \cup A_2 \cup \cdots \cup A_n) = P(A_1) + P(A_2) + \cdots + P(A_n)$$
      이 성립한다는 조건으로, 무한이 있고 없고 차이.  
      유한가법성만으로는 무한 개의 사건에 대한 극한 연산을 다룰 수 없으므로,
      Kolmogorov는 **가산가법성(countable additivity)** 을 확률의 공리로 채택하였다.

실해석학의 측도론에서 이를 **측도의 연속성(continuity from below/above)**  이라 하며,
확률론의 근간이 되는 중요한 성질이다. ∎



#### 예 1.1.3 (Example)
표본공간 $S=[0,1]$이고
$$P((a,b))=b-a$$
일 때 ($a<b$), 한 점의 확률은

$$P(\{b\})
=P\Big(\bigcap_{n=1}^\infty (b-\tfrac1n,b]\Big)
=\lim_{n\to\infty}P((b-\frac1n,b]))
=\lim_{n\to\infty}\frac1n
=0$$

## 2. 조건부확률과 독립성 (Conditional Probability and Independence)
### 2.1 조건부확률 (Conditional Probability)
한 사건이 일어날 가능성은 그 사건에 관한 정보에 따라 다르다.  
사건 $A$가 일어났다는 조건하에서
사건 $B$가 일어날 확률을 **조건부확률**이라 하며

$$P(B\mid A)
=\frac{P(A\cap B)}{P(A)},
\qquad P(A)>0$$
으로 정의한다.  
이는 사건 $A$를 새로운 표본공간으로 간주하는 것이다.

### 정리 1.2.1 조건부확률의 성질 (Properties of Conditional Probability)
(a) **곱셈공식 (Multiplication Rule)**
$$P(A\cap B)=P(B\mid A)P(A)=P(A\mid B)P(B)$$

(b) **전확률공식 (Law of Total Probability)**
서로소 사건 $A_1,A_2,\dots$ 가 $S$를 분할하면 (공통부분이 없음)
$$P(B)=\sum_i P(B\mid A_i)P(A_i)$$

#### 증명 (Proof)
조건부확률의 정의와 가산가법성을 적용하여 바로 얻어진다. ∎
**(a) 곱셈공식의 증명**  
조건부확률의 정의로부터
$$P(B\mid A) = \frac{P(A\cap B)}{P(A)}$$

양변에 $P(A)$를 곱하면
$$P(A\cap B) = P(B\mid A)P(A)$$

마찬가지로
$$P(A\mid B) = \frac{P(A\cap B)}{P(B)}$$

양변에 $P(B)$를 곱하면
$$P(A\cap B) = P(A\mid B)P(B)$$

따라서
$$P(A\cap B) = P(B\mid A)P(A) = P(A\mid B)P(B)$$

**(b) 전확률공식의 증명**

서로소 사건 $A_1, A_2, \ldots$ 가 표본공간 $S$를 분할한다고 하자. 즉,
$$A_i \cap A_j = \varnothing \quad (i \neq j), \qquad \bigcup_{i=1}^\infty A_i = S$$

임의의 사건 $B$에 대하여
$$B = B \cap S = B \cap \Big(\bigcup_{i=1}^\infty A_i\Big) = \bigcup_{i=1}^\infty (B \cap A_i)$$

$A_i$들이 서로소이므로 $B \cap A_i$들도 서로소이다:
$$(B \cap A_i) \cap (B \cap A_j) = B \cap (A_i \cap A_j) = B \cap \varnothing = \varnothing \quad (i \neq j)$$

따라서 가산가법성에 의해
$$P(B) = P\Big(\bigcup_{i=1}^\infty (B \cap A_i)\Big) = \sum_{i=1}^\infty P(B \cap A_i)$$

곱셈공식 $P(B \cap A_i) = P(B \mid A_i)P(A_i)$를 적용하면
$$P(B) = \sum_{i=1}^\infty P(B \mid A_i)P(A_i)$$

**유한 분할의 경우**

특히 $A_1, A_2, \ldots, A_n$이 $S$를 유한 분할하면
$$P(B) = \sum_{i=1}^n P(B \mid A_i)P(A_i)$$

이 공식은 사건 $B$의 확률을 계산할 때, 전체 표본공간을 여러 경우로 나누어
각 경우의 확률을 조건부확률로 계산한 후 합하는 방법을 제공한다. ∎

#### 예 1.2.1 (Laplace's Rule of Succession)
상자 $N+1$개가 있고
$k (\in[0, N])$번째 상자에는 빨간 공 $k$개, 흰 공 $N-k$개가 들어 있다.

상자를 임의로 하나고르자. 이 상자에서 복원추출하는 작업을 $n$번 했을 때 $n$번 연속 빨간 공이 나왔다. 같은 상자에서 다음 번도 빨간 공일 확률은?

**풀이 (Solution)**  
사건을 다음과 같이 정의하자:
- $A_k$: $k$번째 상자를 선택하는 사건
- $B$: $n$번 연속 빨간 공이 나오는 사건
- $C$: 다음 번($(n+1)$번째)에도 빨간 공이 나오는 사건

구하는 확률은 $P(C\mid B)$이다.

**전확률공식 적용**  
상자들이 표본공간을 분할하므로
$$P(B) = \sum_{k=0}^N P(B\mid A_k)P(A_k)$$

각 상자가 선택될 확률은 동일하므로 $P(A_k) = \frac{1}{N+1}$

$k$번째 상자에서 빨간 공이 나올 확률은 $\frac{k}{N}$이고, 복원추출이므로
$$P(B\mid A_k) = \left(\frac{k}{N}\right)^n$$

따라서
$$P(B) = \sum_{k=0}^N \left(\frac{k}{N}\right)^n \cdot \frac{1}{N+1} = \frac{1}{N+1}\sum_{k=0}^N \left(\frac{k}{N}\right)^n$$

마찬가지로
$$P(C\cap B) = P(B\cap C) = \sum_{k=0}^N P(B\cap C\mid A_k)P(A_k) = \frac{1}{N+1}\sum_{k=0}^N \left(\frac{k}{N}\right)^{n+1}$$

$$P(C\mid B) = \frac{P(C\cap B)}{P(B)} = \frac{\frac{1}{N+1}\sum_{k=0}^N \left(\frac{k}{N}\right)^{n+1}}{\frac{1}{N+1}\sum_{k=0}^N \left(\frac{k}{N}\right)^n}$$
$$P(C\mid B)
= \frac{\sum_{k=0}^N (k/N)^{n+1}}
{\sum_{k=0}^N (k/N)^n}$$

$N$이 큰 경우
$$P(C\mid B)\approx \frac{n+1}{n+2}$$

**근삿값 유도**  
$N$이 충분히 클 때, 합을 적분으로 근사할 수 있다.  
$u = \frac{k}{N}$로 치환하면 $k$가 $0$부터 $N$까지 변할 때 $u$는 $0$부터 $1$까지 변하고, $\Delta u = \frac{1}{N}$이다.

따라서 Riemann 합의 극한으로  
$$\sum_{k=0}^N \left(\frac{k}{N}\right)^n \approx N\int_0^1 u^n\,du = N\cdot\frac{1}{n+1} = \frac{N}{n+1}$$

마찬가지로
$$\sum_{k=0}^N \left(\frac{k}{N}\right)^{n+1} \approx N\int_0^1 u^{n+1}\,du = N\cdot\frac{1}{n+2} = \frac{N}{n+2}$$

따라서
$$P(C\mid B) \approx \frac{N/(n+2)}{N/(n+1)} = \frac{n+1}{n+2}$$

이는 **Laplace (1812)**  의 결과로, 오늘까지 해가 뜬 것을 전제로 내일 해가 뜰 확률의 근삿값을 이와같이 생각하여 계산했다.  

### 정리 1.2.2 베이즈 정리 (Bayes' Theorem)
사건 $A_1, A_2, \ldots$ 이 표본공간 $S$를 공통부분 없이 분할하고
$P(A_i)>0\ (i=1,2,\ldots)$ 이며 $P(B)>0$ 일 때,

$$
P(A_j\mid B)\propto P(B\mid A_j)P(A_j)\quad (j=1,2,\ldots)
$$

이고 이 비례식에서 비례상수는 좌변의 합이 1임으로부터 결정된다.

#### 증명
조건부확률의 정의 또는 곱셈공식으로부터
$$
P(A_j\mid B)P(B)=P(B\cap A_j)=P(B\mid A_j)P(A_j)\quad (j=1,2,\ldots)
$$

따라서 전확률공식으로부터
$$
P(A_j\mid B)
=\frac{P(B\mid A_j)P(A_j)}{P(B)}
=\frac{P(B\mid A_j)P(A_j)}
{P(B\mid A_1)P(A_1)+P(B\mid A_2)P(A_2)+\cdots}
$$

한편 비례상수를 $c$라고 하여 위 비례식을 $j$에 대하여 더하면
$$
P(A_1\mid B)+P(A_2\mid B)+\cdots
=c\{P(B\mid A_1)P(A_1)+P(B\mid A_2)P(A_2)+\cdots\}
$$

$$
P(A_1\cup A_2\cup\cdots\mid B)
=c\{P(B\mid A_1)P(A_1)+\cdots\}
$$

$$
1=P(S\mid B)
=c\{P(B\mid A_1)P(A_1)+P(B\mid A_2)P(A_2)+\cdots\}
$$

따라서
$$
c=\frac{1}{P(B\mid A_1)P(A_1)+P(B\mid A_2)P(A_2)+\cdots}
$$

베이즈 정리는 베이지안 추론의 근본이 되는 정리로서
$P(A_1),P(A_2),\ldots$ 는 여러 모형의 가능성을 뜻하고,
$P(A_1\mid B),P(A_2\mid B),\ldots$ 는 실험 결과 $B$의 관측 후 각 모형의 가능성을 뜻한다.
이러한 이유에서 $P(A_j)$, $P(A_j\mid B)$를 각각 사전(prior), 사후(posterior) 확률이라 부른다.

#### 예 1.2.2
한 공장에서 전체 생산량의 20%, 30%, 50%를 세 기계 $M_1,M_2,M_3$로 생산하고 있고
각 기계에서의 불량품 제조 비율은 각각 3%, 2%, 1%로 알려져 있다.
어느 날 이 공장에서 생산된 제품 중 임의로 1개를 택하여 검사하였더니 불량품이었다.
이 제품이 각 기계에서 생산되었을 확률을 구하여라.

**풀이**  
검사 전 한 제품이 기계 $M_1,M_2,M_3$에서 생산되었을 확률은 각각

$$
P(M_1)=0.2,\quad P(M_2)=0.3,\quad P(M_3)=0.5
$$

불량품 사건을 $B$라 하면

$$
P(B\mid M_1)=0.03,\quad P(B\mid M_2)=0.02,\quad P(B\mid M_3)=0.01
$$

베이즈 정리로부터

$$
P(M_1\mid B)\propto 0.2\times0.03,\quad
P(M_2\mid B)\propto 0.3\times0.02,\quad
P(M_3\mid B)\propto 0.5\times0.01
$$

따라서

$$
P(M_1\mid B):P(M_2\mid B):P(M_3\mid B)=6:6:5
$$

즉

$$
P(M_1\mid B)=\frac{6}{17},\quad
P(M_2\mid B)=\frac{6}{17},\quad
P(M_3\mid B)=\frac{5}{17}
$$

### 2.4 사건의 독립성 (Independence of Events)
사건 $A$의 관측 여부가 사건 $B$가 일어날 가능성에 아무런 영향을 주지 않는 것을

$$
P(B\mid A)=P(B)
$$

와 같이 나타낼 수 있고, 이는 곱셈공식으로부터

$$
P(A\cap B)=P(A)P(B)
$$

와 같음을 알 수 있다. 이러한 경우 $A$와 $B$는 서로 독립(mutually independent)이라 한다.  
서로 독립이 아닌 사건들은 서로 종속(mutually dependent)라 한다  

여러 사건의 경우에는 **모든 유한 부분집합**에 대해 위 조건이 성립해야 한다.  
**2개씩:**
$$P(A_i \cap A_j) = P(A_i)P(A_j) \quad \text{for all } i \neq j$$

**3개씩:**
$$P(A_i \cap A_j \cap A_k) = P(A_i)P(A_j)P(A_k) \quad \text{for all } i < j < k$$

**일반적으로 $k$개씩 ($2 \le k \le n$):**
$$P(A_{i_1} \cap A_{i_2} \cap \cdots \cap A_{i_k}) = P(A_{i_1})P(A_{i_2})\cdots P(A_{i_k})$$

**전체:**
$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1)P(A_2)\cdots P(A_n)$$

- 총 $2^n - n - 1$개의 조건을 확인해야 한다
- 아래 예에서는 A,B각각은 사건 C와 서로 독립이지만, A와B를 동시에 관측하면 C와 종속이 된다

#### 예 1.2.3
두 개의 주사위를 던지는 경우 다음 사건들을 독립 여부를 판단하자.
* $A$: 첫 번째 주사위의 눈이 짝수
* $B$: 두 번째 주사위의 눈이 홀수
* $C$: 두 주사위의 눈의 합이 홀수

**풀이**  
각각의 경우의 수를 구하여 전체 경우의 수로 나누면

$$
P(A)=18/36,\quad P(B)=18/36,\quad P(C)=18/36
$$

$$
P(A\cap B)=9/36,\quad P(B\cap C)=9/36,\quad P(C\cap A)=9/36
$$

따라서

$$
P(A)=P(B)=P(C)=1/2
$$

$$
P(A\cap B)=P(B\cap C)=P(C\cap A)=1/4
$$

즉 $A,B,C$는 서로 독립이다.

#### 예 1.2.4
서로 독립인 두 사건 $A$와 $B$에 대하여, $A$와 $B^c$도 서로 독립임을 보이자.

**풀이**  
$A$와 $B$가 서로 독립이므로
$$P(A \cap B) = P(A)P(B)$$

사건 $A$는 다음과 같이 분할할 수 있다:
$$A = (A \cap B) \cup (A \cap B^c)$$

이 두 사건은 서로소이므로 가산가법성에 의해
$$P(A) = P(A \cap B) + P(A \cap B^c)$$

따라서
$$P(A \cap B^c) = P(A) - P(A \cap B)$$

독립성 조건 $P(A \cap B) = P(A)P(B)$를 대입하면
$$P(A \cap B^c) = P(A) - P(A)P(B) = P(A)(1 - P(B))$$

여사건의 확률 $P(B^c) = 1 - P(B)$를 이용하면
$$P(A \cap B^c) = P(A)P(B^c)$$

이는 $A$와 $B^c$가 서로 독립임을 의미한다. ∎

**따름정리**  
같은 방법으로 다음도 성립함을 보일 수 있다:
- $A$와 $B$가 독립이면 $A^c$와 $B$도 독립
- $A$와 $B$가 독립이면 $A^c$와 $B^c$도 독립

## 3. 확률변수와 확률분포 (Random Variables and Distributions)
### 3.1 확률변수 (Random Variable)
야구선수의 타격 결과, 새로 개발된 전구의 수명처럼 여러 가지의 결과가 가능할 때, 각 결과의 가능성을 확률로 나타낼 수 있다.  
여기서 가능한 결과들에 대응시킨 수 값에 관심이 있고, 하나의 값을 갖는게 아니라 여러 가지의 값을 적절한 확률에 따라 갖게되는 변수라 생각할 수 있다.  

일반적으로, 여러 가지의 결과가 가능하고 그 가능성을 확률로 나타낼 수 있는 실험을 랜덤한 실험(random experiment)라 하고,  
이런 실험의 모든 가능한 결과의 집합인 표본공간에서 정의된 실수 값 함수를 확률변수(random variable)이라 한다.  

확률공간 $(\Omega,\mathcal F,P)$에서 정의된 실수값 함수

$$
X:\Omega\to\mathbb R
$$

를 확률변수라 한다.

#### 예 1.3.1 동전을 두 번 던지는 실험에서 앞면이 나오는 횟수
표본공간은
$$
S=\{(T,T),(T,H),(H,T),(H,H)\}
$$
앞면이 나오는 횟수를 $X$라 하면
| 결과     | (T,T) | (T,H) | (H,T) | (H,H) |
| ------ | ----- | ----- | ----- | ----- |
| $X$의 값 | 0     | 1     | 1     | 2     |
| 확률     | $1/4$ | $1/4$ | $1/4$ | $1/4$ |

확률변수의 값에 관한 사건은
$\{X=1\}$, $\{X\le1\}$, $\{1<Y\le3\}$ 등으로 나타내며 이는 표본공간의 부분집합으로서 확률을 갖는다.

### 3.2 확률분포 (Probability Distribution)
확률변수 $X$가 유도하는 측도

$$
P_X(B)=P(X\in B)
$$

를 $X$의 확률분포라 한다.

#### 예 1.3.2
앞면이 나오는 횟수 $X$의 확률분포표
$$
P(X=0)=1/4,\quad
P(X=1)=1/2,\quad
P(X=2)=1/4
$$

| $X$의 값 | 0     | 1     | 2     |
| ------ | ----- | ----- | ----- |
| 확률     | $1/4$ | $1/2$ | $1/4$ |

### 3.3 이산형 확률질량함수 (Probability Mass Function, PMF)
확률변수가 가질 수 있는 값들의 집합을 $\{x_1, x_2, \dots \}$로 나타낼 수 있을 때 그 확류변수를 이산형(discrete type)이라 하고, 각각의 값에 그 값을 가질 확률을 대응시키는 함수, 즉 $f(x_k) = P(X=x_k)$를 X의 확률질량함수(probability mass function, pmf)이라 한다. (밀도함수 즉 pdf로 표현하는 책도 있네)  

#### 확률질량함수의 성질 (Properties of PMF)
확률질량함수 $p_X(x) = P(X=x)$는 다음을 만족한다:

(a) **비음성 (Non-negativity)**
$$p_X(x) \ge 0 \quad \text{for all } x$$

(b) **합이 1 (Normalization)**
$$\sum_{x} p_X(x) = 1$$
여기서 합은 $X$가 가질 수 있는 모든 값에 대해 취한다.

(c) **사건의 확률 계산**
임의의 집합 $B$에 대하여
$$P(X \in B) = \sum_{x \in B} p_X(x)$$

- 사건의 확률을 PMF로 계산
  - 임의의 집합 $B \subseteq \mathbb{R}$에 대하여
    $$P(a < X \le b) = \sum_{x: a < x \le b} p_X(x)$$

#### 예 1.3.3
앞면이 나올 때까지 동전을 던지는 실험
$$
S=\{H,TH,TTH,TTTH,\ldots\}
$$
확률변수 $X$가 가질 수 있는 값은 자연수이고
$$
P(X=x)=(1/2)^x,\quad x=1,2,\ldots
$$
이를 이산형 확률변수라 한다.

### 3.4 연속형 확률질량함수 (Probability density function, PDF)
확률변수가 실수 구간의 값을 갖고 그 확률이 적분으로 주어질 때 이 확률변수를 연속형(continuous type)이라 한다.

$$
P(a\le X\le b)=\int_a^b f(x),dx
$$

이 확률을 정해주는 함수 $f$를 $X$의 확률밀도함수라 한다.

#### 성질
* $f(x)\ge0$
* $\int_{-\infty}^{\infty}f(x) dx=1$
* $P(a < X \le b) = \int_a^b f(x)\,dx$

- 참고
   - 한 점에서의 적분값이 0이다. $P(X=a)=\int_a^a f(x),dx=0$
   - $f(a) \neq P(X=a)$
   - $\Delta x \to 0$일 때, $P(a < X \le a + \Delta x) \approx f(a)\Delta x$



이산형이던 연속형이던 확률밀도함수는 확률변수에 관한 확률 $P(a \le X \le b)$를 결정지어준다. 즉 확률변수 X에 관한 확률이 실직선 위에 어떻게 분포되는가를 나타낸다.  
그래서 P를 X의 확률분포(probability function)또는 간단히 '분포'라 부르며, $$X \sim f(pdf)$$ 와 같이 X의 분포가 pdf f에 의해 정해지는 것으로 표기한다.

#### 예 1.3.4
다음 함수가 확률밀도함수가 되기 위한 상수 $c$를 구하고
$P(1/2\le X\le 3/4)$를 구하여라.

$$
f(x)=
\begin{cases}
cx(1-x), & 0\le x\le1 \\
0, & \text{otherwise}
\end{cases}
$$

**풀이**  
$$
\int_0^1 cx(1-x)dx=1
\Rightarrow c=6
$$

$$
P(1/2\le X\le 3/4)
=\int_{1/2}^{3/4}6x(1-x),dx
=\frac{11}{32}
$$

- 참고  
   집합 $A$에 대한 **지표함수** (indicator function)는 다음과 같이 정의된다:

   $$
   \mathbf{1}_A(x) = I_A(x) = 
   \begin{cases}
   1, & x \in A \\
   0, & x \notin A
   \end{cases}
   $$

   지표함수는 확률론에서 사건의 발생 여부를 수치화하는 데 유용하다.  
   예 1.3.4의 확률밀도함수를 지표함수로 표현하면
   $$
   f(x) = 6x(1-x) \cdot \mathbf{1}_{[0,1]}(x)
   $$
   여기서 $\mathbf{1}_{[0,1]}(x)$는 구간 $[0,1]$의 지표함수이다.  
   지표함수를 사용하면 확률밀도함수를 더 간결하게 표현할 수 있으며,  
   "otherwise" 조건을 명시적으로 나타낼 필요가 없다.  
   일반적으로 구간 $[a,b]$에서만 0이 아닌 확률밀도함수는
   $$
   f(x) = g(x) \cdot \mathbf{1}_{[a,b]}(x)
   $$
   형태로 표현할 수 있다.

## 4. 확률분포의 특성치 (Characteristics of Distributions)
일상생활에서 흔히 사용하는 ‘평균 임금’, ‘평균 성적’에서 평균의 의미는 무엇이며, 어떤 목적으로 평균을 사용하는지 생각해본다.  
이를 위해 두 반의 성적 분포를 비교하여 평균의 의미를 살펴본다.

두 반의 성적 분포를 비교하면, 2반의 성적 분포가 1반에 비해 큰 쪽에 위치하고 있음을 알 수 있다. 한편 두 반의 평균 성적을 계산하면 각각 다음과 같다.
$$
(80\times3+85\times6+90\times1)/10=84.0
$$

$$
(80\times2+85\times5+90\times3)/10=85.5
$$
즉 평균은 분포의 위치를 나타내는 값으로서, 성적 분포를 이루고 있는 점수들의 한 기준이 된다.

이 과정을 확률분포의 관점에서 보면, 평균은 가능한 각 값에 대응하는 상대도수(확률)를 곱하여 더한 값임을 알 수 있다.
이러한 관점에서 확률변수의 평균을 정의한다.

### 4.1 평균 (Mean)
확률변수 $X$의 확률분포가 주어졌을 때, 평균은 분포의 중심 경향을 나타내는 대표값이다.  
확률변수 $X$의 확률밀도함수가 $f$일 때, 확률분포의 평균 $\mu$(mean)는 다음과 같이 정의한다.
$$
\mu=
\begin{cases}
\displaystyle\sum_x x f(x), & X\text{가 이산형일 때} \\
\displaystyle\int_{-\infty}^{\infty} x f(x),dx, & X\text{가 연속형일 때}
\end{cases}
$$
여기서 무한 합이나 이상적분이 **절대수렴**하는 경우에만 평균이 실수로 정의된다.

#### 예 1.4.1
확률변수 $X$의 확률밀도함수가 다음과 같을 때, $X$의 확률분포의 평균 $\mu$를 구하여라.
$$
f(x)=
\begin{cases}
(-x^2+2x)/2, & 0\le x\le1 \\
(-x^2+2x+3)/8, & 1\le x\le3 \\
0, & x<0,\ x>3
\end{cases}
$$

**풀이**  
평균의 정의에 따라
$$
\mu=\int_{-\infty}^{\infty}x f(x),dx
=\int_0^1 x\frac{-x^2+2x}{2},dx+\int_1^3 x\frac{-x^2+2x+3}{8},dx
$$
계산하면
$$
\mu=\frac{11}{8}
$$
확률변수 $X$의 확률분포의 평균은 $X$의 **기댓값(expected value)** 이라고도 부른다.

### 4.2 기댓값 (Expectation)
확률변수 $X$의 확률밀도함수가 $f$일 때, 실수값 함수 $g(x)$에 대하여 이 식이 실수로 정의되면, 그 값을 $g(X)$의 기댓값이라 하고 $E[g(X)]$로 나타낸다.
$$
E[g(X)]=
\begin{cases}
\displaystyle\sum_x g(x)f(x), & X\text{가 이산형일 때} \\
\displaystyle\int_{-\infty}^{\infty} g(x)f(x),dx, & X\text{가 연속형일 때}
\end{cases}
$$
여기서 무한 합이나 이상적분이 **절대수렴**하는 경우에만 기댓값이 실수로 정의된다.

#### Mean vs. Expectation
- 평균($\mu$)은 분포의 특성을 나타내는 **고정된 값**
- 기댓값($E[\cdot]$)은 확률변수의 함수를 확률로 가중평균하는 **연산**
- 수리통계학에서는 평균을 "$E[X]$로 정의되는 모수"로, 기댓값을 "더 일반적인 적분/합 연산자"로 구분한다  

**개념적 차이**  
1. **평균 (Mean, $\mu$)**
   - 확률분포 자체의 특성치
   - 모집단의 고정된 모수(parameter)
   - 분포의 중심 위치를 나타내는 상수

2. **기댓값 (Expected Value, $E[\cdot]$)**
   - 확률변수의 함수에 대한 연산자(operator)
   - $E[g(X)]$는 함수 $g(X)$의 가중평균
   - 더 일반적인 개념

**수리통계적 관점**  
1. **평균은 기댓값의 특수한 경우**
  $$\mu = E[X] = E[g(X)]\quad \text{where } g(x)=x$$
  즉, $g(x)=x$ (항등함수)일 때의 기댓값이 평균이다.

2. **기댓값의 일반성**
  - $E[X^2]$: 2차 적률
  - $E[(X-\mu)^2]$: 분산
  - $E[e^{tX}]$: 적률생성함수
  - $E[g(X)]$: 임의의 함수 $g$에 대한 기댓값

**참고**  
- 지시함수의 기댓값과 확률의 관계: 지시함수(Indicator Function) **1_A**의 기댓값은 사건 A의 확률과 같다  
$$E[1_A] = 1 × P(A) + 0 × P(A^c) = P(A) \\ E[1_A​]=P(A)$$

- 이 성질은 확률론에서 기댓값 계산을 단순화하고, 복잡한 확률 문제를 기댓값 문제로 변환할 때 유용하다

- 예: 지시함수 $\mathbf{1}_A$의 기댓값은 사건 $A$의 확률과 같다: $$E[\mathbf{1}_A] = 1 \times P(A) + 0 \times P(A^c) = P(A)$$

   이 성질을 이용하면 비음 확률변수 $X \ge 0$에 대해 다음이 성립한다(비음 확률변수의 적분 표현):  
   $X \ge 0$이면 모든 $\omega$에 대해
   $$X(\omega) = \int_0^{\infty} \mathbf{1}_{\{t < X(\omega)\}}\,dt$$

   **직관적 이해**  
   - $t < X(\omega)$일 때 $\mathbf{1}_{\{t < X(\omega)\}} = 1$
   - $t \ge X(\omega)$일 때 $\mathbf{1}_{\{t < X(\omega)\}} = 0$
   - 따라서 적분 구간 $[0, X(\omega))$에서만 1이므로 적분값은 정확히 $X(\omega)$

   **기댓값 계산**  
   양변에 기댓값을 취하면
   $$E[X] = E\left[\int_0^{\infty} \mathbf{1}_{\{t < X\}}\,dt\right]$$

   Fubini-Tonelli 정리에 의해 적분과 기댓값의 순서를 교환할 수 있다:
   $$E[X] = \int_0^{\infty} E[\mathbf{1}_{\{t < X\}}]\,dt$$

   지시함수의 기댓값은 확률이므로
   $$E[\mathbf{1}_{\{t < X\}}] = P(t < X) = P(X > t)$$

   따라서
   $$E[X] = \int_0^{\infty} P(X > t)\,dt$$

   **의미**  
   이 공식은 비음 확률변수의 기댓값을 계산하는 또 다른 방법을 제공한다:
   - 확률밀도함수를 이용: $E[X] = \int_0^{\infty} x f(x)\,dx$
   - 생존함수를 이용: $E[X] = \int_0^{\infty} P(X > t)\,dt$

   특히 생존함수 $P(X > t)$가 간단한 형태일 때 유용하다.

#### 예 1.4.2
동전을 두 번 던져 앞면이 나오는 횟수를 $X$라 하고, 상금을 $X^2$(만원)만큼 받는다고 할 때 기대할 수 있는 상금은 얼마인가?  
**풀이**  
앞에서 구한 확률분포는
$$
P(X=0)=\frac14,\quad P(X=1)=\frac12,\quad P(X=2)=\frac14
$$

따라서
$$
E(X^2)=\sum_x x^2P(X=x)
=0^2\cdot\frac14+1^2\cdot\frac12+2^2\cdot\frac14
=1.5
$$
즉 기대 상금은 15,000원이다.

### 4.3 분산과 표준편차 (Variance and Standard Deviation)
확률분포의 평균만으로는 분포의 위치 정보밖에 모른다.  
이를 위해 분포가 평균을 기준으로 얼마나 퍼져 있는지를 나타내는 분산(Var(X))과 표준편차(Sd(X))를 정의한다.  

확률변수 $X$의 평균을 $\mu$라 할 때, 분산과 표준편차는 다음과 같다.
$$
\mathrm{Var}(X)=E[(X-\mu)^2]=
\begin{cases}
\displaystyle\sum_x (x-\mu)^2 f(x), & X\text{가 이산형일 때} \\
\displaystyle\int_{-\infty}^{\infty} (x-\mu)^2 f(x)\,dx, & X\text{가 연속형일 때}
\end{cases}
$$

$$
\mathrm{Sd}(X)=\sqrt{\mathrm{Var}(X)}
$$

#### 예 1.4.3
다음 두 확률분포의 분산을 구하여라.  
(a) $X\sim f_1$ (균등분포)
$$
f_1(x)=I_{[0,1]}(x)
$$

(b) $X\sim f_2$ (삼각분포)
$$
f_2(x)=(2-4|x-\tfrac12|)I_{[0,1]}(x)
$$

**풀이**  
두 분포의 평균은 각각
$$
\mu_1=\int_0^1 x,dx=\frac12
$$

$$
\mu_2=\int_0^{1/2}x(4x),dx+\int_{1/2}^1x(4-4x),dx=\frac12
$$

분산은
$$
\mathrm{Var}(X_1)=\int_0^1(x-\tfrac12)^2dx=\frac1{12}
$$

$$
\mathrm{Var}(X_2)=\int_0^1(x-\tfrac12)^2(2-4|x-\tfrac12|)dx=\frac1{24}
$$

### 기댓값과 분산의 성질
**정리 1.4.1 (기댓값의 성질)**  
(a) 선형성  
$$
E(aX+b)=aE(X)+b\quad(a,b\text{는 확률변수가 아닌 상수})
$$

(b) 일반 선형결합  
c1, c2는 상수
$$
E[c_1g_1(X)+c_2g_2(X)]
=c_1E[g_1(X)]+c_2E[g_2(X)]
$$

(c) 단조성  
$$
g_1(X)\le g_2(X)\Rightarrow E[g_1(X)]\le E[g_2(X)]
$$

**정리 1.4.2 (분산의 성질)**  
(a)  
a, b는 확률변수가 아닌 상수일 때
$$
\mathrm{Var}(aX+b)=a^2\mathrm{Var}(X)
$$

(b)
$$
\mathrm{Var}(X)=E(X^2)-{E(X)}^2
$$

#### 예 1.4.4
(예 1.4.3의 계속)
(a) 균등분포의 경우
$$
E(X)=\frac12,\quad E(X^2)=\int_0^1x^2dx=\frac13
$$

$$
\mathrm{Var}(X)=\frac13-\left(\frac12\right)^2=\frac1{12}
$$

(b) 삼각분포의 경우
$$
E(X)=\frac12,\quad E(X^2)=\frac7{24}
$$

$$
\mathrm{Var}(X)=\frac7{24}-\left(\frac12\right)^2=\frac1{24}
$$

#### 예 1.4.5 확률변수의 표준화
평균이 $\mu$, 표준편차가 $\sigma>0$일 때
$$
Z=\frac{X-E(X)}{\sqrt{\mathrm{Var}(X)}}=\frac{X-\mu}{\sigma}
$$
라 하면
$$
E(Z)=0,\quad \mathrm{Var}(Z)=1
$$

이때 $Z$를 $X$를 **표준화한 확률변수**라 한다.  

한편 다음과 같은 확률분포를 갖는 확률변수에 대해서는 평균이나 분산이 실수로 정의되지 않을 수 있다.
$$
f(k)=P(X=k)=\frac1{k(k+1)},\quad k=1,2,\ldots
$$

절대수렴하는, **두터운 꼬리 분포(heavy-tailed distribution)** 가 아니라서 그렇다.  
적분 방법을 바꾼다고 해결되는 문제가 아니라 분포 자체의 본질적 성질이다.  

**참고: 절대수렴의 중요성**  
확률론에서 평균과 분산을 정의할 때 절대수렴을 요구하는 이유는:
1. 무한합의 순서를 바꿔도 값이 같음을 보장
2. 조건부수렴만으로는 극한값이 재배열에 따라 달라질 수 있음
3. 측도론적으로 Lebesgue 적분이 잘 정의되려면 절대수렴 필요

## 5. 누적분포함수와 생성함수 (Distribution and Generating Functions)
확률밀도함수 외에도 확률분포를 나타내는 방법은 여러 가지가 있다.  
수열에서 일반항, 유한항까지의 합이 서로를 정해줄 수 있듯이,  
이산현 확률변수의 경우에 누적 확률과 확률밀도함수는 서로를 정해줄 수 있다.  

### 5.1 누적분포함수 (Cumulative Distribution Function, CDF)
확률변수 $X$의 누적분포함수(cumulative distribution function, CDF)는 다음과 같이 정의된다.
$$
F(x)=P(X\le x)
$$

#### 이산형 확률변수의 경우
확률변수 $X$가 이산형이고 확률질량함수가 $p(x)$일 때, 누적분포함수는
$$
F(x)=\sum_{k:x_k\le x}p(x_k)=\sum_{k:x_k\le x}P(X=x_k)
$$
여기서 합은 $x_k\le x$를 만족하는 모든 가능한 값에 대해 취한다.  
이산형의 경우 CDF는 계단함수(step function) 형태를 가지며, 각 가능한 값에서 불연속점을 갖는다.

#### 예 1.5.1
앞면이 나올 때까지 동전을 던질 때 시행 횟수 $X$에 대하여
$$
P(X=k)=(1/2)^k,\quad k=1,2,\dots
$$
누적분포함수는
$$
F(n)=1-(1/2)^n
$$
이다.

#### 연속형 확률변수의 경우
확률변수 $X$가 연속형이고 확률밀도함수가 $f(x)$일 때, 누적분포함수는
$$
F(x)=\int_{-\infty}^x f(t)\,dt
$$

연속형의 경우 CDF는 연속함수이며, 미분 가능한 점에서는
$$
F'(x)=f(x)
$$
가 성립한다. 즉, 확률밀도함수는 누적분포함수의 도함수이다.  

또한 연속형 확률변수의 경우
$$
P(a<X\le b)=F(b)-F(a)=\int_a^b f(x)\,dx
$$
이고, 한 점에서의 확률은
$$
P(X=a)=F(a)-F(a-)=0
$$
이므로 $P(a<X\le b)=P(a\le X\le b)=P(a<X<b)=P(a\le X<b)$가 모두 같다.

#### 일반적인 확률분포에서 CDF로부터 확률 복원
**불연속점(점질량, atom)에서**
$$P(X=a) = F(a) - F(a-)$$
**연속인 구간에서**
$$f(x) = F'(x) \quad \text{(거의 모든 } x\text{에서)}$$
이는 **Lebesgue 분해(Lebesgue decomposition)** 에 해당하는 표준적 절차로,
임의의 확률분포를 이산 부분(discrete part), 절대연속 부분(absolutely continuous part),
특이연속 부분(singular continuous part)으로 분해할 수 있다는 측도론의 결과이다.

**❌ 흔한 오해**
> "$F$가 불연속이면 적분이나 $F(b)-F(a)$ 같은 계산은 안 되는 것 아닌가?"
→ **틀렸다.**

**이유:**  
우리가 쓰는 $F(b)-F(a)$는
- 리만적분이 아니라
- 분포측도 $\mu$의 값을 CDF로 표현한 것일 뿐이다

**정리**
1. **연속형**: $P(a < X \le b) = F(b) - F(a) = \int_a^b f(x)\,dx$
   - CDF가 연속이고 미분가능
   - 확률밀도함수 $f$가 존재

2. **이산형**: $P(a < X \le b) = F(b) - F(a) = \sum_{a < x_k \le b} p(x_k)$
   - CDF가 계단함수 (불연속)
   - 확률질량함수 $p$가 존재

3. **혼합형**: $P(a < X \le b) = F(b) - F(a)$
   - CDF가 일부 점에서 불연속
   - $F(b) - F(a)$는 항상 잘 정의됨

**핵심**
- CDF의 불연속성과 무관하게 $P(a < X \le b) = F(b) - F(a)$는 **항상** 성립
- 이는 확률측도의 정의로부터 나오는 본질적 성질
- 불연속점에서의 점프 크기 = 그 점에서의 확률질량: $P(X=b) = F(b) - F(b-)$
- CDF가 불연속이어도 구간 확률 계산에는 문제가 없다. 구간의 열림/닫힘 여부에 따라 $a, b$에서 좌극한 또는 우극한을 적절히 선택하면 된다 (연습문제 1.6, 1.7)

#### 예 1.5.2 표준지수분포(standard exponential distribution)
확률변수 $X$가 확률밀도함수
$$
f(x)=
\begin{cases}
e^{-x}, & x\ge 0 \\
0, & x<0
\end{cases}
$$
를 가질 때, 누적분포함수를 구하여라.

**풀이**  
$x<0$일 때는 $F(x)=0$이다.

$x\ge 0$일 때는
$$
F(x)=\int_{-\infty}^x f(t)\,dt
=\int_0^x e^{-t}\,dt
=[-e^{-t}]_0^x
=1-e^{-x}
$$

따라서 누적분포함수는
$$
F(x)=
\begin{cases}
0, & x<0 \\
1-e^{-x}, & x\ge 0
\end{cases}
$$

또는 지표함수를 사용하여
$$
F(x)=(1-e^{-x})\mathbf{1}_{[0,\infty)}(x)
$$
로 나타낼 수 있다.

#### 누적분포함수의 성질 (정리 1.5.1)
* 단조증가:
  $$
  x_1<x_2 \Rightarrow F(x_1)\le F(x_2)
  $$

* 전체 변동:
  $$
  \lim_{x\to-\infty}F(x)=0,\quad \lim_{x\to\infty}F(x)=1
  $$

* 오른쪽 연속성:
  $$
  \lim_{h\downarrow 0}F(x+h)=F(x)
  $$

또한,
$$
F(a)-F(a^-)=P(X=a)
$$
가 성립한다.

**증명**  
**(1) 단조증가 (Monotonicity)**  
$x_1 < x_2$일 때, $\{X \le x_1\} \subseteq \{X \le x_2\}$이므로
정리 1.1.1(c)의 단조성에 의해
$$F(x_1) = P(X \le x_1) \le P(X \le x_2) = F(x_2)$$

**(2) 전체 변동 (Total Variation)**  
증가열 $A_n = \{X \le n\}$에 대해 $\bigcup_{n=1}^{\infty} A_n = S$이므로
정리 1.1.3(a)의 확률측도의 연속성에 의해
$$\lim_{x \to \infty} F(x) = \lim_{n \to \infty} P(X \le n) = P(S) = 1$$

감소열 $B_n = \{X \le -n\}$에 대해 $\bigcap_{n=1}^{\infty} B_n = \varnothing$이므로
정리 1.1.3(b)에 의해
$$\lim_{x \to -\infty} F(x) = \lim_{n \to \infty} P(X \le -n) = P(\varnothing) = 0$$

**(3) 오른쪽 연속성 (Right Continuity)**  
$h > 0$에 대해 감소열 $B_n = \{X \le x + \frac{1}{n}\}$을 생각하면
$$\bigcap_{n=1}^{\infty} B_n = \{X \le x\}$$

정리 1.1.3(b)에 의해
$$\lim_{h \downarrow 0} F(x+h) = \lim_{n \to \infty} P(X \le x + \tfrac{1}{n}) = P(X \le x) = F(x)$$

**(4) 한 점에서의 확률**  
증가열 $A_n = \{X \le a - \frac{1}{n}\}$에 대해
$$\bigcup_{n=1}^{\infty} A_n = \{X < a\}$$

정리 1.1.3(a)에 의해
$$F(a-) = \lim_{n \to \infty} F(a - \tfrac{1}{n}) = P(X < a)$$

따라서 $$ P(X = a) = P(X \le a) - P(X < a) = F(a) - F(a-)$$ ∎

### 5.2 확률생성함수 (Probability Generating Function, PGF)
음이 아닌 정수 값을 갖는 이산형 확률변수 $X$에 대해 확률생성함수는
$$
G(s)=E[s^X]=\sum_{k=0}^{\infty}s^kP(X=k)
$$
로 정의된다. 여기서 $|s| \le 1$일 때 이 급수가 수렴한다.

#### 성질
(a) **확률질량함수 복원**
$$
P(X=k)=\frac{G^{(k)}(0)}{k!}, \quad k=0,1,2,\ldots
$$

(b) **적률 계산**
$$
E[X]=G'(1), \quad \mathrm{Var}(X)=G''(1)+G'(1)-[G'(1)]^2
$$

(c) **분포 결정성**
$G_X(s)=G_Y(s)$이면 $X$와 $Y$의 분포는 동일하다.

#### 예 1.5.3
$P(X=k)=(1/2)^k$, $k=1,2,\ldots$ 일 때
$$
G(s)=\sum_{k=1}^{\infty}s^k(1/2)^k=\frac{s/2}{1-s/2}, \quad |s|<2
$$

- 이 예에서 처럼, 1보다 큰 s값에 대해서도 절대수렴할 수 있다
- PGF가 $|s|>1$에서도 수렴한다는 사실은 
   - 해당 분포가 지수적 꼬리를 가지며
   - MGF가 존재하고
   - 적률과 분포결정 이론을 강하게 적용할 수 있음을 의미

#### 예 1.5.4
베르누이 확률변수 $X\sim\mathrm{Bernoulli}(p)$에 대해
$$
G(s)=(1-p)+ps=1-p(1-s)
$$

### 5.3 적률생성함수 (Moment Generating Function, MGF)
확률생성함수 $G(s)$에서 $s>0$이면
$$
G(s)=E(s^X)=E(e^{X\log s})=E(e^{tX}), \quad t=\log s
$$
로 쓸 수 있다.

일반적으로, 0을 포함하는 열린구간 $(-h,h)$에서
$$
E(e^{tX})<\infty \quad \forall t\in(-h,h)\ (\exists h>0)
$$
일 때 $M(t) = E(e^{tX})$를 확률변수 $X$의 적률생성함수
(moment generating function, mgf)라 한다.

#### 예 1.5.5
앞면이 나올 때까지 동전을 던질 때
$$
M(t)=\frac{e^t/2}{1-e^t/2},\quad t<\log 2
$$

### 5.4 적률 (Moment)
지수함수의 멱급수 전개식으로부터
$$e^{tX} = \sum_{k=0}^{\infty} \frac{(tX)^k}{k!} = \sum_{k=0}^{\infty} \frac{t^k X^k}{k!}$$
임을 알고 있다. 여기에서 오른쪽 변의 기댓값을 항별로 취하여 더할 수 있다면
$$M(t) = E(e^{tX}) = \sum_{k=0}^{\infty} \frac{t^k}{k!} E(X^k)$$
일 것으로 예상할 수 있다. 이러한 전개식의 각 항에서 나타나는 $X^k$의 기댓값 $E(X^k)$를 $X$의 $k$차 적률(moment)이라 하며 기호로는 $m_k(X)$ 또는 $m_k$로 나타낸다.  
$$
m_k = E(X^k) = 
\begin{cases}
\displaystyle\sum_x x^k f(x), & X\text{가 이산형일 때} \\
\displaystyle\int_{-\infty}^{\infty} x^k f(x)\,dx, & X\text{가 연속형일 때}
\end{cases}
$$

여기서 무한 합이나 이상적분이 **절대수렴**하는 경우에만 적률이 실수로 정의된다.

#### 주요 적률들
* **1차 적률**: $m_1 = E(X) = \mu$ (평균)
* **2차 적률**: $m_2 = E(X^2)$
* **중심적률**: $E[(X-\mu)^k]$
   - 특히 $k=2$일 때: $E[(X-\mu)^2] = \mathrm{Var}(X)$ (분산)
- 참고
   - 적률생성함수가 존재하면 모든 적률이 존재한다
   - 그러나 역은 성립하지 않는다: 모든 적률이 존재해도 적률생성함수가 존재하지 않을 수 있다
      - 반례: 로그정규분포(lognormal distribution)는 모든 적률이 존재하지만, 
         $E(e^{tX})$가 어떤 $t>0$에 대해서도 발산하므로 MGF가 존재하지 않는다

**분산 계산에서의 활용**  
정리 1.4.2(b)에 의해
$$
\mathrm{Var}(X) = E(X^2) - [E(X)]^2 = m_2 - m_1^2
$$
이는 2차 적률과 1차 적률을 이용하여 분산을 계산하는 방법을 제공한다.

#### 적률생성함수의 성질 (정리 1.5.2)
**(a) 적률 생성 성질**  
적률생성함수가 0을 포함하는 열린구간에서 존재하면: 
$$
E(e^{tX})<\infty \quad \forall t\in(-h,h)\ (\exists h>0)
$$
모든 차수의 적률이 존재하며
$$
E(X^k) = M^{(k)}(0) = \left.\frac{d^k M(t)}{dt^k}\right|_{t=0}
$$
여기서 $M^{(k)}(0)$는 $t=0$에서의 $k$차 도함수이다.

**증명**  
적률생성함수의 멱급수 전개
$$M(t) = \sum_{k=0}^{\infty} \frac{t^k}{k!} E(X^k)$$
를 $k$번 미분하면
$$M^{(k)}(t) = \sum_{j=k}^{\infty} \frac{t^{j-k}}{(j-k)!} E(X^j)$$
$t=0$을 대입하면
$$M^{(k)}(0) = \sum_{j=k}^{\infty} \frac{0^{j-k}}{(j-k)!} E(X^j)$$

이 무한합에서 $j=k$인 항만 0이 아니므로
$$M^{(k)}(0) = \frac{0^0}{0!} E(X^k) = E(X^k)$$

따라서 $k$차 적률은 적률생성함수의 $k$차 도함수를 $t=0$에서 구한 값과 같다. ∎

>참고: $0^0 = 1$의 관례 (Convention $0^0 = 1$)
>확률론과 조합론에서는 일반적으로 $0^0 = 1$로 정의한다.
>1. **조합론적 해석**
>   - $n^k$는 $k$개 위치에 $n$개 원소를 배치하는 경우의 수
>   - $0^0$은 0개 위치에 0개 원소를 배치하는 경우의 수
>   - "아무것도 선택하지 않는" 방법은 정확히 한 가지 (공집합)
>
>2. **멱급수의 연속성**
>   - $e^x = \sum_{k=0}^{\infty} \frac{x^k}{k!}$에서 $x=0$일 때
>   - 첫 항: $\frac{0^0}{0!} = 1$ (이래야 $e^0 = 1$)
>
>주의사항
>- 해석학에서 극한 $\lim_{x \to 0^+} x^x = 1$이지만
>- 이중극한 $\lim_{(x,y) \to (0,0)} x^y$는 경로에 따라 다른 값
>- 따라서 문맥에 따라 $0^0$을 정의하거나 미정의로 남겨둠

**다른 증명 적률생성함수의 멱급수 전개**  
적률생성함수
$$
M(t)=E(e^{tX})
$$
가 어떤 $h>0$에 대해
$$
-h<t<h
$$
에서 존재한다고 가정한다.

**1. 지수함수의 멱급수와 기본 부등식**  
지수함수의 멱급수 전개로부터
$$
e^a=\sum_{k=0}^{\infty}\frac{a^k}{k!}
$$
가 성립하므로, 모든 $x,t$에 대하여
$$
\frac{|tx|^k}{k!}\le e^{|tx|}\le e^{tx}+e^{-tx}
$$
가 성립한다.

확률밀도함수(또는 확률질량함수)를 곱해 적분(또는 합)을 취하면
$$
\frac{|t|^k}{k!}E(|X|^k)\le M(t)+M(-t)<\infty
\quad(-h<t<h)
$$
이므로 $X$의 모든 적률 $E(|X|^k)$가 존재한다.

**2. 테일러 정리(적분형 나머지)**  
테일러 정리를 지수함수에 적용하면
$$
e^{tx}-\sum_{k=0}^{n-1}\frac{(tx)^k}{k!}
=\frac{(tx)^n}{(n-1)!}\int_0^1(1-u)^{n-1}e^{utx},du
$$
가 성립한다.

절댓값을 취하면
$$
\left|e^{tx}-\sum_{k=0}^{n-1}\frac{(tx)^k}{k!}\right|
\le \frac{|tx|^n}{(n-1)!}\int_0^1(1-u)^{n-1}e^{u|tx|},du
$$

**3. 나머지항의 지배**  
다음 부등식을 이용한다:
$$
\frac{|y|^n}{(n-1)!}\le \frac{1}{n}e^{|y|},\qquad
e^{2|a|}\le e^{3a}+e^{-3a}.
$$

따라서
$$
\left|e^{tx}-\sum_{k=0}^{n-1}\frac{(tx)^k}{k!}\right|
\le \frac{1}{n}\big(e^{3tx}+e^{-3tx}\big).
$$

**4. 기댓값 취하기**  
확률밀도함수를 곱해 적분(또는 합)을 취하면
$$
\left|
M(t)-\sum_{k=0}^{n-1}\frac{E(X^k)}{k!}t^k
\right|
\le \frac{1}{n}{M(3t)+M(-3t)},
\qquad -h<3t<h.
$$

**5. 극한**  
$|t|<h/3$에서 우변은 유한하고
$$
\frac{1}{n}{M(3t)+M(-3t)}\xrightarrow[n\to\infty]{}0
$$
이므로
$$
M(t)=\sum_{k=0}^{\infty}\frac{E(X^k)}{k!}t^k,
\qquad |t|<\varepsilon,\ \varepsilon=h/3>0.
$$

**결론**  
멱급수 표현으로부터
$$
M^{(k)}(0)=E(X^k),\qquad k=0,1,2,\dots
$$
가 성립한다. ∎

**(b) 분포 결정성 (Uniqueness Theorem)**  
두 확률변수 $X$와 $Y$의 적률생성함수가 0을 포함하는 열린구간에서 일치하면,
즉 $M_X(t) = M_Y(t)$ for all $t \in (-h, h)$ (for some $h > 0$)이면,
$X$와 $Y$는 같은 확률분포를 갖는다. (확률밀도함수, 누적분포함수가 일치)

증명은 생략,  
(적률생성함수를 일반화한 특성함수(characteristic function)을 이용한 증명은 Probability and Measure 3rd ed. 346~347p참고)  

**의미**  
이 성질은 적률생성함수가 확률분포를 **유일하게 결정**함을 의미한다.
따라서 두 확률변수의 분포가 같은지 확인하려면 적률생성함수만 비교하면 된다.

**주의사항**  
- 적률생성함수가 존재하지 않는 분포도 있다
- 모든 적률이 존재해도 적률생성함수가 존재하지 않을 수 있다 (ex: 로그정규분포)
- 이런 경우 특성함수(characteristic function) $\phi(t) = E(e^{itX})$를 사용한다

**(c) 선형변환**  
$Y = aX + b$일 때 ($a, b$는 상수)
$$M_Y(t) = e^{bt} M_X(at)$$

**증명**  
$$M_Y(t) = E(e^{tY}) = E(e^{t(aX+b)}) = e^{bt} E(e^{atX}) = e^{bt} M_X(at)$$

**(d) 독립 확률변수의 합**  
$X$와 $Y$가 독립이면 $Z = X + Y$의 적률생성함수는
$$M_Z(t) = M_X(t) \cdot M_Y(t)$$

**증명**  
독립성에 의해
$$M_Z(t) = E(e^{t(X+Y)}) = E(e^{tX} \cdot e^{tY}) = E(e^{tX}) \cdot E(e^{tY}) = M_X(t) \cdot M_Y(t)$$

이 성질은 독립 확률변수들의 합의 분포를 구할 때 매우 유용하다.

#### 예 1.5.6
표준지수분포의 적률생성함수를 구하고 평균과 분산을 계산하자.

**풀이**  
확률밀도함수가 $f(x) = e^{-x} \mathbf{1}_{[0,\infty)}(x)$일 때

$$M(t) = E(e^{tX}) = \int_0^{\infty} e^{tx} e^{-x}\,dx = \int_0^{\infty} e^{(t-1)x}\,dx$$

$t < 1$일 때 이 적분이 수렴하며
$$M(t) = \left[\frac{e^{(t-1)x}}{t-1}\right]_0^{\infty} = \frac{1}{1-t}, \quad t < 1$$

**멱급수 전개**  
적률생성함수를 멱급수로 전개하면

$$M(t) = \frac{1}{1-t} = \sum_{k=0}^{\infty} t^k, \quad |t| < 1$$

따라서
$$E(e^{tX}) = \sum_{k=0}^{\infty} \frac{E(X^k)}{k!} t^k = \sum_{k=0}^{\infty} t^k$$

계수를 비교하면
$$\frac{E(X^k)}{k!} = 1 \quad \Rightarrow \quad E(X^k) = k!$$

즉, 표준지수분포의 $k$차 적률은 $k!$이다.

**부분적분을 이용한 직접 계산**  
정의에 따라 적률을 직접 귀납적으로 계산하면
$$E(X^k) = \int_0^{\infty} x^k e^{-x}\,dx = k \int_0^{\infty} x^{k-1} e^{-x}\,dx = k \cdot E(X^{k-1})$$

따라서
$$E(X^k) = k!$$

이는 적률생성함수의 멱급수 전개로부터 얻은 결과와 일치한다.

#### 예 1.5.7
베르누이 확률변수 $X \sim \mathrm{Bernoulli}(p)$의 적률생성함수를 구하자.

**풀이**  
$P(X=0) = 1-p$, $P(X=1) = p$이므로

$$M(t) = E(e^{tX}) = (1-p)e^{0} + pe^{t} = 1-p+pe^t$$

또는
$$M(t) = 1 + p(e^t - 1)$$

평균과 분산:
$$E(X) = M'(0) = pe^t\big|_{t=0} = p$$

$$E(X^2) = M''(0) = pe^t\big|_{t=0} = p$$

$$\mathrm{Var}(X) = p - p^2 = p(1-p)$$

### 5.5 누율생성함수 (Cumulant Generating Function)
적률생성함수가 존재할 때
$$
C(t)=\log M(t)
$$
를 **누율생성함수**라 한다.

멱급수 전개로
$$
C(t)=\log M(t)=\sum_{r=1}^{\infty}\frac{c_r}{r!}t^r
$$

* $c_r$: $c_r(X)$나 $C^{(r)}(t=0)$라고도 하며, X의 **$r$차 누율(cumulant)** 이라고 한다.
* $c_1=E(X)$
* $c_2=\mathrm{Var}(X)$

#### 누율과 적률의 관계 (Relation between Cumulants and Moments)
**저차 누율과 적률**
- $c_1 = m_1 = E(X)$ (평균)
- $c_2 = m_2 - m_1^2 = \mathrm{Var}(X)$ (분산)
- $c_3 = m_3 - 3m_2m_1 + 2m_1^3$ (왜도 관련)
- $c_4 = m_4 - 4m_3m_1 - 3m_2^2 + 12m_2m_1^2 - 6m_1^4$ (첨도 관련)

**일반 공식**  
$M(t) = e^{C(t)}$의 양변을 미분하면
$$M'(t) = C'(t)e^{C(t)} = C'(t)M(t)$$

$t=0$에서
$$m_1 = E(X) = M'(0) = C'(0) = c_1$$

다시 미분하면
$$M''(t) = C''(t)M(t) + [C'(t)]^2M(t)$$

$t=0$에서
$$m_2 = M''(0) = C''(0) + [C'(0)]^2 = c_2 + c_1^2$$

따라서
$$c_2 = m_2 - m_1^2 = \mathrm{Var}(X)$$

> 누율, 적률 관계추가 정보: Statistics_01_추가_표준화 확률변수의 왜도_첨도.md 참고!

#### 예 1.5.8 
적률생성함수가 $M(t) = \frac{1}{1-2t}$일 때, 누율생성함수와 누율  
누율생성함수는  
$$C(t) = \log M(t) = \log\left(\frac{1}{1-2t}\right) = -\log(1-2t)$$

**멱급수 전개**  
테일러 급수 $\log(1-x) = -\sum_{r=1}^{\infty}\frac{x^r}{r}$ ($|x|<1$)을 이용하면

$$C(t) = -\log(1-2t) = \sum_{r=1}^{\infty}\frac{(2t)^r}{r} = \sum_{r=1}^{\infty}\frac{2^r}{r}t^r$$

**누율 계산**  
누율생성함수의 정의
$$C(t) = \sum_{r=1}^{\infty}\frac{c_r}{r!}t^r$$
와 비교하면

$$\frac{c_r}{r!} = \frac{2^r}{r}$$

따라서 $r$차 누율은
$$c_r = \frac{r! \cdot 2^r}{r} = (r-1)! \cdot 2^r$$

## 6. 여러 부등식 (Inequalities in Probability)
산술평균, 기하평균의 대소관계같이 항상 성립하는 부등식은 증명에 유용하다. 이런 부등식들을 추가로 정리했다.  

### 6.1 Jensen(젠센) 부등식 (Jensen's Inequality)
볼록함수(convex function)와 확률변수의 기댓값 사이의 관계를 나타내는 중요한 부등식이다.

#### 정리 1.6.1 (Jensen's Inequality)
함수 $\varphi: \mathbb{R} \to \mathbb{R}$가 볼록함수이고, $E[|X|] < \infty$, $E[|\varphi(X)|] < \infty$이면
$$\varphi(E[X]) \le E[\varphi(X)]$$

만약 $\varphi$가 오목함수(concave function)이면 부등호의 방향이 바뀐다:
$$\varphi(E[X]) \ge E[\varphi(X)]$$

#### 볼록함수의 정의 (Definition of Convex Function)
함수 $\varphi$가 **볼록함수**라는 것은 임의의 $x_1, x_2$와 $0 \le \lambda \le 1$에 대해
$$\varphi(\lambda x_1 + (1-\lambda)x_2) \le \lambda \varphi(x_1) + (1-\lambda)\varphi(x_2)$$
가 성립하는 것이다.

기하학적으로, 두 점을 연결한 선분이 함수 그래프보다 위에 있다는 의미이다.  
미분가능한 경우, $\varphi''(x) \ge 0$이면 볼록함수이다.

#### 증명 (Proof - 이산형의 경우)
$X$가 유한개의 값 $x_1, x_2, \ldots, x_n$을 확률 $p_1, p_2, \ldots, p_n$으로 갖는다고 하자.
($\sum_{i=1}^n p_i = 1$)

**수학적 귀납법으로 증명**  
**(1) $n=2$인 경우**  
$\lambda = p_1$, $1-\lambda = p_2$로 놓으면
$$\varphi(p_1 x_1 + p_2 x_2) \le p_1 \varphi(x_1) + p_2 \varphi(x_2)$$
이는 볼록함수의 정의 그 자체이다.

**(2) $n=k$일 때 성립한다고 가정**
$$\varphi\left(\sum_{i=1}^k p_i x_i\right) \le \sum_{i=1}^k p_i \varphi(x_i)$$

**(3) $n=k+1$일 때**
$q = p_1 + p_2 + \cdots + p_k$, $p_{k+1} = 1-q$로 놓으면

$$\varphi\left(\sum_{i=1}^{k+1} p_i x_i\right) = \varphi\left(q \cdot \frac{\sum_{i=1}^k p_i x_i}{q} + p_{k+1} x_{k+1}\right)$$

$n=2$인 경우를 적용하면
$$\le q \varphi\left(\frac{\sum_{i=1}^k p_i x_i}{q}\right) + p_{k+1} \varphi(x_{k+1})$$

귀납가정을 적용하면
$$\le q \sum_{i=1}^k \frac{p_i}{q} \varphi(x_i) + p_{k+1} \varphi(x_{k+1}) = \sum_{i=1}^{k+1} p_i \varphi(x_i)$$

따라서 수학적 귀납법에 의해 모든 유한 $n$에 대해 성립한다. ∎

**연속형의 경우**는 측도론적 논의가 필요하며 생략한다.

#### 응용 예시
**(a) 산술평균-기하평균 부등식 (AM-GM Inequality)**  
$\varphi(x) = -\log x$ (볼록함수, $x > 0$)에 Jensen 부등식을 적용하면
$$-\log E[X] \le E[-\log X] = -E[\log X]$$

따라서
$$\log E[X] \ge E[\log X]$$

양변에 지수를 취하면
$$E[X] \ge \exp(E[\log X])$$

특히 이산형인 경우
$$\frac{x_1 + x_2 + \cdots + x_n}{n} \ge \sqrt[n]{x_1 x_2 \cdots x_n}$$

**(b) 분산의 비음성**  
$\varphi(x) = x^2$ (볼록함수)에 Jensen 부등식을 적용하면
$$[E(X)]^2 = \varphi(E[X]) \le E[\varphi(X)] = E[X^2]$$

따라서
$$\mathrm{Var}(X) = E[X^2] - [E(X)]^2 \ge 0$$

### 6.2 Liapounov(리아푸노프) 부등식 (Lyapunov's Inequality)
적률 간의 크기 관계를 나타내는 부등식으로, Jensen 부등식의 응용이다.

#### 정리 1.6.2 (Lyapunov's Inequality)
$0 < r < s$이고 $E[|X|^s] < \infty$이면
$$[E(|X|^r)]^{1/r} \le [E(|X|^s)]^{1/s}$$

또한 등호는 $P(|X| = c) = 1$ (거의 확실히 상수)일 때만 성립한다.

#### 증명 (Proof)
$t = s/r > 1$로 놓으면 $\varphi(x) = x^t$ ($x \ge 0$)는 볼록함수이다.

확률변수 $Y = |X|^r$에 Jensen 부등식을 적용하면
$$[E(Y)]^t \le E[Y^t]$$

즉,
$$[E(|X|^r)]^{s/r} \le E[(|X|^r)^{s/r}] = E[|X|^s]$$

양변에 $r/s$ 제곱을 취하면 $$[E(|X|^r)]^{1/r} \le [E(|X|^s)]^{1/s}$$
∎

#### 특수한 경우들
**(a) $r=1, s=2$인 경우**  
$$E[|X|] \le \sqrt{E[X^2]}$$

즉, 평균의 절댓값은 제곱평균제곱근(RMS) 이하이다.

**(b) 일반화된 평균의 순서**
$0 < r_1 < r_2 < r_3$에 대해
$$[E(|X|^{r_1})]^{1/r_1} \le [E(|X|^{r_2})]^{1/r_2} \le [E(|X|^{r_3})]^{1/r_3}$$

#### 예 1.6.2
$X$가 $[0,1]$ 구간의 균등분포를 따를 때, Lyapunov 부등식을 확인해보자.

$$E[|X|^r] = \int_0^1 x^r\,dx = \frac{1}{r+1}$$

따라서
$$[E(|X|^r)]^{1/r} = \left(\frac{1}{r+1}\right)^{1/r}$$

$r < s$일 때
$$\left(\frac{1}{r+1}\right)^{1/r} \le \left(\frac{1}{s+1}\right)^{1/s}$$

실제로 $f(r) = (r+1)^{1/r}$는 $r > 0$에서 증가함수이므로 부등식이 성립한다.

### 6.3 Markov, Chebyshev 부등식 (Markov and Chebyshev Inequalities)
#### 정리 1.6.3 (Markov's Inequality)
확률변수 $Z$에 대해 $E(|Z|^r) < \infty \quad (r > 0)$이면 임의의 양수 $k$에 대해
$$P(|Z| \ge k) \le \frac{E[|Z|^r]}{k^r}$$

**증명**  
임의의 $k > 0$에 대해 지표함수(indicator function)를 이용하면
$$P(|Z| \ge k) = E[\mathbf{1}_{\{|Z| \ge k\}}]$$

$|Z| \ge k$일 때 $|Z|^r \ge k^r$이므로 $\frac{|Z|^r}{k^r} \ge 1$

따라서
$$\mathbf{1}_{\{|Z| \ge k\}} \le \frac{|Z|^r}{k^r}$$

양변에 기댓값을 취하면 (기댓값의 단조성, 정리 1.4.1(c))
$$P(|Z| \ge k) = E[\mathbf{1}_{\{|Z| \ge k\}}] \le E\left[\frac{|Z|^r}{k^r}\right] = \frac{E[|Z|^r]}{k^r}$$

**참고: 일반화의 의미**
- $r=1$: 기본 Markov 부등식
- $r=2$: $(Z-\mu)^2$에 적용하면 Chebyshev 부등식 유도
- 형태상 더 빠른 $k^{-r}$ 감쇠를 제공하나, 해당 적률이 유한할 때만 적용 가능

#### 정리 1.6.4 (Chebyshev's Inequality)
확률변수 $X$의 평균을 $\mu = E[X]$, 분산을 $\sigma^2 = \mathrm{Var}(X)\lt \infin$라 할 때, 임의의 $k > 0$에 대해
$$P(|X - \mu| \ge k) \le \frac{\sigma^2}{k^2}$$

**증명 1 (Markov 부등식 이용)**  
$(X-\mu)^2 \ge 0$에 Markov 부등식을 적용하면
$$P((X-\mu)^2 \ge k^2) \le \frac{E[(X-\mu)^2]}{k^2} = \frac{\sigma^2}{k^2}$$

$\{(X-\mu)^2 \ge k^2\} = \{|X-\mu| \ge k\}$이므로
$$P(|X - \mu| \ge k) \le \frac{\sigma^2}{k^2}$$

**증명 2 (직접 계산 - 연속형)**  
확률밀도함수를 $f(x)$라 하면
$$P(|X-\mu| \ge k) = \int_{|x-\mu| \ge k} f(x)\,dx$$

$|x-\mu| \ge k$인 영역에서 $\frac{(x-\mu)^2}{k^2} \ge 1$이므로
$$P(|X-\mu| \ge k) \le \int_{|x-\mu| \ge k} \frac{(x-\mu)^2}{k^2} f(x)\,dx \le \frac{1}{k^2}\int_{-\infty}^{\infty} (x-\mu)^2 f(x)\,dx = \frac{\sigma^2}{k^2}$$

#### 다른 표현
**(a) 표준편차의 배수로 표현**  
$k = c\sigma$ ($c > 0$)로 놓으면
$$P(|X - \mu| \ge c\sigma) \le \frac{1}{c^2}$$

예를 들어:
- $c=2$: $P(|X-\mu| \ge 2\sigma) \le 1/4 = 0.25$
- $c=3$: $P(|X-\mu| \ge 3\sigma) \le 1/9 \approx 0.111$

**(b) 여사건으로 표현**  
$$P(|X - \mu| < k) \ge 1 - \frac{\sigma^2}{k^2}$$

즉, 평균 근처에 있을 확률의 하한을 제공한다.

#### 예 1.6.1 (6시그마 법칙)
확률변수 $X$의 평균과 표준편차를 각각 $\mu$, $\sigma$라 하자.

Chebyshev 부등식에서 $k = 6\sigma$로 놓으면
$$P(|X-\mu| \ge 6\sigma) \le \frac{\sigma^2}{(6\sigma)^2} = \frac{1}{36} \approx 0.028$$

즉, 평균으로부터 $6\sigma$ 이상 떨어진 값이 관측될 확률은 최대 2.8%이다.

**의미**  
- 이는 분포의 형태와 무관하게 성립하는 상한(upper bound)이다
- 실제로 정규분포의 경우 $P(|X-\mu| \ge 6\sigma) \approx 2 \times 10^{-9}$로 훨씬 작다
- 6시그마 품질관리에서는 이 원리를 이용하여 불량률을 극소화한다
   - 공정의 변동을 $\pm 6\sigma$ 범위 내로 유지
   - 실제로는 정규분포를 가정하여 더 엄격한 기준 적용

**Chebyshev 부등식의 한계**  
- 분포의 형태를 고려하지 않으므로 매우 보수적(conservative)인 상한
- 정규분포, 지수분포 등 특정 분포에서는 훨씬 정확한 확률 계산 가능
- 그러나 분포를 모르는 경우에도 사용할 수 있다는 장점

#### 예 1.6.3
어떤 시험의 평균 점수가 70점, 표준편차가 10점일 때, 50점 이하 또는 90점 이상을 받을 학생의 비율은?

**풀이**  
$\mu = 70$, $\sigma = 10$이므로
$$P(|X - 70| \ge 20) \le \frac{100}{400} = 0.25$$

즉, 최대 25%의 학생이 50점 이하 또는 90점 이상을 받는다.

실제로 점수가 정규분포를 따른다면 이 비율은 약 4.6%로 훨씬 낮다.

### 6.4 분산이 0이라는 것의 의미
#### 정리 1.6.5 (Variance Zero Implies Constant)
$$\mathrm{Var}(X) = 0 \iff P(X = E[X]) = 1$$
즉, 분산이 0이면 확률변수는 **거의 확실히(almost surely)** 상수이다.

#### 증명 (Proof)
**($\Leftarrow$) 상수이면 분산이 0**  
$P(X = c) = 1$이면 $E[X] = c$이고
$$\mathrm{Var}(X) = E[(X-c)^2] = 0^2 \cdot 1 = 0$$

**($\Rightarrow$) 분산이 0이면 상수**  
$\mathrm{Var}(X) = 0$이라 하고 $\mu = E[X]$라 하자.

임의의 $\varepsilon > 0$에 대해 Chebyshev 부등식을 적용하면
$$P(|X - \mu| \ge \varepsilon) \le \frac{\mathrm{Var}(X)}{\varepsilon^2} = \frac{0}{\varepsilon^2} = 0$$

따라서
$$P(|X - \mu| \ge \varepsilon) = 0 \quad \text{for all } \varepsilon > 0$$

이는 다음을 의미한다:
$$P(X \neq \mu) = P\left(\bigcup_{n=1}^{\infty} \{|X-\mu| \ge 1/n\}\right) \le \sum_{n=1}^{\infty} P(|X-\mu| \ge 1/n) = 0$$

따라서
$$P(X = \mu) = 1$$

#### 참고: 거의 확실히(Almost Surely)의 의미
"거의 확실히 $X = c$"는 확률론에서 중요한 개념으로 $$P(X = c) = 1$$을 의미한다.  

이는 다음과 같은 의미를 갖는다:
1. $X \neq c$인 사건이 일어날 확률이 정확히 0
2. 하지만 $X \neq c$인 결과가 표본공간에 존재할 수는 있음
3. 단지 그런 결과들의 확률측도가 0

**예시**  
연속형 확률변수 $X$의 경우:
- 임의의 한 점 $a$에 대해 $P(X = a) = 0$
- 하지만 $X$는 실제로 어떤 값을 가져야 함
- "거의 확실히"라는 개념이 이런 측도론적 미묘함을 포착

#### 응용
**(a) 상수의 판별**  
확률변수 $X$가 실질적으로 상수인지 확인하려면 분산을 계산하면 된다.

**(b) 표본분산**  
표본 $X_1, X_2, \ldots, X_n$의 표본분산이 0이면 모든 관측값이 같다.

**(c) 통계적 추정**  
불편추정량(unbiased estimator) $\hat{\theta}$가 $\mathrm{Var}(\hat{\theta}) = 0$이면,
이는 완벽한 추정량으로 항상 참값 $\theta$와 같다.

추가 노트:  
## 측도론적 미묘함: 거의 확실히(Almost Surely)의 의미

### 1. 확률 0 vs 불가능
**핵심 구분**
- **불가능(Impossible)**: 표본공간에 존재하지 않음, $P(\varnothing) = 0$
- **확률 0(Probability Zero)**: 표본공간에 존재하지만 측도가 0, $P(A) = 0$이지만 $A \neq \varnothing$일 수 있음

### 2. 연속형 확률변수의 역설
#### 예시: 균등분포 $X \sim \text{Uniform}[0,1]$
표본공간은 $S = [0,1]$이고, 임의의 한 점 $a \in [0,1]$에 대해
$$P(X = a) = \int_a^a f(x)\,dx = 0$$

**역설적 상황**
- $X$는 반드시 $[0,1]$ 안의 어떤 값을 가져야 함
- 그런데 모든 개별 점의 확률이 0
- 즉, "$X$는 확률 0인 사건이 일어난 결과"

**해결**
- 가산 개의 확률 0 사건들의 합집합도 확률 0: $P(\{a_1, a_2, \ldots\}) = 0$
- 하지만 비가산 개(연속체)의 합은 1이 될 수 있음: $P([0,1]) = 1$
- 이것이 Lebesgue 측도론의 핵심

### 3. 거의 확실히(Almost Surely, a.s.)의 정의
#### 정의
사건 $A$가 **거의 확실히** 일어난다는 것은
$$P(A) = 1 \iff P(A^c) = 0$$

이는 "$A$가 일어나지 않는 것은 확률 0"을 의미한다.

#### 확실히(Surely) vs 거의 확실히(Almost Surely)
| 구분 | Surely | Almost Surely |
|------|--------|---------------|
| 정의 | $A = S$ | $P(A) = 1$ |
| 여사건 | $A^c = \varnothing$ | $P(A^c) = 0$ |
| 예외 가능성 | 불가능 | 측도 0인 예외 존재 가능 |

### 4. 구체적 예시
#### 예시 1: 무한 번 동전 던지기
무한 번 동전을 던질 때, "언젠가는 앞면이 나온다"

- 모든 시행에서 뒷면이 나올 확률: $P(\text{all tails}) = \lim_{n\to\infty}(1/2)^n = 0$
- 따라서 $P(\text{at least one head}) = 1$
- 하지만 "모든 시행에서 뒷면"인 무한수열 $(T,T,T,\ldots)$은 표본공간에 존재

#### 예시 2: 연속형 확률변수의 유리수
$X \sim \text{Uniform}[0,1]$일 때
$$P(X \in \mathbb{Q} \cap [0,1]) = 0$$

- 유리수 집합은 가산무한이므로 측도 0
- 하지만 $X$가 유리수 값을 가지는 것이 불가능한 것은 아님
- 단지 "거의 확실히 무리수"

### 5. 분산이 0일 때의 의미
$$\mathrm{Var}(X) = 0 \implies P(X = \mu) = 1$$

**해석**
- $X$는 거의 확실히 $\mu$와 같다
- 측도 0인 예외적인 결과에서는 $X \neq \mu$일 수 있음
- 하지만 그런 결과들의 확률은 정확히 0

**실용적 의미**
- 실제로는 "$X$는 상수 $\mu$"로 취급 가능
- 측도 0인 예외는 실질적으로 무시 가능
- 확률적 추론에서 "$X = \mu$"로 간주

### 6. 왜 "거의"라는 표현을 쓰는가
#### 수학적 엄밀성
- 측도론에서 "확률 1"과 "항상"은 다름
- $P(A) = 1$이어도 $A^c$가 공집합이 아닐 수 있음
- "거의 확실히"는 이런 측도론적 차이를 명시

#### 예시: 강대수의 법칙(Strong Law of Large Numbers)
표본평균 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$에 대해
$$P\left(\lim_{n\to\infty}\bar{X}_n = \mu\right) = 1$$

- 이는 "거의 모든 시행 경로에서 수렴"
- 하지만 수렴하지 않는 경로도 존재 가능 (측도 0)
- "거의 확실히 수렴"이라고 표현

### 7. 측도론적 통찰
#### Lebesgue 측도의 성질
1. **가산가법성(Countable Additivity)**
   $$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i) \quad \text{(서로소일 때)}$$

2. **가산 개의 영측도 집합의 합**
   $$P(A_i) = 0 \quad (i=1,2,\ldots) \implies P\left(\bigcup_{i=1}^{\infty} A_i\right) = 0$$

3. **비가산 합은 다를 수 있음**
   $$P(\{x\}) = 0 \text{ for all } x \in [0,1], \quad \text{but} \quad P([0,1]) = 1$$

#### 결론
"거의 확실히"라는 개념은:
- 확률론을 측도론 위에서 엄밀하게 구축하기 위해 필수적
- 연속형 확률변수를 다루기 위해 불가피
- "확률적으로 무시 가능한 예외"를 수학적으로 정확하게 표현
