# 1. 수열과 극한 *(Sequences and Limits)*
## (1) 수열의 정의
### Def. 1. [수열] *(Sequence)*
함수 $f:\mathbb N\to\mathbb R$를 수열 $\{a_n\}$이라 하고
$f(n)=a_n$을 $\{a_n\}$의 $n$번째 항이라 한다.

### Def. 2. [부분수열] *(Subsequence)*
수열 $\{a_n\}$에 대하여 자연수 $n_1<n_2<\cdots<n_k<\cdots$에 대해
$$
\{a_{n_k}\}
$$
를 $\{a_n\}$의 **부분수열**이라 한다.
 - ex: $n_k = \{1, 3, 5, ...\}$, $\{a_{n_k}\} = \{a_1, a_3, a_5, ... \}$
### Def. 3. [증가(감소)수열] *(Monotone Sequence)*
1. $\forall n\in\mathbb N,\ a_n\le a_{n+1}$이면 $\{a_n\}$을 **단조증가수열** *(monotone increasing sequence)* 이라 한다.  
- ($a_n<a_{n+1}$이면 **순증가수열**)

2. $\forall n\in\mathbb N,\ a_n\ge a_{n+1}$이면 $\{a_n\}$을 **단조감소수열**
   *(monotone decreasing sequence)* 이라 한다.

### Def. 4. [유계인 수열] *(Bounded Sequence)*

$\exists M>0$ s.t. $\forall n\in\mathbb N,\ |a_n|\le M$이면
$\{a_n\}$을 **유계 수열**이라 한다.

## (2) 수열의 극한 *(Limit of a Sequence)*
### Def. 1. [수열의 수렴] *(Convergence of a Sequence)*
$a\in\mathbb R$이라 하자.
$\forall\varepsilon>0,\ \exists N\in\mathbb N$ s.t.
$$
\forall n\ge N,\ |a_n-a|<\varepsilon
$$
이면 수열 $\{a_n\}$은 $a$로 **수렴**한다고 하고
$$
\lim_{n\to\infty}a_n=a
$$
로 쓴다.

### Def. 2. [수열의 발산] *(Divergence of a Sequence)*
수열 ${a_n}$이 어떤 실수 $a$로도 수렴하지 않으면
${a_n}$은 **발산**한다고 한다.

### Thm. 1. [수열 극한의 유일성] *(Uniqueness of Limit)*
수열 $\{a_n\}$이 수렴하면 그 극한은 유일하다.

### Thm. 2. [수열 극한의 연산] *(Limit Laws for Sequences)*
$\lim a_n=a$, $\lim b_n=b$이면 다음이 성립한다.
1. $\lim(a_n\pm b_n)=a\pm b$
2. $\lim(a_nb_n)=ab$
3. $\lim\dfrac{a_n}{b_n}=\dfrac ab$
   (단, $b\ne0$, $\forall n,\ b_n\ne0$)

## (3) 코시수열 *(Cauchy Sequence)*
### Def. 1. [코시수열] *(Cauchy Sequence)*
$\forall\varepsilon>0,\ \exists N\in\mathbb N$ s.t. $\forall m, n \in \mathbb{N}$ with
$$
m\ge n\gt N,\ |a_m-a_n|<\varepsilon
$$
이면 $\{a_n\}$을 **코시수열**이라 한다.

### Thm. 1. [코시수열과 수렴성] *(Cauchy Criterion)*
$\{a_n\}$이 코시수열이면 $\{a_n\}$은 수렴한다.

### 참고) Def. 2. [실수의 구성적 정의] *(Construction of $\mathbb R$)*
1. 유리수 코시수열의 집합 $\mathbb R^*$에 대해
   동치관계
   $$
   \{a_n\}\sim\{b_n\}\iff \lim_{n\to \infin}(a_n-b_n)=0
   $$
   를 정의한다.

2. 이 동치류의 집합을 $\mathbb R$이라 한다.
  - 예: 1로 수렴하는 수열은 수많음. 이를 집합으로, 즉 동치류로 정의

### 참고) Thm. 2. [실수의 완비성] *(Completeness of $\mathbb R$)*
$\mathbb R$의 공집합이 아닌 부분집합이 위로 유계이면
그 부분집합은 **상한** *(supremum)* 을 갖는다.

# 2. 주요 정리 *(Main Theorems)*
## (1) 단조수렴정리 *(Monotone Convergence Theorem)*
### Thm. 1.
$\{a_n\}$이 단조증가(또는 감소)하고 위(아래)로 유계이면
$\{a_n\}$은 수렴한다.

#### 증명
단조증가하고 위로 유계인 경우만 증명한다. (감소의 경우도 유사)

$\{a_n\}$이 단조증가하고 위로 유계라고 하자.

$A = \{a_n : n \in \mathbb{N}\}$라 하면, $A$는 공집합이 아니고 위로 유계이므로
실수의 완비성에 의해 상한 $\alpha = \sup A$가 존재한다.

이제 $\lim_{n\to\infty} a_n = \alpha$임을 보이자.

$\varepsilon > 0$이 주어졌을 때, $\alpha - \varepsilon$은 $A$의 상한이 아니므로
$$
\exists N \in \mathbb{N} \text{ s.t. } a_N > \alpha - \varepsilon
$$

$\{a_n\}$이 단조증가하므로 $n \ge N$일 때
$$
\alpha - \varepsilon < a_N \le a_n \le \alpha < \alpha + \varepsilon
$$

따라서 $n \ge N$이면 $|a_n - \alpha| < \varepsilon$이므로
$$
\lim_{n\to\infty} a_n = \alpha
$$

### Thm. 2. [축소구간정리] *(Nested Interval Theorem)*
실수의 완비성을 보이는 다른 버전.  

모든 $n\in\mathbb N$에 대해 $[a_n,b_n]$이
1. 닫힌 구간이고
2. $[a_{n+1},b_{n+1}]\subset[a_n,b_n]$이며
3. $\lim(b_n-a_n)=0$이면

$$
\bigcap_{n=1}^\infty[a_n,b_n]=\{\alpha\}
$$
인 $\alpha\in\mathbb R$가 존재한다.

#### 증명
$\{a_n\}$은 단조증가이고 위로 유계이므로, $A = \{a_n : n \in \mathbb{N}\}$라 하면
실수의 완비성에 의해 $\alpha = \sup A$가 존재한다.

$\{b_n\}$은 단조감소이고 아래로 유계이므로, $B = \{b_n : n \in \mathbb{N}\}$라 하면
$\beta = \inf B$가 존재한다.

조건 2에 의해 $\forall n, a_n \le b_n$이므로 $\alpha \le \beta$이다.

단조수렴정리(Thm. 1)에 의해
$$
\lim_{n\to\infty} a_n = \alpha, \quad \lim_{n\to\infty} b_n = \beta
$$

조건 3에 의해 $\lim_{n\to\infty}(b_n - a_n) = 0$이므로
$$
\beta - \alpha = \lim_{n\to\infty}(b_n - a_n) = 0
$$

따라서 $\alpha = \beta$이고, 모든 $n$에 대해 $a_n \le \alpha \le b_n$이므로
$$
\alpha \in \bigcap_{n=1}^\infty [a_n, b_n]
$$

유일성: 만약 $\gamma \in \bigcap_{n=1}^\infty [a_n, b_n]$이면
모든 $n$에 대해 $a_n \le \gamma \le b_n$이고,
극한을 취하면 $\alpha \le \gamma \le \beta = \alpha$이므로 $\gamma = \alpha$이다.

## (2) B–W 정리 *(Bolzano–Weierstrass Theorem)*
### Thm. 1. [샌드위치 정리] *(Squeeze Theorem)*
$a_n\le b_n\le c_n$이고
$\lim a_n=\lim c_n=L$이면
$$
\lim b_n=L
$$

### Thm. 2. [볼차노–바이어슈트라스 정리] *(Bolzano–Weierstrass Theorem)*
위아래로 유계인 수열 $\{a_n\}$은 수렴하는 부분수열을 갖는다.
  - 진동해도 상관없음

#### 증명
$\{a_n\}$이 유계라고 하자. 즉, $\exists M > 0$ s.t. $\forall n \in \mathbb{N}, |a_n| \le M$.

구간 $I_1 = [-M, M]$을 이등분하여 $[-M, 0]$, $[0, M]$ 두 구간을 만들면,
비둘기집 원리에 의해 적어도 하나의 구간에는 무한히 많은 $a_n$이 존재한다.
이 구간을 $I_2$라 하고, $a_{n_1} \in I_2$를 선택한다.

$I_2$를 다시 이등분하여 무한히 많은 $a_n$을 포함하는 구간을 $I_3$라 하고,
$n_2 > n_1$인 $a_{n_2} \in I_3$를 선택한다.

이 과정을 반복하면 닫힌 구간의 수열 $\{I_k\}$와 부분수열 $\{a_{n_k}\}$를 얻는다.
- $I_{k+1} \subset I_k$
- $I_k$의 길이는 $\frac{2M}{2^{k-1}}$이므로 $\lim_{k\to\infty}(\text{length of } I_k) = 0$
- $a_{n_k} \in I_k$

축소구간정리(Thm. 2)에 의해 $\bigcap_{k=1}^\infty I_k = \{\alpha\}$인 $\alpha \in \mathbb{R}$가 존재한다.

임의의 $\varepsilon > 0$에 대해, 충분히 큰 $K$를 선택하면 $I_K$의 길이가 $\varepsilon$보다 작고,
$k \ge K$일 때 $a_{n_k}, \alpha \in I_K\subset I_K$이므로
$$
|a_{n_k} - \alpha| < \varepsilon
$$

따라서 $\lim_{k\to\infty} a_{n_k} = \alpha$이다.

### Cor. [최대 최소정리] *(Extreme Value Theorem)*
$f$가 $[a,b]$에서 연속이면
$\exists a_0,b_0\in[a,b]$ s.t.
$$
f(a_0)\le f(x)\le f(b_0)\quad(\forall x\in[a,b])
$$

#### 증명
$f$가 $[a,b]$에서 연속이므로 최댓값을 가짐을 보이자. (최솟값도 유사)

**Step 1.** $f$가 위로 유계임을 보이자.

귀류법으로, $f$가 위로 유계가 아니라고 가정하면,
$$
\forall n \in \mathbb{N}, \exists x_n \in [a,b] \text{ s.t. } f(x_n) > n
$$

**Step 2.** 수열 $\{x_n\} \subset [a,b]$는 유계이므로 B-W 정리에 의해
수렴하는 부분수열 $\{x_{n_k}\}$가 존재한다. $\lim_{k\to\infty} x_{n_k} = c$라 하면 $c \in [a,b]$이다.

**Step 3.** $f$가 $c$에서 연속이므로
$$
\lim_{k\to\infty} f(x_{n_k}) = f(c)
$$

그런데 $f(x_{n_k}) > n_k \ge k$이므로 $\lim_{k\to\infty} f(x_{n_k}) = \infty$인데,
이는 $f(c)$가 유한한 실수라는 것과 모순이다.

따라서 $f$는 위로 유계이다. 즉, $M = \sup\{f(x) : x \in [a,b]\} < \infty$이다.

**Step 4.** $M$이 최댓값임을 보이자.  
상한의 정의에 의해
$$
\forall n \in \mathbb{N}, \exists x_n \in [a,b] \text{ s.t. } M - \frac{1}{n} < f(x_n) \leq M
$$

수열 $\{x_n\}$은 유계이므로 B-W 정리에 의해 수렴하는 부분수열 $\{x_{n_k}\}$가 존재하고,
$\lim_{k\to\infty} x_{n_k} = b_0 \in [a,b]$이다.

$f$의 연속성에 의해
$$
f(b_0) = \lim_{k\to\infty} f(x_{n_k})
$$

그런데 $f(x_{n_k}) > M - \frac{1}{n_k}$이고 $\lim_{k\to\infty}\left(M - \frac{1}{n_k}\right) = M$이므로,
샌드위치 정리에 의해 $f(b_0) = M$이다.

따라서 $\exists b_0 \in [a,b]$ s.t. $f(b_0) = M = \sup\{f(x) : x \in [a,b]\}$.

# 3. 급수와 극한 *(Series and Limits)*
## (1) 급수의 정의 *(Definition of a Series)*
### Def. 1. [급수] *(Series)*
수열 $\{a_n\}$에 대하여
$$
a_1+a_2+a_3+\cdots=\sum_{n=1}^\infty a_n
$$
을 **(무한)급수**라 한다.

$n$번째 부분합을
$$
S_n=\sum_{k=1}^n a_k
$$
라 한다.

### Def. 2. [재배열급수] *(Rearranged Series)*
$f:\mathbb N\to\mathbb N$이 전단사 함수일 때
$$
\sum_{n=1}^\infty a_{f(n)}
$$
을 재배열급수라 한다.
  - 더해지는 순서가 중요하므로, 의미있는 급수다.

## (2) 급수의 극한 *(Convergence of Series)*
### Def. 1. [급수의 수렴과 발산]
부분합 수열 $\{S_n\}$이 $S$로 수렴하면
$$
\sum_{n=1}^\infty a_n=S
$$
S로 수렴한다 하고, 그렇지 않으면 발산한다고 한다.
- 급수의 수렴, 발산 확인은 수열의 수렴, 발산보다 어렵다
- 케바케로 다양한 제안이 서술되어있다.

#### 예시
1. **기하급수** *(Geometric Series)*
    $$
    \sum_{n=0}^\infty r^n = 1 + r + r^2 + r^3 + \cdots
    $$
    - $|r| < 1$이면 $\dfrac{1}{1-r}$로 수렴
    - $|r| \ge 1$이면 발산

2. **조화급수** *(Harmonic Series)*
    $$
    \sum_{n=1}^\infty \frac{1}{n} = 1 + \frac{1}{2} + \frac{1}{3} + \cdots
    $$
    발산한다.

3. **p-급수** *(p-Series)*
    $$
    \sum_{n=1}^\infty \frac{1}{n^p}
    $$
    - $p > 1$이면 수렴
    - $p \le 1$이면 발산

4. **교대조화급수** *(Alternating Harmonic Series)*
    $$
    \sum_{n=1}^\infty \frac{(-1)^{n+1}}{n} = 1 - \frac{1}{2} + \frac{1}{3} - \frac{1}{4} + \cdots
    $$
    이 급수는 **조건수렴**한다.
    
    - **수렴성**: 교대급수 판정법(Alternating Series Test)에 의해 수렴한다.
        * $\frac{1}{n}$은 단조감소하고
        * $\lim_{n\to\infty}\frac{1}{n} = 0$이므로
        * 급수는 수렴한다. (실제로 $\ln 2$로 수렴)
    
    * $\ln 2$로 수렴하는 이유
    
    부분합을 $S_n = \sum_{k=1}^n \frac{(-1)^{k+1}}{k}$라 하자.
    
    $2n$번째 부분합은
    $$
    \begin{align}
    S_{2n} &= \left(1-\frac{1}{2}\right) + \left(\frac{1}{3}-\frac{1}{4}\right) + \cdots + \left(\frac{1}{2n-1}-\frac{1}{2n}\right) \\
    &= \sum_{k=1}^{n} \left(\frac{1}{2k-1} - \frac{1}{2k}\right) \\
    &= \sum_{k=1}^{n} \frac{1}{2k(2k-1)}
    \end{align}
    $$
    
    한편, $\ln 2$의 적분 표현을 이용하면
    $$
    \ln 2 = \int_1^2 \frac{1}{x}dx
    $$
    
    리만 합과의 관계를 통해, 또는 Taylor 급수를 이용하면
    $$
    \ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4} + \cdots = \sum_{n=1}^\infty \frac{(-1)^{n+1}x^n}{n}
    $$
    
    $x=1$을 대입하면 (수렴반경 경계에서의 수렴성 확인 필요)
    $$
    \ln 2 = 1 - \frac{1}{2} + \frac{1}{3} - \frac{1}{4} + \cdots = \sum_{n=1}^\infty \frac{(-1)^{n+1}}{n}
    $$

    - **절대수렴하지 않음**: 
        $$
        \sum_{n=1}^\infty \left|\frac{(-1)^{n+1}}{n}\right| = \sum_{n=1}^\infty \frac{1}{n}
        $$
        은 조화급수이므로 발산한다.
    
    따라서 이 급수는 조건수렴한다.

### Def. 2. [절대수렴과 조건수렴]
1. $\sum|a_n|$이 수렴하면 $\sum a_n$은 **절대수렴** *(absolutely convergent)* 한다.

2. $\sum a_n$은 수렴하나 $\sum|a_n|$은 발산하면
   **조건수렴** *(conditionally convergent)* 한다.
  - 조건수렴은 재배열급수의 경우 상황이 달라지기도 함

## (3) 여러 가지 정리
### Thm. 1.
$\alpha, \beta \in \mathbb{R}$, 
$\sum a_n=\alpha$, $\sum b_n=\beta$이면
$$
\sum(a_n\pm b_n)=\alpha\pm \beta
$$

### Thm. 2.
$\sum a_n$이 수렴하면
$$
\lim_{n\to\infty}a_n=0
$$

### Thm. 3.
$\sum a_n$이 절대수렴하면 임의의 재배열 급수도 수렴하며 그 합은 같다. (조건수렴은 그렇지 않다)

#### 증명
$\sum a_n$이 절대수렴한다고 하자. 즉, $\sum |a_n| = L < \infty$이다.

$f:\mathbb{N} \to \mathbb{N}$을 전단사 함수라 하고, 재배열급수 $\sum a_{f(n)}$을 고려하자.

**Step 1.** 재배열급수가 수렴함을 보이자.

재배열급수의 부분합을 $T_m = \sum_{k=1}^m a_{f(k)}$라 하자.

임의의 $\varepsilon > 0$에 대해, $\sum |a_n|$이 수렴하므로
$$
\exists N \in \mathbb{N} \text{ s.t. } \sum_{n=N+1}^\infty |a_n| < \frac{\varepsilon}{2}
$$

$M$을 충분히 크게 선택하여 $\{1, 2, \ldots, N\} \subset \{f(1), f(2), \ldots, f(M)\}$이 되도록 하자.

$m, m' \ge M$일 때, $T_m - T_{m'}$에 포함되는 항들은 모두 $n > N$인 $a_n$들이므로
$$
|T_m - T_{m'}| \le \sum_{n=N+1}^\infty |a_n| < \frac{\varepsilon}{2} < \varepsilon
$$

따라서 $\{T_m\}$은 코시수열이고, 코시수열은 수렴하므로 재배열급수는 수렴한다.

**Step 2.** 재배열급수의 합이 원래 급수의 합과 같음을 보이자.

원래 급수의 합을 $S = \sum_{n=1}^\infty a_n$, 재배열급수의 합을 $T = \sum_{n=1}^\infty a_{f(n)}$이라 하자.

임의의 $\varepsilon > 0$에 대해, $\exists N$ s.t. $\sum_{n=N+1}^\infty |a_n| < \varepsilon$이고
$$
\left|S - \sum_{k=1}^N a_k\right| < \varepsilon
$$

$M$을 충분히 크게 선택하여 $\{1, 2, \ldots, N\} \subset \{f(1), f(2), \ldots, f(M)\}$이 되도록 하면
$$
\left|T - \sum_{k=1}^N a_k\right| = \left|\sum_{n=1}^\infty a_{f(n)} - \sum_{k=1}^N a_k\right| < \varepsilon
$$

따라서
$$
|S - T| \le \left|S - \sum_{k=1}^N a_k\right| + \left|T - \sum_{k=1}^N a_k\right| < 2\varepsilon
$$

$\varepsilon$은 임의이므로 $S = T$이다.

# [연습문제]
1. 다음 명제의 참·거짓을 판별하고 설명하시오.  
(1) $\{a_n\}$이 유계 $\Rightarrow \lim a_n$ 존재  
(2) $\lim(a_n+b_n)=\infty\Rightarrow \lim a_n=\infty$ 또는 $\lim b_n=\infty$  
(3) $\{a_n\}$이 수렴하는 부분수열을 가지면 유계이다  
(4) $\{a_n\}$이 수렴 $\Rightarrow \{a_n\}$은 코시수열   
(5) $\forall n \in \mathbb{N}$, $|a_{n+1}-a_n| \le \frac{1}{n}$이면 $\{a_n\}$은 코시수열이다  
(6) $\sum(a_n+b_n)$이 수렴 $\Rightarrow$ $\sum a_n$이 수렴하고 $a_1 + b_1 + a_2 + b+2 + ...$이 수렴한다  
(7) $\sum a_n = \alpha \land \sum b_n = \beta$이면 $\sum a_nb_n = \alpha\beta$  

2. $a_1=1$, $a_{n+1}=\sqrt{1+a_n}$으로 정의된 수열의 수렴성을 판별하시오.

3. 다음 급수의 수렴성을 조사하시오.
   $$
   1-\frac12+\frac13-\frac14+\cdots
   $$
