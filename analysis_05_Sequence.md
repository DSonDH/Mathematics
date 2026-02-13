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
### 추가: [증가(감소)수열] *(Monotone Sequence)*
1. $\forall n\in\mathbb N,\ a_n\le a_{n+1}$이면 $\{a_n\}$을 **단조증가수열** *(monotone increasing sequence)* 이라 한다.  
- ($a_n<a_{n+1}$이면 **순증가수열**)

2. $\forall n\in\mathbb N,\ a_n\ge a_{n+1}$이면 $\{a_n\}$을 **단조감소수열**
   *(monotone decreasing sequence)* 이라 한다.

### Def. 4. [유계인 수열] *(Bounded Sequence)*
$\exists M>0$ s.t. $\forall n\in\mathbb N,\ |a_n|\le M$이면
$\{a_n\}$을 **유계 수열**이라 한다.

### Thm. ⭐유계인 단조 실수열은 항상 수렴한다. **[Monotone Convergence Theorem]**
#### 증명
$\{a_n\}$이 단조증가하고 유계인 경우만 증명한다. (감소의 경우도 유사)  

$\{a_n\}$이 단조증가하고 유계라고 하자.
$A = \{a_n : n \in \mathbb{N}\}$라 하면, $A$는 공집합이 아니고 위로 유계이므로
실수의 완비성에 의해 상한 $\alpha = \sup A$가 존재한다.
이제 $\lim_{n\to\infty} a_n = \alpha$임을 보이자.
$\varepsilon > 0$이 주어졌을 때, $\alpha - \varepsilon$은 $A$의 상한이 아니므로
$$
\exists N \in \mathbb{N} \text{ s.t. } a_N > \alpha - \varepsilon
$$
따라서 모든 $n \ge N$에 대하여
$$
\alpha - \varepsilon < a_N \le a_n \le \alpha \lt \alpha + \varepsilon
$$
즉, $|a_n - \alpha| < \varepsilon$이다.
따라서 $\lim_{n\to\infty} a_n = \alpha$.

- 부분합 수열의 수렴성을 직접 증명할 필요 없이, 단조성과 유계성만 확인하면 수렴을 보장한다.
- 실수의 완비성(completeness)과의 깊은 연결을 보여준다.
- 수열의 극한값을 구체적으로 몰라도 수열의 수렴성을 확인할 수 있다.

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

#### 증명
만일 $\lim_{n\to\infty} a_n = a$이고 $\lim_{n\to\infty} a_n = b$라고 하자.
$a \ne b$라고 하면, $\varepsilon = \frac{|a-b|}{3} > 0$에 대하여
$$
\exists N_1 \in \mathbb{N} \text{ s.t. } \forall n \ge N_1, |a_n - a| < \varepsilon \\
\exists N_2 \in \mathbb{N} \text{ s.t. } \forall n \ge N_2, |a_n - b| < \varepsilon
$$
$n \ge \max\{N_1, N_2\}$일 때
$$
|a - b| = |a - a_n + a_n - b| \le |a - a_n| + |a_n - b| < 2\varepsilon = \frac{2|a-b|}{3}
$$
이는 모순이다. 따라서 $a = b$이다.

### Thm. 2. 수렴하는 수열은 유계이다. *(A Convergent Sequence is Bounded)*
증명  
$\lim_{n\to\infty} a_n = a$이라 하자. $\varepsilon = 1$이라 하면,
$$
\exists N \in \mathbb{N} \text{ s.t. } \forall n \ge N, |a_n - a| < 1
$$
따라서 $n \ge N$일 때 $|a_n| < |a| + 1$이다.
또한, $n = 1, 2, \ldots, N-1$에 대하여
$$
M_1 = \max\{ |a_1|, |a_2|, \ldots, |a_{N-1}| \}
$$
이라 하면, 모든 $n \in \mathbb{N}$에 대하여
$$
|a_n| \le \max\{ M_1, |a| + 1 \}
$$
이므로 $\{a_n\}$은 유계이다.

### Thm. 3. [수열 극한의 연산] *(Limit Laws for Sequences)*
$\lim a_n=a$, $\lim b_n=b$이면 다음이 성립한다.
1. $\lim(a_n\pm b_n)=a\pm b$
2. $\lim(a_nb_n)=ab$
3. $\lim\dfrac{a_n}{b_n}=\dfrac ab$
   (단, $b\ne0$, $\forall n,\ b_n\ne0$)

>### 추가: 실수 수열의 상한과 하한 *(Supremum and Infimum)*
>실수 수열 $\{a_n\}_{n\in\mathbb{N}}\subset\mathbb{R}$의 값집합을
>$$A:=\{a_n:n\in\mathbb{N}\}\subset\mathbb{R}$$
>이라 하자.
>
>**1. 상한 (Supremum)**  
>수열의 상한은 집합 $A$의 상한으로 정의한다: $\sup a_n := \sup A$
>
>(1) 상방유계: $\sup a_n < \infty \Leftrightarrow \exists M\in\mathbb{R}$ s.t. $\forall n\in\mathbb{N}, a_n\le M$
>
>(2) 양의 무한대: $\sup a_n = +\infty \Leftrightarrow \forall K\in\mathbb{R}, \exists n\in\mathbb{N}$ s.t. $a_n > K$
>
>(3) 유한한 상한: $u\in\mathbb{R}$에 대하여 $\sup a_n = u$  
>$\Leftrightarrow \forall n\in\mathbb{N}, a_n \le u$이고 $\forall K\in\mathbb{R}$, $a_n \le K$ (모든 $n$) $\Rightarrow u \le K$  
>$\Leftrightarrow \forall n\in\mathbb{N}, a_n \le u$이고 $ \forall \varepsilon > 0, \exists n\in\mathbb{N}$ s.t. $a_n > u - \varepsilon$  
>  - $\forall n\in\mathbb{N}, a_n \le u$를 상계 조건이라 한다
>  - $\forall K\in\mathbb{R}$, $a_n \le K$ (모든 $n$) $\Rightarrow u \le K$를 최소성 조건이라 한다
>
>**2. 하한 (Infimum)**  
>마찬가지로 $\inf a_n := \inf A$로 정의한다.
>
>(1) 하방유계: $\inf a_n > -\infty \Leftrightarrow \exists m\in\mathbb{R}$ s.t. $\forall n\in\mathbb{N}, a_n\ge m$  
>
>(2) 음의 무한대: $\inf a_n = -\infty \Leftrightarrow \forall K\in\mathbb{R}, \exists n\in\mathbb{N}$ s.t. $a_n < K$
>(3) 유한한 하한: $\ell\in\mathbb{R}$에 대하여 $\inf a_n = \ell$  
>$\Leftrightarrow \forall n\in\mathbb{N}, a_n \ge \ell$이고 $\forall K\in\mathbb{R}$, $a_n \ge K$ (모든 $n$) $\Rightarrow \ell \ge K$  
>$\Leftrightarrow \forall n\in\mathbb{N}, a_n \ge \ell$이고 $\forall \varepsilon > 0, \exists n\in\mathbb{N}$ s.t. $a_n < \ell + \varepsilon$  
>  - $\forall n\in\mathbb{N}, a_n \ge \ell$를 하계 조건이라 한다
>  - $\forall K\in\mathbb{R}$, $a_n \ge K$ (모든 $n$) $\Rightarrow \ell \ge K$를 최대성 조건이라 한다
>
>#### 예시
>**1. $a_n = 1 - \frac{1}{n}$**  
>값집합: $A = \{0, \frac{1}{2}, \frac{2}{3}, \frac{3}{4}, \ldots\}$
>- $\sup a_n = 1$ (상방유계, 극한값)
>- $\inf a_n = 0$ (최솟값)
>
>**2. $a_n = \frac{(-1)^n}{n}$**  
>값집합: $A = \{-1, \frac{1}{2}, -\frac{1}{3}, \frac{1}{4}, \ldots\}$
>- $\sup a_n = \frac{1}{2}$ (최댓값)
>- $\inf a_n = -1$ (최솟값)
>
>**3. $a_n = (-1)^n n$**  
>값집합: $A = \{-1, 2, -3, 4, -5, 6, \ldots\}$
>- $\sup a_n = +\infty$ (위로 무한)
>- $\inf a_n = -\infty$ (아래로 무한)
>
>**4. $a_n = n\left(-\frac{1}{2}\right)^n$**  
>값집합: $A = \{-\frac{1}{2}, \frac{1}{2}, -\frac{3}{8}, \frac{1}{4}, \ldots\}$
>- $\sup a_n = \frac{1}{2}$ (최댓값, $n=2$일 때)
>- $\inf a_n = -\frac{1}{2}$ (최솟값, $n=1$일 때)
>
>### 추가: 실수 수열의 상극한과 하극한 *(Limit Superior and Limit Inferior)*
>실수 수열 $\{a_n\}$에 대하여
>
>**상극한 (Limit Superior)**
>각 $n$에 대하여 $M_n := \sup_{k\ge n} a_k$로 정의하면, $\{M_n\}$은 단조감소수열이다. 따라서 $\{M_n\}$이 아래로 유계이면 수렴하며, 이 극한을 상극한이라 정의한다:
>$$
>\limsup_{n\to\infty} a_n := \overline{\lim_{n\to\infty}} a_n = \lim_{n\to\infty} \sup_{k\ge n} a_k
>$$
>
>**상극한의 다른 정의**  
>상극한의 정의로부터 다음이 성립한다:
>$$\limsup_{n\to\infty} a_n = \bar{l} \in \mathbb{R} \Leftrightarrow \forall \varepsilon > 0, \begin{cases} a_n < \bar{l} + \varepsilon & \text{a.b.f.}(n) \\ a_n > \bar{l} - \varepsilon & \text{i.o.}(n) \end{cases}$$
>- a.b.f. (almost but finitely): 유한개를 제외한 모든 $n$에 대해, 즉 $\exists N$ s.t. $\forall n \ge N$
>- i.o. (infinitely often): 무한히 많은 $n$에 대해
>
>**하극한 (Limit Inferior)**
>$$
>\liminf_{n\to\infty} a_n := \underline{\lim_{n\to\infty}} a_n = \lim_{n\to\infty} \inf_{k\ge n} a_k
>$$
>
>**하극한의 다른 정의**  
>$$\liminf_{n\to\infty} a_n = \underline{l} \in \mathbb{R} \Leftrightarrow \forall \varepsilon > 0, \begin{cases} a_n > \underline{l} - \varepsilon & \text{a.b.f.}(n) \\ a_n < \underline{l} + \varepsilon & \text{i.o.}(n) \end{cases}$$
>
>#### 예제
>**$a_n = \begin{cases} 2^n & n = 1, 2, \ldots, 10 \\ (-1)^n & n \geq 11 \end{cases}$**
>
>값집합: $A = \{2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, -1, 1, -1, 1, \ldots\}$
>
>- $M_n = \sup_{k \geq n} a_k$:
>  - $n \le 10$일 때: $M_n = \sup\{a_n, a_{n+1}, \ldots, a_{10}, -1, 1, -1, 1, \ldots\} = 2^{10} = 1024$
>  - $n > 10$일 때: $M_n = \sup\{-1, 1, -1, 1, \ldots\} = 1$
>  - $\{M_n\} = \{1024, 1024, \ldots, 1024, 1, 1, 1, \ldots\}$ (단조감소)
>  - $\limsup a_n = \lim_{n\to\infty} M_n = 1$
>
>- $m_n = \inf_{k \geq n} a_k$:
>  - $n \le 10$일 때: $m_n = \inf\{2^n, 2^{n+1}, \ldots, 2^{10}, -1, 1, -1, 1, \ldots\} = -1$
>  - $n > 10$일 때: $m_n = \inf\{-1, 1, -1, 1, \ldots\} = -1$
>  - $\{m_n\} = \{-1, -1, -1, \ldots\}$ (단조증가)
>  - $\liminf a_n = \lim_{n\to\infty} m_n = -1$
>
>- 결론: $\liminf a_n = -1 < 1 = \limsup a_n$ (수렴하지 않음. 수열은 처음에는 기하급수적으로 증가했다가 $n > 10$부터 $-1$과 $1$을 진동)
>
>### 상극한과 하극한의 성질
>1. $\liminf a_n \le \limsup a_n$
>2. $\limsup a_n = -\liminf(-a_n)$ and $\liminf a_n = -\limsup(-a_n)$
>3. $\limsup_{n\to\infty} (a_n+b_n) \le \limsup_{n\to\infty} a_n + \limsup_{n\to\infty} b_n$  
>$\liminf_{n\to\infty} (a_n+b_n) \ge \liminf_{n\to\infty} a_n + \liminf_{n\to\infty} b_n$
>4. $a_n \le b_n$ a.b.f.(n)이면 $\liminf a_n \le \liminf b_n$, $\limsup a_n \le \limsup b_n$
>#### 예시
>**$a_n = n\left(-\frac{1}{2}\right)^n$**  
>- $\lim_{k\to\infty} (2k-1)\left(\frac{1}{2}\right)^{2k-1} = \lim_{k\to\infty} \frac{2k-1}{2^{2k-1}} = 0$
>- $\lim_{k\to\infty} 2k\left(\frac{1}{2}\right)^{2k} = \lim_{k\to\infty} \frac{2k}{4^k} = 0$  
>따라서 $\lim_{n\to\infty} a_n = 0$

### Thm. 4. [상극한의 특성화] *(Characterization of Limit Superior)*
유계실수열 $\{x_n\}$의 상극한이 $\alpha \in \mathbb{R}$이면 다음 두 성질이 성립한다.

1. 임의의 $\varepsilon > 0$에 대하여, $\exists N \in \mathbb{N}$ s.t. $\forall n \ge N$, $x_n < \alpha + \varepsilon$
  (즉, 유한개를 제외한 모든 $n$에 대해 $x_n < \alpha + \varepsilon$)

2. 임의의 $\varepsilon > 0$에 대하여, 무한히 많은 $n$에 대하여 $x_n > \alpha - \varepsilon$

역으로, 유계수열 $\{x_n\}$이 어떤 실수 $\alpha \in \mathbb{R}$에 대하여 위 두 성질을 만족하면, $\limsup_{n\to\infty} x_n = \alpha$이다.

#### 증명
**($\Rightarrow$ 방향)** $\limsup_{n\to\infty} x_n = \alpha$라고 하자.

정의에 의해 $M_n := \sup_{k \ge n} x_k$로 정의하면, $\{M_n\}$은 단조감소하고 $\lim_{n\to\infty} M_n = \alpha$이다.

**성질 1 증명:**  
임의의 $\varepsilon > 0$에 대해, $\lim_{n\to\infty} M_n = \alpha$이므로
$$
\exists N \in \mathbb{N} \text{ s.t. } \forall n \ge N, |M_n - \alpha| < \varepsilon
$$

$M_n = \sup_{k \ge n} x_k$이므로 $x_k \le M_n < \alpha + \varepsilon$이다.
따라서 모든 $n \ge N$에 대해 $x_n < \alpha + \varepsilon$이다.

**성질 2 증명:**  
임의의 $\varepsilon > 0$에 대해, $M_n = \sup_{k \ge n} x_k$의 정의에 의해
$$
M_n > \alpha - \varepsilon \text{ (왜냐하면 } M_n \to \alpha)
$$

따라서 상한의 정의에 의해
$$
\exists k \ge n \text{ s.t. } x_k > \alpha - \varepsilon
$$

이를 각 $n$에 대해 반복하면, $\alpha - \varepsilon$보다 큰 항이 무한히 많이 존재한다.

$(\Leftarrow$ 방향) 주어진 두 성질을 만족한다고 하자.

성질 1에 의해 $M := \inf_{n} M_n \le \alpha$이다.
(왜냐하면 $M_n \ge \alpha - \varepsilon$ for all $\varepsilon > 0$)

성질 2에 의해 $M \ge \alpha$이다.
(왜냐하면 $\alpha - \varepsilon$보다 큰 항이 무한히 많으므로 $M_n \ge \alpha - \varepsilon$)

따라서 $M = \alpha$이고, $\limsup_{n\to\infty} x_n = \alpha$이다.

### Thm. 5. [하극한의 특성화] *(Characterization of Limit Inferior)*
유계실수열 $\{x_n\}$의 하극한이 $\beta \in \mathbb{R}$이면 다음 두 성질이 성립한다.

1. 임의의 $\varepsilon > 0$에 대하여, $\exists N \in \mathbb{N}$ s.t. $\forall n \ge N$, $x_n > \beta - \varepsilon$
  (즉, 유한개를 제외한 모든 $n$에 대해 $x_n > \beta - \varepsilon$)

2. 임의의 $\varepsilon > 0$에 대하여, 무한히 많은 $n$에 대하여 $x_n < \beta + \varepsilon$

역으로, 유계수열 $\{x_n\}$이 어떤 실수 $\beta \in \mathbb{R}$에 대하여 위 두 성질을 만족하면, $\liminf_{n\to\infty} x_n = \beta$이다.

#### 증명
**($\Rightarrow$ 방향)** $\liminf_{n\to\infty} x_n = \beta$라고 하자.

정의에 의해 $m_n := \inf_{k \ge n} x_k$로 정의하면, $\{m_n\}$은 단조증가하고 $\lim_{n\to\infty} m_n = \beta$이다.

**성질 1 증명:**  
임의의 $\varepsilon > 0$에 대해, $\lim_{n\to\infty} m_n = \beta$이므로
$$
\exists N \in \mathbb{N} \text{ s.t. } \forall n \ge N, |m_n - \beta| < \varepsilon
$$

$m_n = \inf_{k \ge n} x_k$이므로 $x_k \ge m_n > \beta - \varepsilon$이다.
따라서 모든 $n \ge N$에 대해 $x_n > \beta - \varepsilon$이다.

**성질 2 증명:**  
임의의 $\varepsilon > 0$에 대해, $m_n = \inf_{k \ge n} x_k$의 정의에 의해
$$
m_n < \beta + \varepsilon \text{ (왜냐하면 } m_n \to \beta)
$$

따라서 하한의 정의에 의해
$$
\exists k \ge n \text{ s.t. } x_k < \beta + \varepsilon
$$

이를 각 $n$에 대해 반복하면, $\beta + \varepsilon$보다 작은 항이 무한히 많이 존재한다.

$(\Leftarrow$ 방향) 주어진 두 성질을 만족한다고 하자.

성질 1에 의해 $m := \sup_{n} m_n \ge \beta$이다.
(왜냐하면 $m_n \le \beta + \varepsilon$ for all $\varepsilon > 0$)

성질 2에 의해 $m \le \beta$이다.
(왜냐하면 $\beta + \varepsilon$보다 작은 항이 무한히 많으므로 $m_n \le \beta + \varepsilon$)

따라서 $m = \beta$이고, $\liminf_{n\to\infty} x_n = \beta$이다.


## (3) 코시수열 *(Cauchy Sequence)*
### Def. 1. [코시수열] *(Cauchy Sequence)*
$\forall\varepsilon>0,\ \exists N\in\mathbb N$ s.t. $\forall m, n \in \mathbb{N}$ with
$$
m\ge n\gt N,\ |a_m-a_n|<\varepsilon
$$
이면 $\{a_n\}$을 **코시수열**이라 한다.

### Thm. 1. [코시수열과 수렴성] *(Cauchy Criterion)*
$\{a_n\}$이 코시수열이면 $\{a_n\}$은 수렴한다.

#### 증명
$\{a_n\}$이 코시수열이라고 하자.

**Step 1.** $\{a_n\}$이 유계임을 보이자.  
$\varepsilon = 1$이라 하면, 코시수열의 정의에 의해
$$
\exists N \in \mathbb{N} \text{ s.t. } \forall m, n \ge N, |a_m - a_n| < 1
$$

특히 $n \ge N$일 때 $|a_n - a_N| < 1$이므로
$$
|a_n| \le |a_N| + 1
$$

따라서
$$
M = \max\{|a_1|, |a_2|, \ldots, |a_{N-1}|, |a_N| + 1\}
$$
라 하면, 모든 $n \in \mathbb{N}$에 대해 $|a_n| \le M$이므로 $\{a_n\}$은 유계이다.

**Step 2.** 수렴하는 부분수열이 존재함을 보이자.

$\{a_n\}$이 유계이므로 Bolzano–Weierstrass 정리에 의해
수렴하는 부분수열 $\{a_{n_k}\}$가 존재한다. $\lim_{k\to\infty} a_{n_k} = a$라 하자.

**Step 3.** $\lim_{n\to\infty} a_n = a$임을 보이자.

임의의 $\varepsilon > 0$에 대해, 코시수열의 정의에 의해
$$
\exists N_1 \in \mathbb{N} \text{ s.t. } \forall m, n \ge N_1, |a_m - a_n| < \frac{\varepsilon}{2}
$$

또한 $\{a_{n_k}\}$가 $a$로 수렴하므로
$$
\exists K \in \mathbb{N} \text{ s.t. } \forall k \ge K, |a_{n_k} - a| < \frac{\varepsilon}{2}
$$

$n_k \ge N_1$인 $K$를 택하면, $n \ge N_1$일 때
$$
|a_n - a| \le |a_n - a_{n_k}| + |a_{n_k} - a| < \frac{\varepsilon}{2} + \frac{\varepsilon}{2} = \varepsilon
$$

따라서 $\lim_{n\to\infty} a_n = a$이다.

### Thm. 2. 수렴하는 수열은 코시수열이다. *(A Convergent Sequence is a Cauchy Sequence)*
$\lim_{n\to\infty} a_n = a$이면 $\{a_n\}$은 코시수열이다.

#### 증명
$\lim_{n\to\infty} a_n = a$라고 하자.

임의의 $\varepsilon > 0$에 대해, 수열의 극한 정의에 의해
$$
\exists N \in \mathbb{N} \text{ s.t. } \forall n \ge N, |a_n - a| < \frac{\varepsilon}{2}
$$

$m, n \ge N$일 때
$$
|a_m - a_n| = |a_m - a + a - a_n| \le |a_m - a| + |a - a_n| < \frac{\varepsilon}{2} + \frac{\varepsilon}{2} = \varepsilon
$$

따라서 $\{a_n\}$은 코시수열이다.

### 보조정리. 코시수열은 유계이다. *(A Cauchy Sequence is Bounded)*
$\{a_n\}$이 코시수열이면 $\{a_n\}$은 유계이다.

#### 증명
$\{a_n\}$이 코시수열이라고 하자.

$\varepsilon = 1$이라 하면, 코시수열의 정의에 의해
$$
\exists N \in \mathbb{N} \text{ s.t. } \forall m, n \ge N, |a_m - a_n| < 1
$$

특히 $n \ge N$일 때 $|a_n - a_N| < 1$이므로
$$
|a_n| \le |a_N| + 1
$$

따라서
$$
M = \max\{|a_1|, |a_2|, \ldots, |a_{N-1}|, |a_N| + 1\}
$$
라 하면, 모든 $n \in \mathbb{N}$에 대해 $|a_n| \le M$이므로 $\{a_n\}$은 유계이다.

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

### 정리 부분합수열이 유계인 양수항 급수는 수렴한다.
증명: $\{a_n\}$이 양수항 수열이고, 부분합수열 $\{S_n\}$이 유계라고 하자.
$\{S_n\}$은 단조증가수열이므로 단조수렴정리에 의해 수렴한다.

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

### Thm. 4. [상극한의 특성화] *(Characterization of Limit Superior)*
유계실수열 $\{x_n\}$의 상극한이 $\alpha \in \mathbb{R}$이면 다음 두 성질이 성립한다.

1. 임의의 $\varepsilon > 0$에 대하여, $\exists N \in \mathbb{N}$ s.t. $\forall n \ge N$, $x_n < \alpha + \varepsilon$
  (즉, 유한개를 제외한 모든 $n$에 대해 $x_n < \alpha + \varepsilon$)

2. 임의의 $\varepsilon > 0$에 대하여, 무한히 많은 $n$에 대하여 $x_n > \alpha - \varepsilon$

역으로, 유계수열 $\{x_n\}$이 어떤 실수 $\alpha \in \mathbb{R}$에 대하여 위 두 성질을 만족하면, $\limsup_{n\to\infty} x_n = \alpha$이다.

#### 증명
**($\Rightarrow$ 방향)** $\limsup_{n\to\infty} x_n = \alpha$라고 하자.

정의에 의해 $M_n := \sup_{k \ge n} x_k$로 정의하면, $\{M_n\}$은 단조감소하고 $\lim_{n\to\infty} M_n = \alpha$이다.

**성질 1 증명:**  
임의의 $\varepsilon > 0$에 대해, $\lim_{n\to\infty} M_n = \alpha$이므로
$$
\exists N \in \mathbb{N} \text{ s.t. } \forall n \ge N, |M_n - \alpha| < \varepsilon
$$

$M_n = \sup_{k \ge n} x_k$이므로 $x_k \le M_n < \alpha + \varepsilon$이다.
따라서 모든 $n \ge N$에 대해 $x_n < \alpha + \varepsilon$이다.

**성질 2 증명:**  
임의의 $\varepsilon > 0$에 대해, $M_n = \sup_{k \ge n} x_k$의 정의에 의해
$$
M_n > \alpha - \varepsilon \text{ (왜냐하면 } M_n \to \alpha)
$$

따라서 상한의 정의에 의해
$$
\exists k \ge n \text{ s.t. } x_k > \alpha - \varepsilon
$$

이를 각 $n$에 대해 반복하면, $\alpha - \varepsilon$보다 큰 항이 무한히 많이 존재한다.

$(\Leftarrow$ 방향) 주어진 두 성질을 만족한다고 하자.

성질 1에 의해 $M := \inf_{n} M_n \le \alpha$이다.
(왜냐하면 $M_n \ge \alpha - \varepsilon$ for all $\varepsilon > 0$)

성질 2에 의해 $M \ge \alpha$이다.
(왜냐하면 $\alpha - \varepsilon$보다 큰 항이 무한히 많으므로 $M_n \ge \alpha - \varepsilon$)

따라서 $M = \alpha$이고, $\limsup_{n\to\infty} x_n = \alpha$이다.

### Thm. 5. [하극한의 특성화] *(Characterization of Limit Inferior)*
유계실수열 $\{x_n\}$의 하극한이 $\beta \in \mathbb{R}$이면 다음 두 성질이 성립한다.
1. 임의의 $\varepsilon > 0$에 대하여, $\exists N \in \mathbb{N}$ s.t. $\forall n \ge N$, $x_n > \beta - \varepsilon$
  (즉, 유한개를 제외한 모든 $n$에 대해 $x_n > \beta - \varepsilon$)
2. 임의의 $\varepsilon > 0$에 대하여, 무한히 많은 $n$에 대하여 $x_n < \beta + \varepsilon$
역으로, 유계수열 $\{x_n\}$이 어떤 실수 $\beta \in \mathbb{R}$에 대하여 위 두 성질을 만족하면, $\liminf_{n\to\infty} x_n = \beta$이다.



# [연습문제]
1. 다음 명제의 참·거짓을 판별하고 설명하시오.  
(1) $\{a_n\}$이 유계 $\Rightarrow \lim a_n$ 존재  
(2) $\lim(a_n+b_n)=\infty\Rightarrow \lim a_n=\infty$ 또는 $\lim b_n=\infty$  
  - 거짓
  - $$
    a_n =
    \begin{cases}
    n, & n \text{이 홀수일 때} \\
    0, & n \text{이 짝수일 때}
    \end{cases}
    \\
    b_n =
    \begin{cases}
    0, & n \text{이 홀수일 때} \\
    n, & n \text{이 짝수일 때}
    \end{cases}
    $$

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
