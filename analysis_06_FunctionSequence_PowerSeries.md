함수열·멱급수
# 1. 정의

## Def. 1. [함수열과 함수급수] *(Sequence of Functions and Series of Functions)*
$\varphi:\phi \neq D\subset\mathbb R$이고 모든 $n\in\mathbb N$에 대하여
$$
f_n:D\to\mathbb R
$$
일 때 $\{f_n\}$을 $D$에서의 **함수열** *(sequence of functions)* 이라 한다.

또한 $\{f_n\}$이 함수열일 때
$$
\sum_{n=1}^\infty f_n
$$
을 **함수급수** *(series of functions)* 라 한다.

- 이게 수렴하는지 확인하는게 핵심
- fn이 다항함수면 멱급수

## Def. 2. [멱급수] *(Power Series)*
실수 $c$와 수열 ${a_n}$에 대하여 함수열 $\{f_n\}$이
$$
f_n(x)=a_n(x-c)^n
$$
과 같이 표현될 때, 함수급수
$$
\sum_{n=0}^\infty f_n = \sum_{n=0}^\infty a_n(x-c)^n
$$
을 **멱급수** *(power series)* 라 한다.
- 멱급수로 표현가능한 함수를 해석함수라 한다
## Def. 3. [해석함수] *(Analytic Function)*
어떤 $\delta>0$에 대하여 $(c-\delta,c+\delta)$에서 함수 $f$가 멱급수
$$
f(x)=\sum_{n=0}^\infty a_n(x-c)^n
$$
로 표현될 수 있으면, $f$는 $x=c$에서 **해석적** *(analytic)* 이라 한다.

- 함수 $f$가 열린구간 $I$의 모든 점에서 해석적이면  
멱급수의 미분, 적분 등의 성질을 이용하여 함수의 성질을 분석할 수 있다


# 2. 점별수렴과 균등수렴 *(Pointwise and Uniform Convergence)*
## (1) 함수열의 수렴
- 함수열과 극한함수 간 성질이 항상 보존되지 않는 문제가 있었음
   - 각 함수는 연속인데 극한함수는 불연속
   - 각 함수는 미분가능인데 극한함수는 미분불가능
- 이를 분석하여 어떤 조건에서 성질이 보존되는지 바이어슈트라스가 밝혀냈고, 여기서 점별수렴과 균등수렴이 핵심 개념임
### Def. [점별수렴과 균등수렴] *(Pointwise Convergence and Uniform Convergence)*
$\{f_n\}$과 $f$가 $D$에서 정의된 함수열과 함수라 하자.

① 임의의 $x\in D$와 $\varepsilon>0$에 대하여
$$
\exists N\in\mathbb N\ \text{s.t.}\ n\ge N \Rightarrow |f_n(x)-f(x)|<\varepsilon
$$
이면 $\{f_n\}$은 $D$에서 $f$로 **점별수렴** *(pointwise convergence)* 한다.

② 임의의 $\varepsilon>0$에 대하여
$$
\exists N\in\mathbb N\ \text{s.t.}\ \forall x\in D,\ n\ge N \Rightarrow |f_n(x)|<\varepsilon
$$
이면 $\{f_n\}$은 $D$에서 $f$로 **균등수렴** *(uniform convergence)* 한다.
   - ①은 각 $x$마다 서로 다른 $N$을 선택 가능
   - ②는 ①과 달리 $N$이 $x$에 무관하게 존재

**점별수렴 예시:**
   - $f_n(x) = x^n$을 $[0,1]$에서 생각하면
      - $x \in [0,1)$일 때: $\lim_{n\to\infty} x^n = 0$
      - $x = 1$일 때: $\lim_{n\to\infty} 1^n = 1$
      - 따라서 $f(x) = \begin{cases} 0 & (x \in [0,1)) \\ 1 & (x=1) \end{cases}$로 점별수렴
      - 각 $f_n$은 연속이지만 극한함수 $f$는 불연속

**균등수렴 예시:**
   - $f_n(x) = \frac{x^n}{n}$을 $[0,1]$에서 생각하면
      - 모든 $x \in [0,1]$에 대해 $|f_n(x) - 0| = \frac{x^n}{n} \le \frac{1}{n}$
      - $\sup_{x \in [0,1]} |f_n(x)| = \frac{1}{n} \to 0$
      - 따라서 $f(x) = 0$로 균등수렴

### Thm.
$\{f_n\}$이 $D$에서 균등수렴하면 점별수렴한다.

- 증명

균등수렴하면 정의상 $N$이 $x$에 무관하게 존재한다.
  
  임의의 $\varepsilon>0$에 대하여, 균등수렴의 정의에 의해
  $$
  \exists N\in\mathbb N\ \text{s.t.}\ \forall x\in D,\ n\ge N \Rightarrow |f_n(x)-f(x)|<\varepsilon
  $$
  
  따라서 임의의 $x\in D$를 고정하면, 같은 $N$에 대해
  $$
  n\ge N \Rightarrow |f_n(x)-f(x)|<\varepsilon
  $$
  
  이므로 점별수렴의 정의를 만족한다. □

## (2) 함수급수의 수렴
### Thm. 1. [코시판정법] *(Cauchy Criterion for Series of Functions)*

$f_n:D\to\mathbb R$라 할 때, 다음을 만족하면
$\sum_{n=1}^\infty f_n$은 $D$에서 균등수렴한다.  
$$
\forall\varepsilon>0,\ \exists N\in\mathbb N\ \text{s.t.}\
\forall m>n\ge N,\ \forall x\in D,\\
\left|\sum_{k=n+1}^m f_k(x)\right|<\varepsilon
$$

### Thm. 2. [바이어슈트라스 판정법] *(Weierstrass M-test)*
각 $n\in\mathbb N$에 대해 $f_n:D\to\mathbb R$라 하자.
적당한 양의 상수 $M_n>0$가 존재하여
$$
|f_n(x)|\le M_n\quad(\forall x\in D)
$$
이고
$$
\sum_{n=1}^\infty M_n<\infty
$$
이면
$$
\sum_{n=1}^\infty f_n
$$
은 $D$에서 균등수렴한다.

#### 증명
임의의 $\varepsilon>0$에 대하여, $\sum_{n=1}^\infty M_n$이 수렴하므로 코시판정법에 의해
$$
\exists N\in\mathbb N\ \text{s.t.}\ \forall m>n\ge N,\quad \sum_{k=n+1}^m M_k<\varepsilon
$$

이때 모든 $x\in D$에 대해 (삼각부등식 활용)
$$
\left|\sum_{k=n+1}^m f_k(x)\right|
\le\sum_{k=n+1}^m |f_k(x)|
\le\sum_{k=n+1}^m M_k
<\varepsilon
$$

따라서 Thm. 1 (코시판정법)에 의해 $\sum_{n=1}^\infty f_n$은 $D$에서 균등수렴한다. □

### 예시: 균등수렴 판정
함수급수 $\sum_{n=1}^\infty \frac{\sin nx}{n^2}$가 $\mathbb{R}$에서 균등수렴하는지 확인하자.

**풀이:**
각 $x \in \mathbb{R}$에 대해
$$
\left|\frac{\sin nx}{n^2}\right| \le \frac{1}{n^2}
$$

$M_n = \frac{1}{n^2}$로 두면
$$
\sum_{n=1}^\infty M_n = \sum_{n=1}^\infty \frac{1}{n^2} < \infty
$$

따라서 바이어슈트라스 판정법(Thm. 2)에 의해 주어진 함수급수는 $\mathbb{R}$에서 균등수렴한다.

**반례:**
함수급수 $\sum_{n=1}^\infty \frac{x^n}{n}$을 $[0,1)$에서 생각하자.

이 급수는 각 $x \in [0,1)$에서 점별수렴하지만, $x \to 1^-$일 때
$$
\sup_{x \in [0,1)} \left|\sum_{k=n+1}^m \frac{x^k}{k}\right|
$$
가 0으로 수렴하지 않으므로 $[0,1)$에서 균등수렴하지 않는다.

# 3. 멱급수 *(Power Series)*
## (1) 멱급수의 수렴
### Thm. 1. [근판정법] *(Root Test)*
모든 $n\in\mathbb N$에 대해 $a_n\ge0$이고
$$
\lim_{n\to\infty}\sqrt[n]{a_n}=M
$$
일 때,

① $M<1$이면 $\sum_{n=1}^\infty a_n$은 수렴한다.  
② $M>1$이면 $\sum_{n=1}^\infty a_n$은 발산한다.  

#### 증명
① $M < 1$일 때: $\varepsilon > 0$을 $M + \varepsilon < 1$이 되도록 선택하면, 정의에 의해
$$
\exists N \in \mathbb{N} \text{ s.t. } n \ge N \Rightarrow \sqrt[n]{a_n} < M + \varepsilon
$$
따라서 $n \ge N$일 때
$$
a_n < (M + \varepsilon)^n
$$
$\sum_{n=1}^\infty (M + \varepsilon)^n$은 $M + \varepsilon < 1$이므로 수렴하는 등비급수이다.  
비교판정법에 의해 $\sum_{n=1}^\infty a_n$은 수렴한다.

② $M > 1$일 때: $\varepsilon > 0$을 $M - \varepsilon > 1$이 되도록 선택하면, 정의에 의해 무한히 많은 $n$에 대해
$$
\sqrt[n]{a_n} > M - \varepsilon > 1
$$
따라서 $a_n > 1$이므로 $\lim_{n \to \infty} a_n \ne 0$이다.  
따라서 $\sum_{n=1}^\infty a_n$은 발산한다. □

>(참고)  
>**비교판정법** *(Comparison Test)* 은 **급수의 수렴·발산을 이미 알고 있는 급수와의 항별 비교로 판단하는 방법**이다.  
>$a_n,b_n \ge 0$인 수열에 대하여, 충분히 큰 $n$에 대해
>$$
>0 \le a_n \le b_n
>$$
>라고 하자.
>1. $\displaystyle\sum_{n=1}^\infty b_n$이 수렴하면
>   $\displaystyle\sum_{n=1}^\infty a_n$도 수렴한다.
>
>2. $\displaystyle\sum_{n=1}^\infty a_n$이 발산하면
>   $\displaystyle\sum_{n=1}^\infty b_n$도 발산한다.
>
### Cor. 중요: [멱급수의 수렴반경] *(Radius of Convergence)*
멱급수
$$
\sum_{n=0}^\infty a_n(x-c)^n
$$
에 대하여
$$
\alpha=\limsup_{n\to\infty}\sqrt[n]{|a_n|}
$$
이면 수렴반경 $R=\frac1\alpha$이다.

* $|x-c|<R$에서 절대수렴
* $|x-c|>R$에서 발산
* 멱급수의 수렴반지름 $R$을 구한 뒤에는
반드시 경계점 $|x-c|=R$에서의 수렴 여부를
각 점마다 별도로 판정해야 한다.

$\alpha=0$이면 $R=\infty$,
$\alpha=\infty$이면 $R=0$으로 간주한다.

#### 예시: 멱급수의 수렴반경 계산
**예제 1.** 멱급수 $\displaystyle\sum_{n=1}^\infty \frac{x^n}{n}$의 수렴반경을 구하자.

**풀이:**
$$
a_n = \frac{1}{n},\quad \sqrt[n]{|a_n|} = \frac{1}{n^{1/n}}
$$

$\lim_{n\to\infty} n^{1/n} = 1$이므로
$$
\alpha = \limsup_{n\to\infty} \sqrt[n]{|a_n|} = \lim_{n\to\infty} \frac{1}{n^{1/n}} = 1
$$

따라서 수렴반경은
$$
R = \frac{1}{\alpha} = 1
$$

**수렴구간 판정:**
- $|x| < 1$: 절대수렴
- $x = 1$: $\sum_{n=1}^\infty \frac{1}{n}$은 조화급수로 발산
- $x = -1$: $\sum_{n=1}^\infty \frac{(-1)^n}{n}$은 교대급수 판정법에 의해 수렴

따라서 수렴구간은 $[-1, 1)$이다.

### Def. [수렴반지름과 수렴구간] *(Radius and Interval of Convergence)*
$R$을 멱급수의 **수렴반지름**이라 하고,
멱급수가 수렴하는 점들의 전체 집합을 **수렴구간**이라 한다.

### Thm. 1. [수렴반지름과 균등수렴] *(Uniform Convergence inside Radius)*
멱급수의 수렴반지름이 $R$이고 $0<r<R$이면
$$
\sum_{n=0}^\infty a_n(x-c)^n
$$
은 $[c-r,c+r]$에서 균등수렴한다.

#### 증명
$r < R$이므로 어떤 $r' \in (r, R)$를 선택하자.

$|x - c| \le r < r' < R$인 모든 $x$에 대해
$$
|a_n(x-c)^n| \le |a_n|r'^n
$$

$r' < R = \frac{1}{\alpha}$이므로 $\alpha r' < 1$이다.

따라서 $\limsup_{n\to\infty}\sqrt[n]{|a_n|r'^n} = r'\alpha < 1$이므로

근판정법(Thm. 1)에 의해 $\sum_{n=0}^\infty |a_n|r'^n$은 수렴한다.

$M_n = |a_n|r'^n$으로 두면 $\sum_{n=0}^\infty M_n < \infty$이고

모든 $x \in [c-r, c+r]$에 대해 $|a_n(x-c)^n| \le M_n$이므로

바이어슈트라스 판정법(Thm. 2)에 의해

$\sum_{n=0}^\infty a_n(x-c)^n$은 $[c-r, c+r]$에서 균등수렴한다. □

## (2) 멱급수의 연속
### Thm. 1. [함수열의 연속] *(Continuity under Uniform Convergence)*
구간 $I$에서 연속인 함수열 $\{f_n\}$이 $I$에서 $f$로 균등수렴하면, $f$는 $I$에서 연속이다.

#### 증명
임의의 $x_0 \in I$를 고정하자. 주어진 $\varepsilon > 0$에 대하여

균등수렴의 정의에 의해
$$
\exists N \in \mathbb{N} \text{ s.t. } n \ge N \Rightarrow \sup_{x \in I}|f_n(x) - f(x)| < \frac{\varepsilon}{3}
$$

$f_N$이 $x_0$에서 연속이므로
$$
\exists \delta > 0 \text{ s.t. } |x - x_0| < \delta \Rightarrow |f_N(x) - f_N(x_0)| < \frac{\varepsilon}{3}
$$

따라서 $|x - x_0| < \delta$일 때
$$
\begin{align*}
|f(x) - f(x_0)| &\le |f(x) - f_N(x)| + |f_N(x) - f_N(x_0)| + |f_N(x_0) - f(x_0)|\\
&< \frac{\varepsilon}{3} + \frac{\varepsilon}{3} + \frac{\varepsilon}{3} = \varepsilon
\end{align*}
$$

그러므로 $f$는 $x_0$에서 연속이다. $x_0$는 임의로 선택했으므로 $f$는 $I$에서 연속이다. □

### Cor. [함수급수의 연속] *(Continuity of Series of Functions)*
$I$에서 연속인 함수열 $\{f_n\}$에 대해
$$
\sum_{n=1}^\infty f_n
$$
이 $I$에서 $f$로 균등수렴하면 $f$도 $I$에서 연속이다.
#### 증명
부분합 $S_n(x) = \sum_{k=1}^n f_k(x)$라 하면, $S_n$은 연속함수들의 유한합이므로 $I$에서 연속이다.

$\sum_{n=1}^\infty f_n$이 $I$에서 $f$로 균등수렴하므로, 함수열 $\{S_n\}$은 $I$에서 $f$로 균등수렴한다.

따라서 Thm. 1에 의해 $f$는 $I$에서 연속이다. □

### Lemma. [아벨의 공식] *(Abel's Summation Formula)*
수열 $\{a_k\},\{b_k\}$와 자연수 $n,m$ $(n>m)$에 대하여
$$
\sum_{k=m+1}^n a_k b_k 
= a_n\sum_{k=m+1}^n b_k
- \sum_{j=m+1}^{n-1}(a_{j+1}-a_j)\sum_{k=m+1}^j b_k
$$

#### 증명
$A_j = \sum_{k=m+1}^j b_k$로 정의하자. 그러면 $b_k = A_k - A_{k-1}$ (단, $A_m = 0$)이다.

$$
\begin{align*}
\sum_{k=m+1}^n a_k b_k 
&= \sum_{k=m+1}^n a_k(A_k - A_{k-1})\\
&= \sum_{k=m+1}^n a_k A_k - \sum_{k=m+1}^n a_k A_{k-1}\\
&= \sum_{k=m+1}^n a_k A_k - \sum_{j=m}^{n-1} a_{j+1} A_j\\
&= a_n A_n + \sum_{k=m+1}^{n-1} a_k A_k - \sum_{j=m+1}^{n-1} a_{j+1} A_j \quad \text{(Am=0이므로)}\\
&= a_n A_n + \sum_{j=m+1}^{n-1}(a_j - a_{j+1})A_j\\
&= a_n\sum_{k=m+1}^n b_k - \sum_{j=m+1}^{n-1}(a_{j+1}-a_j)\sum_{k=m+1}^j b_k
\end{align*}
$$

따라서 아벨의 공식이 성립한다. □


### Thm. 2. [아벨정리] *(Abel’s Theorem)*
멱급수
$$
\sum_{n=0}^\infty a_n(x-c)^n
$$
이 수렴반지름이 $R$이고, $x=c+R$에서 수렴하면, 임의의 $r$ $(0<r<R)$에 대하여 이 멱급수는 $[c−r,c+r]$ 에서 균등수렴한다.

- 결론은 Thm. 1. [수렴반지름과 균등수렴]과 같은데
  - Thm.1이 균등수렴을 만들어 주는 정리라면
  - 아벨정리는 끝점에서의 점별수렴이라는 추가 정보가 주어지더라도, 수렴반지름 내부의 닫힌 부분구간에서의 균등수렴 결론이 여전히 성립함을 보장하는 정리이다.

#### 증명
임의의 $\varepsilon > 0$을 택하자.

$\sum_{n=0}^\infty a_n R^n$이 수렴하므로, 코시 판정법에 의해
$$
\exists N \in \mathbb{N} \text{ s.t. } \forall m > n \ge N, \quad \left|\sum_{k=n+1}^m a_k R^k\right| < \varepsilon
$$

이제 $x \in [c+r, c+R]$를 택하자. 아벨의 공식(Lemma)을 이용하여 $b_k = a_k R^k$, $a_k = \left(\frac{x-c}{R}\right)^k$로 두면

$$
\begin{align*}
\left|\sum_{k=n+1}^m a_k(x-c)^k\right| 
&= \left|\sum_{k=n+1}^m \left(\frac{x-c}{R}\right)^k \cdot a_k R^k\right|\\
&= \left|\left(\frac{x-c}{R}\right)^m \sum_{k=n+1}^m a_k R^k - \sum_{j=n+1}^{m-1}\left[\left(\frac{x-c}{R}\right)^{j+1} - \left(\frac{x-c}{R}\right)^j\right]\sum_{k=n+1}^j a_k R^k\right|
\end{align*}
$$

$r \le x - c \le R$이므로 $\frac{x-c}{R} \le 1$이고, $\left|\sum_{k=n+1}^j a_k R^k\right| < \varepsilon$이므로

$$
\begin{align*}
\left|\sum_{k=n+1}^m a_k(x-c)^k\right|
&\le \left(\frac{x-c}{R}\right)^m \varepsilon + \sum_{j=n+1}^{m-1}\left[\left(\frac{x-c}{R}\right)^j - \left(\frac{x-c}{R}\right)^{j+1}\right]\varepsilon\\
&= \varepsilon\left[\left(\frac{x-c}{R}\right)^m + \sum_{j=n+1}^{m-1}\left(\frac{x-c}{R}\right)^j - \sum_{j=n+1}^{m-1}\left(\frac{x-c}{R}\right)^{j+1}\right]\\
&= \varepsilon\left[\left(\frac{x-c}{R}\right)^m + \left(\frac{x-c}{R}\right)^{n+1} - \left(\frac{x-c}{R}\right)^m\right]\\
&= \varepsilon\left(\frac{x-c}{R}\right)^{n+1}\\
&\le \varepsilon
\end{align*}
$$

따라서 코시 판정법(Thm. 1)에 의해 $\sum_{n=0}^\infty a_n(x-c)^n$은 $[c+r, c+R]$에서 균등수렴한다.

마찬가지 방법으로 $[c-R, c-r]$에서도 균등수렴함을 보일 수 있다. □

### 중요! Thm. 3. [멱급수의 연속] *(Continuity of Power Series)*
멱급수
$$
f(x)=\sum_{n=0}^\infty a_n(x-c)^n
$$
는 수렴구간에서 연속이다.

#### 증명
임의의 $x_0 \in (c-R, c+R)$를 택하자.

$|x_0 - c| < r < R$인 $r$을 선택하면, Thm. 1에 의해

$\sum_{n=0}^\infty a_n(x-c)^n$은 $[c-r, c+r]$에서 균등수렴한다.

각 항 $f_n(x) = a_n(x-c)^n$은 연속함수이므로,

Cor. [함수급수의 연속]에 의해

$f(x) = \sum_{n=0}^\infty a_n(x-c)^n$은 $[c-r, c+r]$에서 연속이다.

특히 $x_0 \in [c-r, c+r]$이므로 $f$는 $x_0$에서 연속이다.

$x_0$는 $(c-R, c+R)$의 임의의 점이므로 $f$는 수렴구간에서 연속이다. □

## (3) 멱급수의 미분 *(Differentiation of Power Series)*
### Thm. 1. [함수열의 미분] *(Differentiation under Uniform Convergence)*
유계구간 $I$에서 함수열 $\{f_n\}$이 다음을 만족하면

* 어떤 $x_0\in I$에서 $\{f_n(x_0)\}$가 수렴
  - 즉, 이 구간에서 점별수렴
* $\{f_n\}$이 $I$에서 미분가능
* $\{f_n'\}$이 $I$에서 균등수렴
  - 점별수렴 정도로는 f도 미분가능하다는 보장이 안됨

$f_n\to f$라 할 때 $f$는 $I$에서 미분가능하며
$$
f'(x)=\lim_{n\to\infty}f_n'(x)
$$

#### 증명
임의의 $x_0 \in I$를 고정하고, $h \ne 0$이며 $x_0 + h \in I$라 하자.

$\{f_n(x_0)\}$가 수렴하고 $\{f_n'\}$이 $I$에서 균등수렴하므로,

평균값 정리에 의해 각 $n$에 대해 $x_0$와 $x_0+h$ 사이의 어떤 $\xi_n$이 존재하여
$$
f_n(x_0+h) - f_n(x_0) = f_n'(\xi_n)h
$$

따라서
$$
\frac{f_n(x_0+h) - f_n(x_0)}{h} = f_n'(\xi_n)
$$

$\{f_n'\}$이 $I$에서 어떤 함수 $g$로 균등수렴하므로, 임의의 $\varepsilon > 0$에 대하여
$$
\exists N \in \mathbb{N} \text{ s.t. } \forall n \ge N, \forall x \in I, \quad |f_n'(x) - g(x)| < \varepsilon
$$

$n, m \ge N$일 때
$$
\left|\frac{f_n(x_0+h) - f_n(x_0)}{h} - \frac{f_m(x_0+h) - f_m(x_0)}{h}\right| = |f_n'(\xi_n) - f_m'(\xi_m)|
$$

$$
\le |f_n'(\xi_n) - g(\xi_n)| + |g(\xi_n) - g(\xi_m)| + |g(\xi_m) - f_m'(\xi_m)|
$$

균등수렴에 의해 첫 번째와 세 번째 항은 각각 $\varepsilon$보다 작고, $g$는 Thm. 1 [함수열의 연속]에 의해 연속이므로 $|g(\xi_n) - g(\xi_m)| \to 0$이다.

따라서 $\left\{\frac{f_n(x_0+h) - f_n(x_0)}{h}\right\}$는 코시 수열이므로 수렴한다.

$f(x) = \lim_{n\to\infty} f_n(x)$로 정의하면
$$
\frac{f(x_0+h) - f(x_0)}{h} = \lim_{n\to\infty}\frac{f_n(x_0+h) - f_n(x_0)}{h} = \lim_{n\to\infty}f_n'(\xi_n) = g(x_0)
$$

$h \to 0$일 때 극한을 취하면 $f'(x_0) = g(x_0) = \lim_{n\to\infty}f_n'(x_0)$이다. □

### Cor. [함수열급수의 미분] *(Term-by-term Differentiation)*
다음이 성립하면
$$
\sum_{n=1}^\infty f_n
$$
은 유계구간 $I$에서 미분가능하다.

① $\sum f_n(x_0)$가 수렴  
② $\sum f_n'$이 $I$에서 균등수렴  

이때
$$
\left(\sum_{n=1}^\infty f_n\right)'
=\sum_{n=1}^\infty f_n'
$$

### Lemma. *(Power series and its derivative have the same radius of convergence)*

멱급수
$$
\sum_{n=0}^{\infty} a_n (x-c)^n
$$
의 수렴반지름이 $R$이면, 그 **항별미분으로 얻어지는 멱급수**
$$
\sum_{n=1}^{\infty} n a_n (x-c)^{,n-1}
$$
의 수렴반지름도 역시 $R$이다.

#### 증명
원래 멱급수의 수렴반지름을 $R$이라 하면
$$
\alpha = \limsup_{n\to\infty}\sqrt[n]{|a_n|} = \frac{1}{R}
$$

미분한 멱급수 $\sum_{n=1}^{\infty} n a_n (x-c)^{n-1}$의 수렴반지름을 $R'$이라 하면
$$
\frac{1}{R'} = \limsup_{n\to\infty}\sqrt[n]{|n a_n|} 
= \limsup_{n\to\infty}\sqrt[n]{n}\cdot\sqrt[n]{|a_n|}
$$

$\lim_{n\to\infty}\sqrt[n]{n} = 1$이므로
$$
\frac{1}{R'} = 1 \cdot \limsup_{n\to\infty}\sqrt[n]{|a_n|} = \alpha = \frac{1}{R}
$$

따라서 $R' = R$이다. □

### 중요! Thm. 3. [멱급수의 미분] *(Differentiation of Power Series)*
멱급수
$$
f(x)=\sum_{n=0}^\infty a_n(x-c)^n
$$
의 수렴반지름이 $R$이면 $(c-R,c+R)$에서 미분가능하며
$$
f'(x)=\sum_{n=1}^\infty n a_n(x-c)^{n-1}
$$

#### 증명
임의의 $x \in (c-R, c+R)$를 택하자.

$|x-c| < r < R$인 $r$을 선택하면, 

미분한 멱급수 $\sum_{n=1}^{\infty} n a_n (x-c)^{n-1}$은 Lemma에 의해 수렴반지름이 $R$이므로

Thm. 1 [수렴반지름과 균등수렴]에 의해 $[c-r, c+r]$에서 균등수렴한다.

또한 원래 멱급수는 $x$에서 수렴하므로 Cor. [함수열급수의 미분]의 조건을 만족한다.

따라서 $f$는 $x$에서 미분가능하며
$$
f'(x) = \sum_{n=1}^{\infty} n a_n (x-c)^{n-1}
$$

$x$는 $(c-R, c+R)$의 임의의 점이므로 결론이 성립한다. □

## (4) 멱급수의 적분 *(Integration of Power Series)*
### Thm. 1. [균등수렴과 적분] *(Integration under Uniform Convergence)*
$\{f_n\}$이 $[a,b]$에서 $f$로 균등수렴하고
$f_n\in\mathcal R[a,b]$이면 $f\in\mathcal R[a,b]$이며
$$
\lim_{n\to\infty}\int_a^b f_n=\int_a^b f
$$


#### 증명
임의의 $\varepsilon > 0$을 택하자.

$\{f_n\}$이 $[a,b]$에서 $f$로 균등수렴하므로
$$
\exists N \in \mathbb{N} \text{ s.t. } n \ge N \Rightarrow \sup_{x \in [a,b]}|f_n(x) - f(x)| < \frac{\varepsilon}{3(b-a)}
$$

$f_N \in \mathcal{R}[a,b]$이므로 리만의 판정법에 의해, 어떤 분할 $P = \{x_0, x_1, \ldots, x_m\}$이 존재하여
$$
U(f_N, P) - L(f_N, P) < \frac{\varepsilon}{3}
$$

각 소구간 $[x_{i-1}, x_i]$에서
$$
M_i(f) - m_i(f) \le [M_i(f_N) + \frac{\varepsilon}{3(b-a)}] - [m_i(f_N) - \frac{\varepsilon}{3(b-a)}]
= M_i(f_N) - m_i(f_N) + \frac{2\varepsilon}{3(b-a)}
$$

따라서
$$
\begin{align*}
U(f, P) - L(f, P) &= \sum_{i=1}^m [M_i(f) - m_i(f)]\Delta x_i\\
&\le \sum_{i=1}^m [M_i(f_N) - m_i(f_N)]\Delta x_i + \frac{2\varepsilon}{3(b-a)}\sum_{i=1}^m \Delta x_i\\
&= U(f_N, P) - L(f_N, P) + \frac{2\varepsilon}{3}\\
&< \frac{\varepsilon}{3} + \frac{2\varepsilon}{3} = \varepsilon
\end{align*}
$$

따라서 $f \in \mathcal{R}[a,b]$이다.

이제 $n \ge N$일 때
$$
\left|\int_a^b f_n - \int_a^b f\right| = \left|\int_a^b (f_n - f)\right| \le \int_a^b |f_n - f| < \frac{\varepsilon}{3(b-a)} \cdot (b-a) = \frac{\varepsilon}{3} < \varepsilon
$$

따라서 $\displaystyle\lim_{n\to\infty}\int_a^b f_n = \int_a^b f$이다. □


### Thm. 2. [항별적분] (Term-by-term Integration)

$f_n\in\mathcal R[a,b]$인 함수열 $\{f_n\}$에 대하여
부분합
$$
S_N(x)=\sum_{n=1}^N f_n(x)
$$
이 $[a,b]$에서 어떤 함수 $f$로 균등수렴한다고 하자.
즉,
$$
f(x)=\sum_{n=1}^\infty f_n(x)
$$
이다.

그러면 $f\in\mathcal R[a,b]$이고
$$
\int_a^b f(x)\,dx
=
\sum_{n=1}^\infty\int_a^b f_n(x)\,dx
$$
이다.

#### 증명
각 $f_n \in \mathcal{R}[a,b]$이고 $S_N = \sum_{n=1}^N f_n$이 $[a,b]$에서 $f$로 균등수렴하므로,

Thm. 1 [균등수렴과 적분]에 의해 $f \in \mathcal{R}[a,b]$이고
$$
\lim_{N\to\infty}\int_a^b S_N(x)\,dx = \int_a^b f(x)\,dx
$$

한편
$$
\int_a^b S_N(x)\,dx = \int_a^b \sum_{n=1}^N f_n(x)\,dx = \sum_{n=1}^N \int_a^b f_n(x)\,dx
$$

따라서
$$
\int_a^b f(x)\,dx = \lim_{N\to\infty}\sum_{n=1}^N \int_a^b f_n(x)\,dx = \sum_{n=1}^\infty \int_a^b f_n(x)\,dx
$$
□

### Thm. 3. [멱급수의 적분] *(Integration of Power Series)*
멱급수
$$
f(x)=\sum_{n=0}^\infty a_n(x-c)^n
$$
가 $[a,b]$에서 수렴하면 $f\in\mathcal R[a,b]$ 이고
$$
\int_a^b f(x)\,dx
=\sum_{n=0}^\infty a_n\int_a^b (x-c)^n\,dx
$$

#### 증명
멱급수의 수렴반지름을 $R$이라 하자.

$[a,b]$가 수렴구간 $(c-R, c+R)$에 포함되므로, 적당한 $r$에 대해 $[a,b] \subset [c-r, c+r] \subset (c-R, c+R)$이다.

Thm. 1 [수렴반지름과 균등수렴]에 의해 $\sum_{n=0}^\infty a_n(x-c)^n$은 $[c-r, c+r]$에서 균등수렴한다.

각 항 $a_n(x-c)^n$은 다항함수이므로 $[a,b]$에서 리만적분가능하다.

따라서 Thm. 2 [항별적분]에 의해 $f \in \mathcal{R}[a,b]$이고
$$
\int_a^b f(x)\,dx = \int_a^b \sum_{n=0}^\infty a_n(x-c)^n\,dx = \sum_{n=0}^\infty \int_a^b a_n(x-c)^n\,dx = \sum_{n=0}^\infty a_n\int_a^b (x-c)^n\,dx
$$
□

### Thm. 4. [멱급수의 특이적분] *(Improper Integration of Power Series)*
$f(x)=\sum_{n=0}^\infty a_n(x-c)^n$이 $[a,b)$에서 수렴하고
$$
\sum_{n=0}^\infty \frac{a_n}{n+1}(b-c)^{n+1}
$$
이 수렴하면 $f$는 $[a,b)$에서 특이적분가능하고,
$$
\int_a^b f(x)\,dx
= 
\sum_{n=0}^\infty a_n\int_a^b (x-c)^n\,dx
$$

#### 증명
임의의 $\varepsilon > 0$을 택하자.

$\sum_{n=0}^\infty \frac{a_n}{n+1}(b-c)^{n+1}$이 수렴하므로, 충분히 큰 $N$에 대해
$$
\left|\sum_{n=N}^\infty \frac{a_n}{n+1}(b-c)^{n+1}\right| < \varepsilon
$$

$[a, b-\delta]$ (단, $\delta > 0$는 충분히 작음)에서 멱급수는 균등수렴하므로 Thm. 3에 의해
$$
\int_a^{b-\delta} f(x)\,dx = \sum_{n=0}^\infty a_n\int_a^{b-\delta} (x-c)^n\,dx = \sum_{n=0}^\infty \frac{a_n}{n+1}[(b-\delta-c)^{n+1}-(a-c)^{n+1}]
$$

$\delta \to 0^+$일 때, 항별로 극한을 취하면 연속성에 의해
$$
\lim_{\delta \to 0^+}\int_a^{b-\delta} f(x)\,dx = \sum_{n=0}^\infty \frac{a_n}{n+1}[(b-c)^{n+1}-(a-c)^{n+1}]
$$

따라서 $f$는 $[a,b)$에서 특이적분가능하고
$$
\int_a^b f(x)\,dx = \sum_{n=0}^\infty a_n\int_a^b (x-c)^n\,dx
$$
□

# 4. 다변수 함수의 멱급수 *(Power Series of Multivariable Functions)*

## Def. [다변수 멱급수] *(Power Series in Multiple Variables)*
중심점 $\mathbf{c}=(c_1,c_2,\ldots,c_m)\in\mathbb{R}^m$과 계수 $a_{\mathbf{n}}$ (단, $\mathbf{n}=(n_1,n_2,\ldots,n_m)\in\mathbb{N}^m$)에 대하여 다음 멱급수
$$
f(\mathbf{x}) = \sum_{n_1=0}^\infty \sum_{n_2=0}^\infty \cdots \sum_{n_m=0}^\infty a_{n_1,n_2,\ldots,n_m}(x_1-c_1)^{n_1}(x_2-c_2)^{n_2}\cdots(x_m-c_m)^{n_m}
$$
을 **다변수 멱급수** *(power series in multiple variables)* 라 한다.

간단히 표기하면
$$
f(\mathbf{x}) = \sum_{\mathbf{n}\in\mathbb{N}^m} a_{\mathbf{n}}(\mathbf{x}-\mathbf{c})^{\mathbf{n}}
$$
여기서 $(\mathbf{x}-\mathbf{c})^{\mathbf{n}} = (x_1-c_1)^{n_1}(x_2-c_2)^{n_2}\cdots(x_m-c_m)^{n_m}$이다.

### Thm. [다변수 멱급수의 성질] *(Properties of Multivariable Power Series)*
다변수 멱급수가 중심점 주변의 열린 구간에서 절대수렴하면, 그 합 $f(\mathbf{x})$는 다음을 만족한다:

① $f(\mathbf{x})$는 무한번 미분가능하다.

② 모든 혼합 편미분이 항별미분으로 계산 가능하다:
$$
\frac{\partial^{n_1+n_2+\cdots+n_m}}{\partial x_1^{n_1}\partial x_2^{n_2}\cdots\partial x_m^{n_m}}f(\mathbf{x}) = \sum_{k_1=0}^\infty \sum_{k_2=0}^\infty \cdots \sum_{k_m=0}^\infty a_{k_1+n_1,k_2+n_2,\ldots,k_m+n_m}(x_1-c_1)^{k_1}\cdots(x_m-c_m)^{k_m}
$$

③ $\mathbf{a}=(a_1,a_2,\ldots,a_m)$를 중심점 근처의 점이라 하면, **테일러 급수 전개**가 성립한다:
$$
f(\mathbf{x}) = f(\mathbf{a}) + \sum_{r=1}^\infty \frac{1}{r!}\sum_{n_1+n_2+\cdots+n_m=r} \frac{r!}{n_1!n_2!\cdots n_m!}\left[\frac{\partial^r f}{\partial x_1^{n_1}\partial x_2^{n_2}\cdots\partial x_m^{n_m}}\right]_{\mathbf{x}=\mathbf{a}}(x_1-a_1)^{n_1}\cdots(x_m-a_m)^{n_m}
$$

특히 1차 근사는
$$
f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a}) \cdot (\mathbf{x}-\mathbf{a})
$$

### 예제: $e^{x_1+x_2}$의 멱급수 전개
**방법 1: 곱셈 공식**  
일변수 지수함수의 멱급수로부터
$$
e^{x_1+x_2} = e^{x_1} \cdot e^{x_2} = \left(\sum_{n_1=0}^\infty \frac{x_1^{n_1}}{n_1!}\right)\left(\sum_{n_2=0}^\infty \frac{x_2^{n_2}}{n_2!}\right)
$$

이를 전개하면
$$
e^{x_1+x_2} = \sum_{n_1=0}^\infty \sum_{n_2=0}^\infty \frac{x_1^{n_1}x_2^{n_2}}{n_1!n_2!}
$$

**방법 2: 다항정리 (Multinomial Theorem)**  
$e^t = \sum_{n=0}^\infty \frac{t^n}{n!}$에서 $t = x_1 + x_2$로 대입하면
$$
e^{x_1+x_2} = \sum_{n=0}^\infty \frac{(x_1+x_2)^n}{n!}
$$

다항정리에 의해
$$(x_1+x_2)^n = \sum_{n_1+n_2=n} \binom{n}{n_1}x_1^{n_1}x_2^{n_2} = \sum_{n_1=0}^n \frac{n!}{n_1!n_2!}x_1^{n_1}x_2^{n_2}$$

따라서
$$
e^{x_1+x_2} = \sum_{n=0}^\infty \frac{1}{n!}\sum_{n_1=0}^n \frac{n!}{n_1!n_2!}x_1^{n_1}x_2^{n_2} = \sum_{n_1=0}^\infty \sum_{n_2=0}^\infty \frac{x_1^{n_1}x_2^{n_2}}{n_1!n_2!}
$$

**수렴영역**: 모든 $(x_1,x_2) \in \mathbb{R}^2$에서 수렴 (수렴반경: $R_1=R_2=\infty$)

### (3) 예제: $-\log(1-x_1-x_2)$의 멱급수 전개
일변수 로그함수의 멱급수로부터
$$
-\log(1-t) = \sum_{n=1}^\infty \frac{t^n}{n}, \quad |t|<1
$$

$t = x_1 + x_2$로 치환하면
$$
-\log(1-x_1-x_2) = \sum_{n=1}^\infty \frac{(x_1+x_2)^n}{n}
$$

다항정리에 의해
$$(x_1+x_2)^n = \sum_{n_1+n_2=n} \binom{n}{n_1}x_1^{n_1}x_2^{n_2}$$

따라서
$$
-\log(1-x_1-x_2) = \sum_{n=1}^\infty \frac{1}{n}\sum_{n_1=0}^n \binom{n}{n_1}x_1^{n_1}x_2^{n_2}
$$

모든 항을 명시적으로 나열하면
$$
-\log(1-x_1-x_2) = \sum_{n_1=1}^\infty \sum_{n_2=0}^\infty \frac{1}{n_1+n_2}\binom{n_1+n_2}{n_1}x_1^{n_1}x_2^{n_2}
$$

**낮은 차수 항 예시:**
$$
\begin{align*}
&= (x_1 + x_2) + \frac{1}{2}(x_1^2 + 2x_1x_2 + x_2^2)\\
&\quad + \frac{1}{3}(x_1^3 + 3x_1^2x_2 + 3x_1x_2^2 + x_2^3) + \cdots
\end{align*}
$$

**수렴영역**

수렴하려면 $|x_1+x_2|<1$이어야 하므로
$$
\{(x_1,x_2) : |x_1+x_2|<1\}
$$

이는 기울기가 $\pm 1$인 두 직선 사이의 영역이다.



# [연습문제]
1. 다음 함수열이 $[0,1]$에서  
   ① 점별수렴, ② 균등수렴하는지 판별하시오.  
   (1) $f_n(x)=x^n$  
   (2) $f_n(x)=\dfrac{x^n}{n}$  

2. $\displaystyle\sum_{n=0}^\infty\frac1{2^n}(x-1)^n$의 수렴반지름과 수렴구간을 구하시오.

3. $f(x)=\displaystyle\sum_{n=1}^\infty\frac{\cos nx}{n^2}$가 $(-\infty,\infty)$에서 연속임을 보이시오.

4. $f_n(x)=|x|^{1+1/n}$과 극한함수 $f$가 $(-1,1)$에서 미분가능하지 않은 이유를 설명하시오.

5. 다음을 구하시오.  
   (1) $f(x)=\displaystyle\sum_{n=1}^\infty\frac{x^{2n}}{2^n(2n+1)}$의 $\int_0^1 f(x)\,dx$  
   (2) $f(x)=\displaystyle\sum_{n=1}^\infty\frac{(-1)^n(x-1)^n}{n}$의 $\int_1^2 f(x)\,dx$
