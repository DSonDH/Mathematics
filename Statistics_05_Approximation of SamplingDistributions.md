# 제5장 표본분포의 근사 *(Approximation of Sampling Distributions)*

## 중심극한정리 *(Central Limit Theorem)*

### 이항분포와 정규근사
이항분포의 누적확률은 적절한 표준화를 거치면 표준정규분포의 누적확률로 근사될 수 있다. 즉, 다음과 같은 근사식이 성립한다.

$$
\sum_{x:\, a \le \frac{x-np}{\sqrt{np(1-p)}} \le b}
\binom{n}{x}p^x(1-p)^{n-x}
\approx
\int_a^b \frac{1}{\sqrt{2\pi}}e^{-z^2/2}\,dz,
\quad n \to \infty
$$

이는 앞선 장(제3장 제6절)에서 이미 소개된 바 있다.

### 베르누이 표본과 표본비율의 정규근사
확률변수 $X_1, X_2, \dots, X_n$이 서로 독립이고 동일한 베르누이분포 Bernoulli$(p)$를 따른다고 하자.
이때 불량품의 총 개수는

$$
X_1 + \cdots + X_n \sim \text{Binomial}(n,p)
$$

를 따른다.

표준정규분포 $N(0,1)$를 따르는 확률변수 $Z$에 대하여 다음이 성립한다.

$$
\lim_{n\to\infty}
P\!\left(
a \le \frac{X_1+\cdots+X_n-np}{\sqrt{np(1-p)}} \le b
\right)
= P(a \le Z \le b)
$$

이를 표본비율 $\hat p = (X_1+\cdots+X_n)/n$에 대해 쓰면 다음과 같다.

$$
\lim_{n\to\infty}
P\!\left(
a \le \frac{\hat p - p}{\sqrt{p(1-p)/n}} \le b
\right)
= P(a \le Z \le b)
$$

이와 같이 통계량에 관한 확률이 표준정규분포의 확률로 근사되는 경우가 많다. 이러한 근사식이 성립하는 통계량 중 가장 기본적이고 중요한 것이 표본평균이다.

### 중심극한정리의 역할
중심극한정리(Central Limit Theorem)는 위와 같은 근사의 이론적 근거를 제공하며, 확률근사 이론에서 핵심적인 역할을 한다.

### 정리 5.1.1. 중심극한정리 *(Central Limit Theorem)*
확률변수 $X_1, \dots, X_n$이 서로 독립이고 동일한 분포를 따르며, 분산이 유한하다고 하자.

$$
E(X_1)=\mu,\quad \mathrm{Var}(X_1)=\sigma^2,\quad 0<\sigma<\infty
$$

라고 하면, 표준정규분포 $N(0,1)$를 따르는 확률변수 $Z$에 대하여 다음이 성립한다.

$$
\lim_{n\to\infty}
P\!\left(
\frac{(X_1+\cdots+X_n)/n - \mu}{\sigma/\sqrt{n}} \le x
\right)
= P(Z \le x),
\quad \forall x\in\mathbb{R}
$$

#### 증명 (적률생성함수 방법)
이 정리는 주어진 조건보다 더 일반적인 조건에서도 성립하지만, 여기서는 추가 조건으로 $X_1$의 적률생성함수(mgf, moment generating function)가 존재하는 경우만 다룬다.

표준화된 표본평균을

$$
\frac{\sqrt{n}(\bar X_n-\mu)}{\sigma}
= \frac{(X_1+\cdots+X_n)/n - \mu}{\sigma/\sqrt{n}}
$$

이라 하자.

누적분포함수 대신 적률생성함수를 이용하여, 위 확률변수의 적률생성함수가 표준정규분포의 적률생성함수

$$
\mathrm{mgf}_Z(t)=\exp(t^2/2)
$$

로 수렴함을 보이면 충분하다.

즉,

$$
\lim_{n\to\infty}
\mathrm{mgf}_{\sqrt{n}(\bar X_n-\mu)/\sigma}(t)
= \mathrm{mgf}_Z(t) = \exp(t^2/2)
$$

를 보인다.

독립성에 의해

$$
\mathrm{mgf}_{\sqrt{n}(\bar X_n-\mu)/\sigma}(t)
= E\!\left[
\exp\!\left(
\frac{t}{\sqrt{n}}\sum_{i=1}^n \frac{X_i-\mu}{\sigma}
\right)
\right]
= \left[
\mathrm{mgf}_{(X_1-\mu)/\sigma}\!\left(\frac{t}{\sqrt{n}}\right)
\right]^n
$$

라 하자.

$m(s)=\mathrm{mgf}_{(X_1-\mu)/\sigma}(s)$라 두면, 테일러 전개로

$$
m\!\left(\frac{t}{\sqrt{n}}\right)
= 1+\frac{1}{2}\frac{t^2}{n}+R_{n,t},
\quad \lim_{n\to\infty} nR_{n,t}=0
$$

를 얻는다.

따라서

$$
\log \mathrm{mgf}
= n\log\!\left(1+\frac{1}{2n}t^2+R_{n,t}\right)
= \frac{1}{2}t^2 + nr_{n,t},
\quad \lim_{n\to\infty}nr_{n,t}=0
$$

이므로

$$
\lim_{n\to\infty}
\mathrm{mgf}_{\sqrt{n}(\bar X_n-\mu)/\sigma}(t)
= \exp(t^2/2)
$$

가 성립한다.

#### 각주 67
일반적으로 확률변수열 $Z_n$의 적률생성함수가 어떤 열린 구간에서 $Z$의 적률생성함수로 수렴하고, $Z$의 누적분포함수가 연속이면

$$
\lim_{n\to\infty} P(Z_n\le x)=P(Z\le x)
$$

가 성립하는 것이 알려져 있다.

#### 예 5.1.1. 분포들의 정규근사
**(a) 포아송분포의 정규근사**  
서로 독립이고 $Poisson(\lambda)$를 따르는 $X_1,\dots,X_n$에 대해
$$
E(X_1)=\mathrm{Var}(X_1)=\lambda
$$
이다.

중심극한정리에 의해
$$
\lim_{n\to\infty}
P\!\left(
\frac{(X_1+\cdots+X_n)/n-\lambda}{\sqrt{\lambda/n}} \le x
\right)
= P(Z\le x)
$$

또한 $X_1+\cdots+X_n\sim\text{Poisson}(n\lambda)$이므로
$$
\lim_{n\to\infty}
\sum_{k:\, (k-n\lambda)/\sqrt{n\lambda}\le x}
\frac{e^{-n\lambda}(n\lambda)^k}{k!}
= \int_{-\infty}^x \frac{1}{\sqrt{2\pi}}e^{-z^2/2}dz
$$

따라서 $Y_n\sim\text{Poisson}(n)$에 대해
$$
P(a<Y_n\le b)
\approx
\Phi\!\left(\frac{b-n}{\sqrt{n}}\right)
- \Phi\!\left(\frac{a-n}{\sqrt{n}}\right)
$$
(각주: 이런 근사계산은 중심극한정리에서 정규분포로의 수렴이 균등수렴이므로 가능한 것이다.)

**(b) 감마분포 및 카이제곱분포의 정규근사**  
$X_1,\dots,X_n\sim\text{Gamma}(\alpha,\beta)$라 하면
$$
E(X_1)=\alpha\beta,\quad \mathrm{Var}(X_1)=\alpha\beta^2
$$

중심극한정리에 의해
$$
\lim_{n\to\infty}
P\!\left(
\frac{(X_1+\cdots+X_n)/n-\alpha\beta}{\sqrt{\alpha\beta^2/n}} \le x
\right)
= P(Z\le x)
$$
이므로 표준정규분포로 근사할 수 있다.
특히 $Y_n\sim\chi^2(n)$인 경우, $Y_n$은 $n$개의 독립인 $\chi^2(1)$ 확률변수의 합이므로 $\text{Gamma}(n/2, 2)$와 같다. 따라서 $\alpha=n/2$, $\beta=2$를 대입하면
$$
\lim_{n\to\infty}
P\!\left(
\frac{Y_n-n}{\sqrt{2n}} \le x
\right)
= P(Z\le x)
$$
가 성립한다.

### 다차원 중심극한정리
중심극한정리는 다차원 확률변수에도 성립하며, 많은 통계량의 표본분포를 근사하는데 핵심적인 역할을 한다. 그 증명은 이차원의 경우와 같으므로 생략한다.

### 정리 5.1.2. 다차원 경우의 중심극한정리
확률벡터 $X_1,\dots,X_n$이 서로 독립이고 동일한 분포를 따르며

$$
E(X_1)=\mu,\quad \mathrm{Var}(X_1)=\Sigma
$$

라 하자. 이때 $Z\sim N_k(0,\Sigma)$에 대해

$$
\lim_{n\to\infty}
P\!\left(
\sqrt{n}\left(\frac{X_{1j}+\cdots+X_{nj}}{n}-\mu_j\right)
\le x_j,\; j=1,\dots,k
\right)
= P(Z_1\le x_1,\dots,Z_k\le x_k)
$$
가 성립한다.

참고로,
$$
\lim_{n\to\infty}
P\!\left(
\sqrt{n}(\bar X_n - \mu) \le x
\right)
= P(Z \le x),
\quad \forall x\in\mathbb{R}^k,\quad Z\sim N_k(0,\Sigma)
$$
로 나타내기도 한다. 여기서 $\bar X_n = (X_1+\cdots+X_n)/n$이고, 부등호는 성분별(componentwise) 부등호를 의미한다.

#### 예 5.1.2. 다항분포의 다변량 정규근사
$X_n\sim\text{Multinomial}(n,p_1,\dots,p_k)$라 하면

$$
E(X_1)=p,\quad \mathrm{Var}(X_1)=D(p_j)-pp^\top
$$

중심극한정리에 의해

$$
\lim_{n\to\infty}
P\!\left(
\frac{X_n-np}{\sqrt{n}} \le x
\right)
= P(Z\le x),\quad
Z\sim N_k(0,D(p_j)-pp^\top)
$$
이때 분산행렬은 특이행렬이라 역행렬을 갖지않음에 유의해야 한다.


## 극한분포와 확률적 수렴 *(Limiting Distributions and Convergence in Probability)*

### 극한분포의 정의
중심극한정리에서 근사에 이용되는 분포는 정규분포이며, 이 경우 누적분포함수는 연속이다. 그러나 이항분포의 포아송 근사와 같이, 근사에 이용되는 분포의 누적분포함수가 **연속이 아닐 수도 있다**. 이제 일반적인 경우에 대해 분포의 근사를 정의한다.

#### 예 5.2.1. 불연속 극한의 예
동전을 던져 앞면이 나오면 $0$부터 $1+n^{-1}$ 사이의 수를 균등하게 선택하여 보여주고, 뒷면이 나오면 $1+n^{-1}$을 보여주는 실험에서 관측되는 확률변수를 $X_n,(n=1,2,\dots)$이라 하자.

이때 $X_n$의 누적분포함수는
$$
\mathrm{cdf}_{X_n}(x)
=
\begin{cases}
0, & x<0 \\
\dfrac{1}{2}\dfrac{x}{1+n^{-1}}, & 0\le x<1+n^{-1} \\
1, & x\ge 1+n^{-1}
\end{cases}
$$

이 누적분포함수의 점별 극한으로 정의되는 함수는
$$
G(x)=\lim_{n\to\infty}\mathrm{cdf}_{X_n}(x)
=
\begin{cases}
0, & x<0 \\
\dfrac{1}{2}x, & 0\le x\le 1 \\
1, & x>1
\end{cases}
$$

그러나 $G(x)$는 $x=1$에서 오른쪽 연속이 아니므로 누적분포함수가 아니다.

위 예에서 극한 함수가 연속이 아니므로, $x=1$에서 오른쪽 연속이 되도록 수정한 함수
$$
F(x)=
\begin{cases}
0, & x<0 \\
\dfrac{1}{2}x, & 0\le x<1 \\
1, & x\ge 1
\end{cases}
$$
를 정의한다.

이 함수 $F(x)$는 누적분포함수이며, 예 5.2.1의 확률변수열 $X_n$에 대해
$$
\lim_{n\to\infty}\mathrm{cdf}_{X_n}(x)=F(x)\quad \forall x\neq 1
$$
이 성립한다.

이 경우, 앞면이 나오면 $0$부터 $1$ 사이의 수를 균등하게 선택하고, 뒷면이 나오면 $1$을 보여주는 확률변수를 $X$라 하면, $F(x)$는 $X$의 누적분포함수이다. 따라서 $X_n$의 분포는 $X$의 분포로 근사된다고 말할 수 있다.

이 예시처럼 극한으로 주어지는 분포의 누적분포함수가 연속이 아닌 경우까지 다루기 위해 극한분포를 아래와 같이 정의한다.

### 정의: 극한분포(limiting distribution)
확률변수열 $X_n,(n=1,2,\dots)$과 확률변수 $Z$에 대하여
$$
\lim_{n\to\infty}P(X_n\le x)=P(Z\le x)
$$
가 $Z$의 누적분포함수 $\mathrm{cdf}_Z$가 **연속인 모든 점 $x$** 에서 성립하면, $Z$의 분포를 $X_n$의 **극한분포(limiting distribution)** 또는 **점근분포(asymptotic distribution)** 라 한다.  
기호로는
$$
X_n \xrightarrow{d} Z
$$

$cdf_z$가 연속인 점들의 집합을 $Conti(cdf_z)$라 하면
$$
X_n \xrightarrow{d} Z
\;\Longleftrightarrow\;
\lim_{n\to\infty}\mathrm{cdf}_{X_n}(x)=\mathrm{cdf}_Z(x)
\quad \forall x\in\mathrm{Conti}(\mathrm{cdf}_Z)
$$

**극한분포의 의의**  
극한분포는 표본크기가 커질 때 통계량의 분포가 어떤 분포로 수렴하는지 설명해 준다. 이를 통해 복잡한 표본분포를 단순한 분포(정규분포, 포아송분포 등)로 근사할 수 있으며, 신뢰구간 계산, 검정 등 통계적 추론의 이론적 근거를 제공한다. 실제로 많은 통계량의 분포는 극한분포로 근사하여 실용적으로 활용된다.

#### 예 5.2.2. 이항분포의 포아송 근사
이항분포 $B(n,\lambda/n)$을 따르는 확률변수 $X_n$과 포아송분포 $\mathrm{Poisson}(\lambda)$를 따르는 확률변수 $X$에 대해

$$
P(X_n=k)
=
\binom{n}{k}\left(\frac{\lambda}{n}\right)^k
\left(1-\frac{\lambda}{n}\right)^{n-k}
\;\xrightarrow[n\to\infty]{}\;
\frac{\lambda^k e^{-\lambda}}{k!}
=
P(X=k)
$$
가 성립한다.

또한 이들 확률변수는 $0$ 또는 자연수 값만 가지므로 모든 $x$에 대해
$$
\mathrm{cdf}_{X_n}(x)
=\sum_{k:0\le k\le x}P(X_n=k)
\;\xrightarrow[n\to\infty]{}\;
\sum_{k:0\le k\le x}P(X=k)
=\mathrm{cdf}_X(x)
$$

따라서
$$
B(n,\lambda/n)\;\approx\;\mathrm{Poisson}(\lambda),\quad n\to\infty
$$

#### 예 5.2.3. 최대 순서통계량의 극한분포
균등분포 $U(0,1)$에서의 랜덤표본 $n$개에 기초한 순서통계량을 $U_{(1)}<\cdots<U_{(n)}$이라 하자.

$$
P\{n(1-U_{(n)})\le x\}
=
1-P\left\{U_{(n)}<1-\frac{x}{n}\right\}
$$
$$
P\left\{U_{(n)}<1-\frac{x}{n}\right\}
=\begin{cases}
0, & x>n \\
\left(1-\frac{x}{n}\right)^n, & 0<x\le n \\
1, & x\le 0
\end{cases}
$$
따라서
$$
\lim_{n\to\infty}
P\left\{U_{(n)}<1-\frac{x}{n}\right\}
=
\begin{cases}
e^{-x}, & x>0 \\
1, & x\le 0
\end{cases}
$$
이므로

$$
\lim_{n\to\infty}
P\{n(1-U_{(n)})\le x\}
=
\begin{cases}
1-e^{-x}, & x\ge 0 \\
0, & x<0
\end{cases} \\
\therefore n(1-U_{(n)}) \xrightarrow{d} Z,\quad Z\sim\mathrm{Exp}(1)
$$

### 확률수렴 *(Convergence in Probability)*
**분포수렴과 확률수렴의 비교**  
| 구분 | 분포수렴 (Convergence in Distribution) | 확률수렴 (Convergence in Probability) |
|------|---------------------------------------|---------------------------------------|
| 정의 | $X_n \xrightarrow{d} X$<br>임의의 연속점 $x$에 대해<br>$\lim_{n\to\infty}P(X_n \le x) = P(X \le x)$ | $X_n \xrightarrow{P} c$<br>임의의 $\varepsilon>0$에 대해<br>$\lim_{n\to\infty}P(\|X_n-c\|\ge\varepsilon)=0$ |
| 극한 | 임의의 확률변수 $X$ (분포) | 상수 $c$ (확률변수의 값) |
| 의미 | $X_n$의 분포가 $X$의 분포에 가까워짐 | $X_n$이 상수 $c$에 확률적으로 가까워짐 |
| 강도 | 약한 수렴 (weak convergence) | 강한 수렴 (stronger than distribution) |
| 연산 | 연속함수 $g$에 대해 $X_n\xrightarrow{d}X$이면 $g(X_n)\xrightarrow{d}g(X)$ | 연속함수 $g$에 대해 $X_n\xrightarrow{P}c$이면 $g(X_n)\xrightarrow{P}g(c)$ |
| 관계 | 확률수렴이면 분포수렴이 항상 성립<br>($X_n\xrightarrow{P}c$이면 $X_n\xrightarrow{d}c$) | 분포수렴이 확률수렴을 보장하지 않음 |

**요약:**  
- 확률수렴은 분포수렴을 함의하지만, 그 역은 성립하지 않는다.
- 확률수렴은 값 자체가 상수에 가까워지는 것이고, 분포수렴은 분포의 형태가 가까워지는 것이다.
- 통계적 추정량의 일치성(consistency)은 확률수렴의 개념에 기반한다.
- 중심극한정리 등에서 등장하는 근사분포는 분포수렴의 개념을 사용한다.

### 정리 5.2.1. 극한분포가 상수인 경우
확률변수열 $X_n$의 극한분포가 상수 $c$의 분포일 필요충분조건:
$$
X_n\xrightarrow{d}X,\;P(X=c)=1
\;\Longleftrightarrow\;
\lim_{n\to\infty}P(|X_n-c|\ge\varepsilon)=0
\quad \forall\varepsilon>0
$$

#### 증명
($\Rightarrow$)  
$P(X=c)=1$이면, $\mathrm{cdf}_X(x)$는 $x=c$를 제외한 모든 점에서 연속이다. 극한분포의 정의에 따라
$$
\lim_{n\to\infty}\mathrm{cdf}_{X_n}(x)=\mathrm{cdf}_X(x)\quad(x\neq c)
$$

임의의 $\varepsilon>0$에 대해,
$$
P(|X_n-c|\ge\varepsilon)
= P(X_n\le c-\varepsilon) + P(X_n\ge c+\varepsilon)
= \mathrm{cdf}_{X_n}(c-\varepsilon) + 1 - \mathrm{cdf}_{X_n}(c+\varepsilon)
$$

$n\to\infty$로 보낼 때, 극한분포의 정의에 의해
$$
\mathrm{cdf}_{X_n}(c-\varepsilon) \to \mathrm{cdf}_X(c-\varepsilon),\quad
\mathrm{cdf}_{X_n}(c+\varepsilon) \to \mathrm{cdf}_X(c+\varepsilon)
$$
이고, $X$가 상수 $c$이므로
$$
\mathrm{cdf}_X(c-\varepsilon) = 0,\quad \mathrm{cdf}_X(c+\varepsilon) = 1
$$
따라서
$$
\lim_{n\to\infty}P(|X_n-c|\ge\varepsilon) = 0 + 1 - 1 = 0
$$

($\Leftarrow$)  
주어진 조건
$$
\lim_{n\to\infty}P(|X_n-c|\ge\varepsilon)=0
$$
에서, $x < c$인 경우
$$
\mathrm{cdf}_{X_n}(x) = P(X_n \le x) \le P(X_n \le c-\varepsilon) \le P(|X_n-c|\ge\varepsilon)
$$
이므로 $\lim_{n\to\infty}\mathrm{cdf}_{X_n}(x)=0=\mathrm{cdf}_X(x)$이다.

$x > c$인 경우
$$
1-\mathrm{cdf}_{X_n}(x) = P(X_n > x) \le P(X_n \ge c+\varepsilon) \le P(|X_n-c|\ge\varepsilon)
$$
이므로 $\lim_{n\to\infty}\mathrm{cdf}_{X_n}(x)=1=\mathrm{cdf}_X(x)$이다.

따라서 모든 $x\neq c$에 대해
$$
\lim_{n\to\infty}\mathrm{cdf}_{X_n}(x)=\mathrm{cdf}_X(x)
$$
가 성립한다. $\square$

### 정의: 확률수렴
정리 5.2.1의 조건이 성립할 때, 확률변수 $X_n$이 상수 $c$로 **확률수렴(convergence in probability)** 한다고 하며
$$
X_n\xrightarrow{P}c
\quad\text{또는}\quad
\mathrm{plim}_{n\to\infty}X_n=c
$$
로 쓴다. 즉,
$$
\mathrm{plim}_{n\to\infty}X_n=c
\;\Longleftrightarrow\;
\lim_{n\to\infty}P(|X_n-c|\ge\varepsilon)=0
\quad \forall\varepsilon>0
$$

### 정리 5.2.2. 큰수의 법칙 (대수의 법칙) *(Law of Large Numbers)*
서로 독립이고 동일한 분포를 따르는 확률변수 $X_1,\dots,X_n$에 대해 $E(X_1)$이 존재하면
$$
\mathrm{plim}_{n\to\infty}\frac{1}{n}\sum_{i=1}^n X_i = E(X_1)
$$

#### 증명
주어진 조건하에서 일반적으로 성립하지만, 조건부 증명만 소개한다.  
추가 조건 $\mathrm{Var}(X_1)<\infty$ 하에서 체비셰프 부등식을 적용하면
$$
P(|\bar X_n-E(X_1)|\ge\varepsilon)
\le \frac{\mathrm{Var}(\bar X_n)}{\varepsilon^2}
=\frac{\mathrm{Var}(X_1)}{n\varepsilon^2}
$$
이고, $n\to\infty$로 보내면 확률이 $0$으로 수렴한다. $\square$

**큰수의 법칙 해석**  
랜덤표본 $X_1, \dots, X_n$을 관측할 때, 집합 $A$에 속하는 관측값의 **상대도수(relative frequency)** 는
$$
\frac{1}{n}\sum_{i=1}^n I_A(X_i)
$$
로 나타낼 수 있다.  
이 상대도수에 큰수의 법칙을 적용하면
$$
\mathrm{plim}_{n\to\infty}\frac{1}{n}\sum_{i=1}^n I_A(X_i) = E(I_A(X_1)) = P(X_1 \in A)
$$

즉, 시행 횟수 $n$이 커질수록 상대도수는 확률 $P(X_1 \in A)$에 한없이 가까워진다.  

통계적 추론 관점에서는 표본크기 $n$이 커질수록 이러한 통계량이 모집단의 진짜 값(모수)에 가까워진다는 것을 의미한다. 즉, 표본평균이 모집단 평균에, 표본분산이 모집단 분산에 확률적으로 수렴한다는 사실은 추정량의 **일치성(consistency)**  을 보장한다.

이 개념을 많은 통계량에 적용하려면 이 정의를 아래의 다차원 확률변수로 확장시킬 필요가 있다  
### 다차원 확률수렴
다차원 확률변수 $X_n=(X_{n1},\dots,X_{nk})^t$와 상수벡터 $c=(c_1,\dots,c_k)^t$에 대해
$$
|X_n-c|
=
\sqrt{(X_{n1}-c_1)^2+\cdots+(X_{nk}-c_k)^2}
$$

로 정의하면,
$$
\mathrm{plim}_{n\to\infty}X_n=c
\;\Longleftrightarrow\;
\lim_{n\to\infty}P(|X_n-c|\ge\varepsilon)=0
$$

### 정리 5.2.3. 성분별 확률수렴
다차원 확률변수 $X_n=(X_{n1},\dots,X_{nk})^t$와 상수벡터 $c=(c_1,\dots,c_k)^t$에 대해 다음 조건들은 서로 동치이다.

(a) $\displaystyle\mathrm{plim}_{n\to\infty}X_n=c$

(b) $\displaystyle\lim_{n\to\infty}P\left(\max_{1\le i\le k}|X_{ni}-c_i|\ge\varepsilon\right)=0\quad\forall\varepsilon>0$

(c) 모든 $i=1,\dots,k$에 대해 $\displaystyle\mathrm{plim}_{n\to\infty}X_{ni}=c_i$

#### 증명
벡터 노름의 성질을 이용하여 각 조건의 동치성을 보인다.  
**(a) ⇒ (b):**  
(a)에서 확률수렴의 정의에 따라, 임의의 $\varepsilon>0$에 대해
$$
\lim_{n\to\infty}P(|X_n-c|\ge\varepsilon)=0
$$
이다. 벡터 노름의 성질에 의해
$$
\max_{1\le i\le k}|X_{ni}-c_i| \le |X_n-c|
$$
이므로
$$
P\left(\max_{1\le i\le k}|X_{ni}-c_i|\ge\varepsilon\right)
\le P(|X_n-c|\ge\varepsilon)
$$
따라서 (a)에서 (b)가 성립한다.

**(b) ⇒ (a):**  
반대로, 벡터 노름의 삼각부등식에 의해
$$
|X_n-c| \le \sum_{i=1}^k|X_{ni}-c_i| \le k\max_{1\le i\le k}|X_{ni}-c_i|
$$
이므로
$$
P(|X_n-c|\ge\varepsilon)
\le P\left(\max_{1\le i\le k}|X_{ni}-c_i|\ge\varepsilon/k\right)
$$
(b)에서 임의의 $\varepsilon>0$에 대해
$$
\lim_{n\to\infty}P\left(\max_{1\le i\le k}|X_{ni}-c_i|\ge\varepsilon/k\right)=0
$$
이므로 (a)가 성립한다.

**(b) ⇔ (c):**  
(b)는 모든 성분 $i$에 대해 $|X_{ni}-c_i|$가 $\varepsilon$ 이상일 확률이 0으로 수렴함을 의미한다. 이는 각 성분별로
$$
\lim_{n\to\infty}P(|X_{ni}-c_i|\ge\varepsilon)=0
$$
임을 뜻하므로 (c)와 동치이다.

따라서 (a), (b), (c)는 서로 동치이다. $\square$

### 정리 5.2.4. 다차원 큰수의 법칙  
서로 독립이고 동일한 분포를 따르는 다차원 확률변수  
$$
X_1 = (X_{11}, \dots, X_{1k})^T, \dots, X_n = (X_{n1}, \dots, X_{nk})^T
$$  
에 대해 $E(X_1) = (E(X_{11}), \dots, E(X_{1k}))^T$가 정의될 수 있으면  
$$
\mathrm{plim}_{n\to\infty} \frac{1}{n} \sum_{i=1}^n X_i = E(X_1)
$$  

#### 증명  
각 성분별로 큰수의 법칙이 성립하므로, 정리 5.2.3(성분별 확률수렴)에 의해 결론이 성립한다. $\square$

#### 예 5.2.4. 표본적률의 확률수렴
서로 독립이고 동일한 분포를 따르는 확률변수 $X_1,\dots,X_n$에 대해 $\hat m_r = \frac{1}{n}\sum_{i=1}^{n}X_i^r$을 $k$차 표본적률(sample $k$ th moment)이라 한다.  
$E(|X_1|^k)<\infty$이면
$$
\mathrm{plim}_{n\to\infty}\frac{1}{n}\sum_{i=1}^n X_i^k = E(X_1^k)
$$
즉, $k$차 표본적률이 모집단의 $k$차 모적률(population $k$ th moment)로 확률수렴한다.

이는 $X_1^k,\dots,X_n^k$ 역시 서로 독립이고 동일한 분포를 따르며, $E(|X_1^k|)<\infty$이므로 정리 1.6.2로부터 큰수의 법칙을 그대로 적용할 수 있다. 따라서 표본평균 $\frac{1}{n}\sum_{i=1}^n X_i^k$는 모집단의 적률벡터 $E(X_1^k)$에 확률적으로 가까워진다.

이 결과는 표본평균뿐만 아니라 표본분산, 표본적률 등 다양한 통계량의 일치성(consistency)을 보장하는 근거가 된다.

### 정리 5.2.5. 연속함수와 확률수렴
$X_n\xrightarrow{P}c$이고 실수값 함수 $g$가 $c$에서 연속이면
$$
\mathrm{plim}_{n\to\infty}g(X_n)=g(c)
$$

#### 증명
연속성의 정의에 따라 임의의 $\varepsilon>0$에 대해 어떤 $\delta>0$가 존재하여
$$
|x-c|<\delta \;\Longrightarrow\; |g(x)-g(c)|<\varepsilon
$$
가 성립한다. 이를 대우(contrapositive)로 바꾸면
$$
|g(x)-g(c)|\ge\varepsilon \;\Longrightarrow\; |x-c|\ge\delta
$$
따라서 확률변수 $X_n$에 대해 사건의 포함관계는
$$
\{|g(X_n)-g(c)|\ge\varepsilon\} \subseteq \{|X_n-c|\ge\delta\}
$$
따라서
$$
P(|g(X_n)-g(c)|\ge\varepsilon)
\le P(|X_n-c|\ge\delta)
$$
이때 확률수렴의 정의에 따라 $lim_{n\to\infty}P(|X_n-c|\ge\delta)=0$ 이므로
$$
\lim_{n\to\infty}P(|g(X_n)-g(c)|\ge\varepsilon)=0
$$
즉, $g(X_n)\xrightarrow{P}g(c)$임을 알 수 있다. $\square$  

### 정리 5.2.6. 확률수렴과 사칙연산
$X_n\xrightarrow{P}a$, $Y_n\xrightarrow{P}b$이면 다음이 성립한다.

(a) $\mathrm{plim}(X_n+Y_n)=a+b$

(b) $\mathrm{plim}(X_n-Y_n)=a-b$

(c) $\mathrm{plim}(X_nY_n)=ab$

(d) $\mathrm{plim}(X_n/Y_n)=a/b\;(b\neq0)$

#### 증명
정리 5.2.3에 의해 $(X_n,Y_n)^t\xrightarrow{P}(a,b)^t$임을 알 수 있다. 이제 각 사칙연산에 대해 다음과 같이 보일 수 있다.

(a), (b): 덧셈과 뺄셈 함수 $g(x,y)=x+y$, $g(x,y)=x-y$는 $(a,b)$에서 연속이므로, 정리 5.2.5(연속함수와 확률수렴)에 의해 $\mathrm{plim}(X_n+Y_n) = a+b,\quad \mathrm{plim}(X_n-Y_n) = a-b$  
(c): 곱셈 함수 $g(x,y)=xy$도 $(a,b)$에서 연속이므로 $\mathrm{plim}(X_nY_n) = ab$  
(d): 나눗셈 함수 $g(x,y)=x/y$는 $b\neq0$일 때 $(a,b)$에서 연속이므로 $
\mathrm{plim}(X_n/Y_n) = a/b \\
\square$

#### 예 5.2.5. 표본분산과 표본표준편차의 확률수렴
랜덤표본 $X_1,X_2,\dots,X_n,(n\ge2)$을 이용하여 모분산 추정에 사용되는 **표본분산(sample variance)** 의 정의: $S_n^2=\frac{1}{n-1}\sum_{i=1}^n (X_i-\bar X)^2$는 다음과 같이 두 표본적률의 함수로 나타낼 수 있다.
$$
S_n^2
= \frac{1}{n-1}\sum_{i=1}^n (X_i-\bar X)^2
= \frac{n}{n-1}\left\{
\frac{1}{n}\sum_{i=1}^n X_i^2
- \left(\frac{1}{n}\sum_{i=1}^n X_i\right)^2
\right\}
$$
즉, $(\frac{1}{n}\sum_{i=1}^n X_i,\, \frac{1}{n}\sum_{i=1}^n X_i^2)^T$의 함수로 $S_n^2$를 표현할 수 있다.

한편 예 5.2.4 및 큰수의 법칙으로부터, $E(X_1^2)<+\infty$이면
$$
\mathrm{plim}_{n\to\infty}\frac{1}{n}\sum_{i=1}^n X_i^2=E(X_1^2),
\quad
\mathrm{plim}_{n\to\infty}\frac{1}{n}\sum_{i=1}^n X_i=E(X_1)
$$

따라서 정리 5.2.6(사칙연산의 보존성)에 의해
$$
\mathrm{plim}_{n\to\infty} S_n^2
=
1\times\{E(X_1^2)-[E(X_1)]^2\}
=\mathrm{Var}(X_1)=\sigma^2
$$

즉, 모분산 $\sigma^2$가 양의 실수이면
$$
\mathrm{plim}_{n\to\infty} S_n^2=\sigma^2
$$

또한 표본표준편차 $S_n=\sqrt{S_n^2}$에 대해, 제곱근 함수는 연속이므로 정리 5.2.5로부터
$$
\mathrm{plim}_{n\to\infty} S_n = \sqrt{\mathrm{plim}_{n\to\infty}S_n^2} =\sigma
$$

### 정리 5.2.7. 평균제곱수렴과 확률수렴
큰수의 법칙을 적용하기 어려운 경우에 확률수렴을 밝히는 데 유용한 정리다.  

분산이 실수로 정의될 수 있는 확률변수 $X_n,(n=1,2,\dots)$에 대해
$\lim_{n\to\infty}\mathrm{Var}(X_n)=0,
\quad
\lim_{n\to\infty}E(X_n)=a$ 이면
$$
\mathrm{plim}_{n\to\infty}X_n=a
$$
#### 증명
마르코프 부등식으로부터 임의의 $\varepsilon>0$에 대해
$$
P(|X_n-a|\ge\varepsilon)
\le
\frac{E[(X_n-a)^2]}{\varepsilon^2}
$$

한편
$$
E[(X_n-a)^2]
=
\mathrm{Var}(X_n)+\{E(X_n)-a\}^2
$$
이고, 가정에 의해 $\lim_{n\to\infty}E[(X_n-a)^2]=0$
이다.

따라서
$$
0\le
\lim_{n\to\infty}P(|X_n-a|\ge\varepsilon)
\le 0
$$
이 되어 결론이 성립한다. $\square$

#### 예 5.2.6. 최대 순서통계량의 확률수렴
균등분포 $U(0,1)$에서의 랜덤표본 $n$개에 기초한 순서통계량을 $U_{(1)}<\cdots<U_{(n)}$이라 하자.

$U_{(n)}$의 확률밀도함수는 $\mathrm{pdf}_{U_{(n)}}(x)=n x^{n-1} I_{(0,1)}(x)$ 이므로
$$
E(U_{(n)})=\int_0^1 x\cdot nx^{n-1}dx=\frac{n}{n+1}\\
E(U_{(n)}^2)=\int_0^1 x^2\cdot nx^{n-1}dx=\frac{n}{n+2} \\
\mathrm{Var}(U_{(n)})=\frac{n}{n+2}-\left(\frac{n}{n+1}\right)^2 \\
\therefore \lim_{n\to\infty}\mathrm{Var}(U_{(n)})=0,
\quad
\lim_{n\to\infty}E(U_{(n)})=1
$$
정리 5.2.7로부터 $\mathrm{plim}_{n\to\infty}U_{(n)}=1$

#### 예 5.2.7. 표본분위수의 확률수렴
모집단 분포가 연속이고 누적분포함수 $F(x)$가 순증가함수일 때, 누적확률이 $\alpha$가 되는 값 $F^{-1}(\alpha),(0<\alpha<1)$를 **모집단 $\alpha$ 분위수(quantile)** 라 한다.  

한편, 랜덤표본 $n$개에 기초한 순서통계량을 $X_{(1)}<\cdots<X_{(n)}$이라 하고,
$$
r_n\sim \alpha n,
\quad
\lim_{n\to\infty}\frac{r_n}{n}=\alpha
$$

를 만족하는 자연수 $r_n$에 대해 $X_{(r_n)}$를 **표본분위수(sample quantile)** 라 한다.
> **각주 75:** 이 정의는 표본크기가 충분히 큰 경우에 사용하는 정의이며, 일반적인 표본분위수 정의에서도 모집단 분위수로의 확률수렴이 성립한다.

이때 표본분위수의 분포는 정리4.3.4에 의해 다음과 같이 나타낼 수 있다.
$$
X_{(r_n)}
\overset{d}{\equiv}
h\left(
\frac{1}{n}Z_1+\cdots+\frac{1}{n-r_n+1}Z_{r_n}
\right),
\quad
Z_i\overset{iid}{\sim}\mathrm{Exp}(1)
$$

여기서
$$
h(y)=F^{-1}(1-e^{-y}),\quad y>0
$$

표준지수분포의 평균과 분산이 $1$이므로
$$
Y_n
=
\frac{1}{n}Z_1+\cdots+\frac{1}{n-r_n+1}Z_{r_n}
$$
에 대해

$$
E(Y_n)=\frac{1}{n}+\cdots+\frac{1}{n-r_n+1},
\quad
\mathrm{Var}(Y_n)=\frac{1}{n^2}+\cdots+\frac{1}{(n-r_n+1)^2}
$$

이 합들을 적분으로 근사하면
$$
E(Y_n)\sim \int_0^\alpha \frac{1}{1-x}\,dx=-\log(1-\alpha)
$$

$$
\mathrm{Var}(Y_n)\sim \int_0^\alpha \frac{1}{n(1-x)^2}\,dx=\frac{1}{n}\cdot\frac{\alpha}{1-\alpha}
$$
따라서
$$
\lim_{n\to\infty}\mathrm{Var}(Y_n)=0,
\quad
\lim_{n\to\infty}E(Y_n)=-\log(1-\alpha)
$$
이고, 정리 5.2.7에 의해
$$
\mathrm{plim}_{n\to\infty}Y_n=-\log(1-\alpha)
$$

함수 $h$가 연속이므로 정리 5.2.5에 의해
$$
\mathrm{plim}_{n\to\infty}
h(Y_n)
=
h(-\log(1-\alpha))
=F^{-1}(\alpha)
$$
즉,
$$
\mathrm{plim}_{n\to\infty}X_{(r_n)}=F^{-1}(\alpha)
$$


## 5.3 극한분포의 계산 *(Computation of Limiting Distributions)*
표본분포의 극한(점근)분포를 실제 통계량에 적용하여 계산하는 방법을 다룬다. 중심극한정리, 확률수렴, 연속함수 정리, 그리고 슬럿츠키(Slutsky)의 정리가 핵심 도구로 사용된다.

### 5.3.1 슬럿츠키(Slutsky)의 정리
슬럿츠키의 정리는 서로 다른 수렴 형태(분포수렴, 확률수렴)를 보이는 확률변수열을 조합한 새로운 통계량의 극한분포를 구할 때 매우 유용하다.

**정리:**  
확률변수열 $X_n, Y_n$과 확률변수 $Z$, 상수 $c$에 대해
$$
X_n \xrightarrow{d} Z, \qquad \mathrm{plim}_{n\to\infty} Y_n = c
$$
이면 다음이 성립한다.

- (a) $X_n + Y_n \xrightarrow{d} Z + c$
- (b) $X_n - Y_n \xrightarrow{d} Z - c$
- (c) $Y_n X_n \xrightarrow{d} cZ$
- (d) $X_n / Y_n \xrightarrow{d} Z / c \quad (c \neq 0)$

**설명:**  
$X_n$이 분포수렴하고 $Y_n$이 상수로 확률수렴하면, 이들의 합, 차, 곱, 몫 역시 각각의 연산에 맞는 극한분포로 수렴한다. 

#### 증명
이 정리는 일반적인 경우에 성립하지만, Z의 누적분포함수가 연속함수인 경우만 증명하기로 한다. 증명은 분포함수의 연속성과 확률수렴의 정의를 이용한다.  

**(a) $X_n + Y_n \xrightarrow{d} Z + c$**

$X_n \xrightarrow{d} Z$, $Y_n \xrightarrow{P} c$라 하자. $F_Z$가 연속인 점 $z$에 대해 $P(X_n + Y_n \le z)$의 극한을 구한다.

임의의 $\varepsilon > 0$에 대해
$$
P(X_n \le z - c - \varepsilon) \le P(X_n + Y_n \le z) \le P(X_n \le z - c + \varepsilon) + P(|Y_n - c| \ge \varepsilon)
$$
$Y_n \xrightarrow{P} c$이므로 $P(|Y_n - c| \ge \varepsilon) \to 0$, $X_n \xrightarrow{d} Z$이므로 $P(X_n \le x) \to P(Z \le x)$이다. 따라서
$$
\limsup_{n \to \infty} P(X_n + Y_n \le z) \le P(Z \le z - c + \varepsilon)
$$
$$
\liminf_{n \to \infty} P(X_n + Y_n \le z) \ge P(Z \le z - c - \varepsilon)
$$
$F_Z$가 연속이므로 $\varepsilon \downarrow 0$으로 보내면
$$
\lim_{n \to \infty} P(X_n + Y_n \le z) = P(Z \le z - c) = P(Z + c \le z)
$$
즉, $X_n + Y_n \xrightarrow{d} Z + c$가 성립한다.

**(b) $X_n - Y_n \xrightarrow{d} Z - c$**

$Y_n \xrightarrow{P} c$이면 $-Y_n \xrightarrow{P} -c$이다. (a)를 $X_n + (-Y_n)$에 적용하면 $X_n - Y_n \xrightarrow{d} Z - c$가 성립한다.

**(c) $Y_n X_n \xrightarrow{d} cZ$**

먼저 $c=0$인 경우, 임의의 $\varepsilon > 0$, $k > 0$에 대해
$$
P(|Y_n X_n| \ge \varepsilon) \le P(|X_n| > k) + P(|Y_n| \ge \varepsilon / k)
$$
$X_n \xrightarrow{d} Z$이므로 $\limsup_{n \to \infty} P(|X_n| > k) \le P(|Z| > k)$, $Y_n \xrightarrow{P} 0$이므로 $P(|Y_n| \ge \varepsilon / k) \to 0$. $k \to \infty$로 보내면 $P(|Y_n X_n| \ge \varepsilon) \to 0$, 즉 $Y_n X_n \xrightarrow{P} 0$.

일반 $c$에 대해 $Y_n X_n = (Y_n - c) X_n + c X_n$. 위 결과와 (a)에 의해 $(Y_n - c) X_n \xrightarrow{P} 0$, $c X_n \xrightarrow{d} cZ$이므로 $Y_n X_n \xrightarrow{d} cZ$.

**(d) $X_n / Y_n \xrightarrow{d} Z / c\ (c \neq 0)$**

$Y_n \xrightarrow{P} c$이고 $c \neq 0$이므로 $1 / Y_n \xrightarrow{P} 1 / c$. (c)를 $X_n$과 $1 / Y_n$에 적용하면 $X_n / Y_n \xrightarrow{d} Z / c$가 성립한다.  
$\square$

#### 5.3.1 예시: 스튜던트화된 표본평균의 극한분포
모집단 평균 $\mu$, 분산 $\sigma^2$인 랜덤표본 $X_1,\dots,X_n$에 대해 표본평균이 $\bar X_n$, 표본표준편차가 $S_n$라 하면, **스튜던트화된 표본평균**은 다음과 같다:
$$
T_n = \frac{\bar X_n - \mu}{S_n / \sqrt{n}}
$$

- **중심극한정리:** 에 의해  
    $$
    \frac{\bar X_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} N(0,1)
    $$
- 예 5.2.5 에 의해  
    $$
    \mathrm{plim}_{n\to\infty} S_n = \sigma
    $$

따라서 슬럿츠키의 정리에 의해
$$
\frac{\bar X_n - \mu}{S_n / \sqrt{n}} = \frac{\bar X_n - \mu}{\sigma / \sqrt{n}}/\frac{S_n}{\sigma} \xrightarrow{d} Z/1 = Z, Z\sim N(0,1)
$$

즉, 정규모집단이 아니더라도 표본크기가 커지면 $t$분포가 표준정규분포로 근사된다.

특히, 모집단 분포가 정규분포 $N(\mu, \sigma^2)$인 경우에는
$$
T_n = \frac{\bar X_n - \mu}{S_n / \sqrt{n}} \sim t_{n-1}
$$
이므로, 자유도가 충분히 크면 $t_{n-1}$ 분포는 표준정규분포 $N(0,1)$로 근사된다. 즉,
$$
t_{n-1} \approx N(0,1),\quad n \to \infty
$$

또한 모집단의 분포가 무엇이든 $\sigma^2 > 0$만 정의되면
$$
\lim_{n \to \infty} P\left(-z_{\alpha/2} < T_n \le z_{\alpha/2}\right) = 1 - \alpha
$$
따라서
$$
P\left(\bar X_n - z_{\alpha/2} \frac{S_n}{\sqrt{n}} < \mu < \bar X_n + z_{\alpha/2} \frac{S_n}{\sqrt{n}}\right) \approx 1 - \alpha,\quad n \to \infty
$$
즉, 모평균 $\mu$에 대한 $100(1-\alpha)\%$ **정근(asymptotic) 신뢰구간**은
$$
\left(\bar X_n - z_{\alpha/2} \frac{S_n}{\sqrt{n}},\;\; \bar X_n + z_{\alpha/2} \frac{S_n}{\sqrt{n}}\right)
$$
로 주어진다.

#### 5.3.2 예시: 표본분산의 극한분포
모평균이 $\mu$, 모표준편차가 $\sigma\ (0<\sigma<+\infty)$인 모집단에서 랜덤표본 $X_1,\dots,X_n$을 추출할 때, 표본분산 $S_n^2$의 극한분포를 구해보자. 표본평균을 $\bar X_n$이라 하면
$$
S_n^2 = \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar X_n)^2 
= \frac{1}{n-1} \sum_{i=1}^n \left\{ (X_i - \mu) - (\bar X_n - \mu) \right\}^2 \\
= \frac{1}{n-1} \left\{ \sum_{i=1}^n (X_i - \mu)^2 - n(\bar X_n - \mu)^2 \right\} 
= \frac{n}{n-1} \left\{ \frac{1}{n} \sum_{i=1}^n (X_i - \mu)^2 - (\bar X_n - \mu)^2 \right\}
$$

이제 $Y_i = (X_i - \mu)^2$라 하면 $Y_1, \dots, Y_n$은 서로 독립이고 동일한 분포를 따르며
$$
E(Y_1) = E[(X_1 - \mu)^2] = \sigma^2
$$
이다. $E[(X_1 - \mu)^4] < +\infty$라 가정하면
$$
\operatorname{Var}(Y_1) = E[(X_1 - \mu)^4] - \sigma^4
$$
이므로 중심극한정리에 의해
$$
\sqrt{n} \left( \frac{1}{n} \sum_{i=1}^n Y_i - \sigma^2 \right)
= \sqrt{n} \left( \frac{1}{n} \sum_{i=1}^n (X_i - \mu)^2 - \sigma^2 \right)
\xrightarrow{d} W,\quad W \sim N\left(0,\, E[(X_1 - \mu)^4] - \sigma^4\right)
$$

한편, 중심극한정리로부터
$$
\sqrt{n}(\bar X_n - \mu) \xrightarrow{d} Z,\quad Z \sim N(0, \sigma^2)
$$
이고, 큰수의 법칙에 의해
$$
\bar X_n - \mu \xrightarrow{P} 0
$$
이므로 슬럿츠키의 정리에 의해
$$
\sqrt{n} (\bar X_n - \mu)^2 \xrightarrow{P} 0
$$

따라서
$$
\sqrt{n} \left( \frac{1}{n} \sum_{i=1}^n (X_i - \bar X_n)^2 - \sigma^2 \right)
= \sqrt{n} \left( \frac{1}{n} \sum_{i=1}^n (X_i - \mu)^2 - \sigma^2 \right)
- \sqrt{n} (\bar X_n - \mu)^2
\xrightarrow{d} W
$$

$S_n^2$와 $\frac{1}{n} \sum_{i=1}^n (X_i - \bar X_n)^2$의 차이는 계수 $\frac{n}{n-1}$에서만 발생한다. 실제로
$$
S_n^2 = \frac{n}{n-1} \cdot \frac{1}{n} \sum_{i=1}^n (X_i - \bar X_n)^2
$$
이므로
$$
S_n^2 - \sigma^2
= \left( \frac{1}{n} \sum_{i=1}^n (X_i - \bar X_n)^2 - \sigma^2 \right)
+ \left( \frac{n}{n-1} - 1 \right) \frac{1}{n} \sum_{i=1}^n (X_i - \bar X_n)^2
$$
여기서 $\frac{n}{n-1} - 1 = \frac{1}{n-1}$이므로
$$
\sqrt{n}(S_n^2 - \sigma^2)
= \sqrt{n} \left( \frac{1}{n} \sum_{i=1}^n (X_i - \bar X_n)^2 - \sigma^2 \right)
+ \frac{\sqrt{n}}{n-1} \cdot \frac{1}{n} \sum_{i=1}^n (X_i - \bar X_n)^2
$$
$\frac{\sqrt{n}}{n-1} \to 0$이고, $S_n^2$가 확률적으로 $\sigma^2$에 수렴하므로
$$
\frac{1}{\sqrt{n}} S_n^2 \xrightarrow{P} 0
$$
따라서
$$
\sqrt{n}(S_n^2 - \sigma^2) \xrightarrow{d} W,\quad
W \sim N\left(0,\, E[(X_1 - \mu)^4] - \sigma^4\right)
$$

이제 $E[(X_1 - \mu)^4]$를 첨도 $\rho_4$로 표현하면, $\rho_4 = E\left[ \left( \frac{X_1 - \mu}{\sigma} \right)^4 \right] - 3$이므로
$$
E[(X_1 - \mu)^4] = \sigma^4 (\rho_4 + 3)
$$
따라서
$$
E[(X_1 - \mu)^4] - \sigma^4 = (\rho_4 + 2)\sigma^4
$$
결국
$$
\sqrt{n}(S_n^2 - \sigma^2) \xrightarrow{d} N\left(0,\, (\rho_4 + 2)\sigma^4\right)
$$

여기서 $\rho_4$는 모집단의 **첨도(kurtosis)** 이며, 정규분포에서는 $\rho_4 = 0$이다.

### 정리 5.3.2 연속함수와 극한분포
**정리:**  
다차원 확률변수 $X_n \xrightarrow{d} Z$이고, 함수 $g$가 연속이면
$$
g(X_n) \xrightarrow{d} g(Z)
$$

**설명:**  
분포수렴하는 확률변수에 연속함수를 적용해도 분포수렴이 보존된다. 이는 복잡한 통계량의 극한분포를 구할 때 매우 유용하다.

증명은 이 책의 수준을 넘으므로 생략.  

#### 5.3.3 예시: 이항분포와 카이제곱 근사
이항분포 $X_n \sim B(n, p)$에 대해, 중심극한정리에 의해 표준화된 이항변수는
$$
\frac{X_n - np}{\sqrt{np(1-p)}} \xrightarrow{d} N(0,1)
$$
이제 연속함수 정리(정리 5.3.2)에 따라, 연속함수 $g(x) = x^2$를 적용하면
$$
g\left(\frac{X_n - np}{\sqrt{np(1-p)}}\right) = \frac{(X_n - np)^2}{np(1-p)} \xrightarrow{d} \chi^2(1)
$$
즉, 이항분포의 제곱형 통계량은 자유도 1인 카이제곱분포로 근사된다.

이 결과는 이항검정, 적합도검정 등에서 카이제곱 근사에 이론적 근거를 제공한다.

#### 예 5.3.4 다항분포와 카이제곱근사
확률벡터
$$
(X_{n1},\dots,X_{nk})^t
$$
가 모수 $(p_1,\dots,p_k)$ $(p_j>0,\ \sum_{j=1}^k p_j=1)$를 갖는 다항분포 $\mathrm{Multinomial}(n,p_1,\dots,p_k)$를 따른다고 하자. 즉,
$$
P(X_{n1}=x_1,\dots,X_{nk}=x_k)
=
\frac{n!}{x_1!\cdots x_k!}p_1^{x_1}\cdots p_k^{x_k},
\quad
\sum_{j=1}^k x_j=n
$$
각 성분에 대해 $E(X_{nj})=np_j, \mathrm{Var}(X_{nj})=np_j(1-p_j), \mathrm{Cov}(X_{ni},X_{nj})=-np_ip_j\quad (i\neq j)$
가 성립한다.

다음과 같은 벡터를 고려하자.
$$
Z_n
=
\left(
\frac{X_{n1}-np_1}{\sqrt n},\dots,
\frac{X_{nk}-np_k}{\sqrt n}
\right)^t.
$$
다차원 중심극한정리에 의해 $Z_n\xrightarrow{d}Z$
여기서 $Z$는 평균이 0이고 분산–공분산 행렬이
$$
\Sigma
=
\begin{pmatrix}
p_1(1-p_1) & -p_1p_2 & \cdots & -p_1p_k \\
-p_2p_1 & p_2(1-p_2) & \cdots & -p_2p_k \\
\vdots & \vdots & \ddots & \vdots \\
-p_kp_1 & -p_kp_2 & \cdots & p_k(1-p_k)
\end{pmatrix}
$$
인 $k$차원 정규분포를 따른다.

또한 $\sum_{j=1}^k X_{nj}=n$이 항상 성립하므로
$$
\sum_{j=1}^k Z_{nj}=0
$$
이고, 따라서 $\Sigma$는 계수(rank)가 $k-1$인 특이행렬이다.

이제 다음 통계량을 고려하자.
$$
Q_n
=
\sum_{j=1}^k \frac{(X_{nj}-np_j)^2}{np_j}.
$$
이를 $Z_n$을 이용하여 쓰면
$$
Q_n
=
Z_n^t D^{-1} Z_n,
\quad
D=\mathrm{diag}(p_1,\dots,p_k).
$$

위에서 본 바와 같이 $Z_n\xrightarrow{d}Z$이고, $Z$는 평균이 0이며 공분산행렬이 $\Sigma$인 정규분포를 따른다. 또한 $D^{-1}$는 상수 행렬이므로 연속함수 정리에 의해
$$
Q_n \xrightarrow{d} Q=Z^t D^{-1} Z.
$$

이때 $Z$는 선형 제약조건 $\sum_{j=1}^k Z_j=0$을 만족하므로, $Q$는 자유도 $k-1$인 카이제곱분포를 따른다. 즉,
$$
Q\sim \chi^2(k-1).
$$

따라서
$$
\sum_{j=1}^k \frac{(X_{nj}-np_j)^2}{np_j}
\xrightarrow{d}
\chi^2(k-1)
$$
가 성립한다.

이 결과는 다항분포에 기초한 적합도 검정에서 사용되는 카이제곱 근사의 이론적 근거를 제공한다.

### 정리 5.3.3 일차근사를 이용한 극한분포 계산: 델타 방법 (Delta Method)
다차원 확률변수 $X_n$에 대해
$$
\sqrt{n}(X_n - \theta) \xrightarrow{d} Z
$$
이고, 함수 $g$가 $\theta$에서 미분가능(일차원에서는 도함수 $g'(\theta)$, 다차원에서는 그레디언트 $\nabla g(\theta)$ 존재)하면
$$
\sqrt{n}(g(X_n) - g(\theta)) \xrightarrow{d} g'(\theta) Z
$$
(다차원에서는 $g'(\theta)Z = \nabla g(\theta)^T Z$)

**설명:**  
델타 방법(Delta Method)은 중심극한정리로부터 얻은 표본평균 등 통계량의 극한분포를, 미분가능한 함수 $g$에 적용한 새로운 통계량의 극한분포로 확장하는 데 사용된다.  
함수 $g$가 $\theta$에서 미분가능하면, $X_n$이 $\theta$ 근처에서 $g(X_n)$을 1차 테일러 전개로 근사할 수 있다.  
즉,
$$
g(X_n) \approx g(\theta) + g'(\theta)(X_n - \theta)
$$
따라서
$$
\sqrt{n}(g(X_n) - g(\theta)) \approx g'(\theta) \sqrt{n}(X_n - \theta)
$$
$X_n$의 중심화된 극한분포가 $Z$라면, $g(X_n)$의 극한분포는 $g'(\theta)Z$가 된다.

TODO: 왜 $\sqrt n$이 필요할까? 이거 없으면 수렴이 안되나? 다른 n관련함수는 안되는걸까

**증명:**  
(일차원 $X_n$의 경우)  
$g$가 $\theta$에서 미분가능하므로, 임의의 $X_n$에 대해
$$
g(X_n) = g(\theta) + g'(\theta)(X_n - \theta) + r_n
$$
여기서 $r_n = o(|X_n - \theta|)$는 $X_n \to \theta$일 때 $r_n / |X_n - \theta| \to 0$이다.

따라서
$$
\sqrt{n}(g(X_n) - g(\theta)) = g'(\theta)\sqrt{n}(X_n - \theta) + \sqrt{n} r_n
$$
$X_n \xrightarrow{P} \theta$이므로 $|X_n - \theta| \xrightarrow{P} 0$, $r_n = o(|X_n - \theta|)$이므로 $\sqrt{n} r_n = o_P(\sqrt{n}|X_n - \theta|)$.  
또한 $\sqrt{n}(X_n - \theta) \xrightarrow{d} Z$이므로, 슬럿츠키의 정리에 의해 $\sqrt{n} r_n \xrightarrow{P} 0$이 되어 전체 극한분포는 $g'(\theta)Z$가 된다.

(다차원 $X_n$의 경우도, $g$의 그레디언트 $\nabla g(\theta)$를 사용하여 동일하게 증명된다.)
$\square$

#### 5.3.5 예시: 표본표준편차의 극한분포
표본표준편차 $S_n = \sqrt{S_n^2}$의 극한분포는 델타 방법(Delta Method)을 이용해 구할 수 있다. 앞서 표본분산 $S_n^2$에 대해
$$
\sqrt{n}(S_n^2 - \sigma^2) \xrightarrow{d} N\left(0,\, (\rho_4 + 2)\sigma^4\right)
$$
임을 보였다. 여기서 $\rho_4$는 모집단의 첨도(kurtosis)이다.

이제 $g(x) = \sqrt{x}$를 $x = \sigma^2$에서 미분하면 $g'(\sigma^2) = \frac{1}{2\sqrt{\sigma^2}} = \frac{1}{2\sigma}$이다. 델타 방법에 따라
$$
\sqrt{n}(S_n - \sigma) \approx g'(\sigma^2)\sqrt{n}(S_n^2 - \sigma^2)
$$
따라서
$$
\sqrt{n}(S_n - \sigma) \xrightarrow{d} N\left(0,\, \frac{(\rho_4 + 2)\sigma^4}{4\sigma^2}\right)
= N\left(0,\, \frac{(\rho_4 + 2)\sigma^2}{4}\right)
$$

즉, 표본표준편차 $S_n$의 극한분포는
$$
\sqrt{n}(S_n - \sigma) \xrightarrow{d} Z,\quad Z \sim N\left(0,\, \frac{(\rho_4 + 2)\sigma^2}{4}\right)
$$

**설명:**  
- 표본분산 $S_n^2$의 극한분포를 델타 방법으로 변환하여 표본표준편차 $S_n$의 극한분포를 구한다.
- 분산의 첨도 $\rho_4$가 0인 정규분포의 경우, 분산은 $\sigma^2/2$가 된다.
- 이 결과는 표본표준편차의 신뢰구간 계산 등에서 근사적으로 활용된다.

#### 예 5.3.6 표본상관계수의 극한분포
모평균이 각각 $\mu_1, \mu_2$, 모분산이 각각 $\sigma_1^2, \sigma_2^2$ ($0<\sigma_1<+\infty,\ 0<\sigma_2<+\infty$), 모상관계수가 $\rho$ ($-1<\rho<1$)인 이변량 모집단에서 랜덤표본 $((X_1,Y_1)^T, \dots, (X_n,Y_n)^T)$ ($n>2$)을 추출할 때, 표본상관계수는 다음과 같이 정의된다.

$$
\hat\rho_n = \frac{\sum_{i=1}^n (X_i-\bar X)(Y_i-\bar Y)}{\sqrt{\sum_{i=1}^n (X_i-\bar X)^2}\ \sqrt{\sum_{i=1}^n (Y_i-\bar Y)^2}}
$$

$(X_i, Y_i)$ 대신 각각 $(\frac{X_i-\mu_1}{\sigma_1}, \frac{Y_i-\mu_2}{\sigma_2})$로 치환해도 표본상관계수 값은 변하지 않으므로, $\mu_1=0, \mu_2=0, \sigma_1=1, \sigma_2=1$로 가정해도 무방하다.

또한, 다음과 같이 표기한다.
$$
\bar X_n = \frac{1}{n}\sum_{i=1}^n X_i,\quad
\bar Y_n = \frac{1}{n}\sum_{i=1}^n Y_i,\quad
\overline{(XY)}_n = \frac{1}{n}\sum_{i=1}^n X_i Y_i
$$

$$
\overline{(X^2)}_n = \frac{1}{n}\sum_{i=1}^n X_i^2,\quad
\overline{(Y^2)}_n = \frac{1}{n}\sum_{i=1}^n Y_i^2
$$

함수 $g$를
$$
g(t_1, t_2, t_3, t_4, t_5) = \frac{t_3 - t_1 t_2}{\sqrt{t_4 - t_1^2}\ \sqrt{t_5 - t_2^2}}
$$
로 정의하면,

$$
\hat\rho_n = g(\bar X_n, \bar Y_n, \overline{(XY)}_n, \overline{(X^2)}_n, \overline{(Y^2)}_n)
$$
즉, 표본상관계수는 $(X_i, Y_i, X_iY_i, X_i^2, Y_i^2)^T$의 표본평균의 함수로 나타낼 수 있다.

$Z_i = (X_i, Y_i, X_iY_i, X_i^2, Y_i^2)^T$ ($i=1,\dots,n$)라 하면, $Z_i$는 서로 독립이고 동일한 분포를 따르는 5차원 확률변수이므로 분산행렬이 존재할 때
$$
\sqrt{n}\left(\frac{1}{n}\sum_{i=1}^n Z_i - E(Z_1)\right) \xrightarrow{d} V,\quad V \sim N_5(0, \operatorname{Var}(Z_1))
$$
또한 $E(Z_1) = (0, 0, \rho, 1, 1)^T$, $\rho = g(0,0,\rho,1,1) = g(E(Z_1))$이다.

함수 $g$의 일차편도함수들이 연속이므로, <정리 5.3.3>에 의해
$$
\sqrt{n}(\hat\rho_n - \rho)
= \sqrt{n}\left(g\left(\frac{1}{n}\sum_{i=1}^n Z_i\right) - g(\theta)\right)
\xrightarrow{d} (\dot g(\theta))^T V
$$
여기서 $\theta = E(Z_1) = (0,0,\rho,1,1)^T$이고, $\dot g(\theta) = (0, 0, 1, -\rho/2, -\rho/2)^T$이다.

따라서
$$
(\dot g(\theta))^T V \sim N\left(0, (\dot g(\theta))^T \operatorname{Var}(Z_1) \dot g(\theta)\right)
$$

또는
$$
(\dot g(\theta))^T Z_1 = X_1 Y_1 - \frac{\rho}{2} X_1^2 - \frac{\rho}{2} Y_1^2
$$
이므로 $E(X_1^4)<+\infty$, $E(Y_1^4)<+\infty$일 때
$$
\sqrt{n}(\hat\rho_n - \rho) \xrightarrow{d} W,\quad
W \sim N\left(0, \operatorname{Var}\left(X_1 Y_1 - \frac{\rho}{2} X_1^2 - \frac{\rho}{2} Y_1^2\right)\right)
$$

#### 예 5.3.7 표본상관계수와 분산안정변환 (variance stabilizing transformation)
이변량 정규분포 $N(\mu_1, \mu_2; \sigma_1^2, \sigma_2^2, \rho)$ ($\sigma_1>0, \sigma_2>0, -1<\rho<1$)에서 표본상관계수의 극한분포를 살펴보자.

이 경우 $Y_1 - \rho X_1 \mid X_1 = x_1 \sim N(0, 1-\rho^2)$로, $Y_1 - \rho X_1$과 $X_1$은 조건부분포가 x1에 의존하지 않으므로 서로 독립이다. $\mu_1 = \mu_2 = 0$, $\sigma_1 = \sigma_2 = 1$로 두면

$$
T = \frac{Y_1 - \rho X_1}{\sqrt{1-\rho^2}}
$$

에서 $X_1$과 $T$는 서로 독립이고 각각 $N(0,1)$을 따른다.

$Y_1 = \rho X_1 + \sqrt{1-\rho^2} T$를 표본상관계수의 극한분포식에 대입하면
$$
X_1 Y_1 - \frac{\rho}{2} X_1^2 - \frac{\rho}{2} Y_1^2
= \frac{\rho}{2}(1-\rho^2) X_1^2 + (1-\rho^2)^{3/2} X_1 T - \frac{\rho}{2}(1-\rho^2) T^2
$$
이므로, 계산 과정은 다음과 같다.

먼저 $X_1, Y_1$이 $N(0,1)$이고 $\operatorname{Cov}(X_1, Y_1) = \rho$임을 이용한다.  
$T = X_1 Y_1 - \frac{\rho}{2} X_1^2 - \frac{\rho}{2} Y_1^2$라 하면,

$$
\begin{align*}
E(T) &= E\left[X_1 Y_1 - \frac{\rho}{2} X_1^2 - \frac{\rho}{2} Y_1^2\right] \\
&= E[X_1 Y_1] - \frac{\rho}{2} E[X_1^2] - \frac{\rho}{2} E[Y_1^2] \\
&= \rho - \frac{\rho}{2} \cdot 1 - \frac{\rho}{2} \cdot 1 \\
&= 0
\end{align*}
$$

분산은
$$
\operatorname{Var}(T) = E[T^2] - (E[T])^2 = E[T^2]
$$

$E[T^2]$를 전개하면,
$$
\begin{align*}
E[T^2] &= E\left[ \left( X_1 Y_1 - \frac{\rho}{2} X_1^2 - \frac{\rho}{2} Y_1^2 \right)^2 \right] \\
&= E[X_1^2 Y_1^2] - \rho E[X_1^2 Y_1^2] - \rho E[X_1 Y_1^2 X_1] \\
&\quad + \frac{\rho^2}{4} E[X_1^4] + \rho^2 E[X_1^2 Y_1^2] + \frac{\rho^2}{4} E[Y_1^4]
\end{align*}
$$

표준정규분포의 적률과 공분산 성질을 이용하여 계산하면, 최종적으로
$$
\operatorname{Var}\left(X_1 Y_1 - \frac{\rho}{2} X_1^2 - \frac{\rho}{2} Y_1^2\right) = (1-\rho^2)^2
$$
임을 얻는다.

따라서
$$
\sqrt{n}(\hat\rho_n - \rho) \xrightarrow{d} W,\quad W \sim N(0, (1-\rho^2)^2)
$$

이제 $\hat\rho_n$의 함수 $g(\hat\rho_n)$의 극한분포를 구하면, <정리 5.3.3>에 의해
$$
\sqrt{n}\big(g(\hat\rho_n) - g(\rho)\big) \xrightarrow{d} \dot g(\rho) W,\quad
\dot g(\rho) W \sim N\left(0, [\dot g(\rho)]^2 (1-\rho^2)^2\right)
$$

이 때, 극한분포의 분산이 $\rho$에 의존하지 않도록 하는 $g(\hat\rho_n)$을 $\hat\rho_n$의 분산안정변환이라 한다. 특히
$$
g(\hat\rho_n) = \frac{1}{2} \log\frac{1+\hat\rho_n}{1-\hat\rho_n}
$$
은 피셔 변환(Fisher transformation)이라 하며, $[\dot g(\rho)]^2 (1-\rho^2)^2 = 1$이 되어 극한분포가 $N(0,1)$이 된다. 즉,
$$
\sqrt{n}\big(g(\hat\rho_n) - g(\rho)\big) \xrightarrow{d} Z,\quad Z \sim N(0,1)
$$

#### 예 5.3.8 표본분위수의 극한분포
<예 5.2.7>에서 표본분위수 $X_{(r_n)}$ ($r_n \sim \alpha n,\ 0<\alpha<1$)의 확률수렴을 다루었다. 여기서는 표본분위수의 극한분포를 살펴본다.

모집단 누적분포함수의 역함수를 $F^{-1}$라 하고,
$$
h(y) = F^{-1}(1 - e^{-y}),\quad y > 0
$$
라 하면, <정리 4.3.4>에 의해
$$
X_{(r_n)} \overset{d}{\equiv}
 h\left(\frac{1}{n} Z_1 + \cdots + \frac{1}{n - r_n + 1} Z_{r_n}\right),\quad Z_i \overset{iid}{\sim} \operatorname{Exp}(1)
$$
한편, $Y_n = \frac{1}{n} Z_1 + \cdots + \frac{1}{n - r_n + 1} Z_{r_n}$의 평균과 분산은

$$
E(Y_n) \sim -\log(1-\alpha),\qquad
\operatorname{Var}(Y_n) \sim \frac{1}{n} \frac{\alpha}{1-\alpha}
$$

따라서
$$
W_n = \sqrt{n}\ \frac{Y_n + \log(1-\alpha)}{\sqrt{\alpha/(1-\alpha)}}
$$
에 대해 적률생성함수를 근사해보면
$$
W_n \xrightarrow{d} N(0,1)
$$

즉,
$$
\sqrt{n}\big(Y_n + \log(1-\alpha)\big) \xrightarrow{d} \sqrt{\frac{\alpha}{1-\alpha}}\, W,\quad W \sim N(0,1)
$$

함수 $h$가 미분가능할 때 <정리 5.3.3>에 의해
$$
\sqrt{n}\big(h(Y_n) - h(-\log(1-\alpha))\big) \xrightarrow{d} \dot h(-\log(1-\alpha)) \sqrt{\frac{\alpha}{1-\alpha}}\, W
$$

여기서
$$
\dot h(-\log(1-\alpha)) = \frac{1-\alpha}{f(F^{-1}(\alpha))},\quad f = F'
$$
이므로
$$
\sqrt{n}\big(X_{(r_n)} - F^{-1}(\alpha)\big) \xrightarrow{d} Z,\quad
Z \sim N\left(0, \frac{\alpha(1-\alpha)}{[f(F^{-1}(\alpha))]^2}\right)
$$

## 모의실험을 이용한 근사 *(Simulation-Based Approximations)*
이 절에서는 랜덤표본의 관측값을 이용하여 통계량의 분포를 근사하거나 그 특성을 연구하는 방법을 다룬다. 특정한 분포를 따르는 확률변수의 관측값을 흔히 **난수(random number)** 라고 하며, 최근에는 이러한 난수를 생성할 수 있는 기능이 많은 통계 패키지에 제공되어 있다.

예를 들어, 확률변수 $X$가 특정 분포 $F$를 따른다면, $X$의 관측값 $x_1, x_2, \dots, x_n$을 난수라고 한다. 난수 생성기는 주어진 분포 $F$에 따라 난수 $x_i$를 생성하며, 이를 이용해 통계량의 분포를 근사하거나 모의실험(simulation)을 수행할 수 있다.

특히, 난수 $U$가 균등분포 $U(0,1)$를 따를 때, 임의의 분포 $F$의 난수 $X$는 다음과 같이 생성할 수 있다.

$$
X = F^{-1}(U)
$$

여기서 $F^{-1}$은 분포 $F$의 누적분포함수의 역함수이다. 이 방법을 **역변환법(inverse transform method)** 이라고 한다.

이러한 난수 생성과 모의실험을 통해, 표본분포의 근사, 신뢰구간 계산, 통계적 검정의 특성 평가 등 다양한 통계적 분석이 가능하다.

이러한 난수 생성의 기본이 되는 것은 균등분포 (U(0,1))에서의 관측값으로, 이를 **균등난수(uniform random number)** 라고 한다. 균등난수로부터 확률적 변환에 관한 정리(정리 4.3.3)를 이용하면 임의의 분포로부터의 난수를 생성할 수 있다.

#### 예 5.4.1 로지스틱분포에서의 난수 생성
로지스틱분포 (L(0,1))의 확률밀도함수와 누적분포함수는 각각
$$
f(z)=\frac{e^z}{(1+e^z)^2}, \qquad
F(z)=1-\frac{1}{1+e^z}, \quad -\infty<z<\infty
$$
로 주어진다.
이로부터 누적분포함수의 역함수는
$$
F^{-1}(u) = \log\frac{u}{1-u}
$$
임을 알 수 있다. 따라서 정리 4.3.3에 의해 균등분포 $U(0,1)$를 따르는 확률변수 $U$에 대해
$$
Z = \log\frac{U}{1-U} \sim L(0,1)
$$
이 된다.

이를 이용하면 일반적인 로지스틱분포 $L(\mu,\sigma)$에 대해서도
$$
\sigma Z + \mu = \sigma\log\frac{U}{1-U} + \mu \sim L(\mu,\sigma)
$$
와 같이 난수를 생성할 수 있다.

같은 방식으로, 균등분포 $U(0,1)$를 따르는 확률변수 $U$에 대해
$$
-\log(1-U) \sim \mathrm{Exp}(1), \qquad
\sigma(-\log(1-U)) \sim \mathrm{Exp}(\sigma)
$$
임을 이용하면 지수분포의 난수 생성이 가능하며, 또한 표준정규분포의 누적분포함수 $\Phi$의 역함수 $\Phi^{-1}(u)$를 이용하여
$$
\Phi^{-1}(U) \sim N(0,1), \qquad
\sigma\Phi^{-1}(U) + \mu \sim N(\mu, \sigma^2)
$$
와 같이 정규분포의 난수도 생성할 수 있다.

### 정리 5.4.1 난수를 이용한 정적분의 근사
서로 독립이고 균등분포 $U(0,1)$를 따르는 $U_1, \dots, U_n$과 구간 $[a, b]$에서 연속인 함수 $g(x)$에 대하여 다음이 성립한다.
$$
X_i = (b-a)U_i + a \sim U(a, b), \quad i = 1, \dots, n
$$
일 때,
$$
\mathrm{plim}_{n \to \infty} \frac{b-a}{n} \sum_{i=1}^n g(X_i)
= \int_a^b g(x)\, dx
$$
이는 $X_1, \dots, X_n$이 서로 독립이고 균등분포 $U(a, b)$를 따르므로
$$
E[g(X_1)] = \frac{1}{b-a} \int_a^b g(x)\, dx
$$
이고, 큰수의 법칙에 의해
$$
\frac{1}{n} \sum_{i=1}^n g(X_i) \xrightarrow{P} E[g(X_1)]
$$
이기 때문이다.

이 정리를 이용하여 정적분의 근삿값을 구하는 방법을 **몬테카를로 적분(Monte Carlo integration)** 이라고 한다. 즉, 서로 독립적으로 생성된 균등난수 $u_1, \dots, u_n$에 대해
$$
x_i = (b-a)u_i + a, \qquad y_i = (b-a)g(x_i)
$$
로 두면
$$
\int_a^b g(x)\, dx \approx \frac{1}{n} \sum_{i=1}^n y_i
$$
가 된다.

이는 $Y_i$의 관측값을 이용하여 $E(Y_1)$을 추정하는 것이므로, 이에 대한 점근신뢰구간을 통해 근사의 정확도를 평가할 수 있다.

#### 예 5.4.2 정적분의 몬테칼로 근사
$$
\int_1^3 x^2\,dx = \frac{26}{3}
$$
임은 잘 알려져 있다. 독립적으로 생성된 $U(0,1)$ 난수 $u_1,\dots,u_n$에 대해
$$
x_i = 2u_i + 1, \qquad y_i = 2x_i^2
$$
로 두면
$$
\int_1^3 x^2\,dx \approx \frac{1}{n}\sum_{i=1}^n y_i
$$
로 근사할 수 있다.

난수의 개수 $n$이 커질수록 이러한 근사의 정밀도가 증가함을 수치 결과를 통해 확인할 수 있다.

#### 예 5.4.3 로지스틱분포 (L(0,1))의 분산
로지스틱분포 $L(0,1)$은 $x=0$에 대해 대칭인 분포이며, 분산은 다음과 같이 주어진다.
$$
\operatorname{Var}(X) = \int_{-\infty}^{\infty} x^2\, f(x)\, dx = \int_{-\infty}^{\infty} \frac{x^2 e^x}{(1 + e^x)^2}\, dx = \frac{\pi^2}{3}
$$

정리 5.4.1을 이용하면, 균등분포 $U(0,1)$ 난수 $u_1, \dots, u_n$에 대해
$$
x_i = \log\frac{u_i}{1-u_i}, \qquad y_i = x_i^2
$$
로 두고,

$$
\frac{1}{n} \sum_{i=1}^n y_i \approx E(X^2) = \frac{\pi^2}{3}
$$
로 근사할 수 있다.

이 방법을 통해 로지스틱분포의 분산 근삿값과 점근적 신뢰구간을 계산할 수 있다.

#### 예 5.4.4 표본비율의 극한분포
모비율이 $p$ $(0 < p < 1)$인 베르누이 시행을 독립적으로 $n$번 관측한 결과를 $X_1, \dots, X_n$이라 하면, 표본비율
$$
\hat{p}_n = \frac{1}{n} \sum_{i=1}^n X_i
$$
에 대해 큰수의 법칙과 중심극한정리에 의해
$$
\mathrm{plim}_{n \to \infty} \hat{p}_n = p, \qquad
\frac{\hat{p}_n - p}{\sqrt{p(1-p)/n}} \xrightarrow{d} N(0, 1)
$$
이 성립한다.

슬럿츠키의 정리와 연속함수 정리(정리 5.3.2)를 이용하면
$$
\frac{\hat{p}_n - p}{\sqrt{\hat{p}_n (1 - \hat{p}_n) / n}} \xrightarrow{d} N(0, 1)
$$
이 되며, 이를 통해 모비율 $p$에 대한 $100(1-\alpha)\%$ 점근 신뢰구간
$$
p \in \left[
\hat{p}_n - z_{\alpha/2} \sqrt{\frac{\hat{p}_n (1 - \hat{p}_n)}{n}},
\;
\hat{p}_n + z_{\alpha/2} \sqrt{\frac{\hat{p}_n (1 - \hat{p}_n)}{n}}
\right]
$$
을 얻는다.

#### 예 5.4.5 표본비율의 극한분포를 이용한 원주율 $\pi$의 근사
서로 독립이고 균등분포 $U(0,1)$를 따르는 $U_1, U_2$에 대해 점 $(U_1, U_2)$는 단위정사각형 내부에 무작위로 떨어진 점으로 생각할 수 있다.

이 점이 사분원 내부에 떨어질 확률은
$$
P(U_1^2 + U_2^2 \le 1) = \frac{\pi}{4}
$$
이다.

이를 독립적으로 반복하여 $n$개의 점을 생성하고
$$
X_i = I(U_{i1}^2 + U_{i2}^2 \le 1)
$$
이라 두면, 표본비율
$$
\hat{p}_n = \frac{1}{n} \sum_{i=1}^n X_i
$$
은 $\pi/4$의 근삿값이 되며, $4\hat{p}_n$은 $\pi$의 근삿값으로 사용될 수 있다.

또한 예 5.4.4에서 얻은 점근 신뢰구간을 이용하여 $\pi$에 대한 근사의 정확도를 평가할 수 있다.