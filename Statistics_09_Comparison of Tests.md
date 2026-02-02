# 제9장 검정의 비교 *(Comparison of Tests)*

## 검정법 비교의 기준 *(Criteria for Comparing Tests)*
**단순 가설(simple hypothesis)**: 귀무가설과 대립가설이 각각 하나의 확률밀도함수로 주어지므로, 검정 방법의 비교 기준을 이해하기 쉽다.  
- 모집단 분포: 확률밀도함수 $f(x;\theta)$
- 모수 공간: $\theta \in \Omega = \{\theta_0, \theta_1\}$
- 랜덤표본: $X = (X_1, \dots, X_n)$

검정 문제:
$$
H_0: \theta = \theta_0 \quad \text{vs} \quad H_1: \theta = \theta_1
$$

**검정과 오류 확률**  
**(1) 비랜덤화 검정**  
기각역 $C \subset \mathcal{X}^n$에 대해
$$
X \in C \Rightarrow H_0 \text{ 기각}
$$
오류 확률:
- 제1종 오류: $P_{\theta_0}(X \in C)$
- 제2종 오류: $P_{\theta_1}(X \notin C) = 1 - P_{\theta_1}(X \in C)$

**(2) 랜덤화 검정**  
검정 함수:
$$
\phi(X) = \phi(X_1,\dots,X_n), \quad 0 \le \phi(X) \le 1
$$
- $\phi(X)$: 귀무가설을 기각할 확률

오류 확률:
- 제1종 오류: $E_{\theta_0}[\phi(X)]$
- 제2종 오류: $E_{\theta_1}[1 - \phi(X)]$

**검정 비교의 필요성**  
두 오류 확률을 동시에 작게 만드는 검정이 이상적이지만, 한쪽 오류 확률을 줄이면 다른 쪽 오류 확률이 커지는 trade-off가 존재한다. 따라서 오류를 종합적으로 평가하는 기준이 필요하다.

### 최대오류확률과 베이지안 평균오류확률 *(Maximum Error Probability and Bayesian Average Error Probability)*
**(a) 최대오류확률 기준 *(Maximum Error Probability Criterion)*** 
$$
\max\left\{ E_{\theta_0}[\phi(X)],\; E_{\theta_1}[1-\phi(X)] \right\}
$$
→ 값이 작을수록 좋은 검정 (The smaller the value, the better the test)

**(b) 베이지안 평균오류확률 기준 *(Bayesian Average Error Probability Criterion)***
$$
\pi_0 E_{\theta_0}[\phi(X)] + \pi_1 E_{\theta_1}[1-\phi(X)]
$$
- 가중치: $\pi_0 + \pi_1 = 1,\quad \pi_0 > 0,\; \pi_1 > 0$
- $(\pi_0, \pi_1)$: 사전확률(prior probability)

### 정리 9.1.1 단순 가설의 베이즈 검정
단순 가설 $H_0:\theta=\theta_0 \quad\text{vs}\quad H_1:\theta=\theta_1$
에 대해, 다음 검정
$$
\phi^\pi(x) =
\begin{cases}
1, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} > \dfrac{\pi_0}{\pi_1} \\
0, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} < \dfrac{\pi_0}{\pi_1}
\end{cases}
$$
을 만족하는 검정은 베이지안 평균오류확률을 최소로 한다.

#### 증명
베이지안 평균오류확률은
$$
\pi_0 E_{\theta_0}[\phi(X)] + \pi_1 E_{\theta_1}[1-\phi(X)]
= \int \left[ \pi_0 \phi(x) pdf(x;\theta_0) + \pi_1 (1-\phi(x)) pdf(x;\theta_1) \right] dx
$$
이를 $\phi(x)$에 대해 최소화하려면, 각 $x$에 대해 integrand
$$
\pi_0 \phi(x) pdf(x;\theta_0) + \pi_1 (1-\phi(x)) pdf(x;\theta_1)
$$
을 최소화하는 $\phi(x)$를 선택하면 된다.

$\phi(x)$는 $[0,1]$ 사이의 값(랜덤화 검정)도 가능하지만, integrand가 $\phi(x)$에 대해 선형이므로, 극값은 $\phi(x)=0$ 또는 $\phi(x)=1$에서만 달성된다(비랜덤화 검정).

- $\phi(x)=1$일 때: 값은 $\pi_0 pdf(x;\theta_0)$
- $\phi(x)=0$일 때: 값은 $\pi_1 pdf(x;\theta_1)$

따라서 $\phi(x)=1$이 더 작으려면
$$
\pi_0 pdf(x;\theta_0) < \pi_1 pdf(x;\theta_1)
\implies
\frac{pdf(x;\theta_1)}{pdf(x;\theta_0)} > \frac{\pi_0}{\pi_1}
$$
이므로, 이 경우 $\phi(x)=1$로 하고, 그렇지 않으면 $\phi(x)=0$으로 한다.  
즉,
$$
\phi^\pi(x) =
\begin{cases}
1, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} > \dfrac{\pi_0}{\pi_1} \\
0, & \text{otherwise}
\end{cases}
$$
가 베이지안 평균오류확률을 최소로 한다.

#### 예 9.1.1 (정규분포)
- 표본: $X_1,\dots,X_n \sim N(\mu,1)$
- 가설: $H_0:\mu=\mu_0 \quad\text{vs}\quad H_1:\mu=\mu_1$

가능도비:
$$
\frac{pdf(x;\mu_1)}{pdf(x;\mu_0)} =
\exp\left[
n(\mu_1-\mu_0)
\left(
\bar{x} - \frac{\mu_1+\mu_0}{2}
\right)
\right]
$$

(a) $\mu_1 > \mu_0$일 때
$$
\phi^\pi(x)=
\begin{cases}
1, & \bar{x}-\mu_0 >
\frac{\mu_1-\mu_0}{2}
+ \frac{\log(\pi_0/\pi_1)}{n(\mu_1-\mu_0)} \\
0, & \text{otherwise}
\end{cases}
$$

(b) $\mu_1 < \mu_0$일 때
$$
\phi^\pi(x)=
\begin{cases}
1, & \bar{x}-\mu_0 <
-\frac{\mu_0-\mu_1}{2}
+ \frac{\log(\pi_0/\pi_1)}{n(\mu_0-\mu_1)} \\
0, & \text{otherwise}
\end{cases}
$$

베이지안 평균오류확률이나 최대오류확률 기준은  
귀무가설과 대립가설을 대칭적으로 취급한다.  
이는 제7장에서 다룬 전통적 검정과 다르다.

### 최강력 검정 *(Most Powerful Test)*
전통적 검정 이론에서는
- 제1종 오류 확률을 유의수준 $\alpha$ 이하로 제한하고
- 그 조건에서 검정력
    $$
    \gamma_\phi(\theta_1) = E_{\theta_1}[\phi(X)]
    $$
    을 최대화하는 검정을 찾는다.

**정의: 유의수준 $\alpha$의 최강력 검정**  
단순 가설 $H_0:\theta=\theta_0 \quad\text{vs}\quad H_1:\theta=\theta_1$에서 검정 $\phi^{MP}_\alpha(X)$가 다음을 만족하면 **최강력 검정**(Most Powerful Test, MP test)이라 한다.

- **최강력 검정(Most Powerful Test, MP test)**:  
    주어진 유의수준 $\alpha$에서, 대립가설 하에서 검정력(귀무가설을 기각할 확률)이 가능한 한 가장 큰 검정. 즉, 같은 유의수준을 만족하는 모든 검정 중에서 대립가설 하에서 귀무가설을 기각할 확률이 최대가 되는 검정. 이런 검정을 기호로 $\phi^{MP}_\alpha(X)$ 또는 $\phi^{MP}_\alpha$ 또는 $\phi^*_\alpha(X)$로 쓴다.
    
1. (유의수준)
        $$
        E_{\theta_0}[\phi^{MP}_\alpha(X)] \le \alpha \quad (0\lt \alpha \lt 1)
        $$
2. (최대 검정력)
        $$
        E_{\theta_1}[\phi^{MP}_\alpha(X)] \ge E_{\theta_1}[\phi(X)] \quad \forall \phi: E_{\theta_0}[\phi(X)] \le \alpha
        $$

### 정리 9.1.1 단순 가설의 최강력 검정 (Neyman–Pearson Lemma)
검정
$
\phi^*(x)=
\begin{cases}
1, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} > k \\
\gamma, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} = k \\
0, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} < k
\end{cases}
$
가
$
E_{\theta_0}[\phi^*(X)] = \alpha
$
를 만족하면, 이는 유의수준 $\alpha$의 **최강력 검정**이다.

#### 증명 개요
임의의 검정 $\phi$에 대해
$
E_{\theta_1}[\phi^*(X)] - k E_{\theta_0}[\phi^*(X)]
\ge
E_{\theta_1}[\phi(X)] - k E_{\theta_0}[\phi(X)]
$
임을 보이고,

$E_{\theta_0}[\phi(X)] \le \alpha$를 이용하여
$
E_{\theta_1}[\phi^*(X)]
\ge
E_{\theta_1}[\phi(X)]
$
를 얻는다.

#### 예 9.1.2 (포아송 분포)
- 표본: $X_1,\dots,X_{100} \sim \mathrm{Poisson}(\theta)$
- 가설: $H_0:\theta=0.1 \quad\text{vs}\quad H_1:\theta=0.05$
- 유의수준: $\alpha=0.05$

가능도비:
$
\frac{pdf(x;\theta_1)}{pdf(x;\theta_0)}
=
e^{-100(\theta_1-\theta_0)}
\left(\frac{\theta_1}{\theta_0}\right)^{\sum x_i}
=
e^5 (1/2)^{\sum x_i}
$

따라서 최강력 검정은
$
\phi^*(x)=
\begin{cases}
1, & \sum x_i \le c-1 \\
\gamma, & \sum x_i = c \\
0, & \sum x_i \ge c+1
\end{cases}
$
이며,
$
E_{\theta_0}[\phi^*(X)] = 0.05
$
을 만족하도록 $(c, \gamma)$를 정한다.

→ 계산 결과: $c=5,\quad \gamma=\frac{21}{38}$


## 전역최강력 검정 *(Uniformly Most Powerful Tests)*
모집단 분포가 확률밀도함수 $f(x;\theta)$, $\theta \in \Omega$ 중의 하나인 경우에 랜덤표본 $X=(X_1,\dots,X_n)$을 이용하여 일반적인 가설
$
H_0:\theta\in\Omega_0
\quad\text{vs}\quad
H_1:\theta\in\Omega_1
\quad
(\Omega_0\cap\Omega_1=\varnothing,\ \Omega_0\cup\Omega_1=\Omega)
$
을 유의수준 $\alpha$에서 검정할 때, 대립가설의 각 모수 값에서의 검정력
$
\gamma_\phi(\theta_1)=E_{\theta_1}\phi(X),\quad \theta_1\in\Omega_1
$
을 크게 하는 검정이 좋은 것이다.

### 전역최강력 검정
일반적인 가설
$
H_0:\theta\in\Omega_0
\quad\text{vs}\quad
H_1:\theta\in\Omega_1
$
을 검정할 때, 다음을 만족시키는 검정 $\phi^{UMP}_\alpha$를 유의수준 $\alpha$의 전역최강력 검정이라 한다.

(i) (유의수준)
$
\max_{\theta\in\Omega_0} E_\theta \phi^{UMP}_\alpha(X)\le \alpha
$

(ii) (대립가설 전역에서 최대의 검정력)
$
E_{\theta_1}\phi^{UMP}_\alpha(X)\ge E_{\theta_1}\phi(X),
\quad
\forall\theta_1\in\Omega_1,\;
\forall\phi:\max_{\theta\in\Omega_0}E_\theta\phi(X)\le\alpha
$

#### 예 9.2.1
정규분포 $N(\mu,1)$에서의 랜덤표본을 이용하여
$
H_0:\mu=\mu_0
\quad\text{vs}\quad
H_1:\mu>\mu_0
$
을 유의수준 $\alpha$에서 검정할 때 전역최강력 검정은
$
\phi^*(x)=
\begin{cases}
1, & \bar{x}-\mu_0\ge z_\alpha/\sqrt{n} \\
0, & \bar{x}-\mu_0< z_\alpha/\sqrt{n}
\end{cases}
$
이다.

가능도비는
$
\frac{pdf(x;\mu_1)}{pdf(x;\mu_0)}
=
\exp\left[
n(\mu_1-\mu_0)
\left(
\bar{x}-\frac{\mu_1+\mu_0}{2}
\right)
\right]
$
로 주어지고 이는 $\bar{x}$의 증가함수이므로, $c=\mu_0+z_\alpha/\sqrt{n}$이다.
이 검정은 $\mu_1$의 값에 관계없이 성립하므로 유의수준 $\alpha$의 전역최강력 검정이다.

#### 예 9.2.2
정규분포 $N(\mu,1)$에서
$
H_0(-):\mu\le\mu_0
\quad\text{vs}\quad
H_1:\mu>\mu_0
$
을 검정할 때, 예 9.2.1의 검정이 유의수준 $\alpha$의 전역최강력 검정임을 보인다.

검정력 함수는
$
E_\mu\phi^*(X)
=
P_\mu(\bar{X}-\mu_0\ge z_\alpha/\sqrt{n})
=
P\left(Z\ge \sqrt{n}(\mu_0-\mu)+z_\alpha\right),\quad Z\sim N(0,1)
$
로서 $\mu$의 증가함수이다. 따라서
$
\max_{\mu\le\mu_0}E_\mu\phi^*(X)=E_{\mu_0}\phi^*(X)=\alpha
$
이다.

또한
$
\{\phi:\max_{\mu\le\mu_0}E_\mu\phi(X)\le\alpha\}
\subset
\{\phi:E_{\mu_0}\phi(X)\le\alpha\}
$
이므로, 예 9.2.1의 검정 $\phi^*$는 전역최강력 검정이다.

### 정리 9.2.1 단일모수 지수족과 전역최강력 한쪽 검정
모집단 분포의 확률밀도함수가
$
f(x;\theta)
=
\exp\{g(\theta)T(x)-B(\theta)+S(x)\},
\quad
x\in\mathcal{X},\ \theta\in\Omega\subset\mathbb{R}
$
와 같이 나타내어지는 단일모수 지수족이고, $g(\theta)$가 $\theta$의 증가함수일 때, 가설
$
H_0:\theta\le\theta_0
\quad\text{vs}\quad
H_1:\theta>\theta_0
$
을 유의수준 $\alpha$에서 검정한다고 하자.

이때 다음 조건을 만족시키는 검정 $\phi^*$는 유의수준 $\alpha$의 전역최강력 검정이다.

(a) (가능도비 검정 꼴)
$
\phi^*(x)=
\begin{cases}
1, & T(x_1)+\cdots+T(x_n)>c \\
\gamma, & T(x_1)+\cdots+T(x_n)=c \\
0, & T(x_1)+\cdots+T(x_n)<c
\end{cases}
$

(b) (검정의 크기)
$
E_{\theta_0}\phi^*(X)=\alpha
$

#### 예 9.2.3
포아송분포 $\mathrm{Poisson}(\theta)$, $0<\theta<+\infty$에서 $n=100$개의 랜덤표본을 이용하여
$
H_0:\theta\ge0.1
\quad\text{vs}\quad
H_1:\theta<0.1
$
을 검정할 때 유의수준 $\alpha=0.05$의 전역최강력 검정은
$
\phi^*(x)=
\begin{cases}
1, & x_1+\cdots+x_n\le c-1 \\
\gamma, & x_1+\cdots+x_n=c \\
0, & x_1+\cdots+x_n\ge c+1
\end{cases}
$
이고
$
E_{\theta_0}\phi^*(X)=0.05,\quad (\theta_0=0.1)
$
를 만족시키는 $c=5,\ \gamma=21/38$이다.

#### 예 9.2.4
지수분포 $\mathrm{Exp}(\theta)$, $0<\theta<+\infty$에서 랜덤표본 $X_1,\dots,X_n$을 이용하여
$
H_0:\theta\le\theta_0
\quad\text{vs}\quad
H_1:\theta>\theta_0
$
을 검정할 때 유의수준 $\alpha$의 전역최강력 검정은
$
\phi^*(x)=
\begin{cases}
1, & x_1+\cdots+x_n\ge c \\
0, & x_1+\cdots+x_n<c
\end{cases},
\quad
E_{\theta_0}\phi^*(X)=\alpha
$
이다.

$\theta=\theta_0$일 때
$
\sum_{i=1}^n X_i/\theta_0\sim \mathrm{Gamma}(n,1),\quad
2\sum_{i=1}^n X_i/\theta_0\sim\chi^2(2n)
$
이므로
$
c=\theta_0\chi^2_\alpha(2n)/2
$
이다

#### 예 9.2.5
정규분포 $N(\mu,1)$에서 랜덤표본을 이용하여
$$
H_0:\mu=\mu_0 \quad\text{vs}\quad H_1:\mu\ne\mu_0
$$
을 유의수준 $\alpha$에서 검정할 때, 전역최강력 검정이 존재하지 않음을 보여라.

전역최강력 검정이 존재한다면 임의의 $\mu_1 < \mu_0 < \mu_2$에 대해
$$
H_0:\mu=\mu_0\ \text{vs}\ H_1(\mu_1):\mu=\mu_1, \qquad
H_0:\mu=\mu_0\ \text{vs}\ H_1(\mu_2):\mu=\mu_2
$$
에 대해 모두 유의수준 $\alpha$의 최강력 검정이어야 한다.  
그러나 이를 동시에 만족시키는 상수 $c_1, c_2$는 존재하지 않으므로 전역최강력 검정은 존재하지 않는다.

#### 예 9.2.6
정규분포 $N(\theta,1)$에서 랜덤표본을 이용하여
$$
H_0:\theta=\theta_0 \quad\text{vs}\quad H_1:\theta\ne\theta_0
$$
을 유의수준 $\alpha$에서 검정할 때, 최대가능도비 검정
$$
\phi^*(x)=
\begin{cases}
1, & \sqrt{n}|\bar{x}-\theta_0| \ge z_{\alpha/2} \\
0, & \sqrt{n}|\bar{x}-\theta_0| < z_{\alpha/2}
\end{cases}
$$
에 대해 다음을 보여라.

(a)
$$
E_{\theta_0}[\phi^*(X)] = \alpha, \qquad
\left.\frac{d}{d\theta}E_\theta[\phi^*(X)]\right|_{\theta=\theta_0} = 0
$$

(b)
$$
\forall\phi:\ E_{\theta_0}[\phi(X)] = \alpha,\ 
\left.\frac{d}{d\theta}E_\theta[\phi(X)]\right|_{\theta=\theta_0} = 0
\implies
E_\theta[\phi^*(X)] \ge E_\theta[\phi(X)],\ \forall\theta\ne\theta_0
$$

**[증명]**
(a) 검정력 함수는
$$
E_\theta[\phi^*(X)] = 1-\Phi(z_{\alpha/2}-\delta) + \Phi(-z_{\alpha/2}-\delta),\quad \delta = \sqrt{n}(\theta-\theta_0)
$$
로 나타나므로 성립한다.

(b) $p_\theta(X) = pdf(X;\theta)$, $p'_\theta(X) = \frac{d}{d\theta}pdf(X;\theta)$라 하면
$$
\left.\frac{d}{d\theta}E_\theta[\phi(X)]\right|_{\theta=\theta_0}
= E_{\theta_0}\left[\phi(X)\frac{p'_{\theta_0}(X)}{p_{\theta_0}(X)}\right]
$$
이고, 이를 이용하여 조건을 만족시키는 검정 중에서 $\phi^*$가 검정력이 최대임을 보일 수 있다.

## 비모수적 검정과 점근적 비교 *(Nonparametric Tests and Asymptotic Comparisons)*
모집단 분포에 특정한 형태를 가정하지 않는 경우의 검정에 대해 살펴보자.

#### 예 9.3.1 위치모수 모형에서 부호검정
모집단 분포가 연속형이고 확률밀도함수가 $f(x-\theta)$, $-\infty<\theta<+\infty$의 꼴로서 $\theta$에 관해 대칭($f(-x)=f(x)$)이고, $f$에 대응하는 누적분포함수 $F$가 순증가함수인 모형을 생각한다. 랜덤표본 $X_1,\dots,X_n$을 이용하여
$$
H_0(\theta_0):\theta=\theta_0 \quad\text{vs}\quad H_1:\theta>\theta_0
$$
을 유의수준 $\alpha$에서 검정할 때, 통계량
$$
S_n = \sum_{i=1}^n I(X_i > \theta_0)
$$
을 이용한다.

$S_n$의 분포는
$$
S_n \sim B(n, p(\theta)),\quad p(\theta) = P_\theta(X_1 > \theta_0) = 1 - F(\theta_0 - \theta)
$$
이고, 위의 가설이 $p(\theta)$에 관한 가설
$$
H_0(1/2):p(\theta)=1/2 \quad\text{vs}\quad H_1:p(\theta)>1/2
$$
에 대응한다.

유의수준 $\alpha$의 검정:
$$
\phi_s(X_1,\dots,X_n) =
\begin{cases}
1, & S_n \ge c+1 \\
\gamma, & S_n = c \quad (0 \le \gamma \le 1) \\
0, & S_n \le c-1
\end{cases}
$$
$$
E_{\theta_0}[\phi_s(X)] = P_{\theta_0}(S_n \ge c+1) + \gamma P_{\theta_0}(S_n = c) = \alpha
$$
즉,
$$
\sum_{k=c+1}^n \binom{n}{k}(1/2)^n + \gamma \binom{n}{c}(1/2)^n = \alpha
$$

$\phi_s(x_1,\dots,x_n)$은 각 성분 $x_i$의 증가함수이므로
$$
\max_{\theta \le \theta_0} E_\theta[\phi_s(X)] = E_{\theta_0}[\phi_s(X)] = \alpha
$$
따라서 $\phi_s$는
$$
H_0:\theta \le \theta_0 \quad\text{vs}\quad H_1:\theta > \theta_0
$$
에 대한 유의수준 $\alpha$의 검정이다. 이와 같이 $S_n$을 사용하여 연속형 분포의 중앙값에 대한 검정을 하는 방법을 **부호검정(sign test)**이라고 한다.

부호검정은 모집단 분포에 특정한 함수 형태를 가정하지 않고 사용할 수 있는 반면, 특정 모집단에 적용하면 효율성이 떨어질 수 있다. 효율성 판단에는 특정 대립가설에서의 검정력을 일정 수준으로 유지하기 위한 표본크기 비교를 기준으로 한다.

#### 예 9.3.2 위치모수 모형에서 부호검정의 검정력 근사
예 9.3.1의 가설에 대해 부호검정 통계량 $S_n$의 극한분포에 이항분포의 정규근사를 적용하면
$$
\frac{S_n - n p(\theta)}{\sqrt{n \sigma^2(\theta)}} \xrightarrow{d} N(0,1),\quad \sigma^2(\theta) = p(\theta)(1-p(\theta))
$$
이 성립하므로, 표본크기가 클 때 유의수준 $\alpha$의 기각역은
$$
S_n \ge c_n,\quad
c_n \simeq n p(\theta_0) + \sqrt{n} \sigma(\theta_0) z_\alpha,\quad p(\theta_0) = 1/2,\ \sigma(\theta_0) = 1/2
$$
대립가설 $\theta = \theta_1 (\theta_1 > \theta_0)$에서의 검정력 근사:
$$
\gamma_n(\theta_1) = P_{\theta_1}(S_n \ge c_n)
$$
$$
\simeq 1 - \Phi\left(
-\sqrt{n} \frac{p(\theta_1) - p(\theta_0)}{\sigma(\theta_1)}
+ \frac{\sigma(\theta_0)}{\sigma(\theta_1)} z_\alpha
\right)
$$
고정된 대립가설 $\theta = \theta_1$에 대해
$$
\lim_{n\to\infty} \gamma_n(\theta_1) = 1
$$
귀무가설에 가까운 대립가설 $\theta_{1n} \simeq \theta_0 + K/\sqrt{n}$ ($K>0$)에 대해
$$
\gamma_n(\theta_{1n}) \simeq 1 - \Phi\left(
-\sqrt{n} (\theta_{1n} - \theta_0) \dot{p}(\theta_0)/\sigma(\theta_0) + z_\alpha
\right),\quad
\dot{p}(\theta) = \frac{d}{d\theta}p(\theta) = f(\theta_0 - \theta)
$$
따라서 $\gamma_n(\theta_{1n}) \simeq \gamma$가 되기 위한 표본크기는 근사적으로
$$
n = (2f(0))^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\theta_{1n} - \theta_0} \right)^2
$$

### 정리 9.3.1 검정력의 근사와 표본크기
실수 모수 $\theta$에 관한 가설
$$
H_0(\theta_0):\theta=\theta_0 \quad\text{vs}\quad H_1:\theta>\theta_0
$$
을 유의수준 $\alpha$에서 검정할 때, 크기 $n$인 랜덤표본에 기초한 검정통계량 $T_n$을 이용한 크기 $\alpha$의 기각역이
$$
\sqrt{n} \frac{T_n - \mu(\theta_0)}{\sigma(\theta_0)} \ge t_n
$$
이고, $T_n$에 대해
$$
\sqrt{n} \frac{T_n - \mu(\theta)}{\sigma(\theta)} \xrightarrow{d} N(0,1)
$$
과 같은 점근정규성이 성립한다고 하자.

(a) **검정력 근사**  
귀무가설에 가까운 대립가설 $\theta_{1n} \simeq \theta_0 + K/\sqrt{n}$ ($K>0$)에서
$$
\gamma_n(\theta_{1n}) \simeq 1 - \Phi\left(
-\sqrt{n} (\theta_{1n} - \theta_0) \dot{\mu}(\theta_0)/\sigma(\theta_0) + z_\alpha
\right),\quad
\dot{\mu}(\theta) = \frac{d}{d\theta}\mu(\theta)
$$

(b) **표본크기 근사**  
$\gamma_n(\theta_{1n}) \simeq \gamma$가 되기 위한 표본크기 근사:
$$
N(T_n;\gamma,\theta_{1n}) \simeq \left( \frac{\dot{\mu}(\theta_0)}{\sigma(\theta_0)} \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\theta_{1n} - \theta_0} \right)^2
$$

**[증명]**
검정의 크기가 $\alpha$이므로
$$
P_{\theta_0}\left( \sqrt{n} \frac{T_n - \mu(\theta_0)}{\sigma(\theta_0)} \ge t_n \right) = \alpha
$$
$$
t_n \simeq z_\alpha
$$
대립가설 $\theta = \theta_1$에서의 검정력 근사:
$$
\gamma_n(\theta_1) \simeq 1 - \Phi\left(
-\sqrt{n} \frac{\mu(\theta_1) - \mu(\theta_0)}{\sigma(\theta_1)} + \frac{\sigma(\theta_0)}{\sigma(\theta_1)} z_\alpha
\right)
$$
$\theta_{1n} \simeq \theta_0 + K/\sqrt{n}$에서
$$
\gamma_n(\theta_{1n}) \simeq 1 - \Phi\left(
-\sqrt{n} (\theta_{1n} - \theta_0) \dot{\mu}(\theta_0)/\sigma(\theta_0) + z_\alpha
\right)
$$
표본크기 근사:
$$
N(T_n;\gamma,\theta_{1n}) \simeq \left( \frac{\dot{\mu}(\theta_0)}{\sigma(\theta_0)} \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\theta_{1n} - \theta_0} \right)^2
$$

### 점근상대효율성
실수 모수 $\theta$에 관한 가설
$$
H_0(\theta_0):\theta=\theta_0 \quad\text{vs}\quad H_1:\theta>\theta_0
$$
을 유의수준 $\alpha$에서 검정할 때, 크기 $n$인 랜덤표본에 기초한 검정통계량 $T_{in}$ ($i=1,2$)들이 정리 9.3.1의 조건을 만족한다고 하자. 대립가설 $\theta_{1n} \simeq \theta_0 + K/\sqrt{n}$에서 검정력이 $\gamma$가 되기 위한 표본크기를 $N(T_{in};\gamma,\theta_{1n})$라 할 때, 이들 표본크기의 역수의 극한값
$$
\lim_{n\to\infty} \frac{N^{-1}(T_{1n};\gamma,\theta_{1n})}{N^{-1}(T_{2n};\gamma,\theta_{1n})}
$$
을 $T_{1n}$에 의한 검정 방법 1의 $T_{2n}$에 의한 검정 방법 2에 대한 **점근상대효율성**(asymptotic relative efficiency, ARE)이라 한다.  
기호로는 $\mathrm{ARE}(T_{1n}, T_{2n})$로 나타내며, 정리 9.3.1로부터
$$
\mathrm{ARE}(T_{1n}, T_{2n}) = \frac{(\dot{\mu}_1(\theta_0)/\sigma_1(\theta_0))^2}{(\dot{\mu}_2(\theta_0)/\sigma_2(\theta_0))^2}
$$

#### 예 9.3.3 부호검정의 $t$-검정에 대한 점근상대효율성
모집단 분포가 연속형이고 $\mu$에 대해 대칭이며 확률밀도함수가
$$
\frac{1}{\sigma} f\left(\frac{x-\mu}{\sigma}\right),\quad -\infty<\mu<+\infty,\ \sigma>0
$$
인 경우, 가설
$$
H_0(\mu_0):\mu=\mu_0 \quad\text{vs}\quad H_1:\mu>\mu_0
$$
을 유의수준 $\alpha\ (0<\alpha<1)$에서 검정할 때, 통계량
$$
S_n = \sum_{i=1}^n I(X_i > \mu_0)
$$
을 이용하는 부호검정을 예 9.3.1과 같이 적용할 수 있다. 이때 $\mu_{1n} \simeq \mu_0 + K/\sqrt{n}\ (K>0)$에서 부호검정의 검정력이 $\gamma$가 되기 위한 표본크기는
$$
N(S_n;\gamma,\mu_{1n}) \simeq \left( \frac{2f(0)}{\sigma} \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\mu_{1n} - \mu_0} \right)^2
$$
로 주어진다.

한편 모집단이 정규분포라면
$$
\sqrt{n}(\bar{X} - \mu_0)/S \ge t_\alpha(n-1)
$$
로 주어지는 $t$-검정을 사용한다. $t$-검정의 통계량
$$
T_n = \sqrt{n}(\bar{X} - \mu_0)/S
$$
에 대해 점근정규성이 성립하며,
$$
\sqrt{n}\left( \frac{\bar{X} - \mu_0}{S} - \frac{\mu_{1n} - \mu_0}{\sqrt{\mathrm{Var}(X_1)}} \right) \xrightarrow{d} N(0,1)
$$
여기서
$$
\sqrt{\mathrm{Var}(X_1)} = \sigma \left( \int_{-\infty}^{+\infty} z^2 f(z)\,dz \right)^{1/2}
$$
따라서 $t$-검정의 표본크기 근사식은
$$
N(T_n;\gamma,\mu_{1n}) \simeq \left( \frac{1}{\sqrt{\mathrm{Var}(X_1)}} \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\mu_{1n} - \mu_0} \right)^2
$$

따라서 부호검정의 $t$-검정에 대한 점근상대효율성(ARE)은
$$
\mathrm{ARE}(S_n, T_n) = \frac{(2f(0)/\sigma)^2}{(1/\sqrt{\mathrm{Var}(X_1)})^2} = 4(f(0))^2 \int_{-\infty}^{+\infty} z^2 f(z)\,dz
$$
이며, 이는 표본중앙값의 표본평균에 대한 점근상대효율성과 같다.

**표 9.3.1 부호검정의 $t$-검정에 대한 점근상대효율성**
- 모집단 분포 $N(\mu,\sigma^2)$: $2/\pi \approx 0.636$
- 모집단 분포 $L(\mu,\sigma)$: $\pi^2/12 \approx 0.822$
- 모집단 분포 $DE(\mu,\sigma)$: $2$

부호검정처럼 모집단 분포의 형태를 가정하지 않고 사용할 수 있는 검정 방법을 **비모수적(nonparametric) 검정**이라 하며, 모집단 분포에 대한 가정이 어려운 경우에 유용하다. 대표적 비모수적 검정인 크기 순서를 이용하는 방법을 살펴보자.

#### 예 9.3.4 위치모수 모형에서 부호순위 검정통계량
모집단 분포가 연속형이고 확률밀도함수가
$$
f(x-\theta),\quad -\infty<\theta<+\infty
$$
의 꼴로서 $\theta$에 대해 대칭($f(-x)=f(x)$)인 경우, 랜덤표본 $X_1,\dots,X_n$을 이용하여
$$
H_0(\theta_0):\theta=\theta_0 \quad\text{vs}\quad H_1:\theta>\theta_0
$$
을 검정한다. 이때 $|X_1-\theta_0|,\dots,|X_n-\theta_0|$을 크기순으로 나열할 때 $|X_i-\theta_0|$의 순위를 $R(|X_i-\theta_0|)$라 하며,
$$
R(|X_i-\theta_0|) = 1 + \sum_{j=1}^n I(|X_j-\theta_0| < |X_i-\theta_0|)
$$
이다. 다음 통계량을 **부호순위(signed rank) 검정통계량**이라 한다.
$$
W_n = \sum_{i=1}^n \operatorname{sgn}(X_i-\theta_0) R(|X_i-\theta_0|)
$$
여기서 $\operatorname{sgn}(x)$는 $x$의 부호 함수이다.

### 정리 9.3.2 부호순위 검정통계량의 귀무가설하의 분포
귀무가설 $H_0(\theta_0):\theta=\theta_0$ 하에서 부호순위 검정통계량의 분포는 다음과 같다.

(a)
$$
W_n \overset{d}{=} \sum_{j=1}^n j S(j),\quad S(j):\text{iid},\ P(S(j)=-1)=P(S(j)=+1)=1/2
$$

(b)
$$
\frac{W_n - E_{\theta_0} W_n}{\sqrt{\mathrm{Var}_{\theta_0}(W_n)}} \xrightarrow{d} N(0,1)
$$
$$
E_{\theta_0}(W_n) = 0,\quad \mathrm{Var}_{\theta_0}(W_n) = \sum_{j=1}^n j^2 = \frac{n(n+1)(2n+1)}{6}
$$

### 정리 9.3.3 부호순위 검정통계량의 표현
- 표기
    - $R(|X_i-\theta_0|)$: $|X_1-\theta_0|,\dots,|X_n-\theta_0|$의 순위
    - $\operatorname{sgn}(x)$: 부호 함수
    - 부호순위 검정통계량
        $$
        W_n = \sum_{i=1}^n \operatorname{sgn}(X_i-\theta_0) R(|X_i-\theta_0|),\qquad
        W_n^+ = \sum_{i=1}^n \mathbf{1}(X_i-\theta_0 > 0) R(|X_i-\theta_0|)
        $$
- (a) $W_n$과 $W_n^+$의 관계
    $$
    W_n = 2W_n^+ - \frac{n(n+1)}{2}
    $$

- (b) 귀무가설 $H_0(\theta_0):\theta=\theta_0$ 하에서의 분포 및 정규근사
    $$
    W_n^+ \overset{d}{=} \sum_{j=1}^n j B_j,\qquad B_j \overset{iid}{\sim} \mathrm{Bernoulli}(1/2)
    $$
    $$
    \frac{W_n^+ - n(n+1)/4}{\sqrt{n(n+1)(2n+1)/24}} \overset{d}{\to} N(0,1)\quad(n\to\infty)
    $$

- 참고 항등식 (연속형이면 동점확률 0)
    $$
    \operatorname{sgn}(X_i-\theta_0) = 2\mathbf{1}(X_i-\theta_0 > 0) - 1
    $$

### 정리 9.3.4 한쪽 가설에 대한 부호순위 검정
- 모형(위치모수, 대칭): 모집단 밀도 $f(x-\theta)$, $-\infty<\theta<\infty$, $f(-x)=f(x)$
- 가설
    $$
    H_0:\theta\le \theta_0\quad \text{vs}\quad H_1:\theta>\theta_0
    $$
- 유의수준 $\alpha$의 부호순위 검정(임계값 $c$, 랜덤화 $\gamma\in[0,1]$)
    $$
    \phi_{SR}(X_1,\dots,X_n)=
    \begin{cases}
        1, & W_n \ge c+1 \\
        \gamma, & W_n = c \\
        0, & W_n \le c-1
    \end{cases}
    \qquad E_{\theta_0}[\phi_{SR}(X)] = \alpha
    $$
    또는 $W_n^+$로 동치 표현:
    $$
    \phi_{SR}(X_1,\dots,X_n)=
    \begin{cases}
        1, & W_n^+ \ge c^+ + 1 \\
        \gamma, & W_n^+ = c^+ \\
        0, & W_n^+ \le c^+ - 1
    \end{cases}
    \qquad E_{\theta_0}[\phi_{SR}(X)] = \alpha
    $$

- 성질
    - 검정력 함수 $\gamma_{\phi_{SR}}(\theta) = E_\theta[\phi_{SR}(X)]$는 $\theta$의 증가함수
    - 따라서
        $$
        \max_{\theta\le \theta_0} E_\theta[\phi_{SR}(X)] = E_{\theta_0}[\phi_{SR}(X)] = \alpha
        $$
    - (정규근사에 의한 큰 표본 기각역)
        $$
        \frac{W_n}{\sqrt{n(n+1)(2n+1)/6}} \ge z_\alpha
        \quad\text{또는}\quad
        \frac{W_n^+ - n(n+1)/4}{\sqrt{n(n+1)(2n+1)/24}} \ge z_\alpha
        $$

### 정리 9.3.5 부호순위 검정의 점근정규성
- (a) 점근정규성
    $$
    \frac{W_n^+ - E_\theta(W_n^+)}{\sqrt{\mathrm{Var}_\theta(W_n^+)}} \overset{d}{\to} N(0,1)
    $$
- (b) 평균/분산 근사
    $$
    \mu(\theta) = \frac{1}{2} P_\theta(X_1 + X_2 > 2\theta_0),\qquad
    \sigma^2(\theta) = \mathrm{Cov}_\theta\left( \mathbf{1}(X_1 + X_2 > 2\theta_0),\ \mathbf{1}(X_1 + X_3 > 2\theta_0) \right)
    $$
    이때
    $$
    E_\theta(W_n^+) \simeq n^2 \mu(\theta),\qquad \mathrm{Var}_\theta(W_n^+) \simeq n^3 \sigma^2(\theta)
    $$
    따라서 $W_n^+/n^2$에 대해 앞의 점근적 검정력/표본크기 근사 공식을 적용할 수 있다.

### 정리 9.3.6 부호순위 검정의 검정력 근사와 표본크기
- 정리 9.3.4의 가정(대칭 위치모수 모형) 하에서, 대립가설이
    $$
    \theta_{1n} \simeq \theta_0 + \frac{K}{\sqrt{n}}\quad(K>0)
    $$
    처럼 귀무가설에 근접할 때,

- (a) 검정력 근사
    $$
    \gamma_{\phi_{SR}}(\theta_{1n}) \simeq 1 - \Phi\left( -\sqrt{n} (\theta_{1n} - \theta_0) \frac{\dot\mu(\theta_0)}{\sigma(\theta_0)} + z_\alpha \right)
    $$
    여기서
    $$
    \dot\mu(\theta_0) = \int_{-\infty}^{\infty} f^2(x)\,dx,\qquad \sigma^2(\theta_0) = \frac{1}{12}
    $$

- (b) 목표 검정력 $\gamma$를 얻기 위한 표본크기 근사
    $N(W_n^+;\gamma,\theta_{1n})$를 “검정력 $\gamma$”에 필요한 표본크기라 하면
    $$
    N(W_n^+;\gamma,\theta_{1n}) \simeq \left( \sqrt{12} \int_{-\infty}^{\infty} f^2(x)\,dx \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\theta_{1n} - \theta_0} \right)^2
    $$

#### 예 9.3.5 부호순위 검정의 $t$-검정에 대한 점근상대효율성(ARE)
- 대칭 위치-척도 모형:
    $$
    \frac{1}{\sigma} f\left(\frac{x-\mu}{\sigma}\right),\quad -\infty<\mu<\infty,\ \sigma>0
    $$
    에서
    $$
    H_0(\mu_0):\mu=\mu_0\quad \text{vs}\quad H_1:\mu>\mu_0
    $$

- 부호순위 검정의 표본크기 근사(로컬 대립 $\mu_{1n} \simeq \mu_0 + K/\sqrt{n}$)
    $$
    N(W_n;\gamma,\mu_{1n}) \simeq \left( \sqrt{12} \frac{\int f^2(z)\,dz}{\sigma} \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\mu_{1n} - \mu_0} \right)^2
    $$

- 정규모형에서의 $t$-검정 표본크기 근사
    $$
    N(T_n;\gamma,\mu_{1n}) \simeq \left( \frac{1}{\sqrt{\mathrm{Var}(X_1)}} \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\mu_{1n} - \mu_0} \right)^2
    $$
    $$
    \sqrt{\mathrm{Var}(X_1)} = \sigma \left( \int z^2 f(z)\,dz \right)^{1/2}
    $$

- 따라서 점근상대효율성(ARE)
    $$
    \mathrm{ARE}(W_n, T_n) = 12 \left( \int f^2(z)\,dz \right)^2 \left( \int z^2 f(z)\,dz \right)
    $$

- **표 9.3.2: $\mathrm{ARE}(W_n, T_n)$**
    - $N(\mu,\sigma^2)$: $3/\pi \approx 0.954$
    - $L(\mu,\sigma)$: $\pi^2/9 \approx 1.096$
    - $DE(\mu,\sigma)$: $1.5$
