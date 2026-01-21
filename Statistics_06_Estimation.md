# 제6장 추정 *(Statistical Estimation)*

## 6.1 적률이용 추정법 *(Method of Moments Estimation, MME)*

### 6.1.1 도입 및 기본 개념**적률이용 추정법(Method of Moments Estimation, MME)** 은 모집단의 특성을 나타내는 **모수(parameter)** 를 표본으로부터 추정하는 대표적인 방법 중 하나이다. 이 방법은 모집단의 **적률(moment)** 과 이에 대응하는 **표본적률(sample moment)** 을 일치시키는 방식으로 모수를 추정한다.

랜덤표본 $X_1, \dots, X_n$이 모집단에서 주어졌다고 하자. 예를 들어, 모평균 $\mu = E(X_1)$를 추정할 때 표본평균 $\bar X = \frac{1}{n}(X_1 + \cdots + X_n)$을 사용하는 것처럼, 모집단의 $r$차 적률 $m_r = E(X_1^r)$을 추정할 때는 표본적률
$$
\hat m_r = \frac{1}{n}(X_1^r + \cdots + X_n^r)
$$
을 사용하는 것이 자연스럽다.

일반적으로, 모수가 모집단 적률 $m_1, \dots, m_k$의 함수로 표현되어
$$
\eta = g(m_1, \dots, m_k)
$$
와 같이 주어진다면, 이에 대응하는 **적률이용 추정량(method of moments estimator)** 은
$$
\hat\eta = g(\hat m_1, \dots, \hat m_k)
$$
로 정의한다. 즉, 모집단 적률을 표본적률로 대체하여 모수를 추정한다.

#### 예 6.1.1: 모분산과 모표준편차의 적률이용 추정량
모분산이 $\sigma^2 \ (0 \le \sigma^2 < \infty)$인 모집단에서 랜덤표본 $X_1, \dots, X_n$을 관측했다고 하자.  
모분산은
$$
\sigma^2 = \mathrm{Var}(X_1) = E(X_1^2) - (E(X_1))^2 = m_2 - m_1^2
$$
이에 대응하는 적률이용 추정량은
$$
\hat\sigma^2_{\mathrm{MME}} = \hat m_2 - (\hat m_1)^2 = \frac{1}{n}\sum_{i=1}^n X_i^2 - (\bar X)^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar X)^2
$$
$$
\hat\sigma_{\mathrm{MME}} = \sqrt{\frac{1}{n}\sum_{i=1}^n (X_i - \bar X)^2}
$$
> **설명**: 표본분산 $S_n^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar X)^2$와 달리, 적률이용 추정량은 분모가 $n$임에 유의한다.

#### 예 6.1.2: 적률이용 추정량의 비유일성 *(Non-uniqueness)*
포아송 분포 $Poisson(\lambda)$ $(\lambda>0)$에서 랜덤표본 $X_1, \dots, X_n$을 관측했다고 하자.

이 분포에서는
$$
E(X_1) = \lambda, \quad \mathrm{Var}(X_1) = \lambda
$$
이므로, $\lambda = m_1$ 또는 $\lambda = m_2 - m_1^2$의 두 가지 표현이 모두 성립한다.

따라서 적률이용 추정량은
$$
\hat\lambda_1^{\mathrm{MME}} = \hat m_1 = \bar X, \quad \hat\lambda_2^{\mathrm{MME}} = \hat m_2 - (\hat m_1)^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar X)^2
$$
와 같이 **서로 다른 두 가지**가 가능하다.

> **설명**: 모집단 분포의 특성에 따라 적률들 사이에 함수 관계가 생기므로, 하나의 모수에 대해 여러 적률이용 추정량이 정의될 수 있다.

#### 다차원 모집단에서의 적률이용 추정량
모집단이 다차원인 경우에도 모집단의 결합적률(joint moments)에 대응하여 표본결합적률(sample joint moments)을 이용하는 적률이용 추정량을 정의할 수 있다.  
예를 들어, 이차원 모집단에서 랜덤표본을 $(X_1, Y_1)^\top,\, (X_2, Y_2)^\top,\, \dots,\, (X_n, Y_n)^\top$와 같이 관측했다고 하자.  
이때 모집단의 결합적률 $m_{r,s} = E(X_1^r Y_1^s)$에 대응하는 표본적률은
$$
\hat m_{r,s} = \frac{1}{n}\sum_{i=1}^n X_i^r Y_i^s
$$
따라서 다차원 모수도 결합적률의 함수로 표현할 수 있다면, 모집단 결합적률을 표본결합적률로 대체하여 적률이용 추정량을 얻을 수 있다.

#### 예 6.1.3: 모상관계수의 적률이용 추정량
모평균이 각각 $\mu_1, \mu_2$, 모분산이 각각 $\sigma_1^2, \sigma_2^2$이며 모상관계수가 $\rho \ (-1<\rho<1)$인 이차원 모집단에서 랜덤표본 $(X_1, Y_1), \dots, (X_n, Y_n)$ 을 관측했다고 하자.  
모상관계수는
$$
\rho = \frac{\mathrm{Cov}(X_1, Y_1)}{\sqrt{\mathrm{Var}(X_1)\mathrm{Var}(Y_1)}} = \frac{E(X_1Y_1) - E(X_1)E(Y_1)}{\sqrt{E(X_1^2) - (E(X_1))^2}\sqrt{E(Y_1^2) - (E(Y_1))^2}}
$$
이고, 모집단 결합적률을 표본결합적률로 대체하면,
$$
\hat\rho_n^{\mathrm{MME}} = \frac{\frac{1}{n}\sum_{i=1}^n X_iY_i - \bar X\bar Y}{\sqrt{\frac{1}{n}\sum_{i=1}^n X_i^2 - (\bar X)^2}\sqrt{\frac{1}{n}\sum_{i=1}^n Y_i^2 - (\bar Y)^2}} \\
= 
\frac{\frac{1}{n}\sum_{i=1}^n (X_i-\bar X)(Y_i-\bar Y)}{\sqrt{\frac{1}{n}\sum_{i=1}^n (X_i-\bar X)^2}\sqrt{\frac{1}{n}\sum_{i=1}^n (Y_i-\bar Y)^2}}
$$
가 되며, 모상관계수의 적률이용추정량은 표본상관계수와 동일하다.

### 정리 6.1.1: 적률이용 추정량의 확률수렴 *(Consistency of MME)*
랜덤표본 $X_1,\dots,X_n$으로부터 정의된 표본적률
$$
\hat m_r = \frac{1}{n}\sum_{i=1}^n X_i^r \quad (r=1,\dots,k)
$$
과 모수 $\eta = g(m_1,\dots,m_k)$
에 대하여, 함수 $g$가 연속이면
$$
\hat\eta_n^{\mathrm{MME}} = g(\hat m_1,\dots,\hat m_k)
$$
는
$$
\hat\eta_n^{\mathrm{MME}} \xrightarrow{p} \eta
$$
즉, **확률수렴**한다.

> **설명**: 표본적률이 강법칙에 의해 모적률로 수렴하고, 연속함수 정리에 의해 적률이용 추정량도 모수로 수렴한다.

#### 증명 개요
표본적률 벡터의 확률수렴과 연속함수 정리를 결합하여 얻는다.

### 추정량의 일치성 *(Consistency of Estimators)*
모집단의 특성을 나타내는 모수(parameter)를 추정하는 과정을 **추정(estimation)** 이라 하며, 이때 사용되는 통계량을 **추정량(estimator)** 이라 한다.  
모집단에서 랜덤표본 $X_1, \dots, X_n$을 얻었을 때, 추정대상인 $\eta$의 추정량은
$$
\hat\eta(X_1, \dots, X_n)
$$
또는 간단히 $\hat\eta_n$으로 표기한다.

일반적으로 모집단 분포는 모수 $\theta \in \Omega$에 따라 결정되는 확률밀도함수 $f(x;\theta)$로 가정하며, 추정대상은 $\theta$ 또는 그 함수 $\eta = \eta(\theta)$가 된다.  
이때 표본크기 $n$이 커질수록 추정량 $\hat\eta_n$이 추정대상 $\eta$에 **확률수렴** 하는 성질을 **일치성(consistency)** 이라 하며, 이는 추정량이 갖추어야 할 가장 기본적인 요건이다.

즉, 모든 $\theta \in \Omega$에 대해
$$
\hat\eta_n \xrightarrow{p_\theta} \eta(\theta)
\quad \Leftrightarrow \quad
\lim_{n\to\infty} P_\theta\left(|\hat\eta_n - \eta(\theta)| \ge \epsilon\right) = 0
\quad \forall\, \epsilon > 0
$$
가 성립하면, $\hat\eta_n$은 $\eta$에 대해 일치적(consisitent)이라고 한다.

* 다차원의 경우에도 같은 정리가 성립한다

#### 예 6.1.4: 모분산 적률이용 추정량의 일치성
모분산을 $\sigma^2 = g(m_1,m_2) = m_2 - m_1^2$로 표현하면, $g$는 연속함수이므로 정리6.1.1로부터 예6.1.1에서의 적률이용추정량
$$
\hat\sigma_n^2 = \frac{1}{n}\sum_{i=1}^n (X_i-\bar X)^2
$$
는 $\sigma^2$에 대해 일치성을 가진다.

표본분산
$$
S_n^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i-\bar X)^2 = \frac{n}{n-1}\hat\sigma_n^2
$$
도 역시 일치성을 가진다.

### 정리 6.1.2: 적률이용 추정량의 극한분포 *(Asymptotic Distribution of ME)*
$E(X_1^{2k})<\infty$이고 $g$가 일차 미분 가능하면,
$$
\sqrt{n}(\hat\eta_n - \eta) \xrightarrow{d} Z, \quad Z \sim N(0,\sigma^2)
$$
$$
\sigma^2 = (\nabla g(m))^\top \Sigma \nabla g(m), \quad \Sigma = (m_{r+s} - m_rm_s)
$$
> **설명**: 적률이용 추정량은 중심극한정리에 의해 정규분포로 근사되며,
분산은 적률의 공분산과 $g$의 도함수로 결정된다.


## 6.2 최대가능도 추정법 *(Maximum Likelihood Estimation, MLE)*

### 6.2.1 도입 및 가능도함수
모집단 분포가 확률밀도함수 $f(x;\theta)$ $(\theta\in\Omega)$로 주어진다고 하자. 관측값 $x=(x_1,\dots,x_n)$가 주어졌을 때, 이를 이용하여 실제 모집단을 추측하는 것이 추정의 과정이다. 이때 관측값들이 $\theta$로 표현되는 모형에서 생성될 확률은 대략
$$
P(x_1 \le X_1 < x_1+\Delta x_1,\,\dots,\,x_n \le X_n < x_n+\Delta x_n;\,\theta)
\approx f(x_1;\theta)\cdots f(x_n;\theta)\,|\Delta x_1|\cdots|\Delta x_n|
$$
이므로, 모수의 값이 $\theta$와 $\theta'$인 두 모형에서 $x$가 생성되었을 가능성을 비교하여
$$
\prod_{i=1}^n f(x_i;\theta) > \prod_{i=1}^n f(x_i;\theta') \quad \text{또는} \quad \prod_{i=1}^n f(x_i;\theta) < \prod_{i=1}^n f(x_i;\theta')
$$
인가에 따라 모수의 값이 $\theta$인 모형에서 또는 $\theta'$인 모형에서 관측결과가 생성되었다고 추측할 수 있을 것이다.

이런 가능성의 비교에서는 $\theta \in \Omega$의 변화에 따른 $\prod_{i=1}^n f(x_i;\theta)$의 변화가 주된 관심사이므로, 관측결과 $x$를 고정시키고 $\theta$의 함수로서
$$
L(\theta;x) = \prod_{i=1}^n f(x_i;\theta),\quad \theta \in \Omega
$$
로 정의된 함수를 **가능도함수(likelihood function)** 또는 우도함수라 한다.

그 값이 가장 크게 되는 모수의 값 $\hat\theta(x)$로 추측하자는 것이 **최대가능도 추정법(maximum likelihood estimation, MLE)** 이고, 대응하는 통계량을 최대가능도 추정량이라 한다.
$$
L(\hat\theta(x);x) = \max_{\theta \in \Omega} L(\theta;x)
$$
각각의 관측결과 $x$에 대하여 최대가능도 추정값 $\hat\theta^{\mathrm{MLE}}(x)$가 정의될 때, $\hat\theta^{\mathrm{MLE}} = \hat\theta^{\mathrm{MLE}}(X)$를 최대가능도 추정량 또는 최대우도 추정량이라 한다.

미분의 편의를 위해 보통 곱의 꼴보다는 합의 꼴로 주어지면 미분이 쉬우므로, 가능도함수에 로그를 취하여
$$
\ell(\theta) = \log L(\theta;x) = \sum_{i=1}^n \log f(x_i;\theta)
$$
로 정의된 **로그가능도함수(log likelihood function)** 를 사용한다. 이때 고정된 관측결과 $x$를 생략하고 이들을 간략히 $L(\theta)$ 또는 $\ell(\theta)$로 나타내기도 한다.

#### 예 6.2.1: 포아송 분포에서의 최대가능도 추정
$X_i \sim Poisson(\lambda)\ (\lambda > 0)$인 랜덤표본 $X_1, \dots, X_n$이 주어졌을 때, $\lambda$의 최대가능도 추정량은?  

**풀이**  
포아송 분포의 확률질량함수는 $f(x;\lambda) = \frac{e^{-\lambda}\lambda^x}{x!}$이므로
$$
L(\lambda) = \prod_{i=1}^n \frac{e^{-\lambda}\lambda^{x_i}}{x_i!} = e^{-n\lambda}\lambda^{\sum x_i}/(x_1!\cdots x_n!) \\
\ell(\lambda) = \log L(\lambda) = -n\lambda + \left(\sum_{i=1}^n x_i\right)\log\lambda - \sum_{i=1}^n \log x_i! \\
\ell'(\lambda) = -n + \frac{\sum x_i}{\lambda} = -n + \frac{n\bar X}{\lambda}
$$
일차도함수, 이차도함수는
$$
\ell'(\lambda) = -n + \frac{n\bar X}{\lambda}, \quad
\ell''(\lambda) = -\frac{n\bar X}{\lambda^2}
$$
이로부터 로그가능도함수의 증가, 감소를 조사하면 다음이 성립함을 알 수 있다.

1) $\bar X > 0$인 경우, $\ell(\lambda)$는 $\lambda = \bar X$에서 최대값을 가진다.
2) $\bar X = 0$인 경우, $\ell(\lambda) = -n\lambda - \log(x_1! \cdots x_n!)$로 $\lambda \to 0^+$에서 극대값을 가진다.

그런데 가능도함수는 $\lambda=0$에서 정의되어 있고 연속함수이므로, 어느 경우에나
$$
\sup_{\lambda > 0} L(\lambda; x_1, \dots, x_n) = L(\bar X; x_1, \dots, x_n)
$$
따라서
$$
\max_{\lambda \ge 0} L(\lambda; x_1, \dots, x_n) = L(\bar X; x_1, \dots, x_n) \quad \forall\, x = (x_1, \dots, x_n)^\top
$$
그러므로 $\lambda$의 최대가능도 추정량은
$$
\hat\lambda^{\mathrm{MLE}} = \bar X
$$

이처럼 로그가능도함수의 증가, 감소를 조사하는데 도함수들을 이용한다. 특히 
가능도함수 $L(\theta)$ 또는 로그가능도함수 $\ell(\theta)$를 $\theta$에 대해 최대화하는 문제는, 미분 가능한 경우 **가능도방정식(likelihood equation)**
$$
\frac{\partial}{\partial\theta} \ell(\theta) = 0
$$
의 해를 구하는 것으로 귀결된다. 이 방정식의 해(근) 중에서 실제로 $\ell(\theta)$를 최대화하는 값을 **최대가능도 추정값**으로 선택한다.

이 과정에서 극값의 존재, 유일성, 그리고 극대점임을 판별하는 데 다음의 정리들이 매우 유용하게 사용된다.
### 정리 6.2.1: 최대가능도 추정의 충분조건
$\ell(\theta)$가 열린구간 $\Omega_0$에서 두 번 미분 가능하고, 이차 도함수가 연속함수일 때, 
$$\ell'(\hat\theta)=0,\quad \ell''(\theta)<0 \ \rightarrow \ \hat\theta = \arg\max_{\theta\in\Omega_0} \ell(\theta)$$

#### 증명
$\ell(\theta)$가 $\Omega_0$에서 두 번 미분 가능하다고 가정하자. $\ell'(\hat\theta)=0$이고 $\ell''(\hat\theta)<0$일 때, $\hat\theta$가 극대점임을 보이기 위해 $\theta$를 $\hat\theta$ 근방에서 테일러 전개하면,
$$
\ell(\theta) = \ell(\hat\theta) + \ell'(\hat\theta)(\theta-\hat\theta) + \frac{1}{2}\ell''(\hat\theta)(\theta-\hat\theta)^2 + o((\theta-\hat\theta)^2)
$$
여기서 $\ell'(\hat\theta)=0$이므로,
$$
\ell(\theta) = \ell(\hat\theta) + \frac{1}{2}\ell''(\hat\theta)(\theta-\hat\theta)^2 + o((\theta-\hat\theta)^2)
$$
$\ell''(\hat\theta)<0$이므로 $\theta=\hat\theta$에서 $\ell(\theta)$가 최대값을 가짐을 알 수 있다.

#### 예 6.2.2: 이항분포에서의 최대가능도 추정
$X_i \sim Bernoulli(p)$ $(0 < p < 1)$인 랜덤표본 $X_1, \dots, X_n$이 주어졌을 때, $p$의 최대가능도 추정량은?

**풀이**  
베르누이 분포의 확률질량함수는 $f(x;p) = p^x(1-p)^{1-x}$ $(x=0,1)$이므로, 
$$
L(p) = \prod_{i=1}^n p^{x_i}(1-p)^{1-x_i} = p^{\sum x_i}(1-p)^{n-\sum x_i} \\
\ell(p) = \log L(p) = \left(\sum_{i=1}^n x_i\right)\log p + \left(n-\sum_{i=1}^n x_i\right)\log(1-p)
$$
이로부터 일차도함수, 이차도함수는 각각
$$
\ell'(p) = \frac{\sum x_i}{p} - \frac{n-\sum x_i}{1-p}, \quad
\ell''(p) = -\frac{\sum x_i}{p^2} - \frac{n-\sum x_i}{(1-p)^2}
$$
이다.

이제 $0 < p < 1$일 때, 정리 6.2.1을 적용하여 로그가능도함수의 증가, 감소를 조사하면 다음과 같다.

1) $0 < \hat p < 1$인 경우, $\ell(p)$는 $p = \hat p = \bar X$에서 최대값을 가진다.
2) $\hat p = 0$인 경우, 즉 $x_1 = \cdots = x_n = 0$이면, $\ell(p)$는 $p \to 0^+$에서 극대값을 가진다.
3) $\hat p = 1$인 경우, 즉 $x_1 = \cdots = x_n = 1$이면, $\ell(p)$는 $p \to 1^-$에서 극대값을 가진다.

가능도함수는 $p = 0, 1$에서도 정의되고 연속이므로, 모든 경우에 대해
$$
\sup_{0 \le p \le 1} L(p; x_1, \dots, x_n) = L(\hat p; x_1, \dots, x_n)
$$
또한 $p=0$인 경우는 $x_1 = \dots = x_n = 0$이고, $p=$인 경우는 $x_1 = \dots = x_n = 1$이므로 
$$
\max_{0\leq p\leq 1} L(p; x_1, \dots, x_n) = L(\hat p; x_1, \dots, x_n)$$
따라서 베르누이 분포의 최대가능도 추정량은
$$
\hat p^{\mathrm{MLE}} = \bar X = \frac{1}{n}\sum_{i=1}^n x_i
$$

**참고: 로그가능도함수의 순오목성 *(Strict Concavity of Log-Likelihood)***  
앞서 **정리 6.2.1**에서 사용된 조건  
$$
\ddot l(\theta) < 0,\quad \forall \theta \in \Omega_0
$$  
는 로그가능도함수 $l(\theta)$가 **순오목함수(strictly concave function)** 임을 의미한다.

즉,
$$
l(\alpha \theta_1 + (1-\alpha)\theta_2) > \alpha l(\theta_1) + (1-\alpha) l(\theta_2), 
\quad 0<\alpha<1,\ \theta_1\neq\theta_2
$$
이는 선분의 내부점에서의 함수값이 양 끝점의 함수값의 내분값보다 크다는 것을 의미하며, 다음과 같은 중요한 결론을 준다.

- 로그가능도함수가 순오목이면 가능도방정식 $\dot l(\theta)=0$의 해는 **존재한다면 유일**하다.
- 따라서 해를 찾기만 하면 그것이 곧 최대가능도 추정값이다.

하지만 가능도방정식의 구체적 풀이가 어려울 때도 많고, 그런 경우엔 아래 정리가 매우 유용하다.
#### 정리 6.2.3: 해의 존재와 유일성 *(Existence and Uniqueness of the LE)*
수직선 위의 열린구간 $\Omega_0$에서 정의된 함수 $l(\theta)$가 다음을 만족한다고 하자.

1. $l(\theta)$는 두 번 미분 가능하고 $\ddot l(\theta)$는 연속이다.
2. $\ddot l(\theta) < 0,\ \forall \theta \in \Omega_0$.
3. $\displaystyle \lim_{\theta \to \partial(\Omega_0)} l(\theta) = -\infty$.
    * $\partial(\Omega_0)$는 수직선 위의 구간 $\Omega_0$의 경계(boundary)를 나타내는 기호
    * $\theta \to \partial(\Omega_0)$는 유한구간의 경우엔 구간의 끝점으로 가까이 가고, 무한구간의 경우에는 무한히 커지거나 작아지는 것을 뜻한다.

그러면 방정식  
$$
\dot l(\theta)=0,\quad \theta\in\Omega_0
$$
의 해 $\hat\theta$는 **존재하며 유일**하고,  
$$
l(\hat\theta)=\max_{\theta\in\Omega_0} l(\theta)
$$
를 만족한다.

#### 증명
정의역이 무한구간으로서 $\Omega_0 = (-\infin, +\infin)$인 경우에 증명하기로 한다.  
$\ell(\theta)$는 $\ddot l(\theta)<0$이므로 순오목 함수이다. 또한 $\lim_{\theta\to\pm\infty} l(\theta) = -\infty$이므로, $l(\theta)$는 $\mathbb{R}$ 전체에서 최대값을 갖는다.  
순오목 함수의 최대점은 유일하므로, $\dot l(\theta)=0$의 해 $\hat\theta$가 존재하며 유일하다.  
따라서 $l(\hat\theta)=\max_{\theta\in\Omega_0} l(\theta)$가 성립한다.

정리 6.2.3에서 $\ell(\theta)$가 순오목(strictly concave)이므로, 만약 $\hat\theta_1 \neq \hat\theta_2$에서 모두 최대값을 가진다면,
$$
\ell(\hat\theta_1) = \ell(\hat\theta_2) = M
$$
이고, $0 < \alpha < 1$에 대해 $\theta^* = \alpha\hat\theta_1 + (1-\alpha)\hat\theta_2$라 하면,
$$
\ell(\theta^*) > \alpha \ell(\hat\theta_1) + (1-\alpha)\ell(\hat\theta_2) = M
$$
이 되어 $\theta^*$에서 더 큰 값을 가지므로, $\hat\theta_1, \hat\theta_2$가 최대점이라는 가정과 모순이다.  
따라서 최대점은 유일하다.

### 6.2.4 가능도방정식의 반복해법 *(Iterative Solution of Likelihood Equation)*
가능도방정식  
$$
\dot l(\theta)=0
$$
의 해를 **명시적으로 구하기 어려운 경우**가 많다.

이 경우 테일러 전개를 이용하여  
$$
0 = \dot l(\hat\theta^{(r+1)})
\approx \dot l(\hat\theta^{(r)}) + \ddot l(\hat\theta^{(r)})(\hat\theta^{(r+1)}-\hat\theta^{(r)})
$$
로부터 다음 반복식을 얻는다.
$$
\hat\theta^{(r+1)}
=
\hat\theta^{(r)}
-
\left[\ddot l(\hat\theta^{(r)})\right]^{-1}
\dot l(\hat\theta^{(r)}),
\quad r=0,1,\dots
$$

이를 **뉴턴–랩슨 방법(Newton–Raphson method)** 또는 **일단계 반복법(one-step iteration)** 이라 한다.  
초기값 $\hat\theta^{(0)}$로는 흔히 **적률이용 추정값(MME)** 을 사용한다.

#### 예 6.2.3: 로지스틱분포에서의 최대가능도 추정
로지스틱분포 $L(\theta,1)$ $( -\infty<\theta<\infty )$에서의 랜덤표본 $X_1,\dots,X_n$에 대하여  
$$
pdf(x;\theta)
=\frac{e^{-x+\theta}}{(1+e^{-x+\theta})^2},
\quad -\infty<x<\infty \\
l(\theta)
=n\bar x - n\theta - 2\sum_{i=1}^n \log(1+e^{x_i-\theta}) \\
\dot l(\theta)
=n - 2\sum_{i=1}^n \frac{e^{-x_i+\theta}}{1+e^{-x_i+\theta}},
\quad
\ddot l(\theta)
=-2\sum_{i=1}^n \frac{e^{-x_i+\theta}}{(1+e^{-x_i+\theta})^2} < 0
$$
또한  
$$
\lim_{\theta\to\pm\infty} l(\theta) = -\infty
$$
이므로 **정리 6.2.3**을 적용할 수 있다.

따라서 최대가능도 추정값은 가능도방정식  
$$
n - 2\sum_{i=1}^n \frac{e^{-x_i+\theta}}{1+e^{-x_i+\theta}}=0
$$
의 유일한 해이며, 로지스틱분포 $L(\theta, 1)$의 평균은 $\theta$이므로, 적률이용추정값은 $\bar x$다. 따라서
$$
\hat\theta^{(r+1)}
=\hat\theta^{(r)}
-
\left[\ddot l(\hat\theta^{(r)})\right]^{-1}
\dot l(\hat\theta^{(r)}),
\quad
\hat\theta^{(0)}=\bar x
$$

#### 정리 6.2.4: 모수의 일대일 변환과 최대가능도 추정 *(Invariance roperty)*
모수 $\theta$의 최대가능도 추정량 $\hat\theta^{\mathrm{MLE}}$이 존재할 때,  
$\eta=g(\theta)$가 $\theta$의 **일대일 변환(one-to-one transformation)**이면  
$$
\hat\eta^{\mathrm{MLE}} = g(\hat\theta^{\mathrm{MLE}})
$$
이다.

*증명 개요*  
- 확률밀도함수는 모수의 일대일 변환에 의해 형태가 변하지 않는다.  
- 가능도함수의 최대점은 동일하게 유지된다.

#### 예 6.2.4: 지수분포에서의 최대가능도 추정
지수분포 $Exp(\theta)$ $(\theta>0)$의 확률밀도함수는  
$$
pdf(x;\theta)=\frac{1}{\theta}e^{-x/\theta},\quad x>0
$$

직접 $\theta$를 모수로 하면 로그가능도함수는 순오목이 아니다.  
따라서 $\lambda=1/\theta$로 변환하면  
$$
l(\lambda)=n\log\lambda-n\bar x\lambda,
\quad
\ddot l(\lambda)=-\frac{n}{\lambda^2}<0
$$

이로부터  
$$
\hat\lambda^{\mathrm{MLE}}=\frac{1}{\bar X}
$$
이고, 정리 6.2.4에 의해  
$$
\hat\theta^{\mathrm{MLE}}=\bar X
$$
이다.

#### 정리 6.2.5: 지수족에서의 최대가능도 추정 *(Exponential Family)*
확률밀도함수가  
$$
pdf(x;\eta)
=
\exp\{\eta T(x)-A(\eta)+S(x)\},
\quad x\in\mathcal X,\ \eta\in N
$$
의 꼴이며 다음을 만족한다고 하자.

1. 지지집합 $\mathcal X$는 $\eta$에 의존하지 않는다.
2. $N$은 열린구간이다.
3. $\mathrm{Var}_\eta(T(X_1))>0$.

그러면 최대가능도 추정값 $\hat\eta$는  
$$
A'(\hat\eta)=\frac{1}{n}\sum_{i=1}^n T(x_i)
$$
를 만족한다.

*증명 개요*  
누율생성함수(cumulant generating function)  
$$
cgf_{T(X_1)}(s)=A(s+\eta)-A(\eta)
$$
의 성질을 이용한다.

#### 예 6.2.5: 지수족의 구체적 예
- 베르누이 분포 $Bernoulli(p)$
- 포아송 분포 $Poisson(\lambda)$
- 기하분포 $Geo(p)$
- 지수분포 $Exp(\theta)$
- 파레토 분포 $Pareto(1,\theta)$

모두 지수족에 속하며,  
$$
E_\theta[T(X_1)]
=
\frac{1}{n}\sum_{i=1}^n T(x_i)
$$
형태의 가능도방정식을 갖는다.

#### 예 6.2.6: 이중지수분포에서의 최대가능도 추정 *(Laplace Distribution)*
이중지수분포 $DE(\theta,1)$의 확률밀도함수는  
$$
pdf(x;\theta)=\frac12 e^{-|x-\theta|}
$$

로그가능도함수는  
$$
l(\theta)=-\sum_{i=1}^n|x_i-\theta|-n\log2
$$
이며, 이는 미분 불가능하다.

순서통계량 $X_{(1)}<\cdots<X_{(n)}$에 대해 $n=2m+1$이면  
$$
\hat\theta^{\mathrm{MLE}}=X_{(m+1)}
$$
즉, **표본중앙값(sample median)**이다.

#### 예 6.2.7: 균등분포에서의 최대가능도 추정
$U[0,\theta]$에서  
$$
L(\theta)=\theta^{-n}I(\max X_i\le\theta)
$$
이므로  
$$
\hat\theta^{\mathrm{MLE}}=X_{(n)}=\max X_i
$$

#### 예 6.2.8: 최대가능도 추정량이 유일하지 않은 경우
$U[\theta-1,\theta+1]$에서  
$$
L(\theta)=2^{-n}I(X_{(n)}-1\le\theta\le X_{(1)}+1)
$$

이 구간 내의 모든 값이 최대가능도 추정값이다.  
즉,  
$$
X_{(n)}-1\le \hat\theta^{\mathrm{MLE}}\le X_{(1)}+1
$$
를 만족하는 모든 통계량이 MLE이다.


## 다차원 모수의 최대가능도 추정 *(MLE for Multidimensional Parameters)*

## 최대가능도 추정량의 점근적 성질 *(Asymptotic Properties of Maximum Likelihood Estimators)*

## 최소제곱 추정법 *(Least Squares Estimation)*
