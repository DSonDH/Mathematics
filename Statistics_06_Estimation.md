# 제6장 추정 *(Statistical Estimation)*

## 6.1 적률이용 추정법 *(Method of Moments Estimation, MME)*

### 6.1.1 도입 및 기본 개념
**적률이용 추정법(Method of Moments Estimation, MME)** 은 모집단의 특성을 나타내는 **모수(parameter)** 를 표본으로부터 추정하는 대표적인 방법 중 하나이다. 이 방법은 모집단의 **적률(moment)** 과 이에 대응하는 **표본적률(sample moment)** 을 일치시키는 방식으로 모수를 추정한다.

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
> **참고: 가능도(likelihood)와 확률(probability)의 차이**
>
> - **확률(probability)** 은 모수 $\theta$가 고정되어 있을 때, 어떤 데이터 $x$가 관측될 확률을 의미한다. 즉, $P(X = x \mid \theta)$처럼 **데이터의 함수**이다.
> - **가능도(likelihood)** 는 관측된 데이터 $x$가 고정되어 있을 때, 모수 $\theta$의 값을 바꿔가며 "이 데이터가 얼마나 그럴듯한가"를 측정하는 함수다. 즉, $L(\theta; x) = f(x; \theta)$처럼 **모수의 함수**이다.
> - 가능도는 "확률"이 아니며, $0$과 $1$ 사이의 값일 필요도 없다. 단지, 주어진 데이터에서 어떤 $\theta$가 더 그럴듯한지(가능성이 높은지)를 비교하는 기준이다.
> - 최대가능도 추정(MLE)은 "관측된 데이터를 가장 그럴듯하게 만드는 모수값"을 찾는 방법이다.

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
### 정리 6.2.3: 해의 존재와 유일성 *(Existence and Uniqueness of the LE)*
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
= \frac{e^{x-\theta}}{(1+e^{x-\theta})^2} =\frac{e^{-x+\theta}}{(1+e^{-x+\theta})^2},
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

### 정리 6.2.4: 모수의 일대일 변환과 최대가능도 추정 *(Invariance roperty)*
모수 $\theta$의 최대가능도 추정량 $\hat\theta^{\mathrm{MLE}}$이 존재할 때,  
$\eta=g(\theta)$가 $\theta$의 **일대일 변환(one-to-one transformation)** 이면  
$$
\hat\eta^{\mathrm{MLE}} = g(\hat\theta^{\mathrm{MLE}})
$$

*증명 개요*  
- 확률밀도함수는 모수의 일대일 변환에 의해 형태가 변하지 않는다.  
- 가능도함수의 최대점은 동일하게 유지된다.
(교재의 증명 한문단 생략)

#### 예 6.2.4: 지수분포에서의 최대가능도 추정
지수분포 $Exp(\theta)$ $(\theta>0)$의 확률밀도함수, 로그가능도함수는  
$$
pdf(x;\theta)=\frac{1}{\theta}e^{-x/\theta},\quad x>0 \\
l(\theta) = -n\log\theta - \frac{n\bar x}{\theta} \\
\ddot l(\theta) = \frac{n}{\theta^2} - \frac{2n\bar x}{\theta^3}
$$
로, $\theta > 0$에서 항상 음이 아님을 알 수 있다. 즉, $\bar x > \theta/2$일 때만 $\ddot l(\theta) < 0$이므로, 전체 구간에서 순오목함수가 아니다.  
따라서 $\theta$에 대해 정리 6.2.3을 바로 적용할 수 없고, 증가·감소를 직접 조사해야 한다.

이때 $\lambda = 1/\theta$로 치환하면  
$$
l(\lambda) = n\log\lambda - n\bar x\lambda \\
\ddot l(\lambda) = -\frac{n}{\lambda^2} < 0
$$
로, $\lambda > 0$에서 항상 음이므로 순오목함수가 된다. 따라서 정리 6.2.3을 적용할 수 있다.

가능도방정식  
$$
\frac{d}{d\lambda} l(\lambda) = \frac{n}{\lambda} - n\bar x = 0
$$
의 해는  
$$
\hat\lambda^{\mathrm{MLE}} = \frac{1}{\bar X}
$$
이고, 정리 6.2.4(불변성)에 의해  
$$
\hat\theta^{\mathrm{MLE}} = \bar X
$$

>**지수족(Exponential Family)이란?**  
>확률밀도함수 또는 확률질량함수가 아래와 같은 꼴로 쓸 수 있는 분포의 집합
>$$
>f(x;\eta) = \exp\{\eta T(x) - A(\eta) + S(x)\}
>$$
>여기서 $\eta$는 모수(parameter), $T(x)$는 충분통계량, $A(\eta)$는 정규화 상수, $S(x)$는 $x$에만 의존하는 함수.
>- $T(x)$: **충분통계량(sufficient statistic)** 역할을 하는 함수로, 표본 $x$의 정보를 요약하여 모수 $\eta$와 관련된 부분만 남기는 함수. 예를 들어, 베르누이 분포에서는 $T(x) = x$.
>- $A(\eta)$: **정규화 상수(normalizing constant)** 또는 **누율생성함수(cumulant generating function)**로, 분포가 적분해서 1이 되도록 조정하는 $\eta$의 함수.
>- $S(x)$: $x$에만 의존하는 함수로, 모수 $\eta$와는 무관한 부분. (많은 경우 $S(x)=0$이거나 상수.)
>
>**지수족의 예시:**  
>베르누이, 포아송, 지수, 정규(평균 또는 분산 고정), 감마, 이항, 다항, 기하, 파레토 등 많은 분포가 있다.
>
>**지수족의 장점:**  
>충분통계량이 존재하고, 최대가능도 추정이 단순해지며, 통계적 추론이 수학적으로 다루기 쉬워짐.
### 정리 6.2.5: 지수족에서의 최대가능도 추정 *(Exponential Family)*
$$
pdf(x;\eta)
=
\exp\{\eta T(x)-A(\eta)+S(x)\},
\quad x\in\mathcal X,\ \eta\in N
$$
확률밀도함수 형태이 이러하며, 다음을 만족한다고 하자.
1. 지지집합 $\mathcal X$는 $\eta$에 의존하지 않는다. 즉 분포의 토대 $\mathcal X$가 모수에 따라 변하지 않음.
2. $N$은 열린구간이다. 모수공간 N이 수직선 위의 열린구간.
3. $\mathrm{Var}_\eta(T(X_1))>0$. 즉 $T(x)$는 상수 아님.

(참고: 이런 조건을 만족하는 확률밀도함수 형태의 집합을 단일모수 지수족(single parameter exponential family of pdf's)라 한다.)  

여기서 가능도방정식의 근 $\hat\eta$이 존재하면 $\hat\eta$는 $\eta$의 최대가능도 추정값이다.
$$
A'(\hat\eta)=E_{\eta}[T(X_1)] = \frac{1}{n}\sum_{i=1}^n T(x_i)
$$  

#### 증명
가능도함수의 로그를 $\eta$에 대해 미분하면 $A'(\eta) = E_\eta[T(X_1)]$가 된다.  
가능도함수의 로그는  
$$
l(\eta) = n\eta \bar T - nA(\eta) + \sum_{i=1}^n S(x_i)
$$
이고, $\bar T = \frac{1}{n}\sum_{i=1}^n T(x_i)$.

미분하면  
$$
\frac{d}{d\eta} l(\eta) = n\bar T - nA'(\eta)
$$
가능도방정식은  
$$
A'(\hat\eta) = \bar T
$$

조건 1: $\mathcal X$가 $\eta$에 의존하지 않으므로 $l(\eta)$는 $\eta$에 대해 두 번 미분 가능.  
조건 2: $N$이 열린구간이므로 극대점이 내부에 존재.  
조건 3: $\mathrm{Var}_\eta(T(X_1)) > 0$이므로 $A''(\eta) > 0$이고, $l(\eta)$는 순오목함수.

따라서 $A'(\hat\eta) = \bar T$의 해 $\hat\eta$는 존재하며 유일하고, 최대가능도 추정값이 된다.

#### 지수족에서 모수의 일대일 변환과 가능도방정식
$$
pdf(x;\theta) = \exp\{g(\theta) T(x) - A(\eta(\theta)) + S(x)\}
$$
이고, $\eta = g(\theta)$로 모수화하여 정리 6.2.5의 조건이 만족되면, $\theta$를 직접 모수로 쓸 때도 가능도방정식은  
$$
E_\theta[T(X_1)] = \frac{1}{n}\sum_{i=1}^n T(x_i)
$$
즉, $\theta$에 대한 가능도방정식의 해 $\hat\theta$가 존재하면, $\hat\theta$는 $\theta$의 최대가능도 추정값이 된다.  
불변성 정리(정리 6.2.4)에 의해, $g = g(\theta)$의 최대가능도 추정량은 $g(\hat\theta)$로 얻어진다.

#### 예 6.2.5: 정리6.2.5 조건을 만족시키는 지수족의 구체적 예
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

**1. 베르누이 분포 $Bernoulli(p)$**  
$$
\text{모수 변환:}\quad \eta = \log\left(\frac{p}{1-p}\right)
$$

$$
f(x;\eta) = \left(\frac{e^\eta}{1+e^\eta}\right)^x \left(\frac{1}{1+e^\eta}\right)^{1-x},\quad x=0,1
$$

$$
L(\eta) = \prod_{i=1}^n \left(\frac{e^\eta}{1+e^\eta}\right)^{x_i} \left(\frac{1}{1+e^\eta}\right)^{1-x_i}
= \left(\frac{e^\eta}{1+e^\eta}\right)^{\sum x_i} \left(\frac{1}{1+e^\eta}\right)^{n-\sum x_i} \\
\ell(\eta) = \left(\sum x_i\right)\eta - n\log(1+e^\eta) \\
\frac{d}{d\eta}\ell(\eta) = \sum x_i - n\frac{e^\eta}{1+e^\eta} = 0
$$

정리하면:
$$
\frac{e^\eta}{1+e^\eta} = \frac{1}{n}\sum_{i=1}^n x_i = \bar X
$$

따라서 최대가능도 추정량은
$$
\hat\eta^{\mathrm{MLE}} = \log\left(\frac{\bar X}{1-\bar X}\right)
$$

즉, $p$의 최대가능도 추정량 $\hat p^{\mathrm{MLE}} = \bar X$은 모수변환 전의 결과와 같음을 볼 수 있다.

**2. 포아송 분포 $Poisson(\lambda)$**  
모수 변환: $\eta = \log \lambda$  
$$
f(x;\eta) = \frac{e^{-e^\eta}e^{\eta x}}{x!},\quad x=0,1,2,\dots
$$
$$
L(\eta) = \prod_{i=1}^n \frac{e^{-e^\eta}e^{\eta x_i}}{x_i!} = e^{-n e^\eta} e^{\eta \sum x_i} / \prod x_i!
$$
$$
\ell(\eta) = -n e^\eta + \left(\sum x_i\right)\eta - \sum \log x_i!
$$
미분하여 0으로 두면:
$$
\frac{d}{d\eta}\ell(\eta) = -n e^\eta + \sum x_i = 0 \\
e^\eta = \frac{1}{n}\sum x_i = \bar X \\
\hat\eta^{\mathrm{MLE}} = \log \bar X
$$
따라서 $\lambda = e^\eta$의 최대가능도 추정량은 $\hat\lambda^{\mathrm{MLE}} = e^{\hat\eta^{\mathrm{MLE}}} = \bar X$로, 변환 전과 동일하다.

**3. 기하분포 $Geo(p)$**  
모수 변환: $\eta = \log(1-p)$  
$$
f(x;\eta) = e^{\eta(x-1)}(1-e^\eta),\quad x=1,2,\dots,\ \eta<0
$$
$$
L(\eta) = (1-e^\eta)^n e^{\eta \sum (x_i-1)}
$$
$$
\ell(\eta) = n\log(1-e^\eta) + \eta \sum (x_i-1)
$$
미분하여 0으로 두면:
$$
\frac{d}{d\eta}\ell(\eta) = -\frac{n e^\eta}{1-e^\eta} + \sum (x_i-1) = 0 \\
\frac{n e^\eta}{1-e^\eta} = \sum (x_i-1) \\
\frac{n}{1-e^{-\eta}} = \sum x_i \\
e^{-\eta} = 1 - \frac{n}{\sum x_i} \\
p = 1 - e^\eta = \frac{n}{\sum x_i}
$$
따라서 최대가능도 추정량은 $\hat p^{\mathrm{MLE}} = \frac{n}{\sum x_i}$로, 변환 전과 동일하다.

**4. 지수분포 $Exp(\theta)$**  
모수 변환: $\eta = -1/\theta$  
$$
f(x;\eta) = -\eta e^{\eta x},\quad x>0,\ \eta<0
$$
$$
L(\eta) = (-\eta)^n e^{\eta \sum x_i}
$$
$$
\ell(\eta) = n\log(-\eta) + \eta \sum x_i
$$
미분하여 0으로 두면:
$$
\frac{d}{d\eta}\ell(\eta) = -\frac{n}{\eta} + \sum x_i = 0 \\
-\frac{n}{\eta} + \sum x_i = 0 \implies \eta = -\frac{n}{\sum x_i}
$$
따라서 $\theta = -1/\eta = \frac{1}{n}\sum x_i = \bar X$로, 변환 전과 동일하다.


**5. 파레토 분포 $Pareto(1,\theta)$**  
파레토 분포 $Pareto(1,\theta)$의 확률밀도함수는  
$$
f(x;\theta) = \theta x^{-\theta-1},\quad x \ge 1,\ \theta > 0
$$
즉, 파레토 분포는 $\theta$와 $x$의 상호작용 항이 없으므로 모수변환 없이 바로 최대가능도 추정량을 구할 수 있다.

가능도함수와 로그가능도함수는  
$$
L(\theta) = \prod_{i=1}^n \theta x_i^{-\theta-1} = \theta^n \prod_{i=1}^n x_i^{-\theta-1}
$$
$$
\ell(\theta) = n\log\theta - (\theta+1)\sum_{i=1}^n \log x_i
$$

로그가능도함수를 $\theta$에 대해 미분하여 0으로 두면  
$$
\frac{d}{d\theta}\ell(\theta) = \frac{n}{\theta} - \sum_{i=1}^n \log x_i = 0
$$
따라서  
$$
\hat\theta^{\mathrm{MLE}} = \frac{n}{\sum_{i=1}^n \log x_i}
$$

> 지금까지는 가능도함수가 미분가능한 경우였다. 미분 불가능하거나 불연속인 경우에는 가능도함수 또는 로그가능도함수의 증가, 감소를 조사하여 최대가능도 추정값을 찾아야한다. 최대가능도 추정량이 유일하지 않을수도 있다!
#### 예 6.2.6: 이중지수분포에서의 최대가능도 추정 *(Laplace Distribution)*
이중지수분포 $DE(\theta,1)$로부터의 랜덤표본을 $X_1, \dots, X_n$ ($n=2m+1$)이라 할 때, $\theta$의 최대가능도 추정량은?

**풀이**  
$$
f(x;\theta) = \frac{1}{2}e^{-|x-\theta|},\quad -\infty < x < \infty \\
L(\theta) = \prod_{i=1}^n \frac{1}{2}e^{-|x_i-\theta|} = 2^{-n} \exp\left(-\sum_{i=1}^n |x_i-\theta|\right)\\
\ell(\theta) = -n\log 2 - \sum_{i=1}^n |x_i-\theta|
$$
따라서 $\theta$의 최대가능도 추정량은 $\sum_{i=1}^n |x_i-\theta|$를 최소로 하는 $\theta$, 즉 **표본의 중앙값(median)** 이다.  
$n=2m+1$ (홀수)일 때, 오름차순으로 정렬한 표본의 $m+1$번째 값이 최대가능도 추정량이 된다.
> **참고: 절댓값의 합을 최소로 하는 값은 중앙값이다 (중앙값 최소화 법칙)**  
> 임의의 실수 $x_1, \dots, x_n$에 대해 함수 $S(\theta) = \sum_{i=1}^n |x_i - \theta|$를 최소로 하는 $\theta$는 $x_i$들의 **중앙값(median)** 이다.  
>  
> **이유:**  
> $S(\theta)$는 $\theta$가 $x_i$를 지날 때마다 기울기가 $\pm1$씩 변하는 조각별 선형 함수(piecewise linear function)로,  
> $S(\theta)$의 최소값은 $x_i$들 중 중앙값에서 달성된다.  
>  
> **수학적 근거:**  
> $S(\theta)$를 $\theta$에 대해 미분하면,  
> $$
> S'(\theta) = \sum_{i=1}^n \mathrm{sgn}(\theta - x_i)
> $$
> 이고, $S'(\theta) = 0$이 되는 $\theta$가 바로 중앙값이다.  

#### 예 6.2.7: 균등분포에서의 최대가능도 추정
$X_1, \dots, X_n$이 $U[0, \theta]$에서의 랜덤표본이라고 하자. $\theta$의 최대가능도 추정량을 구하라.

**풀이**  
균등분포 $U[0, \theta]$의 확률밀도함수는
$$
f(x;\theta) = \begin{cases}
\frac{1}{\theta}, & 0 \le x \le \theta \\
0, & \text{otherwise}
\end{cases} \\
L(\theta) = \prod_{i=1}^n f(x_i;\theta) = \begin{cases}
\theta^{-n}, & \theta \ge X_{(n)} \\
0, & \text{otherwise}
\end{cases}
$$
여기서 $X_{(n)} = \max\{x_1, \dots, x_n\}$이다.

즉, 모든 표본값이 $\theta$ 이하일 때만 가능도함수가 0이 아니고, $\theta$가 $X_{(n)}$ 이상일 때 $L(\theta) = \theta^{-n}$이다. $\theta$가 커질수록 $L(\theta)$는 감소하므로, 최대값은 $\theta = X_{(n)}$에서 달성된다.  
따라서 $\theta$의 최대가능도 추정량은 $X_{(n)}$이다.
$$
\hat\theta^{\mathrm{MLE}} = X_{(n)} = \max\{X_1, \dots, X_n\}
$$

#### 예 6.2.8: 최대가능도 추정량이 유일하지 않은 경우
$X_1, \dots, X_n$이 $U[\theta-1,\,\theta+1]$에서의 랜덤표본이라고 하자. $\theta$의 최대가능도 추정량을 구하라.

**풀이**  
균등분포 $U[\theta-1,\,\theta+1]$의 확률밀도함수는
$$
f(x;\theta) = \begin{cases}
\frac{1}{2}, & \theta-1 \le x \le \theta+1 \\
0, & \text{otherwise}
\end{cases} \\
L(\theta) = \prod_{i=1}^n f(x_i;\theta) = \begin{cases}
2^{-n}, & \theta-1 \le x_i \le \theta+1\ \forall i \\
0, & \text{otherwise}
\end{cases}
$$
즉,
$$
\theta \ge \max_i(x_i) - 1 \quad\text{and}\quad \theta \le \min_i(x_i) + 1
$$
따라서
$$
X_{(n)} - 1 \le \theta \le X_{(1)} + 1
$$
($X_{(1)} = \min_i(x_i)$, $X_{(n)} = \max_i(x_i)$)

이 구간 내의 모든 $\theta$에 대해 $L(\theta) = 2^{-n}$로 최대값을 가지므로,  
$$
X_{(n)} - 1 \le \hat\theta^{\mathrm{MLE}} \le X_{(1)} + 1
$$
를 만족하는 모든 값이 최대가능도 추정량이 된다.

즉, **최대가능도 추정량이 유일하지 않고, 위 구간 내의 모든 값이 MLE**이다.


## 6.3 다차원 모수의 최대가능도 추정 *(Maximum Likelihood Estimation for Multidimensional Parameters)*

### 6.3.1 다차원 모수의 최대가능도 추정 *(MLE for Multidimensional Parameters)*
정규분포 $N(\mu, \sigma^2)$에서 $(\mu, \sigma^2)$처럼 **모수가 다차원 벡터(multidimensional vector)** 인 경우에도 최대가능도 추정(maximum likelihood estimation, MLE)은 일변량과 동일한 원리로 정의된다.

#### (a) 가능도함수와 로그가능도함수 *(Likelihood and Log-Likelihood Functions)*

모집단 분포의 확률밀도함수가 $f(x; \theta), \ \theta = (\theta_1, \dots, \theta_k)^\top \in \Omega$
이고, 랜덤표본 $X_1, \dots, X_n$의 관측값이 $x = (x_1, \dots, x_n)$일 때, **가능도함수(likelihood function)** 는
$$
L(\theta; x) = \prod_{i=1}^n f(x_i; \theta) \\
l(\theta; x) = \log L(\theta; x) = \sum_{i=1}^n \log f(x_i; \theta)
$$

#### (b) 모수 벡터의 최대가능도 추정 *(MLE of Parameter Vector)*
가능도함수를 최대화하는
$$
\hat\theta^{\mathrm{MLE}}(x) = \arg\max_{\theta \in \Omega} L(\theta; x)
$$
를 **최대가능도 추정값(maximum likelihood estimator, MLE)** 이라 한다.

#### (c) 각 성분의 최대가능도 추정 *(MLE for Each Component)*
$\hat\theta^{\mathrm{MLE}}(x)$의 각 성분은 대응하는 모수의 최대가능도 추정값이 된다:
$$
\hat\theta^{\mathrm{MLE}}(x) = \left(
\hat\theta_1^{\mathrm{MLE}}(x), \dots, \hat\theta_k^{\mathrm{MLE}}(x)
\right)^\top
$$

> **설명:** 다차원 모수의 최대가능도 추정값을 구하는 과정은 다변수 함수의 극댓값을 구하는 것과 같다. 각 변수에 대해 편미분하여 **가능도방정식(likelihood equation)** 을 세우고, 그 해를 찾는다.

#### 예시 6.3.1: 이중지수분포에서의 최대가능도 추정 *(MLE for Double Exponential/Laplace Distribution)*
이중지수분포(double exponential distribution, Laplace distribution) $DE(\mu, \sigma), \ -\infty < \mu < +\infty,\ \sigma > 0$
로부터의 랜덤표본을$X_1, \dots, X_n \quad (n = 2m + 1)$이라 할 때, $\theta = (\mu, \sigma)^\top$의 최대가능도 추정량을 구하여라.


이중지수분포 $DE(\mu, \sigma)$의 확률밀도함수는
$$
pdf(x; \theta) = \frac{1}{2\sigma} e^{-|x-\mu|/\sigma}, \qquad -\infty < x < +\infty \\
l(\mu, \sigma) = -\sum_{i=1}^n \frac{|x_i - \mu|}{\sigma} + n\log\sigma + n\log 2
$$
<예시 6.2.5>에서와 같이, 이 로그가능도함수는 $\mu$에 대해 **중앙값**에서 최대가 됨을 알 수 있다.
$$
\hat\mu = x_{(m+1)} \quad (n = 2m + 1)
$$
이제 $\sigma$에 대해 $l(\hat\mu, \sigma)$의 증가·감소를 조사하면,
$$
\frac{\partial}{\partial\sigma} l(\hat\mu, \sigma) = -\frac{1}{\sigma^2} \sum_{i=1}^n |x_i - \hat\mu| + \frac{n}{\sigma}
$$
이므로, $l(\hat\mu, \sigma)$는
$$
\hat\sigma = \frac{1}{n} \sum_{i=1}^n |x_i - \hat\mu|
$$
에서 최대가 된다.

따라서 최대가능도 추정량은
$$
\hat\theta^{\mathrm{MLE}} =
\begin{pmatrix}
\hat\mu^{\mathrm{MLE}} \\
\hat\sigma^{\mathrm{MLE}}
\end{pmatrix}
=
\begin{pmatrix}
X_{(m+1)} \\
\frac{1}{n} \sum_{i=1}^n |X_i - X_{(m+1)}|
\end{pmatrix}
$$

#### 예시 6.3.2: 정규분포에서의 최대가능도 추정 *(MLE for Normal Distribution)*
정규분포(normal distribution) $N(\mu,\sigma^2),\ -\infty<\mu<+\infty,\ \sigma^2>0$로부터의 랜덤표본 $X_1,\dots,X_n$이 주어졌을 때, $\theta=(\mu,\sigma^2)^\top$의 최대가능도 추정량을 구하여라.

정규분포의 확률밀도함수는
$$
pdf(x;\theta) = (2\pi\sigma^2)^{-1/2} \exp\left\{ -\frac{1}{2}\frac{(x-\mu)^2}{\sigma^2} \right\},\qquad -\infty < x < +\infty \\
l(\mu,\sigma^2) = -\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2 - \frac{n}{2}\log\sigma^2 - \frac{n}{2}\log 2\pi
$$
(1) $\mu$에 대한 최대화:
$$
\sum_{i=1}^n (x_i-\mu)^2 = \sum_{i=1}^n (x_i-\bar x)^2 + n(\bar x-\mu)^2
$$
따라서 $\mu$에 대해 최대화하면
$$
\hat\mu = \bar x = \frac{1}{n}\sum_{i=1}^n x_i
$$
에서 최대가 된다.

(2) $\sigma^2$에 대한 최대화:

$\mu = \hat\mu$로 고정하고 $\sigma^2$에 대해 미분하면,
$$
\frac{\partial}{\partial\sigma^2}l(\hat\mu,\sigma^2) = -\frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (x_i-\bar x)^2 - \frac{n}{2\sigma^2}
$$
이므로, 증가·감소를 조사하면
$$
\hat\sigma^2 = \frac{1}{n}\sum_{i=1}^n (x_i-\bar x)^2
$$
에서 최대가 된다.

따라서 최대가능도 추정량은
$$
\hat\theta^{\mathrm{MLE}} =
\begin{pmatrix}
\hat\mu^{\mathrm{MLE}} \\
\hat\sigma^{2,\mathrm{MLE}}
\end{pmatrix}
=
\begin{pmatrix}
\bar X \\
\frac{1}{n}\sum_{i=1}^n (X_i-\bar X)^2
\end{pmatrix}
$$

### 정리 6.3.1 모수의 함수와 최대가능도 추정 *(Function of Parameter and Maximum Likelihood Estimation)*
모수 벡터 $\theta = (\theta_1, \dots, \theta_k)^\top$의 최대가능도 추정량 $\hat\theta^{\mathrm{MLE}}$이 존재한다고 하자.  
또한 $\eta = g(\theta) = (g_1(\theta), \dots, g_m(\theta))^\top$가 $\theta$의 **일대일 변환(one-to-one transformation)** 이라고 하자.  
그러면 $\eta$의 최대가능도 추정량은  
$$
\hat\eta^{\mathrm{MLE}} = g(\hat\theta^{\mathrm{MLE}}) = \left(
g_1(\hat\theta^{\mathrm{MLE}}), \dots, g_m(\hat\theta^{\mathrm{MLE}})
\right)^\top
$$  
특히, $\eta_j = g_j(\theta)$일 때  
$$
\hat\eta_j^{\mathrm{MLE}} = g_j(\hat\theta^{\mathrm{MLE}})
$$  
#### 증명
$\eta = g(\theta)$가 일대일이므로, $\theta$와 $\eta$ 사이에 역함수 $\theta = h(\eta)$가 존재한다.  
따라서  
$$
L_\eta(\eta; x) = L_\theta(h(\eta); x) \\
\max_\eta L_\eta(\eta; x) = \max_\theta L_\theta(\theta; x)
$$  
그러므로 $\theta$의 최대가능도 추정값 $\hat\theta^{\mathrm{MLE}}$에 대응하는  
$$
\hat\eta^{\mathrm{MLE}} = g(\hat\theta^{\mathrm{MLE}})
$$  
가 $\eta$의 최대가능도 추정값이 된다.

#### 예시 6.3.3 모수의 함수에 대한 최대가능도 추정 *(MLE for Function of Parameter)*
(1) 정규분포에서 $|\mu|$의 최대가능도 추정  
정규분포 $N(\mu, \sigma^2)$에서 모수 $\theta = (\mu, \sigma^2)^\top$를 생각하자.  
이때 $\eta = (|\mu|, \operatorname{sgn}(\mu), \sigma^2)^\top$ 는 $\theta$의 일대일 변환.

앞의 예시 6.3.2로부터 
$
\hat\mu^{\mathrm{MLE}} = \bar X, \
\hat\sigma^{2, \mathrm{MLE}} = \frac{1}{n} \sum_{i=1}^n (X_i - \bar X)^2
$이므로, 정리 6.3.1에 의해  

$$
\widehat{\eta}^{\mathrm{MLE}} = |\hat\mu^{\mathrm{MLE}}| = |\bar X|
, \quad \widehat{\operatorname{sgn}(\mu)}^{\mathrm{MLE}} = \operatorname{sgn}(\bar X), \quad \widehat{\sigma^2}^{\mathrm{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar X)^2
$$  

(2) 베르누이 분포에서 분산의 최대가능도 추정  
$Bernoulli(p), \ 0 < p < 1$ 에서 $\theta = p$ 라고 하자.  
분산(variance)은 $\sigma^2 = p(1-p)$  
베르누이 분포의 최대가능도 추정량은 $\hat p^{\mathrm{MLE}} = \bar X$  
이므로, 정리 6.3.1에 의해  

$$
\hat\sigma^{2, \mathrm{MLE}} = \hat p^{\mathrm{MLE}} (1 - \hat p^{\mathrm{MLE}}) = \bar X (1 - \bar X)
$$  

### 6.3.2 다차원 가능도방정식과 최대값 판정 *(Multivariate Likelihood Equation and Maximum Test)*
다차원 모수의 최대가능도 추정값은 로그가능도함수의 최대값을 주는 점으로 정의된다.  
로그가능도함수가 $\Omega$의 내부점에서 미분 가능하다면, 최대가능도 추정값 $\hat\theta$는 다음 가능도방정식(likelihood equations)을 만족한다.
$$
\dot l(\theta; x) =
\begin{pmatrix}
\frac{\partial l}{\partial\theta_1} \\
\frac{\partial l}{\partial\theta_2} \\
\vdots \\
\frac{\partial l}{\partial\theta_k}
\end{pmatrix}
= 0
$$
즉,
$$
\frac{\partial l(\theta; x)}{\partial\theta_j} = 0, \qquad j = 1, \dots, k
$$
를 동시에 만족하는 해가 최대가능도 추정값의 후보가 된다.

### 정리 6.3.2 다차원 최대값 판정 정리 *(Multivariate Maximum Test Theorem)*

열린 집합 $\Omega_0 \subset \mathbb{R}^k$ (열린구간다면체)에서 정의된 함수 $l(\theta)$에 대해 다음을 가정하자.

1. $l(\theta)$는 $\Omega_0$에서 두 번 연속 미분 가능하고 이차편도함수들이 연속함수다.
  - $\Omega_0 = (a_1, b_1)\times \dots \times (a_k, b_k)$
2. 어떤 $\hat\theta \in \Omega_0$가 $\dot l(\hat\theta) = 0$을 만족한다.
3. 모든 $\theta \in \Omega_0$와 모든 $c \in \mathbb{R}^k \setminus \{0\}$에 대하여$c^\top \ddot l(\theta) c < 0$이 성립한다.

그러면
$$
l(\hat\theta) = \max_{\theta \in \Omega_0} l(\theta)
$$
> **설명:**  
> 조건 (3)은 로그가능도함수의 헤시안 행렬(Hessian matrix) $\ddot l(\theta)$가 모든 $\theta$에서 **음의 정부호(negative definite)** 임을 의미한다.  
> 즉, $l(\theta)$는 $\Omega_0$에서 **순오목(strictly concave)** 함수가 된다.

### 정리 6.3.3 다차원 최대가능도 추정값의 존재와 유일성 *(Existence and Uniqueness of Multivariate MLE)*
정리 6.3.2의 1번, 3번가정에 더하여 다음 조건이 성립한다고 하자.  
$$
\lim_{\theta \to \partial(\Omega_0)} l(\theta) = -\infty
$$
>- $\partial(\Omega_0)$는 열린 구간 다면체 $\Omega_0$의 경계(boundary)를 나타내는 기호다. $\theta \to \partial(\Omega_0)$는 $\theta$의 각 성분이 유한 구간의 끝점으로 가까워지거나, 무한히 커지거나 작아지는 상황을 의미한다.
  
그러면 가능도방정식
$$
\dot l(\theta) = 0
$$
의 해는 $\Omega_0$에서 **존재하며 유일**하다.  
따라서 이 해는 유일한 최대가능도 추정값이다.

> **설명:**  
> - 조건 (4)에 의해 로그가능도함수는 경계(boundary)로 갈수록 작아진다.  
> - 연속성과 순오목성에 의해 최대값은 내부에서 하나만 존재한다.

### 정리 6.3.4 다중모수 지수족에서의 최대가능도 추정 *(MLE for Multi-parameter Exponential Family)*
확률밀도함수가 다음과 같은 형태를 갖는 분포족을 고려하자.
$$
pdf(x; \eta) =
\exp\left\{
\sum_{j=1}^k \eta_j T_j(x) - A(\eta) + S(x)
\right\},
\qquad x \in \mathcal{X},\quad \eta = (\eta_1, \dots, \eta_k)^\top \in N
$$

다음 조건들이 성립한다고 하자. (이 조건을 만족하는 형태의 pdf들의 집합을 다중모수 지수족, multi-parameter exponential family of pdf's라 한다)

1. 지지집합 $\mathcal{X}$는 $\eta$에 의존하지 않는다.
2. 모수공간 $N \subset \mathbb{R}^k$는 열린 집합이다.
3. 충분통계량 벡터 *(sufficient statistic vector)*
    $$
    T(X_1) = (T_1(X_1), \dots, T_k(X_1))^\top
    $$
    에 대하여 공분산행렬 *(covariance matrix)*
    $$
    \mathrm{Var}_\eta(T(X_1))
    $$
    가 양의 정부호(positive definite)이다.
    - 모두가 0은 아닌 어떤 실수 $c_1, \dots, c_k$에 대해서도 $c^\top T(x) = c_1 T_1(x) + \cdots + c_k T_k(x)$는 상수가 아니다.

이때 로그가능도함수는 순오목함수이며, 최대가능도 추정값 $\hat\eta$는 $\eta$의 최대가능도 추정값이고 다음 방정식의 해로 주어진다.
$$
\nabla A(\hat\eta) = \frac{1}{n} \sum_{i=1}^n T(x_i)
$$
즉,
$$
\frac{\partial A(\hat\eta)}{\partial\eta_j} =E_\eta[T(X_1)]= \frac{1}{n} \sum_{i=1}^n T_j(x_i), \qquad j = 1, \dots, k
$$

> **설명:**  
> 누율생성함수(cumulant generating function)의 성질로부터 $\nabla^2 A(\eta) = \mathrm{Var}_\eta(T(X_1))$임을 알 수 있다.  
> 가정 (iii)에 의해 $\nabla^2 A(\eta)$는 양의 정부호이므로, 로그가능도함수는 순오목함수.  
> 따라서 정리 6.3.3을 적용할 수 있으며, 최대가능도 추정값은 위의 방정식의 유일한 해가 된다.

**모수의 일대일 변환과 최대가능도 추정**  
모수 $\eta = g(\theta)$가 $\theta$의 **일대일 변환(one-to-one transformation)** 이면, $\theta$의 최대가능도 추정량 $\hat\theta^{\mathrm{MLE}}$을 이용해
$$
\hat\eta^{\mathrm{MLE}} = g(\hat\theta^{\mathrm{MLE}})
$$
로 $\eta$의 최대가능도 추정량을 얻을 수 있다.  
즉, 모수의 함수도 최대가능도 추정량을 통해 직접 추정할 수 있다.
$$
E_\theta[T(X_1)]= \frac{1}{n} \sum_{i=1}^n T_j(x_i),\ \theta \in \Omega
$$
이 가능도방정식의 근 $\hat\theta$ 로 주어진다.

#### 예시 6.3.4 정규분포의 지수족 표현 *(Exponential Family Representation of Normal Distribution)*
정규분포 *(normal distribution)* $N(\mu, \sigma^2)$의 경우
$$
pdf(x; \mu, \sigma^2) = (2\pi\sigma^2)^{-1/2} \exp\left\{ -\frac{1}{2\sigma^2}(x-\mu)^2 \right\}
$$
이를 지수족 형태로 다시 쓰면,
$$
pdf(x; \mu, \sigma^2) = \exp\left\{
\frac{\mu}{\sigma^2} x
+ \left(-\frac{1}{2\sigma^2}\right) x^2
- \left(
\frac{\mu^2}{2\sigma^2}
+ \frac{1}{2}\log\sigma^2
+ \frac{1}{2}\log 2\pi
\right)
\right\}
$$
따라서 정규분포 $N(\mu, \sigma^2)$의 확률밀도함수를 지수족 형태로 표현하려면, 모수 $\mu, \sigma^2$를 다음과 같이 변환한다:
$$
\eta_1 = \frac{\mu}{\sigma^2}, \qquad \eta_2 = -\frac{1}{2\sigma^2}
$$
이렇게 하면 확률밀도함수는
$$
pdf(x; \eta) = \exp\left\{ \eta_1 x + \eta_2 x^2 - A(\eta) \right\}
$$
의 꼴로 쓸 수 있다. 여기서 $A(\eta)$는 정규화 상수로, $\eta_1, \eta_2$에 대해
$$
A(\eta) = -\frac{\eta_1^2}{4\eta_2} + \frac{1}{2}\log(-2\eta_2) + \frac{1}{2}\log 2\pi
$$
가능도방정식은
$$
E_\theta(X_1) = \frac{1}{n}\sum_{i=1}^n x_i, \qquad
E_\theta(X_1^2) = \frac{1}{n}\sum_{i=1}^n x_i^2
$$
정규분포에서 $E_\theta(X_1) = \mu$, $E_\theta(X_1^2) = \mu^2 + \sigma^2$이므로, 표본평균 $\bar x$와 표본제곱평균 $\overline{x^2}$를 각각 대응시켜
$$
\hat\mu = \bar x = \frac{1}{n}\sum_{i=1}^n x_i
$$
$$
\hat\sigma^2 = \overline{x^2} - (\bar x)^2 = \frac{1}{n}\sum_{i=1}^n x_i^2 - \left(\frac{1}{n}\sum_{i=1}^n x_i\right)^2
$$
따라서 정규분포의 최대가능도 추정량은
$$
\hat\mu^{\mathrm{MLE}} = \bar X, \qquad
\hat\sigma^{2,\mathrm{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar X)^2
$$

#### 예시 6.3.5 이변량 정규분포에서의 최대가능도 추정 *(MLE for Bivariate Normal Distribution)*
이변량 정규분포 $(X, Y) \sim N_2(\mu_1, \mu_2, \sigma_1^2, \sigma_2^2, \rho)$의 확률밀도함수는 다음과 같다.
$$
f(x, y; \mu_1, \mu_2, \sigma_1^2, \sigma_2^2, \rho) =
\frac{1}{2\pi \sigma_1 \sigma_2 \sqrt{1-\rho^2}}
\exp\left\{
-\frac{1}{2(1-\rho^2)}
\left[
\frac{(x-\mu_1)^2}{\sigma_1^2}
- 2\rho \frac{(x-\mu_1)(y-\mu_2)}{\sigma_1 \sigma_2}
+ \frac{(y-\mu_2)^2}{\sigma_2^2}
\right]
\right\}
$$
이를 지수족 형태로 변환하면, 다음과 같이 모수변환을 할 수 있다.
- $\eta_1 = \frac{\mu_1}{\sigma_1^2 (1-\rho^2)}$
- $\eta_2 = \frac{\mu_2}{\sigma_2^2 (1-\rho^2)}$
- $\eta_3 = -\frac{1}{2\sigma_1^2 (1-\rho^2)}$
- $\eta_4 = -\frac{1}{2\sigma_2^2 (1-\rho^2)}$
- $\eta_5 = \frac{\rho}{\sigma_1 \sigma_2 (1-\rho^2)}$

따라서 확률밀도함수는
$$
f(x, y; \eta) = \exp\left\{
\eta_1 x + \eta_2 y + \eta_3 x^2 + \eta_4 y^2 + \eta_5 xy - A(\eta)
\right\}
$$
꼴로 쓸 수 있다.

정리 6.3.2(다차원 최대값 판정 정리)의 조건을 확인하면,
- 정의역은 열린 집합이고,
- 로그가능도함수는 $\eta$에 대해 두 번 연속 미분 가능하며,
- 충분통계량 벡터 $(X, Y, X^2, Y^2, XY)$의 공분산행렬이 양의 정부호이므로,
- 로그가능도함수의 헤시안이 음의 정부호(순오목)임을 알 수 있다.

따라서 정리 6.3.2를 적용할 수 있으며, 최대가능도 추정값은 가능도방정식의 유일한 해로 결정된다.
모수 $\theta = (\mu_1, \mu_2, \sigma_1^2, \sigma_2^2, \rho)^\top$에 대해, 크기 $n$인 랜덤표본 $(X_1, Y_1), \dots, (X_n, Y_n)$에 기초한 **가능도방정식**은 다음과 같다.
가능도방정식은 다음과 같이 각 충분통계량의 표본평균과 모평균을 일치시키는 형태로 주어진다.
$$
\begin{aligned}
E_\theta(X_1) &= \frac{1}{n}\sum_{i=1}^n X_i \\
E_\theta(Y_1) &= \frac{1}{n}\sum_{i=1}^n Y_i \\
E_\theta(X_1^2) &= \frac{1}{n}\sum_{i=1}^n X_i^2 \\
E_\theta(Y_1^2) &= \frac{1}{n}\sum_{i=1}^n Y_i^2 \\
E_\theta(X_1 Y_1) &= \frac{1}{n}\sum_{i=1}^n X_i Y_i
\end{aligned}
$$
각각의 가능도방정식을 풀면, 최대가능도 추정값은 다음과 같이 주어진다.

- 평균의 최대가능도 추정값:
    $$
    \hat\mu_1 = \bar X = \frac{1}{n}\sum_{i=1}^n X_i, \qquad
    \hat\mu_2 = \bar Y = \frac{1}{n}\sum_{i=1}^n Y_i
    $$
- 분산의 최대가능도 추정값:
    $$
    \hat\sigma_1^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar X)^2, \qquad
    \hat\sigma_2^2 = \frac{1}{n}\sum_{i=1}^n (Y_i - \bar Y)^2
    $$
- 상관계수의 최대가능도 추정값:
    $$
    \hat\rho^{\mathrm{MLE}} =
    \frac{
        \sum_{i=1}^n (X_i - \bar X)(Y_i - \bar Y)
    }{
        \sqrt{\sum_{i=1}^n (X_i - \bar X)^2}
        \sqrt{\sum_{i=1}^n (Y_i - \bar Y)^2}
    }
    $$

이는 표본상관계수(sample correlation coefficient)와 일치한다.

#### 예시 6.3.6 다항분포에서의 최대가능도 추정 *(MLE for Multinomial Distribution)*
$(X_1, \dots, X_k) \sim Multin(n; p_1, \dots, p_k), \ \sum_{j=1}^k p_j = 1$
$$
P(X_1 = x_1, \dots, X_k = x_k) =
\frac{n!}{x_1! \cdots x_k!} \prod_{j=1}^k p_j^{x_j} \\
l(p_1, \dots, p_k) =
\sum_{j=1}^k x_j \log p_j
+ \log n! - \sum_{j=1}^k \log x_j!
$$

**풀이 1**  
제약조건 $\sum_{j=1}^k p_j = 1$ 하에서 **라그랑주 승수법**을 이용하여 최대화한다.  
라그랑주 함수는
$$
\mathcal{L}(p_1, \dots, p_k, \lambda) =
\sum_{j=1}^k x_j \log p_j + \lambda \left(1 - \sum_{j=1}^k p_j\right)
$$

각 $p_j$에 대해 편미분하여 0으로 두면,
$$
\frac{\partial \mathcal{L}}{\partial p_j} = \frac{x_j}{p_j} - \lambda = 0
\implies
p_j = \frac{x_j}{\lambda}
$$

제약조건 $\sum_{j=1}^k p_j = 1$을 대입하면,
$$
\sum_{j=1}^k p_j = \sum_{j=1}^k \frac{x_j}{\lambda} = \frac{n}{\lambda} = 1
\implies
\lambda = n
$$

따라서
$$
\hat p_j^{\mathrm{MLE}} = \frac{x_j}{n}
$$
즉, 각 범주의 상대도수(비율)이다.

**풀이 2: 정리 6.3.4의 일대일 모수변환을 이용한 방법**  
다항분포의 확률질량함수는  
$$
P(X_1 = x_1, \dots, X_k = x_k) =
\frac{n!}{x_1! \cdots x_k!} \prod_{j=1}^k p_j^{x_j}
$$
로, $p_j$는 $0 < p_j < 1$, $\sum_{j=1}^k p_j = 1$이다.

이를 지수족 형태로 쓰면,  
$$
P(x; \eta) = \exp\left\{ \sum_{j=1}^k x_j \log p_j + \log n! - \sum_{j=1}^k \log x_j! \right\}
$$
여기서 충분통계량은 $T_j(x) = x_j$, 자연모수는 $\eta_j = \log p_j$이다. 단, $p_j = e^{\eta_j}$이므로 $p_j$와 $\eta_j$는 일대일 대응한다.

정리 6.3.4에 따라, 최대가능도 추정값 $\hat\eta_j$는  
$$
E_{\eta}[T_j(X)] = \frac{1}{n} \sum_{i=1}^n T_j(x^{(i)}) = \frac{x_j}{n}
$$
여기서 $E_{\eta}[T_j(X)] = E_p[X_j] = n p_j / n = p_j$이므로,  
$$
\hat p_j^{\mathrm{MLE}} = \frac{x_j}{n}
$$
즉, 일대일 함수 변환을 통해서도 각 범주의 상대도수가 최대가능도 추정량임을 알 수 있다.


## 6.4 최대가능도 추정량의 점근적 성질 *(Asymptotic Properties of Maximum Likelihood Estimators)*

### 6.4.1 도입: 점근적 성질의 필요성
표본크기 $n$이 커질수록 추정량이 참값에 가까워지는 **일치성(consistency)** 은 추정량의 기본 요건이다.  
하지만 통계적 추론(신뢰구간, 가설검정 등)을 위해서는 **극한분포(asymptotic distribution)** 까지 알아야 한다.  
이 절에서는 최대가능도 추정량(MLE)의 일치성과 극한분포를 일반적인 조건과 함께 정리한다. (로그가능도함수의 여러 성질로 최대가능도 추정량의 극한분포를 유도할 수 있음)

#### 예 6.4.1 최대가능도 추정량의 일치성과 극한분포
**(a) 베르누이 분포 $Bernoulli(p)$, $0 \le p \le 1$**  
MLE: $\hat p_n^{\mathrm{MLE}} = \bar X = \frac{1}{n}\sum_{i=1}^n X_i$ (예 6.2.2, 6.2.5 참고)
- 일치성:  
    $$
    \hat p_n^{\mathrm{MLE}} \xrightarrow{P_p} p \\
    \Leftrightarrow \\
    \lim_{n\to\infty} P_p\left(|\hat p_n^{\mathrm{MLE}} - p| \ge \epsilon\right) = 0 \quad \forall\, \epsilon > 0
    $$
- 극한분포:  
    $$
    \sqrt{n}(\hat p_n^{\mathrm{MLE}} - p) \xrightarrow{d} N(0, p(1-p))
    $$

**(b) 포아송 분포 $Poisson(\lambda)$, $\lambda \ge 0$**  
MLE: $\hat\lambda_n^{\mathrm{MLE}} = \bar X = \frac{1}{n}\sum_{i=1}^n X_i$ (예 6.2.1, 6.2.5)
- 일치성:  
    $$
    \hat\lambda_n^{\mathrm{MLE}} \xrightarrow{P_\lambda} \lambda
    $$
- 극한분포:  
    $$
    \sqrt{n}(\hat\lambda_n^{\mathrm{MLE}} - \lambda) \xrightarrow{d} N(0, \lambda)
    $$

**(c) 지수분포 $Exp(\theta)$, $\theta > 0$**  
MLE: $\hat\theta_n^{\mathrm{MLE}} = \bar X = \frac{1}{n}\sum_{i=1}^n X_i$ (예 6.2.4, 6.2.5)
- 일치성:  
    $$
    \hat\theta_n^{\mathrm{MLE}} \xrightarrow{P_\theta} \theta
    $$
- 극한분포:  
    $$
    \sqrt{n}(\hat\theta_n^{\mathrm{MLE}} - \theta) \xrightarrow{d} N(0, \theta^2)
    $$

**(d) 정규분포 $N(\mu, \sigma^2)$**  
MLE: $
\hat\mu_n^{\mathrm{MLE}} = \bar X, \
\hat\sigma_n^{2,\mathrm{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar X)^2$ (예 6.3.4)

- 일치성:  
    $$
    \hat\mu_n^{\mathrm{MLE}} \xrightarrow{P_\theta} \mu, \qquad
    \hat\sigma_n^{2,\mathrm{MLE}} \xrightarrow{P_\theta} \sigma^2
    $$
- 극한분포:  
    $$
    \sqrt{n}(\bar X - \mu) \sim N(0, \sigma^2) \text{  (표본크기 관계없음, 정리6.1.1)}
    $$
    $$
    \frac{n\hat\sigma_n^{2,\mathrm{MLE}}}{\sigma^2} \sim \chi^2(n-1)
    \text{  (표본크기 관계없음, 정리4.2.2)}
    $$
    $$
    \sqrt{n}(\hat\sigma_n^{2,\mathrm{MLE}} - \sigma^2) \xrightarrow{d} N(0, 2\sigma^4)
    $$

**(e) 이변량 정규분포 $N(\mu_1, \mu_2; \sigma_1^2, \sigma_2^2, \rho)$**  
MLE (상관계수, 예 6.3.5):  
$$
\hat\rho_n^{\mathrm{MLE}} =
\frac{\sum_{i=1}^n (X_i - \bar X)(Y_i - \bar Y)}
{\sqrt{\sum_{i=1}^n (X_i - \bar X)^2} \sqrt{\sum_{i=1}^n (Y_i - \bar Y)^2}}
$$ (예 6.2.1, 6.2.5)

- 극한분포:  
    $$
    \sqrt{n}(\hat\rho_n^{\mathrm{MLE}} - \rho) \xrightarrow{d} N(0, (1-\rho^2)^2)
    $$

위 경우들은 최대가능도 추정량이 일치정을 갖고 표본분포의 극한분포가 정규분포인 예였다. 이는 최대가능도 추정량이 적률이용추정량과 일치하는 예로, 그 성질 또한 적률이용추정량의 성질에서 밝힌 것이다.  
아래는 일치하지 않는 경우도 다루며, 일반적인 경우에 최대가능도 추정량이 어떤 성질을 갖는지 소개한다.
### 최대가능도 추정량의 일치성: 직관적 배경
로그가능도함수 $\ell_n(\theta) = \sum_{i=1}^n \log f(X_i; \theta), \
\bar\ell_n(\theta) = \frac{1}{n}\ell_n(\theta)$  

$\theta^0$를 모수의 참값이라 하면, 대수의 법칙(큰수의 법칙)에 의해  
$$
\bar\ell_n(\theta) =\frac{1}{n}\sum_{i=1}^n \log f(X_i; \theta)\xrightarrow{P_{\theta^0}} E_{\theta^0}[\log f(X_1; \theta)]
$$
따라서, 아래와 같은 추측을 할 수 있다 (최대가능도 추정량이 유일하게 정해질 수 있다는 전제하에 argmax사용가능)  
$$
\hat\theta_n = \arg\max_{\theta \in \Omega} \bar\ell_n(\theta)
\quad\Rightarrow\quad
\theta^0 = \arg\max_{\theta \in \Omega} E_{\theta^0}[\log f(X_1; \theta)]
$$
- 이 추측은 "표본 로그가능도 평균 $\bar\ell_n(\theta)$를 최대화하는 $\hat\theta_n$이, 표본크기 $n$이 커질수록 참값 $\theta^0$에 가까워진다"는 직관을 표현한 것이다.
- 이 직관을 엄밀화하기 위해 **쿨백–라이블러 괴리도(Kullback–Leibler divergence)** 를 도입한다.
- 결론은 이 추측은 참으로, 확률수렴하는 일치성을 보여준다.
### 정리 6.4.1 쿨백–라이블러 괴리도
두 분포가 서로 다른 정도를 나타내는 측도.  
확률밀도함수 $f(x; \theta)$, $\theta \in \Omega$가  
- (R0) **식별가능성**: $f(\cdot; \theta) = f(\cdot; \theta^0) \Rightarrow \theta = \theta^0$  
- (R1) **공통의 토대**: $\{x: f(x; \theta) > 0\}$이 $\theta$에 의존하지 않음

을 만족하면  
$$
KL(\theta, \theta^0) = -E_{\theta^0}\left[\log\frac{f(X; \theta)}{f(X; \theta^0)}\right]
$$
라고 하면 다음이 성립한다:
- $KL(\theta, \theta^0) \ge 0$  
- $KL(\theta, \theta^0) = 0 \iff \theta = \theta^0$

#### 증명
(i) **$KL(\theta, \theta^0) \ge 0$의 증명**  
- (R1) 공통 토대 조건에 의해 $f(x; \theta^0) > 0$이면 $f(x; \theta) > 0$이므로, $T(x) := \frac{f(x; \theta)}{f(x; \theta^0)}$는 $X$에서 양수이고 $P_{\theta^0}$-거의 모든 곳에서 정의된다.
- 모든 $t > 0$에 대해 $-\log t \ge 1 - t$ (볼록함수의 성질, $t=1$에서 등호).
- $t = T(X)$를 대입하면 $-\log T(X) \ge 1 - T(X)$.
- 양변에 $E_{\theta^0}$를 취하면
    $$
    KL(\theta, \theta^0) = -E_{\theta^0}\left[\log T(X)\right] \ge 1 - E_{\theta^0}[T(X)]
    $$
- $E_{\theta^0}[T(X)] = \int_X \frac{f(x; \theta)}{f(x; \theta^0)} f(x; \theta^0) dx = \int_X f(x; \theta) dx = 1$.
- 따라서 $KL(\theta, \theta^0) \ge 0$.

(ii) **$KL(\theta, \theta^0) = 0 \iff \theta = \theta^0$의 증명**  
- 위 부등식에서 등호는 $T(X) = 1$ $P_{\theta^0}$-a.s.일 때만 성립.
- 즉, $f(X; \theta) = f(X; \theta^0)$ $P_{\theta^0}$-a.s.
- (R1) 공통 토대에 의해 $P_{\theta^0}$-거의 모든 곳은 $X$ 전체에서 "거의 모든 $x$"와 동치.
- 따라서 $f(\cdot; \theta) = f(\cdot; \theta^0)$ (거의 모든 $x \in X$).
- (R0) 식별가능성에 의해 $\theta = \theta^0$.
- 역방향($\theta = \theta^0 \implies KL(\theta, \theta^0) = 0$)은 정의에서 즉시 성립.

**정리:**  
쿨백–라이블러 괴리도 $KL(\theta, \theta^0)$는 항상 0 이상이며, 0이 되는 유일한 경우는 $\theta = \theta^0$일 때이다.

### 최대가능도 추정량의 일치성: 일반 조건
1. **균등 확률수렴**  
     $$
     \sup_{|\theta - \theta^0| \le K} \left| \bar\ell_n(\theta) - E_{\theta^0}[\log f(X_1; \theta)] \right| \xrightarrow{P_{\theta^0}} 0, \ \forall K > 0 
     $$
2. **기대 로그가능도의 연속성**  
     $E_{\theta^0}[\log f(X_1; \theta)]$은 $\theta$에 대해 연속
3. **로그가능도의 단봉성 (순오목 단봉 로그가능도)**  
     $$
     c^\top \nabla^2 \ell(\theta) c < 0, \quad \forall c \neq 0, \ \forall \theta \in \Omega, \lim_{\theta \to \partial(\Omega)} l_n(\theta) = - \infin
     $$
### 정리 6.4.2 최대가능도 추정량(MLE)의 일치성(consistency)
모수공간 $\Omega \subset \mathbb{R}^k$에 대해, 다음을 가정한다.

1. **(R0) 식별 가능성(identifiability)**
    $$
    f(\cdot;\theta)=f(\cdot;\theta^0)\ \Rightarrow\ \theta=\theta^0
    $$
2. **(R1) 공통 토대(common support)**
    $$
    \{x:f(x;\theta)>0\}=\mathcal{X}\quad(\theta\in\Omega\ \text{에 의존하지 않음})
    $$
3. **(순오목 단봉 로그가능도: strict concavity + boundary condition)**
    로그가능도함수 $l_n(\theta)$가 $\theta$에 대해 두 번 편미분 가능하고, 이차 편도함수들이 연속함수이며
    $$
    c^\top \ddot l_n(\theta)c<0,\quad \forall c\neq 0,\ \forall \theta\in\Omega
    $$
    또한 경계로 갈 때
    $$
    \lim_{\theta\to \partial(\Omega)} l_n(\theta)=-\infty
    $$
4. **(기대 로그가능도의 연속성)**
    $$
    \forall \theta^0\in\Omega,\quad E_{\theta^0}[\log f(X_1;\theta)]\ \text{가 존재하고 }\theta\ \text{에 관한 연속함수이다}
    $$

위 조건이 만족되면, 크기 $n$ 랜덤표본에 대한 최대가능도 추정량 $\hat\theta_n^{\mathrm{MLE}}$는

* **가능도방정식(likelihood equation)의 유일한 근(unique root)** 이며
    $$
    i_n(\hat\theta_n^{\mathrm{MLE}})=0
    $$
    를 만족한다. (여기서 $i_n(\theta)$는 점수함수(score function), 즉 로그가능도의 기울기 벡터이다.)

* **일치성(consistency)** 을 가진다.
    $$
    \hat\theta_n^{\mathrm{MLE}}\xrightarrow{p_{\theta}} \theta,\qquad \forall \theta\in\Omega
    $$

#### 예 6.4.2 로지스틱분포 모형 $L(\theta,1)$에서 MLE의 일치성
$-\infty<\theta<\infty$로지스틱분포 $L(\theta,1)$의 확률밀도함수는
$$
f(x;\theta)=\frac{e^{x-\theta}}{(1+e^{x-\theta})^2}
=\frac{e^{-x+\theta}}{(1+e^{-x+\theta})^2},\qquad -\infty<x<\infty
$$
표본 $x_1,\dots,x_n$에 대해
$$
l_n(\theta)
= n\bar x-n\theta-2\sum_{i=1}^n \log(1+e^{x_i-\theta})
= -n\bar x+n\theta-2\sum_{i=1}^n \log(1+e^{-x_i+\theta})\\
\dot l_n(\theta) = n-2\sum_{i=1}^n \frac{e^{-x_i+\theta}}{1+e^{-x_i+\theta}}=0
$$
이 방정식의 근으로 주어지는 $\hat\theta_n^{\mathrm{MLE}}$는 정리 6.4.2의 조건을 만족하므로 **일치성을 가진다**

#### 예 6.4.3 감마분포 모형 $Gamma(\alpha,\beta)$에서 MLE의 일치성
$\alpha>0,\ \beta>0,\quad \theta=(\alpha,\beta)^\top$ 확률밀도함수
$$
f(x;\theta)=\frac{1}{\Gamma(\alpha)\beta^\alpha}x^{\alpha-1}e^{-x/\beta},\ \mathbf{1}_{(x>0)} \\
l_n(\theta)
=-n\log\Gamma(\alpha)-n\alpha\log\beta
+(\alpha-1)\sum_{i=1}^n \log x_i
-\sum_{i=1}^n \frac{x_i}{\beta}
$$

표기 단순화를 위해
$$
\overline{\log x}=\frac{1}{n}\sum_{i=1}^n \log x_i,\qquad
\bar x=\frac{1}{n}\sum_{i=1}^n x_i,\qquad
\psi(\alpha)=\frac{\partial}{\partial \alpha}\log\Gamma(\alpha)
$$
($\psi$는 digamma function이라 부르는 함수이다.)
$$
\begin{cases}
\frac{\partial l_n(\theta)}{\partial \alpha}
= -n\psi(\alpha)-n\log\beta+n\overline{\log x}=0 \\[6pt]
\frac{\partial l_n(\theta)}{\partial \beta}
= -\frac{n\alpha}{\beta}+\frac{n\bar x}{\beta^2}=0
\end{cases}
$$
이 방정식의 근으로 주어지는 $\big(\hat\alpha_n^{\mathrm{MLE}},\hat\beta_n^{\mathrm{MLE}}\big)$는 정리 6.4.2에 의해 **일치성을 가진다**  
(참고로 두 번째 식은 $\beta=\bar x/\alpha$ 형태로 정리되는 것이 보통이다.)

### 도입설명: 최대가능도 추정량(MLE)의 극한분포를 향한 직관적 전개(점근 정규성의 "아이디어")
이 절부터는 "왜 $\sqrt{n}(\hat\theta_n-\theta)$가 정규분포로 가는가"를 전개한다.  
평균 로그가능도: $\bar l_n(\theta)=\frac{1}{n}l_n(\theta)=\frac{1}{n}\sum_{i=1}^n \log f(X_i;\theta)$

**1차 테일러 전개 기반의 핵심 근사**  
최대가능도 추정량을 $\hat\theta_n$이라 하면(일차원 설명),
$$
0=\bar l_n'(\hat\theta_n)\approx \bar l_n'(\theta)+\bar l_n''(\theta)(\hat\theta_n-\theta)
$$
따라서
$$
\sqrt{n}(\hat\theta_n-\theta)\approx \big(-\bar l_n''(\theta)\big)^{-1}\sqrt{n}\,\bar l_n'(\theta)
$$

**정보량(information) 정의로의 연결**  
큰수의 법칙을 쓰면(적절한 조건하에서)
$$
-\bar l_n''(\theta)
=\frac{1}{n}\sum_{i=1}^n\Big[-\frac{\partial^2}{\partial\theta^2}\log f(X_i;\theta)\Big]
\xrightarrow{p_\theta}
E_\theta\Big[-\frac{\partial^2}{\partial\theta^2}\log f(X_1;\theta)\Big]
$$
따라서
$$
I(\theta)=E_\theta\Big[-\frac{\partial^2}{\partial\theta^2}\log f(X_1;\theta)\Big]
$$
로 둔다.

**점수(score)의 중심극한정리**  
$\frac{\partial}{\partial\theta}\log f(X_i;\theta)$는 독립 동일분포이므로 중심극한정리에 의해
$$
\sqrt{n}\Big(
    n^{-1}\sum_{i=1}^n \frac{\partial}{\partial\theta}\log f(X_i;\theta) - E_\theta\left[\frac{\partial}{\partial\theta}\log f(X_1;\theta)\right] 
\Big)
\ \xrightarrow{d}\
N\Big(0,\ \mathrm{Var}_\theta\Big(\frac{\partial}{\partial\theta}\log f(X_1;\theta)\Big)\Big) \\
\therefore 
\sqrt{n}\Big(
    \bar l_n'(\theta) - E_\theta\left[\frac{\partial}{\partial\theta}\log f(X_1;\theta)\right]
\Big)
\ \xrightarrow{d}\
N\Big(0,\ \mathrm{Var}_\theta\Big(\frac{\partial}{\partial\theta}\log f(X_1;\theta)\Big)\Big)
$$

또한 아래에서 보이듯(정리 6.4.3) 적절한 조건하에
$$
E_\theta\Big[\frac{\partial}{\partial\theta}\log f(X_1;\theta)\Big] 
= \int \left(\frac{\partial}{\partial\theta}\log f(x;\theta) \right) f(x;\theta)\,dx \\
= \int \left(\frac{1}{f(x;\theta)} \frac{\partial}{\partial\theta} f(x;\theta) \right) f(x;\theta)\,dx \\
= \int \frac{\partial}{\partial\theta} f(x;\theta)\,dx
= \frac{\partial}{\partial\theta} \int f(x;\theta)\,dx 
= \frac{\partial}{\partial\theta} 1 
= 0
$$
이므로
$$
\sqrt{n}\,\bar l_n'(\theta)\xrightarrow{d}
N\Big(0,\ \mathrm{Var}_\theta\Big(\frac{\partial}{\partial\theta}\log f(X_1;\theta)\Big)\Big)
$$

**결론**:
$$
\sqrt{n}(\hat\theta_n-\theta)
\approx I(\theta)^{-1}\cdot \sqrt{n}\,\bar l_n'(\theta)
\ \xrightarrow{d}\
N\Big(0,\ I(\theta)^{-2}\,\mathrm{Var}_\theta\Big(\frac{\partial}{\partial\theta}\log f(X_1;\theta)\Big)\Big)
$$
가 된다. 그리고 $\mathrm{Var}(\text{score})=I(\theta)$가 성립하면(정리 6.4.3) 더 단순해진다.
- 따라서 적절한 조건하에서 최대가능도 추정량의 극한분포가 정규분포가 된다.
- 이런 성질을 점근정규성(asymptotic normality) 라 한다
### 최대가능도 추정량의 점근 정규성(asymptotic normality)을 위한 기본 조건 (R0)–(R5)
아래 조건들은 이후 정리 6.4.3, 6.4.4에서 사용되는 "정규성 전개가 성립하도록 하는" 정칙조건(regularity conditions)이다.
* **(R0) 식별 가능성(identifiability)**
    $$
    f(\cdot;\theta)=f(\cdot;\theta^0)\Rightarrow \theta=\theta^0
    $$
* **(R1) 공통 토대(common support)**
    $$
    \{x:f(x;\theta)>0\}=\mathcal{X}\quad(\theta\ \text{에 무관})
    $$
* **(R2) 열린 모수공간(open parameter space)**
    $$
    \Omega\ \text{는 } \mathbb{R}^k\ \text{의 열린집합이다}
    $$
* **(R3) 미분가능한 로그가능도(differentiability)**
    모든 관측결과 $(x_1,\dots,x_n)\in\mathcal{X}^n$에 대해
    $$
    l_n(\theta)=\sum_{i=1}^n \log f(x_i;\theta)
    $$
    의 1차/2차 편도함수 $l_n'(\theta),\ \ddot l_n(\theta)$가 존재하고 모두 연속함수이다.
    (다차원에서 $i_n(\theta)$는 score vector, $\ddot l_n(\theta)$는 Hessian matrix이다.)
* **(R4) 적분(또는 합)과 미분의 교환(interchangeability)**
    랜덤표본 $X=(X_1,\dots,X_n)^\top$의 함수 $u(X)$에 대해 $E_\theta[u(X)]$가 존재하면, 모수에 관한 미분을 적분/합 안으로 넣을 수 있다고 가정한다.

    * 연속형인 경우:
        $$
        \partial_\theta^r E_\theta[u(X)]
        =\int_{\mathbb{R}^n} u(x)\,\partial_\theta^r(pdf(x;\theta))\,dx
        $$
    * 이산형인 경우:
        $$
        \partial_\theta^r E_\theta[u(X)]
        =\sum_{x\in\mathbb{R}^n} u(x)\,\partial_\theta^r(pdf(x;\theta))
        $$
        여기서
        $$
        pdf(x;\theta)=\prod_{i=1}^n f(x_i;\theta)
        $$
* **(R5) 정보량(information)의 존재 및 가역성(invertibility)**
    모든 $\theta\in\Omega$에 대해
    $$
    I(\theta)=\mathrm{Var}_\theta\Big(\frac{\partial}{\partial\theta}\log f(X_1;\theta)\Big)
    $$
    가 (스칼라 또는 행렬로) 잘 정의되며, 그 역수(또는 역행렬)가 존재한다.

$l_n'(\theta)$를 점수함수(score function)아라 부른다. 이는 최대가능도 추정량의 성질을 아래 정리와 같이 잘 보여주는 함수다.
### 정리 6.4.3 점수함수(score function)의 성질
조건 (R0)–(R5)가 만족되면 다음이 성립한다.

**(a) 점수의 기대값은 0이다**  
$$
E_\theta\Big[\frac{\partial}{\partial\theta}\log f(X_1;\theta)\Big]=0
$$

**(b) 정보량의 두 표현(바틀렛 항등식 형태, Bartlett identity)**  
$$
I(\theta)
=\mathrm{Var}_\theta\Big(\frac{\partial}{\partial\theta}\log f(X_1;\theta)\Big)
=E_\theta\Big[-\frac{\partial^2}{\partial\theta^2}\log f(X_1;\theta)\Big]
$$

#### 증명
이산형, 다차원의 경우도 같은 방법으로 증명가능하므로 일차원 연속형만 증명한다.  
* $ \int_{\mathcal{X}} f(x;\theta)\,dx = 1 $ 양변을 $\theta$로 미분하면
    $$
    \int_{\mathcal{X}} \frac{\partial}{\partial\theta}f(x;\theta)\,dx=0
    $$
* $\frac{\partial}{\partial\theta}\log f=\frac{f_\theta}{f}$를 이용하면 (R2)–(R4)로
    $$
    E_\theta\Big[\frac{\partial}{\partial\theta}\log f(X_1;\theta)\Big]=0
    $$
* 한 번 더 미분하여 정리하면
    $$
    E_\theta\Big[\Big(\frac{\partial}{\partial\theta}\log f(X_1;\theta)\Big)^2\Big]
    =-E_\theta\Big[\frac{\partial^2}{\partial\theta^2}\log f(X_1;\theta)\Big]
    $$
    이고 (a)로 평균이 0이므로 분산과 연결되어 (b)가 나온다.

>조건 (R5)와 정리6.4.3의 $I(\theta)$를 일차원 모수의 경우에는 정보량(information number), 다차원 모수의 경우에는 정보량 행렬이라 하며, >추정량의 효율성을 연구하는 데 매우 중요한 역할을 한다.
### 정리 6.4.4 최대가능도 추정량의 점근 정규성 (Asymptotic Normality of MLE)
전제(기본 조건: (R0)~(R5))
- (R6) **일치성(consistency)** + **가능도방정식의 유일근**
    $$
    i_n(\theta)=\sum_{i=1}^n \frac{\partial}{\partial\theta}\log f(X_i;\theta)=0
    $$
    의 해 $\hat\theta_n^{\mathrm{MLE}}$가 **단 하나**이고, 또한 일치성을 가진다. 즉
    $$
    \hat\theta_n^{\mathrm{MLE}}\xrightarrow{P_\theta}\theta
    $$
- (R7) **3차 도함수의 지배(dominating bound for third derivative)**
    (스칼라 설명 기준) 어떤 확률변수 $M(X_1)$가 존재하여
    $$
    \max_{\theta\in\Omega}\left| \partial_\theta^3 \log f(X_1;\theta)\right|\le M(X_1),
    \qquad E_\theta[M(X_1)]<\infty
    $$

위 조건들이 만족되면
$$
\sqrt{n}\,(\hat\theta_n^{\mathrm{MLE}}-\theta)\ \xrightarrow{d}\ N\left(0\,,\,[I(\theta)]^{-1}\right)
$$
#### 증명
다차원 모수의 경우도 같은 방법으로 밝힐 수 있으므로 일차원의 경우만 증명한다.  
표현을 간단히 하기 위해 "표본당(per-observation)" 표기를 둔다.
- $\bar l'_n(\theta)=l'_n(\theta)/n$ (점수의 평균)
- $\ddot{\bar{l}_n}(\theta) = \frac{1}{n} \ddot{l}_n(\theta)$ (2차 도함수의 평균)
- $\bar l_n^{(3)}(\theta)=\partial_\theta^3 l_n(\theta)/n$

**(1단계) 가능도방정식의 1차 근사 + 잉여항(remainder)**  
가능도방정식 $\bar{\dot{l}_n}(\hat\theta_n)=0$에 대해 테일러 전개를 적용하면
$$
0 = \bar{\dot{l}_n}(\hat\theta_n)
= \bar{\dot{l}_n}(\theta)
+ \bar l_n''(\theta)(\hat\theta_n - \theta)
+ \frac{1}{2}\bar l_n^{(3)}(\theta_n^*)(\hat\theta_n - \theta)^2,
\quad (|\theta_n^* - \theta| \leq |\hat\theta_n - \theta|)
$$

**(2단계) 잉여항의 확률수렴**  
$$
R_n = \frac{1}{2}\,\bar l_n^{(3)}(\theta_n^*)(\hat\theta_n - \theta)
$$
라 하면, 임의의 양수 $\epsilon > 0$과 $K > 0$에 대하여
$$
P_\theta(|R_n| \geq \epsilon)
= P_\theta\left(\left|\frac{1}{2}\bar l_n^{(3)}(\theta_n^*)\right| \gt K, |R_n| \geq \epsilon\right) + P_\theta\left(\left|\frac{1}{2}\bar l_n^{(3)}(\theta_n^*)\right| \leq K, |R_n| \geq \epsilon\right) \\
\leq
P_\theta\left(|\bar l_n^{(3)}(\theta_n^*)| \geq K\right)
+ P_\theta\left(|\hat\theta_n - \theta| \geq \frac{2\epsilon}{K}\right)
$$

조건 (R7)과 마르코프 부등식에 의해
$$
P_\theta\left(|\bar l_n^{(3)}(\theta_n^*)| \gt K\right)
\leq
P_\theta\left(\frac{1}{n}\sum_{i=1}^n M(X_i) \geq K\right)
\leq
E_\theta\left[\frac{1}{n}\sum_{i=1}^n M(X_i)\right]/K
 = \frac{E_\theta[M(X_1)]}{K}
$$

또한, 조건 (R6)의 일치성에 의해
$$
P_\theta\left(|\hat\theta_n - \theta| \geq \frac{2\epsilon}{K}\right) \to 0 \quad (n \to \infty)
$$
따라서 $K \to \infty$로 보내면
$$
0 \leq \limsup_{n \to \infty} P_\theta(|R_n| \geq \epsilon) \leq \frac{E_\theta[M(X_1)]}{K} \to 0
$$
즉,
$$
R_n \xrightarrow{P_\theta} 0
$$

**(3단계) $\hat\theta_n-\theta$를 점수로 표현**  
(1), (2) 결과를 합치면
$$
0=\bar{\dot{l}_n}(\theta)+\big(\bar{\ddot l}_n(\theta)+R_n\big)(\hat\theta_n-\theta), \ R_n \xrightarrow{P_\theta}0
$$
조건(R5), 정리6.4.3과 큰수의 법칙을 활용하면,
$$

-\bar{\ddot l}_n(\theta)=\frac1n\sum_{i=1}^n\Big[-\partial_\theta^2\log f(X_i;\theta)\Big]
\xrightarrow{P_\theta}E_\theta\Big[-\partial_\theta^2\log f(X_1;\theta)\Big]=I(\theta)
$$
$$
\therefore \sqrt{n}(\hat\theta_n-\theta)=\big(I(\theta)+r_n\big)^{-1}\sqrt{n}\,\bar{\dot{l}_n}(\theta),
\qquad r_n\xrightarrow{P_\theta}0
$$

**(4단계) 중심극한정리(CLT) + 슬럿스키(Slutsky)**  
점수(score) 함수의 평균 $\bar{\dot{l}_n}(\theta)$는 독립 동일분포(i.i.d.) 표본 $X_1, \dots, X_n$에 대해  
$$
\bar{\dot{l}_n}(\theta) = \frac{1}{n}\sum_{i=1}^n \frac{\partial}{\partial\theta}\log f(X_i;\theta)
$$
각 항 $\frac{\partial}{\partial\theta}\log f(X_i;\theta)$는 평균이 0(정리 6.4.3(a) 참고), 분산 $I(\theta)$를 갖는 i.i.d. 확률변수  
따라서 중심극한정리(CLT)에 의해  
$$
\sqrt{n}\left(\bar{\dot{l}_n}(\theta) - E_\theta\left[\frac{\partial}{\partial\theta}\log f(X_1;\theta)\right]\right)
= \sqrt{n}\left(\frac{1}{n}\sum_{i=1}^n \frac{\partial}{\partial\theta}\log f(X_i;\theta) - 0\right)
\xrightarrow{d} N(0, I(\theta))
$$
즉, 점수의 표본평균은 $\sqrt{n}$으로 정규화하면 평균 0, 분산 $I(\theta)$인 정규분포로 수렴.  

따라서 3단계의 점수화와 정리5.3.1의 극한분포 계산법으로부터
$$
\sqrt{n}(\hat\theta_n-\theta) = (I(\theta)+r_n)^{-1}\sqrt n \bar{\dot{l}_n}(\theta), \quad r_n\xrightarrow{P_\theta}0\\
\sqrt{n}(\hat\theta_n-\theta) \xrightarrow{d} [I(\theta)]^{-1}Z, \quad Z\sim N(0, I(\theta))\\
\therefore \sqrt{n}(\hat\theta_n-\theta) \xrightarrow{d}N\left(0,[I(\theta)]^{-1}\right)
$$

#### 해석
- 근사식
    $$
    \sqrt{n}(\hat\theta_n^{\mathrm{MLE}}-\theta)
    = [I(\theta)]^{-1}\sqrt{n}\,\bar{\dot{l}_n}(\theta)+o_n, \quad o_n \xrightarrow{P_\theta} 0
    $$
    이 성립하는것을 알 수 있다.
- 즉 **추정오차는 "점수(Score)의 크기"에 의해 결정**되며, **정밀도는 정보량 $I(\theta)$** 가 클수록 좋아진다(분산이 $[I(\theta)]^{-1}$로 감소).
- 그래서 $\sqrt{n}\bar{\dot{l}_n}(\theta)$를 최대가능도 추정량의 추정점수를 나타내는 점수함수로 부르고, $I(\theta)$를 최대가능도 추정량의 추정 정밀도를 나타내는 정보량이라 부른다.

### 정리 6.4.5 일단계(one-step) 반복법에 의한 MLE 근사
예 6.4.2나 예 6.4.3처럼 최대가능도 추정량의 구체적인 형태를 모르고 가능도방정식의 근으로 주어지는 경우에는 추정량을 근사해야 한다. 근사하는 방법을 소개한다.  

정리 6.4.4의 조건 (R0)~(R7)을 만족한다고 하고, 추가로 초기값 $\hat\theta_n^{(0)}$이 다음을 만족한다고 둔다.

- (R8) **초기 추정량의 $\sqrt{n}$-수준 안정성**
    $$
    \sqrt{n}\,(\hat\theta_n^{(0)}-\theta)\xrightarrow{d}Z
    $$
    인 어떤 확률변수 $Z$가 존재한다 (즉, $\hat\theta_n^{(0)}$가 이미 $\sqrt{n}$ 스케일에서 "그럭저럭" 맞는다).

**일단계 갱신(one-step Newton update)**  
가능도방정식의 1차 근사를 이용해 다음을 정의한다.
$$
\hat\theta_n^{(1)}
=\hat\theta_n^{(0)} +\left[-\bar {\ddot l}_n\big(\hat\theta_n^{(0)}\big)\right]^{-1}\bar{\dot{l}_n}\big(\hat\theta_n^{(0)}\big)
$$
(다차원은 역행렬로 동일)

결론: $\hat\theta_n^{(1)}$는 최대가능도 추정량과 같은 극한분포를 갖는다.
$$
\sqrt{n}\,(\hat\theta_n^{(1)}-\theta)\xrightarrow{d}N\left(0,[I(\theta)]^{-1}\right)
$$
즉, **정확한 MLE를 끝까지 풀지 않아도** "좋은 초기값 + 한 번의 뉴턴 갱신"이면 **점근적으로 MLE와 동등한 효율**을 얻는다.

#### 예 6.4.4 로지스틱 분포 $L(\theta,1)$ (location 모수)
$$
f(x;\theta)=\frac{e^{x-\theta}}{(1+e^{x-\theta})^2}
=\frac{e^{-x+\theta}}{(1+e^{-x+\theta})^2},
\qquad -\infty<x<\infty \\
l_n(\theta)=n\bar x-n\theta-2\sum_{i=1}^n\log\big(1+e^{x_i-\theta}\big)
= -n\bar x+n\theta-2\sum_{i=1}^n\log\big(1+e^{-x_i+\theta}\big) \\
i_n(\theta)=n-2\sum_{i=1}^n\frac{e^{-x_i+\theta}}{1+e^{-x_i+\theta}}=0 \\
\ddot l_n(\theta)= -2\sum_{i=1}^n\frac{e^{-x_i+\theta}}{\big(1+e^{-x_i+\theta}\big)^2}
$$
따라서 MLE는 위 방정식의 **유일근**으로 주어진다.

초기값을 표본평균으로 두면 $\hat\theta_n^{(0)}=\bar X_n$  
일단계 갱신:
$$
\hat\theta_n^{(1)}
=\hat\theta_n^{(0)}
+\left[-\bar l_n\big(\hat\theta_n^{(0)}\big)\right]^{-1}\bar{\dot{l}_n}\big(\hat\theta_n^{(0)}\big)
$$

로지스틱 분포에서 $\mathrm{Var}(X)=\pi^2/3$이므로 점근분포:
$$
\sqrt{n}\,(\hat\theta_n^{(0)}-\theta)\xrightarrow{d}N\left(0,\frac{\pi^2}{3}\right)
$$
또한 변환 $X_1-\theta=\log\frac{U}{1-U}$, $U\sim U(0,1)$ 및
$$
\frac{1}{1+e^{-(X_1-\theta)}}=U
$$
를 이용해 정보량을 계산하면
$$
I(\theta)=E_\theta[-\ddot l_1(\theta)]
=2E[U(1-U)]=\frac13
$$
따라서 정리 6.4.5에 의해
$$
\sqrt{n}\,(\hat\theta_n^{(1)}-\theta)\xrightarrow{d}N(0,3)
$$
이 된다. 즉, 일단계 갱신만으로도 MLE 효율($[I(\theta)]^{-1}=3$)을 달성한다.

#### 예 6.4.5 감마 분포 $Gamma(\alpha,\beta)$ (shape-scale)
$$
f(x;\theta)=\frac{1}{\Gamma(\alpha)\beta^\alpha}x^{\alpha-1}e^{-x/\beta}\mathbf{1}_{(x>0)},
\qquad \theta=(\alpha,\beta)^\top,\ \alpha>0,\ \beta>0 \\
l_n(\theta)=
-n\log\Gamma(\alpha)-n\alpha\log\beta
+(\alpha-1)\sum_{i=1}^n\log x_i
-\sum_{i=1}^n\frac{x_i}{\beta}
$$
요약기호:
$$
\overline{\log x}=\frac1n\sum_{i=1}^n\log x_i,\qquad
\bar x=\frac1n\sum_{i=1}^n x_i,\qquad
\Psi(\alpha)=\frac{\partial}{\partial\alpha}\log\Gamma(\alpha)
$$
($\Psi$는 digamma 함수이다. $\Psi'(\alpha)$는 trigamma이다.)

점수벡터, 헤시안:
$$
i_n(\theta)=
\begin{pmatrix}
\partial_\alpha l_n(\theta)\\
\partial_\beta l_n(\theta)
\end{pmatrix}
=
\begin{pmatrix}
-n\Psi(\alpha)-n\log\beta+n\overline{\log x}\\
-n\alpha/\beta+n\bar x/\beta^2
\end{pmatrix} \\
\ddot l_n(\theta)=
\begin{pmatrix}
\partial_{\alpha\alpha}^2l_n & \partial_{\alpha\beta}^2l_n\\
\partial_{\beta\alpha}^2l_n & \partial_{\beta\beta}^2l_n
\end{pmatrix}
=
\begin{pmatrix}
-n\Psi'(\alpha) & -n/\beta\\
-n/\beta & n\alpha/\beta^2-2n\bar x/\beta^3
\end{pmatrix}
$$

감마에서 $E[X]=\alpha\beta$, $\mathrm{Var}(X)=\alpha\beta^2$이므로 적률추정량 $\hat\theta_n^{\mathrm{MME}}=(\hat\alpha_n,\hat\beta_n)^\top$는 연립방정식 해로 주어진다:
$$
\begin{cases}
\hat\alpha_n\hat\beta_n=\bar X_n\\
\hat\alpha_n(\hat\beta_n)^2=\frac1n\sum_{i=1}^n(X_i-\bar X_n)^2
\end{cases}
$$
책에서는 이를 $\hat\theta_n^{(0)}$로 둔다:
$$
\hat\theta_n^{(0)}=\hat\theta_n^{\mathrm{MME}}
$$

정보량 행렬과 역행렬:
$$
I(\theta)=E_\theta[-\ddot l_1(\theta)]
=
\begin{pmatrix}
\Psi'(\alpha) & 1/\beta\\
1/\beta & \alpha/\beta^2
\end{pmatrix} \\
[I(\theta)]^{-1}
=\frac{1}{\alpha\Psi'(\alpha)-1}
\begin{pmatrix}
\alpha & -\beta\\
-\beta & \Psi'(\alpha)\beta^2
\end{pmatrix}
$$

결론(일단계 추정량의 점근분포):  
일단계(one-step)로 만든 $\hat\theta_n^{(1)}=(\hat\alpha_n^{(1)},\hat\beta_n^{(1)})^\top$에 대해
$$
\sqrt{n}\big(\hat\theta_n^{(1)}-\theta\big)\xrightarrow{d}N\left(0,[I(\theta)]^{-1}\right)
$$
가 성립한다.


## 최소제곱 추정법 *(Least Squares Estimation)*
### 선형회귀모형과 최소제곱의 목적
관측치 $i=1,\dots,n$에 대해
$$
Y_i = x_{i0}\beta_0 + x_{i1}\beta_1 + \cdots + x_{ip}\beta_p + e_i
$$

오차항 가정은
- $E(e_i) = 0$
- $\operatorname{Var}(e_i) = \sigma^2$
- $\operatorname{Cov}(e_i, e_j) = 0$ for $i \neq j$
- $-\infty < \beta_j < \infty$ $(j=0,\dots,p)$, $0 < \sigma^2 < \infty$

여기서 핵심은 **평균반응 $E(Y_i \mid x_{i0},\dots,x_{ip})$** 를 설명변수의 선형함수로 두는 모형이라는 점이다.

**최소제곱(least squares)의 기준**  
모형이 제시하는 평균반응
$$
E(Y_i) = x_{i0}\beta_0 + \cdots + x_{ip}\beta_p
$$
가 실제 관측 $Y_i$에 "가깝도록" 만들기 위해, 오차제곱합(SSE)을 최소화한다.
$$
\sum_{i=1}^{n} \left\{ Y_i - (x_{i0}\beta_0 + \cdots + x_{ip}\beta_p) \right\}^2
$$
를 최소로 하는 $\beta$를 선택한다.

### 행렬(벡터) 표기와 최소제곱 추정량의 정의
**(1) 행렬 표기**  
다음을 정의한다.

- 반응변수 벡터  
    $$
    Y = (Y_1, \dots, Y_n)^\top
    $$
- 설계행렬(design matrix)  
    $$
    X =
    \begin{pmatrix}
    x_{10} & x_{11} & \cdots & x_{1p} \\
    \vdots & \vdots &        & \vdots \\
    x_{n0} & x_{n1} & \cdots & x_{np}
    \end{pmatrix}
    \quad (n \times (p+1))
    $$
- 회귀계수 벡터  
    $$
    \beta = (\beta_0, \dots, \beta_p)^\top
    $$
- 오차벡터  
    $$
    e = (e_1, \dots, e_n)^\top
    $$

그러면 모형은
$$
Y = X\beta + e
$$
이고,
$$
E(e) = 0, \quad \operatorname{Var}(e) = \sigma^2 I_n, \quad \operatorname{rank}(X) = p+1
$$
로 쓴다. $\operatorname{rank}(X) = p+1$은 $X^\top X$가 가역(invertible)임을 보장하는 핵심 조건이다.

**(2) 최소제곱 추정량(LSE) 정의**  
오차제곱합은
$$
|Y - X\beta|^2
$$
이므로 최소제곱 추정량은
$$
\hat\beta^{LSE} = \arg\min_{\beta \in \mathbb{R}^{p+1}} |Y - X\beta|^2
$$
로 정의된다.

### 정리 6.5.1: 투영(projection) 관점의 최소제곱
$$
\Pi = X(X^\top X)^{-1} X^\top
$$
를 정의한다. $\Pi$는 $Y$를 $X$의 열공간(column space)으로 직교투영하는 행렬이다.

**(a) 투영행렬의 성질**
- $\Pi^\top = \Pi$ (대칭)
- $\Pi^\top (I - \Pi) = 0$ (직교성: $\Pi$ 부분과 $(I-\Pi)$ 부분이 서로 직교)
- $\Pi X = X$ (열공간에 있는 벡터는 투영해도 그대로)

즉, $\Pi$는 "직교투영"을 정확히 구현하는 행렬이다.

**(b) SSE의 직교분해와 추정량 형태**  
임의의 $\beta$에 대해 잔차벡터 $Y - X\beta$는
$$
Y - X\beta = \Pi(Y - X\beta) + (I - \Pi)Y
$$
로 분해되고, (a)의 직교성 때문에 노름제곱이
$$
|Y - X\beta|^2 = |\Pi(Y - X\beta)|^2 + |(I - \Pi)Y|^2
$$
로 분해된다.

여기서 $|(I - \Pi)Y|^2$는 $\beta$와 무관한 상수항이므로, 최소화를 위해서는 $|\Pi(Y - X\beta)|^2$를 0으로 만들면 된다. 따라서 최적해는
$$
X\hat\beta^{LSE} = \Pi Y
$$
를 만족해야 하고, $\operatorname{rank}(X) = p+1$이면 해가 유일하여
$$
\boxed{
\hat\beta^{LSE} = (X^\top X)^{-1} X^\top Y
}
$$
가 된다. 또한 적합값(fitted value)은
$$
\widehat{Y} = X\hat\beta^{LSE} = \Pi Y
$$
이다.

### 오차분산 $\sigma^2$의 추정
책에서는 오차항 분산의 추정량으로 평균오차제곱(MSE)을 사용한다.
$$
\boxed{
\hat\sigma^2 = \frac{|Y - X\hat\beta^{LSE}|^2}{n - p - 1}
}
$$
분모 $n - p - 1$은 자유도(표본크기 $n$에서 추정한 모수 개수 $p+1$을 뺀 값)이다.
### 정리 6.5.2: 최소제곱 추정량의 성질(불편성, 분산, 정규성)
선형회귀모형 $Y = X\beta + e$, $E(e) = 0$, $\operatorname{Var}(e) = \sigma^2 I_n$, $\operatorname{rank}(X) = p+1$에서

**(a) $\hat\beta^{LSE}$의 기댓값과 분산**  
$$
E(\hat\beta^{LSE}) = \beta
$$
즉, 최소제곱 추정량은 **불편추정량**이다.

또한
$$
\operatorname{Var}(\hat\beta^{LSE}) = \sigma^2 (X^\top X)^{-1}
$$

**해석**  
- $X^\top X$가 "설명변수 정보량"을 모으는 행렬이고,
- $(X^\top X)^{-1}$가 커질수록(설명변수들이 서로 비슷하거나, 표본이 부족하거나) 분산이 커진다.

**(b) $\hat\sigma^2$의 불편성**  
$$
E(\hat\sigma^2) = \sigma^2
$$
즉, 위의 $\hat\sigma^2$도 **불편추정량**이다.

책의 증명 흐름은 다음 아이디어를 쓴다.

- 잔차는 $(I-\Pi)Y = (I-\Pi)e$
- $\hat\sigma^2 = \frac{e^\top (I-\Pi) e}{n-p-1}$
- $\operatorname{trace}$ 성질을 이용해 $E(e^\top A e) = \sigma^2 \operatorname{trace}(A)$ 형태로 계산
- $\operatorname{trace}(I-\Pi) = n-(p+1)$ 임을 이용해 결론 도출

**(c) 오차가 정규분포일 때의 정확한 분포(추론의 기반)**  
추가로 $e \sim N_n(0, \sigma^2 I_n)$를 가정하면

- $$
    \hat\beta^{LSE} \sim N\left(\beta,\ \sigma^2 (X^\top X)^{-1}\right)
    $$
- $$
    \frac{(n-p-1)\hat\sigma^2}{\sigma^2} \sim \chi^2(n-p-1)
    $$
- 그리고 $\hat\beta^{LSE}$와 $\hat\sigma^2$는 서로 독립이다.

**해석**  
이 (c)가 회귀분석의 $t$-검정, $F$-검정, 신뢰구간의 표준 결과로 연결되는 출발점이다.

### 설명변수 직교화(orthogonalization) 동기
- 설명변수 행렬 $X$의 열들이 서로 직교(orthogonal)하면 $X^\top X$가 대각행렬이 되어 계산과 해석이 쉽다.
- 하지만 일반적으로는 직교하지 않으므로, **열공간을 보존하면서(동일한 모형공간) 직교화된 표현으로 바꾸어** 추정과 해석을 편하게 할 수 있다.

이를 위해 $X$를 두 블록으로 나눈다.
$$
X = (X_0, X_1),\quad X_0: n \times p_0,\ X_1: n \times p_1,\ p_0 + p_1 = p+1
$$

**(1) 부분투영행렬 정의**  
- $X_0$에 대한 투영:
        $$
        \Pi_0 = X_0 (X_0^\top X_0)^{-1} X_0^\top
        $$
- $X_1$에서 $X_0$ 성분을 제거한(잔차화한) 행렬:
        $$
        X_{1|0} = (I - \Pi_0) X_1
        $$
- $X_{1|0}$에 대한 투영:
        $$
        \Pi_{1|0} = X_{1|0} (X_{1|0}^\top X_{1|0})^{-1} X_{1|0}^\top
        $$

이때 $X_0$의 열공간과 $X_{1|0}$의 열공간은 서로 직교가 된다(구성상 $(I-\Pi_0)$로 $X_0$에 직교인 성분만 남겼기 때문).

또한 전체 투영이
$$
\Pi = \Pi_0 + \Pi_{1|0}
$$
처럼 "서로 직교인 두 투영의 합"으로 분해된다는 점이 핵심이다.

**(2) 평균반응의 재표현**  
$$
X\beta = X_0\beta_0 + X_1\beta_1
$$
를
$$
X\beta = X_0\gamma_0 + X_{1|0}\beta_1
$$
꼴로도 쓸 수 있음을 보인다(책에서는 $\gamma_0 = \beta_0 + (X_0^\top X_0)^{-1} X_0^\top X_1 \beta_1$ 형태를 제시한다). 요지는 "$X_1$의 $X_0$방향 성분을 $X_0$로 흡수"하고, 나머지 직교 성분만 $X_{1|0}$가 담당하도록 바꾸는 것이다.

### 정리 6.5.3: 직교화된 표현에서의 최소제곱 결과
**(a) 적합값의 분해와 각 부분의 추정**  
최소제곱 적합값은
$$
X\hat\beta^{LSE} = X_0\hat\gamma_0^{LSE} + X_{1|0}\hat\beta_1^{LSE}
$$
로 분해된다.

그리고 각 항은 단순한 투영 형태로 주어진다.
$$
X_0\hat\gamma_0^{LSE} = \Pi_0 Y
$$
$$
X_{1|0}\hat\beta_1^{LSE} = \Pi_{1|0} Y
$$

또한 $\hat\beta_0^{LSE}$는 $\hat\gamma_0^{LSE}$와 $\hat\beta_1^{LSE}$를 이용해
$$
\hat\beta_0^{LSE} = \hat\gamma_0^{LSE} - (X_0^\top X_0)^{-1} X_0^\top X_1 \hat\beta_1^{LSE}
$$
처럼 복원된다("흡수했던 성분을 되돌려 원래 계수로 환산"하는 단계이다).

**(b) 공분산 0 및 평균/분산**  
직교화의 가장 큰 이점은, 두 블록의 추정이 분리되고 상관이 사라진다는 점이다.

$$
\operatorname{Cov}(\hat\gamma_0^{LSE}, \hat\beta_1^{LSE}) = 0
$$

또한
$$
E(\hat\gamma_0^{LSE}) = \gamma_0,\quad \operatorname{Var}(\hat\gamma_0^{LSE}) = \sigma^2 (X_0^\top X_0)^{-1}
$$
$$
E(\hat\beta_1^{LSE}) = \beta_1,\quad \operatorname{Var}(\hat\beta_1^{LSE}) = \sigma^2 (X_{1|0}^\top X_{1|0})^{-1}
$$

즉 $X_1$을 "$X_0$의 영향 제거 후"에 회귀하는 형태로 분산이 정리된다.

#### 예 6.5.1: 절편이 포함된 선형회귀모형(중심화로 직교화)
평균반응을
$$
E(Y_i\mid x_{i1},\dots,x_{ip}) = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip}
$$
로 둔다. (즉 $x_{i0} = 1$.)

각 설명변수의 표본평균을
$$
\bar x_j = \frac{1}{n}\sum_{i=1}^n x_{ij}
$$
라 하면, 평균반응을 다음처럼 다시 쓸 수 있다.
$$
\beta_0 + \sum_{j=1}^p \beta_j x_{ij}
= \gamma_0 + \sum_{j=1}^p \beta_j (x_{ij} - \bar x_j)
$$
여기서 $\gamma_0 = \beta_0 + \sum_{j=1}^p \beta_j \bar x_j$ 이다.

즉,
- $X_0 = \mathbf{1}$ (전부 1인 열벡터)
- $X_{1|0}$는 각 설명변수에서 평균을 뺀 "중심화(centered) 변수"들로 구성된다.

이렇게 하면 중심화된 설명변수 열들은 $\mathbf{1}$과 직교가 된다(각 열의 합이 0이므로).

이 경우 정리 6.5.3(a)에서
$$
\hat\gamma_0^{LSE} = \bar Y
$$
가 된다(절편 성분은 $Y$의 평균으로 추정된다는 결론).

또한 $X_{1|0}^\top X_{1|0}$와 $X_{1|0}^\top Y$는 "편차곱합"으로 정리된다. 책은
- $S_{jk} = \sum_{i=1}^n (x_{ij} - \bar x_j)(x_{ik} - \bar x_k)$
- $S_{jY} = \sum_{i=1}^n (x_{ij} - \bar x_j)(Y_i - \bar Y)$

로 두면
$$
X_{1|0}^\top X_{1|0} = (S_{jk}),\quad X_{1|0}^\top Y = (S_{jY})
$$
가 되어, 기울기 추정량 $\hat\beta_1, \dots, \hat\beta_p$는
$$
\begin{pmatrix}
S_{11} & \cdots & S_{1p} \\
\vdots & \ddots & \vdots \\
S_{p1} & \cdots & S_{pp}
\end{pmatrix}
\begin{pmatrix}
\hat\beta_1 \\ \vdots \\ \hat\beta_p
\end{pmatrix}
=
\begin{pmatrix}
S_{1Y} \\ \vdots \\ S_{pY}
\end{pmatrix}
$$
을 푸는 것과 같아진다.