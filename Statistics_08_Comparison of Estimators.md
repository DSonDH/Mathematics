# 제8장 추정량의 비교 *(Comparison of Estimators)*

## 추정량 비교의 기준 *(Criteria for Comparing Estimators)*

### 평균제곱오차 *(Mean Squared Error, MSE)*
모집단 분포가 확률밀도함수 $f(x;\theta),\ \theta \in \Omega$로 주어지고, 랜덤표본 $(X_1,\dots,X_n)$을 이용하여 모수 $\theta$ 또는 모수의 함수 $\eta=\eta(\theta)$를 추정한다고 하자.

좋은 추정량이란 참값에 가까운 추정값을 제공하는 추정량이다. 이를 정량적으로 평가하는 가장 보편적인 기준이 **평균제곱오차(MSE)** 이다.

$$
\mathrm{MSE}(\hat\eta,\theta)
= E_\theta\left[(\hat\eta(X_1,\dots,X_n)-\eta(\theta))^2\right]
$$

- 편향(bias)과 분산(variance)을 동시에 반영하는 손실 기준이다.
- 추정량 비교의 기본 도구로 사용된다.

#### 예 8.1.1 비교가 가능한 경우
균등분포 $X_i \sim U(0,\theta),\ \theta>0,\ n\ge 2$에서 $\theta$의 추정량으로 다음 두 가지를 고려한다.

- 최대가능도추정량(MLE): $\hat\theta^{\mathrm{MLE}} = X_{(n)}$ (표본최댓값)
- 적률추정량(MME): $\hat\theta^{\mathrm{MME}} = 2\bar X$

각각의 평균제곱오차는

$$
\mathrm{MSE}(\hat\theta^{\mathrm{MLE}},\theta)
= E_\theta[(X_{(n)}-\theta)^2]
= \frac{2}{(n+1)(n+2)}\theta^2
$$

$$
\mathrm{MSE}(\hat\theta^{\mathrm{MME}},\theta)
= E_\theta[(2\bar X-\theta)^2]
= \frac{1}{3n}\theta^2
$$

따라서 모든 $\theta>0$에 대해
$$
\mathrm{MSE}(\hat\theta^{\mathrm{MLE}},\theta)
\le
\mathrm{MSE}(\hat\theta^{\mathrm{MME}},\theta)
$$

또한 수렴 속도 측면에서도
$$
\mathrm{MSE}(\hat\theta^{\mathrm{MLE}},\theta)\sim n^{-2},\qquad
\mathrm{MSE}(\hat\theta^{\mathrm{MME}},\theta)\sim n^{-1}
$$

즉, 이 경우에는 **MLE가 MME보다 명백히 우수**하다.

#### 예 8.1.2 비교가 어려운 경우
정규분포 $X_i \sim N(\theta,1),\ -\infty<\theta<\infty$에서 다음 두 추정량을 고려한다.

- 상수 추정량: $\hat\theta^{(0)}=0$
- 최대가능도추정량: $\hat\theta^{\mathrm{MLE}}=\bar X$

각각의 평균제곱오차는
$$
\mathrm{MSE}(\hat\theta^{(0)},\theta)=\theta^2,\qquad
\mathrm{MSE}(\hat\theta^{\mathrm{MLE}},\theta)=\frac{1}{n}
$$

- $|\theta|<1/\sqrt{n}$이면 $\hat\theta^{(0)}$가 더 작다.
- $|\theta|$가 크면 $\hat\theta^{\mathrm{MLE}}$가 더 작다.

즉, **MSE 기준에서는 어느 한 추정량이 전 구간에서 우월하다고 말할 수 없다.**

### 혼합추정량 *(Hybrid Estimator)*
위 문제를 완화하기 위해 검정 결과를 활용한 혼합추정량을 고려한다.

$$
\hat\theta^{H}=
\begin{cases}
0, & |\bar X|<1.96/\sqrt{n} \\
\bar X, & |\bar X|\ge 1.96/\sqrt{n}
\end{cases}
$$

이때 평균제곱오차는 표준정규확률변수 $Z\sim N(0,1)$를 이용하여

$$
E_\theta(\hat\theta^{H}-\theta)^2
= n^{-1}\left[
(\sqrt{n}\theta)^2P(|Z+\sqrt{n}\theta|<1.96)
+E\{Z^2 I(|Z+\sqrt{n}\theta|\ge1.96)\}
\right]
$$

- 평균제곱오차는 $|\sqrt{n}\theta|$의 함수이다.
- 수렴 속도는 $\mathrm{MSE}(\hat\theta^{H},\theta)\sim n^{-1}$로 $\hat\theta^{\mathrm{MLE}}$와 동일하다.
- $\theta=0$ 근방에서는 표본평균보다 더 작은 MSE를 가질 수 있다.

### 최대평균제곱오차 *(Maximum Mean Squared Error)*
모수값에 따라 우열이 바뀌는 문제를 피하기 위해, **모수 전 범위에서의 최악 성능**을 기준으로 비교한다.

$$
\max_{\theta\in\Omega}
E_\theta[(\hat\eta-\eta(\theta))^2]
$$

이를 최소화하는 추정량을 **최소최대 평균제곱오차 추정량**이라 한다.

### 베이지안 평균제곱오차 *(Bayesian Mean Squared Error)*
사전밀도함수(prior) $\pi(\theta)$가 주어질 때, 평균제곱오차의 가중평균을 기준으로 한다.

$$
r(\pi,\hat\eta)
=\int_\Omega E_\theta[(\hat\eta-\eta(\theta))^2]\pi(\theta)\,d\theta
$$

이를 최소로 하는 추정량을 **베이지안 평균제곱오차 추정량**이라 한다.

### 평균제곱상대오차 *(Mean Squared Relative Error)*
$\eta(\theta)>0$인 경우, 다음 기준을 사용하기도 한다.

$$
E_\theta\left[\left(\frac{\hat\eta}{\eta}-1\right)^2\right]
$$

이에 대응하여 최대평균제곱상대오차, 베이지안 평균제곱상대오차 개념이 정의된다.

#### 예 8.1.3 최대평균제곱상대오차 최소화
정규분포 $X_i\sim N(\mu,\sigma^2)$에서 분산의 추정량을
$$
\hat\sigma_c^2
=c\sum_{i=1}^n(X_i-\bar X)^2,\quad c>0
$$
로 둘 때,

$$
\sum_{i=1}^n\frac{(X_i-\bar X)^2}{\sigma^2}
\sim \chi^2(n-1)
$$

따라서 평균제곱상대오차는
$$
E_\theta\left[\left(\frac{\hat\sigma_c^2}{\sigma^2}-1\right)^2\right]
=2(n-1)c^2+((n-1)c-1)^2
$$

이를 최소화하면 $c=\frac{1}{n+1}$

즉,
$$
\hat\sigma^2=\frac{1}{n+1}\sum_{i=1}^n(X_i-\bar X)^2
$$

### 불편추정량 *(Unbiased Estimator)*
$E_\theta[\hat\eta]=\eta(\theta),\quad \forall\theta\in\Omega$

불편추정량의 경우
$\mathrm{MSE}(\hat\eta,\theta)=\mathrm{Var}_\theta(\hat\eta)$
이므로, 분산을 최소화하는 것이 곧 MSE를 최소화하는 것이다.

### 전역최소분산불편추정량 *(Uniformly Minimum Variance Unbiased Estimator, UMVUE)*
추정량 $\hat\eta^*$가 다음을 만족하면 UMVUE라 한다.
1. $E_\theta[\hat\eta^*]=\eta(\theta)$
2. 임의의 불편추정량 $\hat\eta^{UE}$에 대해
    $\mathrm{Var}_\theta(\hat\eta^*) \le \mathrm{Var}_\theta(\hat\eta^{UE}),\quad \forall\theta\in\Omega$

### 정리 8.1.1 최소분산불편추정량의 유일성 *(Uniqueness of UMVUE)*
UMVUE의 분산이 유한하면, UMVUE는 거의 확실하게 유일하다.

#### 증명
$\hat\eta_1, \hat\eta_2$가 모두 UMVUE라고 하자.
$\hat\eta_3 = \frac{\hat\eta_1+\hat\eta_2}{2}$ 역시 불편추정량이다.

UMVUE의 성질로부터
$\mathrm{Var}_\theta(\hat\eta_3) \ge \mathrm{Var}_\theta(\hat\eta_1) = \mathrm{Var}_\theta(\hat\eta_2)$

이를 전개하면
$\mathrm{Var}_\theta(\hat\eta_1-\hat\eta_2)=0$

또한
$E_\theta(\hat\eta_1-\hat\eta_2)=0$

따라서
$P_\theta(\hat\eta_1=\hat\eta_2)=1,\quad \forall\theta\in\Omega$
이다. □


## 충분통계량 *(Sufficient Statistics)*

### 충분성의 동기 *(Motivation of Sufficiency)*
랜덤표본 $X_1,\dots,X_n$을 이용해 모수 $\theta$를 추론할 때, 표본 전체가 아니라 **일부 정보만으로도 동일한 추론 정확도**를 얻을 수 있다면 데이터 저장·계산 측면에서 유리하다.

이러한 **자료 축약(reduction)**의 핵심 개념이 **충분성(sufficiency)**이다.

#### 예 8.2.1 두 번의 베르누이 시행 *(Bernoulli Trials)*
서로 독립인 $X_1,X_2 \sim \mathrm{Bernoulli}(\theta),\ 0<\theta<1$에서 $(X_1,X_2)$ 대신 $Y=X_1+X_2$만 관측한다고 하자.

- $Y=0 \Rightarrow (0,0)$
- $Y=2 \Rightarrow (1,1)$
- $Y=1 \Rightarrow (1,0)$ 또는 $(0,1)$

조건부확률은
$P_\theta(X_1=1,X_2=0\mid Y=1) = P_\theta(X_1=0,X_2=1\mid Y=1) = \frac12$

이는 $\theta$에 의존하지 않는다.

**결론:** 성공 횟수 $Y$만 알면 순서 정보는 $\theta$ 추론에 불필요하다.

### 충분통계량의 정의 *(Definition of Sufficient Statistic)*
모집단 분포가 $f(x;\theta),\ \theta\in\Omega$이고, 통계량 $Y=u(X_1,\dots,X_n)$에 대하여

$$
P_{\theta_1}\big((X_1,\dots,X_n)\in A\mid Y=y\big)
=
P_{\theta_2}\big((X_1,\dots,X_n)\in A\mid Y=y\big)
$$

가 모든 집합 $A$, 모든 $y$, 모든 $\theta_1,\theta_2\in\Omega$에 대해 성립하면 $Y$를 $\theta$에 대한 **충분통계량**이라 한다.

즉, **조건부 분포가 모수에 의존하지 않음**.

#### 예 8.2.2 베르누이 독립시행에서의 충분통계량
$X_1,\dots,X_n \sim \mathrm{Bernoulli}(\theta)$

$Y=\sum_{i=1}^n X_i \sim \mathrm{Binomial}(n,\theta)$

조건부확률은
$$
P_\theta(X_1=x_1,\dots,X_n=x_n\mid Y=y)
=
\binom{n}{y}^{-1} I\Big(\sum x_i=y\Big)
$$

이는 $\theta$에 무관하다.

**결론:** $Y=\sum_{i=1}^n X_i$는 $\theta\in(0,1)$에 대한 충분통계량이다.

### 분해 정리 *(Factorization Theorem)*

### 정리 8.2.1 분해 정리
통계량 $Y=u(X_1,\dots,X_n)$이 $\theta$에 대한 충분통계량일 **필요충분조건**은

$$
\prod_{i=1}^n f(x_i;\theta) = k_1(u(x),\theta)\,k_2(x)
$$

를 만족하는 함수 $k_1, k_2$가 존재하는 것이다.

#### 증명 (충분조건)
결합확률을 분해하면
$$
P_\theta(X=x\mid Y=y)
= \frac{k_1(y,\theta)k_2(x)}{\sum_{z:u(z)=y}k_1(y,\theta)k_2(z)}
= \frac{k_2(x)}{\sum_{z:u(z)=y}k_2(z)}
$$

$\theta$에 의존하지 않는다.

#### 증명 (필요조건)
충분성으로부터
$P_\theta(X=x\mid Y=u(x))=k_2(x)$,
$P_\theta(Y=u(x))=k_1(u(x),\theta)$

이므로
$P_\theta(X=x)=k_1(u(x),\theta)k_2(x)$

### 대표적 분포에서의 충분통계량

#### 예 8.2.3 포아송 분포 *(Poisson)*
$X_i\sim\mathrm{Poisson}(\theta)$

$$
\prod f(x_i;\theta)
= e^{-n\theta}\theta^{\sum x_i}\prod \frac{1}{x_i!}
$$

따라서 $\sum X_i$는 충분통계량이다.

#### 예 8.2.4 지수분포 *(Exponential)*
$X_i\sim \mathrm{Exp}(\theta)$

$$
\prod f(x_i;\theta)
= \theta^{-n}\exp\Big(-\frac{1}{\theta}\sum x_i\Big)\prod I_{(0,\infty)}(x_i)
$$

따라서 $\sum X_i$는 충분통계량이다.

#### 예 8.2.5 감마분포 *(Gamma)*
$X_i \sim \mathrm{Gamma}(\alpha, \beta)$

결합밀도는
$\left(\sum x_i,\; \prod x_i\right)$
로 분해 가능하다.

따라서
$u(X) = \left(\sum X_i,\, \prod X_i\right)$
는 $(\alpha, \beta)$에 대한 충분통계량이다.

### 지수족에서의 충분통계량 *(Exponential Family)*

### 정리 8.2.2 다중모수 지수족
확률밀도함수가
$$
f(x;\theta) = \exp\left\{ \sum_{j=1}^k g_j(\theta) T_j(x) - A(g(\theta)) + S(x) \right\}
$$
꼴이면,
$$
\sum_{i=1}^n T(X_i)
$$
는 충분통계량이다.

### 충분통계량의 함수 *(Function of Sufficient Statistic)*

### 정리 8.2.3
$Y$가 충분통계량이고 $W = g(Y)$가 **일대일 함수**이면 $W$도 충분통계량이다.

#### 예 8.2.6 정규분포 *(Normal)*
$X_i \sim N(\mu, \sigma^2)$

충분통계량:
$\left(\sum X_i,\, \sum X_i^2\right)$

동치 표현:
$\left(\bar X,\, \sum (X_i - \bar X)^2\right)$

#### 예 8.2.7 베타분포 *(Beta$(\alpha, 1)$)*
$f(x;\alpha) = \alpha x^{\alpha-1}$

충분통계량:
$\sum \log X_i$
또는
$\overline{\log X}$

#### 예 8.2.8 두 모수 지수분포 *(Shifted Exponential)*
$f(x;\mu,\sigma) = \frac{1}{\sigma} e^{-(x-\mu)/\sigma} I_{[\mu,\infty)}(x)$

충분통계량:
$\left(\sum X_i,\, \min X_i\right)$

#### 예 8.2.9 균등분포 $U[\theta_1, \theta_2]$
$f(x;\theta_1,\theta_2) = (\theta_2-\theta_1)^{-n} I(\theta_1 \le \min X_i,\, \max X_i \le \theta_2)$

충분통계량:
$(\min X_i,\, \max X_i)$

### 최대가능도추정량과 충분통계량

### 정리 8.2.4
MLE $\hat\theta^{\mathrm{MLE}}$가 유일하면, 이는 **임의의 충분통계량의 함수**이다.

즉,
$$
\hat\theta^{\mathrm{MLE}} = h(S)
$$

### Rao–Blackwell 정리 *(Estimator Improvement)*

### 정리 8.2.5 Rao–Blackwell 정리
$\hat\eta$가 추정량이고 $Y$가 충분통계량이면
$$
\hat\eta^{RB} = E(\hat\eta \mid Y)
$$
는 항상
$$
\mathrm{MSE}(\hat\eta^{RB}, \theta) \le \mathrm{MSE}(\hat\eta, \theta)
$$

#### 예 8.2.10 균등분포 $U(0, \theta)$
초기 추정량:
$\hat\theta = 2\bar X,\quad \mathrm{MSE} = \frac{\theta^2}{3n}$

충분통계량:
$X_{(n)} = \max X_i$

Rao–Blackwell 개선:
$\hat\theta^{RB} = \frac{n+1}{n} X_{(n)}$

$\mathrm{MSE}(\hat\theta^{RB}, \theta) = \frac{\theta^2}{n(n+2)} < \frac{\theta^2}{3n}$


## 최소분산불편추정 *(Minimum Variance Unbiased Estimation)*
앞 절에서 다음 사실을 확인하였다.

- 충분통계량을 이용하면 추정량을 개선할 수 있다.
- Rao–Blackwell 정리를 통해 **평균제곱오차(MSE)** 를 감소시킬 수 있다.

이제 관심은 다음 질문으로 수렴한다.

> **불편추정량(unbiased estimator) 중에서, 모든 모수값에 대해 분산이 가장 작은 추정량은 무엇인가?**

이에 대한 해답이 **최소분산불편추정**이며, 그 궁극적 결과가 **전역최소분산불편추정량(UMVUE)** 이다.

### 완비통계량 *(Complete Statistic)*
모집단 분포가 $f(x;\theta),\ \theta\in\Omega$이고, 통계량 $Y = u(X_1, \dots, X_n)$에 대해
$$
E_\theta[g(Y)] = 0\ \forall\theta\in\Omega \implies P_\theta(g(Y)=0) = 1\ \forall\theta\in\Omega
$$
이면 $Y$를 $\theta$에 대한 **완비통계량**이라 한다.

- 평균이 0인 함수는 거의 확실하게 0이어야 한다.
- 즉, $Y$ 안에는 "중복되는 정보"가 존재하지 않는다.

### 완비충분통계량 *(Complete Sufficient Statistic)*
통계량 $Y$가

- 충분통계량(sufficient statistic)이고
- 완비통계량(complete statistic)이면

이를 **완비충분통계량**이라 한다.

이 개념은 UMVUE 존재·유일성의 핵심 전제이다.

### 전역최소분산불편추정량 *(Uniformly Minimum Variance Unbiased Estimator, UMVUE)*
모수의 함수 $\eta=\eta(\theta)$에 대한 불편추정량 $\hat\eta^*$가
$$
\mathrm{Var}_\theta(\hat\eta^*) \le \mathrm{Var}_\theta(\hat\eta)
\quad \forall \theta\in\Omega
$$
를 모든 불편추정량 $\hat\eta$에 대해 만족하면, $\hat\eta^*$를 **전역최소분산불편추정량(UMVUE)** 라 한다.

### 완비충분통계량과 UMVUE

### 정리 8.3.1 완비충분통계량을 이용한 UMVUE
모집단 분포 $f(x;\theta)$에서 랜덤표본 $(X_1,\dots,X_n)$에 대해 통계량
$$
Y = u(X_1,\dots,X_n)
$$
이 $\theta$에 대한 **완비충분통계량**이라고 하자.

- **(a) Rao–Blackwell 형태**  
    $\eta=\eta(\theta)$의 임의의 불편추정량 $\hat\eta^{UE}$에 대해
    $$
    \hat\eta^{RB}(Y) = E(\hat\eta^{UE} \mid Y)
    $$
    로 정의하면, $\hat\eta^{RB}(Y)$는 $\eta(\theta)$의 UMVUE이다.

- **(b) 함수형 UMVUE**  
    $Y$의 함수 $\delta(Y)$가 $\eta(\theta)$의 불편추정량이면, $\delta(Y)$는 $\eta(\theta)$의 UMVUE이다.

**증명의 핵심 논리**
- Rao–Blackwell 정리에 의해 분산은 감소한다.
- 완비성에 의해 같은 평균을 가지는 두 함수는 거의 확실하게 동일하다.
- 따라서 UMVUE는 유일하다.

#### 예 8.3.1 베르누이 독립시행
$X_i \sim \mathrm{Bernoulli}(\theta),\ 0<\theta<1$

- 충분통계량: $Y = \sum X_i$
- $Y \sim \mathrm{Binomial}(n, \theta)$  
    다항식 전개의 유일성으로 $Y$는 완비통계량이다.

$$
E_\theta(Y/n) = \theta
$$

따라서
$$
\hat\theta = \frac{1}{n}\sum_{i=1}^n X_i
$$
는 $\theta$의 UMVUE이다.

#### 예 8.3.2 포아송 분포
$X_i \sim \mathrm{Poisson}(\theta),\ \theta>0$

- 충분통계량: $Y = \sum X_i$
- $Y \sim \mathrm{Poisson}(n\theta)$  
    멱급수 전개의 유일성으로 $Y$는 완비통계량이다.

$$
E_\theta(Y/n) = \theta
$$

따라서
$$
\hat\theta = \frac{1}{n}\sum_{i=1}^n X_i
$$
는 UMVUE이다.

#### 예 8.3.3 지수분포
$X_i \sim \mathrm{Exp}(\theta),\ \theta>0$

- 충분통계량: $Y = \sum X_i$
- $Y \sim \mathrm{Gamma}(n, \theta)$  
    라플라스 변환의 유일성으로 $Y$는 완비통계량이다.

$$
E_\theta(Y/n) = \theta
$$

따라서 표본평균은 UMVUE이다.

### 다중모수 지수족에서의 완비성

### 정리 8.3.2 다중모수 지수족의 완비충분통계량
확률밀도함수가
$$
f(x;\eta) = \exp\left\{ \sum_{j=1}^k \eta_j T_j(x) - A(\eta) + S(x) \right\}
$$
꼴이고,  
1. 분포의 토대가 모수에 의존하지 않음  
2. 모수공간이 열린 직사각형을 포함  
3. $(T_1,\dots,T_k)$의 비자명한 선형결합이 상수가 아님  
이면
$$
\sum_{i=1}^n T(X_i)
$$
는 **완비충분통계량**이다.

#### 예 8.3.4 정규분포
$X_i \sim N(\mu, \sigma^2)$

- 완비충분통계량: $\left(\sum X_i,\; \sum X_i^2\right)$  
- 동치 표현: $(\bar X,\; S^2)$

따라서  
- $\bar X$는 $\mu$의 UMVUE  
- $S^2$는 $\sigma^2$의 UMVUE

이다.

#### 예 8.3.5 제한된 모수공간을 갖는 정규모집단의 경우
$X_1,\dots,X_n \sim N(\mu, \mu^2),\ \mu>0$

- 지수족 표현에서 $\eta_1 = 1/\mu,\ \eta_2 = -1/(2\mu^2)$,  
    모수공간 $\mathcal N = \{ (\eta_1, \eta_2): \eta_2 = -\tfrac12 \eta_1^2,\ \eta_1>0 \}$은 2차원 열린집합을 포함하지 않음.
- 정리 8.3.2를 적용할 수 없음.
- 실제로 $Y = (\sum X_i,\, \sum X_i^2)$에 대해
    $$
    \mathbb E_\mu\left[\frac{1}{2n}Y_2 - \frac{1}{n(n+1)}Y_1^2\right] = 0\quad(\forall\mu>0)
    $$
    인 비자명한 함수가 존재.

**결론:**  
$Y$는 $\mu$에 대한 완비통계량이 아니며, 표본평균 $\bar X$는 $\mu$의 UMVUE가 아니다.

#### 예 8.3.6 균등분포 $U[0,\theta]$에서의 완비충분통계량
$X_1,\dots,X_n \sim U[0,\theta],\ \theta>0$

- 충분통계량: $Y = \max_{1\le i\le n} X_i$
- $Y$의 밀도:
    $$
    f_Y(y;\theta) = \frac{n}{\theta^n} y^{n-1} \mathbf{1}_{[0,\theta]}(y)
    $$
- $E_\theta[g(Y)] = 0\ \forall\theta>0$이면 $g(Y)=0$이므로 $Y$는 완비충분통계량.

또한
$$
\mathbb E_\theta\left(\frac{n+1}{n}Y\right) = \theta
$$

따라서
$$
\hat\theta^{UMVUE} = \frac{n+1}{n} \max_{1\le i\le n} X_i
$$

#### 예 8.3.7 베르누이 분포에서 모분산의 불편추정
$X_1,\dots,X_n \sim \mathrm{Bernoulli}(\theta),\ 0<\theta<1$

- 완비충분통계량: $Y = \sum_{i=1}^n X_i$
- 모분산: $\eta = \theta(1-\theta)$
- 불편추정량: $\hat\eta^{UE} = X_1(1-X_2)$

Rao–Blackwell 개선:
$$
\hat\eta^{RB}(Y) = \mathbb E[X_1(1-X_2)\mid Y] = \frac{n}{n-1}\hat\theta(1-\hat\theta),\quad \hat\theta = \frac{Y}{n}
$$

따라서
$$
\hat\eta^{UMVUE} = \frac{n}{n-1}\hat\theta(1-\hat\theta)
$$

#### 예 8.3.8 포아송 분포에서 모평균의 함수의 불편추정
$X_1,\dots,X_n \sim \mathrm{Poisson}(\theta),\ \theta>0$

- 완비충분통계량: $Y = \sum_{i=1}^n X_i \sim \mathrm{Poisson}(n\theta)$
- 추정 대상: $\eta = e^{-2\theta}$

멱급수 전개의 유일성을 이용하면
$$
\hat\eta^{UMVUE} = (1-2/n)^Y
$$

- $n=1$에서는 $(-1)^{X_1}$ 형태의 비직관적 추정량이 됨

#### 예 8.3.9 정규분포에서 신뢰도의 불편추정
- 모형: $X_1,\dots,X_n \sim N(\theta,1),\quad n\ge2$
- 완비충분통계량: $\bar X$
- 추정 대상: $\eta = P_\theta(X_1 > a)$
- 불편추정량: $\mathbf{1}_{(a,\infty)}(X_1)$
- Rao–Blackwell 개선:
    $$
    \hat\eta^{UMVUE}
    = P(X_1 > a \mid \bar X)
    = 1 - \Phi\left(\sqrt{\frac{n}{n-1}}(a - \bar X)\right)
    $$

### 정리 8.3.3 완비충분통계량과 보조통계량의 독립성
- 통계량 $Z = v(X_1,\dots,X_n)$의 분포가 $\theta$에 의존하지 않으면 $Z$는 **보조통계량(ancillary statistic)** 이다.
- $Y$가 $\theta$에 대한 완비충분통계량이면 $Z \perp Y$ (독립).
- 핵심 결과: 완비충분통계량이 주어지면, 그와 독립인 보조통계량을 이용해 조건부기대값 계산이 단순해진다.

#### 예 8.3.10 지수분포에서 신뢰도의 불편추정
- 모형: $X_1,\dots,X_n \sim \mathrm{Exp}(\theta),\ \theta>0$
- 완비충분통계량: $Y = \sum_{i=1}^n X_i$
- 추정 대상: $\eta = P_\theta(X_1 > a) = e^{-a/\theta}$
- 보조통계량: $\frac{X_1}{\sum_{i=1}^n X_i} \sim \mathrm{Beta}(1, n-1)$
- 독립성 $\left(\frac{X_1}{\sum X_i}\right) \perp Y$를 이용하면
    $$
    \hat\eta^{UMVUE}
    = \left(1 - \frac{a}{n\bar X}\right)^{n-1}
    \mathbf{1}_{(a/n,\infty)}(\bar X)
    $$
- $n$이 크면 $\hat\eta^{UMVUE} \to e^{-a/\bar X}$로 MLE와 동일한 극한거동을 가진다.

## 8.4 추정량의 점근적 비교 *(Asymptotic Comparison of Estimators)*
- 표본크기 $n$이 커질 때 추정량 $\hat\eta_n$의 **극한분포(asymptotic distribution)** 로 성능을 비교한다.
- 보통 $\sqrt{n}(\hat\eta_n - \eta(\theta)) \xrightarrow{d} N(0,\,\sigma^2(\theta))$ 꼴이면, $\sigma^2(\theta)$가 작을수록 점근적으로 효율적이다.
- 평균제곱오차는 $MSE_\theta(\hat\eta_n) = E_\theta[(\hat\eta_n - \eta(\theta))^2] = \mathrm{Var}_\theta(\hat\eta_n) + \mathrm{Bias}_\theta(\hat\eta_n)^2$로 분해된다.

#### 예 8.4.1 로지스틱분포에서의 추정량 비교
- 모형: $L(\theta,1)$, $-\infty < \theta < \infty$ (위치모수)
- 후보 추정량:
    - 표본중앙값: $\hat\theta_n^{med} = X_{(\lfloor(n+1)/2\rfloor)}$
    - 표본평균: $\hat\theta_n^{mean} = \bar X$
    - 최대가능도추정량(MLE): $\hat\theta_n^{MLE}$
- 점근분포:
    $$
    \sqrt{n}(\hat\theta_n^{med} - \theta) \xrightarrow{d} N(0, 4)
    $$
    $$
    \sqrt{n}(\hat\theta_n^{mean} - \theta) \xrightarrow{d} N\left(0, \frac{\pi^2}{3}\right)
    $$
    $$
    \sqrt{n}(\hat\theta_n^{MLE} - \theta) \xrightarrow{d} N(0, 3)
    $$
- 점근분산이 작은 순(효율이 좋은 순): $\hat\theta_n^{MLE}\ (3) < \hat\theta_n^{mean}\ \left(\frac{\pi^2}{3}\right) < \hat\theta_n^{med}\ (4)$

### 점근상대효율성(ARE, asymptotic relative efficiency) 정의
- 두 추정량 $\hat\theta_n^{1}, \hat\theta_n^{2}$에 대해
    $$
    \sqrt{n}(\hat\theta_n^{i} - \theta) \xrightarrow{d} N(0, \sigma_i^2(\theta)),\quad i=1,2
    $$
    이면,
    $$
    ARE(\hat\theta_n^1, \hat\theta_n^2) = \frac{\sigma_1^{-2}(\theta)}{\sigma_2^{-2}(\theta)}
    $$
- $ARE > 1$이면 1번이 더 효율적(점근분산이 더 작음)이라는 뜻.

#### 예 8.4.2 로지스틱분포에서 추정오차한계와 점근상대효율성
- 표본중앙값으로 95% 점근신뢰구간:
    $$
    \theta \in \left[\hat\theta_n^{med} - 1.96\sqrt{\frac{4}{n}},\ \hat\theta_n^{med} + 1.96\sqrt{\frac{4}{n}}\right]
    $$
- 이를 $d$ 이하로 만들기 위한 표본크기:
    $$
    n_{med} \simeq \left(\frac{1.96}{d}\right)^2 \cdot 4
    $$
- 표본평균의 경우:
    $$
    n_{mean} \simeq \left(\frac{1.96}{d}\right)^2 \cdot \frac{\pi^2}{3}
    $$
- 따라서
    $$
    \frac{n_{mean}^{-1}}{n_{med}^{-1}} = \frac{12}{\pi^2} \approx 1.2 = ARE(\hat\theta_n^{mean}, \hat\theta_n^{med})
    $$
- 해석: 같은 오차한계를 달성하려면 **표본중앙값이 표본평균보다 대략 1.2배 더 많은 표본**을 필요로 한다.

#### 예 8.4.3 위치모수 모형에서 중앙값/평균/MLE 비교
- 연속형 위치모수 모형: 밀도 $f(x-\theta)$, 대칭 $f(-x)=f(x)$
- 점근분산
    - 표본중앙값: $\sqrt{n}(\hat\theta_n^{med} - \theta) \xrightarrow{d} N\left(0,\, (4f^2(0))^{-1}\right)$
    - 표본평균(2차모멘트 존재): $\sqrt{n}(\hat\theta_n^{mean} - \theta) \xrightarrow{d} N(0, \sigma_f^2)$, $\sigma_f^2 = \int_{-\infty}^{\infty} x^2 f(x)\,dx$
    - MLE(정규성 조건): $\sqrt{n}(\hat\theta_n^{MLE} - \theta) \xrightarrow{d} N(0, I_f^{-1})$, $I_f = \int_{-\infty}^{\infty} \left(\frac{f'(x)}{f(x)}\right)^2 f(x)\,dx$

#### 표 8.4.1 (점근분산 값 요약)
| 분포 | $(4f^2(0))^{-1}$ | $\sigma_f^2$ | $I_f^{-1}$ |
|------|------------------|--------------|------------|
| $N(\theta,1)$ | $\pi/2$ | $1$ | $1$ |
| $L(\theta,1)$ | $4$ | $\pi^2/3$ | $3$ |
| $DE(\theta,1)$ | $1$ | $2$ | $1$ |

#### 표 8.4.2 (ARE 요약)
- $ARE(\hat\theta_n^{mean}, \hat\theta_n^{med})$
    - $N(\theta,1)$: $\pi/2$
    - $L(\theta,1)$: $12/\pi^2$
    - $DE(\theta,1)$: $1/2$
- $ARE(\hat\theta_n^{mean}, \hat\theta_n^{MLE})$
    - $N(\theta,1)$: $1$
    - $L(\theta,1)$: $9/\pi^2$
    - $DE(\theta,1)$: $1/2$
- $ARE(\hat\theta_n^{med}, \hat\theta_n^{MLE})$
    - $N(\theta,1)$: $2/\pi$
    - $L(\theta,1)$: $3/4$
    - $DE(\theta,1)$: $1$

#### 예 8.4.4 베타분포 $\mathrm{Beta}(\alpha,1)$에서 추정량 비교
- 모형: $X_i \sim \mathrm{Beta}(\alpha,1)$, $\alpha>0$, 밀도 $f(x;\alpha)=\alpha x^{\alpha-1}I_{(0,1)}(x)$
- 모평균: $m_1 = \frac{\alpha}{\alpha+1}$

### 적률이용추정량(MME)
$$
\hat\alpha_n^{\mathrm{MME}} = \frac{\bar X}{1-\bar X}
$$

$$
\sqrt{n}(\hat\alpha_n^{\mathrm{MME}}-\alpha) \xrightarrow{d} N\left(0,\, \frac{\alpha(\alpha+1)^2}{\alpha+2}\right)
$$

### 최대가능도추정량(MLE)
가능도방정식:
$$
\sum_{i=1}^n \log X_i + \frac{n}{\alpha} = 0
$$
따라서
$$
\hat\alpha_n^{\mathrm{MLE}} = \frac{1}{-\overline{\log X}}
$$

$$
\sqrt{n}(\hat\alpha_n^{\mathrm{MLE}}-\alpha) \xrightarrow{d} N(0,\,\alpha^2)
$$

### 점근상대효율성
$$
ARE(\hat\alpha_n^{\mathrm{MME}},\,\hat\alpha_n^{\mathrm{MLE}})
= \frac{\alpha(\alpha+2)}{(\alpha+1)^2}
$$

따라서 MLE가 점근적으로 더 효율적이다.

### 정리 8.4.1 정보량 부등식 *(information inequality, Cramér–Rao 유형)*
- (MLE 점근정규성 조건들과 유사한 정칙성) 하에서, 실수값 모수 $\eta=\eta(\theta)$의 추정량 $\hat\eta_n$에 대해
$$
\mathrm{Var}_\theta(\hat\eta_n) \ge
\left(\frac{\partial}{\partial\theta}E_\theta(\hat\eta_n)\right)^{\!t}
[nI(\theta)]^{-1}
\left(\frac{\partial}{\partial\theta}E_\theta(\hat\eta_n)\right),
\quad \forall\theta
$$

#### 증명 아이디어:
    - 점수함수(score) $\dot l_n(\theta)=\frac{\partial}{\partial\theta}\sum_{i=1}^n \log f(X_i;\theta)$를 두고
    - 코시–슈바르츠 부등식으로 $\mathrm{Var}(\hat\eta_n)\,\mathrm{Var}(\dot l_n)\ge \mathrm{Cov}(\hat\eta_n,\dot l_n)^2$를 적용하여 도출

- 다차원 모수의 경우도 유사하게
$$
c^t\left(\mathrm{Var}_\theta(\hat\eta_n)-
\left(\frac{\partial}{\partial\theta}E_\theta(\hat\eta_n)\right)^{\!t}
[nI(\theta)]^{-1}
\left(\frac{\partial}{\partial\theta}E_\theta(\hat\eta_n)\right)\right)c \ge 0
$$
형태가 성립

### 정리 8.4.2 불편추정량에 대한 정보량 부등식
- $\theta=(\theta_1,\dots,\theta_k)^t$의 **불편추정량** $\hat\theta_n^{\mathrm{UE}}$에 대해
$$
c^t\mathrm{Var}_\theta(\hat\theta_n^{\mathrm{UE}})c \ge \frac{1}{n}c^tI^{-1}(\theta)c,
\quad \forall c,\ \forall\theta
$$

- 행렬로 쓰면 $\mathrm{Var}_\theta(\hat\theta_n^{\mathrm{UE}})-I^{-1}(\theta)/n$이 음이 아닌 정부호(positive semidefinite)임을 의미

#### 예 8.4.5 $\mathrm{Beta}(\alpha,1)$에서의 불편추정과 정보량 부등식
- $\mathrm{Beta}(\alpha,1)$는 지수족이며
$$
Y = \sum_{i=1}^n \log X_i
$$
가 $\alpha$에 대한 완비충분통계량임

- 변환 아이디어:
    - $X_i^\alpha \sim U(0,1) \implies -\alpha\log X_i \sim \mathrm{Exp}(1) \implies -\alpha\sum\log X_i \sim \mathrm{Gamma}(n,1)$

- 계산 결과:
$$
E_\alpha\left[\frac{1}{-\sum\log X_i}\right]=\frac{\alpha}{n-1}\quad (n\ge 2)
$$
$$
\mathrm{Var}_\alpha\left[\frac{1}{-\sum\log X_i}\right]=\frac{\alpha^2}{(n-1)^2(n-2)}\quad (n\ge 3)
$$

- 따라서 UMVUE:
$$
\hat\alpha_n^{\mathrm{UMVUE}} = \frac{n-1}{-\sum_{i=1}^n \log X_i}
$$
$$
\mathrm{Var}_\alpha(\hat\alpha_n^{\mathrm{UMVUE}}) = \frac{\alpha^2}{n-2}
$$

- 정보량 부등식의 하한:
$$
[nI(\alpha)]^{-1} = \frac{\alpha^2}{n}
$$
보다 크지만, 점근적으로는 같아진다:
$$
\lim_{n\to\infty}\frac{\mathrm{Var}_\alpha(\hat\alpha_n^{\mathrm{UMVUE}})}{[nI(\alpha)]^{-1}}=1
$$

- 또한 MLE와의 관계:
$$
\hat\alpha_n^{\mathrm{UMVUE}} = \frac{n-1}{n}\hat\alpha_n^{\mathrm{MLE}}
$$

### (정리 8.4.2 이후) 점근분산 하한과 Fisher의 추측
- 정리 8.4.2로부터 (일차원 모수에서) 점근정규성
    $\sqrt{n}(\hat\theta_n-\theta)\Rightarrow N(0,\sigma^2(\theta))$를 갖는 추정량은 대략
$$
\sigma^2(\theta) \ge \frac{1}{I(\theta)}
$$
같은 하한을 "추측"할 수 있다.

- 한편 MLE는
$$
\sqrt{n}(\hat\theta_n^{\mathrm{MLE}}-\theta)\Rightarrow N\left(0,\frac{1}{I(\theta)}\right)
$$
이므로, "점근정규성을 갖는 추정량들 중 MLE가 최소 극한분산을 갖는다"는 형태의 추측을 **Fisher의 추측(Fisher’s conjecture)** 이라 한다.

- 이 추측은 일반적으로는 성립하지 않고, 추가적인 **균등수렴(uniform convergence)** 조건 하에서만 성립함이 알려져 있다.

#### 예 8.4.6 감마분포 $\mathrm{Gamma}(\alpha,\beta)$에서의 추정량 비교
- 모형: $\mathrm{Gamma}(\alpha,\beta)$, $\theta=(\alpha,\beta)^t$
- MLE는 가능도방정식을 푸는 수치적 방법(일단계 반복법, one-step iteration)으로 근사

초기값: 적률이용추정량(MME)

$$
\hat\theta_n^{\mathrm{MME}} = (\hat\alpha_n, \hat\beta_n)^t
$$
는 연립방정식
$$
\hat\alpha_n\hat\beta_n = \bar X_n,\qquad
\hat\alpha_n(\hat\beta_n)^2 = \frac{1}{n}\sum_{i=1}^n (X_i-\bar X_n)^2
$$
의 해로 주어진다.

$$
\sqrt{n}(\hat\theta_n^{\mathrm{MME}}-\theta)\xrightarrow{d}N(0,\Sigma(\theta))
$$
$$
\Sigma(\theta)=
\begin{pmatrix}
2\alpha(\alpha+1) & -2(\alpha+1)\beta \\
-2(\alpha+1)\beta & (2+3/\alpha)\beta^2
\end{pmatrix}
$$

정보량 행렬과 역행렬

$$
I(\theta) = E_\theta[-\ddot l_1(\theta)] =
\begin{pmatrix}
\Psi'(\alpha) & 1/\beta \\
1/\beta & \alpha/\beta^2
\end{pmatrix},
\quad
I^{-1}(\theta) = \frac{1}{\alpha\Psi'(\alpha)-1}
\begin{pmatrix}
\alpha & -\beta \\
-\beta & \Psi'(\alpha)\beta^2
\end{pmatrix}
$$

일단계 반복법(one-step) 추정량

- $\hat\theta_n^{(0)}=\hat\theta_n^{\mathrm{MME}}$에서 시작해 뉴턴 형태로 한 번 갱신한 $\hat\theta_n^{(1)}$에 대해
$$
\sqrt{n}(\hat\theta_n^{(1)}-\theta)\xrightarrow{d}N(0,I^{-1}(\theta))
$$
가 성립 (즉, $\hat\theta_n^{(1)}$의 극한분포가 MLE의 극한분포와 동일)

성분별 점근상대효율성

- $\alpha$에 대해
$$
ARE(\hat\alpha_n^{\mathrm{MME}},\,\hat\alpha_n^{\mathrm{MLE}})
= \frac{[\alpha/(\alpha\Psi'(\alpha)-1)]^{-1}}{[2\alpha(\alpha+1)]^{-1}}
$$

- $\beta$에 대해
$$
ARE(\hat\beta_n^{\mathrm{MME}},\,\hat\beta_n^{\mathrm{MLE}})
= \frac{[\Psi'(\alpha)\beta^2/(\alpha\Psi'(\alpha)-1)]^{-1}}{[(2+3/\alpha)\beta^2]^{-1}}
$$

#### 표 두개