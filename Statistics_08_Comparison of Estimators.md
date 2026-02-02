# 제8장 추정량의 비교 *(Comparison of Estimators)*

## 추정량 비교의 기준 *(Criteria for Comparing Estimators)*

### 평균제곱오차 *(Mean Squared Error, MSE)*
모집단 분포가 확률밀도함수 $f(x;\theta),\ \theta \in \Omega$로 주어지고, 랜덤표본 $(X_1,\dots,X_n)$을 이용하여 모수 $\theta$ 또는 모수의 함수 $\eta=\eta(\theta)$를 추정한다고 하자.

좋은 추정량이란 참값에 가까운 추정값을 제공하는 추정량이다. 이를 정량적으로 평가하는 가장 보편적인 기준이 **평균제곱오차(Mean Squared Error, MSE)** 이다.

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
= \frac{2}{(n+1)(n+2)}\theta^2 \\
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
- 수렴 속도는 $\mathrm{MSE}(\hat\theta^{H},\theta)\sim n^{-1}$로 $\hat\theta^{\mathrm{MLE}} = \bar X$와 동일하다.
  - (TMI: Mill's ratio라고 불리는 근사식으로 $\mathrm{MSE}(\hat\theta^{H},\theta)\approx n^{-1}$ 증명 가능)
- $\theta=0$ 근방에서는 표본평균보다 더 작은 MSE를 가질 수 있다.
- 표본크기가 크면 혼합추정량이 표본평균과 거의 같은 성능
- 표본크기가 작으면 거의 모든 경우에 표본평균이 더 좋다

> 미지인 모수값의 범위에 따라 더 우수한 추정량이 바뀌니까 비교가 어렵다. 따라서 모수값에 의존하지 않는 비교기준이 필요하다. 대표적으로 최대평균제곱오차(maximum MSE), 베이지안평균제곱오차(Bayesian MSE)가 있다.
### 최대평균제곱오차 *(Maximum Mean Squared Error)*
모수값에 따라 우열이 바뀌는 문제를 피하기 위해, **모수 전 범위에서의 최악 성능**을 기준으로 비교한다.

평균제곱오차의 최댓값인 최대평균제곱오차
$$
\max_{\theta\in\Omega}
E_\theta[(\hat\eta-\eta(\theta))^2]
$$
를 최소화하는 추정량을 **최소최대(minimax) 평균제곱오차 추정량**이라 한다.

즉, minimax 평균제곱오차 추정량은
$$
\max_{\theta\in\Omega} E_\theta[(\hat\eta^*-\eta(\theta))^2] = \min_{\hat\eta} \max_{\theta\in\Omega} E_\theta[(\hat\eta-\eta(\theta))^2]
$$

### 베이지안 평균제곱오차 *(Bayesian Mean Squared Error)*
사전확률분포(사전밀도함수, prior distribution) $\pi(\theta)$가 주어졌다고 하자. 이때, 평균제곱오차(Mean Squared Error, MSE)를 모수공간 전체에 걸쳐 사전확률로 가중평균하여 다음과 같이 정의한다.
$$
r(\pi,\hat\eta)
=\int_\Omega E_\theta[(\hat\eta-\eta(\theta))^2]\pi(\theta)\,d\theta
$$
즉, 추정량 $\hat\eta$의 평균제곱오차를 사전확률분포 $\pi(\theta)$로 가중평균한 값을 **베이지안 평균제곱오차(Bayesian Mean Squared Error)** 라 한다.

이 값을 최소로 만드는 추정량 $\hat\eta^\pi$을 사전밀도함수 $\pi$에 대한 **베이지안 평균제곱오차 추정량(Bayesian MSE estimator)** 이라고 한다.  
사전밀도함수 $\pi$는 양수 또는 0의 값을 갖는 함수 또는 확률밀도함수로, 가중치의 정도를 반영하는 함수다. 

> **(참고) 평균제곱상대오차 *(Mean Squared Relative Error)***  
> $\eta(\theta)>0$인 경우, 평균제곱오차 대신 **평균제곱상대오차(Mean Squared Relative Error, MSRE)** 기준을 사용하는 경우도 많다. 이는 추정량의 상대적 정확도를 평가할 때 유용하다.  
> $$
> E_\theta\left[\left(\frac{\hat\eta}{\eta}-1\right)^2\right]
> $$
> 와 같이 정의하며, 최대평균제곱상대오차, 베이지안 평균제곱상대오차 등도 유사하게 정의된다.

#### 예 8.1.3 최대평균제곱상대오차 최소화
정규분포 $X_i\sim N(\mu,\sigma^2)$에서 모분산의 추정량을
$$
\hat\sigma_c^2
= c\sum_{i=1}^n(X_i-\bar X)^2,\quad c>0
$$
로 둘 때, 최대평균제곱상대오차를 최소로 하는 추정량은?
$$
\max_{\theta \in \Omega} E_\theta\left[\left(\frac{\hat\sigma_c^2}{\sigma^2}-1\right)^2\right]
$$

**풀이**
1. $\sum_{i=1}^n (X_i - \bar X)^2$는 $\sigma^2$에 대해 $(n-1)$ 자유도를 갖는 카이제곱 분포:
    $$
    \sum_{i=1}^n (X_i - \bar X)^2 \sim \sigma^2 \cdot \chi^2_{n-1}
    $$
    따라서
    $$
    \frac{\hat\sigma_c^2}{\sigma^2} = c \cdot \chi^2_{n-1}
    $$
    (여기서 $\chi^2_{n-1}$은 자유도 $n-1$인 표준화된 카이제곱 변수)

2. 평균제곱상대오차(MSRE)는
    $$
    E_\theta\left[\left(\frac{\hat\sigma_c^2}{\sigma^2}-1\right)^2\right]
    = E\left[(cY-1)^2\right]
    $$
    $\chi^2_k$의 평균은 $k$, 분산은 $2k$이므로, $Y = \frac{1}{n-1}\chi^2_{n-1}$, $E[Y]=1$, $Var(Y)=\frac{2}{n-1}$

3. $E[(cY-1)^2] = c^2 E[Y^2] - 2c E[Y] + 1$

    $E[Y^2] = Var(Y) + (E[Y])^2 = \frac{2}{n-1} + 1 = \frac{n+1}{n-1}$

    따라서
    $$
    E[(cY-1)^2] = c^2 \frac{n+1}{n-1} - 2c + 1
    $$

4. 이를 $c>0$에 대해 최소화하면,
    $$
    \frac{d}{dc} \left( c^2 \frac{n+1}{n-1} - 2c + 1 \right) = 2c \frac{n+1}{n-1} - 2 = 0
    $$
    $$
    c^* = \frac{n-1}{n+1}
    $$

5. 결론: **최대평균제곱상대오차를 최소로 하는 $c$는 $c^* = \frac{n-1}{n+1}$** 이고,
    $$
    \hat\sigma_{c^*}^2 = \frac{n-1}{n+1} \sum_{i=1}^n (X_i - \bar X)^2
    $$

### 불편추정량 *(Unbiased Estimator)*
한편 평균제곱오차를 기준으로 추정량을 비교할 때 비교대상인 추정량을 특정한 성질을 갖는 것으로 제한하여 비교하기도 한다. 특히 
$$E_\theta[\hat\eta(X_1, \dots, X_n)]=\eta(\theta),\quad \forall\theta\in\Omega$$
를 만족하는 추정량을 치우침 없는 추정량 또는 **불편추정량(unbiased estimator)** 라 하며, 이들 중에서 평균제곱오차를 최소로 하는 추정량을 찾기도 한다.

>불편추정량임을 나타내는 위 항등식은 간단히
>$$
>E_\theta[\hat\eta^{UE}]^{\theta \in \Omega}
>$$
>와 같이 표기하기도 한다.

불편추정량의 경우
$$
\mathrm{MSE}(\hat\eta, \theta) = E_\theta\left[(\hat\eta - \eta(\theta))^2\right]
= \operatorname{Var}_\theta(\hat\eta) + \left(E_\theta[\hat\eta] - \eta(\theta)\right)^2
$$
인데, (bias, variance decomposition!!) 불편추정량이면 $E_\theta[\hat\eta] = \eta(\theta)$이므로  
$$
\mathrm{MSE}(\hat\eta, \theta) = \operatorname{Var}_\theta(\hat\eta)
$$
즉, 분산을 최소화하는 것이 곧 MSE를 최소화하는 것, 최적의 추정량이다.

>#### 불편추정량의 의의
>불편추정량은 '장기적으로 평균을 내면 항상 정답을 맞힌다'는 성질을 가진다. 즉, 같은 실험을 무한히 반복했을 때 추정값들의 평균이 항상 참값 $\eta(\theta)$로 수렴한다.
>
>- 체계적 오차(편향)가 없으므로, 표본을 아무리 많이 모아도 >평균적으로는 참값에 도달한다.
>- 추정량 비교의 기준선 역할: "평균적으로는 맞히자"라는 최소한의 공정성 조건을 제공한다.
>- 이론적으로 MSE가 분산 하나로 환원되어 분석이 단순해진다.
>
>**주의점**
>- 불편성은 좋은 추정량의 충분조건이 아니다. 분산이 너무 크면 실제로는 매우 나쁜 추정량일 수 있다.
>- 약간의 편향이 있더라도 MSE가 더 작은(즉, 전체적으로 더 나은) 추정량이 존재할 수 있다.
>- 따라서 불편성은 최적성의 조건이 아니라, 하나의 제약조건에 불과하다.

불편추정량 중에서, 모수 $\theta$의 참값이 무엇이든 그 분산을 최소로 하는 추정량이 있다면, 그 추정량을 전역최소분산불편추정량 **(Uniformly Minimum Variance Unbiased Estimator, UMVUE)** 라 한다
### 전역최소분산불편추정량 *(Uniformly Minimum Variance Unbiased Estimator, UMVUE)*
추정량 $\hat\eta^*$가 다음을 만족하면 UMVUE라 하고 $\hat\eta^{UMVUE}$로 나타내기도 한다.
1. $E_\theta[\hat\eta^*]=\eta(\theta)$
2. 임의의 불편추정량 $\hat\eta^{UE}$에 대해
    $\mathrm{Var}_\theta(\hat\eta^*) \le \mathrm{Var}_\theta(\hat\eta^{UE}),\quad \forall\theta\in\Omega$

>#### UMVUE의 의의
>UMVUE는 불편추정량 중에서 분산이 가장 작은, 즉 평균적으로는 항상 맞히면서도 가장 덜 흔들리는 추정량이다.  
>즉, **불편성**이라는 제약을 절대 포기하지 않을 때, 그 안에서 더 이상 개선이 불가능한 "최적의 불편추정량"이 바로 UMVUE다.
>
>추후 다양한 이론이 UMVUE에서 시작된다!
>- UMVUE에서 “Uniformly”란 **특정 $\theta$에서만 좋은 것이 아니라, 모든 $\theta \in \Omega$에 대해 분산이 최소**임을 뜻한다.
>- 즉, **국소 최적(local optimum)이 아니라 전역 최적(global optimum)** 으로, 모수공간 전체에서 항상 최적의 불편추정량임을 보장한다.
>- 이는 일부 $\theta$에서만 분산이 최소인 추정량과 구별되는 중요한 특징이다.

### 정리 8.1.1 최소분산불편추정량의 유일성 *(Uniqueness of UMVUE)*
UMVUE의 분산이 유한하면, UMVUE는 유일하다 (거의 확실하게).
$$
P_\theta(\hat\eta_1 = \hat\eta_2) = 1,\quad \forall\theta\in\Omega
$$

#### 증명
$\hat\eta_1, \hat\eta_2$가 모두 UMVUE라고 하자. 즉, 두 추정량 모두 $\eta(\theta)$의 불편추정량이며, 모든 $\theta \in \Omega$에 대해 분산이 최소이다.

이제 $\hat\eta_3 = \frac{\hat\eta_1 + \hat\eta_2}{2}$를 고려하자. $\hat\eta_3$ 역시 $\eta(\theta)$의 불편추정량임을 쉽게 확인할 수 있다:
$$
E_\theta[\hat\eta_3] = E_\theta\left[\frac{\hat\eta_1 + \hat\eta_2}{2}\right] = \frac{E_\theta[\hat\eta_1] + E_\theta[\hat\eta_2]}{2} = \frac{\eta(\theta) + \eta(\theta)}{2} = \eta(\theta)
$$

UMVUE의 정의에 따라, $\hat\eta_1$과 $\hat\eta_2$는 모두 불편추정량 중에서 분산이 최소이므로,
$$
\mathrm{Var}_\theta(\hat\eta_1) \le \mathrm{Var}_\theta(\hat\eta_3), \qquad
\mathrm{Var}_\theta(\hat\eta_2) \le \mathrm{Var}_\theta(\hat\eta_3)
$$
또한, $\hat\eta_1$과 $\hat\eta_2$의 분산은 같으므로(둘 다 UMVUE이므로),
$$
\mathrm{Var}_\theta(\hat\eta_1) = \mathrm{Var}_\theta(\hat\eta_2) \le \mathrm{Var}_\theta(\hat\eta_3)
$$

한편, $\hat\eta_3$의 분산은 다음과 같다:
$$
\mathrm{Var}_\theta(\hat\eta_3) = \mathrm{Var}_\theta\left(\frac{\hat\eta_1 + \hat\eta_2}{2}\right)
= \frac{1}{4} \mathrm{Var}_\theta(\hat\eta_1 + \hat\eta_2)
= \frac{1}{4} \left( \mathrm{Var}_\theta(\hat\eta_1) + \mathrm{Var}_\theta(\hat\eta_2) + 2\mathrm{Cov}_\theta(\hat\eta_1, \hat\eta_2) \right) \\
= \frac{1}{4} \mathrm{Var}_\theta(\hat\eta_1) + \frac{1}{4} \mathrm{Var}_\theta(\hat\eta_2) + \frac{1}{2} \mathrm{Cov}_\theta(\hat\eta_1, \hat\eta_2)
$$

위에서 $\mathrm{Var}_\theta(\hat\eta_1) \le \mathrm{Var}_\theta(\hat\eta_3)$이므로,
$$
\mathrm{Var}_\theta(\hat\eta_1) \le \frac{1}{4} \mathrm{Var}_\theta(\hat\eta_1) + \frac{1}{4} \mathrm{Var}_\theta(\hat\eta_2) + \frac{1}{2} \mathrm{Cov}_\theta(\hat\eta_1, \hat\eta_2)
$$
즉,
$$
\frac{3}{4} \mathrm{Var}_\theta(\hat\eta_1) - \frac{1}{4} \mathrm{Var}_\theta(\hat\eta_2) \le \frac{1}{2} \mathrm{Cov}_\theta(\hat\eta_1, \hat\eta_2) \\
\therefore \mathrm{Var}_\theta(\hat\eta_1) \le \mathrm{Cov}_\theta(\hat\eta_1, \hat\eta_2)
$$
이때,
$$
\mathrm{Var}_\theta(\hat\eta_1 - \hat\eta_2)
= \mathrm{Var}_\theta(\hat\eta_1) + \mathrm{Var}_\theta(\hat\eta_2) - 2\,\mathrm{Cov}_\theta(\hat\eta_1, \hat\eta_2)
$$
위에서 $\mathrm{Var}_\theta(\hat\eta_1) \le \mathrm{Cov}_\theta(\hat\eta_1, \hat\eta_2)$, $\mathrm{Var}_\theta(\hat\eta_2) \le \mathrm{Cov}_\theta(\hat\eta_1, \hat\eta_2)$이므로,
$$
\mathrm{Var}_\theta(\hat\eta_1 - \hat\eta_2) \le 0 \\
\therefore \mathrm{Var}_\theta(\hat\eta_1 - \hat\eta_2) = 0
$$
정리 1.6.4(분산 0의 의미)에 따라, $\hat\eta_1 - \hat\eta_2 = 0$이 거의 확실하게 성립한다. 즉,
$$
P_\theta(\hat\eta_1 = \hat\eta_2) = 1,\quad \forall\theta\in\Omega
$$
따라서 UMVUE는 유일하다.

## 충분통계량 *(Sufficient Statistics)*

### 충분성의 동기 *(Motivation of Sufficiency)*
랜덤표본 $X_1,\dots,X_n$을 이용해 모수 $\theta$를 추론할 때, 표본 전체가 아니라 **일부 정보만으로도 동일한 추론 정확도**를 얻을 수 있다면 데이터 저장·계산 측면에서 유리하다.

이러한 **자료 축약(reduction)** 의 핵심 개념이 **충분성(sufficiency)** 이다.

#### 예 8.2.1 두 번의 베르누이 시행 *(Bernoulli Trials)*
서로 독립인 $X_1,X_2 \sim \mathrm{Bernoulli}(\theta),\ 0<\theta<1$에서 $(X_1,X_2)$ 대신 $Y=X_1+X_2$만 관측한다고 하자.

- $Y=0 \Rightarrow (0,0)$
- $Y=2 \Rightarrow (1,1)$
- $Y=1 \Rightarrow (1,0)$ 또는 $(0,1)$

조건부확률은
$P_\theta(X_1=1,X_2=0\mid Y=1) = P_\theta(X_1=0,X_2=1\mid Y=1) = \frac12$  
이는 $\theta$에 의존하지 않는다.  
즉, 성공 횟수 $Y$만 알면 순서 정보는 $\theta$ 추론에 불필요하다.

>추가 예시:  
>$X_1, X_2 \sim N(\mu, 1)$에서 $(X_1, X_2)$ 대신 $Y = X_1 + X_2$만 알 때,  
>$P_\mu(X_1 = x_1, X_2 = x_2 \mid Y = y)$는 $\mu$에 무관하다.  
>즉, $Y$만으로 $\mu$ 추론에 필요한 정보가 모두 담겨 있으므로 $Y$는 $\mu$에 대한 충분통계량이다.
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
$X_1,\dots,X_n \sim \mathrm{Bernoulli}(\theta)$, $Y=\sum_{i=1}^n X_i \sim \mathrm{Binomial}(n,\theta)$

조건부확률은
$$
P_\theta(X_1=x_1,\dots,X_n=x_n\mid Y=y)
=
\binom{n}{y}^{-1} I\Big(\sum x_i=y\Big)
$$
이는 $\theta$에 무관하다.  
**결론:** $Y=\sum_{i=1}^n X_i$는 $\theta\in(0,1)$에 대한 충분통계량이다.

### 정리 8.2.1 분해 정리 *(Factorization Theorem)*
통계량 $Y=u(X_1,\dots,X_n)$이 $\theta$에 대한 충분통계량일 **필요충분조건**은
$$
\prod_{i=1}^n f(x_i;\theta) = k_1(u(x),\theta)\,k_2(x)
$$
를 만족하는 "함수" $k_1, k_2$가 존재하는 것이다.

**정리 8.2.1 (분해 정리) 증명**
이산형만 증명하겠다. 일반적으로 성립한다.  
#### (⇒) 충분조건: 분해형이면 충분통계량이다
분포를
$$
\prod_{i=1}^n f(x_i;\theta) = k_1(u(x),\theta)\,k_2(x)
$$
꼴로 쓸 수 있다고 하자. $Y = u(X_1,\dots,X_n)$에 대해, $Y=y$일 때 $X=x$의 조건부확률은
$$
P_\theta(X=x \mid Y=y)
= \frac{P_\theta(X=x)}{P_\theta(Y=y)}
= \frac{k_1(u(x),\theta)\,k_2(x)}{\sum_{z:u(z)=y} k_1(u(z),\theta)\,k_2(z)}
$$
여기서 $u(x)=y$이므로 $k_1(u(x),\theta) = k_1(y,\theta)$로 고정된다:
$$
= \frac{k_1(y,\theta)\,k_2(x)}{k_1(y,\theta)\sum_{z:u(z)=y} k_2(z)}
= \frac{k_2(x)}{\sum_{z:u(z)=y} k_2(z)}
$$
즉, $\theta$에 무관하다. 따라서 $Y$는 충분통계량이다.

#### (⇐) 필요조건: 충분통계량이면 분해형이 된다
$Y = u(X_1,\dots,X_n)$이 충분통계량이라고 하자. $x$에 대해 $y = u(x)$로 두고,
$$
P_\theta(X=x \mid Y=y) = k_2(x)
$$
꼴로 쓸 수 있다. $P_\theta(Y=y)$를 $k_1(y,\theta)$로 두면,
$$
P_\theta(X=x) = P_\theta(X=x \mid Y=y) \cdot P_\theta(Y=y)
= k_2(x)\,k_1(y,\theta)
$$
즉,
$$
\prod_{i=1}^n f(x_i;\theta) = k_1(u(x),\theta)\,k_2(x)
$$
형태로 쓸 수 있다.
#### 예 8.2.3 포아송 분포 *(Poisson)*  
$X_i \sim \mathrm{Poisson}(\theta)$의 결합확률질량함수는  
$$
\prod_{i=1}^n f(x_i;\theta) = \prod_{i=1}^n \frac{e^{-\theta}\theta^{x_i}}{x_i!} = e^{-n\theta}\theta^{\sum x_i} \prod_{i=1}^n \frac{1}{x_i!}
$$
여기서 $k_1\left(\sum x_i,\,\theta\right) = e^{-n\theta}\theta^{\sum x_i}$, $k_2(x) = \prod_{i=1}^n \frac{1}{x_i!}$라 하면, 분해정리에 의해 $\sum X_i$는 $\theta$에 대한 충분통계량이다.

#### 예 8.2.4 지수분포 *(Exponential)*  
$X_i \sim \mathrm{Exp}(\theta)$의 결합확률밀도함수는  
$$
\prod_{i=1}^n f(x_i;\theta) = \prod_{i=1}^n \frac{1}{\theta} \exp\left(-\frac{x_i}{\theta}\right) I_{(0,\infty)}(x_i) = \theta^{-n} \exp\left(-\frac{1}{\theta} \sum x_i\right) \prod_{i=1}^n I_{(0,\infty)}(x_i)
$$
여기서 $k_1\left(\sum x_i,\,\theta\right) = \theta^{-n} \exp\left(-\frac{1}{\theta} \sum x_i\right)$, $k_2(x) = \prod_{i=1}^n I_{(0,\infty)}(x_i)$이므로, 분해정리에 의해 $\sum X_i$는 $\theta$에 대한 충분통계량이다.

#### 예 8.2.5 감마분포 *(Gamma)*  
$X_i \sim \mathrm{Gamma}(\alpha, \beta)$의 결합확률밀도함수는  
$$
\prod_{i=1}^n f(x_i;\alpha,\beta) = \prod_{i=1}^n \frac{1}{\Gamma(\alpha)\beta^\alpha} x_i^{\alpha-1} e^{-x_i/\beta}
= \frac{1}{[\Gamma(\alpha)]^n \beta^{n\alpha}} \left(\prod_{i=1}^n x_i\right)^{\alpha-1} \exp\left(-\frac{1}{\beta} \sum_{i=1}^n x_i\right)
$$
여기서 $k_1\left(\sum x_i,\,\sum \log x_i,\,\alpha,\beta\right) = \beta^{-n\alpha} \left(\prod x_i\right)^{\alpha-1} \exp\left(-\frac{1}{\beta} \sum x_i\right)$, $k_2(x) = 1$로 둘 수 있다. 즉, 분해정리에 의해 $u(X) = \left(\sum X_i,\, \sum \log X_i\right)$가 $(\alpha, \beta)$에 대한 충분통계량이다.

#### 증명  
분해정리(정리 8.2.1)에 따르면, 결합확률밀도함수  
$$
\prod_{i=1}^n f(x_i;\theta) = \exp\left\{ \sum_{i=1}^n \sum_{j=1}^k g_j(\theta) T_j(x_i) - nA(g(\theta)) + \sum_{i=1}^n S(x_i) \right\}
$$  
에서 $g_j(\theta)\sum_{i=1}^n T_j(x_i)$ 부분이 $\theta$와 $x$를 연결하므로,  
$$
\sum_{i=1}^n T(X_i)
$$  
가 충분통계량임을 알 수 있다.

### 정리 8.2.3 충분통계량의 일대일 함수 *(Function of Sufficient Statistic)*
$Y$가 충분통계량이고 $W = g(Y)$가 **일대일 함수**이면 $W$도 충분통계량이다.

#### 증명
$Y$가 충분통계량이므로, 임의의 집합 $A$와 모든 $y$, $\theta_1, \theta_2$에 대해
$$
P_{\theta_1}\big((X_1,\dots,X_n)\in A \mid Y=y\big)
= P_{\theta_2}\big((X_1,\dots,X_n)\in A \mid Y=y\big)
$$

$W = g(Y)$가 일대일 함수이므로, $W=w$일 때 $Y$는 유일하게 결정된다. 즉, $Y = g^{-1}(w)$.

따라서
$$
P_{\theta}\big((X_1,\dots,X_n)\in A \mid W=w\big)
= P_{\theta}\big((X_1,\dots,X_n)\in A \mid Y=g^{-1}(w)\big)
$$
이므로, $W$에 대한 조건부분포 역시 $\theta$에 무관하다.  
결론적으로, $W$도 충분통계량이다.

#### 예 8.2.6 정규모집단의 경우
$X_i \sim N(\mu, \sigma^2)$  
충분통계량: $\left(\sum X_i,\, \sum X_i^2\right)$  
동치 표현: $\left(\bar X,\, \sum (X_i - \bar X)^2\right)$

#### 예 8.2.7 $Beta(\alpha, 1)$의 경우
$f(x;\alpha) = \alpha x^{\alpha-1}$  
충분통계량:
$\sum \log X_i$ 또는 $\overline{\log X}$

#### 예 8.2.8 두 개의 모수를 갖는 지수분포 *(Shifted Exponential)*
모집단 분포의 토대가 모수에 의존하면 정리8.2.2를 적용할 수 없지만, 그 토대를 나타내는 지표함수를 고려하여 분해정리를 이용하면 충분통계량을 찾을 수 있다.  

$f(x;\mu,\sigma) = \frac{1}{\sigma} e^{-(x-\mu)/\sigma} I_{[\mu,\infty)}(x)$  
결합확률밀도함수는
$$
\prod_{i=1}^n f(x_i;\mu,\sigma)
= \prod_{i=1}^n \frac{1}{\sigma} e^{-(x_i-\mu)/\sigma} I_{[\mu,\infty)}(x_i)
= \sigma^{-n} \exp\left(-\frac{1}{\sigma} \sum_{i=1}^n (x_i-\mu)\right) \prod_{i=1}^n I_{[\mu,\infty)}(x_i)
$$
여기서 $\prod_{i=1}^n I_{[\mu,\infty)}(x_i) = I_{[\mu,\infty)}(\min x_i)$이므로,
$$
= \sigma^{-n} \exp\left(-\frac{1}{\sigma} \sum_{i=1}^n (x_i-\mu)\right) I_{[\mu,\infty)}(\min x_i)
$$
이 등식의 우변을 $k_1\left(\sum x_i,\,\min x_i,\,\mu,\,\sigma\right)$과 $k_2(x_1,\dots,x_n)=1$의 곱으로 나타낼 수 있으므로, 분해정리에 의해  
$(\sum X_i,\,\min X_i)$는 $(\mu,\sigma)$에 대한 충분통계량이다.

#### 예 8.2.9 균등분포 $U[\theta_1, \theta_2]$에서의 충분통계량
$X_1, \dots, X_n \sim U[\theta_1, \theta_2]$일 때, 결합확률밀도함수는
$$
f(x;\theta_1,\theta_2) = (\theta_2-\theta_1)^{-n} I(\theta_1 \le \min X_i,\, \max X_i \le \theta_2)
$$

분해정리에 따라, $k_1(\min X_i, \max X_i, \theta_1, \theta_2) = (\theta_2-\theta_1)^{-n} I(\theta_1 \le \min X_i,\, \max X_i \le \theta_2)$, $k_2(x) = 1$로 쓸 수 있으므로,  
충분통계량은 $(\min X_i,\, \max X_i)$

즉, 표본의 최솟값과 최댓값만 알면 $\theta_1, \theta_2$에 대한 모든 정보가 보존된다.

### 정리 8.2.4 최대가능도추정량과 충분통계량
MLE $\hat\theta^{\mathrm{MLE}}$가 유일하면, 이는 **임의의 충분통계량 $S=(X_1, \dots, X_n)$** 의 함수다.
$$
\hat\theta^{\mathrm{MLE}}(X_1, \dots, X_n) = f(S)
$$

#### 증명
MLE $\hat\theta^{\mathrm{MLE}}$는 표본 $(X_1, \dots, X_n)$에서 가능도를 최대화하는 값이다. 충분통계량 $S = S(X_1, \dots, X_n)$이 존재하면, 분해정리에 의해 결합확률(밀도)함수는
$$
L(\theta; X_1, \dots, X_n) = k_1(S, \theta) k_2(X_1, \dots, X_n)
$$
꼴로 쓸 수 있다. $k_2$는 $\theta$에 무관하므로, $\theta$에 대한 최대화는 $k_1(S, \theta)$만 고려하면 된다. 즉,
$$
\hat\theta^{\mathrm{MLE}} = \arg\max_\theta L(\theta; X_1, \dots, X_n) = \arg\max_\theta k_1(S, \theta)
$$
따라서 $\hat\theta^{\mathrm{MLE}}$는 $S$의 함수로 표현된다.

### 정리 8.2.5 Rao–Blackwell 정리 *(Estimator Improvement)* 
정리8.2.4에서 유일하게 존재한다고 가정된 최대가능도 추정량 $Y=\hat\theta^{MLE}(X_1, \dots, X_n)$가 추가적으로 $\theta\in\Omega$에 대한 충분통계량이면, 이는 임의의 다른 충분통계량 $S=u(X_1, \dots, X_n)$의 함수로 나타내지는 충분통계량으로서, 이런 충분통계량을 최소충분통계량(minimal sufficient statistic)라 한다.  
예8.2.2부터 예8.2.9까지 모형에서의 최대가능도추정량은 모두 유일하게 존재하고 모수에 관한 충분통계량이므로, 이들 모두 모수에 관한 최소충분통계량이다.  

$\hat\eta$가 추정량이고 $Y$가 충분통계량이면
$$
\hat\eta^{RB} = E(\hat\eta(X_1, \dots, X_n) \mid Y)
$$
는 항상
$$
\mathrm{MSE}(\hat\eta^{RB}, \theta) \le \mathrm{MSE}(\hat\eta, \theta), \ \forall \theta \in \Omega
$$

>Rao–Blackwell 정리의 의의
>- 임의의 추정량이 주어졌을 때, 충분통계량에 대한 조건부기대값을 취하면 항상 평균제곱오차(MSE)가 감소하거나 같아진다.
>- 즉, **충분통계량을 활용하면 추정량을 반드시 개선**할 수 있다.
>- 이 과정을 반복하면, 충분통계량의 함수로 표현되는 "최적의" 추정량(특히 불편추정량의 경우 UMVUE)에 도달할 수 있다.
>- 실전적으로는, 임의의 불편추정량을 Rao–Blackwell화하면 항상 더 나은(또는 같은) 불편추정량을 얻을 수 있다는 점에서 매우 강력한 추정량 개선 도구이다.

#### 증명
$\hat\eta^{RB} = E(\hat\eta \mid Y)$로 정의한다. 평균제곱오차(MSE)는
$$
\mathrm{MSE}(\hat\eta, \theta) = E_\theta\left[(\hat\eta - \eta(\theta))^2\right]
$$
조건부기대값의 분해(분산의 법칙)에 의해
$$
E_\theta\left[(\hat\eta - \eta(\theta))^2\right]
= E_\theta\left[\,E_\theta\left[(\hat\eta - \eta(\theta))^2 \mid Y\right]\,\right] \\
= E_\theta\left[\,\operatorname{Var}_\theta(\hat\eta \mid Y) + (E_\theta[\hat\eta \mid Y] - \eta(\theta))^2\,\right] \\
= E_\theta\left[\operatorname{Var}_\theta(\hat\eta \mid Y)\right] + E_\theta\left[(\hat\eta^{RB} - \eta(\theta))^2\right] \\
\therefore \mathrm{MSE}(\hat\eta, \theta) = E_\theta\left[\operatorname{Var}_\theta(\hat\eta \mid Y)\right] + \mathrm{MSE}(\hat\eta^{RB}, \theta)
$$
이고, $E_\theta[\operatorname{Var}_\theta(\hat\eta \mid Y)] \ge 0$이기 때문에
$$
\mathrm{MSE}(\hat\eta^{RB}, \theta) \le \mathrm{MSE}(\hat\eta, \theta)
$$
즉, 충분통계량에 대한 조건부기대값으로 추정량을 개선하면 항상 MSE가 감소하거나 같아진다.

#### 예 8.2.10 균등분포 $U(0, \theta)$
결합확률밀도함수는
$$
\prod_{i=1}^n f(x_i;\theta) = \prod_{i=1}^n \frac{1}{\theta} I_{(0,\theta)}(x_i) = \theta^{-n} I_{(0 < \min x_i < \max x_i < \theta)}
$$
이므로, 분해정리에 따라 $k_1(\max x_i, \theta) = \theta^{-n} I_{(\max x_i < \theta)}$, $k_2(x) = I_{(0 < \min x_i)}$로 쓸 수 있다. 따라서 $Y = \max X_i$가 $\theta > 0$에 대한 충분통계량이다.

초기 추정량: $\hat\theta = 2\bar X$이고, $\hat\theta$는 $\theta$의 불편추정량.  
평균제곱오차(MSE)는
$$
\mathrm{MSE}(\hat\theta, \theta) = E_\theta[(2\bar X - \theta)^2] = 4Var_{\theta}(\bar X) = \frac{\theta^2}{3n}
$$

**Rao–Blackwell 개선:**  
Rao–Blackwell 정리에 따라, 충분통계량 $X_{(n)}$에 대한 조건부기대값을 취하면 더 나은 추정량을 얻을 수 있다.
- 개선된 추정량: $\hat\theta^{RB} = E(2\bar X \mid X_{(n)})$
- $U(0, \theta)$에서 $X_{(n)} = t$일 때, $X_1, \dots, X_n$의 조건부분포는 $[0, t]$에서 균등하게 분포하며, $X_{(n)} = t$는 $n$개 중 하나가 $t$이고 나머지는 $[0, t]$에서 iid 균등이다.
- 따라서 $E(\bar X \mid X_{(n)} = t) = \frac{n t}{n+1}$

따라서  
- $X_{(n)} = \max(X_1, \dots, X_n)$일 때, $E(\bar X \mid X_{(n)} = t) = \frac{n t}{n+1}$이므로  
- Rao–Blackwell 개선 추정량은  
    $$
    \hat\theta^{RB} = 2 \cdot \frac{n X_{(n)}}{n+1} = \frac{2n}{n+1} X_{(n)}
    $$
- 하지만 실제로 $\frac{n+1}{n} X_{(n)}$가 $\theta$의 불편추정량이므로, Rao–Blackwell 개선 추정량은  
    $$
    \hat\theta^{RB} = \frac{n+1}{n} X_{(n)}
    $$
즉, 표본의 최댓값에 $(n+1)/n$을 곱한 것이 $\theta$의 불편추정량이자 Rao–Blackwell 개선 추정량이다.

**MSE 개선 전후 비교**  
$$
f_{X_{(n)}}(x) = n \frac{x^{n-1}}{\theta^n},\quad 0 < x < \theta
$$
- $E[X_{(n)}] = \frac{n}{n+1}\theta$
- $E[X_{(n)}^2] = \frac{n}{n+2}\theta^2$
$$
\mathrm{MSE}(\hat\theta^{RB}, \theta)
= E\left[\left(\frac{n+1}{n} X_{(n)} - \theta\right)^2\right]
= \left(\frac{n+1}{n}\right)^2 E[X_{(n)}^2] - 2\theta\frac{n+1}{n} E[X_{(n)}] + \theta^2 \\
= \left(\frac{n+1}{n}\right)^2 \cdot \frac{n}{n+2}\theta^2 - 2\theta\frac{n+1}{n} \cdot \frac{n}{n+1}\theta + \theta^2 \\
= \frac{(n+1)^2}{n(n+2)}\theta^2 - 2\theta^2 + \theta^2
= \frac{(n+1)^2 - 2n(n+2) + n(n+2)}{n(n+2)}\theta^2
= \frac{\theta^2}{n(n+2)}
$$

**비교:**  
$$
\mathrm{MSE}(\hat\theta^{RB}, \theta) = \frac{\theta^2}{n(n+2)} < \frac{\theta^2}{3n} = \mathrm{MSE}(\hat\theta, \theta)
$$

즉, Rao–Blackwell 개선을 통해 충분통계량을 사용하면 MSE가 더 작아진다.


## 최소분산불편추정 *(Minimum Variance Unbiased Estimation)*
앞 절에서 다음 사실을 확인하였다:
- 충분통계량을 이용하면 추정량을 개선할 수 있다.
- Rao–Blackwell 정리를 통해 **평균제곱오차(MSE)** 를 감소시킬 수 있다.  
그렇다면, **불편추정량(unbiased estimator) 중에서, 모든 모수값에 대해 분산이 가장 작은 추정량은 무엇인가?**  
이에 대한 해답이 **최소분산불편추정**이며, 그 궁극적 결과가 **전역최소분산불편추정량(UMVUE)** 이다.

### 완비통계량 *(Complete Statistic)*
모집단 분포가 $f(x;\theta),\ \theta\in\Omega$이고, 통계량 $Y = u(X_1, \dots, X_n)$에 대해
$$
E_\theta[g(Y)] = 0\ \forall\theta\in\Omega \implies P_\theta(g(Y)=0) = 1\ \forall\theta\in\Omega
$$
이면 $Y$를 $\theta$에 대한 **완비통계량**이라 한다.

- 평균이 0인 함수는 거의 확실하게 0이어야 한다.
- 즉, $Y$ 안에는 "중복되는 정보"가 존재하지 않는다.
  - 쓸모없는 흔들림이 없다. (흔들림은 평균하면 0이니까)
통계량 $Y$가 충분통계량(sufficient statistic)이며 완비통계량(complete statistic)이면 이를 **완비충분통계량(Complete Sufficient Statistic)** 이라 한다.  
이 개념은 UMVUE 존재·유일성의 핵심 전제이다.

완비통계량은 **최소분산불편추정량(UMVUE)의 존재와 유일성**을 보장하는 핵심 개념이다.  
- 충분통계량만으로는 UMVUE가 여러 개 존재할 수 있지만, 완비성까지 만족하면 UMVUE가 유일하게 결정된다.
- 완비충분통계량이 존재하면, 그 함수로 표현되는 불편추정량 중에서 분산이 가장 작은 추정량(UMVUE)을 항상 찾을 수 있다.
- Rao–Blackwell 정리와 결합하여, 임의의 불편추정량을 완비충분통계량의 함수로 개선하면 반드시 최적(UMVUE)에 도달한다.

즉, **완비통계량은 추정이론에서 "최적의 불편추정량"을 찾는 데 필수적인 역할**을 한다.

### 정리 8.3.1 완비충분통계량을 이용한 (Uniformly Minimum Variance Unbiased Estimator, UMVUE))
>모수의 함수 $\eta=\eta(\theta)$에 대한 불편추정량 $\hat\eta^*$가
>$$
>\mathrm{Var}_\theta(\hat\eta^*) \le \mathrm{Var}_\theta(\hat\eta)
>\quad \forall \theta\in\Omega
>$$
>를 모든 불편추정량 $\hat\eta$에 대해 만족하면, $\hat\eta^*$를 **전역최소분산불편추정량(UMVUE)** 라 한다.

모집단 분포 $f(x;\theta)$에서 랜덤표본 $(X_1,\dots,X_n)$에 대해 통계량 $Y = u(X_1,\dots,X_n) $이 $\theta$에 대한 **완비충분통계량**이라고 하자.
- **(a) Rao–Blackwell 형태**  
    $\eta=\eta(\theta)$의 임의의 불편추정량 $\hat\eta^{UE}$에 대해
    $$
    \hat\eta^{RB}(Y) = E(\hat\eta^{UE} \mid Y)
    $$
    로 정의하면, $\hat\eta^{RB}(Y)$는 $\eta(\theta)$의 UMVUE이다.

- **(b) 함수형 UMVUE**  
    $Y$의 함수 $\delta(Y)$가 $\eta(\theta)$의 불편추정량이면, $\delta(Y)$는 $\eta(\theta)$의 UMVUE이다.

#### 증명
(a) Rao–Blackwell 형태:  
(1) 조건부기댓값의 성질로부터  
$$
E_\theta[\hat\eta^{RB}(Y)] = E_\theta[E(\hat\eta^{UE} \mid Y)] = E_\theta[\hat\eta^{UE}] = \eta(\theta)
$$
즉, $\hat\eta^{RB}(Y)$는 $\eta(\theta)$의 불편추정량이다.

(2) 정리 8.2.5(Rao–Blackwell 정리)에 의해, 임의의 불편추정량 $\hat\eta^{UE}$에 대해  
$$
\mathrm{Var}_\theta(\hat\eta^{RB}(Y)) \le \mathrm{Var}_\theta(\hat\eta^{UE}),\quad \forall\theta\in\Omega
$$
특히, $\hat\eta^{RB}(Y)$는 $Y$의 함수이므로, $Y$의 함수로 표현되는 모든 불편추정량 중에서도 분산이 최소이다.

(3) $Y$가 완비통계량이므로, $Y$의 함수로 표현되는 불편추정량이 둘 이상 존재한다면 그 차이는 0이어야 한다(완비성의 정의). 즉, $\hat\eta^{RB}(Y)$는 유일하다.

따라서 (1), (2), (3)으로부터 $\hat\eta^{RB}(Y)$는 $\eta(\theta)$의 UMVUE이다.

(b) 함수형 UMVUE:  
(a)의 증명 (2)에서 $\hat\eta^{RB}(Y)$의 역할을 임의의 $Y$의 함수 $\delta(Y)$로 바꾸면, $\delta(Y)$가 $\eta(\theta)$의 불편추정량이면 위와 동일하게 $\delta(Y)$가 UMVUE임을 알 수 있다.


#### 예 8.3.1 베르누이 독립시행
$X_i \sim \mathrm{Bernoulli}(\theta),\ 0<\theta<1$  

충분통계량: $Y = \sum X_i$ (예8.2.2)

$Y = \sum_{i=1}^n X_i \sim \mathrm{Binomial}(n, \theta)$이므로, $Y$의 확률질량함수는
$$
P_\theta(Y = y) = \binom{n}{y} \theta^y (1-\theta)^{n-y},\quad y=0,1,\dots,n
$$
$Y$가 $\theta$에 대한 완비통계량임을 보이려면, 임의의 함수 $g$에 대해
$$
E_\theta[g(Y)] = 0\quad \forall\,\theta\in(0,1)
$$
이면 $P_\theta(g(Y)=0)=1$임을 보여야 한다.

$E_\theta[g(Y)]$를 전개하면
$$
E_\theta[g(Y)] = \sum_{y=0}^n g(y) \binom{n}{y} \theta^y (1-\theta)^{n-y} = 0,\quad \forall\,\theta\in(0,1)
$$
이 식은 $\theta$에 대한 다항식(정확히는 $n$차 다항식)이다. 이 다항식이 $(0,1)$의 모든 $\theta$에 대해 0이므로, 계수(즉, $g(y)\binom{n}{y}$)가 모두 0이어야 한다. 즉,
$$
g(y)\binom{n}{y} = 0,\quad y=0,1,\dots,n
$$
따라서 $g(y)=0$ ($y=0,1,\dots,n$)이고, $Y$가 $\theta$에 대한 완비충분통계량이다. 

정리 8.3.1에 따르면, $Y$의 함수로 표현되는 $\theta$의 불편추정량은 곧 $\theta$의 UMVUE가 된다. 실제로 $E_\theta(Y/n) = \theta$이므로, $\hat\theta = Y/n = \frac{1}{n}\sum_{i=1}^n X_i$는 $\theta$의 UMVUE임이 엄밀히 성립한다.

#### 예 8.3.2 포아송 분포
$X_i \sim \mathrm{Poisson}(\theta),\ \theta>0$

- 충분통계량: $Y = \sum X_i$
- $Y \sim \mathrm{Poisson}(n\theta)$이므로, $Y$의 확률질량함수는
    $$
    P_\theta(Y = y) = \frac{(n\theta)^y}{y!} e^{-n\theta},\quad y=0,1,2,\dots
    $$
- $Y$가 $\theta$에 대한 완비통계량임을 보이려면, 임의의 함수 $g$에 대해
    $$
    E_\theta[g(Y)] = 0\quad \forall\,\theta>0
    $$
    이면 $P_\theta(g(Y)=0)=1$임을 보여야 한다.
- $E_\theta[g(Y)]$를 전개하면
    $$
    E_\theta[g(Y)] = \sum_{y=0}^\infty g(y) \frac{(n\theta)^y}{y!} e^{-n\theta} = 0,\quad \forall\,\theta>0
    $$
    이 식은 $\theta$에 대한 멱급수이므로, 모든 $\theta>0$에서 0이 되려면 각 계수 $g(y)/y!$가 모두 0이어야 한다. 즉,
    $$
    g(y) = 0,\quad y=0,1,2,\dots
    $$
    따라서 $Y$는 $\theta$에 대한 완비충분통계량이다.
- $\theta$의 불편추정량은 $E_\theta(Y/n) = \theta$이므로, $\hat\theta = Y/n = \frac{1}{n}\sum_{i=1}^n X_i$가 $\theta$의 UMVUE이다.

#### 예 8.3.3 지수분포
$X_i \sim \mathrm{Exp}(\theta),\ \theta>0$

- 충분통계량: $Y = \sum X_i$
- $Y \sim \mathrm{Gamma}(n, \theta)$이므로, $Y$의 확률밀도함수는
    $$
    f_Y(y;\theta) = \frac{1}{\Gamma(n)\theta^n} y^{n-1} e^{-y/\theta},\quad y>0
    $$
- $Y$가 $\theta$에 대한 완비통계량임을 보이려면, 임의의 함수 $g$에 대해
    $$
    E_\theta[g(Y)] = 0\quad \forall\,\theta>0
    $$
    이면 $P_\theta(g(Y)=0)=1$임을 보여야 한다.

- $E_\theta[g(Y)]$를 전개하면
    $$
    E_\theta[g(Y)] = \int_0^\infty g(y) \frac{1}{\Gamma(n)\theta^n} y^{n-1} e^{-y/\theta} dy = 0,\quad \forall\,\theta>0
    $$
- 변수변환 $t = y/\theta$를 하면 $y = t\theta$, $dy = \theta dt$이므로
    $$
    E_\theta[g(Y)] = \int_0^\infty g(t\theta) \frac{1}{\Gamma(n)} t^{n-1} e^{-t} dt = 0,\quad \forall\,\theta>0
    $$
- 이 식이 모든 $\theta>0$에서 0이 되려면, $g(y)$가 거의 모든 $y>0$에 대해 0이어야 한다(라플라스 변환의 유일성). 즉, $g(Y)=0$이 거의 확실하게 성립한다.
  - 라플라스 변환 $L_T(s)$:
    $$
    L_T(s) := E[e^{-sT}] = \int_0^\infty e^{-st} f_T(t)\,dt,\quad s \ge 0
    $$
   - 라플라스 변환은 함수 f와 일대일 대응이라 알려져 있다.
- 따라서 $Y$는 $\theta$에 대한 완비충분통계량이다.
- $E_\theta(Y/n) = \theta$이므로, 표본평균 $\bar X = Y/n$은 $\theta$의 불편추정량이자 UMVUE이다.

### 정리 8.3.2 다중모수 지수족의 완비충분통계량
확률밀도함수가
$$
f(x;\eta) = \exp\left\{ \sum_{j=1}^k \eta_j T_j(x) - A(\eta) + S(x) \right\}
$$
꼴이고, 조건 1,2,3을 만족하면  
1. 분포의 토대$\{x:f(x;\eta)>0\}$가 모수에 의존하지 않음  
2. 모수공간$N$이 $k$차원 공간 $R^k$의 열린구간 다면체 $(a_1,b_1)\times \dots \times (a_k,b_k)$를 포함
3. $(T_1,\dots,T_k)$의 비자명한 선형결합이 상수가 아님  
3. $a_1 T_1(x) + \dots + a_k T_k(x)$가 상수 함수가 되려면 $a_1 = \dots = a_k = 0$이어야 한다.

이면
$$
\sum_{i=1}^n T(X_i) = \left(\sum_{i=1}^n T_1(X_i),\; \dots,\; \sum_{i=1}^n T_k(X_i)\right)^T
$$
는 $\eta$에 관한 **완비충분통계량**이다.

이 정리를 적용할 때 지수족의 표현에 사용되는 모수$\eta$는 
$\eta = g(\theta) = (g_1(\theta), \dots, g_k(\theta))^T, \theta \in \Omega$
와 같은 일대일변환을 통해 정의되는 모수다. 즉 이 정리의 확률밀도함수가 
$$
pdf(x;\theta) = \exp\left\{ \sum_{j=1}^k g_j(\theta) T_j(x) - A(g(\theta)) + S(x) \right\}
$$
의 꼴로 모수 $\theta \in \Omega$를 이용하여 주어진다. 따라서 조건2는 
$$\{(g_1(\theta), \dots, g_k(\theta))^T : \theta \in \Omega \}\supseteq(a_1,b_1)\times \dots \times(a_k,b_k)$$
를 만족시키는 열린구간 다면체 $(a_1,b_1)\times \dots \times(a_k,b_k)$가 존재하는 것을 뜻한다.

(증명은 라플라스 변환의 유일성을 이용, 책의 범위를 넘으므로 생략)

#### 예 8.3.4 정규모집단에서 랜덤표본으로부터의 완비충분통계량
$X_i \sim N(\mu, \sigma^2)$에서 완비충분통계량과 UMVUE를 구하라.

**1. 충분통계량 찾기**  
정규분포의 결합확률밀도함수는
$$
\prod_{i=1}^n f(x_i;\mu,\sigma^2) = (2\pi\sigma^2)^{-n/2} \exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2\right)
$$
이를 분해정리(정리 8.2.1) 형태로 쓰면,
$$
= (2\pi\sigma^2)^{-n/2} \exp\left(-\frac{1}{2\sigma^2}\left(\sum x_i^2 - 2\mu\sum x_i + n\mu^2\right)\right)
$$
즉, $\sum x_i$와 $\sum x_i^2$만 알면 결합확률밀도함수를 모두 표현할 수 있으므로,  
$Y = (\sum X_i,\, \sum X_i^2)$는 $(\mu, \sigma^2)$에 대한 충분통계량이다.

**2. 완비성 확인**  
정규분포는 지수족이며, $T_1(x) = x$, $T_2(x) = x^2$로 표현된다.  
정리 8.3.2의 조건(모수공간이 열린집합 포함, 비자명한 선형결합이 상수 아님 등)을 모두 만족하므로,  
$Y = (\sum X_i,\, \sum X_i^2)$는 $(\mu, \sigma^2)$에 대한 완비충분통계량이다.

**3. UMVUE 구하기**  
- $\mu$의 불편추정량: 표본평균 $\bar X = \frac{1}{n}\sum X_i$  
    $E[\bar X] = \mu$이므로 $\bar X$는 $\mu$의 UMVUE이다.
- $\sigma^2$의 불편추정량: $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar X)^2$  
    $E[S^2] = \sigma^2$이므로 $S^2$는 $\sigma^2$의 UMVUE이다.

#### 예 8.3.5 제한된 모수공간을 갖는 정규모집단의 경우
$X_1,\dots,X_n \sim N(\mu, \mu^2),\ \mu>0$

**1. 지수족 표현**  
$$
f(x;\mu) = \frac{1}{\sqrt{2\pi}\mu} \exp\left(-\frac{(x-\mu)^2}{2\mu^2}\right) \\
= \exp\left(-\frac{1}{2}\log(2\pi) - \log\mu - \frac{(x-\mu)^2}{2\mu^2}\right) \\
= \exp\left(-\frac{1}{2}\log(2\pi) - \log\mu - \frac{x^2 - 2\mu x + \mu^2}{2\mu^2}\right) \\
= \exp\left(-\frac{1}{2}\log(2\pi) - \log\mu - \frac{x^2}{2\mu^2} + \frac{x}{\mu} - \frac{1}{2}\right)
$$
즉, $T_1(x) = x$, $T_2(x) = x^2$, $g_1(\mu) = 1/\mu$, $g_2(\mu) = -1/(2\mu^2)$

**2. 모수공간 확인**  
모수공간 $\mathcal N = \{ (\eta_1, \eta_2): \eta_1 = 1/\mu > 0,\ \eta_2 = -1/(2\mu^2) \}$은  
$\eta_2 = -\frac12 \eta_1^2$로 $\eta_1>0$에서 1차원 곡선만을 이룸(2차원 열린집합 아님).  
따라서 정리 8.3.2(완비성 보장)를 적용할 수 없다.

**3. 실제로 완비성 불만족 확인**  
$Y_1 = \sum X_i$, $Y_2 = \sum X_i^2$에 대해
$$
\mathbb E_\mu\left[\frac{1}{2n}Y_2 - \frac{1}{n(n+1)}Y_1^2\right] = 0\quad(\forall\mu>0)
$$
즉, $g(Y_1, Y_2) = \frac{1}{2n}Y_2 - \frac{1}{n(n+1)}Y_1^2$는 평균이 0이지만, $g$가 항상 0은 아니므로 완비성 불만족.  
따라서 $Y = (\sum X_i,\, \sum X_i^2)$는 $\mu$에 대한 완비통계량이 아니고, $\bar X$는 $\mu$의 UMVUE가 아니다.

#### 예 8.3.6 균등분포 $U[0,\theta]$에서의 완비충분통계량
$X_1,\dots,X_n \sim U[0,\theta],\ \theta>0$

**1. 충분통계량**  
결합밀도는
$$
f(x_1,\dots,x_n;\theta) = \theta^{-n} \mathbf{1}_{0 < x_{(1)} < x_{(n)} < \theta}
$$
즉, $Y = X_{(n)} = \max X_i$만 알면 $\theta$에 대한 정보가 모두 보존됨.

**2. 완비성 확인**  
$Y$의 밀도는
$$
f_Y(y;\theta) = \frac{n}{\theta^n} y^{n-1} \mathbf{1}_{[0,\theta]}(y)
$$
임의의 함수 $g$에 대해
$$
E_\theta[g(Y)] = \int_0^\theta g(y) \frac{n}{\theta^n} y^{n-1} dy = 0\quad\forall\theta>0
$$
이 식이 모든 $\theta>0$에서 0이 되려면 $g(y)=0$ (거의 모든 $y$에 대해)이어야 하므로 완비성 성립.

**3. UMVUE 구하기**  
$$
E_\theta[Y] = \int_0^\theta y \cdot \frac{n}{\theta^n} y^{n-1} dy = \frac{n}{\theta^n} \int_0^\theta y^n dy \\
= \frac{n}{\theta^n}\left[ \frac{y^{n+1}}{n+1} \right]_0^\theta
= \frac{n}{n+1} \theta
$$
이므로, $\frac{n+1}{n}Y$는 $\theta$의 불편추정량이자 UMVUE.
- 완비충분통계량: $Y = \max X_i$
- $\hat\theta^{UMVUE} = \frac{n+1}{n} \max X_i$

#### 예 8.3.7 베르누이 분포에서 모분산의 불편추정
$X_1,\dots,X_n \sim \mathrm{Bernoulli}(\theta),\ 0<\theta<1$

**1. 완비충분통계량**  
$Y = \sum_{i=1}^n X_i$ (예 8.3.1에서 이미 완비충분통계량임을 확인)

**2. 모분산의 불편추정량**  
모분산: $\eta = \theta(1-\theta)$

불편추정량 후보로 $X_1(1-X_2)$를 생각할 수 있다.  
$$
E[X_1(1-X_2)] = E[X_1] - E[X_1 X_2] = \theta - \theta^2 = \theta(1-\theta)
$$
즉, $X_1(1-X_2)$는 $\theta(1-\theta)$의 불편추정량이다.

**3. Rao–Blackwell 개선**  
Rao–Blackwell 정리에 따라, 완비충분통계량 $Y$에 대한 조건부기대값을 취하면 UMVUE를 얻을 수 있다.
$$
\hat\eta^{RB}(Y) = E[X_1(1-X_2)\mid Y]
$$

$Y$가 주어졌을 때, $X_1$과 $X_2$의 조건부분포는 교환가능하다. $Y=y$일 때, $n$개 중 $y$개가 1이고 $n-y$개가 0이다.

- $X_1$이 1일 확률: $P(X_1=1\mid Y=y) = \frac{y}{n}$
- $X_2$가 0일 확률 (단, $X_1=1$로 이미 1개 사용): $P(X_2=0\mid X_1=1, Y=y) = \frac{n-y}{n-1}$

따라서,
$$
E[X_1(1-X_2)\mid Y=y] = P(X_1=1\mid Y=y) \cdot P(X_2=0\mid X_1=1, Y=y)
= \frac{y}{n} \cdot \frac{n-y}{n-1}
$$

이를 $\hat\theta = Y/n$으로 쓰면,
$$
E[X_1(1-X_2)\mid Y] = \frac{n}{n-1}\hat\theta(1-\hat\theta)
$$

$\therefore \boxed{\hat\eta^{UMVUE} = \frac{n}{n-1}\hat\theta(1-\hat\theta)}$  

#### 예 8.3.8 포아송 분포에서 모평균의 함수의 불편추정
$X_1,\dots,X_n \sim \mathrm{Poisson}(\theta),\ \theta>0$

**1. 완비충분통계량**  
포아송 분포의 완비충분통계량은 $Y = \sum_{i=1}^n X_i$이다.  
$Y \sim \mathrm{Poisson}(n\theta)$

**2. 추정 대상**  
$\eta = e^{-2\theta}$

**3. UMVUE 구하기**  
$Y$의 확률질량함수는 $P(Y=y) = \frac{(n\theta)^y}{y!}e^{-n\theta}$이다.  

여기서 $e^{-2\theta}$의 불편추정량을 찾기 위해, $Y$의 함수 $(1-2/n)^Y$를 고려한다. 왜냐하면, 포아송 분포의 적률생성함수(MGF)는 $M_Y(t) = E[e^{tY}] = \exp(n\theta(e^t-1))$이므로, $E[a^Y] = \exp(n\theta(a-1))$ 형태로 쉽게 계산할 수 있다. $a = 1-2/n$을 대입하면,
$$
E[(1-2/n)^Y] = \exp(n\theta((1-2/n)-1)) = \exp(-2\theta)
$$
즉, $(1-2/n)^Y$는 $e^{-2\theta}$의 불편추정량이 된다.

- $n=1$이면 $(1-2/1)^{X_1} = (-1)^{X_1}$로, $X_1$이 홀수면 $-1$, 짝수면 $1$이 된다.
  - 즉, 맹목적인 불편추정은 특이한 결과를 가져오기도 한다.

#### 예 8.3.9 정규분포에서 신뢰도의 불편추정
$X_1,\dots,X_n \sim N(\theta,1),\ n\ge2$

**1. 완비충분통계량**  
정규분포에서 평균 $\theta$의 완비충분통계량은 표본평균 $\bar X$이다.

**2. 추정 대상**  
신뢰도: $\eta = P_\theta(X_1 > a)$

**3. 불편추정량 및 Rao–Blackwell 개선**  
- $X_1$만 이용한 불편추정량은 $\mathbf{1}_{(a,\infty)}(X_1)$이다.  
    $E[\mathbf{1}_{(a,\infty)}(X_1)] = P_\theta(X_1 > a) = 1 - \Phi(a-\theta)$

- Rao–Blackwell 정리에 따라, 완비충분통계량 $\bar X$에 대한 조건부기대값을 취하면 UMVUE가 된다:
        $$
        \hat\eta^{UMVUE} = E[\mathbf{1}_{(a,\infty)}(X_1) \mid \bar X]
        $$
- $X_1$과 $\bar X$의 결합분포에서, $X_1 - \bar X$와 $\bar X$는 서로 독립이다.  
    - 왜냐하면 $X_1 - \bar X$는 $X_1$과 $X_2,\dots,X_n$의 선형결합이고, $\bar X$는 $X_1,\dots,X_n$의 평균이므로, 정규분포의 성질(공분산 계산)로부터 두 변수는 공분산이 0이고, 정규분포에서는 공분산 0이면 독립이다.
    - 즉, $X_1 - \bar X$와 $\bar X$는 독립인 정규분포이다.
- 따라서 $X_1 = \bar X + (X_1 - \bar X)$이고, $\bar X$가 주어졌을 때 $X_1$의 조건부분포는  
    $X_1 \mid \bar X \sim N\left(\bar X,\,\frac{n-1}{n}\right)$  (즉, $\bar X$를 중심으로 분산 $\frac{n-1}{n}$인 정규분포)
- 이제 $P_\theta(X_1 > a \mid \bar X = y)$를 계산하면,
    $$
    P(X_1 > a \mid \bar X = y)
    = P(X_1 - \bar X > a - y \mid \bar X = y)
    $$
    $X_1 - \bar X \mid \bar X = y \sim N(0,\,\frac{n-1}{n})$이므로,
    $$
    = P\left(Z > \sqrt{\frac{n}{n-1}}(a - y)\right)
    $$
    여기서 $Z \sim N(0,1)$.
- 즉,
    $$
    P(X_1 > a \mid \bar X = y) = 1 - \Phi\left(\sqrt{\frac{n}{n-1}}(a - y)\right)
    $$
- 따라서 Rao–Blackwell 개선 추정량은
    $$
    \hat\eta^{UMVUE}
    = 1 - \Phi\left(\sqrt{\frac{n}{n-1}}(a - \bar X)\right)
    $$

### 정리 8.3.3 완비충분통계량과 보조통계량의 독립성
- 통계량 $Z = v(X_1,\dots,X_n)$의 분포가 $\theta$에 의존하지 않으면 $Z$는 **보조통계량(ancillary statistic)** 이다.
  - 모수 $\theta$에 의존하지 않는 분포를 갖는 통계량을 $\theta$에 관한 보조통계량이라 한다
- $Y$가 $\theta$에 대한 완비충분통계량이면 $Z \perp Y$ (독립).
- 의의: 완비충분통계량이 주어지면, 그와 독립인 보조통계량을 이용해 조건부기대값 계산이 단순해진다.

#### 증명
$Z = v(X_1, \dots, X_n)$의 분포가 $\theta$에 의존하지 않으므로, 임의의 집합 $B$에 대해
$$
P_\theta(Z \in B) = P(Z \in B), \quad \forall \theta \in \Omega
$$

또한, $Y = u(X_1, \dots, X_n)$가 $\theta$에 대한 완비충분통계량이라고 하자. 조건부확률을 생각하면,
$$
P_\theta(Z \in B \mid Y) = P(v(X_1, \dots, X_n) \in B \mid Y)
$$
이 값이 $\theta$에 의존할 수도 있지만, 아래와 같이 보조통계량의 정의와 완비성으로부터 독립임을 보일 수 있다.

조건부기대값의 성질로,
$$
E_\theta[P(Z \in B \mid Y)] = E_\theta[E[I_B(Z) \mid Y]] = E_\theta[I_B(Z)] = P_\theta(Z \in B) = P(Z \in B)
$$
즉, $g(Y) := P(Z \in B \mid Y) - P(Z \in B)$로 두면,
$$
E_\theta[g(Y)] = 0, \quad \forall \theta \in \Omega
$$
$Y$가 완비통계량이므로, $g(Y) = 0$이 거의 확실하게 성립한다. 즉,
$$
P(Z \in B \mid Y) = P(Z \in B)
$$
따라서 $Z$와 $Y$는 독립이다.

#### 예 8.3.10 지수분포에서 신뢰도의 불편추정
- **모형:** $X_1,\dots,X_n \sim \mathrm{Exp}(\theta),\ \theta>0$
- **완비충분통계량:** $Y = \sum_{i=1}^n X_i$
- **추정 대상:a에서의 신뢰도** $\eta = P_\theta(X_1 > a) = e^{-a/\theta}$

1. **불편추정량 후보:**  
    $X_1$만 이용하면, $\mathbf{1}_{(a,\infty)}(X_1)$이 $\eta$의 불편추정량이다.  
    $E_\theta[\mathbf{1}_{(a,\infty)}(X_1)] = P_\theta(X_1 > a) = e^{-a/\theta}$
2. **보조통계량 활용:**  
    $Y = \sum X_i$는 완비충분통계량이고, $\frac{X_1}{Y}$는 $\theta$에 무관한 보조통계량($\mathrm{Beta}(1, n-1)$ 분포)이며 $Y$와 독립이다 (정리8.3.3).
3. **UMVUE 계산:**  
    $X_1 = uY$로 두면, $P(X_1 > a \mid Y = y) = P(u > a/y) = \int_{a/y}^1 (n-1)(1-u)^{n-2} du = (1 - a/y)^{n-1}$ ($a < y$일 때).
    - 여기서 $X_1 = uY$로 두었으므로, $X_1 > a$는 $uY > a$와 동치이고, $Y = y$이므로 $u > a/y$가 된다. 즉, $P(X_1 > a \mid Y = y) = P(u > a/y)$로 쓸 수 있다.
    - $u = X_1/Y$는 $Y$와 독립이며, $u \sim \mathrm{Beta}(1, n-1)$ 분포를 따른다. 따라서 $P(u > a/y)$는 $u$의 누적분포함수를 이용해 계산할 수 있다.

    따라서
    $$
    \hat\eta^{UMVUE} = \left(1 - \frac{a}{Y}\right)^{n-1} \mathbf{1}_{(a, \infty)}(Y)
    $$
    또는 $Y = n\bar X$이므로,
    $$
    \hat\eta^{UMVUE} = \left(1 - \frac{a}{n\bar X}\right)^{n-1} \mathbf{1}_{(a/n, \infty)}(\bar X)
    $$
4. **점근적 거동:**  
    $n \to \infty$에서 $\left(1 - \frac{a}{n\bar X}\right)^{n-1} \to e^{-a/\bar X}$ (MLE와 동일).

**정리:**  
- UMVUE:  
  $$
  \boxed{
     \hat\eta^{UMVUE} = \left(1 - \frac{a}{n\bar X}\right)^{n-1} \mathbf{1}_{(a/n, \infty)}(\bar X)
  }
  $$
- $n \to \infty$에서 $e^{-a/\bar X}$로 수렴 (MLE와 동일)


## 8.4 추정량의 점근적 비교 *(Asymptotic Comparison of Estimators)*
- 표본크기 $n$이 커질 때 추정량 $\hat\eta_n$의 **극한분포(asymptotic distribution)** 로 성능을 비교한다.
- 보통 $\sqrt{n}(\hat\eta_n - \eta(\theta)) \xrightarrow{d} N(0,\,\sigma^2(\theta))$ 꼴이면, $\sigma^2(\theta)$가 작을수록 점근적으로 효율적이다 (추정 오차가 작은것을 뜻한다).
- 평균제곱오차는 $MSE_\theta(\hat\eta_n) = E_\theta[(\hat\eta_n - \eta(\theta))^2] = \mathrm{Var}_\theta(\hat\eta_n) + \mathrm{Bias}_\theta(\hat\eta_n)^2$로 분해된다.
  - bias: 편의 또는 바이어스
  - bias term은 $n^{-2}$의 속도로 0에 가까워지고 variance term은 $n^{-1}$의 속도로 0에 가까워지므로 표본분포가 커지면 극한분포의 분산을 비교기준으로 한다
- 그래서, 극한분포의 분산을 이용한 추정량의 비교에 대해 알아보자!

#### 예 8.4.1 로지스틱분포에서의 추정량 비교
- 모형: $L(\theta,1)$, $-\infty < \theta < \infty$ (위치모수)
- 후보 추정량:
    - 표본중앙값: $\hat\theta_n^{med} = X_{(\lfloor(n+1)/2\rfloor)}$
    - 표본평균: $\hat\theta_n^{mean} = \bar X$
    - 최대가능도추정량(MLE): $\hat\theta_n^{MLE}$
- 점근분포:
  - 예5.3.8
    $$
    \sqrt{n}(\hat\theta_n^{med} - \theta) \xrightarrow{d} N(0, 4)
    $$
  - 중심극한정리, 예5.4.3
    $$
    \sqrt{n}(\hat\theta_n^{mean} - \theta) \xrightarrow{d} N\left(0, \frac{\pi^2}{3}\right)
    $$
  - 예6.4.4
    $$
    \sqrt{n}(\hat\theta_n^{MLE} - \theta) \xrightarrow{d} N(0, 3)
    $$
- 점근분산이 작은 순(효율이 좋은 순): $\hat\theta_n^{MLE}\ (3) < \hat\theta_n^{mean}\ \left(\frac{\pi^2}{3}\right) < \hat\theta_n^{med}\ (4)$

### 점근상대효율성(ARE, asymptotic relative efficiency)
- 두 추정량 $\hat\theta_n^{1}, \hat\theta_n^{2}$에 대해
    $$
    \sqrt{n}(\hat\theta_n^{i} - \theta) \xrightarrow{d} N(0, \sigma_i^2(\theta)),\quad i=1,2
    $$
    이면,
    $$
    ARE(\hat\theta_n^1, \hat\theta_n^2) = \frac{\sigma_1^{-2}(\theta)}{\sigma_2^{-2}(\theta)}
    $$
- $ARE > 1$이면 1번이 더 효율적(점근분산이 더 작음)이라는 뜻.
- 점근상대효울성이 커진다는 것은 표본크기가 커지면서 추정의 정밀도가 상대적으로 높아지는 것
- 더 적은 샘플크기로도 요구되는 추정오차한계를 만족시킬 수 있음

#### 예 8.4.2 로지스틱분포에서 추정오차한계와 점근상대효율성
- 표본중앙값으로 95% 점근신뢰구간:
    $$
    \theta \in \left[\hat\theta_n^{med} - 1.96\sqrt{\frac{4}{n}},\ \hat\theta_n^{med} + 1.96\sqrt{\frac{4}{n}}\right]
    $$
    - $1.96\sqrt{\frac{4}{n}}$: 표본중앙값의 95%점근추정오차한계
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
- 즉, 같은 오차한계를 달성하려면 **표본중앙값이 표본평균보다 대략 1.2배 더 많은 표본**을 필요로 한다.

#### 예 8.4.3 위치모수 모형에서 중앙값/평균/MLE 비교
예 8.4.1 처럼 모집단분포가 연속형이고, 확률밀도함수가 $f(x)=f(x-\theta)$ 즉 대칭이다.  
랜덤표본 $X_1, \dots, X_n$을 이용한 $\theta$추정량으로서 표본중앙값, 표본평균, 최대가능도추정량을 비교해본다.
- 점근분산
    - 표본중앙값: 예5.3.7결과: $\sqrt{n}(\hat\theta_n^{med} - \theta) \xrightarrow{d} N\left(0,\, (4f^2(0))^{-1}\right)$
    - 표본평균(2차모멘트 존재): $\sqrt{n}(\hat\theta_n^{mean} - \theta) \xrightarrow{d} N(0, \sigma_f^2)$, $\sigma_f^2 = \int_{-\infty}^{\infty} x^2 f(x)\,dx$
    - MLE(정규성 조건): 정리6.4.4조건 만족 시 $\sqrt{n}(\hat\theta_n^{MLE} - \theta) \xrightarrow{d} N(0, I_f^{-1})$, $I_f = \int_{-\infty}^{\infty} \left(\frac{f'(x)}{f(x)}\right)^2 f(x)\,dx$

**표 8.4.1 (점근분산 값 요약)**  
| 분포             | $N(\theta,1)$ | $L(\theta,1)$ | $DE(\theta,1)$ |
|------------------|:-------------:|:-------------:|:--------------:|
| $(4f^2(0))^{-1}$ | $\pi/2$       | $4$           | $1$            |
| $\sigma_f^2$     | $1$           | $\pi^2/3$     | $2$            |
| $I_f^{-1}$       | $1$           | $3$           | $1$            |

**표 8.4.2 (ARE 요약)**  
| 분포            | $N(\theta,1)$ | $L(\theta,1)$ | $DE(\theta,1)$ |
|-----------------|:-------------:|:-------------:|:--------------:|
| $ARE(\hat\theta_n^{mean}, \hat\theta_n^{med})$ | $\pi/2$        | $12/\pi^2$    | $1/2$           |
| $ARE(\hat\theta_n^{mean}, \hat\theta_n^{MLE})$ | $1$            | $9/\pi^2$     | $1/2$           |
| $ARE(\hat\theta_n^{med}, \hat\theta_n^{MLE})$  | $2/\pi$        | $3/4$         | $1$             |

#### 예 8.4.4 베타분포 $\mathrm{Beta}(\alpha,1)$에서 추정량 비교
- 모형: $X_i \sim \mathrm{Beta}(\alpha,1)$, $\alpha>0$, 밀도 $f(x;\alpha)=\alpha x^{\alpha-1}I_{(0,1)}(x)$
- 모평균: $m_1 = \frac{\alpha}{\alpha+1}$

**적률이용추정량(MME)**  
$$
\hat\alpha_n^{\mathrm{MME}} = \frac{\bar X}{1-\bar X} \\
$$

모평균 $m_1 = \frac{\alpha}{\alpha+1}$의 적률추정량 $\bar X$는 중심극한정리에 의해
$$
\sqrt{n}(\bar X - m_1) \xrightarrow{d} N(0,\,\sigma^2)
$$
여기서 $\sigma^2 = \mathrm{Var}(X_1) = E[X_1^2] - (E[X_1])^2$이다.

$\mathrm{Beta}(\alpha,1)$에서
- $E[X_1] = m_1 = \frac{\alpha}{\alpha+1}$
- $E[X_1^2] = \frac{\alpha}{\alpha+2}$

따라서
$$
\sigma^2 = \frac{\alpha}{\alpha+2} - \left(\frac{\alpha}{\alpha+1}\right)^2
$$

적률이용추정량은 $g(\bar X)$ 꼴로, $g(m_1) = \frac{m_1}{1-m_1} = \alpha$이다.

델타 방법에 의해
$$
\sqrt{n}(g(\bar X) - g(m_1)) \xrightarrow{d} N(0,\, [g'(m_1)]^2 \sigma^2)
$$
여기서
$$
g'(x) = \frac{1}{(1-x)^2},\quad g'(m_1) = \frac{1}{(1-m_1)^2} = (\alpha+1)^2
$$

따라서
$$
\sqrt{n}(\hat\alpha_n^{\mathrm{MME}}-\alpha) \xrightarrow{d} N\left(0,\, (g'(m_1))^2 \sigma^2\right)
= N\left(0,\, (\alpha+1)^4 \left[\frac{\alpha}{\alpha+2} - \left(\frac{\alpha}{\alpha+1}\right)^2\right]\right) \\
= N\left(0,\, \frac{\alpha(\alpha+1)^2}{\alpha+2}\right)
$$

**최대가능도추정량(MLE)**  
가능도방정식: $\sum_{i=1}^n \log X_i + \frac{n}{\alpha} = 0$
$$
\therefore \hat\alpha_n^{\mathrm{MLE}} = \frac{1}{-\overline{\log X}}
$$

$$
\sqrt{n}(\hat\alpha_n^{\mathrm{MLE}}-\alpha) \xrightarrow{d} N(0,\,\alpha^2)
$$

**점근상대효율성**  
$$
ARE(\hat\alpha_n^{\mathrm{MME}},\,\hat\alpha_n^{\mathrm{MLE}})
= \frac{\alpha(\alpha+2)}{(\alpha+1)^2}
$$

따라서 MLE가 점근적으로 더 효율적이다.

### 정리 8.4.1 정보량 부등식 *(information inequality, Cramér–Rao 유형)*
위의 예들에서 점근상대효율성이 가장 큰 추정량은 최대가능도추정량이었다. 이런 최대가능도추정량의 점근적 효율성은 아래 정보량 부등식으로 설명할 수도 있다.
- (MLE 점근정규성 조건들과 유사한 정칙성) 하에서, 실수값 모수 $\eta=\eta(\theta)$의 추정량 $\hat\eta_n$에 대해
$$
\mathrm{Var}_\theta(\hat\eta_n) \ge
\left(\frac{\partial}{\partial\theta}E_\theta(\hat\eta_n)\right)^{\!t}
[nI(\theta)]^{-1}
\left(\frac{\partial}{\partial\theta}E_\theta(\hat\eta_n)\right),
\quad \forall\theta
$$

- 다차원 모수의 경우도 유사하게 아래 부등식이 성립
$$
c^t\left(\mathrm{Var}_\theta(\hat\eta_n)-
\left(\frac{\partial}{\partial\theta}E_\theta(\hat\eta_n)\right)^{\!t}
[nI(\theta)]^{-1}
\left(\frac{\partial}{\partial\theta}E_\theta(\hat\eta_n)\right)\right)c \ge 0
$$

#### 증명
점수함수(score function)와 코시–슈바르츠 부등식을 이용한다.
정보량 부등식의 증명에는 다음과 같은 정칙성 조건이 필요하다:
- **(R1) 지지집합(support)이 $\theta$에 의존하지 않음:**  
    $f(x;\theta)$의 정의역(지지집합)이 $\theta$에 따라 변하지 않는다.

- **(R2) 미분과 적분의 교환 가능:**  
    $\frac{\partial}{\partial\theta} \log f(x;\theta)$가 존재하고,  
    $$
    \frac{\partial}{\partial\theta} \int g(x) f(x;\theta)\,dx = \int g(x) \frac{\partial}{\partial\theta} f(x;\theta)\,dx
    $$
    가 성립한다.

- **(R3) 점수함수의 평균이 0:**  
    $$
    E_\theta[\dot l_n(\theta)] = 0
    $$

이제 증명을 정리하면 다음과 같다:
1. **점수함수 정의**  
        $$
        \dot l_n(\theta) = \frac{\partial}{\partial\theta} \sum_{i=1}^n \log f(X_i;\theta)
        $$
        (다변수의 경우 $\theta=(\theta_1,\dots,\theta_k)^t$는 $k$차원 벡터, $\dot l_n(\theta)$도 $k$차원 벡터)

2. **코시–슈바르츠 부등식 적용**  
        임의의 추정량 $\hat\eta_n$에 대해,
        $$
        \operatorname{Var}_\theta(\hat\eta_n)\,\operatorname{Var}_\theta(\dot l_n(\theta)) \ge \operatorname{Cov}_\theta(\hat\eta_n,\,\dot l_n(\theta))^2
        $$
        (다변수의 경우, 임의의 벡터 $a$, $b$에 대해  
        $$
        a^t \mathrm{Var}_\theta(\hat\eta_n) a \cdot b^t \mathrm{Var}_\theta(\dot l_n(\theta)) b \ge \left(a^t \mathrm{Cov}_\theta(\hat\eta_n, \dot l_n(\theta)) b\right)^2
        $$)

3. **공분산 계산**  
        $$
        \operatorname{Cov}_\theta(\hat\eta_n,\,\dot l_n(\theta))
        = E_\theta\left[(\hat\eta_n - E_\theta[\hat\eta_n])\,\dot l_n(\theta)\right]
        $$
        정칙성 조건(R2)에 의해,
        $$
        E_\theta\left[\hat\eta_n\,\dot l_n(\theta)\right]
        = \frac{\partial}{\partial\theta} E_\theta[\hat\eta_n]
        $$
        따라서,
        $$
        \operatorname{Cov}_\theta(\hat\eta_n,\,\dot l_n(\theta))
        = \frac{\partial}{\partial\theta} E_\theta[\hat\eta_n]
        $$
        (다변수의 경우, $\mathrm{Cov}_\theta(\hat\eta_n, \dot l_n(\theta)) = \frac{\partial}{\partial\theta} E_\theta[\hat\eta_n]$는 $m\times k$ 행렬)

4. **정보량 대입**  
        점수함수의 분산은 정보량 $I_n(\theta)$와 같다:
        $$
        \operatorname{Var}_\theta(\dot l_n(\theta)) = nI(\theta)
        $$
        ($I(\theta)$는 $k\times k$ 양정정부호 행렬)

5. **최종 부등식**  
        위 결과들을 코시–슈바르츠 부등식에 대입하면,
        $$
        \operatorname{Var}_\theta(\hat\eta_n) \ge
        \frac{\left(\frac{\partial}{\partial\theta} E_\theta[\hat\eta_n]\right)^2}{nI(\theta)}
        $$
        (다변수/행렬의 경우,
        $$
        \mathrm{Var}_\theta(\hat\eta_n) \succeq
        \left(\frac{\partial}{\partial\theta} E_\theta[\hat\eta_n]\right)
        [nI(\theta)]^{-1}
        \left(\frac{\partial}{\partial\theta} E_\theta[\hat\eta_n]\right)^t
        $$
        여기서 $\succeq$는 정부호(positive semidefinite) 관계)

즉, 단변수/다변수 모두 정보량 부등식이 성립한다.

### 정리 8.4.2 불편추정량에 대한 정보량 부등식
- $\theta=(\theta_1,\dots,\theta_k)^t$의 **불편추정량** $\hat\theta_n^{\mathrm{UE}}$에 대해
$$
c^t\mathrm{Var}_\theta(\hat\theta_n^{\mathrm{UE}})c \ge \frac{1}{n}c^tI^{-1}(\theta)c,
\quad \forall c,\ \forall\theta
$$

- 행렬로 쓰면 $\mathrm{Var}_\theta(\hat\theta_n^{\mathrm{UE}})-I^{-1}(\theta)/n$이 음이 아닌 정부호(positive semidefinite)임을 의미

#### 증명
$\hat\theta_n^{\mathrm{UE}}$가 $\theta$의 불편추정량이므로 $E_\theta[\hat\theta_n^{\mathrm{UE}}]=\theta$이다.  
정리 8.4.1(정보량 부등식)에 $\eta(\theta)=\theta$를 대입하면
$$
\mathrm{Var}_\theta(\hat\theta_n^{\mathrm{UE}}) \ge [nI(\theta)]^{-1}
$$
(다변수의 경우 $c^t\mathrm{Var}_\theta(\hat\theta_n^{\mathrm{UE}})c \ge \frac{1}{n}c^tI^{-1}(\theta)c$).  
이는 $\frac{\partial}{\partial\theta}E_\theta[\hat\theta_n^{\mathrm{UE}}]=I$ (항등행렬)이므로 바로 성립한다.

#### 예 8.4.5 $\mathrm{Beta}(\alpha,1)$에서의 불편추정과 정보량 부등식
- $\mathrm{Beta}(\alpha,1)$는 지수족으로 정리8.3.2의 조건을 만족함
- 통계량 $Y = \sum_{i=1}^n \log X_i$가 $\alpha$에 대한 완비충분통계량임
- 변환 아이디어:
    - $X_i^\alpha \sim U(0,1) \implies -\alpha\log X_i \sim \mathrm{Exp}(1) \implies -\alpha\sum\log X_i \sim \mathrm{Gamma}(n,1)$

- 계산 결과:
$$
E_\alpha\left[\frac{1}{-\sum\log X_i}\right]=\frac{\alpha}{n-1}\quad (n\ge 2) \\
\mathrm{Var}_\alpha\left[\frac{1}{-\sum\log X_i}\right]=\frac{\alpha^2}{(n-1)^2(n-2)}\quad (n\ge 3)
$$

- 따라서 UMVUE:
$$
\hat\alpha_n^{\mathrm{UMVUE}} = \frac{n-1}{-\sum_{i=1}^n \log X_i} \\
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

- 또한 예8.4.4로부터 MLE와의 관계:
$$
\hat\alpha_n^{\mathrm{UMVUE}} = \frac{n-1}{n}\hat\alpha_n^{\mathrm{MLE}}
$$
이고, 최소분산불편추정량은 최대가능도추정량과 같은 극한분포를 갖는다.
즉, $\sqrt{n}(\hat\theta_n^{\mathrm{MLE}} - \theta) \xrightarrow{d} N(0,\,\alpha^2)$와 같이 두 추정량은 점근적으로 동일한 효율성을 가진다.

### 피셔의 추측 (Fisher's conjecture)
정리 8.4.2에 따르면, 일차원 모수의 경우
$$
\operatorname{Var}_\theta\left(\sqrt{n}(\hat\theta_n - \theta)\right) \geq \frac{1}{I(\theta)}
$$
임을 알 수 있다. 이로부터, 점근정규성을 갖는 임의의 추정량 $\hat\theta_n$의 극한분포 분산 $\sigma^2(\theta)$에 대해
$$
\sigma^2(\theta) \geq \frac{1}{I(\theta)}
$$
라는 하한을 "추측"할 수 있다.

한편, 정리 6.4.4에 의해 최대가능도추정량(MLE)의 경우
$$
\sqrt{n}(\hat\theta_n^{\mathrm{MLE}} - \theta) \xrightarrow{d} N\left(0,\,\frac{1}{I(\theta)}\right)
$$
가 성립하므로, 점근정규성을 갖는 추정량들 중에서 MLE가 최소의 극한분포 분산을 갖는다고 볼 수 있다.

이러한 내용을 **피셔의 추측(Fisher's conjecture)** 라 하며, "점근정규성을 갖는 추정량들 중 MLE가 최소 극한분산을 갖는다"는 형태의 주장이다. 다만, 이 추측은 일반적으로는 항상 성립하지 않고, 추가적인 **균등수렴(uniform convergence)** 조건이 만족될 때에만 성립함이 알려져 있다.

즉, 최대가능도추정량은 이러한 조건 하에서 **점근적으로 효율적인 추정량(asymptotically efficient estimator)** 이라 하며, 이러한 점근적 효율성은 다차원 모수함수의 추정에도 확장된다.

#### 예 8.4.6 감마분포 $\mathrm{Gamma}(\alpha,\beta)$에서의 추정량 비교
- 모형: $\mathrm{Gamma}(\alpha,\beta)$, $\theta=(\alpha,\beta)^t$
- MLE는 가능도방정식을 푸는 수치적 방법(일단계 반복법, one-step iteration)으로 근사

초기값: 적률이용추정량(MME) $\hat\theta_n^{\mathrm{MME}} = (\hat\alpha_n, \hat\beta_n)^t$
는 연립방정식
$$
\hat\alpha_n\hat\beta_n = \bar X_n,\qquad
\hat\alpha_n(\hat\beta_n)^2 = \frac{1}{n}\sum_{i=1}^n (X_i-\bar X_n)^2
$$
의 해로 주어지고, 정리6.1.2로부터 그 극한분포를 구하면:
$$
\sqrt{n}(\hat\theta_n^{\mathrm{MME}}-\theta)\xrightarrow{d}N(0,\Sigma(\theta)) \\
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
(참고: $\Psi'(\alpha)$와 $\Psi''(\alpha)$는 각각 다이감마(digamma), 트라이감마(trigamma) 함수로, 로그감마함수 $\log\Gamma(\alpha)$의 1차, 2차 도함수다.)

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

- (표 생략) 적률이용추정량은 최대가능도추정량에 비해 점근적으로 효율성이 낮다. 이는 일반적으로 성립하며, 최대가능도추정량이 점근적으로 가장 효율적인 추정량임이 알려져 있다!
