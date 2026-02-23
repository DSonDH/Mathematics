# Chapter 2 단순회귀에 관한 추론
앞 장에서 단순회귀모형의 기본 가정 중 하나로 $E(Y \mid X=x) = \mu_{y\cdot x}$에 대해
$$\mu_{y\cdot x} = \beta_0 + \beta_1 x$$

가 성립한다고 하였다. 이는 모집단에서 $x$와 $y$ 사이에 선형관계가 존재함을 의미한다. $\beta_0, \beta_1$는 모집단의 모수(parameter)이다.  
모집단의 모든 관측값을 일일히 확일할 수 없으므로, 표본을 이용하여 $\beta_0, \beta_1$ 및 $\mu_{y\cdot x}$에 대한 추정과 구간추정을 수행한다. 표본 $(x_i, y_i), i=1,\dots,n$을 이용하면 회귀모형은

$$\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x$$

으로 추정된다. 물론 이추정량들은 통계량이므로 분포를 가지며, 이를 이용하여 모수에 대한 구간추정과 가설검정을 수행할 수 있다. 이제 1장에서의 단순회귀모형 가정이 모두 성립한다고 전제하여 $\beta_0, \beta_1$ 및 $\mu_{y\cdot x}$에 대한 구간추정을 살펴보자.
- prediction band: 특정 $x$에서의 평균반응 $\mu_{y\cdot x}$에 대한 구간추정은 confidence band라고도 불리는 반면, 개별 관측값 $y$에 대한 구간추정은 prediction band라고도 불린다. 이는 개별 관측값이 평균반응보다 더 큰 변동성을 가지기 때문이다.


## 2.1 구간추정 (Interval Estimation)

### 2.1.1 $\beta_1$의 신뢰구간
기울기 $\beta_1$의 추정량은
$$\hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$$
였는데, 이를 다음과 같이 쓸 수 있다.
$$\hat{\beta}_1 = \sum a_i y_i, \quad a_i = \frac{x_i - \bar{x}}{\sum (x_i - \bar{x})^2}$$
따라서 $\hat{\beta}_1$은 서로 독립인 $y_i$들의 선형결합이고, $y_i$는 정규분포를 따르므로 $\hat{\beta}_1$도 정규분포를 따른다.

**기대값**  
$$E(\hat{\beta}_1) = \sum a_i E(y_i) = \sum a_i (\beta_0 + \beta_1 x_i) = \beta_1$$
즉, $\hat{\beta}_1$은 불편추정량이다.

**분산**  
$$\mathrm{Var}(\hat{\beta}_1) = \sigma^2 \sum a_i^2 = \frac{\sigma^2}{S_{xx}} \\
S_{xx} = \sum (x_i - \bar{x})^2$$

$\sigma^2$의 추정량은 $MSE = \frac{SSE}{n-2}$ 이므로,
$$\widehat{\mathrm{Var}}(\hat{\beta}_1) = \frac{MSE}{S_{xx}}$$
표준편차는
$$\hat{\sigma}_{\hat{\beta}_1} = \sqrt{\frac{MSE}{S_{xx}}}$$

**신뢰구간**  
- $\sigma^2$를 아는 경우:
$$\hat{\beta}_1 \pm z_{\alpha/2} \frac{\sigma}{\sqrt{S_{xx}}}$$
- $\sigma^2$를 모르는 경우:
$$\hat{\beta}_1 \pm t_{\alpha/2}(n-2) \sqrt{\frac{MSE}{S_{xx}}}$$

### 2.1.2 $\beta_0$의 신뢰구간
모집단에서 회귀모형의 절편은 $\beta_0$이지만, 표본에서는 $\hat{\beta}_0$으로 추정된다.
$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

**기대값**  
$$E(\hat{\beta}_0) = E(\bar{y} - \hat{\beta}_1 \bar{x}) = E(\bar{y}) - \bar{x} E(\hat{\beta}_1) = \beta_0$$
즉, 불편추정량이다.

**분산**  
$$\mathrm{Var}(\hat{\beta}_0) = \sigma^2 \left( \frac{1}{n} + \frac{\bar{x}^2}{S_{xx}} \right)$$
따라서 추정 분산은
$$\widehat{\mathrm{Var}}(\hat{\beta}_0) = MSE \left( \frac{1}{n} + \frac{\bar{x}^2}{S_{xx}} \right)$$

표준편차는
$$\hat{\sigma}_{\hat{\beta}_0} = \sqrt{ MSE \left( \frac{1}{n} + \frac{\bar{x}^2}{S_{xx}} \right) }$$

**신뢰구간**  
- $\sigma^2$를 아는 경우:
$$\hat{\beta}_0 \pm z_{\alpha/2} \sqrt{ \sigma^2 \left( \frac{1}{n} + \frac{\bar{x}^2}{S_{xx}} \right) }$$
- $\sigma^2$를 모르는 경우:
$$\hat{\beta}_0 \pm t_{\alpha/2}(n-2) \sqrt{ MSE \left( \frac{1}{n} + \frac{\bar{x}^2}{S_{xx}} \right) }$$

### 2.1.3 $\mu_{y\cdot x} = E(Y|X=x)$의 신뢰구간
어떤 특정한 $x$에서의 평균반응은 $\mu_{y\cdot x} = \beta_0 + \beta_1 x$ 이며, 추정값은 $\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x$

**기대값**  
$$E(\hat{y}) = \mu_{y\cdot x}$$
즉, 불편추정량이다.

**분산**  
$$\mathrm{Var}(\hat{y}) = \sigma^2 \left( \frac{1}{n} + \frac{(x - \bar{x})^2}{S_{xx}} \right)$$
- 분산은 $x$의 함수로서, $\bar{x}$에서 멀어질수록 증가하고, $\bar{x}$에서 최소가 된다.
- 표본크기 $n$이 커질수록 분산이 감소한다.

추정 분산은
$$\widehat{\mathrm{Var}}(\hat{y}) = MSE \left( \frac{1}{n} + \frac{(x - \bar{x})^2}{S_{xx}} \right)$$

**신뢰구간**
- $\sigma^2$를 아는 경우:
$$\hat{y} \pm z_{\alpha/2} \sqrt{ \sigma^2 \left( \frac{1}{n} + \frac{(x - \bar{x})^2}{S_{xx}} \right) }$$
- $\sigma^2$를 모르는 경우:
$$\hat{y} \pm t_{\alpha/2}(n-2) \sqrt{ MSE \left( \frac{1}{n} + \frac{(x - \bar{x})^2}{S_{xx}} \right) }$$

### 2.1.4 개별 관측값 $y$의 신뢰구간 (예측구간)
$y$의 기댓값이 아닌, 개별적인 관측값 $y_0$에 대한 구간추정은 예측구간(prediction interval)이라고 한다.

하나의 새로운 관측값 $y_0$에 대한 예측분산은
$$\mathrm{Var}(y_0) = \mathrm{Var}(\hat{y}) + \mathrm{Var}(\epsilon_0) = \sigma^2 \left( 1 + \frac{1}{n} + \frac{(x - \bar{x})^2}{S_{xx}} \right)$$
이고, $$\widehat{\mathrm{Var}}(\hat{y}_0) = MSE \left( 1 + \frac{1}{n} + \frac{(x - \bar{x})^2}{S_{xx}} \right)$$

따라서 예측구간은

- $\sigma^2$를 아는 경우:
$$\hat{y} \pm z_{\alpha/2} \sqrt{ \sigma^2 \left( 1 + \frac{1}{n} + \frac{(x - \bar{x})^2}{S_{xx}} \right) }$$

- $\sigma^2$를 모르는 경우:
$$\hat{y} \pm t_{\alpha/2}(n-2) \sqrt{ MSE \left( 1 + \frac{1}{n} + \frac{(x - \bar{x})^2}{S_{xx}} \right) }$$

- 예측구간은 신뢰구간보다 항상 넓다. 이는 개별 관측값이 평균반응보다 더 큰 변동성을 가지기 때문이다.


## 2.2 가설검정 (Hypothesis Testing)
모집단의 회귀직선
$$\mu_{y\cdot x} = \beta_0 + \beta_1 x$$
의 모수에 대해 특정 값을 취하는지 아닌지 여부를 검정하는 절차를 다룬다.  
일반적으로 모수 $\theta$의 불편추정량 $\hat{\theta}$가 정규분포를 따르고 분산이 알려진 경우, 검정통계량은
$$Z_0 = \frac{\hat{\theta} - \theta_0}{\sqrt{\mathrm{Var}(\hat{\theta})}}$$

분산이 알려져 있지 않으면 표본으로 추정하여
$$t_0 = \frac{\hat{\theta} - \theta_0}{\sqrt{\widehat{\mathrm{Var}}(\hat{\theta})}}$$
을 사용하며, 단순회귀의 경우 자유도는 $n-2$이다.

### 2.2.1 $\beta_1$의 검정
가설은 다음과 같이 설정한다.
$$H_0 : \beta_1 = \beta_{10}, \qquad H_1 : \beta_1 \neq \beta_{10}$$
표본 $(x_i, y_i), i=1,2,\dots,n$을 취하여 최소제곱법에 의한 회귀식 $\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x$을 구한다.

$$\widehat{\mathrm{Var}}(\hat{\beta}_1) = \frac{MSE}{S_{xx}}$$
따라서 검정통계량은
$$t_0 = \frac{\hat{\beta}_1 - \beta_{10}}{\sqrt{MSE/S_{xx}}}$$
이며 자유도 $n-2$의 t-분포를 따른다.

양측검정에서
$$|t_0| > t_{\alpha/2}(n-2)$$
이면 귀무가설을 기각한다.

### 2.2.2 $\beta_0$의 검정
가설:
$$H_0 : \beta_0 = \beta_{00}, \qquad H_1 : \beta_0 < \beta_{00} \quad \text{(단측 예시)}$$

$$\widehat{\mathrm{Var}}(\hat{\beta}_0) = MSE \left( \frac{1}{n} + \frac{\bar{x}^2}{S_{xx}} \right)$$

검정통계량:
$$t_0 = \frac{\hat{\beta}_0 - \beta_{00}}{\sqrt{ MSE\left(\frac{1}{n}+\frac{\bar{x}^2}{S_{xx}}\right) }}$$

단측검정에서는
$$t_0 < -t_\alpha(n-2)$$
이면 기각한다.

### 2.2.3 $\mu_{y\cdot x}$의 검정
특정 $x$에서 평균반응 $\mu_{y\cdot x}$이 특정값 $\mu_{y\cdot x}^0$과 같은지 여부를 검정한다.
$$H_0 : \mu_{y\cdot x} = \mu_{y\cdot x}^0, \qquad H_1 : \mu_{y\cdot x} > \mu_{y\cdot x}^0$$

$$\widehat{\mathrm{Var}}(\hat{y}) = MSE \left( \frac{1}{n} + \frac{(x-\bar{x})^2}{S_{xx}} \right)$$

검정통계량:
$$t_0 = \frac{\hat{y} - \mu_{y\cdot x}^0}{\sqrt{ MSE \left( \frac{1}{n} + \frac{(x-\bar{x})^2}{S_{xx}} \right) }}$$

단측검정에서는
$$t_0 > t_\alpha(n-2)$$
이면 기각한다.

### 2.2.4 개별 예측값 $y_s$의 검정
특정 $x$에서의 개별 관측값 $y_s$가 특정값 $y_0$과 같은지 여부를 검정한다.
$$H_0 : y_s = y_0, \qquad H_1 : y_s \neq y_0$$

$$\widehat{\mathrm{Var}}(\hat{y}_s) = MSE \left( 1 + \frac{1}{n} + \frac{(x-\bar{x})^2}{S_{xx}} \right)$$

검정통계량:
$$t_0 = \frac{\hat{y}_s - y_0}{\sqrt{ MSE \left( 1+\frac{1}{n}+\frac{(x-\bar{x})^2}{S_{xx}} \right) }}$$

양측검정에서는
$$|t_0| > t_{\alpha/2}(n-2)$$
이면 기각한다.


## 2.3 상관계수의 검정 (Test of Correlation Coefficient)
이 절에서는 모집단 상관계수 $\rho$에 대한 추론을 다룬다.
표본상관계수 $r$의 분포는 정규분포가 아니며, $\rho$에 의존한다. 이를 해결하기 위해 Fisher의 z-변환을 사용한다.

> 참고: Fisher's z-transformation
> r을 g(r)로 변환하여 정규분포에 근사시키는 방법. r의 분포가 비대칭적이므로, z-변환을 통해 정규분포에 근사시킬 수 있다.
> $$g(r) = \frac{1}{2}\ln\left(\frac{1+r}{1-r}\right)$$

**Fisher 변환**  
$$Z' = \frac{1}{2}\ln\left(\frac{1+r}{1-r}\right)$$
표본이 충분히 클 때 (대략 $n \ge 25$)
$$E(Z') = \frac{1}{2} \ln\left(\frac{1+\rho}{1-\rho}\right) \\
\mathrm{Var}(Z') = \frac{1}{n-3}$$

표준화된 통계량 $$Z = \frac{Z' - E(Z')}{\sqrt{\widehat{\mathrm{Var}}(Z')}} = \frac{Z' - E(Z')}{\sqrt{1/(n-3)}} \sim N(0,1)$$

**신뢰구간**  
$$Z' \pm z_{\alpha/2} \sqrt{\frac{1}{n-3}}$$
을 계산한 후 역변환하여 $\rho$의 구간을 구한다.

**가설검정**  
$$H_0 : \rho = \rho_0, \qquad H_1 : \rho \neq \rho_0$$

$$Z = \frac{Z' - E(Z')}{\sqrt{1/(n-3)}}$$

$$|Z| > z_{\alpha/2}$$
이면 기각한다.

**회귀계수와의 관계**  
설명변수 $x$가 확률변수가 아닌 경우,
$$H_0 : \beta_1 = 0$$
은
$$H_0 : \rho = 0$$
과 동등하다. 이때 검정통계량은
$$t_0 = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}}$$
이며 자유도 $n-2$의 t-분포를 따른다.

> 회귀분석 모형에서 설명변수 $x$가 확률변수가 아니므로 모상관계수 $\rho$는 정의되지 않지만, 표본상관계수 $r$은 통계량 값으로, 정의는 문제가 없어서 여전히 계산할 수 있다.  
> 이 경우 $r$이 0에 가까울수록 회귀계수 $\hat{\beta}_1$이 0에 가까워지고, $r$이 1 또는 -1에 가까울수록 $\hat{\beta}_1$의 절대값이 커진다. 따라서 $r$이 0에서 멀어질수록 회귀계수 $\hat{\beta}_1$이 유의미하게 다르다고 판단할 가능성이 높아진다.


## 2.4 모형의 타당성 (Model Adequacy)
앞 절까지의 추론은 다음 단순선형회귀모형을 전제로 한다.
$$y = \beta_0 + \beta_1 x + \varepsilon, \qquad \varepsilon \sim N(0,\sigma^2)$$
즉, 설명변수 $x$와 반응변수 $y$ 사이에 선형관계가 존재하며, 오차 $\varepsilon$는 평균이 0이고 분산이 $\sigma^2$인 정규분포를 따르며 오차는 서로 독립이라는 가정이다.  
이 절에서는 이러한 가정이 자료에 적합한지 검토하는 방법을 다룬다. 이를 타당성 검토라 한다.

### 2.4.1 적합결여검정 (Lack-of-Fit Test)
설명변수 $x$의 각 수준에서 반복측정이 존재할 때, 선형모형이 적절한지 검정할 수 있다. 각 $x_i$ 수준에서 $n_i$개의 관측값이 있다고 하자.
$$
y_{i1}, y_{i2}, \dots, y_{in_i} \ \text{for } i=1,2,\dots,k
$$
전체 표본수는
$$n = \sum_{i=1}^{k} n_i$$
최소제곱추정에 의한 회귀식은
$$\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$$

하나의 점 $x_i$에서는 $n_i$개의 관측값이 존재하지만, 회귀모형에 의한 예측값은 하나뿐이다. 따라서 오차제곱합은
$$SSE = \sum_{i=1}^{k}\sum_{j=1}^{n_i} (y_{ij} - \hat{y}_i)^2$$
이를 다음과 같이 분해할 수 있다.
$$SSE = \sum_{i=1}^{k}\sum_{j=1}^{n_i}(y_{ij}-\bar{y}_i)^2 + \sum_{i=1}^{k} n_i (\bar{y}_i - \hat{y}_i)^2$$

첫 항: **순오차제곱합, pure error sum of squares**
$$SSPE = \sum_{i=1}^{k}\sum_{j=1}^{n_i}(y_{ij}-\bar{y}_i)^2$$

두 번째 항: **적합결여제곱합, lack-of-fit sum of squares**
$$SSLF = \sum_{i=1}^{k} n_i (\bar{y}_i - \hat{y}_i)^2$$

따라서
$$SSE = SSPE + SSLF$$

**자유도**  
$$df(SSPE) = n - k, \quad df(SSLF) = k - 2$$
- $SSPE$는 각 $x_i$ 수준에서의 관측값과 평균값 간의 변동이므로, 자유도는 각 수준에서의 관측값 수에서 평균값을 빼준 $n_i - 1$을 모두 더한 $n-k$가 된다.
- $SSLF$는 회귀모형에 의한 예측값과 각 수준의 평균값 간의 변동이므로, 자유도는 회귀모형에서 추정되는 모수의 수인 2 (절편과 기울기)를 빼준 $k-2$가 된다.

**평균제곱**  
$$MSPE = \frac{SSPE}{n-k}, \quad MSLF = \frac{SSLF}{k-2}$$
- 적협결여평균제곱 (lack-of-fit mean square) $MSLF$는 회귀모형이 설명하지 못하는 변동의 평균값을 나타낸다.
- 순오차평균제곱 (pure error mean square) $MSPE$는 각 $x_i$ 수준에서의 관측값 간의 변동의 평균값을 나타낸다.

**검정통계량**  
$$F_0 = \frac{MSLF}{MSPE}$$
가설:
$$H_0 : E(Y|X=x) = \beta_0 + \beta_1 x \quad H_1 : E(Y|X=x) \neq \beta_0 + \beta_1 x \\
F_0 > F_\alpha(k-2, n-k)$$
이면 선형모형이 부적절하다고 판단한다.

### 2.4.2 잔차의 검토 (Residual Analysis)
잔차는 $e_i = y_i - \hat{y}_i$이고, OLS의 성질로부터
$$\sum e_i = 0, \qquad \sum x_i e_i = 0, \qquad \sum \hat{y}_i e_i = 0$$

**잔차 산점도 해석**
- 랜덤한 패턴 → 선형성 가정 적절
- 곡선형 패턴 → 다항모형 고려
- 분산 증가 패턴 → 이분산 가능성 / 가중회귀모형 고려
- 구조적 패턴 → 모형 오적합

### 2.4.3 잔차의 성질
잔차: $$e_i = y_i - \hat{y}_i = (y_i - \mu_{y\cdot x_i}) + (\mu_{y\cdot x_i} - \hat{y}_i) = \varepsilon_i - (\hat{\beta}_0 - \beta_0) - (\hat{\beta}_1 - \beta_1)x_i$$

평균: $E(e_i) = 0$  
분산: 
$$\mathrm{Var}(e_i) = \left[ 1 - \frac{1}{n} - \frac{(x_i-\bar{x})^2}{S_{xx}} \right]\sigma^2$$
공분산:
$$\mathrm{Cov}(e_i,e_j) = \left[ \frac{1}{n} + \frac{(x_i-\bar{x})(x_j-\bar{x})}{S_{xx}} \right]\sigma^2 \quad (i\neq j)$$
즉, 잔차는 서로 독립이 아니다.
- 오차항은 서로 독립이지만, 잔차는 서로 독립이 아니다. 이는 잔차가 오차항과 회귀계수 추정량의 선형결합으로 표현되기 때문이다. 회귀계수 추정량은 모든 관측값에 의존하므로, 잔차도 모든 관측값에 의존하게 된다. 따라서 잔차는 서로 독립이 아니다.

모형 타당성 점검의 핵심 요소
1. 선형성(linearity)
2. 등분산성(homoscedasticity)
3. 정규성(normality)
4. 독립성(independence)


## 2.5 오차의 자기상관 (Autocorrelation of Errors)
설명변수 (x)가 시간(time)을 나타내는 경우, 오차항 사이의 독립성 가정이 특히 중요해진다.
잔차가 시간의 흐름에 따라 일정한 주기(cycle)나 패턴을 보인다면 오차들 사이에 상관관계가 존재할 가능성이 있다.

단순선형회귀모형
$$y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$$
에서 오차가 다음과 같이 주어진다고 하자.

$$\varepsilon_i = \rho \varepsilon_{i-1} + \delta_i$$
* $\rho \neq 0$
* $E(\delta_i)=0$
* $\mathrm{Var}(\delta_i)=\sigma_\delta^2$
* $\mathrm{Cov}(\delta_i,\delta_j)=0 \quad (i\neq j)$

이면 이를 **1차 자기상관(first-order autocorrelation)** 이라고 한다.

**오차의 전개**  
식 (4.46)을 반복 대입하면
$$\varepsilon_i = \rho \varepsilon_{i-1} + \delta_i = \rho (\rho \varepsilon_{i-2} + \delta_{i-1}) + \delta_i = \rho^2 \varepsilon_{i-2} + \rho \delta_{i-1} + \delta_i \\
= \rho^3 \varepsilon_{i-3} + \rho^2 \delta_{i-2} + \rho \delta_{i-1} + \delta_i = \cdots \\ = \sum_{j=0}^{\infty} \rho^j \delta_{i-j}$$

$E(\delta_i)=0$이고 $Cov(\delta_i,\delta_j)=0$이므로,
$$
E(\varepsilon_i)=0, \quad \forall i \\
\mathrm{Var}(\varepsilon_i) = \sigma_\delta^2 \sum_{j=0}^{\infty} \rho^{2j} = \frac{\sigma_\delta^2}{1-\rho^2} $$

$$
\mathrm{Cov}(\varepsilon_i,\varepsilon_{i-1}) = E(\varepsilon_i \varepsilon_{i-1}) = E\left( \sum_{j=0}^{\infty} \rho^j \delta_{i-j} \cdot \sum_{k=0}^{\infty} \rho^k \delta_{i-1-k} \right) =
\rho \frac{\sigma_\delta^2}{1-\rho^2}
$$
일반적으로
$$\mathrm{Cov}(\varepsilon_i,\varepsilon_{i-j}) = \rho^j \frac{\sigma_\delta^2}{1-\rho^2}$$

따라서 일차자기상관이 있는 단순선형회귀모형에서 오차벡터 $\varepsilon=(\varepsilon_1,\dots,\varepsilon_n)'$의 분산-공분산행렬은
$$\mathrm{Var}(\varepsilon) = E(\varepsilon \varepsilon') = \frac{\sigma_\delta^2}{1-\rho^2} \begin{pmatrix} 1 & \rho & \rho^2 & \cdots \\ \rho & 1 & \rho & \cdots \\ \rho^2 & \rho & 1 & \cdots \\ \vdots & \vdots & \vdots & \ddots \end{pmatrix}$$

### Durbin–Watson 검정
일차자기상관계수(first-order autocorrelation coefficient) $\rho$가 0인지 여부를 검정하는 절차를 다룬다.  
오차의 자기상관 여부를 검정하기 위한 통계량은 Durbin–Watson 통계량이다.

잔차 $e_i = y_i - \hat{y}_i$를 이용하여
$$d = \frac{\sum_{i=2}^{n}(e_i - e_{i-1})^2}{\sum_{i=1}^{n} e_i^2}$$
로 정의한다.

- 오차항 $\varepsilon_i$의 자기상관계수 $\rho$가 0이면 
  - $E(\varepsilon_i -\varepsilon_{i-1})^2 = E(\varepsilon_i^2) + E(\varepsilon_{i-1}^2) - 2E(\varepsilon_i \varepsilon_{i-1})$에서 $E(\varepsilon_i \varepsilon_{i-1})=0$이므로 $E(\varepsilon_i -\varepsilon_{i-1})^2 = 2\sigma^2$가 된다. 따라서 $d$의 기대값은 2가 된다.
- 양의 자기상관($\rho > 0$)이면 
  - $E(\varepsilon_i \varepsilon_{i-1}) > 0$이므로 $E(\varepsilon_i -\varepsilon_{i-1})^2 < 2\sigma^2$가 된다. 따라서 $d$의 기대값은 2보다 작아진다.
- 음의 자기상관($\rho < 0$)이면 
  - $E(\varepsilon_i \varepsilon_{i-1}) < 0$이므로 $E(\varepsilon_i -\varepsilon_{i-1})^2 > 2\sigma^2$가 된다. 따라서 $d$의 기대값은 2보다 커진다.

잔차의 표본 자기상관계수는
$$\hat{\rho} = \frac{\sum_{i=2}^{n} e_i e_{i-1}}{\sqrt{\sum e_i^2}\sqrt{\sum e_{i-1}^2}}$$
로 정의되는데, 표본이 충분히 크면
$$d = \frac{\sum_{i=2}^{n}(e_i - e_{i-1})^2}{\sum_{i=1}^{n} e_i^2} = \frac{\sum e_i^2 + \sum e_{i-1}^2 - 2\sum e_i e_{i-1}}{\sum e_i^2}
\approx 2(1-\hat{\rho})$$

따라서
* $\rho=0$이면 $d\approx 2$
* 양의 자기상관($\rho>0$)이면 $d<2$
* 음의 자기상관($\rho<0$)이면 $d>2$

#### 양의 자기상관 검정
가설:
$$H_0: \rho=0 \qquad H_1: \rho>0$$

판정:
* $d < d_L$ → $H_0$ 기각
* $d > d_U$ → $H_0$ 채택
* $d_L < d < d_U$ → 불확정

#### 음의 자기상관 검정
$$H_0: \rho=0 \qquad H_1: \rho<0$$

상한과 하한을
$$d_L^* = 4 - d_U, \qquad d_U^* = 4 - d_L$$
로 변환하여 동일한 절차를 적용한다.

#### 오차항에 자기상관이 존재할 때의 회귀계수 성질
자기상관이 존재하더라도 OLS 추정량은
$$E(\hat{\beta}_0)=\beta_0, \qquad E(\hat{\beta}_1)=\beta_1$$
즉, 불편성을 유지한다.

그러나
* 최소분산추정량(MVUE)이 아님
* 분산이 비효율적
* 표준오차가 왜곡됨
* t-검정, F-검정이 신뢰할 수 없게 됨

이러한 경우 해결방법은
1. 잔차로부터 $\hat{\rho}$를 추정
  - $\hat{\rho} = \frac{\sum_{i=2}^{n} e_i e_{i-1}}{\sum e_i^2}$
  - 일반적인 시계열 모형을 가정하고 $\hat{\rho}$를 추정해도 됨
2. 일반화최소제곱법(GLS, generalized least squares) 적용
