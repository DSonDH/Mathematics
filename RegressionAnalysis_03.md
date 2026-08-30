# Chapter 3 기초적 회귀분석의 기타 논제 (Additional Topics in Basic Regression)
본 장은 단순선형회귀(simple linear regression)와 관련하여 기존 장에서 다루지 않은 주요 주제들을 정리한 것이다. 특히 비선형관계를 선형모형으로 변환하는 방법, 역변환, Box–Cox 변환 등 실무적 자료분석에서 자주 활용되는 기법들을 체계적으로 다룬다.

## 3.1 모형의 변환 (Model Transformation)
실제 자료에서는 설명변수 (x)와 반응변수 (y)의 관계가 선형모형(linear model)

$$y = \beta_0 + \beta_1 x + \varepsilon$$

을 따르지 않는 경우가 많다. 이러한 비선형모형(nonlinear model)에 대해 적절한 함수변환(functional transformation)을 적용하면 선형회귀분석으로 처리할 수 있다.

### 3.1.1 로그변환 (Logarithmic Transformation)

#### (1) 지수형 모형 (Exponential Model)

$$y = a_0 a_1^x \varepsilon$$

* $(a_0, a_1)$: 회귀모수(regression parameters)
* $(\varepsilon)$: 오차항(error term), 평균 1, 분산 $\sigma^2$

양변에 로그(logarithm)를 취하면

$$\log y = \log a_0 + x \log a_1 + \log \varepsilon$$

변수 치환:
* $y' = \log y$
* $\beta_0 = \log a_0$
* $\beta_1 = \log a_1$
* $\varepsilon' = \log \varepsilon$

선형모형으로 변환됨:

$$y' = \beta_0 + \beta_1 x + \varepsilon'$$

최소제곱추정(least squares estimation)을 통해 $(\hat{\beta}_0, \hat{\beta}_1)$을 구한 후,

$$\hat{a}_0 = \text{antilog}(\hat{\beta}_0), \quad \hat{a}_1 = \text{antilog}(\hat{\beta}_1)$$

따라서 원래 모형의 추정식은

$$\hat{y} = \hat{a}_0 \hat{a}_1^x$$

#### (2) 멱함수 모형 (Power Function Model)

$$y = a_0 x^{a_1} \varepsilon$$

로그변환하면

$$\log y = \log a_0 + a_1 \log x + \log \varepsilon$$

변수 치환:
* $y' = \log y$
* $x' = \log x$
* $\beta_0 = \log a_0$
* $\beta_1 = a_1$

선형모형:

$$y' = \beta_0 + \beta_1 x' + \varepsilon'$$

역변환 후 추정식:

$$\hat{y} = \hat{a}_0 x^{\hat{a}_1}$$

#### (3) 기타 선형화 가능한 모형
1. **지수-선형 모형**

    $$y = e^{\beta_0 + \beta_1 x} \varepsilon$$

    → 자연로그(natural logarithm) 취하면 선형화.

2. **로지스틱 모형(Logistic Model)**

$$y = \frac{1}{1 + e^{\beta_0 + \beta_1 x + \varepsilon}}$$

양변 변환:

$$\ln\left(\frac{1}{y} - 1\right) = \beta_0 + \beta_1 x + \varepsilon$$

이는 로짓변환(logit transformation)에 해당한다.

#### (4) 모형 선택 기준
* 산점도(scatter plot) 확인
* 여러 모형 적합 후 결정계수 ($R^2$) 비교
* 적합도 검정(goodness-of-fit test)
* 회귀진단(regression diagnostics)

### 3.1.2 역변환 (Reciprocal Transformation)
비선형 모형:

$$y = \beta_0 + \beta_1 \left(\frac{1}{x}\right) + \varepsilon$$

변환:

$$x' = \frac{1}{x}$$

선형모형:

$$y = \beta_0 + \beta_1 x' + \varepsilon$$

추가 가능한 형태:
1. $y' = 1/y$
2. $x' = 1/x$
3. $y' = 1/y,\; x' = 1/x$

적용 상황:
* $x$ 증가에 따라 $y$가 일정 값으로 **수렴(convergence)** 하는 곡선(curvilinear relationship)을 보일 때 적절하다.

### 3.1.3 Box–Cox 변환 (Box–Cox Transformation)
로그변환, 역변환, 멱변환(power transformation)을 포괄하는 일반적 변환체계이다. 정규성(normality), 등분산성(homoscedasticity), 선형성(linearity)을 동시에 만족시키기 위한 방법이다.

#### (1) 변환 정의
반응변수 $y > 0$ 가정:

$$y^{(\lambda)} =
\begin{cases}
\frac{y^\lambda - 1}{\lambda}, & \lambda \neq 0 \\
\ln(y), & \lambda = 0
\end{cases}$$

특수한 경우:
* $\lambda = 0$: 로그변환
* $\lambda = 1$: 원자료와 거의 동일
* $\lambda = -1$: 역변환
* $\lambda = 1/2$: 제곱근변환(square root transformation)

#### (2) 변환 후 회귀모형

$$y^{(\lambda)} = \beta_0 + \beta_1 x + \varepsilon$$

형태로 선형모델을 만족한다는 가정 하에 모수를 추정해보자.  
$\lambda$에 따라 회귀계수가 달라지며, 잔차제곱합(SSE, sum of squared errors):

$$SSE_\lambda = \sum_{i=1}^{n} \left(y_i^{(\lambda)} - \hat{\beta}_0 - \hat{\beta}_1 x_i\right)^2$$

#### (3) $\lambda$ 선택 방법
* $SSE_\lambda$ 최소화
* 최대가능도추정(maximum likelihood estimation)
* 정규성 및 등분산성 개선 여부 확인

#### (4) 표준화된 Box–Cox 절차
변환식:

$$z_i^{(\lambda)} =
\begin{cases}
\frac{y_i^\lambda - 1}{\lambda}
\left(\frac{1}{\left(\prod y_i\right)^{1/n}}\right)^{\lambda-1},
& \lambda \neq 0 \\
\ln(y_i)\left(\prod y_i\right)^{1/n},
& \lambda = 0
\end{cases}$$

절차 (Box–Cox procedure, 자세한건 참고문헌(5.3) 참조):
1. 변환값 계산
2. 선형회귀 적합, SSE 계산
3. SSE 최소가 되는 $\lambda$ 선택

#### (5) 확장 모형
* Yeo–Johnson 변환(Yeo–Johnson Transformation): 음수값 허용
* Cook–Weisberg 절차(Cook–Weisberg Procedure): 이론적 기반 확장


## 3.2 x의 측정오차 (Measurement Error in X)
기존 회귀모형에서는 설명변수 $x$가 오차 없이 관측된다고 가정하였다. 그러나 실제 자료에서는 $x$ 역시 측정오차(measurement error)를 포함할 수 있다.

### 3.2.1 모형 설정과 기본 가정
* 참값(true value): $x_i$
* 관측값(observed value): $x_i^*$
* 측정오차:

  $$\delta_i = x_i^* - x_i$$

가정:
* $E(\delta_i) = 0$
* $\mathrm{Var}(\delta_i) = \sigma_\delta^2$
* $\delta_i \sim N(0, \sigma_\delta^2)$
* $\varepsilon_i$와 $\delta_i$는 서로 독립(independence)

참값 모형:

$$y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$$

이를 관측값으로 표현하면

$$y_i = \beta_0 + \beta_1 x_i^* + (\varepsilon_i - \beta_1 \delta_i)$$

- 이는 일반적인 회귀직선 모형처럼 보이지만 큰 차이가 있다
  - 오차항이 $x_i^*$와 상관되어 있다
  - 고전적 회귀가정(classical regression assumption)이 위배된다

### 3.2.2 공분산 구조
설명변수와 오차항의 공분산:

$$\mathrm{Cov}(x_i^*, \varepsilon_i - \beta_1 \delta_i) = E[(x_i^* - E(x_i^*))( \varepsilon_i - \beta_1 \delta_i - E(\varepsilon_i - \beta_1 \delta_i))]$$

$$= E[(x_i + \delta_i - x_i)(\varepsilon_i - \beta_1 \delta_i)]$$

$$= E[\delta_i \varepsilon_i] - \beta_1 E[\delta_i^2]
= -\beta_1 \sigma_\delta^2$$

즉, $x_i^*$와 오차항이 상관되어 고전적 회귀가정(classical regression assumption)이 위배된다.

### 3.2.3 최소제곱추정량의 편의 (Bias)
표본 회귀계수:

$$\hat{\beta}_1 = \frac{\sum (x_i^* - \bar{x}^*)(y_i - \bar{y})}{\sum (x_i^* - \bar{x}^*)^2}$$

대수법칙(law of large numbers)에 의해

$$\hat{\beta}_1 = \frac{\sum (x_i^* - \bar{x}^*)(y_i - \bar{y})}{\sum (x_i^* - \bar{x}^*)^2} \to \frac{\mathrm{Cov}(x_i^*, y_i)}{\mathrm{Var}(x_i^*)} \\
= \frac{\mathrm{Cov}(x_i^*, \beta_0 + \beta_1 x_i + \varepsilon_i)}{\mathrm{Var}(x_i^*)} 
= \beta_1 \left(1 - \frac{\sigma_\delta^2}{\mathrm{Var}(x_i^*)}\right)$$

따라서
* 편의(bias) 존재
* 일치성(consistency) 불만족

이를 감쇠편의(attenuation bias)라고 한다.


## 3.3 x의 수준 선택 (Choice of Levels of x)
실험계획법(design of experiments)의 관점에서 설명변수 $x$의 관측수준(level)을 어떻게 선택할 것인지가 중요한 문제이다.

- 정해진 $x$값에서 몇 번 $y$를 관측할 것인가?
- $x$값을 어떻게 분포시킬 것인가? 등등

**실험 목적**
1. 기울기 $\beta_1$을 정확히 추정
2. 절편 $\beta_0$을 정확히 추정
3. 평균반응 $E(Y|X=x)$ 예측
4. 개별반응 예측

관련 분산식:

$$\mathrm{Var}(\hat{\beta}_1) = \sigma^2 \left[ \frac{1}{\sum (x_i - \bar{x})^2} \right]\\
\mathrm{Var}(\hat{\beta}_0) = \sigma^2 \left[ \frac{1}{n} + \frac{\bar{x}^2}{\sum (x_i - \bar{x})^2} \right]\\
\mathrm{Var}(\hat{y}) = \sigma^2 \left[ \frac{1}{n} + \frac{(x_i - \bar{x})^2}{\sum (x_i - \bar{x})^2} \right]\\
\mathrm{Var}(\hat{y}_0) = \sigma^2 \left[ 1 + \frac{1}{n} + \frac{(x_i - \bar{x})^2}{\sum (x_i - \bar{x})^2} \right]$$

**기울기 추정 정밀화**  
기울기 분산 최소화를 위해서는

$$S_{xx} = \sum (x_i - \bar{x})^2$$

를 최대화해야 한다.

실험영역(experimental region) $[x_a, x_b]$에서
* $n/2$를 $x_a$
* $n/2$를 $x_b$ 에 배치하면 $S_{xx}$가 최대가 된다.
* 즉, $x$의 수준을 양 끝점에 배치하는 것이 기울기 추정 정밀화에 가장 효과적이다.
* 하지만, 이런 극단적 배치는 비선형성(nonlinearity) 가능성을 간과할 수 있다.
* 비선형 가능성이 있는 경우에는 최소 세 수준을 사용하는 것이 바람직하다.

**절편 추정 정밀화**  
절편의 분산을 줄이려면 $\bar{x} = 0$이 되도록 설계하는 것이 이상적이다.

**예측 정확도 향상**  
관심 있는 $x$ 값이 $\bar{x}$에 가까울수록 예측분산이 작다.


## 3.4 두 회귀선의 비교 (Comparison of Two Regression Lines)
두 모집단(population)을 비교할때, 회귀직선이 동일한지 검정하는 문제이다.

* 소득 당 소비전력량의 관계를 보고싶은데, 서울 모델이랑 부산 모델이랑 같을까?
* 지역 간 소득–소비 관계 비교
* 생산라인 간 공정 특성 비교
* **두 회귀모형이 같다면, 두 모집단을 합쳐서 분석해서 더 정확한 추정량을 얻을 수 있다.**
* **두 회귀모델이 달라도, 기울기는 동일할 수 있는데, 이 경우 절편만 다른 것으로 해석할 수 있다.**

### 3.4.1 두 회귀직선의 검정 (Test of Equality of Two Regression Lines)

#### (1) 완전모형 (Full Model)

$$y_{ij} = \beta_{0i} + \beta_{1i} x_{ij} + \varepsilon_{ij}$$

* $i = 1, 2$ (두 모집단)
* $\varepsilon_{ij} \sim N(0, \sigma^2)$
* 등분산성(equal variance) 가정

귀무가설:

$$H_0: \beta_{01} = \beta_{02}, \quad \beta_{11} = \beta_{12}$$

#### (2) 축소모형 (Reduced Model)

$$y_{ij} = \beta_0 + \beta_1 x_{ij} + \varepsilon_{ij}$$

#### (3) 검정통계량 (F-test)
**순서1: 완전모형 적합**  
각 모집단별로 회귀식을 개별적으로 적합하여 잔차제곱합 $SSE(F)$를 계산한다.  
$i$의 SSE를 $SSE_i$라고 하면

$$SSE(F) = SSE_1 + SSE_2$$

**순서2: 축소모형 적합**  
두 모집단을 합쳐서 하나의 회귀식을 적합하여 잔차제곱합 $SSE(R)$을 계산한다.
  
$$SSE(R) = \sum_{i=1}^2 \sum_{j=1}^{n_i} (y_{ij} - \hat{\beta}_0 - \hat{\beta}_1 x_{ij})^2$$

- $\hat{\beta}_0$, $\hat{\beta}_1$은 두 모집단을 합쳐서 추정한 회귀계수이다.

**순서3: F-통계량 계산** 
 
$$F_0 = \frac{[SSE(R) - SSE(F)]/(df_R - df_F)}{SSE(F)/df_F}$$

* 자유도 $df_F = (n_1 - 2) + (n_2 - 2) = n_1 + n_2 - 4$
* 자유도 $df_R = (n_1 - 1) + (n_2 - 1) = n_1 + n_2 - 2$
* $F_0$는 $F$ 분포를 따른다
  - 증명 step1: $SSE(F)$와 $SSE(R)$이 독립임을 보인다.
  - 증명 step2: $SSE(F)/\sigma^2$는 자유도 $n_1 + n_2 - 4$인 카이제곱분포를 따른다.
  - 증명 step3: $[SSE(R) - SSE(F)]/\sigma^2$는 자유도 2인 카이제곱분포를 따른다.
  
**순서4: 가설검정**  
판정:

$$F_0 > F_{\alpha}(2, n_1 + n_2 - 4) \Rightarrow H_0 \text{ 기각}$$

두 회귀직선이 통계적으로 유의하게 다르다고 결론 내린다.

### 3.4.2 두 기울기의 검정 (Test of Equality of Slopes)
회귀직선이 동일하지 않을 경우 기울기만 비교할 수 있다.

가설:

$$H_0: \beta_{11} - \beta_{12} = 0$$

검정통계량:

$$t_0 = \frac{\hat{\beta}_{11} - \hat{\beta}_{12}}{\sqrt{\mathrm{Var}(\hat{\beta}_{11} - \hat{\beta}_{12})}}$$

분산 (두 표본 사이에 공분산이 없다고 가정):

$$\mathrm{Var}(\hat{\beta}_{11} - \hat{\beta}_{12}) = MSE(F) \left[ \frac{1}{\sum (x_{1j} - \bar{x}_1)^2} + \frac{1}{\sum (x_{2j} - \bar{x}_2)^2} \right]$$

자유도:

$$df = n_1 + n_2 - 4$$

판정:

$$|t_0| > t_{\alpha/2}(df) \Rightarrow H_0 \text{ 기각}$$


## 3.5 이차형식의 분포 (Distribution of Quadratic Forms)
총제곱합 $y^T y$와 같은 이차형식(quadratic form)의 분포는 회귀분석에서 매우 중요하다. 특히, 잔차제곱합(SSE)과 회귀제곱합(SSR)은 이차형식으로 표현되며, 이들의 분포를 이해하는 것은 F-검정과 같은 가설검정의 근간이 된다.

예를들어, 총제곱합(총변동) $\sum (y_i - \bar{y})^2$는 $y^T y$로 표현할 수 있다. 또한, 회귀제곱합(모형변동) $\sum (\hat{y}_i - \bar{y})^2$도 이차형식으로 표현 가능하다. 잔차제곱합(오차변동) $\sum (y_i - \hat{y}_i)^2$ 역시 이차형식으로 나타낼 수 있다.

$$ 
\sum (y_i - \bar{y})^2 
= \sum y_i^2 - 2\bar{y} \sum y_i + n \bar{y}^2 = \sum y_i^2 - n \bar{y}^2 \\
= y^T y - \frac{1}{n} (1^T y)^2 = y^T \left(I_n - \frac{1}{n} 1 1^T\right) y \\
= y^T A y
$$

- 여기서 $A = I_n - \frac{1}{n} 1 1^T$는 대칭행렬(symmetric matrix)이며, 멱등행렬(idempotent matrix)이다.

### 3.5.1 다변량정규분포 (Multivariate Normal Distribution)
확률벡터 $\mathbf{y} = (y_1,\dots,y_n)^T$가 $\mathbf{y} \sim N(\mu, V)$이면 밀도함수는
밀도함수는

$$f(\mathbf{y})=(2\pi)^{-n/2}|\mathbf{V}|^{-1/2} \exp\left[-\frac{1}{2} (\mathbf{y-\mu})^T \mathbf{V}^{-1} (\mathbf{y-\mu})\right]$$

특수한 경우:
* $\mathbf{y \sim N(0, I_n)}$ 이면

     $$\mathbf{y}^T \mathbf{y} = \sum_{i=1}^n y_i^2 \sim \chi^2(n)$$

* 두 독립 카이제곱 변수 $Q_1 \sim \chi^2(n_1)$, $Q_2 \sim \chi^2(n_2)$ 이면

     $$\frac{Q_1/n_1}{Q_2/n_2} \sim F(n_1,n_2)$$

* $\mathbf{y \sim N(0,1)}$, $Q \sim \chi^2(n)$ 독립이면

     $$\frac{y}{\sqrt{Q/n}} \sim t(n)$$

### 3.5.2 비중심 χ² 및 F 분포 (Noncentral χ² and F Distributions)
만약 $\mathbf{y \sim N(\mu, I_n)}$ 이면

$$\mathbf{y}^T \mathbf{y} \sim \chi^2(n,\lambda), \quad \lambda=\frac{1}{2} \mathbf{\mu}^T\mathbf{\mu}$$

(비중심 카이제곱분포)
- $\lambda = \frac{1}{2} \mathbf{\mu}^T\mathbf{\mu}$는 비중심성 매개변수(noncentrality parameter, 비중심모수)라고 불린다.

또한 $Q_1 \sim \chi^2(n_1,\lambda)$, $Q_2 \sim \chi^2(n_2, \lambda)$이고 독립이면

$$\frac{Q_1/n_1}{Q_2/n_2} \sim F(n_1,n_2,\lambda)$$

### 3.5.3 일반 이차형식의 분포
TODO: FIXME: 다시 증명 검토해보기

#### 정리 3.1: 
$\mathbf{y \sim N(\mathbf{\mu},\mathbf{V})}$ 이면

$$E(\mathbf{y}^T \mathbf{A} \mathbf{y})=\mathrm{tr}(\mathbf{A}\mathbf{V})+\mathbf{\mu}^T \mathbf{A}\mathbf{\mu}\\
\mathrm{Cov}(\mathbf{y},\mathbf{y}^T \mathbf{A} \mathbf{y})=2\mathbf{V}\mathbf{A}\mathbf{\mu}$$

>**증명**
>
>$$
>E(\mathbf{y}\mathbf{y}^T) 
>= E[(\mathbf{y}-\mathbf{\mu})(\mathbf{y}-\mathbf{\mu})^T+\mathbf{y}\mathbf{\mu}^T+\mathbf{\mu}\mathbf{y}^T-\mathbf{\mu}\mathbf{\mu}^T] 
>= V+\mathbf{\mu}\mathbf{\mu}^T
>$$
>
>를 활용한다.
>
>$$E(\mathbf{y}^T \mathbf{A} \mathbf{y}) = E[\mathrm{tr}(\mathbf{y}^T \mathbf{A} \mathbf{y})] = E[\mathrm{tr}(\mathbf{A} \mathbf{y} \mathbf{y}^T)] = \mathrm{tr}(\mathbf{A} E[\mathbf{y}\mathbf{y}^T]) \\
>= \mathrm{tr}(\mathbf{A} (\mathbf{V} + \mathbf{\mu}\mathbf{\mu}^T)) = \mathrm{tr}(\mathbf{A}\mathbf{V}) + \mathrm{tr}(\mathbf{A}\mathbf{\mu}\mathbf{\mu}^T) = \mathrm{tr}(\mathbf{A}\mathbf{V}) + \mathbf{\mu}^T \mathbf{A}\mathbf{\mu}$$
>
>$$\mathrm{Cov}(\mathbf{y},\mathbf{y}^T \mathbf{A} \mathbf{y}) = E[(\mathbf{y} - \mathbf{\mu})(\mathbf{y}^T \mathbf{A} \mathbf{y} - E(\mathbf{y}^T \mathbf{A} \mathbf{y}))] 
>= E[(\mathbf{y} - \mathbf{\mu})\left(\mathbf{y}^T \mathbf{A} \mathbf{y} - \mathrm{tr}(\mathbf{A}\mathbf{V}) - \mathbf{\mu}^T \mathbf{A}\mathbf{\mu}\right)] \\
>= E[(\mathbf{y} - \mathbf{\mu})\left((\mathbf{y}-\mathbf{\mu}+\mathbf{\mu})^T \mathbf{A}(\mathbf{y}-\mathbf{\mu}+\mathbf{\mu}) - \mathrm{tr}(\mathbf{A}\mathbf{V}) - \mathbf{\mu}^T \mathbf{A}\mathbf{\mu}\right)] \\
>= E[(\mathbf{y} - \mathbf{\mu})\Big( (\mathbf{y}-\mathbf{\mu})^T \mathbf{A}(\mathbf{y}-\mathbf{\mu}) + (\mathbf{y}-\mathbf{\mu})^T \mathbf{A}\mathbf{\mu} + \mathbf{\mu}^T \mathbf{A}(\mathbf{y}-\mathbf{\mu}) + \mathbf{\mu}^T \mathbf{A}\mathbf{\mu} -\mathrm{tr}(\mathbf{A}\mathbf{V}) - \mathbf{\mu}^T \mathbf{A}\mathbf{\mu} \Big)]
>$$
>
>이때 $\mathbf{A}=\mathbf{A}^T$이므로, 
>$\mu^T\mathbf{A}(\mathbf{y}-\mu) =\left[\mu^T\mathbf{A}(\mathbf{y}-\mu)\right]^T =(\mathbf{y}-\mu)^T\mathbf{A}^T\mu =(\mathbf{y}-\mu)^T\mathbf{A}\mu$ 이므로 두 교차항은 같고, 
>$(\mathbf{y}-\mu)^T\mathbf{A}\mu+\mu^T\mathbf{A}(\mathbf{y}-\mu)
>=2(\mathbf{y}-\mu)^T\mathbf{A}\mu.$  
>따라서 
>
>$$
>= E[(\mathbf{y} - \mathbf{\mu})\Big( (\mathbf{y}-\mathbf{\mu})^T \mathbf{A}(\mathbf{y}-\mathbf{\mu}) - \mathrm{tr}(\mathbf{A}\mathbf{V}) + 2(\mathbf{y}-\mathbf{\mu})^T \mathbf{A}\mathbf{\mu} \Big)] \\
>= E[(\mathbf{y} - \mathbf{\mu})\left((\mathbf{y}-\mathbf{\mu})^T \mathbf{A}(\mathbf{y}-\mathbf{\mu}) - \mathrm{tr}(\mathbf{A}\mathbf{V})\right) + 2E(\mathbf{y} - \mathbf{\mu})(\mathbf{y}-\mathbf{\mu})^T \mathbf{A}\mathbf{\mu}]
>\\ = E[(\mathbf{y} - \mathbf{\mu})(\mathbf{y}-\mathbf{\mu})^T \mathbf{A}(\mathbf{y}-\mathbf{\mu})] - E[(\mathbf{y} - \mathbf{\mu})\mathrm{tr}(\mathbf{A}\mathbf{V})] + 2E(\mathbf{y} - \mathbf{\mu})(\mathbf{y}-\mathbf{\mu})^T \mathbf{A}\mathbf{\mu}]$$
>
>마지막 식의 첫 항이 0인 이유는 중심 정규분포의 3차 모멘트(변수 세개를 곱합것의 기댓값)가 원점대칭 홀함수라서 0이기 때문이다. $\mathbf{x}=\mathbf{y}-\mathbf{\mu}$ 라 두면 $\mathbf{x}\sim N(\mathbf{0},\mathbf{V})$ 이고, 성분으로 풀었을 때 $i$번째 성분은
>
>$$E[x_i(\mathbf{x}^T\mathbf{A}\mathbf{x})]
>=E\left[x_i\sum_{j,k}A_{jk}x_jx_k\right]
>=\sum_{j,k}A_{jk}E[x_ix_jx_k].$$
>
>중심 다변량 정규분포에서는 모든 홀수 차 모멘트가 0이므로 $E[x_ix_jx_k]=0$ 이다. 따라서 $E[\mathbf{x}(\mathbf{x}^T\mathbf{A}\mathbf{x})]=\mathbf{0}.$  
>두번째 항은 자명하게 0이고 세번째 항은 분산 정의에 의한다. 따라서
>
>$$
>= \mathbf{0} + \mathbf{0} + 2\mathbf{V}\mathbf{A}\mathbf{\mu}
>$$

#### 정리 3.2

$$\mathrm{Var}(\mathbf{y}^T \mathbf{A} \mathbf{y}) = 2\mathrm{tr}[(\mathbf{A}\mathbf{V})^2]+4\mathbf{\mu}^T \mathbf{A} \mathbf{V} \mathbf{A}\mathbf{\mu}$$

>**증명**
>
>$x=y-\mu$ 라 두면 $x\sim N(0,V)$
>
>그러면 $y^TAy =(x+\mu)^TA(x+\mu) =x^TAx+2x^TA\mu+\mu^TA\mu.$
>
>$\mu^TA\mu$ 는 상수이므로 $\operatorname{Var}(y^TAy) = \operatorname{Var}(x^TAx+2x^TA\mu).$  
>일반적으로 두 확률변수 $U,V$에 대해
>$\operatorname{Var}(U+V) =\operatorname{Var}(U)+\operatorname{Var}(V)+\operatorname{Cov}(U,V)+\operatorname{Cov}(V,U)$ 이므로
>
>
>$$\operatorname{Var}(y^TAy) = \operatorname{Var}(x^TAx)+4\operatorname{Var}(x^TA\mu) +4\operatorname{Cov}(x^TAx,x^TA\mu).$$
>
>이제 각 항을 계산하면 된다. 먼저 $\operatorname{Cov}(x^TAx,x^TA\mu)=0$ 이다. 실제로 $E[x^TA\mu]=0$ 이므로
>
>$$
>\begin{aligned}
>\operatorname{Cov}(x^TAx,x^TA\mu)
>&=
>E[(x^TAx)(x^TA\mu)]\\
>&=
>E\left[
>\left(\sum_{i,j}a_{ij}x_ix_j\right)
>\left(\sum_k (A\mu)_k x_k\right)
>\right]\\
>&=
>\sum_{i,j,k}a_{ij}(A\mu)_kE[x_ix_jx_k]\\
>&=0.
>\end{aligned}
>$$
>
>마지막 등식은 평균이 $0$ 인 정규분포에서 세 변수의 곱의 기댓값이 $0$ 이기 때문이다.
>
>다음으로 $x^TA\mu$ 는 스칼라이므로
>
>$$
>\operatorname{Var}(x^TA\mu) =E[(x^TA\mu)^2] =E[x^TA\mu\mu^TAx]\\
>=E[\mu^TAxx^TA\mu] =\mu^TAE[xx^T]A\mu =\mu^TAVA\mu. \\
>\therefore 4\operatorname{Var}(x^TA\mu)  = \boxed{4\mu^TAVA\mu}.
>$$
>
>이제 $\operatorname{Var}(x^TAx)$ 를 보자.  
>$E[x^TAx]=\operatorname{tr}(AV)$ 이므로, $\operatorname{Var}(x^TAx) = E[(x^TAx)^2] - [\operatorname{tr}(AV)]^2.$
>
>이제 $x^TAx=\sum_{i,j}a_{ij}x_ix_j$ 이므로
>
>$$
>\begin{aligned}
>E[(x^TAx)^2]
>&=
>E\left[
>\sum_{i,j}a_{ij}x_ix_j
>\sum_{k,l}a_{kl}x_kx_l
>\right]\\
>&=
>\sum_{i,j,k,l}
>a_{ij}a_{kl}
>E[x_ix_jx_kx_l].
>\end{aligned}
>$$
>
>평균이 $0$ 인 다변량 정규분포에서는
>
>$$
>\boxed{
>E[x_ix_jx_kx_l]
>=
>V_{ij}V_{kl}
>+
>V_{ik}V_{jl}
>+
>V_{il}V_{jk}
>}
>$$
>
>가 성립한다 (정규분포의 **4차 모멘트 공식(Isserlis/Wick 공식)**). 이를 대입하면
>
>$$
>\begin{aligned}
>E[(x^TAx)^2]
>&=
>\sum_{i,j,k,l}
>a_{ij}a_{kl}V_{ij}V_{kl}\\
>&\quad+
>\sum_{i,j,k,l}
>a_{ij}a_{kl}V_{ik}V_{jl}\\
>&\quad+
>\sum_{i,j,k,l}
>a_{ij}a_{kl}V_{il}V_{jk}.
>\end{aligned}
>$$
>
>첫 번째 항은 $\left(\sum_{i,j}a_{ij}V_{ij}\right)^2 = [\operatorname{tr}(AV)]^2.$  
>나머지 두 항은 $A,V$ 가 대칭이므로 각각 $\operatorname{tr}(AVAV) = \operatorname{tr}((AV)^2)$ 가 된다. 따라서
>
>$$
>E[(x^TAx)^2]
>= [\operatorname{tr}(AV)]^2 + 2\operatorname{tr}((AV)^2).
>$$
>
>그러므로
>
>$$
>\begin{aligned}
>\operatorname{Var}(x^TAx)
>&=
>E[(x^TAx)^2]
>-
>[E(x^TAx)]^2\\
>&=
>[\operatorname{tr}(AV)]^2
>+
>2\operatorname{tr}((AV)^2)
>-
>[\operatorname{tr}(AV)]^2\\
>&=
>\boxed{2\operatorname{tr}((AV)^2)}.
>\end{aligned}
>$$
>
>이들을 합치면
>
>$$
>\begin{aligned}
>\operatorname{Var}(y^TAy)
>&=
>\operatorname{Var}(x^TAx)
>+4\operatorname{Var}(x^TA\mu)
>+4\operatorname{Cov}(x^TAx,x^TA\mu)\\
>&=
>2\operatorname{tr}((AV)^2)
>+
>4\mu^TAVA\mu
>+0.
>\end{aligned}
>$$
>
>따라서
>
>$$
>\boxed{
>\operatorname{Var}(y^TAy)
>=
>2\operatorname{tr}((AV)^2)
>+
>4\mu^TAVA\mu
>}
>$$

#### 정리 3.3

$\mathbf y\sim N(\mu,V)$이고 $A$가 대칭행렬, $V$가 양의 정부호 행렬이라고 하자. 그러면

$$
\mathbf y^T A\mathbf y \sim \chi^2\left(r(A),\frac12\mu^T A\mu\right)
$$

가 되기 위한 필요충분조건은

$AVAV=AV$ 즉, $(AV)^2=AV$ 이어서 $AV$가 멱등행렬인 것이다.

>**증명**
>
>먼저 $V$가 양의 정부호이므로 $V^{1/2}$와 $V^{-1/2}$가 존재하여  표준정규벡터를 정의할 수 있다: $z=V^{-1/2}(y-\mu).$ 그러면 $z\sim N(0,I)$ 이고 $y=\mu+V^{1/2}z.$ 또 $\delta=V^{-1/2}\mu$ 라고 두면 $y=V^{1/2}(z+\delta).$
>
>따라서 이차형식은 $y^TAy =(z+\delta)^TV^{1/2}AV^{1/2}(z+\delta).$ 여기서 $B=V^{1/2}AV^{1/2}$ 라고 두자. $A$와 $V^{1/2}$가 대칭이므로 $B$도 대칭행렬이다. 따라서 $\boxed{y^TAy=(z+\delta)^TB(z+\delta)}$
>
>이제 **조건 $AVAV=AV$ 과 $B$의 멱등성이 동치** 임을 보이자.  
>$B^2 = V^{1/2}AV^{1/2}V^{1/2}AV^{1/2} = V^{1/2}AVAV^{1/2}.$ 따라서 $AVAV=AV$이면 $B^2 = V^{1/2}AV^{1/2}= B.$ 즉 $B$는 멱등행렬이다.  
>반대로 $B^2=B$이면 $V^{1/2}AVAV^{1/2} = V^{1/2}AV^{1/2}.$ 양변의 왼쪽에 $V^{-1/2}$, 오른쪽에 $V^{-1/2}$를 곱하면 $AVAV=AV.$  
>따라서 $\boxed{AVAV=AV\iff B^2=B}$, 
>
>이제 $B$는 대칭이며 멱등행렬이다. 대칭행렬이므로 직교행렬 $P$를 이용하여 $B=PDP^T$ 로 대각화할 수 있다.또한 $B^2=B$이므로 $B$의 고유값 $\lambda_i$는 $\lambda_i^2=\lambda_i$ 를 만족한다. 따라서 $\lambda_i\in\{0,1\}.$ 즉 $B$의 rank를 $r$이라 하면 적절히 순서를 정하여 $D=\operatorname{diag} (\underbrace{1,\ldots,1}_{r}, \underbrace{0,\ldots,0}_{n-r})$ 
>로 쓸 수 있다.  
>이제 $w=P^Tz,\quad \gamma=P^T\delta$ 라 두자. $P$는 직교행렬이고 $z\sim N(0,I)$이므로 $w\sim N(0,I).$ 따라서 $w_1,\ldots,w_n$은 서로 독립인 $N(0,1)$ 확률변수이다. 그러면
>
>$$
>\begin{aligned}
>y^TAy
>&=(z+\delta)^TB(z+\delta)\\
>&=(z+\delta)^TPDP^T(z+\delta)\\
>&=(w+\gamma)^TD(w+\gamma)\\
>&=
>\sum_{i=1}^r(w_i+\gamma_i)^2.
>\end{aligned}
>$$
>
>각각 $w_i+\gamma_i\sim N(\gamma_i,1)$ 이므로 정의에 의해 $\sum_{i=1}^r(w_i+\gamma_i)^2$ 는 자유도 $r$, 비중심성 모수 $\sum_{i=1}^r\gamma_i^2$ 인 비중심 카이제곱분포를 따른다. 이 비중심성 모수를 원래 행렬로 표현해보자:  
>$\sum_{i=1}^r\gamma_i^2 =\gamma^TD\gamma =\delta^TPDP^T\delta =\delta^TB\delta.$
>
>그런데 $\delta=V^{-1/2}\mu, \quad B=V^{1/2}AV^{1/2}$ 이므로 $\delta^TB\delta = \mu^TV^{-1/2} V^{1/2}AV^{1/2} V^{-1/2}\mu =\mu^TA\mu.$
>
>따라서  $\boxed{ y^TAy \sim \chi^2 \left(r,\frac12\mu^TA\mu\right)}$
>
>또한 $V^{1/2}$는 가역행렬이므로 $r(B)=r(V^{1/2}AV^{1/2})=r(A).$  
>따라서$r=r(A).$
>
>**왜 이것이 필요조건이기도 한가?**
>
>$B$는 대칭이므로 일반적으로 $B=P\operatorname{diag}(\lambda_1,\ldots,\lambda_n)P^T$ 라고 쓸 수 있다. 그러면 $y^TAy = \sum_{i=1}^n \lambda_i(w_i+\gamma_i)^2.$  
>즉 일반적인 정규벡터의 이차형식은 $\sum_i\lambda_i\chi_1^2(\gamma_i^2)$ 형태의 **가중된 카이제곱합**이다.
>
>이것이 하나의 일반적인 카이제곱분포가 되려면 살아 있는 항들의 계수가 모두 $1$이어야 한다. 즉 $\lambda_i$ 는 0 또는 1 이어야 한다. 따라서 $\lambda_i^2=\lambda_i$ 이고, $B^2=B.$ 
>
>앞에서 보였듯 이것은 $AVAV=AV$ 와 동치이다. 따라서 필요충분조건이 증명된다.

**단순회귀 예시**

단순선형회귀모형 $y_i=\beta_0+\beta_1x_i+\epsilon_i, \quad \epsilon_i\overset{iid}{\sim}N(0,\sigma^2)$ 를 생각하자. 벡터로 나타내면

$$
y= \begin{pmatrix}
y_1\\
\vdots\\
y_n
\end{pmatrix}
\sim N(\mu,\sigma^2I_n), \quad
\mu=
\begin{pmatrix}
\beta_0+\beta_1x_1\\
\vdots\\
\beta_0+\beta_1x_n
\end{pmatrix}
$$

즉 $V=\sigma^2I_n.$ 총제곱합은 $SST=\sum_{i=1}^n(y_i-\bar y)^2.$ 이를 행렬로 표현해보자. $\mathbf1=(1,\ldots,1)^T$이고 $J_n=\mathbf1\mathbf1^T$ 라 하면 $\bar y\mathbf1 = \frac1nJ_ny.$ 따라서 $y-\bar y\mathbf1 = \left(I_n-\frac1nJ_n\right)y.$ 여기서 $M=I_n-\frac1nJ_n$ 라고 두면

$$
SST = (y-\bar y\mathbf1)^T(y-\bar y\mathbf1).
$$

$M$은 대칭이고 멱등이므로 $M^T=M, M^2=M$ 이다. 따라서 $SST = (My)^T(My) = y^TMy$  
따라서 $\frac{SST}{\sigma^2} = y^TAy$ 에서

$$
\boxed{
A=\frac1{\sigma^2} \left(I_n-\frac1nJ_n\right) = \frac1{\sigma^2}M
}
$$

**1. $AV$가 멱등임을 확인**  
$V=\sigma^2I_n$이므로 $AV = \frac1{\sigma^2}M (\sigma^2I_n) =M.$ 따라서 $(AV)^2=M^2=M=AV.$ 즉 정리 3.3의 조건을 만족한다.

**2. $r(A)=n-1$ 증명**  
$A=\frac1{\sigma^2}M$ 이므로 $M$의 랭크와 같다. 멱등행렬 $M$의 랭크는 trace값과 같다. $\operatorname{tr}(M)= \operatorname{tr}(I_n)-\frac1n \operatorname{tr}(J_n) = n - \frac1n n= n-1$ 따라서 $r(A)=r(M)=n-1.$

**3. $\mu^TA\mu$ 계산**  
평균벡터는  $\mu = \beta_0\mathbf1+\beta_1x$ 이다. 여기서 $x=(x_1,\ldots,x_n)^T.$ 따라서 $\mu^TA\mu = \frac1{\sigma^2} \mu^TM\mu = \frac1{\sigma^2} (\beta_0\mathbf1+\beta_1x)^T M (\beta_0\mathbf1+\beta_1x).$

그런데 $M\mathbf1=0$ 이므로 절편 $\beta_0$가 포함된 항은 모두 사라진다. 따라서 $\mu^TA\mu = \frac{\beta_1^2}{\sigma^2}x^TMx.$

이제 $Mx = x-\bar x\mathbf1$ 이므로 $x^TMx = x^T\left(x-\bar x\mathbf1\right) = \sum_{i=1}^nx_i(x_i-\bar x)=\sum_{i=1}^n(x_i-\bar x)^2 =S_{xx}.$  
또는 $M^2=M$을 이용하면 더 직관적으로 $x^TMx = x^TM^TMx = (Mx)^T(Mx) = \sum_{i=1}^n(x_i-\bar x)^2 =S_{xx}$

$$
\boxed{\therefore \mu^TA\mu=\frac{\beta_1^2S_{xx}}{\sigma^2}}.
$$

- 여기서 중요한 해석은 **절편 $\beta_0$는 총제곱합의 비중심성 모수에 전혀 영향을 주지 않고, 기울기 $\beta_1$만 영향을 준다**는 것이다. $M=I-J/n$이 모든 관측치에 공통으로 더해지는 $\beta_0$ 성분을 제거하기 때문이다. 
- 총제곱합은 전체적인 평균 수준 $\beta_0$에는 영향을 받지 않는다. 총제곱합의 비중심성 모수는 $x$에 따른 평균함수의 체계적인 변화량을 나타내며, 실제 총제곱합에는 이 체계적 변화와 오차에 의한 무작위 변동이 함께 포함된다..


#### 정리 3.4 (위 예시의 특수형)

$\mathbf A$를 $n\times n$ 실대칭행렬이라고 하자.

1. $\mathbf y\sim N(\mathbf0,I_n)$이면

$$
\mathbf y^T\mathbf A\mathbf y\sim\chi^2(p)
$$

일 필요충분조건은 $\mathbf A$가 계수, 즉 랭크가 $p$인 멱등행렬인 것이다.

2. $\mathbf y\sim N(\boldsymbol\mu,\sigma^2I_n)$이면

$$
\frac{\mathbf y^T\mathbf y}{\sigma^2}
\sim
\chi^2\left(
n,\frac{\boldsymbol\mu^T\boldsymbol\mu}{2\sigma^2}
\right).
$$

3. $\mathbf y\sim N(\boldsymbol\mu,I_n)$이면

$$
\mathbf y^T\mathbf A\mathbf y
\sim
\chi^2\left(
p,\frac12\boldsymbol\mu^T\mathbf A\boldsymbol\mu
\right)
$$

일 필요충분조건은 $\mathbf A$가 계수 $p$인 멱등행렬인 것이다.

>**1번 증명**
>
>$\mathbf y\sim N(\mathbf0,I_n)$ 이므로 정리 3.3에서 $\boldsymbol\mu=\mathbf0, \mathbf V=I_n$ 으로 놓는다. 이때 정리 3.3의 멱등 조건은 $\mathbf A\mathbf V = \mathbf A I_n = \mathbf A$ 가 멱등행렬이라는 조건이 된다. 따라서
>
>$$
>(\mathbf A\mathbf V)^2=\mathbf A\mathbf V
>\iff
>\mathbf A^2=\mathbf A.
>$$
>
>또한 $r(\mathbf A)=p$라고 하면 정리 3.3에 의해
>
>$$
>\mathbf y^T\mathbf A\mathbf y \sim
>\chi^2\left(p,\frac12\mathbf0^T\mathbf A\mathbf0\right).
>$$
>
>그런데 $\frac12\mathbf0^T\mathbf A\mathbf0=0$ 이므로 $\chi^2\left(p,0\right)=\chi^2(p).$  
>따라서
>
>$$
>\mathbf y^T\mathbf A\mathbf y\sim\chi^2(p) \iff 
>\mathbf A^2=\mathbf A,\quad r(\mathbf A)=p
>$$
>
>즉 $\mathbf A$가 계수 $p$인 멱등행렬일 필요충분조건을 얻는다.
>
>**2번 증명**
>
>$\mathbf y\sim N(\boldsymbol\mu,\sigma^2I_n)$ 이고, 관심 있는 이차형식은 $\frac{\mathbf y^T\mathbf y}{\sigma^2}$ 이다. 이를 정리 3.3의 형태로 나타내면
>
>$$
>\frac{\mathbf y^T\mathbf y}{\sigma^2}
>=
>\mathbf y^T \left(\frac1{\sigma^2}I_n\right) \mathbf y.
>$$
>
>따라서 $\mathbf A=\frac1{\sigma^2}I_n, \mathbf V=\sigma^2I_n$ 으로 놓는다.
>
>그러면 $\mathbf A\mathbf V = \left(\frac1{\sigma^2}I_n\right) \left( \sigma^2I_n \right) = I_n$  
>$I_n$은 $I_n^2=I_n$ 을 만족하므로 멱등행렬이다. 또한 $\sigma^2>0$이므로 $r(\mathbf A) = r\left(\frac1{\sigma^2}I_n\right) =n$
>
>비중심성 모수는
>
>$$
>\frac12\boldsymbol\mu^T\mathbf A\boldsymbol\mu =
>\frac12 \boldsymbol\mu^T \left(\frac1{\sigma^2}I_n\right) \boldsymbol\mu = \frac{\boldsymbol\mu^T\boldsymbol\mu}{2\sigma^2}.
>$$
>
>따라서 정리 3.3에 의해
>
>$$
>\boxed{
>\frac{\mathbf y^T\mathbf y}{\sigma^2}
>\sim
>\chi^2\left(
>n,
>\frac{\boldsymbol\mu^T\boldsymbol\mu}{2\sigma^2}
>\right)
>}.
>$$
>
>여기서 주의할 점은 $\mathbf A=\frac1{\sigma^2}I_n$ 자체는 일반적으로 멱등행렬이 아니라는 것이다. 정리 3.3에서 확인해야 하는 것은 $\mathbf A$가 아니라 $\mathbf A\mathbf V=I_n$ 의 멱등성이다.
>
>**3번 증명**
>
>$\mathbf y\sim N(\boldsymbol\mu,I_n)$ 이므로 정리 3.3에서 $\mathbf V=I_n$ 으로 놓는다.
>
>그러면 $\mathbf A\mathbf V = \mathbf A I_n = \mathbf A.$
>
>따라서 정리 3.3의 조건인 $\mathbf A\mathbf V$의 멱등성은 곧 $\mathbf A$의 멱등성과 같다.
>
>이제 $r(\mathbf A)=p$라고 하면 정리 3.3에 의해
>
>$$
>\boxed{
>\mathbf y^T\mathbf A\mathbf y
>\sim
>\chi^2\left(
>p,
>\frac12\boldsymbol\mu^T\mathbf A\boldsymbol\mu
>\right)
>\iff
>\mathbf A^2=\mathbf A,\quad r(\mathbf A)=p
>}.
>$$

#### 정리 3.5

$\mathbf y\sim N(\boldsymbol\mu,\mathbf V)$이고, $\mathbf A$와 $\mathbf B$가 모두 $n\times n$ 대칭 멱등행렬이라고 하자. 그러면

$$
\mathbf y^T\mathbf A\mathbf y
\quad\text{와}\quad
\mathbf B\mathbf y
$$

가 독립일 필요충분조건은

$$
\boxed{\mathbf B\mathbf V\mathbf A=0}
$$

인 것이다.

**충분성 증명**

$\mathbf B\mathbf V\mathbf A=0$ 이라고 하자.  
$\mathbf A\mathbf y$와 $\mathbf B\mathbf y$는 정규벡터 $\mathbf y$의 선형변환이므로 결합정규분포를 따른다. 두 벡터 사이의 공분산행렬은

$$
\operatorname{Cov}(\mathbf A\mathbf y,\mathbf B\mathbf y)
= \mathbf A\operatorname{Cov}(\mathbf y)\mathbf B^T = \mathbf A\mathbf V\mathbf B^T.
$$

$\mathbf A$, $\mathbf B$, $\mathbf V$가 모두 대칭이므로 $(\mathbf B\mathbf V\mathbf A)^T = \mathbf A\mathbf V\mathbf B = \mathbf A\mathbf V\mathbf B^T=0$.  
공동정규벡터는 공분산이 $0$이면 독립이므로 $\mathbf A\mathbf y \perp\!\!\!\perp \mathbf B\mathbf y.$  
마지막으로 $\mathbf y^T\mathbf A\mathbf y = (\mathbf A\mathbf y)^T(\mathbf A\mathbf y)$ 는 $\mathbf A\mathbf y$의 함수이므로 $\mathbf y^T\mathbf A\mathbf y \perp\!\!\!\perp \mathbf B\mathbf y$

**필요성 증명**

다음과 같이 중심화한다: $\boldsymbol\varepsilon = \mathbf y-\boldsymbol\mu \sim N_n(\mathbf0, \mathbf V).$ 

$\mathbf B\boldsymbol\mu$는 상수벡터이고 $\mathbf B\boldsymbol\varepsilon = \mathbf B\mathbf y-\mathbf B\boldsymbol\mu$ 이므로 $\mathbf y^T\mathbf A\mathbf y \perp\!\!\!\perp \mathbf B\boldsymbol\varepsilon.$ 따라서 임의의 $\mathbf c\in\mathbb R^n$에 대해 $\mathbf y^T\mathbf A\mathbf y \perp\!\!\!\perp (\mathbf c^T\mathbf B\boldsymbol\varepsilon)^2.$

그러므로 $\operatorname{Cov} \left(
\mathbf y^T\mathbf A\mathbf y, (\mathbf c^T\mathbf B\boldsymbol\varepsilon)^2 \right)=0.$

한편 $\mathbf y=\boldsymbol\mu+\boldsymbol\varepsilon$ 이므로 $\mathbf y^T\mathbf A\mathbf y = \boldsymbol\mu^T\mathbf A\boldsymbol\mu + 2\boldsymbol\mu^T\mathbf A\boldsymbol\varepsilon + \boldsymbol\varepsilon^T\mathbf A\boldsymbol\varepsilon.$

따라서 공분산의 선형성에 의해

$$
0 = \operatorname{Cov}
\left(
\boldsymbol\mu^T\mathbf A\boldsymbol\mu, (\mathbf c^T\mathbf B\boldsymbol\varepsilon)^2
\right) +
\operatorname{Cov}
\left(
2\boldsymbol\mu^T\mathbf A\boldsymbol\varepsilon, (\mathbf c^T\mathbf B\boldsymbol\varepsilon)^2
\right) +
\operatorname{Cov}
\left(
\boldsymbol\varepsilon^T\mathbf A\boldsymbol\varepsilon, (\mathbf c^T\mathbf B\boldsymbol\varepsilon)^2
\right).
$$

첫 번째 항은 $\boldsymbol\mu^T\mathbf A\boldsymbol\mu$가 상수이므로 $0$이다.

두 번째 항에서 $E[\boldsymbol\mu^T\mathbf A\boldsymbol\varepsilon]=0$ 이고, $(\boldsymbol\mu^T\mathbf A\boldsymbol\varepsilon) (\mathbf c^T\mathbf B\boldsymbol\varepsilon)^2$ 은 $\boldsymbol\varepsilon$에 관한 총 3차 다항식이다. 중심정규벡터의 홀수 총차수 적률은 $0$이므로 두 번째 항도 $0$이다.  
따라서 $\operatorname{Cov} \left(\boldsymbol\varepsilon^T\mathbf A\boldsymbol\varepsilon,(\mathbf c^T\mathbf B\boldsymbol\varepsilon)^2\right)=\operatorname{Cov} \left(\boldsymbol\varepsilon^T\mathbf A\boldsymbol\varepsilon,\boldsymbol\varepsilon^T \mathbf B\mathbf c\mathbf c^T\mathbf B \boldsymbol\varepsilon\right)=0$ 이다.

중심정규벡터의 두 이차형식에 대해서는 
$\operatorname{Cov} \left(\boldsymbol\varepsilon^T\mathbf M\boldsymbol\varepsilon, \boldsymbol\varepsilon^T\mathbf N\boldsymbol\varepsilon\right) =2\operatorname{tr}(\mathbf M\mathbf V\mathbf N\mathbf V)$ 가 성립한다. 

>왜냐하면, $Q=\boldsymbol\varepsilon^T\mathbf M\boldsymbol\varepsilon, \quad R=\boldsymbol\varepsilon^T\mathbf N\boldsymbol\varepsilon
>$ 라고 하자. 그러면 $\operatorname{Var}(Q+R) = \operatorname{Var}(Q)+\operatorname{Var}(R) +2\operatorname{Cov}(Q,R)$ 중심정규벡터에 대한 정리 3.2에 의해 $\operatorname{Var} \left(\boldsymbol\varepsilon^T\mathbf M\boldsymbol\varepsilon\right) = 2\operatorname{tr}\bigl[(\mathbf M\mathbf V)^2\bigr].$
>
>따라서 $2\operatorname{Cov}(Q,R) = 2\operatorname{tr} \left[\{(\mathbf M+\mathbf N)\mathbf V\}^2\right] - 2\operatorname{tr}\bigl[(\mathbf M\mathbf V)^2\bigr] - 2\operatorname{tr}\bigl[(\mathbf N\mathbf V)^2\bigr]\\ = 2\operatorname{tr}(\mathbf M\mathbf V\mathbf N\mathbf V) + 2\operatorname{tr}(\mathbf N\mathbf V\mathbf M\mathbf V).$
>
>Trace의 순환성에 의해 $\operatorname{tr}(\mathbf N\mathbf V\mathbf M\mathbf V) = \operatorname{tr}(\mathbf M\mathbf V\mathbf N\mathbf V)$ 이므로 $2\operatorname{Cov}(Q,R) = 4\operatorname{tr}(\mathbf M\mathbf V\mathbf N\mathbf V).$


따라서 $0 = 2\operatorname{tr} \left( \mathbf A\mathbf V \mathbf B\mathbf c\mathbf c^T\mathbf B \mathbf V \right) = 2\mathbf c^T \mathbf B\mathbf V\mathbf A\mathbf V\mathbf B \mathbf c.$  
$\mathbf A$가 대칭 멱등행렬이므로 $\mathbf A=\mathbf A^T\mathbf A.$ 따라서

$$
\mathbf c^T \mathbf B\mathbf V\mathbf A\mathbf V\mathbf B \mathbf c
= \mathbf c^T \mathbf B\mathbf V\mathbf A^T\mathbf A \mathbf V\mathbf B\mathbf c =(\mathbf A\mathbf V\mathbf B\mathbf c)^T (\mathbf A\mathbf V\mathbf B\mathbf c) = \|\mathbf A\mathbf V\mathbf B\mathbf c\|^2.
$$

그러므로 모든 $\mathbf c\in\mathbb R^n$에 대해 $\|\mathbf A\mathbf V\mathbf B\mathbf c\|^2=0.$ 이고, $\mathbf A\mathbf V\mathbf B=0.$  
이를 전치하면 $\mathbf B\mathbf V\mathbf A=0$ 이다.


#### 정리 3.6
$\mathbf y\sim N(\mu,V)$ 일때, 두 이차형식 $\mathbf{y}^T \mathbf{A} \mathbf{y}$, $\mathbf{y}^T \mathbf{B} \mathbf{y}$ 독립 $\iff \mathbf{A}\mathbf{V}\mathbf{B}=0$

>**증명**
>
>대칭성에 의해 $\mathbf B\mathbf V\mathbf A=0 \iff (\mathbf B\mathbf V\mathbf A)^T = \mathbf A\mathbf V\mathbf B=0.$  
>정리 3.5에 의해 $\mathbf y^T\mathbf A\mathbf y \perp\!\!\!\perp \mathbf B\mathbf y.$ 그리고 $\mathbf y^T\mathbf B\mathbf y$ 는 $\mathbf B\mathbf y$ 의 함수이므로 $\mathbf y^T\mathbf A\mathbf y \perp\!\!\!\perp \mathbf y^T\mathbf B\mathbf y.$  
>이것으로 충분성이 바로 증명된다.
>
>필요성에서는 $\mathbf y^T\mathbf A\mathbf y \perp\!\!\!\perp \mathbf y^T\mathbf B\mathbf y$ 로부터 곧바로 $\mathbf y^T\mathbf A\mathbf y \perp\!\!\!\perp \mathbf B\mathbf y$ 라고 하면 안 된다. 일반적으로 어떤 확률변수 $X$ 가 $g(Z)$ 와 독립이라고 해서 $X$ 가 $Z$ 와 독립인 것은 아니기 때문이다. 필요성에는 위에서 사용한 공동정규벡터의 제곱노름 보조결과를 사용하면,
>
>$$
>\begin{aligned}
>\mathbf y^T\mathbf A\mathbf y
>\perp\!\!\!\perp
>\mathbf y^T\mathbf B\mathbf y
>&\iff
>\|\mathbf A\mathbf y\|^2
>\perp\!\!\!\perp
>\|\mathbf B\mathbf y\|^2\\
>&\iff
>\mathbf A\mathbf y
>\perp\!\!\!\perp
>\mathbf B\mathbf y\\
>&\iff
>\mathbf A\mathbf V\mathbf B=0.
>\end{aligned}
>$$
>
>다만 가운데 동치는 단순한 "함수의 독립성" 만으로 성립하는 것이 아니라, $\mathbf A\mathbf y,\mathbf B\mathbf y$ 가 공동정규벡터라는 특수한 성질을 사용한 결과다. 이 보조결과를 앞에서 증명하지 않았다면 정리 3.6의 필요성은 별도로 적률생성함수 등을 사용해 증명해야 한다.

#### 정리 3.7

$\mathbf y\sim N_n(\boldsymbol\mu,\mathbf V), \mathbf A=\sum_{j=1}^p\mathbf A_j$ 이고, 각 $\mathbf A_j$가 대칭행렬이며 $r(\mathbf A_j)=k_j, r(\mathbf A)=k$ 라고 하자.

다음 세 가지가 모두 성립하기 위한 필요충분조건을 생각한다.

$$
\mathbf y^T\mathbf A_j\mathbf y \sim \chi^2
\left(
k_j,\frac12\boldsymbol\mu^T\mathbf A_j\boldsymbol\mu
\right),
\qquad j=1,\ldots,p,
$$

$$
\mathbf y^T\mathbf A_1\mathbf y,\ldots,
\mathbf y^T\mathbf A_p\mathbf y
\quad\text{가 서로 독립이고},
$$

$$
\mathbf y^T\mathbf A\mathbf y
\sim
\chi^2
\left(
k,\frac12\boldsymbol\mu^T\mathbf A\boldsymbol\mu
\right).
$$

그 필요충분조건은 다음 두 조건 중 하나다.

1. 다음 세 조건 중 어느 두 조건이 성립한다.

$$
(C_1):\quad
\mathbf A_j\mathbf V
\text{가 모든 }j\text{에 대해 멱등이다},
$$

$$
(C_2):\quad
\mathbf A_i\mathbf V\mathbf A_j=0
\quad\text{for every }i<j,
$$

$$
(C_3):\quad
\mathbf A\mathbf V
\text{가 멱등이다}.
$$

2. $\mathbf A\mathbf V$가 멱등이고

$$
\sum_{j=1}^p k_j=k
$$

이다.

>**증명** 
>
>정리 3.3에 의해 $\mathbf y^T\mathbf A_j\mathbf y \sim \chi^2 \left( k_j,\frac12\boldsymbol\mu^T\mathbf A_j\boldsymbol\mu \right)$ 일 필요충분조건은 $\mathbf A_j\mathbf V$ 가 멱등인 것이다. 따라서 각 이차형식의 카이제곱분포 조건은 정확히 $(C_1)$이다.
>
>정리 3.6에 의해 $i\neq j$일 때 $\mathbf y^T\mathbf A_i\mathbf y \perp\!\!\!\perp \mathbf y^T\mathbf  A_j\mathbf y$ 일 필요충분조건은 $\mathbf A_i\mathbf V\mathbf A_j=0$ 인 것이다. 따라서 이차형식들의 독립성 조건은 $(C_2)$이다.
>
>마지막으로 정리 3.3에 의해 $\mathbf y^T\mathbf A\mathbf y \sim \chi^2 \left(k,\frac12\boldsymbol\mu^T\mathbf A\boldsymbol\mu \right)$ 일 필요충분조건은 $\mathbf A\mathbf V$ 가 멱등인 것이다. 따라서 전체 이차형식의 카이제곱분포 조건은 $(C_3)$이다.
>
>그러므로 정리에서 요구하는 모든 결론이 성립할 필요충분조건은 $C_1$, $C_2$,$C_3$ 이 모두 성립하는 것이다.
>
>이제 세 조건 중 어느 두 조건이 성립하면 나머지 하나도 성립함을 보이면 된다.
>
>---
>
>**공분산행렬의 표준화**
>
>다음과 같이 정의한다: $\mathbf C_j = \mathbf V^{1/2}\mathbf A_j\mathbf V^{1/2}, \quad \mathbf C = \mathbf V^{1/2}\mathbf A\mathbf V^{1/2} = \sum_{j=1}^p{\mathbf C_j}.$
>
>각 $\mathbf A_j$가 대칭이므로 $\mathbf C_j$도 대칭이다. 또한 $\mathbf V^{1/2}$이 가역이므로 $r(\mathbf C_j)=r(\mathbf A_j)=k_j, \quad r(\mathbf C)=r(\mathbf A)=k.$
>
>그리고 다음 동치가 성립한다.
>
>$$
>(C_1)
>\iff
>\mathbf C_j^2=\mathbf C_j
>\quad\text{for every }j,
>$$
>
>$$
>(C_2)
>\iff
>\mathbf C_i\mathbf C_j=0
>\quad\text{for every }i<j,
>$$
>
>$$
>(C_3)
>\iff
>\mathbf C^2=\mathbf C.
>$$
>
>따라서 $(C_1),(C_2),(C_3)$ 대신 각각
>
>* 각 $\mathbf C_j$의 멱등성,
>* 서로 다른 $\mathbf C_i,\mathbf C_j$의 직교성,
>* $\mathbf C=\sum_j\mathbf C_j$의 멱등성
>
>을 증명하면 된다.
>
>---
>
>**세 조건 중 어느 두 조건이면 나머지도 성립한다**
>
>1. $(C_1)+(C_2)\Rightarrow(C_3)$ 증명
>
>각 $\mathbf C_j$가 멱등이고 서로 다른 $i,j$에 대해 $\mathbf C_i\mathbf C_j=0$ 이라고 하자. 그러면
>
>$$
>\mathbf C^2 = \left(\sum_{j=1}^p\mathbf C_j\right)^2
>= \sum_{j=1}^p\mathbf C_j^2 + \sum_{i\neq j}\mathbf C_i\mathbf C_j
>= \sum_{j=1}^p\mathbf C_j = \mathbf C.
>$$
>
>따라서 $(C_3)$이 성립한다.
>
>2. $(C_1)+(C_3)\Rightarrow(C_2)$ 증명
>
>각 $\mathbf C_j$와 $\mathbf C$가 멱등이라고 하자. $\mathbf C^2=\mathbf C$ 에 trace를 취하면 $\operatorname{tr}(\mathbf C^2) = \operatorname{tr}(\mathbf C).$
>
>한편
>
>$$
>\begin{aligned}
>\operatorname{tr}(\mathbf C^2)
>&=
>\sum_{j=1}^p\operatorname{tr}(\mathbf C_j^2)
>+
>2\sum_{i<j}\operatorname{tr}(\mathbf C_i\mathbf C_j)\\
>&=
>\sum_{j=1}^p\operatorname{tr}(\mathbf C_j)
>+
>2\sum_{i<j}\operatorname{tr}(\mathbf C_i\mathbf C_j),
>\end{aligned}
>$$
>
>이고 $\operatorname{tr}(\mathbf C^2) = \operatorname{tr}(\mathbf C) = \sum_{j=1}^p\operatorname{tr}(\mathbf C_j).$ 따라서 $\sum_{i<j}\operatorname{tr}(\mathbf C_i\mathbf C_j)=0.$
>
>대칭 멱등행렬에 대해서는
>
>$$
>\begin{aligned}
>\operatorname{tr}(\mathbf C_i\mathbf C_j)
>&= \operatorname{tr} \left[ (\mathbf C_i\mathbf C_j)^T (\mathbf C_i\mathbf C_j) \right]\\
>&= \operatorname{tr} \left[ \mathbf C_j^T\mathbf C_i\mathbf C_j \right]\\
>&= \operatorname{tr} \left[ \mathbf C_i\mathbf C_j^T\mathbf C_j \right] &\because\text{trace 순환성}\\
>&=
>\|\mathbf C_i\mathbf C_j\|_F^2
>\geq0.
>\end{aligned}
>$$
>
>따라서 각 항이 모두 $0$이어야 하므로 $(C_2)$가 성립한다.
>
>3. $(C_2)+(C_3)\Rightarrow(C_1)$
>
>서로 다른 $i,j$에 대해 $\mathbf C_i\mathbf C_j=0$ 이고 $\mathbf C$가 멱등이라고 하자. $\mathbf C_i,\mathbf C_j$는 대칭이므로 $\mathbf C_j\mathbf C_i = (\mathbf C_i\mathbf C_j)^T = 0.$ 따라서 서로 다른 행렬들의 열공간은 서로 직교한다.
>
>또한 $0 = \mathbf C^2-\mathbf C = \sum_{j=1}^p (\mathbf C_j^2-\mathbf C_j)$ 이다. 각 벡터 $(\mathbf C_j^2-\mathbf C_j)\mathbf x$ 는 $\mathcal R(\mathbf C_j)$에 속하고, 이 열공간들은 서로 직교한다.
>
>따라서 모든 $\mathbf x$에 대해 직교하는 벡터들의 합이 $0$이므로 각각이 $0$이어야 한다.
>
>$$
>(\mathbf C_j^2-\mathbf C_j)\mathbf x=0
>\quad\text{for every }\mathbf x.
>$$
>
>그러므로 $\mathbf C_j^2=\mathbf C_j$ 이고, $(C_1)$이 성립한다.
>
>---
>
>**조건 2와의 동치**
>
>1. $(C_1),(C_2),(C_3)\Rightarrow$ 조건 2
>
>$(C_1)$과 $(C_2)$에 의해 $\mathbf C_j$들은 서로 직교하는 대칭 멱등행렬이다. 따라서 열공간들의 합은 직합이고
>
>$$
>r(\mathbf C) = \sum_{j=1}^p r(\mathbf C_j).
>$$
>
>즉, $k=\sum_{j=1}^p k_j.$
>
>또한 $(C_3)$에 의해 $\mathbf A\mathbf V$는 멱등이다. 따라서 조건 2가 성립한다.
>
>2. 조건 2 $\Rightarrow(C_1),(C_2),(C_3)$
>
>조건 2를 가정하면 $\mathbf C^2=\mathbf C = \mathbf V^{1/2}\mathbf A\mathbf V^{1/2}$ 이고 $r(\mathbf C) = \sum_{j=1}^p r(\mathbf C_j).$
>
>$\mathcal R$을 열공간/치역이라 하면, 항상 $\mathcal R(\mathbf C) \subseteq \mathcal R(\mathbf C_1)+\cdots+\mathcal R(\mathbf C_p)$ 이고,
>
>$$
>\dim\mathcal R(\mathbf C) =
>r(\mathbf C) = \sum_{j=1}^p r(\mathbf C_j).
>$$
>
>한편
>
>$$
>\dim\left(
>\mathcal R(\mathbf C_1)+\cdots+\mathcal R(\mathbf C_p)
>\right)
>\leq
>\sum_{j=1}^p r(\mathbf C_j).
>$$
>
>따라서
>
>$$
>\boxed{
>\mathcal R(\mathbf C)
>=
>\mathcal R(\mathbf C_1)
>\oplus\cdots\oplus
>\mathcal R(\mathbf C_p)
>}
>$$
>
>이다.
>
>임의의 $\mathbf x\in\mathcal R(\mathbf C_i)$를 택한다. 그러면 $\mathbf C$가 멱등이므로 $\mathbf x\in\mathcal R(\mathbf C) \iff \mathbf C\mathbf x=\mathbf x.$
>
>그런데
>
>$$
>\mathbf C\mathbf x = \sum_{j=1}^p\mathbf C_j\mathbf x,
>\qquad \mathbf C_j\mathbf x\in\mathcal R(\mathbf C_j).
>$$
>
>위의 합은 직합이므로 표현의 유일성에 의해
>
>$$
>\mathbf C_i\mathbf x=\mathbf x, \quad \mathbf C_j\mathbf x=0 \quad(j\neq i).
>$$
>
>모든 $\mathbf u\in\mathbb R^n$에 대해 $\mathbf C_i\mathbf u\in\mathcal R(\mathbf C_i)$ 이므로 $\mathbf C_i^2\mathbf u = \mathbf C_i\mathbf u.$
>
>따라서 $\mathbf C_i^2=\mathbf C_i,$ 즉 $(C_1)$이 성립한다.
>
>또한 $i\neq j$이면 $\mathbf C_j\mathbf u\in\mathcal R(\mathbf C_j)$이므로 $\mathbf C_i\mathbf C_j\mathbf u=0$ 이다. 따라서 $\mathbf C_i\mathbf C_j=0,$ 즉 $(C_2)$가 성립한다. $(C_3)$은 처음부터 가정되어 있다.
>
>따라서 조건 2는 $(C_1),(C_2),(C_3)$ 모두를 함의한다.
>
#### 정리 3.8 (Cochran, 코크란 정리)
$\mathbf y\sim N(\mathbf0,I_n)$이고, $\mathbf A_1,\ldots,\mathbf A_p$가 $n\times n$ 대칭행렬이라고 하자. 또한

$$
\sum_{j=1}^p\mathbf A_j=I_n, \qquad r(\mathbf A_j)=k_j
$$

라고 하자. 그러면

$$
\mathbf y^T\mathbf A_j\mathbf y\sim\chi^2(k_j), \qquad j=1,\ldots,p
$$

이고 이 이차형식들이 서로 독립일 필요충분조건은

$$
\boxed{\sum_{j=1}^p k_j=n}
$$

>**증명**
>
>정리 3.7에서 $\boldsymbol\mu=\mathbf0, \mathbf V=I_n, \mathbf A=\sum_{j=1}^p\mathbf A_j$ 로 놓는다. 가정에 의해 $\mathbf A = \sum_{j=1}^p\mathbf A_j =I_n$  
>따라서 $\mathbf A\mathbf V = I_nI_n = I_n$ 이다. $I_n$은 $I_n^2=I_n$ 을 만족하므로 멱등행렬이고, $r(\mathbf A)=r(I_n)=n.$
>
>정리 3.7의 두 번째 필요충분조건은 $\mathbf A\mathbf V$ 가 멱등행렬이고 $\sum_{j=1}^p k_j=r(\mathbf A)$ 인 것이다. 현재 $\mathbf A\mathbf V=I_n$의 멱등성은 자동으로 성립하고 $r(\mathbf A)=n$이므로, 정리 3.7의 조건은 $\sum_{j=1}^p k_j=n$ 으로 축약된다.
>
>따라서 정리 3.7에 의해 $\sum_{j=1}^p k_j=n$ 일 필요충분조건은
>
>$$
>\mathbf y^T\mathbf A_j\mathbf y
>\sim \chi^2\left( k_j, \frac12\mathbf0^T\mathbf A_j\mathbf0 \right)
>$$
>
>이고 이 이차형식들이 서로 독립인 것이다.  
>그런데 $\frac12\mathbf0^T\mathbf A_j\mathbf0=0$ 이므로 비중심 카이제곱분포는 중심 카이제곱분포가 된다. 즉 $\chi^2(k_j,0)=\chi^2(k_j).$
>
>따라서
>
>$$
>\boxed{
>\sum_{j=1}^p k_j=n
>\iff
>\begin{cases}
>\mathbf y^T\mathbf A_j\mathbf y\sim\chi^2(k_j),
>&j=1,\ldots,p,\\[2mm]
>\mathbf y^T\mathbf A_1\mathbf y,\ldots,
>\mathbf y^T\mathbf A_p\mathbf y
>\text{가 서로 독립이다}.
>\end{cases}
>}
>$$
>
>이다.
>
>필요성은 분포의 합을 이용해서도 확인할 수 있다. 이차형식들이 서로 독립이고 각각 $\chi^2(k_j)$를 따른다면
>
>$$
>\sum_{j=1}^p\mathbf y^T\mathbf A_j\mathbf y
>\sim
>\chi^2\left(\sum_{j=1}^p k_j\right).
>$$
>
>한편
>
>$$
>\begin{aligned}
>\sum_{j=1}^p\mathbf y^T\mathbf A_j\mathbf y
>&=
>\mathbf y^T
>\left(\sum_{j=1}^p\mathbf A_j\right)
>\mathbf y\\
>&=
>\mathbf y^TI_n\mathbf y\\
>&=
>\mathbf y^T\mathbf y.
>\end{aligned}
>$$
>
>그리고 $\mathbf y\sim N(\mathbf0,I_n)$이므로 $\mathbf y^T\mathbf y\sim\chi^2(n).$
>
>따라서 $\chi^2\left(\sum_{j=1}^p k_j\right) = \chi^2(n)$ 이어야 하므로 자유도가 같아야 한다.
>
>$$
>\boxed{\sum_{j=1}^p k_j=n}.
>$$
>
>즉 Cochran의 정리는 정리 3.7에서 $\boldsymbol\mu=\mathbf0$, $\mathbf V=I_n$, $\mathbf A=I_n$으로 둔 특수한 경우다.


## 3.6 평균제곱의 기대값 (Expected Mean Squares)
회귀분석의 제곱합, 평균제곱, SSR, SSE, SST, MSR, MSE등은 모두 이차형식(quadratic form)으로 표현할 수 있다. 따라서 이들의 기대값을 구하기 위해서는 이차형식의 기대값을 구하는 문제로 귀결된다.

회귀직선 대체모형:

$$y_i=\beta_0'+\beta_1(x_i-\bar{x})+\varepsilon_i, \quad \varepsilon_i\sim N(0,\sigma^2)$$

오차분산 $\sigma^2$는 다음과 같이 정의할 수 있다: 
$\bar\varepsilon = \sum \varepsilon_i / n$ 이면

$$\sigma^2 = E\left[\frac{1}{n-1} \sum(\varepsilon_i - \bar\varepsilon)^2\right]$$

- $\bar y = \beta_0'+\bar{\varepsilon}$  
- $Var(\bar y) = Var(\bar\varepsilon) = \sigma^2 / n$
- $\hat\beta_0' = \bar y$
- $\operatorname{Var}(\hat\beta_1') = E(\hat\beta^2_1)-\beta^2_1$

### 제곱합 기대값

$$
\begin{aligned}
E(SST) 
&= E\left(\sum (y_i - \bar{y})^2\right) \\
&= E\left(\sum (\beta_0'+\beta_1(x_i-\bar{x})+\varepsilon_i - (\beta_0'+\bar{\varepsilon}))^2\right) \\
&= E\left(\sum (\beta_1^2(x_i-\bar{x})^2+(\varepsilon_i-\bar{\varepsilon})^2)\right) \\
&= E(\beta_1^2S_{xx})+E\left[\sum(\varepsilon_i-\bar{\varepsilon})^2\right]\\
&= \beta_1^2S_{xx}+(n-1)\sigma^2
\end{aligned}
$$

> 고전적 선형회귀모형에서는 보통 $E(\varepsilon \mid X)=0 $ 을 가정한다. 여기서 $X=(x_1,\dots,x_n)$이다. $x_i$들을 주어진 값으로 조건화하면 $E\left[ \sum_{i=1}^n(x_i-\bar x)\varepsilon_i \,\middle|\,X \right]
= \sum_{i=1}^n(x_i-\bar x) E(\varepsilon_i\mid X) =0.$

한편, 절편이 포함된 OLS에서는 정규방정식에 의해 잔차의 합이 0이므로 (잔차는 $e$, 오차항은 $\varepsilon$) $\sum_{i=1}^n e_i=0$ 이다. 그런데 $y_i=\hat y_i+ e_i$이므로 표본평균을 취하면 $\bar y=\bar{\hat y}+\bar{e}=\bar{\hat y}.$

한편 $\bar{\hat y}=\frac{1}{n}\sum_{i=1}^n(\hat\beta_0+\hat\beta_1x_i)
=\hat\beta_0+\hat\beta_1\bar x$ 이므로 역시 $\bar y=\hat\beta_0+\hat\beta_1\bar x.$

$$
\begin{aligned}
E(SSR) &= E\left(\sum (\hat{y}_i - \bar{y})^2\right) \\
&= E\left(\sum \left(\hat\beta_0+\hat\beta_1 x_i-(\hat\beta_0+\hat\beta_1\bar x)\right)^2 \right) \\
&= E\left(\sum (\hat\beta_1 (x_i - \bar{x}))^2\right) \\
&= S_{xx} E(\hat\beta_1^2) = S_{xx} \left[ Var(\hat\beta_1) + (E(\hat\beta_1))^2 \right] \\
&= S_{xx} \left[ \frac{\sigma^2}{S_{xx}} + \beta_1^2 \right] =\sigma^2+\beta_1^2 S_{xx}
\end{aligned}
$$

뒤에서 두번째 등식은 2.1.1의 결과로 유도했다. 따라서
 
$$
\begin{aligned}
E(SSE) &= E(SST) - E(SSR) = (n-2)\sigma^2 \\
E(MSR) &= E(SSR)/1 =\sigma^2+\beta_1^2 S_{xx}\\
E(MSE) &=E(SSE)/(n-2)=\sigma^2
\end{aligned}
$$

### F-검정

$$F_0=\frac{MSR}{MSE}$$

가설:

$$H_0:\beta_1=0 \quad H_1:\beta_1\neq0$$

| 요인 | 제곱합 | 자유도 | 평균제곱 | 평균제곱의 기대값 |
|------|--------|--------|---------|-------------------|
| 회귀 | SSR | 1 | MSR = SSR/1 | $\sigma^2 + \beta_1^2 S_{xx}$ |
| 잔차 | SSE | n-2 | MSE = SSE/(n-2) | $\sigma^2$ |
| 계 | SST | n-1 | | |


## 3.7 반복측정값의 회귀분석
각 $x_i$에서 $n_i$회 반복:

$$y_{ij}=\beta_0+\beta_1 x_i+\varepsilon_{ij}$$

2장에서는 적합결여검정을 위한 F-검정만 하였으므로 여기서는 추정문제를 살펴보자.

기호: $n=\sum_i n_i, \quad T_i=\sum_j y_{ij}, \quad T=\sum_i T_i, \quad \bar x = \frac{\sum_i n_i x_i}{n}$

**제곱합**

$$S_{xx}=\sum_i n_i(x_i-\bar{x})^2 = \sum_i n_i x_i^2 - \frac{(\sum_i n_i x_i)^2}{n} \\
S_{yy}=\sum_{i,j}(y_{ij}-\bar{y})^2 = \sum_i \sum_j y_{ij}^2 - \frac{T^2}{n} \\
S_{xy}=\sum_i n_i (x_i-\bar{x})(\bar{y}_i-\bar{y}) = \sum_i x_i T_i-\frac{(\sum_i n_i x_i)T}{n}
$$

**추정량**

$$\hat\beta_1=\frac{S_{xy}}{S_{xx}}, \quad \hat\beta_0=\bar{y}-\hat\beta_1\bar{x}$$

**적합결여 분해**

$$
SST = S_{yy}, \quad
SSR = S_{xy}^2 / S_{xx} \\
SSE = SST - SSR = SSPE + SSLF
$$

* SSPE: 순오차(pure error)
* SSLF: 적합결여(lack of fit)

F-통계량:

$$F_0=\frac{MSLF}{MSPE}$$

**가설검정:**

$$H_0: \text{선형모형이 적절} \quad H_1: \text{선형모형이 부적절}$$

판정: $F_0 > F_\alpha(m-2, n-m)$ 이면 $H_0$ 기각
**적합결여검정의 분산분석표 (ANOVA Table for Lack of Fit Test)**
| 요인      | 제곱합  | 자유도 | 평균제곱              | F-통계량                     |
| ------- | ---- | --- | ----------------- | ------------------------- |
| 회귀      | SSR  | 1   | MSR = SSR         |                           |
| 잔차      | SSE  | n-2 | MSE = SSE/(n-2)   |                           |
| ├─ 순오차  | SSPE | n-m | MSPE = SSPE/(n-m) |                           |
| └─ 적합결여 | SSLF | m-2 | MSLF = SSLF/(m-2) | $F_0 = \frac{MSLF}{MSPE}$ |
| 계       | SST  | n-1 |                   |                           |


## 3.8 고차원 회귀에서의 단순선형회귀
설명변수가 매우 많은 초고차원(ultra high dimensional) 상황에서

* 변수선별(screening) 절차 필요
* 단순선형회귀가 1차 스크리너로 사용 가능
  - 각 설명변수 $x_j$에 대해 단순선형회귀 적합

    $$y_i = \beta_{0j} + \beta_{1j} x_{ij} + \varepsilon_{ij}$$

  - $|\hat\beta_{1j}|$가 큰 변수들을 선별하여 다변량회귀에 포함
  
고차원 환경에서 이론적으로도 단순회귀 기반 screening이 유효함이 알려져 있다.
