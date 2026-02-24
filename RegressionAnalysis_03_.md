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
* $x$ 증가에 따라 $y$가 일정 값으로 수렴(convergence)하는 곡선(curvilinear relationship)을 보일 때 적절하다.

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
$$\mathrm{Cov}(x_i^*, \varepsilon_i - \beta_1 \delta_i) \\
= E[(x_i^* - E(x_i^*))( \varepsilon_i - \beta_1 \delta_i - E(\varepsilon_i - \beta_1 \delta_i))]$$
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


## 3.3 x의 수준 선택 (Choice of Levels of X)
실험계획법(design of experiments)의 관점에서 설명변수 $x$의 관측수준(level)을 어떻게 선택할 것인지가 중요한 문제이다.

- 정해진 $x$값에서 몇 번 $y$를 관측할 것인가?
- $x$값을 어떻게 분포시킬 것인가? 등등

### 3.3.1 실험 목적
1. 기울기 $\beta_1$을 정확히 추정
2. 절편 $\beta_0$을 정확히 추정
3. 평균반응 $E(Y|X=x)$ 예측
4. 개별반응 예측

관련 분산식:
$$\mathrm{Var}(\hat{\beta}_1) = \sigma^2 \left[ \frac{1}{\sum (x_i - \bar{x})^2} \right]\\
\mathrm{Var}(\hat{\beta}_0) = \sigma^2 \left[ \frac{1}{n} + \frac{\bar{x}^2}{\sum (x_i - \bar{x})^2} \right]\\
\mathrm{Var}(\hat{y}) = \sigma^2 \left[ \frac{1}{n} + \frac{(x_i - \bar{x})^2}{\sum (x_i - \bar{x})^2} \right]\\
\mathrm{Var}(\hat{y}_0) = \sigma^2 \left[ 1 + \frac{1}{n} + \frac{(x_i - \bar{x})^2}{\sum (x_i - \bar{x})^2} \right]$$

### 3.3.2 기울기 추정 정밀화
기울기 분산 최소화를 위해서는
$$S_{xx} = \sum (x_i - \bar{x})^2$$
를 최대화해야 한다.

실험영역(experimental region) $[x_a, x_b]$에서
* $n/2$를 $x_a$
* $n/2$를 $x_b$ 에 배치하면 $S_{xx}$가 최대가 된다.
* 즉, $x$의 수준을 양 끝점에 배치하는 것이 기울기 추정 정밀화에 가장 효과적이다.
* 하지만, 이런 극단적 배치는 비선형성(nonlinearity) 가능성을 간과할 수 있다.
* 비선형 가능성이 있는 경우에는 최소 세 수준을 사용하는 것이 바람직하다.

### 3.3.3 절편 추정 정밀화
절편의 분산을 줄이려면 $\bar{x} = 0$이 되도록 설계하는 것이 이상적이다.

### 3.3.4 예측 정확도 향상
관심 있는 $x$ 값이 $\bar{x}$에 가까울수록 예측분산이 작다.


## 3.4 두 회귀선의 비교 (Comparison of Two Regression Lines)
두 모집단(population)에 대해 회귀직선이 동일한지 검정하는 문제이다.

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
= \sum y_i^2 - 2\bar{y} \sum y_i + n \bar{y}^2 \\
= \sum y_i^2 - n \bar{y}^2 \\
= y^T y - \frac{1}{n} (1^T y)^2 \\
= y^T \left(I_n - \frac{1}{n} 1 1^T\right) y \\
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

#### 정리 3.1: 
$\mathbf{y \sim N(\mathbf{\mu},\mathbf{V})}$ 이면
$$E(\mathbf{y}^T \mathbf{A} \mathbf{y})=\mathrm{tr}(\mathbf{A}\mathbf{V})+\mathbf{\mu}^T \mathbf{A}\mathbf{\mu}\\
\mathrm{Cov}(\mathbf{y},\mathbf{y}^T \mathbf{A} \mathbf{y})=2\mathbf{V}\mathbf{A}\mathbf{\mu}$$

증명:  
$$E(\mathbf{y}^T \mathbf{A} \mathbf{y}) = E[\mathrm{tr}(\mathbf{y}^T \mathbf{A} \mathbf{y})] = E[\mathrm{tr}(\mathbf{A} \mathbf{y} \mathbf{y}^T)] = \mathrm{tr}(\mathbf{A} E[\mathbf{y}\mathbf{y}^T]) \\
= \mathrm{tr}(\mathbf{A} (\mathbf{V} + \mathbf{\mu}\mathbf{\mu}^T)) = \mathrm{tr}(\mathbf{A}\mathbf{V}) + \mathrm{tr}(\mathbf{A}\mathbf{\mu}\mathbf{\mu}^T) = \mathrm{tr}(\mathbf{A}\mathbf{V}) + \mathbf{\mu}^T \mathbf{A}\mathbf{\mu}$$

$$\mathrm{Cov}(\mathbf{y},\mathbf{y}^T \mathbf{A} \mathbf{y}) = E(\mathbf{y} - \mathbf{\mu})(\mathbf{y}^T \mathbf{A} \mathbf{y} - E(\mathbf{y}^T \mathbf{A} \mathbf{y})) \\
= E(\mathbf{y} - \mathbf{\mu})(\mathbf{y}^T \mathbf{A} \mathbf{y} - \mathrm{tr}(\mathbf{A}\mathbf{V}) - \mathbf{\mu}^T \mathbf{A}\mathbf{\mu}) \\
= E(\mathbf{y} - \mathbf{\mu})(\mathbf{(y- \mu)}^T \mathbf{A} \mathbf{(y- \mu)} - \mathrm{tr}(\mathbf{A}\mathbf{V}) + 2\mathbf{(y- \mu)}^T \mathbf{A}\mathbf{\mu}) \\
= E(\mathbf{y} - \mathbf{\mu})(\mathbf{(y- \mu)}^T \mathbf{A} \mathbf{(y- \mu)} - \mathrm{tr}(\mathbf{A}\mathbf{V})) + 2E(\mathbf{y} - \mathbf{\mu})\mathbf{(y- \mu)}^T \mathbf{A}\mathbf{\mu} \\
= 2\mathbf{V}\mathbf{A}\mathbf{\mu}
$$

#### 정리 3.2
$$\mathrm{Var}(\mathbf{y}^T \mathbf{A} \mathbf{y}) = 2\mathrm{tr}(\mathbf{A}\mathbf{V})^2+4\mathbf{\mu}^T \mathbf{A} \mathbf{V} \mathbf{A}\mathbf{\mu}$$

#### 정리 3.3
$$\mathbf{y}^T \mathbf{A} \mathbf{y} \sim \chi^2(r(\mathbf{A}),\tfrac{1}{2} \mathbf{\mu}^T \mathbf{A}\mathbf{\mu})$$
가 되기 위한 필요충분조건은
$$\mathbf{A}\mathbf{V}\mathbf{A}\mathbf{V}=\mathbf{A}\mathbf{V}$$
즉 $\mathbf{A}\mathbf{V}$가 멱등행렬(idempotent matrix)인 것이다.

#### 정리 3.4 (특수형)
1. $\mathbf{y \sim N(0,I_n)}$이면 $\mathbf{y}^T \mathbf{A} \mathbf{y} \sim \chi^2(p)$ ⇔ $\mathbf{A}$가 계수 $p$인 멱등행렬.
2. $\mathbf{y \sim N(\mathbf{\mu},I_n)}$이면 $\mathbf{y}^T\mathbf{A} \mathbf{y} \sim \chi^2(p,\tfrac{1}{2} \mathbf{\mu}^T\mathbf{A}\mathbf{\mu}) \Leftrightarrow \mathbf{A}$가 계수 $p$인 멱등행렬.

#### 정리 3.5
$\mathbf{y}^T \mathbf{A} \mathbf{y}$와 $\mathbf{B}\mathbf{y}$가 독립 ⇔
$\mathbf{B}\mathbf{V}\mathbf{A}=0$
- 증명: A, B가 대칭멱등행렬임을 가정

#### 정리 3.6
두 이차형식 $\mathbf{y}^T \mathbf{A} \mathbf{y}$, $\mathbf{y}^T \mathbf{B} \mathbf{y}$ 독립 ⇔
$$\mathbf{A}\mathbf{V}\mathbf{B}=0$$
- 증명: A, B가 대칭멱등행렬임을 가정

#### 정리 3.7
$\mathbf{A}=\sum_{j=1}^p \mathbf{A}_j$ 이고 각 $\mathbf{A}_j$가 대칭이며 $r(\mathbf{A}_j)=k_j$, $r(\mathbf{A})=k$이면
$$\mathbf{y}^T \mathbf{A}_j \mathbf{y} \sim \chi^2(k_j,\tfrac{1}{2} \mathbf{\mu}^T \mathbf{A}_j\mathbf{\mu})$$
$\mathbf{y}^T \mathbf{A} \mathbf{y}$는 서로 독립이고
$\mathbf{y}^T \mathbf{A} \mathbf{y} \sim \chi^2(k,\tfrac{1}{2} \mathbf{\mu}^T \mathbf{A}\mathbf{\mu})$ 이 성립하기 위한 필요충분조건은 다음의 1 또는 2이다.
1. $\mathbf{A}_j \mathbf{V} \mathbf{A}_j = \mathbf{A}_j$ (즉, $\mathbf{A}_j \mathbf{V}$가 멱등행렬)이고 $\mathbf{A}_j \mathbf{V} \mathbf{A}_l = 0$ ($j \neq l$)
2. $\mathbf{A}_j \mathbf{V}$가 멱등행렬이고 $\sum_{j=1}^p k_j = k$

#### 정리 3.8 (Cochran 정리)
정리 3.7에서 $\mathbf{\mu}=0$이고 $V=I_n=A$이면
$$\mathbf{y \sim N(0,I_n)},\quad \sum_{j=1}^p \mathbf{A}_j=I_n$$
$A_j$는 대칭행렬이면
$$\mathbf{y}^T \mathbf{A}_j \mathbf{y} \sim \chi^2(k_j)$$
서로 독립분포 ⇔ $\sum_{j=1}^p k_j=n$


## 3.6 평균제곱의 기대값 (Expected Mean Squares)
회귀분석의 제곱합, 평균제곱, SSR, SSE, SST, MSR, MSE등은 모두 이차형식(quadratic form)으로 표현할 수 있다. 따라서 이들의 기대값을 구하기 위해서는 이차형식의 기대값을 구하는 문제로 귀결된다.

회귀직선 대체모형:
$$y_i=\beta_0'+\beta_1(x_i-\bar{x})+\varepsilon_i, \quad \varepsilon_i\sim N(0,\sigma^2)$$

오차분산 $\sigma^2$는 다음과 같이 정의할 수 있다: 
$\bar\varepsilon = \sum \varepsilon_i / n$ 이면
$$\sigma^2 = E[\frac{1}{n-1} \sum(\varepsilon_i - \bar\varepsilon)^2]$$
$E(y_i) = \beta_0' + \beta_1 (x_i - \bar{x})$  
$Var(\bar y) = Var(\bar\varepsilon) = \sigma^2 / n$

### 제곱합 기대값
$$
E(SST) = E\left(\sum (y_i - \bar{y})^2\right) \\
= E\left(\sum (y_i - E(y_i) + E(y_i) - \bar{y})^2\right) \\
E\left(\sum (y_i - E(y_i))^2\right) + E\left(\sum (E(y_i) - \bar{y})^2\right) + 2E\left(\sum (y_i - E(y_i))(E(y_i) - \bar{y})\right) \\
= E\left(\sum \varepsilon_i^2\right) + E\left(\sum (\beta_1 (x_i - \bar{x}))^2\right) + 2E\left[\sum \varepsilon_i (\beta_1 (x_i - \bar{x}))\right] \\
= E\left(\sum \varepsilon_i^2\right) + E\left(\sum (\beta_1 (x_i - \bar{x}))^2\right) + 2\beta_1 E\left[\sum \varepsilon_i (x_i - \bar{x})\right] \\
= E\left(\sum \varepsilon_i^2\right) + E\left(\sum (\beta_1 (x_i - \bar{x}))^2\right) + 0 \\
= (n-1)\sigma^2 + \beta_1^2 S_{xx}
$$
> $E[\varepsilon_i]=0$이고 $\varepsilon_i$는 $x_i$와 독립이므로 $E[\sum \varepsilon_i (x_i - \bar{x})] = \sum (x_i - \bar{x})E[\varepsilon_i] = 0$

$$
E(SSR) = E\left(\sum (\hat{y}_i - \bar{y})^2\right) = E\left(\sum (\hat\beta_1 (x_i - \bar{x}))^2\right) \\
= S_{xx} E(\hat\beta_1^2) = S_{xx} \left[ Var(\hat\beta_1) + (E(\hat\beta_1))^2 \right] \\
= S_{xx} \left[ \frac{\sigma^2}{S_{xx}} + \beta_1^2 \right] =\sigma^2+\beta_1^2 S_{xx}
$$
따라서 
$$
E(SSE)= E(SST) - E(SSR) = (n-2)\sigma^2 \\
E(MSR) = E(SSR)/1 =\sigma^2+\beta_1^2 S_{xx}\\
E(MSE)=E(SSE)/(n-2)=\sigma^2$$

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

기호:
$$n=\sum_i n_i, \quad T_i=\sum_j y_{ij}, \quad T=\sum_i T_i, \quad \bar x = \frac{\sum_i n_i x_i}{n}$$

### 제곱합
$$S_{xx}=\sum_i n_i(x_i-\bar{x})^2 = \sum_i n_i x_i^2 - \frac{(\sum_i n_i x_i)^2}{n} \\
S_{yy}=\sum_{i,j}(y_{ij}-\bar{y})^2 = \sum_i \sum_j y_{ij}^2 - \frac{T^2}{n} \\
S_{xy}=\sum_i n_i (x_i-\bar{x})(\bar{y}_i-\bar{y}) = \sum_i x_i T_i-\frac{(\sum_i n_i x_i)T}{n}
$$

### 추정량
$$\hat\beta_1=\frac{S_{xy}}{S_{xx}}, \quad \hat\beta_0=\bar{y}-\hat\beta_1\bar{x}$$

### 적합결여 분해
$$
SST = S_{yy} \\
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
