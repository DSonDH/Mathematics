# Chapter 1 단순회귀분석 (Simple Linear Regression)

## 1.1 서론 (Introduction)
대다수의 연구는 여러 변수(variables) 간의 관계를 규명하는 데 목적이 있다. 예를 들어, 나이와 키의 관계, 기계 온도와 제품 강도의 관계, 광고비와 판매액의 관계 등이 있다. 이때 결과에 해당하는 변수를 반응변수(response variable) 또는 종속변수(dependent variable)라 하고, 이에 영향을 주는 변수를 설명변수(explanatory variable) 또는 독립변수(independent variable)라 한다.

이처럼 변수 간의 함수관계를 통계적으로 추정하고 해석하는 방법을 회귀분석(regression analysis)이라 한다. 회귀분석은 하나의 설명변수와 하나의 반응변수 사이의 관계뿐 아니라 여러 변수 간의 관계도 다룰 수 있으나, 본 장에서는 가장 기본적인 형태인 단순회귀분석(simple regression analysis)을 다룬다.

단순회귀분석은 하나의 설명변수와 하나의 반응변수 사이의 선형관계(linear relation)를 가정하는 분석 방법이다.

### 역사적 배경
회귀(regression)라는 용어는 영국의 통계학자 Galton이 처음 사용하였다. 그는 아버지의 키와 아들의 키 사이에 완전한 비례관계(45도 직선)가 아니라 평균으로 회귀(regression to the mean)하는 경향이 있음을 관찰하였다.

Pearson은 1,078가구의 부자 간 신장 자료를 분석하여 다음과 같은 관계를 추정하였다.
$$\widehat{E}(Y|X=x) = 33.887 + 0.514x$$
* $X$: 아버지의 키
* $Y$: 아들의 평균 키

여기서 주목할 점은 직선의 기울기가 1보다 작다는 것이다. 예를 들어 아버지의 키가 평균보다 10cm 크다면, 아들의 평균 키는 평균보다 약 5.14cm만 더 크다는 의미이다. 즉, 극단적 특성(매우 큰 키)을 가진 부모의 자녀는 극단적이지 않은(평균에 더 가까운) 특성을 보이는 경향이 있다. 이 현상을 **평균으로의 회귀(regression to the mean)** 라 한다.

이 현상은 이변량 정규분포(bivariate normal distribution) 하에서 다음과 같이 설명된다. 확률벡터 $(X,Y)^\top$가 평균벡터 $\mu$와 공분산행렬 $\Sigma$를 가지는 이변량 정규분포를 따른다고 하자.
$$E(Y|X=x) = \mu_Y + \rho \frac{\sigma_Y}{\sigma_X}(x-\mu_X)$$
$\rho$는 상관계수(correlation coefficient)이다. $|\rho|<1$이면 평균으로 회귀하는 현상이 발생한다. 이는 회귀분석이 추정하는 X와 Y 사이의 관계식으로 조건부 기대값인 $E(Y|X=x)$를 추정하는데 기인하는 현상이다.

## 1.2 회귀분석의 기본개념 (Basic Concepts of Regression Analysis)
### 1.2.1 산점도 (Scatter Plot)
두 변수 간 관계를 시각적으로 파악하기 위해 산점도(scatter plot)를 사용한다. 예를 들어 광고비(X)와 총판매액(Y)을 산점도로 나타내면 대략적인 선형관계 여부를 확인할 수 있다.
* 직선형 패턴 → 선형회귀(linear regression) 적합 가능
* 곡선형 패턴 → 곡선회귀(curvilinear regression) 또는 다항회귀(polynomial regression) 필요

### 1.2.2 기본 가정 (Basic Assumptions)
설명변수 $X$와 반응변수 $Y$ 사이의 직선회귀모형(straight line regression model)을 적용하기 위해 다음 가정을 둔다.

#### (1) 선형성 (Linearity)
$$E(Y|X=x) = \beta_0 + \beta_1 x$$

여기서 선형이란 $x$에 대한 선형이 아니라 회귀계수(regression coefficients) $\beta_0, \beta_1$에 대한 선형을 의미한다. 따라서 x에 대한 이차곡선인 $E(Y|X=x) = \beta_0 + \beta_1 x + \beta_2 x^2$도 선형회귀모형에 포함된다.

#### (2) 정규성 및 등분산성 (Normality and Homoscedasticity)
주어진 $x$에서 반응변수 $Y$는 정규분포를 따른다고 가정한다.
$$Y|X=x \sim N(\beta_0 + \beta_1 x, \sigma^2)$$
* 평균은 $x$에 따라 변함
* 분산은 $x$에 관계없이 일정

$$\text{Var}(Y|X=x) = \sigma^2$$

이를 등분산성(homoscedasticity)이라 한다.
분산이 $x$에 따라 달라지면 이분산성(heteroscedasticity)이라 한다.

#### (3) 독립성 (Independence)
설명변수 $X$는 고정값(fixed value)으로 간주하며,
오차항(error term) $\varepsilon_i$들은 서로 독립이다.

$$\text{Cov}(\varepsilon_i,\varepsilon_j|X)=0 \quad (i\ne j)$$

TODO:
### 회귀모형 표현
위 가정 하에서 단순회귀모형은
> 단순회귀모형: 이 모형이 회귀계수로 볼때도 선형, 설명변수로 볼 때도 선형이며, 설명변수가 단 하나뿐인 것
> 이런 모형을 설명변수가 하나인 '일차 모형(first-order model)'이라고도 한다

$$Y = \beta_0 + \beta_1 X + \varepsilon$$

또는 관측치 표현으로
$$y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$$
* $y_i$: i번째 관측 반응값
* $x_i$: i번째 설명변수 값
* $\varepsilon_i$: 오차항(error term)
* $\varepsilon_i \sim N(0,\sigma^2)$ (독립적)

또한
$$Y = E(Y|X=x) + \varepsilon$$
이며

$$\text{Var}(Y|X)=\sigma^2$$
이다.


## 1.3 회귀선의 추정 (Estimation of Regression Line)
표본자료 $(x_1,y_1),\dots,(x_n,y_n)$가 주어졌다고 하자.
모집단 회귀식은

$$E(Y|X=x)=\beta_0+\beta_1 x$$

이나, 실제 분석에서는 모수(parameter) $\beta_0,\beta_1$를 알 수 없으므로 이를 추정(estimation)해야 한다. 추정된 회귀선은

$$\hat{y}=\hat{\beta}_0+\hat{\beta}_1 x$$
으로 표현된다. 여기서

* $\hat{\beta}_0$: 절편(intercept)
* $\hat{\beta}_1$: 기울기(slope)
이다.

### 1.3.1 최소제곱법 (Method of Least Squares)
각 관측치에 대해 오차(residual)
$$e_i=y_i-\hat{y}_i$$
를 정의한다. 최소제곱법은 오차제곱합(sum of squared errors, SSE)

$$S(\beta_0,\beta_1)=\sum_{i=1}^{n}(y_i-\beta_0-\beta_1 x_i)^2$$

을 최소로 만드는 $\beta_0,\beta_1$를 추정값 $\hat{\beta}_0,\hat{\beta}_1$로 정의하는 방법이다.

#### 1) 정규방정식 (Normal Equations)
위 목적함수를 각각 $\beta_0,\beta_1$에 대해 편미분(partial derivative)하여 0으로 놓으면

$$\frac{\partial S}{\partial \beta_0}=-2\sum (y_i-\beta_0-\beta_1 x_i)=0 \\
\frac{\partial S}{\partial \beta_1}=-2\sum x_i (y_i-\beta_0-\beta_1 x_i)=0$$

을 얻는다. 이를 정리하면 정규방정식(normal equations)

$$\hat{\beta}_0 n+\hat{\beta}_1\sum x_i=\sum y_i \\
\hat{\beta}_0\sum x_i+\hat{\beta}_1\sum x_i^2=\sum x_i y_i$$

을 얻는다.

#### 2) 해의 존재와 최소조건
이차편미분(second order partial derivative) 행렬이 양정치(positive definite)이면 극값은 최소값이다.

$$\frac{\partial^2 S}{\partial \beta_0^2}=2n>0$$

행렬식(determinant)

$$|J|=4n\sum (x_i-\bar{x})^2>0$$

이므로 최소해가 보장된다.

#### 3) 최소제곱추정량 (Least Squares Estimators)
정규방정식을 풀면

$$\hat{\beta}_1=\frac{\sum x_i y_i-\frac{(\sum x_i)(\sum y_i)}{n}}{\sum x_i^2-\frac{(\sum x_i)^2}{n}}=\frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sum (x_i-\bar{x})^2} \\
\hat{\beta}_0=\bar{y}-\hat{\beta}_1\bar{x}$$

을 얻는다.

다음 기호를 정의하면

$$S_{xx}=\sum (x_i-\bar{x})^2\\
S_{yy}=\sum (y_i-\bar{y})^2\\
S_{xy}=\sum (x_i-\bar{x})(y_i-\bar{y})$$

기울기는
$$\hat{\beta}_1=\frac{S_{xy}}{S_{xx}}$$
으로 간단히 표현된다.

#### 4) 대체모형 (Centered Form)
회귀식은 다음과 같이 표현할 수도 있다.

$$\hat{y}=\bar{y}+\hat{\beta}_1(x-\bar{x})$$
또는
$$\hat{y}-\bar{y}=\hat{\beta}_1(x-\bar{x})$$

이 식은 추정된 회귀선이 항상 평균점 $(\bar{x},\bar{y})$을 지난다는 것을 의미한다.

#### 5) 예제 결과
광고비–판매액 자료에 대해 계산하면
$$\hat{\beta}_1=2.609, \quad \hat{\beta}_0=-2.270$$

따라서 적합된 회귀직선은
$$\hat{y}=-2.270+2.609x$$

또는 중심화된 형태로
$$\hat{y}=18.6+2.609(x-8)$$
로 쓸 수 있다.

### 1.3.2 최대가능도추정법 (Method of Maximum Likelihood Estimation)
오차항이
$$\varepsilon_i\sim N(0,\sigma^2)$$
을 따른다고 가정하자.

그러면 가능도함수(likelihood function)는
$$L=\prod_{i=1}^{n}\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_i-\beta_0-\beta_1 x_i)^2}{2\sigma^2}\right)$$

로그가능도(log-likelihood)는
$$\ln L=-\frac{n}{2}\ln(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum (y_i-\beta_0-\beta_1 x_i)^2$$
이다.

이를 $\beta_0,\beta_1$에 대해 미분하면 최소제곱법에서 얻은 정규방정식과 동일한 조건이 나온다.
따라서,

> 정규성(normality) 가정 하에서 최대가능도추정량(MLE)은 최소제곱추정량과 일치한다.

단, 최소제곱법은 정규성 가정 없이도 적용 가능하다는 점이 차이이다.

### 1.3.3 적합된 회귀선의 성질 (Properties of Fitted Regression Line)
$$e_i=y_i-\hat{y}_i$$
를 잔차(residual)라 한다. 최소제곱추정량을 사용할 때 다음 성질이 성립한다.

#### (1) 잔차의 합은 0
$$\sum e_i=0$$
따라서
$$\sum y_i=\sum \hat{y}_i$$

#### (2) 잔차제곱합은 최소
$$\sum e_i^2$$
은 가능한 모든 직선 중 최소가 된다.

#### (3) 잔차와 $x_i$의 가중합은 0
$$\sum x_i e_i=0$$

#### (4) 잔차와 적합값의 가중합은 0
$$\sum \hat{y}_i e_i=0$$

즉, 적합값과 잔차의 표본상관계수(sample correlation coefficient)는 0이다.

#### (5) 평균점은 회귀선 위에 존재
$$(\bar{x},\bar{y})$$
는 적합된 회귀직선 위에 있다.


## 1.4 회귀선의 정도 (Goodness of Fit)
회귀선 자체는 평균적 관계를 설명할 뿐, 자료가 그 직선 주위에 얼마나 밀집해 있는지는 별도의 지표가 필요하다. 이를 정량화하는 대표적 척도가 다음 두 가지이다.

* 추정값의 표준오차 (standard error of estimate)
* 결정계수 (coefficient of determination, $R^2$)

### 1.4.1 추정값의 표준오차 (Standard Error of Estimate)
선형회귀모형
$$Y = \beta_0 + \beta_1 x + \varepsilon$$
에서 오차항 $\varepsilon$는
* $E(\varepsilon)=0$
* $\mathrm{Var}(\varepsilon)=\sigma^2$
를 만족한다고 가정한다.

회귀식을 적합한 뒤 잔차(residual)
$$e_i = y_i - \hat{y}_i$$
를 정의하면, 오차분산 $\sigma^2$의 불편추정량(unbiased estimator)은

$$s_{y \cdot x}^2=\frac{\sum (y_i - \hat{y}_i)^2}{n-2}$$

따라서 추정값의 표준오차는
$$s_{y \cdot x}=\sqrt{\frac{\sum (y_i - \hat{y}_i)^2}{n-2}}$$

해석
* 값이 작을수록 자료가 회귀선에 밀집되어 있음
* 모든 점이 회귀선 위에 있으면 $s_{y \cdot x}=0$
* 단위는 종속변수 $Y$의 단위와 동일

### 1.4.2 결정계수 (Coefficient of Determination)
총변동(total variation)을 다음과 같이 정의한다.

$$SST = \sum (y_i - \bar{y})^2$$

이는 다음 두 부분으로 분해된다.
$$SST = SSE + SSR$$

여기서
* $SSE = \sum (y_i - \hat{y}_i)^2$ (설명되지 않는 변동, residual sum of squares)
* $SSR = \sum (\hat{y}_i - \bar{y})^2$ (회귀에 의해 설명되는 변동)

결정계수는
$$R^2=\frac{SSR}{SST}=1 - \frac{SSE}{SST}$$

#### 성질
$$0 \le R^2 \le 1$$
* $R^2 = 1$: 완전한 선형관계
* $R^2 = 0$: 선형설명력 없음

#### 해석
$R^2$는
> 종속변수 총변동 중 회귀식이 설명하는 비율
을 의미한다.

### 1.4.3 단순회귀에서의 추가 성질
단순선형회귀에서는

$$R^2 = r_{xy}^2$$
즉, 결정계수는 표본상관계수의 제곱과 동일하다.

또한,
$$SSR = \hat{\beta}_1^2 S_{xx} \\
\hat{\beta}_1 = \frac{S_{xy}}{S_{xx}}$$
이므로,

$$R^2=\frac{S_{xy}^2}{S_{xx} S_{yy}}=r_{xy}^2$$
가 성립한다.


## 1.5 상관분석 (Correlation Analysis)
결정계수는 선형관계의 "설명력"을 나타내지만, 관계의 방향(양/음)을 나타내지는 못한다. 이를 보완하는 척도가 상관계수이다.

### 1.5.1 모집단 상관계수 (Population Correlation)
이변량 분포 $(X,Y)$에서 모집단 상관계수는
$$\rho_{XY}=\frac{\mathrm{Cov}(X,Y)}{\sqrt{\mathrm{Var}(X)\mathrm{Var}(Y)}}$$

표준편차로 표현하면
$$\rho_{XY}=\frac{\sigma_{XY}}{\sigma_X \sigma_Y}$$

### 범위
$$-1 \le \rho \le 1$$

### 1.5.2 표본상관계수 (Sample Correlation)
표본자료 $(x_i,y_i)$에 대해

$$S_{xx}=\sum (x_i-\bar{x})^2 \\
S_{yy}=\sum (y_i-\bar{y})^2 \\
S_{xy}=\sum (x_i-\bar{x})(y_i-\bar{y})$$
라 하면,

$$r_{xy}=\frac{S_{xy}}{\sqrt{S_{xx}S_{yy}}}$$

#### 성질
$$-1 \le r_{xy} \le 1$$

* $r=1$: 완전한 양의 선형관계
* $r=-1$: 완전한 음의 선형관계
* $r=0$: 선형관계 없음 (비선형관계는 존재 가능)

### 1.5.3 회귀계수와 상관계수의 관계
$$\hat{\beta}_1=r_{xy}\frac{s_y}{s_x}$$
또는
$$r_{xy}=\hat{\beta}_1 \frac{s_x}{s_y}$$

따라서
* 회귀계수와 상관계수는 같은 부호
* 단위 변환에 따라 회귀계수는 변하지만 상관계수는 변하지 않음

### 1.5.4 표준화 변수와의 관계
표준화 변수
$$x^*=\frac{x-\bar{x}}{s_x}, \quad y^*=\frac{y-\bar{y}}{s_y}$$
를 사용하면 회귀식은
$$\hat{y}^* = r_{xy} x^*$$

즉,
> 표준화 단위에서의 회귀계수는 상관계수와 동일하다.


## 1.6 분산분석 (Analysis of Variance, ANOVA)
단순회귀에서 총변동은 다음과 같이 분해된다.

$$SST = \sum (y_i - \bar{y})^2 \\
SST = SSR + SSE$$

여기서
* $SSR = \sum (\hat{y}_i - \bar{y})^2$ : 회귀에 의해 설명되는 제곱합
* $SSE = \sum (y_i - \hat{y}_i)^2$ : 잔차 제곱합

또한 단순회귀에서
$$SSR = \hat{\beta}_1^2 S_{xx} = \frac{S_{xy}^2}{S_{xx}}$$


### 1.6.1 분산분석표
| 요인 | 제곱합 | 자유도 | 평균제곱            |
| -- | --- | --- | --------------- |
| 회귀 | SSR | 1   | MSR = SSR/1     |
| 잔차 | SSE | n−2 | MSE = SSE/(n−2) |
| 합계 | SST | n−1 |                 |

검정통계량은
$$F_0 = \frac{MSR}{MSE}\\ 
F_0 \sim F(1, n-2)$$
를 따른다 (귀무가설 하).

### 1.6.2 회귀의 유의성 검정

귀무가설:
$$H_0 : \beta_1 = 0$$

대립가설:
$$H_1 : \beta_1 \ne 0$$

판정기준:
$$F_0 > F_{\alpha}(1, n-2)$$
이면 회귀선은 유의(significant)하다고 판단한다.

해석적으로,
* SSR이 SSE보다 상대적으로 크면
* 즉 설명변동이 오차변동보다 충분히 크면
* 회귀식은 통계적으로 유의하다.

단순회귀에서는
$$F_0 = t^2$$
관계가 성립한다.


## 1.7 원점을 지나는 회귀 (Regression Through the Origin)
어떤 이론적 이유로
$$X=0 \Rightarrow Y=0$$
가 명확히 성립하는 경우 절편을 제거한 모형을 사용한다.

### 1.7.1 모형
$$y_i = \beta_1 x_i + \varepsilon_i$$
오차제곱합:
$$S(\beta_1)=\sum (y_i - \beta_1 x_i)^2$$

이를 최소화하면
$$\hat{\beta}_1=\frac{\sum x_i y_i}{\sum x_i^2}$$

### 1.7.2 분산 추정
자유도는 1개의 모수만 추정하므로
$$s^2=\frac{\sum (y_i - \hat{\beta}_1 x_i)^2}{n-1}$$
(일반회귀에서는 $n-2$였음에 주의.)

### 1.7.3 제곱합 분해
$$SSR = \sum \hat{y}_i^2 = \hat{\beta}_1^2 \sum x_i^2 \\
SSE = \sum (y_i - \hat{y}_i)^2$$
자유도:
* 회귀: 1
* 잔차: $n-1$

검정통계량:
$$F_0 = \frac{SSR/1}{SSE/(n-1)}$$

### 1.7.4 주의점
* 원점 통과 회귀에서는 잔차합이 0이 아니다.
* $R^2$의 정의도 일반회귀와 동일하게 해석하면 오류가 발생할 수 있다.
* 절편을 제거하는 것은 강한 구조적 가정이므로 신중해야 한다.


## 1.8 가중회귀 (Weighted Regression)
기본 회귀는 등분산 가정
$$\mathrm{Var}(\varepsilon_i)=\sigma^2$$
을 전제로 한다.

그러나 실제로는
$$\mathrm{Var}(\varepsilon_i)=\sigma_i^2=\frac{\sigma^2}{w_i}$$
처럼 관측마다 분산이 다를 수 있다.

이 경우 가중최소제곱법(Weighted Least Squares, WLS)을 사용한다.

### 1.8.1 목적함수
$$Q(\beta_0,\beta_1)=\sum w_i (y_i - \beta_0 - \beta_1 x_i)^2$$
를 최소화한다.

### 1.8.2 가중평균
$$\bar{x}_w=\frac{\sum w_i x_i}{\sum w_i} \\
\bar{y}_w=\frac{\sum w_i y_i}{\sum w_i}$$

### 1.8.3 추정량
$$\hat{\beta}_1=\frac{\sum w_i (x_i-\bar{x}_w)(y_i-\bar{y}_w)}{\sum w_i (x_i-\bar{x}_w)^2} \\
\hat{\beta}_0=\bar{y}_w - \hat{\beta}_1 \bar{x}_w$$

### 1.8.4 변동의 정의
$$SST=\sum w_i (y_i - \bar{y}_w)^2 \\
SSR=\frac{\left[\sum w_i (x_i-\bar{x}_w)(y_i-\bar{y}_w)\right]^2}{\sum w_i (x_i-\bar{x}_w)^2} \\
SSE = SST - SSR$$
가중잔차합은
$$\sum w_i (y_i - \hat{y}_i)=0$$

### 1.8.5 의미
* 분산이 큰 관측값에는 작은 가중치
* 분산이 작은 관측값에는 큰 가중치
* 이 경우 추정량은 최소분산선형불편추정량(BLUE)이 된다 (Gauss–Markov 정리 하)
