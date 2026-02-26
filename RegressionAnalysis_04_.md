# Chapter 4 기초적 중회귀분석 (Multiple Linear Regression)

## 4.1 서론 (Introduction)
제3장에서 다룬 단순회귀모형(simple linear regression model)은 하나의 설명변수(explanatory variable)만을 포함하였다. 그러나 실제 자연·사회 현상에서는 반응변수(response variable) $y$가 여러 요인에 의해 동시에 영향을 받는 경우가 일반적이다.

예를 들어 총판매액이 광고비뿐 아니라 상점 규모, 위치, 종업원 수 등에 의해 함께 영향을 받는다고 가정할 수 있다. 이러한 경우 하나의 설명변수만 사용하는 단순회귀는 정보 손실을 초래한다.

따라서 여러 설명변수를 동시에 포함하는 모형을 고려한다. 이를 **중회귀모형(multiple regression model)** 또는 보다 정확히는 **중선형회귀모형(multiple linear regression model)** 이라 한다.

여기서 "선형(linear)"이라는 의미는 설명변수에 대해 선형이라는 뜻이 아니라 **회귀계수(regression coefficients)에 대해 선형(linear in parameters)** 이라는 뜻이다.

## 4.2 설명변수가 둘인 경우 (Two-Predictor Case)

**(1) 모형 설정**  
반응변수 $y$와 두 설명변수 $x_1, x_2$ 사이의 관계를 다음과 같이 가정한다.
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon$$
* $\beta_0, \beta_1, \beta_2$ : 모수(parameters), 회귀계수(regression coefficients)
* $\varepsilon$ : 오차항(error term)

i번째 관측값에 대해
$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \varepsilon_i \quad (i=1,\dots,n)$$

**(2) 오차 가정 (Error Assumptions)**  
$$\varepsilon_i \overset{i.i.d.}{\sim} N(0,\sigma^2)$$
* 평균 $E(\varepsilon_i)=0$
* 분산 $\text{Var}(\varepsilon_i)=\sigma^2$
* 공분산 $\text{Cov}(\varepsilon_i,\varepsilon_j)=0 \quad (i\neq j)$

이는 독립 동일분포(independent and identically distributed, i.i.d.) 정규오차 가정이다.

**(3) 최소제곱추정 (Least Squares Estimation)**  
예측값(predicted value)은
$$\hat y_i = \hat\beta_0 + \hat\beta_1 x_{i1} + \hat\beta_2 x_{i2}$$

잔차(residual)는
$$e_i = y_i - \hat y_i$$

오차제곱합(sum of squared errors, SSE)은
$$S = \sum_{i=1}^n (y_i - \hat\beta_0 - \hat\beta_1 x_{i1} - \hat\beta_2 x_{i2})^2$$

이를 최소화하는 $\beta$가 최소제곱추정량(least squares estimator)이다.

**(4) 정규방정식 (Normal Equations)**  
각 계수에 대해 편미분 후 0으로 놓으면 $\frac{\partial S}{\partial \hat\beta_j} = 0$세 개의 연립방정식이 얻어진다.
$$\sum y_i = n\hat\beta_0 + \hat\beta_1 \sum x_{i1} + \hat\beta_2 \sum x_{i2}\\
\sum x_{i1}y_i = \hat\beta_0 \sum x_{i1} + \hat\beta_1 \sum x_{i1}^2 + \hat\beta_2 \sum x_{i1}x_{i2}\\
\sum x_{i2}y_i = \hat\beta_0 \sum x_{i2} + \hat\beta_1 \sum x_{i1}x_{i2} + \hat\beta_2 \sum x_{i2}^2$$
이를 정규방정식(normal equations)이라 한다.  
설명변수가 많아질수록 이 방식은 복잡해지므로 행렬표현을 사용한다.

## 4.3 행렬의 사용 (Matrix Formulation)
**(1) 벡터 및 행렬 표현**  
모형을 벡터형태로 쓰면
$$y = X\beta + \varepsilon$$
* $y$ : $n\times1$ 반응벡터(response vector)
* $X$ : $n\times(p+1)$ 설계행렬(design matrix)
  - p: 설명변수의 수
* $\beta$ : $(p+1)\times1$ 회귀계수 벡터
* $\varepsilon$ : 오차벡터(error vector)
  - $E(\varepsilon)=0_n$, $\text{Var}(\varepsilon)=\sigma^2 I_n$

두 설명변수의 경우
$$X=\begin{pmatrix}1 & x_{11} & x_{12} \\1 & x_{21} & x_{22} \\\vdots & \vdots & \vdots \\1 & x_{n1} & x_{n2}\end{pmatrix}$$

**(2) 최소제곱해 (Least Squares Solution)**  
오차제곱합은 $S = (y-X\beta)^T (y-X\beta)$  
이를 $\beta$에 대해 미분하면
$$X^T X \hat\beta = X^T y$$

이를 행렬형 정규방정식(matrix normal equation)이라 한다.

**(3) 해의 존재 조건**  
$X^T X$가 가역행렬(invertible matrix)일 때
$$\hat\beta = (X^T X)^{-1} X^T y$$
가역이 되기 위한 조건은 $\text{rank}(X)=p+1$

즉 설명변수들 사이에 완전한 선형종속(linear dependence)이 없어야 한다.


## 4.4 분산분석 (Analysis of Variance, ANOVA)
회귀모형에서의 분산분석은 반응변수의 총변동(total variation)을
* 회귀식에 의해 설명되는 변동 (variation due to regression)
* 잔차에 의한 변동 (variation due to residuals)

으로 분해하는 과정이다.

중회귀모형 을 가정한다.
$$y = X\beta + \varepsilon, \qquad \varepsilon \sim N(0_n, \sigma^2 I_n)$$

### 4.4.1 총변동의 분해 (Decomposition of Total Variation)

**(1) 총제곱합 (Total Sum of Squares, SST)**  
총변동은 다음과 같이 정의한다.
$$SST = \sum_{i=1}^n (y_i - \bar y)^2$$

행렬형으로는
$$SST = \mathbf{y}^T \mathbf{y} - n(\bar y)^2\\
= \mathbf{y}^T\left(I_n - \frac{J_n}{n}\right)\mathbf{y}$$
* $I_n$ : 단위행렬 (identity matrix)
* $J_n$ : 모든 원소가 1인 행렬

>자유도에 대한 고찰  
>자유도(degree of freedom)는 $n-1$이다. 이는 $\sum (y_i-\bar y)=0$이라는 하나의 선형 제약이 존재하기 때문이다.  
>또한, 위 식은 이차형식으로 표현되므로, 3.5절의 정리로부터 자유도는 $n-1$의 $카이제곱분포(chi-square distribution)$를 따른다. 이처럼 자유도를 정규분포를 가정하고 이를 이용한 카이제곱분포와 연결하여 생각하면 이해도 쉽고 계산도 쉬워진다.

**(2) 잔차제곱합 (Sum of Squares due to Error, SSE)**  
예측값 벡터는 $\hat{\mathbf{y}} = X\hat\beta$  
잔차는 $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$  
잔차제곱합은
$$SSE = (\mathbf{y}-\hat{\mathbf{y}})^T (\mathbf{y}-\hat{\mathbf{y}})\\ 
= \mathbf{y}^T\mathbf{y} - 2\hat\beta^T X^T \mathbf{y} + \hat\beta^T X^T X \hat\beta \\
= \mathbf{y}^T\left[I_n - X(X^T X)^{-1}X^T\right]\mathbf{y} \\
= \mathbf{y}^T \mathbf{y} - \hat\beta^T X^T \mathbf{y}$$
자유도는 $\mathbf{y}^T \mathbf{y}$에서 $\hat\beta^T X^T \mathbf{y}$를 빼는 과정에서 $p+1$개의 선형 제약이 추가되므로 $n - p - 1$  

이 값은 $X$의 랭크(rank)가 $p+1$일 때 성립한다.

**(3) 회귀제곱합 (Sum of Squares due to Regression, SSR)**  
회귀에 의해 설명되는 변동은 $SSR = SST - SSE$ 또는 정의로부터
$$SSR = \sum_{i=1}^n (\hat y_i - \bar y)^2 \\
= \hat\beta^T X^T \mathbf{y} - n(\bar y)^2 \\
= \mathbf{y}^T X (X^T X)^{-1} X^T \mathbf{y} - n(\bar y)^2$$

자유도는 $p$이다. 이는 회귀식이 $p$개의 설명변수를 포함하기 때문이다.

**(4) 변동의 분해식**
$$SST = SSR + SSE$$
자유도도 $(n-1) = p + (n-p-1)$ 로 분해된다.

**(5) 평균제곱 (Mean Squares)**  
* 회귀평균제곱 (Mean Square due to Regression, MSR)
$$MSR = \frac{SSR}{p}$$

* 잔차평균제곱 (Mean Square Error, MSE)
$$MSE = \frac{SSE}{n-p-1}$$

>**중회귀의 분산분석표 (ANOVA Table for Multiple Regression)**  
>| 요인 | 제곱합 | 자유도 | 평균제곱 | $F_0$ | $F_\alpha$ |
>| --- | --- | --- | --- | --- | --- |
>| 회귀 | SSR | p | $MSR=\frac{SSR}{p}$ | $\frac{MSR}{MSE}$ | $F_\alpha(p,n-p-1)$ |
>| 잔차 | SSE | n-p-1 | $MSE=\frac{SSE}{n-p-1}$ | | |
>| 전체 | SST | n-1 | | | |

**(6) F-검정 (F-test)**  
귀무가설 $H_0 : \beta_1 = \beta_2 = \cdots = \beta_p = 0$

대립가설 $H_1 : \text{적어도 하나의 } \beta_j \neq 0$

검정통계량은
$$F_0 = \frac{MSR}{MSE}$$

정규오차 가정 하에서
$$F_0 \sim F(p, n-p-1)$$
을 따른다.

**(7) 기대값 계산**  
$$E(y) = X\beta, \qquad \text{Var}(y) = \sigma^2 I_n$$

(i) SSE의 기대값  
$$E(SSE) = E\left[y^T\left(I_n - X(X^T X)^{-1}X^T\right)y\right] \\
= \text{tr}\left[\left(I_n - X(X^T X)^{-1}X^T\right) \text{Var}(y)\right] + E(y)^T \left(I_n - X(X^T X)^{-1}X^T\right) E(y) \\
= \text{tr}\left[\left(I_n - X(X^T X)^{-1}X^T\right) \sigma^2 I_n\right] + \beta^T X^T \left(I_n - X(X^T X)^{-1}X^T\right) X\beta \\
= \sigma^2 \text{tr}\left(I_n - X(X^T X)^{-1}X^T\right) + 0 \\
= \sigma^2 (n - \text{rank}(X)) = (n-p-1)\sigma^2$$
따라서
$$E(MSE) = \sigma^2$$
즉, MSE는 오차분산 $\sigma^2$의 불편추정량(unbiased estimator)이다.

(ii) SSR의 기대값  
$$E(SSR) = p\sigma^2 + \beta^T X^T \left(I_n - \frac{J_n}{n}\right) X\beta$$
따라서
$$E(MSR) = \sigma^2 + \frac{1}{p} \beta^T X^T \left(I_n - \frac{J_n}{n}\right) X\beta$$

이때 $X^T \left(I_n - \frac{J_n}{n}\right) X$는 양의 준정부호행렬(positive semi-definite matrix)이므로
$$\beta^T X^T \left(I_n - \frac{J_n}{n}\right) X\beta \ge 0$$
등호가 성립하는 경우는 $\beta_1 = \beta_2 = \cdots = \beta_p = 0$ 일 때뿐이다. 따라서 분산분석표의 F검정 가설은 모든 회귀계수가 0이라는 가설을 검정하는 것이다.

**(8) F-검정의 해석**  
$$\frac{E(MSR)}{E(MSE)} = 1 + \frac{1}{\sigma^2 p} \beta^T X^T \left(I_n - \frac{J_n}{n}\right) X\beta$$
* 모든 $\beta_j = 0$이면 $E(MSR) = E(MSE)$
* 적어도 하나가 0이 아니면 $E(MSR) > E(MSE)$

즉, F-검정은 회귀식이 유의미한 설명력을 가지는지 여부를 검정하는 절차이다.


## 4.5 회귀모형의 정도 (Model Fit)
추정된 회귀모형이 자료를 얼마나 잘 설명하는지, 그리고 예측이 어느 정도 정확한지를 평가하는 지표들을 정리한다.

### 4.5.1 평균제곱오차 MSE (Mean Square Error)
잔차평균제곱(residual mean square)은
$$MSE=\frac{SSE}{n-p-1}=\frac{(\mathbf{y}-\hat{\mathbf{y}})^T(\mathbf{y}-\hat{\mathbf{y}})}{n-p-1} =\frac{\mathbf{y}^T[I_n-X(X^TX)^{-1}X^T]\mathbf{y}}{n-p-1}$$
앞 절에서 보였듯이 $E(MSE)=\sigma^2$ 이므로 MSE는 오차분산 $\sigma^2$의 불편추정량(unbiased estimator)이다. MSE가 작을수록 관측값들이 추정된 회귀평면(regression hyperplane) 주위에 밀집해 있음을 의미한다.

### 4.5.2 F-검정에 의한 모형 유의성
추정된 회귀모형이 관측값들을 통계적으로 유의미하게 설명하는지 여부를 검정한다.  

검정통계량은 $F_0=\frac{MSR}{MSE}$ 자유도는 $(p, n-p-1)$  
귀무가설 $H_0:\beta_1=\cdots=\beta_p=0$ 하에서
$$F_0 \sim F(p,n-p-1)$$

$F_0$가 임계값 $F_\alpha(p,n-p-1)$보다 크면 모형은 통계적으로 유의하다.

### 4.5.3 결정계수 $R^2$ (Coefficient of Determination)
추정된 회귀모형이 반응변수의 변동을 얼마나 설명하는지를 나타내는 지표로 결정계수(coefficient of determination) $R^2$가 있다.

회귀모형이 설명하는 변동의 비율은
$$R^2=\frac{SSR}{SST} =1-\frac{SSE}{SST} = \frac{\mathbf{\hat\beta}^T X^T \mathbf{y}-n(\bar y)^2}{\mathbf{y}^T\mathbf{y}-n(\bar y)^2}$$
* $0\le R^2\le 1$
* 모든 관측값이 완전히 설명되면 $R^2=1$
* 설명력이 거의 없으면 $R^2\approx0$

### 4.5.4 회귀계수 추정의 분산
회귀계수 추정이 무엇보다 중요한 경우.  
최소제곱추정량 $\hat\beta=(X^TX)^{-1}X^Ty$에 대해
$$E(\hat\beta)= E\left[(X^TX)^{-1}X^Ty\right] = (X^TX)^{-1}X^TE(y) = (X^TX)^{-1}X^TX\beta = \beta$$
즉 불편추정량이다.

분산–공분산행렬(variance–covariance matrix)은
$$\text{Var}(\hat\beta)= \text{Var}\left[(X^TX)^{-1}X^Ty\right] = (X^TX)^{-1}X^T \text{Var}(y) X(X^TX)^{-1} =
\sigma^2(X^TX)^{-1}$$
그런데 $\text{Var}(\hat\beta)$의 $(i,j)$ 원소는 $\text{Cov}(\hat\beta_i,\hat\beta_j)$이므로, 만약 $(X^TX)^{-1}$의 $(i,j)$ 원소를 $c_{ij}$라 하면
$$\text{Var}(\hat\beta_i)=c_{ii}\sigma^2\\
\text{Cov}(\hat\beta_i,\hat\beta_j)=c_{ij}\sigma^2$$

만약 우리가 특별히 관심있는 설명변수가 $x_j$라고 하면, $\hat\beta_j$의 표준오차(standard error), 즉 $\sqrt{c_{jj}\sigma^2}$를 작게 설계할 필요가 있다. 설계행렬(design matrix)의 구조에 따라 계수의 분산이 달라지므로 실험설계(experimental design)가 중요한 이유가 여기에 있다. (단, 관측값이 많아질수록 계수의 분산이 작아지는 것은 아니다. 설명변수들의 상관관계에 따라 달라진다.)

### 4.5.5 예측값의 분산
추정 이후 새로 주어진 $x$에 대한 예측값(predicted value) $\hat y$의 분산도 관심이 있다.

임의의 설명변수 벡터 $x^T=(1,x_1,\dots,x_p)$ 에서 평균반응의 추정량은 $\hat y= \mathbf{x}^T \mathbf{\hat\beta}$ 이고
$$\text{Var}(\hat y)=\sigma^2 \mathbf{x}^T(X^TX)^{-1}\mathbf{x}$$
즉 예측값 $\hat y$의 분산은 설명변수 벡터 $x$와 설계행렬 $X$의 구조에 의해 결정된다.

개별 관측값 예측의 경우, 새로운 관측값 $y_{new} = \mathbf{x}^T\hat\beta + \varepsilon_{new}$의 분산은 추정된 회귀식의 오차와 새로운 관측값의 오차 모두를 포함해야 한다.
$$\text{Var}(\hat y_{new})=\text{Var}(\mathbf{x}^T\hat\beta) + \text{Var}(\varepsilon_{new}) = \sigma^2\mathbf{x}^T(X^TX)^{-1}\mathbf{x} + \sigma^2 = \sigma^2\left[1+\mathbf{x}^T(X^TX)^{-1}\mathbf{x}\right]$$
따라서 개별 관측값의 예측분산은 평균반응의 추정분산보다 항상 크다. 이는 개별값 예측이 평균값 추정보다 불확실성이 크다는 의미이며, 이를 반영하여 더 넓은 예측구간(prediction interval)을 구성하게 된다.

단순회귀의 경우 이는
$$\text{Var}(\hat y)=\sigma^2\left[\frac{1}{n}+\frac{(x-\bar x)^2}{\sum (x_i-\bar x)^2}\right]$$
와 일치한다.


## 4.6 절편 없는 중회귀모형 (Regression without Intercept)
일반 중회귀모형은 절편(intercept term) $\beta_0$을 포함한다. 그러나 설명변수가 0일 때 반드시 $y=0$이어야 하는 구조적 제약이 있으면 절편을 제거한다.

**4.6.1 모형**  
$$y_i=\beta_1x_{i1}+\cdots+\beta_px_{ip}+\varepsilon_i =X\beta+\varepsilon$$

단, 여기서 $X$는 첫 열이 1이 아닌 $n\times p$ 행렬이다.

**4.6.2 제곱합**  
총제곱합은 평균보정을 하지 않는다. $SST=y^Ty$  
자유도는 $n$이다.

잔차제곱합은 $SSE=y^Ty-\hat\beta^TX^Ty$  
자유도는 $n-p$이다.

회귀제곱합은 $SSR=\hat\beta^TX^Ty$  
자유도는 $p$이다.

**4.6.3 분산분석표**  
| 요인 | 제곱합 | 자유도 |
| -- | --- | --- |
| 회귀 | SSR | p   |
| 잔차 | SSE | n-p |
| 전체 | SST | n   |

검정통계량:
$$F_0=\frac{MSR}{MSE} \sim F(p,n-p)$$

**4.6.4 결정계수**  
절편이 없는 경우
$$R^2=\frac{\hat\beta^TX^Ty}{y^Ty}$$
로 정의된다.

이 경우 $R^2$는 0과 1 사이가 아닐 수도 있다.
따라서 해석에 주의가 필요하다.

**4.6.5 예측값 분산**  
$$\text{Var}(\hat y)=\sigma^2 x^T(X^TX)^{-1}x$$
형태는 동일하나, $X^TX$는 $p\times p$ 행렬이다.


## 4.7 제곱합의 분포 (Distribution of Sum of Squares)
분산분석표에 나타나는 제곱합(SST, SSR, SSE)의 분포를 이론적으로 분석하고, 왜 $F_0$가 $F$분포를 따르는지 설명한다.

**1. 기본 가정**  
중회귀모형
$$y = X\beta + \varepsilon, \qquad \varepsilon \sim N(0_n, \sigma^2 I_n)$$
즉,
$$y \sim N(X\beta, \sigma^2 I_n)$$

**2. 이차형과 카이제곱 분포**  
정리: $y \sim N(\mu, \sigma^2 I)$이고 $A$가 대칭이고 멱등(idempotent) 행렬이며 $\text{rank}(A)=r$이면

$$\frac{y^T A y}{\sigma^2} \sim \chi^2(r,\lambda)$$
이며 비중심모수(noncentrality parameter)

$$\lambda = \frac{\mu^T A \mu}{2\sigma^2}$$
이 결과를 SST, SSR, SSE에 적용한다.

**3. SST의 분포**  
$$SST = y^T\left(I_n - \frac{J_n}{n}\right)y$$
행렬 $A = I_n - \frac{J_n}{n}$는 대칭이고 멱등이며 rank = $n-1$이므로, 
$$\frac{SST}{\sigma^2} \sim \chi^2\left(n-1, \frac{\beta^T X^T (I-\frac{J}{n}) X\beta}{2\sigma^2}\right)$$

**4. SSR의 분포**  
$$SSR = y^T\left[X(X^TX)^{-1}X^T - \frac{J_n}{n}\right]y$$
여기서 행렬 $B = X(X^TX)^{-1}X^T - \frac{J_n}{n}$는 대칭이고 멱등이며 rank = $p$이다. 또한
$$(X\beta)^T\left[X(X^TX)^{-1}X^T - \frac{J_n}{n}\right](X\beta) = \beta^T X^T \left[X(X^TX)^{-1}X^T - \frac{J_n}{n}\right] X\beta$$
이므로 비중심모수는 $$\lambda = \frac{\beta^T X^T (I-\frac{J}{n}) X\beta}{2\sigma^2}$$
 
따라서
$$\frac{SSR}{\sigma^2} \sim \chi^2(p,\lambda)$$

**5. SSE의 분포**  
$$SSE = y^T\left[I_n - X(X^TX)^{-1}X^T\right]y$$
여기서 행렬 $C = I_n - X(X^TX)^{-1}X^T$는 대칭이고 멱등이며 rank = $n-p-1$이다. 또한
$$(X\beta)^T\left[I_n - X(X^TX)^{-1}X^T\right](X\beta) = 0$$
이므로 비중심모수는 0이다. 따라서
$$\frac{SSE}{\sigma^2} \sim \chi^2(n-p-1)$$
(중심 카이제곱분포)

**6. SSR과 SSE의 독립성**  
두 $y$의 이차형식인 $\frac{SSR}{\sigma^2}$과 $\frac{SSE}{\sigma^2}$가 서로 독립인지 확인해보자.
$$\mathbf{y} \sim N(X\beta, \sigma^2 I_n) \\
\begin{pmatrix}\frac{SSR}{\sigma^2} \\ \frac{SSE}{\sigma^2}\end{pmatrix} = \begin{pmatrix}\mathbf{y}^T A \mathbf{y} \\ \mathbf{y}^T B \mathbf{y}\end{pmatrix} \\
$$
여기서
$$A=\frac{X(X^TX)^{-1}X^T - J/n}{\sigma^2}, \quad B=\frac{I - X(X^TX)^{-1}X^T}{\sigma^2}$$
라 하면 필요충분조건인
$$AVB = A\sigma^2 I B = \mathbf{0}_{n \times n}$$
이 성립한다.  
따라서 SSR과 SSE는 독립이다.

**7. F-통계량의 분포**  
noncentral F distrubution이론에서 논의된 바와 같이 $\frac{SSR}{\sigma^2} \sim \chi^2(p,\lambda)$이고 $\frac{SSE}{\sigma^2} \sim \chi^2(n-p-1)$이며 두 카이제곱이 서로 독립이므로,
$$F_0 = \frac{MSR}{MSE} = \frac{(SSR/p)}{(SSE/(n-p-1))}$$
는 비중심 F분포(noncentral F distribution)를 따르고,
$$F_0 \sim F(p,n-p-1,\lambda)$$
여기서
$$\lambda = \frac{\beta^T X^T (I-\frac{J}{n}) X\beta}{2\sigma^2}$$

**귀무가설 하에서**  
$$H_0:\beta_1=\cdots=\beta_p=0$$
이면 $\lambda=0$이므로
$$F_0\sim F(p,n-p-1)$$
이 된다.

이것이 분산분석 F-검정의 이론적 근거이다.


## 4.8 변수의 직교화 (Orthogonalization of Variables)
핵심 개념은 "부분효과(partial effect)"를 행렬적으로 명확히 하는 것이다. 변수의 직교화라는 개념을 통해 중회귀모형에서 회귀계수의 의미를 이해해보자.

**1. 모형 분할**  
$n$개의 관측값과 $p$개의 설명변수를 포함하는 중회귀모형을 두 부분으로 나눈다:  
$(y_i, \mathbf{x}_{1i}, \mathbf{x}_{2i})$로 표현되는 관측값과 설명변수들을 다음과 같이 분할한다.
$$y_i = \mathbf{x}_{i1}^\top\beta_1 + \mathbf{x}_{i2}^\top\beta_2 + \varepsilon_i$$
* $\mathbf{x}_{i1}$: $(q+1)$개의 이미 고려된 변수들, 절편항을 포함하고 있다
* $\mathbf{x}_{i2}$: $(p-q)$개의 추가 변수들

**2. 단계적 해석**  
1단계: 반응변수 $y$를 $\mathbf{x}_1$에 대해서만 회귀한다.
$$y_i = \mathbf{x}_{i1}^\top \mathbf{\alpha_1} + e_i$$
잔차: $e_i = y_i - \hat y_{i1}$

2단계: X₂를 X₁에 대해 직교화
$$\mathbf{x}_{2.1i} = \mathbf{x}_{2i} - \mathbf{x}_{i1}^\top    (\mathbf{X}_1^T\mathbf{X}_1)^{-1}\mathbf{X}_1^T\mathbf{x}_{2i}$$
이는 $\mathbf{X}_{2.1}=(I - H_1)\mathbf{X}_2$ 와 동일하며
$$H_1 = \mathbf{X}_1(\mathbf{X}_1^T\mathbf{X}_1)^{-1}\mathbf{X}_1^T$$
즉, X₂에서 X₁에 의해 설명되는 부분을 제거한 것이다.

3단계: 잔차에 대한 회귀
$$e_i = \mathbf{x}_{2.1i}^\top \mathbf{\alpha_2} + \varepsilon_i$$
최소제곱추정량 $\mathbf{\alpha_2} = (\mathbf{X}_{2.1}^T \mathbf{X}_{2.1})^{-1} \mathbf{X}_{2.1}^T y$
  - $\mathbf{\alpha_2}$는 $\mathbf{x}_{2.1}$에 대한 회귀계수이므로, $\mathbf{x}_2$에서 $\mathbf{x}_1$ 효과를 제거한 후의 순수한 부분끼리의 회귀계수이다.
    - 최소제곱추정량의 $\hat\beta_2$와 동일하다...(*)

**3. 결과**  
행렬적으로 전개하면
$$\hat\beta_2 = [\mathbf{X}_2^T(I-H_1)\mathbf{X}_2]^{-1} \mathbf{X}_2^T(I-H_1)y$$
즉
$$\hat\beta_2 = \mathbf{\alpha_2}$$

**4. 해석**  
중회귀에서 $\hat\beta_2$는
* $y$에서 $X_1$ 효과를 제거하고
* $X_2$에서도 $X_1$ 효과를 제거한 후
* 남은 순수 부분끼리의 회귀계수

이는 다중공선성(multicollinearity) 해석의 핵심 구조이다.

**(*)의 증명**  
$$\hat\beta_2 = [\mathbf{X}_2^T(I-H_1)\mathbf{X}_2]^{-1} \mathbf{X}_2^T(I-H_1)y$$
$$\mathbf{\alpha_2} = (\mathbf{X}_{2.1}^T \mathbf{X}_{2.1})^{-1} \mathbf{X}_{2.1}^T y$$
여기서 $\mathbf{X}_{2.1}=(I-H_1)\mathbf{X}_2$ 이므로
$$\mathbf{\alpha_2} = [\mathbf{X}_2^T(I-H_1)^T(I-H_1)\mathbf{X}_2]^{-1} \mathbf{X}_2^T(I-H_1)^T y$$
$I-H_1$는 대칭이고 멱등이므로 $(I-H_1)^T(I-H_1) = I-H_1$ 이다. 따라서
$$\mathbf{\alpha_2} = [\mathbf{X}_2^T(I-H_1)\mathbf{X}_2]^{-1} \mathbf{X}_2^T(I-H_1)y = \hat\beta_2$$
