# Chapter 8 회귀진단 (Regression Diagnostics)

## 8.1 기본적인 개념 (Basic Concepts)
회귀진단(regression diagnostics)은 회귀모형을 적합한 뒤, 그 모형이 자료를 설명하는 데 적절한지, 그리고 특정 관측값들이 모형 추정에 비정상적으로 큰 영향을 주고 있지는 않은지를 점검하는 절차이다. 반응변수(response variable) $y$와 설명변수(explanatory variable) $x_1, x_2, \dots, x_p$에 대하여, $n$개의 자료
$$
(x_{i1}, x_{i2}, \dots, x_{pi}, y_i), \quad i=1,2,\dots,n
$$
가 관측되었다고 하자. 일반적인 중회귀모형(multiple regression model)은 다음과 같이 둔다.
$$
y = X\beta + \varepsilon,\qquad \varepsilon \sim N(0_n,\sigma^2 I_n)
$$
여기서 $X$는 $n \times (p+1)$ 행렬이며, 계수(rank)는 $p+1$이라고 가정한다. 최소제곱법(method of least squares)에 의한 회귀계수 추정량(estimator)은 다음과 같다.
$$
\hat{\beta} = (X^T X)^{-1}X^T y
$$
이러한 회귀분석(regression analysis)을 수행할 때, 자료로부터 모형의 타당성을 확인하는 절차를 회귀진단(regression diagnostics)이라 한다. 회귀진단은 다음과 같은 사항을 점검하기 위해 사용된다.
1. 회귀식의 선형성(linearity)
2. 오차항(error term)의 정규성(normality)
3. 오차항의 등분산성(equal variance, homoscedasticity)
4. 모형 추정에 특이한 점(unusual point)의 존재 여부

여기서 특이한 점(unusual point)이란 다음 세 가지를 뜻한다.
* 설명변수 값이 극단적으로 큰 **지렛대점(high leverage point)**
* 관측값 $y$가 모형의 예측값 $\hat{y}$와 크게 다른 **이상점(outlier)**
* 모형 추정에 큰 영향을 주는 **영향점(influential point)** 또는 **영향 관측값(influential observation)**

회귀진단과 관련하여, 특히 다음 여섯 가지 사항을 점검하는 것이 중요하다.
### 8.1.1 오차항 가정의 검토 (Checking the Error-Term Assumptions)
식 $\varepsilon \sim N(0_n,\sigma^2 I_n)$는 오차항이 다음 성질을 가진다는 뜻이다.
* 정규성(normality)
* 독립성(independence)
* 등분산성(equal variance)

이 가정이 성립해야 최소제곱추정량 $\hat{\beta}$가 최량선형불편추정량(best linear unbiased estimator; BLUE)이 되며, 분산분석표(analysis of variance table)에서의 $F$-검정도 유효하게 성립한다.  
일반적으로는 잔차분석(residual analysis)을 통하여 오차항 가정을 검토한다.   잔차분석 방법은 앞 장에서 이미 소개되었으므로 여기서는 생략.

### 8.1.2 적절한 모형의 선택 (Selection of an Appropriate Model)
$y = X\beta + \varepsilon$의 모형은 1차 선형모형(first-order linear model)으로 볼 수 있다. 그러나 실제로는 반응변수 $y$와 $p$개의 설명변수 사이에서 이 모형이 가장 적절한지 검토할 필요가 있다.  
경우에 따라서는 변수변환(variable transformation)을 통해 더 적합한 모형을 구성할 수 있다. 예를 들어 $j$번째 설명변수 $x_j$를 다음과 같이 변환할 수 있다.
* $\sqrt{x_j}$
* $\log x_j$
* $1/x_j$

또한 $y$와 연관성이 낮은 설명변수는 모형에서 제거하는 것이 더 바람직할 수도 있다.

### 8.1.3 설명변수 간 상관관계의 검토 (Checking Correlation Among Explanatory Variables)
설명변수들 사이에 상관관계(correlation)가 크면 $X^T X$의 역행렬(inverse matrix)을 구하기 어렵거나 계산 정확도가 낮아진다. 또한 회귀계수 추정값 $\hat{\beta}_j$들의 분산(variance)이 과도하게 커져 회귀방정식 추정의 의미가 약해진다.  
따라서 설명변수들 사이의 상관관계를 점검해야 하며, 상관관계가 지나치게 크다면 다음과 같은 방법을 고려할 수 있다.
* 일부 설명변수 제거
* 편의추정(biased estimation) 방법 사용

이는 다중공선성(multicollinearity) 문제와 직접 관련된다.

### 8.1.4 지렛대점 탐색 (Search for Leverage Points)
지렛대(leverage)란 $i$번째 관측값의 설명변수 값이 나머지 데이터의 설명변수 값들로부터 얼마나 멀리 떨어져 있는지를 나타내는 개념이다.  
큰 지렛대를 가지는 관측값은 설명변수 값이 다른 관측값들과 멀리 떨어져 있으며, 설명변수 공간(explanatory-variable space)에서 주변에 이웃 관측값이 거의 없는 점이다.  
단순선형회귀(simple linear regression)에서는 설명변수 값이 다른 관측값들과 크게 다른 점을 찾으면 되므로 큰 지렛대점을 식별하는 것이 상대적으로 쉽다. 그러나 중회귀모형(multiple regression model, $p>2$)에서는 각 개별 설명변수만 보면 극단적이지 않아도, 전체 $p$차원 설명변수 벡터로 보면 큰 지렛대점이 될 수 있어 식별이 쉽지 않다.  
회귀분석에서는 이러한 큰 지렛대점을 찾기 위해 해트 행렬(hat matrix)
$$
H = X(X^T X)^{-1}X^T = (h_{ij}), \qquad i,j=1,2,\dots,n
$$
의 대각원소(diagonal element) $h_{ii}$를 사용한다. $h_{ii}$가 큰 관측값은 큰 지렛대점을 가진다고 판단한다.

### 8.1.5 이상점 탐색 (Search for Outliers)
$n$개의 반응변수 $y_1,y_2,\dots,y_n$을 관측할 때, 측정 또는 실험상의 오류 등 다양한 원인으로 이상치가 포함될 수 있다. 일반적으로 이러한 이상점(outlier)은 관측값이 모형의 예측값과 크게 달라 큰 잔차(residual)를 남기므로 식별이 비교적 쉽다.  
이상점이 발견되면, 단순 측정오류(measurement error)인 경우에는 해당 관측값을 제외할 수 있다. 그러나 단순 오류가 아니라면 문제의 성격에 따라 적절한 대응이 필요하다.

### 8.1.6 영향점 탐색 (Search for Influential Observations)
회귀분석의 결과, 즉 회귀계수의 추정값 또는 결정계수(coefficient of determination) 등의 값이 몇 개의 관측값에 의해 크게 영향을 받을 수 있다. 이처럼 회귀분석 결과에 큰 영향을 주는 관측값을 영향점(influential point) 또는 영향 관측값(influential observation)이라 한다.  
예를 들어 $i$번째 관측값 $(y_i, x_{i1}, \dots, x_{ip})$이 큰 영향을 주는 영향점이라고 하자. 그러면 $i$번째 관측값을 포함한 경우와 제외한 경우의 회귀분석 결과, 특히 회귀계수 추정값 $\hat{\beta}$에 큰 차이가 나타난다. 따라서 영향점 판별에는 회귀계수 추정량의 변화가 중요한 지표가 된다.  
또한 다음과 같은 경우에 영향점이 될 가능성이 있다.
* 비정상적인 설명변수 값을 갖는 지렛대점(high leverage point)
* 비정상적인 반응변수 값을 갖는 이상점(outlier)

이 장에서는 특히 다음 세 가지를 중점적으로 다룬다.
* 큰 지렛대점을 갖는 관측값(high leverage case)
* 이상점(outlier)
* 영향점(influential observation)
## 8.2 지렛대점의 검출 (Detection of Leverage Points)
### 8.2.1 해트 행렬 $H = X(X^T X)^{-1}X^T$의 성질 (Properties of the Hat Matrix)
회귀모형에서 $E(y)$의 추정벡터(estimator vector)는 적합값(fitted value) 벡터 $\hat{y}$이며, $\hat{y} = X\hat{\beta}$이다. 최소제곱추정량을 대입하면
$\hat{y} = X(X^T X)^{-1}X^T y = Hy$가 된다. 여기서 $n \times n$ 행렬 $H$를 **해트 행렬(hat matrix)** 이라 부른다. 그 이유는 관측벡터 $y$에 "hat"을 씌워 적합값 $\hat{y}$를 만들기 때문이다.  
해트 행렬의 원소는 다음과 같이 표현된다.
#### 8.2.1.1 해트 행렬 원소의 표현 (Representation of the Elements)
$$
h_{ij} = x_i^T (X^T X)^{-1}x_j = x_j^T (X^T X)^{-1}x_i = h_{ji}\\
h_{ii} = x_i^T (X^T X)^{-1}x_i
$$
여기서
$x_i^T = (1, x_{i1}, x_{i2}, \dots, x_{ip})$는 $i$번째 관측값에 대응하는 설계행렬(design matrix) $X$의 행벡터(row vector)이다.

#### 8.2.1.2 멱등성과 계수 및 대각합 (Idempotence, Rank, and Trace)
해트 행렬은
$HH = H$를 만족하므로 멱등행렬(idempotent matrix)이다.  
또한 그 계수(rank)는 $\gamma(H) = \gamma(X) = p+1$  
더불어 대각합(trace)은 $\operatorname{tr}(H) = \sum_{i=1}^n h_{ii} = p+1$  
즉, 해트 행렬의 대각원소들의 합은 설명변수 개수 $p$와 절편(intercept)을 포함한 총 모수 개수와 같다.

#### 8.2.1.3 멱등성으로부터 나오는 관계 (Relation Derived from Idempotence)
$H = HH$ 이므로 각 대각원소에 대하여
$$
h_{ii} = \sum_{j=1}^n h_{ij}^2 = h_{ii}^2 + \sum_{j\ne i} h_{ij}^2
$$
이 식은 대각원소 $h_{ii}$와 비대각원소(off-diagonal element) $h_{ij}$ 사이의 중요한 관계를 제공한다.

#### 8.2.1.4 값의 범위 (Range of the Elements)
해트 행렬 $H$는 양반정치행렬(positive semidefinite matrix)이므로
$h_{ii} \ge 0$이다. 또한 앞의 관계로부터 $h_{ii}^2 \le h_{ii}$이므로
$$
0 \le h_{ii} \le 1
$$
같은 방식으로 $i \ne j$인 경우에 대해
$$
h_{ij}^2 \le h_{ii} - h_{ii}^2 \le \frac14
$$
이므로
$$
-\frac12 \le h_{ij} \le \frac12
$$
> 식의 구조로부터 $h_{ii}$의 실제 하한(lower bound)은 $1/n$이다.

#### 8.2.1.5 해트 행렬과 설계행렬의 관계 (Relation Between the Hat Matrix and Design Matrix)
해트 행렬은 $HX = X(X^T X)^{-1}X^T X = X$를 만족한다.  
따라서 행렬 $X$의 $j$번째 열벡터(column vector) ($j=1,2,\dots,p+1$)에 대하여
$$
H
\begin{pmatrix}
X_{1j}\\
X_{2j}\\
\vdots\\
X_{nj}
\end{pmatrix}
= \begin{pmatrix}
X_{1j}\\
X_{2j}\\
\vdots\\
X_{nj}
\end{pmatrix}
$$
특히 첫째 열이 절편에 해당하는 상수항(constant term)이면 $H\mathbf{1}_n = \mathbf{1}_n$이고, 따라서
$$
\sum_{i=1}^n h_{ij} = \sum_{j=1}^n h_{ij} = 1
$$
즉, 해트 행렬의 각 행의 합과 각 열의 합은 모두 1이다.

#### 8.2.1.6 단순회귀에서의 해트 값 (Hat Values in Simple Linear Regression)
단순회귀(simple regression)에서 $\sum_i (x_i-\bar{x})^2 = S_{xx}$ 라고 두면, 해트 행렬 원소는
$$
h_{ij} = x_i^T (X^T X)^{-1} x_j = \frac1n + \frac{(x_i-\bar{x})(x_j-\bar{x})}{S_{xx}} \\
h_{ii} = \frac1n + \frac{(x_i-\bar{x})^2}{S_{xx}} \\
\therefore \sum_{i=1}^n h_{ii} = 1 + \frac{\sum_i (x_i-\bar{x})^2}{S_{xx}} = 2
$$
이는 단순회귀에서 $p=1$이므로 $\operatorname{tr}(H)=p+1=2$와 일치한다.  
또한 $h_{ii}$의 최소값은 $x_i=\bar{x}$일 때 $1/n$이고, 최대값은 $x_i$가 평균 $\bar{x}$에서 가장 멀리 떨어져 있을 때 발생한다.

#### 8.2.1.7 중회귀에서의 해트 값 (Hat Values in Multiple Regression)
중회귀모형(multiple regression model)에서는
$$
h_{ii} = \frac1n + (\mathbf{x}_i-\bar{\mathbf{x}})^T ({\chi}^T \chi)^{-1}(\mathbf{x}_i-\bar{\mathbf{x}}) \\
\mathbf{x}_i^T = (x_{i1}, x_{i2}, \dots, x_{ip}), \qquad
\bar{\mathbf{x}}^T = (\bar{x}_1, \bar{x}_2, \dots, \bar{x}_p) \\
\bar{x}_j = \sum_{i=1}^n x_{ij}/n
$$
또한 $\chi$는 각 설명변수 열에서 평균을 뺀 중심화된(centered) 행렬로,
$$
\chi =
\begin{pmatrix}
x_{11}-\bar{x}_1 & x_{12}-\bar{x}_2 & \cdots & x_{1p}-\bar{x}_p\\
x_{21}-\bar{x}_1 & x_{22}-\bar{x}_2 & \cdots & x_{2p}-\bar{x}_p\\
\vdots & \vdots & \ddots & \vdots\\
x_{n1}-\bar{x}_1 & x_{n2}-\bar{x}_2 & \cdots & x_{np}-\bar{x}_p
\end{pmatrix}
$$
이 식은 $h_{ii}$가 단순히 각 설명변수의 개별 크기가 아니라, **설명변수 벡터 전체가 평균 벡터로부터 얼마나 멀리 떨어져 있는지**를 반영함을 보여준다.

### 8.2.2 해트 행렬 $H$의 대각원소 (Diagonal Elements of the Hat Matrix)
식 $\hat{y} = Hy$로부터 $i$번째 적합값(fitted value) $\hat{y}_i$는
$$
\hat{y}_i
= h_{i1}y_1 + h_{i2}y_2 + \cdots + h_{in}y_n
= \sum_{j=1}^n h_{ij}y_j
$$
여기서 가중치(weight) $h_{ij}$는 각 관측값 $y_j$가 $i$번째 적합값 $\hat{y}_i$에 기여하는 정도를 뜻한다.  
또한 앞에서 $h_{ii} = \sum_{j=1}^n h_{ij}^2$ 가 성립함을 보였으므로, 대각원소 $h_{ii}$는 $i$번째 관측값이 전체 적합값들에 미치는 영향 정도, 즉 지렛대(leverage)를 나타낸다고 해석할 수 있다.  
더 나아가 $H$의 $i$번째 대각원소는
$$
\frac{\partial \hat{y}_i}{\partial y_i} = h_{ii}
$$
를 만족한다. 따라서 $h_{ii}$가 1에 가까울수록, $i$번째 관측값 $y_i$가 자기 자신의 적합값 $\hat{y}_i$에 큰 영향을 준다는 뜻이다.

한편 해트 행렬의 트레이스(trace)는 $\operatorname{tr}(H)=\sum_{i=1}^n h_{ii}=p+1$이므로, 대각원소의 평균(mean)은
$$
\bar{h}=\frac{p+1}{n}
$$
Belsley 등의 기준:
* $h_{ii} > 2\bar{h}$ 이면 큰 지렛대점을 의심할 수 있다.
* 표본이 작은 경우에는 $2\bar{h}$보다 큰 값이 많아질 수 있으므로 $3\bar{h}$를 기준으로 사용하기도 한다.

다만 **큰 지렛대점을 가진다고 해서 곧바로 영향점(influential point)인 것은 아니다.**
또한 큰 지렛대점을 통해 관측값의 특이성을 평가할 때는 **반응변수 $y$를 제외하고 설명변수 값에만 의존하여 판단한다**는 점에 유의해야 한다.

### 8.2.3 마할라노비스 거리 (Mahalanobis Distance)
관측값 $x_i^T = (x_{i1}, x_{i2}, \dots, x_{ip})$가 관측값들의 평균 $\bar{x}^T = (\bar{x}_1, \bar{x}_2, \dots, \bar{x}_p)$로부터 어느 정도 멀리 떨어져 있는지를 측정하는 척도로 마할라노비스 거리(Mahalanobis distance)가 사용된다. 이는 다변량통계분석(multivariate statistical analysis)에서 널리 쓰이는 거리 개념이다.

$i$번째 관측값의 마할라노비스 거리 $M(i)$는 다음과 같이 정의된다.
$$
M(i) = (\mathbf{x}_i-\bar{\mathbf{x}})^T \left[\frac{{\chi}^T \chi}{n-1} \right]^{-1} (\mathbf{x}_i-\bar{\mathbf{x}})
$$
이 식에 앞서 얻은 $h_{ii}$의 표현을 이용하면,
$$
M(i) = (n-1)\left(h_{ii}-\frac1n\right)
$$
로 간단히 쓸 수 있다. 즉, $h_{ii}$가 클수록 $M(i)$도 비례하여 커진다. 따라서 마할라노비스 거리 $M(i)$가 큰 관측값은 다른 $n-1$개의 관측값들로부터 멀리 떨어져 있으므로, 지렛대점(leverage point)으로 주목할 필요가 있다.


## 8.3 이상점 탐색 (Outlier Detection)
이상점(outlier)은 자료 전처리(data preprocessing) 과정에서 식별할 수도 있지만, 회귀분석(regression analysis)에서는 **잔차(residual)** 에 기반하여 판별하는 것이 일반적이다. 이는 관측값(observation)이 모형의 적합값(fitted value)보다 지나치게 크거나 작을 때 잔차가 크게 나타나기 때문이다.  
회귀진단(regression diagnostics)에서 이상점 탐색은 단순히 잔차의 절대값만 보는 것이 아니라, 지렛대(leverage), 분산(variance), 표준화(standardization), 그리고 관측값 제거 시의 변동까지 고려하여 수행한다.

### 8.3.1 잔차의 성질 (Properties of Residuals)
보통 잔차벡터(residual vector)는 다음과 같이 정의된다. $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = \mathbf{y} - X\hat{\boldsymbol{\beta}}$, 최소제곱추정량(least squares estimator) $\hat{\boldsymbol{\beta}} = (X^TX)^{-1}X^T\mathbf{y}$를 대입하면, 
$$
\mathbf{e} = \mathbf{y} - X(X^TX)^{-1}X^T\mathbf{y}
$$
이고, 해트 행렬(hat matrix) $H = X(X^TX)^{-1}X^T$를 이용하면
$$
\mathbf{e} = (I_n - H)\mathbf{y}
$$
이로부터 잔차벡터의 기댓값 벡터(expected value vector)와 분산-공분산 행렬(variance-covariance matrix)은
$$
E(\mathbf{e})=0_n,\qquad \mathrm{Var}(\mathbf{e})=\sigma^2(I_n-H)
$$
따라서 $i$번째 잔차 $e_i$에 대해서는
$$
E(e_i)=0,\qquad \mathrm{Var}(e_i)=\sigma^2(1-h_{ii})
$$
를 얻는다. 즉, 각 잔차의 분산은 모두 동일하지 않고, 해트 행렬의 대각원소(diagonal element) $h_{ii}$에 따라 달라진다. 특히 $h_{ii}$가 큰 관측값은 잔차의 분산이 작아지는 경향이 있다.  
또한 $e_i$와 $e_j$의 공분산(covariance)은
$$
\mathrm{Cov}(e_i,e_j)=-h_{ij}\sigma^2
$$
이고, 따라서 상관계수(correlation coefficient)는
$$
\rho_{ij} =\frac{-h_{ij}\sigma^2}{\sqrt{\sigma^4(1-h_{ii})(1-h_{jj})}}
= \frac{-h_{ij}}{\sqrt{(1-h_{ii})(1-h_{jj})}}
$$
즉, 잔차들은 서로 독립(independent)이 아니며, 해트 행렬의 비대각원소(off-diagonal element)에 의해 상관되어 있다. 이 점은 이상점 판별에서 단순한 독립 정규표본처럼 취급할 수 없음을 뜻한다.

### 8.3.2 이상점을 찾는 방법 (Methods for Detecting Outliers)
이 절에서는 이상점을 찾는 대표적인 방법으로 다음 두 가지를 다룬다.

1. 표준화 잔차(standardized residual)
2. 스튜던트화 잔차(studentized residual)

#### 표준화 잔차 (Standardized Residual)
관측벡터 $\mathbf{y}$가 정규분포(normal distribution)
$\mathbf{y} \sim N(X\boldsymbol{\beta},\sigma^2 I_n)$를 따르면, 식 (8.16)으로부터 잔차벡터는
$\mathbf{e} \sim N(0_n,\sigma^2(I_n-H))$를 따른다.  
따라서 $i$번째 잔차는
$$
e_i \sim N(0,\sigma^2(1-h_{ii}))
$$
이 식은 중요한 의미를 가진다. 잔차의 분산이 $(1-h_{ii})$에 비례하므로, **큰 지렛대점(high leverage point)은 본질적으로 작은 잔차를 가질 가능성이 있다.** 따라서 단순히 잔차 크기만 비교하면 큰 지렛대점을 가진 이상점을 놓칠 수 있다.  
이를 보정하기 위해
$$
\frac{e_i}{\sigma\sqrt{1-h_{ii}}} \sim N(0,1)
$$
를 생각할 수 있다. 그러나 $\sigma$는 보통 미지(unknown)이므로, 이를 불편추정량(unbiased estimator)
$$
s = \sqrt{\frac{\mathbf{e}^T\mathbf{e}}{n-p-1}} = \sqrt{\frac{\mathbf{y}^T(I_n-H)\mathbf{y}}{n-p-1}}
$$
로 대체한다. 그러면 표준화 잔차(standardized residual)
$$
r_i=\frac{e_i}{s\sqrt{1-h_{ii}}}
\tag{8.20}
$$
를 얻는다. 다만 이 $r_i$의 표본분포(sampling distribution)는 단순한 $t$-분포로 근사할 수 없다. 이유는 분자 $e_i$와 분모 $s$가 서로 독립이 아니기 때문이다. 특히 $|e_i|$가 크면 $s$ 역시 커지는 경향이 있다.  
식 (8.20)의 $r_i$를 **내적 스튜던트화 잔차(internally studentized residual)** 라고도 부른다.

#### 스튜던트화 잔차 (Studentized Residual)
표준화 잔차의 한계를 보완하기 위해, $i$번째 관측값 $y_i$를 제외한 나머지 $n-1$개 자료로부터 오차표준편차(error standard deviation)를 추정한다. 즉, 식 (8.20)의 $s$ 대신, $i$번째 관측값을 제외하고 계산한 $s(i)$를 사용한다. 그러면 스튜던트화 잔차(studentized residual)
$$
r_i^*=\frac{e_i}{s(i)\sqrt{1-h_{ii}}}
\tag{8.21}
$$
를 얻는다. 이를 $r_i$와 구별하여 **외적 스튜던트화 잔차(externally studentized residual)** 라고 부른다. 직관적으로는 $i$번째 관측값이 이상점이면 그 점 자체가 전체 오차분산 추정에 영향을 미치므로, 그 점을 뺀 뒤 분산을 다시 추정하는 것이 더 공정한 판단 기준이 된다.

#### $s(i)^2$와 $\hat{\beta}(i)$의 관계 (Relations for $s(i)^2$ and $\hat{\beta}(i)$)
처음에는 $r_i^*$를 계산하려면 $i=1,\dots,n$ 각각에 대해 별도의 회귀모형을 $n$번 적합해야 할 것처럼 보인다. 그러나 그럴 필요는 없다: 다음 관계식이 성립한다.
$$
s^2(i) = \frac{(n-p-1)s^2-\dfrac{e_i^2}{1-h_{ii}}}{n-p-2}
\tag{8.22}
$$
즉, 전체 자료를 한 번만 적합한 결과로부터 각 $s(i)$를 모두 계산할 수 있다.  
또한 $i$번째 관측값 $y_i$와 설명변수 벡터 $\mathbf{x}_i=(1,x_{i1},x_{i2},\dots,x_{ip})^T$를 제외한 $n-1$개의 관측값으로 구한 회귀계수추정량을 $\hat{\boldsymbol{\beta}}(i)$라 하면,
$$
\hat{\boldsymbol{\beta}}(i) = \hat{\boldsymbol{\beta}} - \frac{(X^TX)^{-1}\mathbf{x}_i e_i}{1-h_{ii}}
$$
가 성립한다. 즉, 한 관측값을 제거했을 때 회귀계수가 얼마나 변하는지도 전체 적합 결과와 잔차, 지렛대 값만으로 계산 가능하다.

#### 이상점 검정을 위한 이론적 절차 (Theoretical Procedure for Outlier Testing)
$y_i$가 이상점인지 판정하기 위한 이론적 절차는 다음과 같다.

**Step1** 전체 $n$개 데이터에서 $y_i$를 제거한다.

**Step2** $i$번째 관측값을 제외한 $n-1$개 데이터로부터 $\boldsymbol{\beta}$와 $\sigma^2$의 추정값 $\hat{\boldsymbol{\beta}}(i)$, $s^2(i)$를 구한다.

**Step3** $E(y_i)$의 추정값을 $\tilde{y}_i = \mathbf{x}_i^T\hat{\boldsymbol{\beta}}(i)$로 구한다. 이때 그 분산은
$$
\mathrm{Var}(\tilde{y}_i) = \mathbf{x}_i^T\left[X(i)^TX(i)\right]^{-1}\mathbf{x}_i \sigma^2
$$
이다. 여기서 $X(i)$는 $X$행렬에서 $i$번째 행을 제거한 행렬이다. $\sigma^2$ 대신 $s^2(i)$를 대입하면 분산의 추정값을 얻을 수 있다.

**Step4** 만약 $y_i$가 이상점이라면 $E(y_i-\tilde{y}_i)\neq 0$일 것이고, 이상점이 아니라면 $E(y_i-\tilde{y}_i)=0$이다.  
여기서 $\tilde{y}_i$를 계산할 때 $y_i$는 사용되지 않았으므로 $y_i$와 $\tilde{y}_i$는 서로 독립이고, $y_i$와 $s^2(i)$도 독립이다.  
또한
$$
\mathrm{Var}(y_i-\tilde{y}_i)
= \sigma^2 \left\{1+\mathbf{x}_i^T\left[X(i)^TX(i)\right]^{-1}\mathbf{x}_i
\right\}
\tag{8.24}
$$
가 되므로, $\sigma^2$ 대신 $s^2(i)$를 대입하면 이상점이 아닐 때
$$
t_i= \frac{y_i-\tilde{y}_i}{s(i)\sqrt{1+\mathbf{x}_i^T[X(i)^TX(i)]^{-1}\mathbf{x}_i}}
$$
는 자유도(degree of freedom) $n-p-2$인 $t$-분포를 따른다.

#### $t_i$와 스튜던트화 잔차의 동치성 (Equivalence of $t_i$ and the Studentized Residual)
식 (8.24)는 계산상 복잡해 보이지만, Beckman과 Trussell은 다음 관계를 밝혔다.
$$
t_i = r_i \left( \frac{n-p-2}{n-p-1-r_i^2} \right)^{1/2} 
\tag{8.25}
$$
그리고 식 (8.20), (8.21), (8.22), (8.25)의 관계를 이용하면
$$
r_i^* = \frac{e_i}{s(i)\sqrt{1-h_{ii}}} = r_i \left( \frac{n-p-2}{n-p-1-r_i^2} \right)^{1/2}
= t_i
\tag{8.26}
$$
즉, **외적 스튜던트화 잔차(externally studentized residual) $r_i^*$** 는 $t_i$와 동일하며, 자유도 $n-p-2$인 $t$-분포를 따른다. 따라서 $i$번째 관측값 $y_i$가 이상점인지 검정하는 방법은
$$
|r_i^*| \ge t_{\alpha/2}(n-p-2)
$$
이면, 유의수준(significance level) $\alpha$에서 $y_i$를 이상점으로 판정하는 것이다.

또한 자료 수 $n$이 크면 식 (8.26)의 제곱근 항이 1에 가까워지므로, **스튜던트화 잔차(studentized residual)** 와 **표준화 잔차(standardized residual)** 의 차이는 거의 없어지고, 두 값은 단조함수(monotone function) 관계에 있으므로 잔차 크기의 순서(order)도 동일해진다.

#### 예제 8.1
두 변수 $x, y$에 대하여 다음의 21개 데이터가 주어졌다고 하자. 이 데이터는 Andrews와 Pregibon, John과 Draper, Michy 등의 논문에서 사용된 것으로, 회귀진단에서 자주 사용되는 자료라고 설명한다. 문제는 **단순회귀모형(simple linear regression model)** 을 적합하고, **지렛대점(leverage point)** 과 **이상점(outlier)** 의 유무를 판정하는 것이다.

주어진 데이터는 다음과 같다.
| 실험번호 | $x$ | $y$ |
| ---- | --: | --: |
| 1    |  15 |  95 |
| 2    |  26 |  71 |
| 3    |  10 |  83 |
| 4    |   9 |  91 |
| 5    |  15 | 102 |
| 6    |  20 |  87 |
| 7    |  18 |  93 |
| 8    |  11 | 100 |
| 9    |   8 | 104 |
| 10   |  20 |  94 |
| 11   |   7 | 113 |
| 12   |   9 |  96 |
| 13   |  10 |  83 |
| 14   |  11 |  84 |
| 15   |  11 | 102 |
| 16   |  10 | 100 |
| 17   |  12 | 105 |
| 18   |  42 |  57 |
| 19   |  17 | 121 |
| 20   |  11 |  86 |
| 21   |  10 | 100 |

전체 21개 데이터를 이용하여 회귀직선(regression line)을 구하면
$\hat{y}_i = 109.874 - 1.127x_i$를 얻는다.  
이 단순선형회귀모형(simple linear regression model)의 분산분석표(analysis of variance table)는 다음과 같다.
| 요인              | 제곱합 (sum of squares) | 자유도 (df) | 평균제곱 (mean square) | $F_0$ |
| --------------- | -------------------: | -------: | -----------------: | ----: |
| 회귀 (regression) |              1604.08 |        1 |            1604.08 | 13.20 |
| 잔차 (residual)   |              2308.59 |       19 |            121.505 |       |
| 계 (total)       |                 3912 |       20 |                    |       |

여기서 $F_0 = 13.20 > F_{0.05}(1,19)=4.38$이므로 회귀직선은 유의하다(significant).  
또한 잔차의 평균제곱(mean square error, MSE)이 121.505이므로 $\sigma$의 추정값은
$$
s=\sqrt{MSE}=\sqrt{121.505}=11.0229
$$

이제 해트 행렬의 대각원소 $h_{ii}$, 잔차 $e_i$, 표준화 잔차 $r_i$, 스튜던트화 잔차 $r_i^*$를 구하면 표와 같은 결과를 얻는다.

| 실험번호 |    $e_i$ | $h_{ii}$ |   $r_i$ | $r_i^*$ |
| ---- | -------: | -------: | ------: | ------: |
| 1    |   2.0310 |   0.0479 |  0.1888 |  0.1840 |
| 2    |  -9.5721 |   0.1545 | -0.9444 | -0.9416 |
| 3    | -15.6040 |   0.0628 | -1.4623 | -1.5108 |
| 4    |  -8.7309 |   0.0705 | -0.8216 | -0.8143 |
| 5    |   9.0310 |   0.0479 |  0.8397 |  0.8329 |
| 6    |  -0.3741 |   0.0726 | -0.0315 | -0.0306 |
| 7    |   3.4120 |   0.0580 |  0.3189 |  0.3112 |
| 8    |   2.5230 |   0.0567 |  0.2357 |  0.2297 |
| 9    |   3.1421 |   0.0799 |  0.2972 |  0.2899 |
| 10   |   6.6666 |   0.0726 |  0.6280 |  0.6177 |
| 11   |  11.0151 |   0.0908 |  1.0480 |  1.0508 |
| 12   |  -3.7309 |   0.0705 | -0.3511 | -0.3428 |
| 13   | -15.6040 |   0.0628 | -1.4623 | -1.5108 |
| 14   | -13.4770 |   0.0567 | -1.2588 | -1.2798 |
| 15   |   4.5230 |   0.0567 |  0.4225 |  0.4132 |
| 16   |   1.3961 |   0.0628 |  0.1308 |  0.1274 |
| 17   |   8.6500 |   0.0521 |  0.8060 |  0.7983 |
| 18   |  -5.5403 |   0.6516 | -0.8515 | -0.8451 |
| 19   |  30.2850 |   0.0531 |  2.8234 |  3.6070 |
| 20   | -11.4770 |   0.0567 | -1.0720 | -1.0765 |
| 21   |   1.3961 |   0.0628 |  0.1308 |  0.1274 |

이 결과를 해석하면,  
첫째, 큰 지렛대점(high leverage point)의 기준으로 $2\bar{h}$ 또는 $3\bar{h}$를 사용한다.
여기서 단순회귀이므로 $p=1$이고, 평균 지렛대(mean leverage)는
$\bar{h}=\frac{p+1}{n}=\frac{2}{21}$이므로 $2\bar{h}=\frac{4}{21}=0.1905,\quad 3\bar{h}=\frac{6}{21}=0.2857$

표에서 보면 **18번째 관측값**의 $h_{18,18}=0.6516$이므로, 이는 $2\bar{h}$, $3\bar{h}$를 모두 크게 초과하는 **극단적으로 큰 지렛대점(extremely large leverage point)** 이다. 실제로 이 점은 $x=42$로 다른 $x$값들에 비해 현저히 멀리 떨어져 있다.

둘째, 이상점(outlier)은 외적 스튜던트화 잔차 $r_i^*$로 판정한다.
이때 $t_{0.025}(n-p-2)=t_{0.025}(18)=2.101$ 이므로, $|r_i^*| > 2.101$인 경우 이상점으로 판정할 수 있다.

표에서 이를 만족하는 것은 **19번째 관측값**뿐이다.
실제로 19번째 관측값은 $r_{19}=2.8234,\quad r_{19}^*=3.6070$으로 나타나며, 잔차도 $e_{19}=30.2850$으로 가장 크다. 따라서 **19번째 관측값은 이상점(outlier)**이다.

정리하면 예제의 결론은 다음과 같다.
* **18번째 관측값**: 매우 큰 지렛대점(high leverage point)
* **19번째 관측값**: 이상점(outlier)


## 8.4 영향점 탐색 (Influential Points)
영향점(influential observation)은 앞 절들에서 간단히 소개한 바와 같이, **특정 관측값을 포함하거나 제외할 때 회귀분석 결과가 크게 달라지게 만드는 관측값**이다. 이 절에서는 영향점을 찾아내는 데 널리 사용되는 몇 가지 대표적인 척도(measure)를 소개한다.

### 8.4.1 DFFITS 
DFFITS는 Belsley 등이 제안한 영향점 탐색 척도이다. 핵심 아이디어는 **전체 자료로 추정한 회귀계수 $\hat{\boldsymbol{\beta}}$** 와 **$i$번째 관측값을 제외한 뒤의 회귀계수 $\hat{\boldsymbol{\beta}}(i)$** 사이의 차이가 얼마나 큰지 측정하는 것이다.  
여기서 $\hat{\boldsymbol{\beta}}(i)$는 $y_i$를 제외하고 $n-1$개 자료에서 구한 $\boldsymbol{\beta}$의 최소제곱추정벡터(least squares estimate vector)이다.  
$i$번째 관측값의 DFFITS는
$$
\mathrm{DFFITS}(i) = \operatorname{sgn}(e_i)\cdot
\sup_{\lambda}
\frac{\left|\lambda^T(\hat{\boldsymbol{\beta}}-\hat{\boldsymbol{\beta}}(i))\right|}
{s(i){\lambda^T(X^TX)^{-1}\lambda}^{1/2}}
\tag{8.27}
$$
로 정의된다. 여기서 $\operatorname{sgn}(e_i)$는 부호함수(sign function)로,
* $e_i>0$이면 $(+)$
* $e_i=0$이면 $(0)$
* $e_i<0$이면 $(-)$

식 (8.27)은 다음과 같이 쓸 수도 있다.
$$
\mathrm{DFFITS}(i)
= \operatorname{sgn}(e_i)
\frac{\left[(\hat{\boldsymbol{\beta}}-\hat{\boldsymbol{\beta}}(i))^TX^TX(\hat{\boldsymbol{\beta}}-\hat{\boldsymbol{\beta}}(i))\right]^{1/2}}
{s(i)}
\tag{8.28}
$$
또한 Belsley 등은 이것이 다음과 동치임을 보였다.
$$
\mathrm{DFFITS}(i) = \frac{\hat{y}_i-\tilde{y}_i(i)}
{\sqrt{\mathrm{Var}(\hat{y}_i)\text{의 추정값}}}
\tag{8.29}
$$
여기서 $\tilde{y}_i(i)=x_i^T\hat{\boldsymbol{\beta}}(i)$는 $i$번째 데이터를 제외한 $n-1$개 자료에서 얻은 적합값(fitted value)이다.  
이 값은 계산 편의상 다음처럼 더 간단히 표현된다.
$$
\mathrm{DFFITS}(i)
= \left(\frac{h_{ii}}{1-h_{ii}}\right)^{1/2}
\frac{e_i}{s(i)\sqrt{1-h_{ii}}}
= \left(\frac{h_{ii}}{1-h_{ii}}\right)^{1/2}r_i^*
\tag{8.30}
$$
즉, DFFITS는 **해트 행렬의 대각원소 $h_{ii}$** 와 **외적 스튜던트화 잔차(externally studentized residual) $r_i^*$** 의 결합으로 표현된다.  
대략적으로 $|r_i^*|\ge 2$이고 $h_{ii}\ge (p+1)/n$일 때 $i$번째 관측값을 영향점으로 판정한다면,
$$
|\mathrm{DFFITS}(i)|
\ge 2\left[\frac{(p+1)/n}{1-(p+1)/n}\right]^{1/2}
= 2\left[\frac{p+1}{n-p-1}\right]^{1/2}
\geq 2\left[\frac{p+1}{n}\right]^{1/2}
\tag{8.31}
$$
이 되므로, 보통 $|\mathrm{DFFITS}(i)| \ge 2\sqrt{\frac{p+1}{n}}$ 정도를 영향점의 경험적 기준(rule of thumb)으로 본다.

### 8.4.2 Cook의 통계량 (Cook's Statistic)
Cook은 식 (8.28)에서 $s(i)$ 대신 전체 자료로부터 구한 $s$를 대입한 뒤, 제곱하고 $p+1$로 나눈 양을 제안하였다. 이를 **Cook의 통계량(Cook's distance / Cook's statistic)** 이라 한다.
$$
D(i) = \frac{(\hat{\boldsymbol{\beta}}-\hat{\boldsymbol{\beta}}(i))^TX^TX(\hat{\boldsymbol{\beta}}-\hat{\boldsymbol{\beta}}(i))}{(p+1)s^2}
\tag{8.32}
$$
$D(i)$가 클수록, $i$번째 관측값을 제거했을 때 회귀계수 추정값 변화가 크다는 뜻이므로 영향점으로 의심한다.  
이 값은 계산 편의상 다음과 같이 쓸 수 있다.
$$
D(i) = \frac{1}{p+1}\cdot
\frac{h_{ii}e_i^2}{s^2(1-h_{ii})^2}
\tag{8.33}
$$
또한 표준화 잔차(standardized residual) $r_i$와의 관계는
$$
D(i) = \frac{r_i^2}{p+1}\cdot \frac{h_{ii}}{1-h_{ii}}
\tag{8.34}
$$
이 식으로부터 다음 사실을 알 수 있다.
* $r_i^2$가 크면 $D(i)$도 커진다.
* $h_{ii}$가 크면 $D(i)$도 커진다.
* 즉, **잔차가 크고 지렛대도 큰 관측값**은 Cook의 통계량이 커져 영향점이 되기 쉽다.

6장에서 $\beta$의 공동신뢰영역(joint confidence region)이
$$
\frac{(\hat{\boldsymbol{\beta}}-\boldsymbol{\beta})^TX^TX(\hat{\boldsymbol{\beta}}-\boldsymbol{\beta})}{(p+1)s^2}
\ge F_{\alpha}(p+1,n-p-1)
$$
이었고, 이 식에서 $\boldsymbol{\beta}$ 대신 $\hat{\boldsymbol{\beta}}(i)$를 대입하여 생각하면
$$
D(i)\ge F_{\alpha}(p+1,n-p-1)
$$
이면 해당 관측값을 영향점으로 판정할 수 있다. 또한 Cook은 경험적으로
$$
D(i)\ge F_{0.50}(p+1,n-p-1)
$$
이면 일단 영향을 크게 주는 측정값으로 의심해도 좋다고 제안하였다.

### 8.4.3 Andrews-Pregibon의 통계량 (Andrews-Pregibon Statistic)
Andrews와 Pregibon은 행렬 $X$와 벡터 $\mathbf{y}$를 함께 고려하는 다음의 통계량을 제안하였다.
$$
AP(i)=\frac{|X^{*}(i)^\top X^{*}(i)|}{|X^{*\top} X^*|}
\tag{8.35}
$$
* $X^*=(X,\mathbf{y})$
* $X^*(i)=(X(i),\mathbf{y}(i))$

즉, $X^*(i)$는 $X^*$ 행렬에서 $i$번째 행을 제거한 것이다.  
John과 Draper는 이 통계량이 다음과 같음을 밝혔다.
$$
AP(i) = \left[ 1-\frac{e_i^2}{(1-h_{ii})SSE} \right](1-h_{ii})
= 1-h_{ii}-\frac{e_i^2}{(n-p-1)s^2}
\tag{8.36}
$$
이 등식으로부터 $AP(i)$ 계산이 매우 쉬워진다.  
해석은 다음과 같다.
* $h_{ii}$가 크면 $AP(i)$는 작아진다.
* $e_i^2$가 크면 $AP(i)$도 작아진다.
* 따라서 **가장 작은 $AP(i)$** 값을 갖는 관측값을 영향점으로 의심할 수 있다.

즉, Andrews-Pregibon의 통계량은 다른 척도들과 달리 **작을수록 영향이 큰 점**이라는 방향을 가진다.

### 8.4.4 COVRATIO
회귀계수 추정값의 분산-공분산행렬(variance-covariance matrix)은
$\mathrm{Var}(\hat{\boldsymbol{\beta}})=\sigma^2(X^TX)^{-1}$이고, $\mathrm{Var}(\hat{\boldsymbol{\beta}}(i))=\sigma^2[X(i)^TX(i)]^{-1}$  
여기서 $\sigma^2$ 대신 각각 $s^2$, $s^2(i)$를 대입한 뒤, 그 행렬식(determinant)의 비율을 취한 값을 **COVRATIO**라고 한다. 이 값은 일반화분산(generalized variance)의 비율이다.
$$
\mathrm{COVRATIO}(i) = \frac{|s^2(i)[X(i)^TX(i)]^{-1}|}{|s^2(X^TX)^{-1}|} \\
= \left[\frac{s^2(i)}{s^2}\right]^{p+1}\cdot \frac{1}{1-h_{ii}}
$$
이 되고, 식 (8.21), (8.22)의 관계를 사용하면
$$
\mathrm{COVRATIO}(i)
= \frac{1}{\left[1+\dfrac{(r_i^*)^2-1}{n-p-1}\right]^{p+1}(1-h_{ii})}
\tag{8.38}
$$
이 값의 해석은 다음과 같다.
* $\mathrm{COVRATIO}(i)$가 **1에 가까우면** $y_i$는 회귀계수 추정의 분산-공분산 구조에 별 영향을 주지 않는다.
* $\mathrm{COVRATIO}(i)$가 **1에서 멀어질수록** 해당 관측값은 회귀계수 추정의 안정성에 큰 영향을 준다.

Belsley 등은 다음 기준을 제안하였다.
$$
|\mathrm{COVRATIO}(i)-1|
\ge
\frac{3(p+1)}{n}
$$
이면 $i$번째 관측값을 영향을 크게 주는 측정값으로 볼 수 있다.

### 8.4.5 FVARATIO
적합값 $\hat{y}_i$와, $i$번째 자료를 제거한 뒤의 적합값 $\hat{y}_i(i)$의 분산은 각각
$\mathrm{Var}(\hat{y}_i)=h_{ii}\sigma^2, \ \mathrm{Var}(\hat{y}_i(i)) = \frac{h_{ii}}{1-h_{ii}}\sigma^2
$로 주어진다.  
여기서 $\sigma^2$를 각각 $s^2$, $s^2(i)$로 대체하고 그 비율을 취하면 FVARATIO를 얻는다. 식 (8.21)의 정의를 사용하면
$$
\mathrm{FVARATIO}(i)
= \frac{s^2(i)}{s^2(1-h_{ii})}
= \frac{e_i^2}{(r_i^*)^2(1-h_{ii})^2s^2}
$$
Belsley 등은 다음 기준을 제안하였다.
* $\mathrm{FVARATIO}(i)\le 1-\dfrac{3}{n}$ 또는
* $\mathrm{FVARATIO}(i)\ge 1+\dfrac{2p+3}{n}$

이면 $i$번째 관측값을 영향을 크게 주는 측정값으로 본다.  
즉 FVARATIO 역시 **1에서 벗어나는 정도**로 영향력을 평가하는 척도이다.


## 8.5 이상한 관측값 탐지: 지렛대점, 이상점과 영향점 (Detection of Leverage, Outliers, and Influential Points)
이 절은 앞에서 따로따로 살펴본 **지렛대점(leverage point)**, **이상점(outlier)**, **영향점(influential point)** 의 척도들을 한 번에 정리하고, 이들 사이의 관계 및 실제 데이터 해석 방법을 설명한다.

### 8.5.1 측도의 종합검토 (Comprehensive Review of Measures)
앞 절들에서 다음을 다루었다.
* 지렛대점을 찾는 해트 행렬의 대각원소(diagonal element of hat matrix)와 마할라노비스 거리(Mahalanobis distance)
* 이상점을 탐지하는 두 종류의 잔차(residual)
* 영향점을 검출하는 다섯 가지 방법

#### 이상한 관측값 탐지에 사용되는 척도 정리 (Summary of Measures)
**지렛대점 (Leverage Point)**
1. 해트 행렬 $(H)$의 대각원소(diagonal element)
    $$h_{ii}$$
2. 마할라노비스 거리(Mahalanobis distance)
    $$M(i)=(n-1)\left(h_{ii}-\frac{1}{n}\right)$$

**이상점 (Outlier)**
1. 표준화 잔차(standardized residual)
    $$r_i=\frac{e_i}{s\sqrt{1-h_{ii}}}$$
2. 스튜던트화 잔차(studentized residual)
    $$r_i^*=\frac{e_i}{s(i)\sqrt{1-h_{ii}}}
    = r_i \left( \frac{n-p-2}{n-p-1-r_i^2} \right)^{1/2}$$

**영향점 (Influential Point)**
1. DFFITS
    $$\mathrm{DFFITS}(i)= \left(\frac{h_{ii}}{1-h_{ii}}\right)^{1/2}r_i^*$$
2. Cook의 통계량(Cook's statistic)
    $$D(i)=\frac{h_{ii}}{(p+1)(1-h_{ii})}r_i^2$$
3. Andrews-Pregibon의 통계량(Andrews-Pregibon statistic)
    $$AP(i)=1-h_{ii}-\frac{e_i^2}{(n-p-1)s^2}$$
4. COVRATIO
    $$\mathrm{COVRATIO}(i)
    = \frac{1} {\left[1+\dfrac{(r_i^*)^2-1}{n-p-1}\right]^{p+1}(1-h_{ii})}$$
5. FVARATIO
    $$\mathrm{FVARATIO}(i)
    = \frac{e_i^2}{(r_i^*)^2(1-h_{ii})^2s^2}$$

#### 지렛대점, 이상점, 영향점의 관계 (Relations Among Leverage, Outliers, and Influential Points)
실제 자료에서 하나의 측정값이 이상점이면 동시에 영향점이 되고, 영향점이면 또한 이상점이 되기도 한다. 그러나 이것이 항상 참인 것은 아니고, 어떤 경우에는 이상점을 제거해도 회귀분석 결과(회귀계수, 결정계수 등)에 큰 변화가 없어 영향점이 아닌 경우가 있다. 반대로 어떤 경우에는 잔차가 작은 측정값이, 잔차가 큰 측정값보다 오히려 회귀결과에 더 큰 영향을 줄 수도 있다.

이 절에서는 위에 정리한 아홉 가지 척도에 대해 다음 사항을 생각해 볼 수 있다  

(1) **지렛대점 척도 $h_{ii}$, $M(i)$**  
이 두 척도는 반응변수 $y$값과는 직접 관련이 없고, 설명변수행렬 $X$에만 관련된 척도이다. $h_{ii}$와 $M(i)$가 크면 $i$번째 관측값 $x_i$가 다른 $x_j$ $(j\ne i)$들과 멀리 떨어져 있다고 해석할 수 있다.
따라서 두 값이 클 때 $i$번째 관측값을 큰 지렛대점(leverage point)이라 부른다.

(2) **이상점 척도 $r_i$, $r_i^*$**  
이 두 방법은 모두 잔차 $e_i$의 함수이므로 $|e_i|$가 커지면 값도 커진다.
일반적으로 잔차가 매우 크면 이상점이라고 볼 수 있다. 또한 이 척도들은 $h_{ii}$의 함수이기도 하므로 $h_{ii}$가 커지면 값도 커질 수 있다.  
따라서 $h_{ii}$가 크고 $e_i$도 크면 이상점으로 판정할 가능성이 커진다.

(3) **영향점 척도 DFFITS, $D(i)$, $AP(i)$, COVRATIO, FVARATIO**  
이 척도들은 모두 추정된 회귀계수의 변화를 검출하는 방식으로 설계되었지만, 결국 모두 잔차 $e_i$와 $h_{ii}$의 함수로 나타난다.  
따라서 **$h_{ii}$와 $e_i$가 모두 큰 경우**, 회귀결과에 큰 영향을 주는 측정값일 가능성이 높다.

### 8.5.2 이상점과 영향점의 취급 (Treatment of Outliers and Influential Points)
이상점(outlier)으로 판정되었다고 해서 그것을 무조건 버리고 나머지 관측값만으로 회귀분석을 다시 하는 것은 좋은 방법이 아니다. 먼저 **왜 그 점이 이상점이 되었는지 원인을 규명**해야 한다.

그 이유는 다음과 같이 다양할 수 있다.
* 실험이 잘못되었을 가능성
* 데이터가 잘못 기록되었을 가능성
* 원료가 잘못 사용되었을 가능성
* 데이터 측정오류(measurement error)가 있었을 가능성

데이터가 실험으로 관측된 경우, 이상점의 원인이 밝혀진다면 가능하면 다시 실험을 수행하여 데이터를 대체하는 것이 바람직하다. 재실험이 불가능한 경우에는 해당 데이터를 제거하는 것도 고려할 수 있다.

또한 영향점으로 판정된 경우에도, 먼저 왜 그 점이 큰 영향을 주는지 해석해 보아야 한다.

예를 들어 실험 데이터에서 다른 실험점들로부터 멀리 떨어져 있는 것이 원인이라면, 그 사이에 몇 개의 실험점을 추가하여 실험을 다시 수행한 뒤 회귀분석을 해보는 것이 바람직하다.

만약 다른 실험점으로부터 멀리 떨어져 있지 않음에도 잔차가 커서 영향을 크게 주는 측정값으로 판정되었다면, 그것은 동시에 이상점일 가능성이 크므로 이상점의 처리와 동일한 방식으로 접근할 수 있다.

결론적으로, 이상점과 영향점에 대한 검토는 회귀진단(regression diagnostics)에서 필수적이며, 분석결과를 더 정확히 이해하기 위해 반드시 필요한 과정이다.

#### 예제 8.2
앞의 예제 8.1에서 사용된 데이터를 다시 사용하여 회귀직선을 적합하고, 표 8.3에 있는 모든 척도를 계산하여 **지렛대점, 이상점, 영향점이 무엇인지**를 구하고, 그 결과를 비교 검토하는 예제이다.

표 8.3에는 다음 여섯 가지 척도의 값이 제시된다.
* $\mathrm{DFFITS}(i)$
* $D(i)$
* $M(i)$
* $AP(i)$
* $\mathrm{COVRATIO}(i)$
* $\mathrm{FVARATIO}(i)$

| 실험번호 | DFFITS$(i)$ | $D(i)$ |  $M(i)$ | $AP(i)$ | COVRATIO$(i)$ | FVARATIO$(i)$ |
| ---- | ----------: | -----: | ------: | ------: | ------------: | ------------: |
| 1    |      0.0413 | 0.0009 |  0.0061 |  0.9503 |        1.1659 |        1.1066 |
| 2    |     -0.4025 | 0.0825 |  2.1379 |  0.8058 |        1.1970 |        1.1899 |
| 3    |     -0.3911 | 0.0717 |  0.3039 |  0.8317 |        0.9363 |        0.9996 |
| 4    |     -0.2243 | 0.0256 |  0.4585 |  0.8964 |        1.1151 |        1.0953 |
| 5    |      0.1869 | 0.0177 |  0.0061 |  0.9167 |        1.0850 |        1.0675 |
| 6    |     -0.0086 | 0.0000 |  0.5000 |  0.9273 |        1.2013 |        1.1382 |
| 7    |      0.0772 | 0.0031 |  0.2074 |  0.9370 |        1.1702 |        1.1145 |
| 8    |      0.0563 | 0.0017 |  0.1810 |  0.9406 |        1.1742 |        1.1157 |
| 9    |      0.0854 | 0.0038 |  0.6448 |  0.9159 |        1.1997 |        1.1418 |
| 10   |      0.1728 | 0.0154 |  0.5000 |  0.9081 |        1.1521 |        1.1146 |
| 11   |      0.3320 | 0.0548 |  0.8027 |  0.8567 |        1.0878 |        1.0938 |
| 12   |     -0.0944 | 0.0047 |  0.4585 |  0.9234 |        1.1833 |        1.1283 |
| 13   |     -0.3911 | 0.0717 |  0.3039 |  0.8317 |        0.9363 |        0.9996 |
| 14   |     -0.3137 | 0.0476 |  0.1810 |  0.8647 |        0.9923 |        1.0256 |
| 15   |      0.1013 | 0.0054 |  0.1810 |  0.9345 |        1.1590 |        1.1085 |
| 16   |      0.0330 | 0.0006 |  0.3039 |  0.9363 |        1.1867 |        1.1253 |
| 17   |      0.1872 | 0.0179 |  0.0898 |  0.9155 |        1.0964 |        1.0755 |
| 18   |     -1.1558 | 0.6781 | 12.0498 |  0.3351 |        2.9587 |        2.9142 |
| 19   |      0.8537 | 0.2233 |  0.1086 |  0.5497 |        0.3964 |        0.6470 |
| 20   |     -0.2638 | 0.0345 |  0.1810 |  0.8863 |        1.0426 |        1.0513 |
| 21   |      0.0330 | 0.0006 |  0.3039 |  0.9363 |        1.1867 |        1.1253 |

(1) 예제 8.1의 결과에 의해 **19번째 관측값은 이상점(outlier)**이다.

(2) $h_{ii}$의 값을 보면
$$h_{ii}\ge 2(p+1)/n = 2(1+1)/21 = 0.1905$$
인 관측값은 **18번째 관측값뿐**이다. 또한 18번째 관측값은 가장 큰 $M(i)$ 값을 가진다. 실제로 18번째 관측값은 $x=42$로 다른 $x$들로부터 멀리 떨어져 있으므로 **지렛대점(leverage point)**이라 판단할 수 있다.

(3) DFFITS를 보면
$$|\mathrm{DFFITS}(i)|\ge 2\sqrt{(p+1)/n} = 2\sqrt{(1+1)/21} \approx 0.6172$$
인 경우는 **18번째와 19번째 관측값**이고, 그중 18번째가 조금 더 크다.

(4) $D(i)$를 보면
$$F_{0.50}(2,19)=0.719$$
이므로, 18번째 관측값은 Cook의 통계량 기준에서 영향점에 가장 가까운 값으로 볼 수 있다. 즉 영향을 주는 측정값에 가장 근사하다

(5) $AP(i)$는 값이 작을수록 영향을 크게 주는 관측값인데, **18번째 관측값**이 가장 작은 $AP(i)$를 가진다.

(6) COVRATIO는
$$\mathrm{COVRATIO}(i)\ge 1+\frac{3(p+1)}{n} = 1+\frac{3(1+1)}{21} \approx 1.2857$$
이거나
$$\mathrm{COVRATIO}(i)\le 1-\frac{3(p+1)}{n} \approx 0.7143$$

이면 영향을 크게 주는 관측값으로 볼 수 있다. 표에서는 **18번째와 19번째 관측값**이 모두 해당한다.

(7) 마지막으로 FVARATIO는
$$1-\frac{3}{n}=1-\frac{3}{21}=0.8571$$
보다 작거나,
$$1+\frac{2p+3}{n} = 1+\frac{2+3}{21} \approx 1.2381$$

보다 크면 영향을 크게 주는 관측값으로 판정할 수 있다. 여기에서도 **18번째와 19번째 관측값**이 모두 해당한다.

위 결과를 종합하면 다음 결론을 얻는다.

* **18번째 관측값**: 지렛대점(leverage point)이면서 영향점(influential point)
* **19번째 관측값**: 이상점(outlier)이면서 영향점(influential point)
