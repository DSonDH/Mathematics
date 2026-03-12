# Chapter 10 일반화 회귀분석 (Generalized Regression Analysis)

## 10.1 일반화최소제곱추정 (Generalized Least Squares, GLS)

지금까지의 선형회귀모형(linear regression model)은  $\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon}$에서 $\boldsymbol{\varepsilon} \sim N(\mathbf{0},\sigma^2 I_n)$

$$\text{Cov}(\varepsilon_i,\varepsilon_j)=\begin{cases}0 & i\neq j \\ \sigma^2 & i=j\end{cases}$$

즉 오차들이 서로 **독립(independent)**, 분산이 모두 동일 (**등분산성 homoscedasticity**)임을 가정한 것이다. 

그러나 실제 데이터에서는 다음과 같은 상황이 빈번하다.
* 오차분산이 서로 다름 (heteroscedasticity)
* 오차들이 상관됨 (autocorrelation)

따라서 더 일반적인 오차 구조를 고려한다.

$$\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon},\qquad \boldsymbol{\varepsilon} \sim N(\mathbf{0},\sigma^2 V)$$

* $V$ : $n\times n$ **정칙행렬(nonsingular matrix)**
* $V \neq I_n$

이 경우를 **일반다중선형회귀모형 (general multiple linear regression)** 이라 한다.

### OLS 추정량의 문제

OLS 추정량은 $\hat{\beta}=(X^TX)^{-1}X^Ty$ 그러나 $\text{Var}(\varepsilon)=\sigma^2 V$ 일 때는 이 추정량이 여전히 **불편(unbiased)** 이지만 **최량선형불편추정량(BLUE: best linear unbiased estimator)** 은 아니다. 따라서 새로운 추정량이 필요하다.

### 오차 공분산 구조의 분해

$V$는 대칭이고 양정치 행렬이므로 다음 분해가 가능하다.

$$C^TVC = A$$

* $C$ : 직교행렬 (orthogonal matrix)
* $A$ : $V$의 고유값으로 이루어진 대각행렬

따라서 $V = CAC^T$ 또는 $V=(CA^{1/2})(CA^{1/2})^T$  
여기서 $T = CA^{1/2}$ 라 두면

$$V = TT^T$$

### 선형변환 (Transformation)

모형 $y = X\beta + \varepsilon$ 양변에 $T^{-1}$을 곱한다.

$$T^{-1}y = T^{-1}X\beta + T^{-1}\varepsilon$$

다음과 같이 정의한다.

$$z=T^{-1}y \\ W=T^{-1}X \\ v=T^{-1}\varepsilon$$

그러면 모형은

$$z = W\beta + v$$

### 변환된 오차의 분산

$$\text{Var}(\mathbf{v})=\text{Var}(T^{-1}\boldsymbol{\varepsilon}) = T^{-1}\text{Var}(\boldsymbol{\varepsilon})(T^{-1})^T\\
= T^{-1}(\sigma^2V)(T^{-1})^T = \sigma^2 T^{-1}TT^T(T^{-1})^T\\
= \sigma^2 I_n$$

따라서 변환된 모형은

$$\mathbf{z}=W\boldsymbol{\beta}+\mathbf{v}, \quad \mathbf{v}\sim N(\mathbf{0},\sigma^2 I_n)$$

즉 **표준 선형회귀모형**이 된다.

### GLS 추정량

변환된 모형에서 OLS를 적용하면

$$\hat{\boldsymbol{\beta}}^*=(W^TW)^{-1}W^T\mathbf{z}$$

여기서 $W=T^{-1}X,\quad \mathbf{z}=T^{-1}\mathbf{y}$ 이므로

$$\hat{\boldsymbol{\beta}}^* =
\left[X^T(T^T)^{-1}T^{-1}X\right]^{-1}
X^T(T^T)^{-1}T^{-1}\mathbf{y}$$

또한 $V^{-1}=(T^T)^{-1}T^{-1}$ 이므로

$$\boxed{\hat{\boldsymbol{\beta}}^*=(X^TV^{-1}X)^{-1}X^TV^{-1}\mathbf{y}}$$

이를 **일반화최소제곱추정량 (generalized least squares estimator, GLS)** 이라 한다.

> 이 추정량은 또한, $\hat{\boldsymbol{\beta}}^* = \text{min}_{\boldsymbol{\beta}}(\mathbf{y}-X\boldsymbol{\beta})^\top V^{-1}(\mathbf{y}-X\boldsymbol{\beta})$  즉, $V^{-1}$로 가중된 잔차제곱합을 최소화하는 $\boldsymbol{\beta}$이기도 하다.


## 10.2 일반화최소제곱추정량의 특성 (Properties of GLS Estimator)

GLS 추정량 $\hat{\boldsymbol{\beta}}^*=(X^TV^{-1}X)^{-1}X^TV^{-1}\mathbf{y}$ 의 성질을 분석해보자

### 기대값 (Expectation)

$$E(\hat{\boldsymbol{\beta}}^*) = (X^TV^{-1}X)^{-1}X^TV^{-1}E(\mathbf{y})\\ E(\mathbf{y})=X\boldsymbol{\beta}$$

따라서

$$E(\hat{\boldsymbol{\beta}}^*) = (X^TV^{-1}X)^{-1}X^TV^{-1}X\boldsymbol{\beta} = \boldsymbol{\beta}$$

즉, GLS 추정량은 **불편추정량(unbiased estimator)** 이다.

### 분산 (Variance)

$$Var(\hat{\boldsymbol{\beta}}^*)
=Var[(X^TV^{-1}X)^{-1}X^TV^{-1}\mathbf{y}]\\
=(X^TV^{-1}X)^{-1}X^TV^{-1}Var(\mathbf{y})V^{-1}X(X^TV^{-1}X)^{-1}\\
Var(\mathbf{y})=\sigma^2V$$

따라서

$$Var(\hat{\boldsymbol{\beta}}^*) =\sigma^2 (X^TV^{-1}X)^{-1}$$

### OLS와 GLS 비교

OLS 추정량: $\hat{\boldsymbol{\beta}}=(X^TX)^{-1}X^T\mathbf{y}$  
GLS 추정량: $\hat{\boldsymbol{\beta}}^*=(X^TV^{-1}X)^{-1}X^TV^{-1}\mathbf{y}$

OLS는 $V=I_n$인 특수한 경우이다.

### 잔차 벡터
잔차 벡터는

$$\mathbf{e}^* = \mathbf{y} - X\hat{\boldsymbol{\beta}}^* \\
= \mathbf{y} - X(X^\top V^{-1}X)^{-1}X^\top V^{-1}\mathbf{y} \\
= \left[I_n - X(X^\top V^{-1}X)^{-1}X^\top V^{-1}\right]\mathbf{y} $$
로 $\mathbf{y} \sim N(X\boldsymbol{\beta}, \sigma^2 V)$이므로, $\mathbf{e}^*$의 기댓값 벡터와 분산 공분산 행렬이 각각

$$
E(\mathbf{e}^*) = [I_n - X(X^\top V^{-1}X)^{-1}X^\top V^{-1}]X\boldsymbol{\beta} = 0 \\
Var(\mathbf{e}^*) = [I_n - X(X^\top V^{-1}X)^{-1}X^\top V^{-1}](\sigma^2V) \\
= \sigma^2[V-X(X^\top V^{-1}X)^{-1}X^\top]$$

### GLS 잔차제곱합

GLS에서의 잔차제곱합(residual sum of squares)은

$$SSE(\hat{\boldsymbol{\beta}}^*)=(\mathbf{y}-X\hat{\boldsymbol{\beta}}^*)^TV^{-1}(\mathbf{y}-X\hat{\boldsymbol{\beta}}^*) \\
=(\mathbf{e}^*)^TV^{-1}(\mathbf{e}^*)$$

### SSE의 기대값

$$E[SSE(\hat{\boldsymbol{\beta}}^*)] = E[(\mathbf{e}^*)^TV^{-1}(\mathbf{e}^*)] = \text{tr}[V^{-1}Var(\mathbf{e}^*)] \\
= tr[V^{-1}(\sigma^2V - \sigma^2X(X^\top V^{-1}X)^{-1}X^\top)] \\
= \sigma^2 tr[I_n - X(X^\top V^{-1}X)^{-1}X^\top V^{-1}] \\
= \sigma^2 (n - \text{rank}(X)) = \sigma^2 (n-p-1)$$

따라서 $\sigma^2$의 불편추정량은

$$\boxed{\tilde{\sigma}^2=\frac{SSE(\hat{\boldsymbol{\beta}}^*)}{n-p-1}}$$

는 $\sigma^2$의 **불편추정량(unbiased estimator)** 이 된다.

### 회귀계수 분산의 추정
$Var(\hat{\boldsymbol{\beta}}^*)$의 불편추정량은 $Var(\hat{\boldsymbol{\beta}}^*)=\sigma^2(X^TV^{-1}X)^{-1}$ 이므로

$$\widehat{Var}(\hat{\boldsymbol{\beta}}^*)=\tilde{\sigma}^2 (X^TV^{-1}X)^{-1}$$

### OLS 분산추정량의 문제

만약 OLS에서 사용한 $\hat{\sigma}^2=MSE$를 그대로 사용하면

$$\hat{\sigma}^2 (X^TV^{-1}X)^{-1}$$

은 $Var(\hat{\boldsymbol{\beta}}^*)$의 불편추정량이 아니다. 왜냐하면 $E(\hat{\sigma}^2)\neq \sigma^2$이기 때문이다.

### 특수한 경우

#### 1. 이분산 구조 (Heteroscedasticity)

$$V=\begin{bmatrix}\sigma_1^2 & 0 & \cdots & 0 \\ 0 & \sigma_2^2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \sigma_n^2\end{bmatrix}$$

이 경우 GLS는 **가중회귀 (weighted regression)** 와 동일하다.

#### 2. 1차 자기상관 (First-order autocorrelation)

$$V=c\begin{bmatrix}1 & \rho & \rho^2 & \cdots \\ \rho & 1 & \rho & \cdots \\ \rho^2 & \rho & 1 & \cdots \\ \vdots & \vdots & \vdots & \ddots\end{bmatrix}$$

이는 시계열 데이터에서 자주 등장하는 구조로 오차 $\varepsilon_t$가 $\varepsilon_{t-1}$과 상관이 있는 경우이다.

#### 3. OLS와 GLS가 동일한 경우

McElroy 결과에 따르면, $V=(1-\rho)I_n+\rho J_n$ 일 때 $\hat{\boldsymbol{\beta}}_{OLS}=\hat{\boldsymbol{\beta}}_{GLS}$ 가 성립한다.


## 10.3 다변량회귀 (Multivariate Regression)

지금부터는 반응변수(response variable)가 하나가 아니라 **여러 개**인 경우를 다룬다. 앞에서 본 여러 회귀모형에서는 설명변수(explanatory variable)의 수는 하나 이상일 수 있었지만, 반응변수는 항상 하나였다. 그러나 실제 응용에서는 같은 설명변수 집합으로 서로 관련된 여러 반응변수를 동시에 설명해야 하는 경우가 많다.  

이때 사용하는 모형이 **다변량 선형회귀모형 (multivariate linear regression model)** 이다.

> 참고로 반응변수는 하나, 설명변수 하나: 단변량 회귀 (univariate regression)  
> 반응변수 하나, 설명변수 여럿: 다중회귀 (multiple regression)  

### 기본 설정

설명변수의 수가 $p$개이고, 반응변수의 수가 $g$개라고 하자. 각 반응변수 $y_h$에 대해 회귀식을 하나씩 세우면 다음과 같다.

$$\mathbf{y}_h=\boldsymbol{\beta}_{0h}+\boldsymbol{\beta}_{1h}x_1+\boldsymbol{\beta}_{2h}x_2+\cdots+\boldsymbol{\beta}_{ph}x_p+\boldsymbol{\varepsilon}_h,
\qquad h=1,2,\cdots,g$$

* $h$번째 반응변수마다 하나의 선형회귀식이 있다.
* 각 반응변수는 동일한 설명변수 $x_1,\dots,x_p$를 사용할 수도 있고,  
뒤에서 보듯 일반화하면 서로 다른 설명변수 행렬을 가질 수도 있다.
* 회귀계수는 반응변수마다 달라질 수 있으므로 $\boldsymbol{\beta}_{0h},\boldsymbol{\beta}_{1h},\dots,\boldsymbol{\beta}_{ph}$처럼 첨자 $h$를 붙인다.

표본의 크기가 $n$이라면, $h$번째 반응변수에 대해 $n$개의 관측값을 모아 벡터로 만들 수 있다. 그러면 식을 행렬형으로 쓰면:

$$\mathbf{y}_h=X\boldsymbol{\beta}_h+\boldsymbol{\varepsilon}_h,\qquad h=1,2,\cdots,g \tag{10.14}$$

* $\mathbf{y}_h$: $n\times1$ 반응벡터
* $X$: $n\times (p+1)$ 계획행렬 (design matrix)
* $\boldsymbol{\beta}_h$: $(p+1)\times1$ 회귀계수 벡터
* $\boldsymbol{\varepsilon}_h$: $n\times1$ 오차벡터

특히 $X$는 보통 첫 열이 모두 1인 절편항(intercept term)을 포함하는 계획행렬이며, 계수의 수는 $p+1$개이다.

### 오차에 대한 가정

다변량 회귀에서 핵심은 **방정식들 사이의 오차 상관**을 어떻게 다루느냐이다. 먼저 각 방정식의 오차벡터에 대해 다음을 가정한다:

$$\boldsymbol{\varepsilon}_h \sim N(\mathbf{0_n},\sigma_h^2 I_n) \tag{10.15}$$

이는 $h$번째 방정식 내부에서는

* 오차의 평균이 0
* 분산이 $\sigma_h^2$
* 서로 다른 시점 또는 관측치 사이에서는 독립

또한 서로 다른 두 방정식 $h,q$의 오차벡터 사이에 대해 다음을 가정한다: 

$$E(\boldsymbol{\varepsilon}_h \boldsymbol{\varepsilon}_q^T)=\sigma_{hq}I_n \tag{10.16}$$

이 식은 매우 중요하다. 먼저 $h=q$이면

$$E(\boldsymbol{\varepsilon}_h\boldsymbol{\varepsilon}_h^T)=\text{Var}(\boldsymbol{\varepsilon}_h)=\sigma_h^2 I_n$$

가 되므로 (10.15)와 연결된다. 즉 $\sigma_{hh}=\sigma_h^2$라 해석하면 된다.  
오차벡터를 성분별로 쓰면

$$\boldsymbol{\varepsilon}_h^T=(\varepsilon_{h1},\varepsilon_{h2},\dots,\varepsilon_{hn}),\qquad
\boldsymbol{\varepsilon}_q^T=(\varepsilon_{q1},\varepsilon_{q2},\dots,\varepsilon_{qn})$$

이고, (10.16)은 성분 수준에서

$$E(\varepsilon_{hi}\varepsilon_{qj})=
\begin{cases}
0, & i\neq j\\
\sigma_{hq}, & i=j
\end{cases} \tag{10.17}$$

이 가정을 문장으로 풀면 다음과 같다.

* 같은 시점 $i$에서 관측된 서로 다른 반응변수들의 오차는 상관될 수 있다.
* 그러나 서로 다른 시점 $i\neq j$에서 관측된 값들 사이에는 상관이 없다고 본다.

즉, $g$개의 반응변수를 $n$번 관찰할 때, **같은 관찰시점 내부의 동시적 상관(contemporaneous correlation)** 은 허용하지만, **서로 다른 관찰시점 간의 상관은 허용하지 않는다**는 뜻이다.

여기서

$$E(\boldsymbol{\varepsilon}_{hi}\boldsymbol{\varepsilon}_{qi})=\sigma_{hq}$$

를 $h$번째 반응변수와 $q$번째 반응변수의 오차 사이의 **동시공분산 (contemporaneous covariance)** 이라고 부른다.

>이러한 구조는 특히 계량경제학 (econometrics)에서 자주 나타난다 (시계열 데이터로, 같은 시점에 여러 반응변수를 관측하는 경우. 설명변수를 흔히 내생변수(endogenous variable)이라 하고, 반응변수를 외생변수(exogenous variable)이라 한다)

### 전체 시스템을 하나의 큰 회귀식으로 묶기

이제 $g$개의 방정식을 하나로 묶는다. 각 방정식이 $\mathbf{y}_h=X\boldsymbol{\beta}_h+\boldsymbol{\varepsilon}_h,\qquad h=1,\dots,g$ 로 주어졌으므로, 이들을 세로로 쌓아 하나의 큰 벡터식으로 만들 수 있다. 

$$\mathbf{y}=
\begin{bmatrix}
\mathbf{y}_1\\
\mathbf{y}_2\\
\vdots\\
\mathbf{y}_g
\end{bmatrix}, 
\boldsymbol{\beta}=
\begin{bmatrix}
\boldsymbol{\beta}_1\\
\boldsymbol{\beta}_2\\
\vdots\\
\boldsymbol{\beta}_g
\end{bmatrix},
\boldsymbol{\varepsilon}=
\begin{bmatrix}
\boldsymbol{\varepsilon}_1\\
\boldsymbol{\varepsilon}_2\\
\vdots\\
\boldsymbol{\varepsilon}_g
\end{bmatrix},
Z= \begin{bmatrix}
X & O_{n\times (p+1)} & \cdots & O_{n\times (p+1)}\\
O_{n\times (p+1)} & X & \cdots & O_{n\times (p+1)}\\
\vdots & \vdots & \ddots & \vdots\\
O_{n\times (p+1)} & O_{n\times (p+1)} & \cdots & X
\end{bmatrix}$$

를 만들면 전체 시스템은 간단히

$$\mathbf{y}=Z\boldsymbol{\beta}+\boldsymbol{\varepsilon} \tag{10.18}$$

* $\mathbf{y}$: $gn\times1$
* $\boldsymbol{\beta}$: $g(p+1)\times1$
* $\boldsymbol{\varepsilon}$: $gn\times1$
* $Z$: $gn\times g(p+1)$

즉, 다변량 회귀는 수학적으로는 "아주 큰 하나의 선형회귀모형"으로 재구성된다. 다만 오차의 분산-공분산 구조가 특수하다는 점이 핵심이다.


### 전체 오차벡터의 분산-공분산행렬

이제 $\boldsymbol{\varepsilon}$의 분산-공분산행렬을 구한다. 블록 형태로 쓰면

$$Var(\boldsymbol{\varepsilon})=
\begin{bmatrix}
E(\boldsymbol{\varepsilon}_1\boldsymbol{\varepsilon}_1^T) & E(\boldsymbol{\varepsilon}_1\boldsymbol{\varepsilon}_2^T) & \cdots & E(\boldsymbol{\varepsilon}_1\boldsymbol{\varepsilon}_g^T)\\
E(\boldsymbol{\varepsilon}_2\boldsymbol{\varepsilon}_1^T) & E(\boldsymbol{\varepsilon}_2\boldsymbol{\varepsilon}_2^T) & \cdots & E(\boldsymbol{\varepsilon}_2\boldsymbol{\varepsilon}_g^T)\\
\vdots & \vdots & \ddots & \vdots\\
E(\boldsymbol{\varepsilon}_g\boldsymbol{\varepsilon}_1^T) & E(\boldsymbol{\varepsilon}_g\boldsymbol{\varepsilon}_2^T) & \cdots & E(\boldsymbol{\varepsilon}_g\boldsymbol{\varepsilon}_g^T)
\end{bmatrix}$$

가정 (10.16)에 의해 각 블록은 $\sigma_{hq}I_n$ 꼴이므로

$$Var(\boldsymbol{\varepsilon})=
\begin{bmatrix}
\sigma_1^2 I_n & \sigma_{12}I_n & \cdots & \sigma_{1g}I_n\\
\sigma_{21}I_n & \sigma_2^2 I_n & \cdots & \sigma_{2g}I_n\\
\vdots & \vdots & \ddots & \vdots\\
\sigma_{g1}I_n & \sigma_{g2}I_n & \cdots & \sigma_g^2 I_n
\end{bmatrix}$$

이를 더 간단히 쓰기 위해

$$\Sigma=
\begin{bmatrix}
\sigma_1^2 & \sigma_{12} & \cdots & \sigma_{1g}\\
\sigma_{21} & \sigma_2^2 & \cdots & \sigma_{2g}\\
\vdots & \vdots & \ddots & \vdots\\
\sigma_{g1} & \sigma_{g2} & \cdots & \sigma_g^2
\end{bmatrix}$$

라 두면

$$Var(\boldsymbol{\varepsilon})=\Sigma\otimes I_n=\Omega
\tag{10.19}$$

여기서 $\otimes$는 **크로네커 곱 (Kronecker product)** 이다.

>**크로네커 곱**  
>행렬 $A=(a_{ij})$와 행렬 $B$에 대해 $A\otimes B$는 $A$의 각 성분 $a_{ij}$를 행렬 $B$로 확대시켜 만든 큰 블록행렬이다. 즉
>
>$$A\otimes B = (a_{ij}B)$$

따라서 $\Sigma\otimes I_n$는 $\Sigma$의 각 원소 $\sigma_{hq}$를 $\sigma_{hq}I_n$라는 블록으로 바꾼 행렬이다. 위에서 유도한 $Var(\boldsymbol{\varepsilon})$와 정확히 같은 구조가 된다.

이 결과는 다변량 회귀의 오차구조를 압축적으로 표현한 것이다.

* $\Sigma$: 같은 시점에서 여러 반응변수 사이의 공분산 구조
* $I_n$: 서로 다른 시점 사이에는 상관이 없음을 나타냄

즉, 오차구조가 "반응변수 방향의 상관 × 관측시점 방향의 독립" 구조를 갖는다는 뜻이다.

### 왜 GLS를 써야 하는가

전체 시스템은 $\mathbf{y}=Z\boldsymbol{\beta}+\boldsymbol{\varepsilon}$ 인데 $Var(\boldsymbol{\varepsilon})=\Omega=\Sigma\otimes I_n$  

일반적으로 $\Omega\neq \sigma^2 I$이므로, 보통최소제곱 (ordinary least squares, OLS)의 표준 가정이 성립하지 않는다. 따라서 $\boldsymbol{\beta}$를 효율적으로 추정하려면 **일반화최소제곱 (generalized least squares, GLS)** 을 써야 한다.

GLS 추정량은 일반식에 따라

$$\hat{\boldsymbol{\beta}}^{*}=(Z^T\Omega^{-1}Z)^{-1}Z^T\Omega^{-1}\mathbf{y}
\tag{10.20}$$

이 추정량은 $\Omega$가 알려져 있을 때 $\boldsymbol{\beta}$의 **최량선형불편추정량 (best linear unbiased estimator, BLUE)** 이다.

### 각 방정식을 따로 OLS로 추정하면 무엇이 나오는가

한편 식 (10.14)의 각 방정식에 대해 각각 OLS를 적용할 수 있다. 즉

$$\hat{\boldsymbol{\beta}}_h=(X^TX)^{-1}X^T\mathbf{y}_h,\qquad h=1,2,\dots,g$$

이들을 하나로 쌓으면

$$\hat{\boldsymbol{\beta}}=
\begin{bmatrix}
\hat{\boldsymbol{\beta}}_1\\
\hat{\boldsymbol{\beta}}_2\\
\vdots\\
\hat{\boldsymbol{\beta}}_g
\end{bmatrix}
=
\begin{bmatrix}
(X^TX)^{-1}X^T\mathbf{y}_1\\
(X^TX)^{-1}X^T\mathbf{y}_2\\
\vdots\\
(X^TX)^{-1}X^T\mathbf{y}_g
\end{bmatrix}$$

이 식은 블록행렬로 다시 쓰면

$$\hat{\boldsymbol{\beta}}
=
\begin{bmatrix}
X^TX & O_{p+1} & \cdots & O_{p+1}\\
O_{p+1} & X^TX & \cdots & O_{p+1}\\
\vdots & \vdots & \ddots & \vdots\\
O_{p+1} & O_{p+1} & \cdots & X^TX
\end{bmatrix}^{-1}
\begin{bmatrix}
X & O_{n\times(p+1)} & \cdots & O_{n\times(p+1)}\\
O_{n\times(p+1)} & X & \cdots & O_{n\times(p+1)}\\
\vdots & \vdots & \ddots & \vdots\\
O_{n\times(p+1)} & O_{n\times(p+1)} & \cdots & X
\end{bmatrix}^T
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_g
\end{bmatrix}$$

이고, 이는 간단히

$$\hat{\boldsymbol{\beta}}=(Z^TZ)^{-1}Z^T\mathbf{y}
\tag{10.21}$$

즉, 각 방정식별 OLS 추정량을 쌓은 벡터는 전체 시스템에 대해 $\Omega$를 무시하고 OLS를 적용한 것과 동일하다.

### 흥미로운 결과: 특정 경우에는 OLS와 GLS가 같다

이제 매우 중요한 결과가 나온다. 식 (10.14)의 다변량 회귀에서 가정 (10.15), (10.16)이 성립하고, **모든 방정식의 계획행렬이 동일하게 $X$** 인 경우에는

$$\hat{\boldsymbol{\beta}}^{*}=\hat{\boldsymbol{\beta}}$$

가 성립한다. 즉, GLS를 쓰든 각 방정식을 따로 OLS로 추정하든 결과가 같다.

이것은 직관적으로는 다소 놀라운 결과이다. 오차들 사이에 동시공분산이 있으므로 보통은 GLS가 OLS보다 더 효율적일 것 같지만, **모든 방정식의 설명변수 행렬이 완전히 동일하면** 그 상관구조를 이용해도 계수 추정 자체는 달라지지 않는다.

다만 이 결과는 어디까지나 **동일한 $X$** 를 갖는 경우에만 성립한다. 방정식마다 설명변수 구조가 달라지면 더 이상 성립하지 않는다.


#### 증명: 크로네커 곱의 성질을 이용한 유도

이 결과를 크로네커 곱의 성질로 보인다. 필요한 성질은 다음과 같다.

$$I_m\otimes I_n = I_{mn}\\
(A\otimes B)(C\otimes D)=AC\otimes BD\\
(A\otimes B)^{-1}=A^{-1}\otimes B^{-1}\\
(A\otimes B)^T=A^T\otimes B^T \tag{10.22}$$

이제 $Z$와 $\Omega$를 크로네커 곱으로 표현하면 $Z=I_g\otimes X,\quad \Omega=\Sigma\otimes I_n$ 따라서

$$\Omega^{-1}=\Sigma^{-1}\otimes I_n$$

**$(Z^T\Omega^{-1}Z)^{-1}Z^T\Omega^{-1}\mathbf{y}$가 $(Z^TZ)^{-1}Z^T\mathbf{y}$와 같음을 보이는 것이 목표이다.**  

**$(Z^T\Omega^{-1}Z)^{-1}$ 계산**  
먼저 $Z^T\Omega^{-1} = (I_g\otimes X)^T(\Sigma^{-1}\otimes I_n)$에서,  
$(I_g\otimes X)^T = I_g^T\otimes X^T = I_g\otimes X^T$ 이므로

$$Z^T\Omega^{-1} = (I_g\otimes X^T)(\Sigma^{-1}\otimes I_n) = (I_g\Sigma^{-1})\otimes (X^T I_n) = \Sigma^{-1}\otimes X^T$$

이제 여기에 다시 $Z$를 곱하면

$$Z^T\Omega^{-1}Z =(\Sigma^{-1}\otimes X^T)(I_g\otimes X) =(\Sigma^{-1}I_g)\otimes (X^TX) =\Sigma^{-1}\otimes (X^TX)$$

따라서

$$(Z^T\Omega^{-1}Z)^{-1} =[\Sigma^{-1}\otimes (X^TX)]^{-1} =\Sigma\otimes (X^TX)^{-1}$$

한편 $Z^T\Omega^{-1}y =(\Sigma^{-1}\otimes X^T)y$ 이므로 GLS 추정량은

$$\hat{\boldsymbol{\beta}}^{*} =(Z^T\Omega^{-1}Z)^{-1}Z^T\Omega^{-1}\mathbf{y} =[\Sigma\otimes (X^TX)^{-1}][(\Sigma^{-1}\otimes X^T)\mathbf{y}] \\
=[(\Sigma\Sigma^{-1})\otimes((X^TX)^{-1}X^T)]\mathbf{y} =[I_g\otimes (X^TX)^{-1}X^T]\mathbf{y}$$

**$(Z^TZ)^{-1}Z^T\mathbf{y}$ 계산**  
$Z^T Z=(I_g\otimes X)^T(I_g\otimes X)=I_g\otimes X^TX$ 이므로 $(Z^TZ)^{-1}=I_g\otimes (X^TX)^{-1}$ 이고,  
또 $Z^T\mathbf{y}=(I_g\otimes X^T)\mathbf{y}$ 이어서

$$(Z^TZ)^{-1}Z^T\mathbf{y}
=[I_g\otimes (X^TX)^{-1}][I_g\otimes X^T]\mathbf{y}\\
=[I_g\otimes (X^TX)^{-1}X^T]\mathbf{y}$$

따라서 $\hat{\boldsymbol{\beta}}^{*}=(Z^TZ)^{-1}Z^T\mathbf{y}=\hat{\boldsymbol{\beta}}$

>**이 결과의 의미**
>
>이 결과는 실무적으로 매우 중요하다. 왜냐하면 $\Omega$를 추정하고 GLS를 계산하는 것은 계산량이 많고 번거롭기 때문이다. 그런데 만약 모든 방정식이 동일한 설명변수 행렬 $X$를 갖는다면, 굳이 복잡한 GLS를 할 필요 없이 **각 방정식을 OLS로 추정한 뒤 묶기만 해도** GLS와 동일한 결과를 얻는다.
>
>즉,
>* 방정식 간 오차의 상관이 있어도,
>* 모든 식의 설명변수 구조가 같다면,
>* 계수 추정량 자체는 OLS와 GLS가 일치한다.
>
>다만 이것은 **계수 추정량의 일치**에 관한 결과이지, 가설검정이나 분산추정에서 모든 문제가 자동으로 사라진다는 뜻은 아니다. 오차 상관구조를 반영한 분산-공분산행렬 계산은 여전히 중요할 수 있다.

### 더 일반적인 경우: 방정식마다 설명변수 행렬이 다른 경우

앞에서는 모든 방정식에서 동일한 $X$를 사용했다. 그러나 현실에서는 각 반응변수에 대해 설명변수 구성이 다를 수 있다. 그러면 모형은

$$\mathbf{y}_h=X_h\boldsymbol{\beta}_h+\boldsymbol{\varepsilon}_h,\qquad h=1,2,\dots,g \tag{10.23}$$

* $X_h$: $h$번째 방정식의 계획행렬
* $\boldsymbol{\beta}_h$: $h$번째 방정식의 회귀계수 벡터
* $\boldsymbol{\varepsilon}_h$: $h$번째 오차벡터

이때도 오차에 대한 가정 (10.15), (10.16)은 그대로 유지한다고 하자. 그러면 전체 시스템은 여전히

$$\mathbf{y}=Z\boldsymbol{\beta}+\boldsymbol{\varepsilon}$$

로 쓸 수 있지만, 이제 $Z$는

$$Z=
\begin{bmatrix}
X_1 & O & \cdots & O\\
O & X_2 & \cdots & O\\
\vdots & \vdots & \ddots & \vdots\\
O & O & \cdots & X_g
\end{bmatrix}$$

이 경우 GLS 추정량은

$$\hat{\boldsymbol{\beta}}^{*}=(Z^T\Omega^{-1}Z)^{-1}Z^T\Omega^{-1}\mathbf{y} \tag{10.24}$$

이번에는 일반적으로

$$\hat{\boldsymbol{\beta}}^{*}\neq \hat{\boldsymbol{\beta}}$$
즉, 각 방정식별 OLS를 따로 한 결과를 단순히 묶는 것으로는 더 이상 BLUE를 얻지 못한다. 이 경우에는 GLS를 사용해야 한다.

이것이 바로 **겉보기에는 서로 무관해 보이는 회귀식들이 오차 상관 때문에 사실상 정보를 공유한다**는 점이다. 방정식별 설명변수가 다르면, 다른 방정식의 정보가 해당 방정식의 추정 정밀도를 개선할 수 있다. 이것이 SUR(Seemingly Unrelated Regression)류 모형의 핵심 직관이다.

### $\Omega$가 알려져 있지 않을 때의 처리

현실에서는 $\Omega=\Sigma\otimes I_n$를 직접 알 수 없는 경우가 거의 대부분이다. 특히 $\Sigma$의 원소들 $\sigma_{hq}$를 모르는 경우가 많다. 이때는 먼저 $\Sigma$를 추정한 뒤, 그 추정값을 GLS 공식에 대입하는 방법을 사용한다. 이것이 **실행가능 일반화최소제곱 (feasible GLS, FGLS)** 이다.

먼저 각 방정식별로 OLS를 수행하여

$$\hat{\boldsymbol{\beta}}_h=(X_h^T X_h)^{-1}X_h^T\mathbf{y}_h$$

를 구한다.

그다음 잔차벡터를

$$\mathbf{e}_h=\mathbf{y}_h-X_h\hat{\boldsymbol{\beta}}_h,\qquad h=1,2,\dots,g$$

로 계산한다.

이제 $\Sigma$의 추정량을 $S=(S_{hq})$로 두고, 각 원소를

$$S_{hq}=\frac{\mathbf{e}_h^T \mathbf{e}_q}{n-p_h-1} \tag{10.25}$$

로 정의한다.

여기서 $p_h+1$은 $X_h$의 열의 개수이므로, $p_h$는 절편을 제외한 설명변수의 수로 볼 수 있다. 분모 $n-p_h-1$은 해당 방정식의 자유도(degree of freedom)이다.

이렇게 얻은 $S$를 이용하면 $\Omega$의 추정량을

$$S\otimes I$$

로 두고, 실행가능 GLS 추정량을

$$\tilde{\boldsymbol{\beta}}
=[Z^T(S\otimes I)^{-1}Z]^{-1}[Z^T(S\otimes I)^{-1}\mathbf{y}] \tag{10.26}$$

로 정의한다.

>**실행가능 GLS의 점근적 성질**
>
>Zellner의 결과에 따르면, $\Omega$가 미지일 때 식 (10.26)의 $\tilde{\boldsymbol{\beta}}$는 식 (10.24)의 이상적 GLS 추정량 $\hat{\boldsymbol{\beta}}^{*}$에 대해 좋은 점근적 성질을 갖는다.
>
>* $S$가 $\Omega$ 또는 $\Sigma$의 **일치추정량 (consistent estimator)** 역할을 하며,
>* $\sqrt{n}(\tilde{\boldsymbol{\beta}}-\boldsymbol{\beta})$와 $\sqrt{n}(\hat{\boldsymbol{\beta}}^{*}-\boldsymbol{\beta})$가 같은 **점근 분산-공분산행렬 (asymptotic variance-covariance matrix)** 을 가진다.
>
>이 뜻은 표본 수가 커질수록 $\tilde{\boldsymbol{\beta}}$가 이상적인 GLS 추정량과 거의 같은 효율성을 갖는다는 것이다. 즉, $\Omega$를 몰라도 데이터를 이용해 $\Omega$를 추정한 뒤 GLS를 수행하면 큰 표본에서는 매우 타당한 방법이 된다.

>**보충 설명: 왜 같은 (X)일 때는 GLS가 이득이 없는가**
>
>직관적으로 보면, 각 방정식이 같은 설명변수 정보를 이미 동일하게 사용하고 있으므로, 방정식 간 오차상관을 추가로 반영해도 계수 추정에 새롭게 활용할 정보가 생기지 않는다. 반면 $X_1,X_2,\dots,X_g$가 서로 다르면, 어떤 방정식이 가진 설명변수 구조가 다른 방정식의 오차상관과 결합되어 추가 정보를 제공한다. 그래서 그때 비로소 GLS가 OLS보다 효율적이 된다.


## 10.4 자기상관 (Autocorrelation)

자기상관(autocorrelation)은 회귀모형의 오차항(error term)들이 서로 독립이 아니라 **시차(lag)를 두고 상관관계**를 갖는 현상을 말한다.  
특히 시계열자료(time series data)에서는 인접한 시점의 오차가 서로 연관되는 경우가 매우 흔하다.  
이 절에서는 이러한 자기상관이 존재할 때 **보통최소제곱 (ordinary least squares, OLS)** 이 아니라 **일반화최소제곱 (generalized least squares, GLS)** 으로 회귀계수를 추정하는 방법을 설명한다.

### 10.4.1 일차자기상관모형의 GLS추정 (GLS Estimation for First-Order Autocorrelation Model)

먼저 단순회귀모형(simple regression model)을 생각한다. $y_i=\beta_0+\beta_1 x_i+\varepsilon_i,\quad i=1,2,\cdots,n $

여기서 오차 $\varepsilon_i$들이 서로 독립이라고 가정하지 않고, 다음과 같은 **일차 자기상관(first-order autocorrelation)** 구조를 갖는다고 가정한다.

$$\varepsilon_i=\rho \varepsilon_{i-1}+\delta_i,\qquad |\rho|<1
\tag{10.28}$$

또한 교란항(disturbance) $\delta_i$에 대해서는

$$\delta_i \sim N(0,\sigma_\delta^2),\qquad \text{Cov}(\delta_i,\delta_j)=0,\quad i\neq j$$

를 가정한다.

이 모형을 **일차자기상관회귀모형 (first-order autocorrelation regression model)** 이라고 부르며, $\rho$를 **자기상관계수 (autocorrelation coefficient)** 라고 한다.

일차자기상관에 관한 기본적인 논의는 앞 장에서 이미 했고, 여기서는 주로 $(\beta_0,\beta_1)$을 어떻게 추정할지를 다룬다.

### 오차벡터의 분산-공분산행렬

오차벡터를 $\boldsymbol{\varepsilon}^T=(\varepsilon_1,\varepsilon_2,\cdots,\varepsilon_n)$ 라 하자. 일차 자기상관 구조를 가지면 $\boldsymbol{\varepsilon}$의 분산-공분산행렬은 다음과 같은 꼴이 된다.

$$\text{Var}(\boldsymbol{\varepsilon})=\sigma_\varepsilon^2 V = \frac{\sigma_\delta^2}{1-\rho^2}
\begin{bmatrix}
1 & \rho & \rho^2 & \cdots & \rho^{n-1}\\
\rho & 1 & \rho & \cdots & \rho^{n-2}\\
\rho^2 & \rho & 1 & \cdots & \rho^{n-3}\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
\rho^{n-1} & \rho^{n-2} & \rho^{n-3} & \cdots & 1
\end{bmatrix}$$

즉, 같은 대각선에서는 1이고, 시차가 1이면 $\rho$, 시차가 2이면 $\rho^2$, 일반적으로 시차가 $k$이면 $\rho^k$가 된다. 이 구조는 AR(1)형 오차의 전형적인 공분산 구조이다.

$$\text{Var}(\varepsilon_i)=\frac{\sigma_\delta^2}{1-\rho^2}=\sigma_\varepsilon^2$$
따라서 $(\beta_0,\beta_1)$의 BLUE(best linear unbiased estimator)는 OLS가 아니라 GLS로 얻어야 한다.

$$ \hat{\boldsymbol{\beta}}^{*}=(X^TV^{-1}X)^{-1}X^TV^{-1}\mathbf{y} \tag{10.29}$$

실제로 자기상관계수 $\rho$를 모르는 경우가 많다. 이때는 $\rho$를 먼저 추정한 뒤, 그 추정값을 이용하여$V$를 추정한 뒤, 그 추정된 $V$로 GLS를 수행하는 방법을 사용한다. 하지만 이 방법은 식(10.29)는 BLUE가 아니게된다.  

자기상관계수를 추정하는 방법으로, 단순회귀모형의 $\beta$를 OLS로 먼저 추정하여 잔차(residual)를 구한 뒤, 그 잔차를 이용하여 $\rho$를 추정하는 방법이 있다. 

$$\hat{\rho} = \frac{\sum_{i=2}^n (e_{i-1} - \bar{e}_{-1})(e_i - \bar{e})}{\sum_{i=2}^n (e_{i-1} - \bar{e}_{-1})^2} \tag{10.30} \\
\bar{e}=\frac{1}{n-1}\sum_{i=2}^n e_i,\qquad \bar{e}_{-1}=\frac{1}{n-1}\sum_{i=2}^n e_{i-1}$$

($\bar{e}$와 $\bar{e}_{-1}$는 거의 0에 가까우므로 보통 0으로 놓고 계산한다)

$\hat \rho$를 $V$에 대입하여 $\hat V$를 구하면 

$$ \hat{V}=
\begin{bmatrix}
1 & \hat{\rho} & \hat{\rho}^2 & \cdots & \hat{\rho}^{n-1}\\
\hat{\rho} & 1 & \hat{\rho} & \cdots & \hat{\rho}^{n-2}\\
\hat{\rho}^2 & \hat{\rho} & 1 & \cdots & \hat{\rho}^{n-3}\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
\hat{\rho}^{n-1} & \hat{\rho}^{n-2} & \hat{\rho}^{n-3} & \cdots & 1
\end{bmatrix}, 
\hat{V}^{-1} = 
\begin{bmatrix}
1 & -\hat{\rho} & 0 & \cdots & 0\\
-\hat{\rho} & 1+\hat{\rho}^2 & -\hat{\rho} & \cdots & 0\\
0 & -\hat{\rho} & 1+\hat{\rho}^2 & \cdots & 0\\
\vdots & \vdots & \vdots & \ddots & -\hat{\rho}\\
0 & 0 & 0 & \cdots & 1
\end{bmatrix}$$

따라서 실행가능 GLS (feasible GLS, FGLS) 추정량은 
$$\tilde{\boldsymbol{\beta}}=[X^T\hat{V}^{-1}X]^{-1}X^T\hat{V}^{-1}\mathbf{y} \tag{10.31}$$

하지만, BLUE가 아니며, 표본크기 $n$이 커질수록 계산이 복잡해지므로 이 공식을 사용하지 않고 데이터 변환(transformation)을 이용하여 OLS로 추정하는 방법이 더 실용적이다. 다음 절에서 그 방법을 설명한다.
### 데이터 변환의 핵심 아이디어

자기상관이 있는 상태에서 바로 OLS를 적용하면 오차 독립 가정이 깨져 효율적인 추정량을 얻지 못한다.  
따라서 핵심은 **오차를 서로 비상관(uncorrelated)** 이 되도록 데이터 변환(transformation)을 하는 것이다.

식 (10.28) $\varepsilon_i=\rho\varepsilon_{i-1}+\delta_i$ 를 이용하면, 원래 회귀식 $y_i=\beta_0+\beta_1x_i+\varepsilon_i$ 에서 한 시점 전 식에 $\rho$를 곱한 것을 빼는 방식의 변환을 생각할 수 있다.

$$y_i' = y_i-\rho y_{i-1},\qquad x_i' = x_i-\rho x_{i-1} \tag{10.35}$$

$$y_i' = y_i-\rho y_{i-1} = (\beta_0+\beta_1 x_i+\varepsilon_i) - \rho(\beta_0+\beta_1 x_{i-1}+\varepsilon_{i-1})\\
= \beta_0(1-\rho) + \beta_1(x_i-\rho x_{i-1}) + (\varepsilon_i-\rho\varepsilon_{i-1}) \\
=\beta_0'+\beta_1 x_i' + \varepsilon_i' \tag{10.36}$$

여기서 $\beta_0'=\beta_0(1-\rho)$, $x_i'=x_i-\rho x_{i-1}$, $\varepsilon_i'=\varepsilon_i-\rho\varepsilon_{i-1}$ 이다.

즉 원래 자기상관이 있는 모형을, 변환된 자료 $(x_i',y_i')$에 대한 새로운 회귀모형으로 바꾸면 오차가 서로 독립이 되는 구조를 만든다.

### 변환된 오차의 분산, 공분산 구조
변환된 오차를 $\varepsilon_i'$라 두면 

$$ \text{Var}(\varepsilon_i') = \text{Var}(\varepsilon_i-\rho\varepsilon_{i-1}) = \text{Var}(\varepsilon_i) + \rho^2 \text{Var}(\varepsilon_{i-1}) - 2\rho \text{Cov}(\varepsilon_i,\varepsilon_{i-1})\\
= \sigma_\varepsilon^2 + \rho^2 \sigma_\varepsilon^2 - 2\rho (\rho \sigma_\varepsilon^2) = (1-\rho^2)\sigma_\varepsilon^2$$

$$\text{Cov}(\varepsilon_i',\varepsilon_{i-1}')
= E[(\varepsilon_i-\rho \varepsilon_{i-1})(\varepsilon_{i-1}-\rho \varepsilon_{i-2})]\\
= E[\varepsilon_i\varepsilon_{i-1}
-\rho \varepsilon_i\varepsilon_{i-2}
-\rho \varepsilon_{i-1}^2
+\rho^2 \varepsilon_{i-1}\varepsilon_{i-2}]$$

가 되고, 이를 자기상관 구조에 따라 정리하면 $\text{Cov}(\varepsilon_i',\varepsilon_{i-1}')=0$.  
같은 방식으로 일반적인 경우도 $\text{Cov}(\varepsilon_i',\varepsilon_j')=0,\quad i\neq j$ 임을 보일 수 있다.

따라서 오차벡터 $\boldsymbol{\varepsilon}'= (\varepsilon_1',\varepsilon_2',\cdots,\varepsilon_n')^\top$ 의 분산-공분산행렬은

$$\text{Var}(\boldsymbol{\varepsilon}')=(1-\rho^2)\sigma_\varepsilon^2 I_n$$

그런데

$$\sigma_\varepsilon^2=\frac{\sigma_\delta^2}{1-\rho^2}$$

이므로 실제로는

$$\text{Var}(\boldsymbol{\varepsilon}')=\sigma_\delta^2 I_n$$

와 같은 의미이다. 즉 변환된 모형에서는 오차가 서로 독립이고 동일분산을 가져 OLS를 적용할 수 있다. 따라서 변환된 회귀식에서 $(\beta_0',\beta_1)$의 BLUE는 OLS로 구할 수 있다. 

* 식 (10.36)의 $(\beta_0',\beta_1)$의 BLUE는 OLS 추정법으로 얻어진다.
* $\beta_1$을 먼저 구하고
* $\beta_0'=\beta_0(1-\rho)$ 관계를 이용하여 $\beta_0$를 복원한다.

즉 변환 후 OLS를 수행하는 것이 원래 자기상관 모형에서의 GLS와 동치가 된다.

### $\rho$가 알려진 경우와 알려지지 않은 경우

**$\rho$가 알려진 경우**  

$$y_i' = y_i-\rho y_{i-1},\qquad x_i' = x_i-\rho x_{i-1}$$

로 직접 변환한 뒤, 변환된 자료에 대해 OLS를 적용하면 된다. 이 경우는 이상적인 GLS 상황이다.

**$\rho$가 알려지지 않은 경우**  

그러나 실제로는 $\rho$를 모르므로 먼저 데이터를 이용하여 $\rho$를 추정해야 한다. 다음과 같은 추정량을 제시한다.

$$\hat{\rho} = \frac{\sum_{i=2}^n e_{i-1}e_i}{\sum_{i=2}^n e_{i-1}^2}
\tag{10.37}$$

여기서 $e_i$는 먼저 원자료에 OLS를 적용해서 얻은 잔차(residual)이다. 
이 식은 사실상 잔차 $e_i$를 이용하여 AR(1) 계수 $\rho$를 회귀식 형태로 추정한 것이다.

### $\hat{\rho}$를 이용한 실제 데이터 변환

$\rho$ 대신 $\hat{\rho}$를 사용하면 데이터 변환은

$$y_i' = y_i-\hat{\rho}y_{i-1},\quad x_i' = x_i-\hat{\rho}x_{i-1}
\tag{10.38}$$

변환된 자료에 대해 OLS를 적용하면

$$\hat{\beta}_1 = \frac{\sum_{i=1}^{n}(x_i'-\bar{x}')(y_i'-\bar{y}')}{\sum_{i=1}^{n}(x_i'-\bar{x}')^2}\\
\hat{\beta}_0 = \frac{1}{1-\hat{\rho}}\left[\bar{y}'-\hat{\beta}_1\bar{x}'\right]
\tag{10.39}$$

여기서

$$\bar{x}'=\frac{1}{n}\sum_{i=1}^{n}x_i',\qquad
\bar{y}'=\frac{1}{n}\sum_{i=1}^{n}y_i'$$

엄밀히 말하면 변환은 보통 $i=2,\dots,n$부터 유효하며, 첫 관측치는 별도 처리 또는 소거하는 경우가 많다. 표 10.2에서도 $i=1$에서 값 대신 $\times$ 표시가 있는 것으로 보아 첫 번째 자료는 변환식에 직접 들어가지 않음을 알 수 있다.

>#### 정리
>1. 원모형:
>    $$y_i=\beta_0+\beta_1x_i+\varepsilon_i$$
>
>2. 한 시점 전 식에 $\rho$를 곱함:
>    $$\rho y_{i-1}=\rho\beta_0+\rho\beta_1x_{i-1}+\rho\varepsilon_{i-1}$$
>
>3. 두 식을 뺌:
>    $$y_i-\rho y_{i-1}
>    = \beta_0(1-\rho)+\beta_1(x_i-\rho x_{i-1})+(\varepsilon_i-\rho\varepsilon_{i-1})$$
>
>4. 새 변수 정의:
>    $$y_i'=y_i-\rho y_{i-1},\quad
>    x_i'=x_i-\rho x_{i-1},\quad
>    \varepsilon_i'=\varepsilon_i-\rho\varepsilon_{i-1}$$
>
>그러면
>
>$$y_i'=\beta_0'+\beta_1x_i'+\varepsilon_i', \quad \beta_0'=\beta_0(1-\rho)$$
>
>5. $\varepsilon_i'=\delta_i$


#### 예제 10.1

어떤 큰 회사의 **연구개발비**와 **총판매액**에 대한 분기별 데이터를 이용하여 다음 단순회귀모형을 적합하려고 한다.

$$y_i=\beta_0+\beta_1x_i+\varepsilon_i,\qquad i=1,2,\cdots,20$$

* $y_i$: 총판매액(억원)
* $x_i$: 연구개발비(백만원)

먼저 **일차자기상관계수 $\rho=0$인지 검정**하라고 한다. 경제 자료에서는 $\rho$가 대체로 양(positive)인 경우가 많으므로 다음과 같은 가설을 둔다.

$$H_0:\rho=0, \quad H_1:\rho>0$$

**1단계: 원자료에 OLS 적용**

먼저 Durbin–Watson (d) 통계량을 계산하기 위하여 원자료에 OLS를 적용해서 $(\beta_0,\beta_1)$을 추정하고 잔차를 구한다.

OLS 추정치는 $\hat{\beta}_0=89.2339, \hat{\beta}_1=2.0242$ 따라서 잔차는 $e_i=y_i-\hat{y}_i = y_i-(89.2339+2.0242x_i)$ 로 계산된다.

표 10.1에는 각 분기별 $(y_i,x_i,e_i)$가 제시되어 있다. 잔차열을 보면 초반 6분기는 모두 음(negative), 그 다음 11분기는 모두 양(positive), 마지막에는 다시 음으로 바뀌어 있다. 이런 패턴은 잔차들 사이에 양의 자기상관이 있음을 직관적으로 시사한다.

**2단계: Durbin–Watson 검정**

Durbin–Watson (d) 통계량은

$$d= \frac{\sum_{i=2}^{20}(e_i-e_{i-1})^2}{\sum_{i=2}^{20}e_i^2} = \frac{946.71}{3030.69} \approx 0.312$$

유의수준 $\alpha=0.01$에서 $n=20, p=1$이므로 표값이 $d_L=0.95,\qquad d_U=1.15$ 여기서 $d=0.312<d_L=0.95$ 이므로 귀무가설 $H_0:\rho=0$은 기각된다.  따라서 $\rho>0$ 인 양의 일차자기상관이 존재한다고 판단한다.

**3단계: $\rho$의 추정**

다음으로 식 (10.37)을 이용하여 $\rho$를 추정한다.

$$\hat{\rho} = \frac{\sum_{i=2}^{20}e_{i-1}e_i}{\sum_{i=2}^{20}e_{i-1}^2} = \frac{2261.300}{2537.930} \approx 0.891$$

즉 자기상관계수의 추정치는 매우 큰 양수로 나타난다.

**4단계: 변환자료 구성**

이제 $\hat{\rho}=0.891$을 사용하여 데이터를 변환한다. $y_i' = y_i-0.891y_{i-1}, x_i' = x_i-0.891x_{i-1}$

표 10.2에는 이렇게 변환된 $(y_i',x_i')$ 값들이 정리되어 있다. 첫 번째 관측치는 이전 시점이 없으므로 $\times$로 표시되어 있다.

표의 일부를 보면 예를 들어

* $i=2$: $y_2'=41.917,\; x_2'=16.362$
* $i=3$: $y_3'=44.232,\; x_3'=16.959$
* $i=4$: $y_4'=46.353,\; x_4'=17.398$

와 같은 식으로 계산되어 있다.

**5단계: 변환자료에 대한 OLS 추정**

변환된 데이터에 대해 식 (10.39)의 OLS 추정량을 적용하면

$$\hat{\beta}_0 = \frac{\hat{\beta}_0'}{1-\hat{\rho}} = \frac{40.2457}{1-0.891} \approx 369.2266\\
\hat{\beta}_1=0.4862$$

즉 자기상관을 무시하고 원자료에 OLS를 적용했을 때의 추정치

$$\hat{\beta}_0=89.2339,\qquad \hat{\beta}_1=2.0242$$

와, 자기상관을 제거한 뒤의 GLS형 추정치

$$\hat{\beta}_0=369.2266,\qquad \hat{\beta}_1=0.4862$$

사이에 큰 차이가 나타난다.

교재의 결론도 바로 이것이다. 앞의 추정치는 자기상관을 무시한 단순 OLS 결과이고, 뒤의 추정치는 오차 간 자기상관을 제거한 뒤 OLS를 적용한 것이므로 보다 적절한 GLS 추정 결과이다.
