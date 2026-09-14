# Chapter 13 편의추정 (Biased Estimation)

지금까지 사용해 온 선형회귀모형(linear regression model)은 $\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\epsilon},\quad \boldsymbol{\epsilon} \sim N(\mathbf{0}_n,\sigma^2 I_n)$ 이다. 이 모형에서 회귀계수 $\boldsymbol{\beta}$는 최소제곱법(least squares method)을  여 $\hat{\boldsymbol{\beta}}=(X^TX)^{-1}X^T\mathbf{y}$ 로 추정한다. 소제곱추정량(least squares estimator)은 **최량 선형 불편 추정량(BLUE; Best Linear Unbiased Estimator)** 이다. 즉 선형 불편 추정량(linear unbiased estimator)들 가운데 분산이 가장 작은 추정량이다. 또한 최소제곱추정량의 분산은 $\text{Var}(\hat{\boldsymbol{\beta}})=\sigma^2 (X^TX)^{-1}$ 

그러나 설명변수들 사이에 강한 상관관계(strong correlation)가 존재하면 $X^TX$ (그람행렬) 의 정칙성(non-singularity)에 문제가 발생한다. 이 경우 $(X^TX)^{-1}$의 값이 매우 커지는 현상이 나타난다. 이러한 상황에서는 최소제곱추정량이 매우 불안정해진다.

이 장에서는 설명변수 사이에 강한 상관관계가 존재할 때 발생하는 문제를 분석하고, 최소제곱추정량 대신 사용할 수 있는 편의추정량(biased estimator)을 다룬다.

편의추정 방법을 적용하기 전에 변수의 **표준화(standardization)** 가 필요하다. 능형회귀(ridge regression)에서는 벌점함수(penalty function)

$$\sum_{j=1}^{p}\beta_j^2$$

를 사용한다. 이 값은 설명변수의 단위(scale)에 영향을 받지 않고 모든 변수에 균등하게 작용해야 한다. 또한 주성분회귀(principal component regression)와 부분최소제곱회귀(partial least squares regression)에서도 분산을 기준으로 차원축소 수행하므로 변수의 단위 차이를 제거하기 위해 표준화를 수행한다.

**표준화된 변수(standardized variables)를 사용하면 회귀모형에서 절편항(intercept)이 사라진다.**  
최종 결과를 정리할 떄는 표준화 이전 변수에 대한 회귀추정량으로 환원해야함! 

편의추정에 의해 구한 회귀추정량을

$$\hat{\beta}=(\hat{\beta}_1,\hat{\beta}_2,\dots,\hat{\beta}_p)^T$$

라 하면, 원래(최종) 변수 기준 회귀계수는 다음과 같이 환원된다.

$$\tilde{\beta}_0 = \bar{y}-\sum_{j=1}^{p}\hat{\beta}_j\frac{s_y}{s_j}\bar{x}_j \\
\tilde{\beta}_j=\frac{s_y}{s_j}\hat{\beta}_j,
\qquad j=1,2,\dots,p
\tag{13.1}
$$

* $\bar{y}$ : 반응변수의 표본평균
* $s_y^2$ : 반응변수의 표본분산
* $\bar{x}_j$ : $j$번째 설명변수의 표본평균
* $s_j^2$ : $j$번째 설명변수의 표본분산


## 13.1 다중공선성의 문제 (Multicollinearity)

다중공선성은 중회귀모형(multiple regression model)에서 두 개 이상의 설명변수 사이에 선형관계(linear relationship)가 존재하는 현상이다. 행렬 $X$의 어떤 열(column)이 다른 열 또는 여러 열의 선형결합(linear combination)으로 표현될 때 이러한 현상이 나타난다.

$X$가 $n\times p$ 행렬일 때 완전한 다중공선성(perfect multicollinearity)이 존재하면 $X$의 계수가 $p$보다 작아진다. 이 경우 $X^TX$의 역행렬이 존재하지 않는다. 따라서 $\hat{\boldsymbol{\beta}}=(X^TX)^{-1}X^T\mathbf{y}$를 사용할 수 없다.

완전한 다중공선성이 아니더라도 설명변수 사이에 매우 높은 상관관계가 존재하면 **근사적 다중공선성(approximate multicollinearity)** 이 발생한다. 이 경우 $(X^TX)^{-1}$는 존재하지만 계산이 매우 불안정해진다.

이를 이해하기 위해 설명변수가 두 개인 경우를 고려한다.

$$
y_i=\beta_1 x_{i1}+\beta_2 x_{i2}+\epsilon_i, \qquad
\epsilon_i\sim N(0,\sigma^2), \quad i=1,2,\dots,n \tag{13.2}
$$

모든 변수는 표준화된 변수라고 가정한다.

$$\sum y_i=\sum x_{i1}=\sum x_{i2}=0 \\ \sum y_i^2=\sum x_{i1}^2=\sum x_{i2}^2=1$$

이 가정 아래에서는 두 설명변수의 내적이 곧 두 변수의 상관계수가 되므로, 이후 다중공선성 계산이 매우 간단해진다. 이 경우 절편항 $\beta_0$는 필요하지 않다. 이때

$$
X^TX= \begin{pmatrix}
1 & r_{12}\\
r_{12} & 1
\end{pmatrix}, \quad  
X^Ty= \begin{pmatrix}
r_{1y}\\
r_{2y}
\end{pmatrix}
$$

* $r_{12}$ : $x_1$과 $x_2$ 사이의 표본상관계수(sample correlation coefficient)
* $r_{jy}$ : $x_j$와 $y$ 사이의 표본상관계수

따라서 최소제곱추정량은

$$
\hat{\boldsymbol{\beta}}
= \begin{pmatrix}
\hat{\beta}_1\\
\hat{\beta}_2
\end{pmatrix}
= (X^TX)^{-1}X^T\mathbf{y} 
= \begin{pmatrix}
\frac{1}{1-r_{12}^2} & \frac{-r_{12}}{1-r_{12}^2}\\
\frac{-r_{12}}{1-r_{12}^2} & \frac{1}{1-r_{12}^2}
\end{pmatrix}
\begin{pmatrix}
r_{1y}\\
r_{2y}
\end{pmatrix}
$$

분산-공분산 행렬(variance-covariance matrix)은 $\text{Var}(\hat{\boldsymbol{\beta}})=\sigma^2(X^TX)^{-1}$ 이므로

$$
\text{Var}(\hat{\beta}_j) = \frac{\sigma^2}{1-r_{12}^2}, \quad j=1,2
\tag{13.3}
$$

따라서 $x_1$과 $x_2$ 사이 상관관계가 커져 $|r_{12}| \rightarrow 1$ 이면 분모 $1-r_{12}^2$가 0에 가까워진다. 이 경우 회귀계수 추정량의 분산이 매우 커진다. 즉 $\hat{\beta}_1$과 $\hat{\beta}_2$는 $\beta_1$과 $\beta_2$의 추정량으로 신뢰하기 어려워진다.  

또한 $\sigma^2$의 불편추정량(unbiased estimator)으로 MSE를 사용하면 $\beta_j$의 $100(1-\alpha)\%$ 신뢰구간(confidence interval)은

$$
\hat{\beta}_j \pm t_{\alpha/2}(n-3) \sqrt{\frac{MSE}{1-r_{12}^2}} \tag{13.4}
$$

$1-r_{12}^2$가 작아지면 신뢰구간의 폭이 매우 커지므로 구간추정(interval estimation)의 의미도 약해진다.

일반적인 선형회귀모형 $y=X\beta+\epsilon, \quad \epsilon\sim N(0_n,\sigma^2 I_n)$ 에서도 동일한 문제가 발생한다. $\text{Var}(\hat{\boldsymbol{\beta}})=\sigma^2 (X^TX)^{-1}$ 이므로

$$ \sum_{j=1}^{p} \text{Var}(\hat{\beta}_j) = \sigma^2\,tr[(X^TX)^{-1}] \tag{13.5}$$

여기서 $X^TX$의 고유값을 $\lambda_j$라 하면, $\text{tr}(A^{-1}) = \sum \lambda_i^{-1}$, ($\lambda_i$는 $A$의 고윳값) 이므로

$$\sum_{j=1}^{p} \text{Var}(\hat{\beta}_j) = \sigma^2 \sum_{j=1}^{p}\lambda_j^{-1} \tag{13.6} $$

설명변수 사이에 강한 선형관계가 존재하면 $X^TX$는 **거의 특이행렬(near singular matrix)** 이 된다. 이 경우 가장 작은 고유값이 0에 가까워지고, 그 역수는 매우 커진다. 결과적으로 회귀계수 추정량의 분산이 크게 증가한다.

이러한 문제를 해결하기 위해 최소제곱추정량 대신 다음과 같은 편의추정량을 사용할 수 있다.

* 능형회귀추정량 (ridge regression estimator)
* 주성분회귀추정량 (principal component regression estimator)
* 부분최소제곱추정량 (partial least squares estimator)

이들을 살펴보기 전에, 다중공선성을 탐지하는 방법을 먼저 살펴본다. 행렬 $X^TX$에서 0이거나 0에 가까운 고윳값이 존재하면 다중공선성이 있다고 할 수 있다.

다른 방법으로는, $X$가 $n\times p$ 행렬이고 표준화되어 있다고 가정한다: $\sum_{i=1}^{n}x_{ij}=0, \quad \sum_{i=1}^{n}x_{ij}^2=1$ 일때, 최소제곱추정량은 $\hat{\boldsymbol{\beta}}=(X^TX)^{-1}X^T\mathbf{y}$ 이고 $E(\hat{\beta}_j)=\beta_j, \quad \text{Var}(\hat{\beta}_j)=c_{jj}\sigma^2$  
여기서 $C=(X^TX)^{-1}$ 이고 $c_{jj}$는 그 $j$번째 대각원소(diagonal element)이다.  
행렬을 $X=(x_j,X_j^*)$ 로 두 원소로 나누면

$$ c_{jj} = (x_j^Tx_j-x_j^TX_j^*(X_j^{*T}X_j^*)^{-1}X_j^{*T}x_j)^{-1}$$

이때 $x_j$를 반응변수로 하고 나머지 $p-1$개의 설명변수로 회귀했을 때의 결정계수(coefficient of determination)를 $R_j^2$라 하면 $c_{jj}=(1-R_j^2)^{-1}$ 이 된다. 따라서

$$\text{Var}(\hat{\beta}_j)=\frac{\sigma^2}{1-R_j^2}$$

* $R_j^2=0$: $x_j$는 다른 설명변수들과 직교(orthogonal)
* $R_j^2\rightarrow 1$: $x_j$는 다른 설명변수의 선형결합에 가깝다

즉 $R_j^2$가 1에 가까울수록 다중공선성이 강하다.

$$c_{jj}=(1-R_j^2)^{-1}$$

을 **분산팽창계수 (Variance Inflation Factor; VIF)** 라 한다. 일반적으로 $VIF > 10$ 이면 다중공선성이 존재한다고 판단한다.


## 13.2 능형회귀 (Ridge Regression)

지금까지 다루어 온 중회귀모형(multiple regression model)에서 행렬 $X$와 $y$가 표준화(standardized)되었다고 하고 $X$의 열의 수를 $p$라 하자. 그러면 $X^TX$와 $X^Ty$는 각각 상관계수행렬(matrices of correlation coefficients)이 된다.  
회귀계수 $\beta$의 최소제곱추정량(least squares estimator)을 $\hat{\boldsymbol{\beta}}=(X^TX)^{-1}X^T\mathbf{y}$라 할 때, $\hat{\boldsymbol{\beta}}$와 $\beta$ 사이의 거리의 제곱을

$$
L^2=(\hat{\boldsymbol{\beta}}-\beta)^T(\hat{\boldsymbol{\beta}}-\beta)
\tag{13.7}
$$

이라 두면, $\hat{\boldsymbol{\beta}}$의 평균제곱오차(mean squared error)는

$$
\begin{aligned}
MSE(\hat{\boldsymbol{\beta}})
&=E(L^2)
=E\left[(\hat{\boldsymbol{\beta}}-\beta)^T(\hat{\boldsymbol{\beta}}-\beta)\right] \\
&=\text{tr}\left(E\left[(\hat{\boldsymbol{\beta}}-\beta)(\hat{\boldsymbol{\beta}}-\beta)^T\right]\right) \\
&=E\left[\text{tr}((\hat{\boldsymbol{\beta}}-\beta)(\hat{\boldsymbol{\beta}}-\beta)^T)\right] \\
&=\sum_{j=1}^{p} \text{Var}(\hat{\boldsymbol{\beta}}_j) \\
&=\sigma^2\sum_{j=1}^{p}\lambda_j^{-1}
\end{aligned}
\tag{13.8}
$$



>$X^TX$는 대칭행렬이므로 고유분해할 수 있다. $X^TX=Q\Lambda Q^T,$ 여기서 $Q^TQ=QQ^T=I, \quad \Lambda= \text{diag}(\lambda_1,\ldots,\lambda_p).$  
>$X$가 완전열계수이면 모든 고윳값이 양수이므로 $(X^TX)^{-1} = Q\Lambda^{-1}Q^T$ 이고, $\Lambda^{-1} = \text{diag} \left(\frac1{\lambda_1},\ldots,\frac1{\lambda_p}\right)$ 이다. 즉, 역행렬의 고윳값은 원래 행렬 고윳값의 역수다.
>
>따라서
>
>$$
>MSE(\hat{\boldsymbol\beta}) = \sigma^2\text{tr}\{(X^TX)^{-1}\}
>= \sigma^2\text{tr}(Q\Lambda^{-1}Q^T) \\
>= \sigma^2\text{tr}(\Lambda^{-1}Q^TQ)
>= \sigma^2\text{tr}(\Lambda^{-1}) = \boxed{\sigma^2\sum_{j=1}^p\frac1{\lambda_j}}.
>$$


가 된다. 여기서 $\lambda_j$는 행렬 $X^TX$의 고유값이다. 따라서 설명변수 사이에 완전에 가까운 다중공선성이 존재하면 $\lambda_j$들 가운데 0에 가까운 값이 생기고, $\hat{\boldsymbol{\beta}}$는 $\beta$로부터 멀어져 평균제곱오차 기준에서 좋은 추정량이 되기 어렵다.

이 단점을 보완하기 위하여 다음과 같은 추정량을 사용한다. (Hoerl, Kennard 가 제안)

$$ \hat{\boldsymbol{\beta}}(k)=(X^TX+kI_p)^{-1}X^T\mathbf{y},\qquad k>0 \tag{13.9}$$

이를 **능형회귀추정량(ridge regression estimator)** 이라 한다. 여기서 $k$는 양의 상수(positive constant)이며 보통 $0<k<1$ 범위 안에 둔다. 모든 설명변수가 서로 직교(orthogonal)하여 $X^TX=I_p$가 되는 경우에는

$$\hat{\boldsymbol{\beta}}(k)=\frac{1}{1+k}\hat{\boldsymbol{\beta}}$$

가 된다. 따라서 능형회귀추정량은 최소제곱추정량의 크기를 축소한 **축소추정량(shrinkage estimator)** 이다.  
또한 기대값(expectation)은

$$
\begin{aligned}
E[\hat{\boldsymbol{\beta}}(k)]
&=(X^TX+kI_p)^{-1}X^TE(y) \\
&=(X^TX+kI_p)^{-1}X^TX\beta \neq \beta
\end{aligned}
$$

이므로, $\hat{\boldsymbol{\beta}}(k)$는 편의추정량(biased estimator)이다. 분산은

$$Var[\hat{\boldsymbol{\beta}}(k)] = \sigma^2(X^TX+kI_p)^{-1}X^TX(X^TX+kI_p)^{-1} \tag{13.10}$$

잔차제곱합(sum of squared errors, SSE)은

$y−X \{\hat\beta + [\hat\beta ( k)−\hat\beta ]\}=(y−X \hat\beta ) −X[\hat\beta ( k)−\hat\beta ]$ 이므로 $e=y-X\hat{\boldsymbol{\beta}}, \quad d=\hat{\boldsymbol{\beta}}(k)-\hat{\boldsymbol{\beta}}.$ 라 하면, 

$$
\begin{aligned}
[y-X\hat{\boldsymbol{\beta}}(k)]^T[y-X\hat{\boldsymbol{\beta}}(k)]
&=(e^T-d^TX^T)(e-Xd) \\
&=e^Te-e^TXd-d^TX^Te+d^TX^TXd \\
&=(y-X\hat{\boldsymbol{\beta}})^T(y-X\hat{\boldsymbol{\beta}})
+[\hat{\boldsymbol{\beta}}(k)-\hat{\boldsymbol{\beta}}]^T X^TX [\hat{\boldsymbol{\beta}}(k)-\hat{\boldsymbol{\beta}}]
\end{aligned}
\tag{13.11}
$$

이므로 ($X^Te=0$), 능형회귀추정량 $\hat{\boldsymbol{\beta}}(k)$로 적합한 모형의 잔차제곱합은 최소제곱추정량 $\hat{\boldsymbol{\beta}}$로 적합한 모형의 잔차제곱합보다 크다.

#### 평균제곱오차의 분해

어떤 $\boldsymbol{\alpha}=(\alpha_1,\alpha_2,\cdots,\alpha_p)^T$가 존재한다. 
> 주석: $X^TX$의 고윳값분해로 얻은 직교행렬 $P$에 대해 $\boldsymbol{\alpha}=P^T\boldsymbol{\beta}$로 둘 수 있다.

이때 능형회귀추정량의 평균제곱오차는 아래를 만족시킨다.

$$
\begin{aligned}
MSE[\hat{\boldsymbol{\beta}}(k)]
&=E[L^2(k)]
=E\left[(\hat{\boldsymbol{\beta}}(k)-\boldsymbol{\beta})^T(\hat{\boldsymbol{\beta}}(k)-\boldsymbol{\beta})\right] \\
&=\sigma^2\text{tr}\left[(X^TX+kI_p)^{-1}X^TX(X^TX+kI_p)^{-1}\right]
+k^2\boldsymbol{\beta}^T(X^TX+kI_p)^{-2}\boldsymbol{\beta} \\
&=\sigma^2\sum_{j=1}^{p}\lambda_j(\lambda_j+k)^{-2}
+\sum_{j=1}^{p}\frac{k^2\alpha_j^2}{(\lambda_j+k)^2}
\end{aligned}
\tag{13.12}
$$

가 된다. 여기서 첫째 항은 분산에 해당하고, 둘째 항은 편의(bias)의 제곱에 해당한다. 즉 능형회귀는 분산을 줄이는 대신 편의를 도입한다. 식 (13.8)과 비교하면 

$$ E[L^2(k)]<E(L^2) \tag{13.13}$$

를 만족하는 $k>0$가 항상 존재함을 Hoerl와 Kennard가 증명했다. 따라서 평균제곱오차(mean squared error) 관점에서는 능형회귀추정량이 최소제곱추정량보다 우수할 수 있다.  
일반적으로 $k$가 작을 때에는 분산이 급격히 감소하고 편의의 제곱은 서서히 증가하므로, $E[L^2(k)]$가 $E(L^2)$보다 작아지는 $k>0$의 값이 존재한다. 그러나 $k$가 지나치게 커지면 편의의 제곱이 급격히 증가하여 오히려 $E[L^2(k)]>E(L^2)$가 된다.

이 절의 핵심은 다음과 같다.

* 최소제곱추정량은 불편(unbiased)하지만 분산이 매우 클 수 있다.
* 능형회귀추정량은 편의(bias)를 허용하는 대신 분산을 줄인다.
* 적절한 $k$를 선택하면 전체 평균제곱오차는 더 작아질 수 있다.

>**증명**
>
>MSE 함수를
>
>$$
>R(k)=E[L^2(k)] =
>\sum_{j=1}^p \frac{\sigma^2\lambda_j+k^2\alpha_j^2}
>{(\lambda_j+k)^2}
>$$
>
>라고 하자. $k=0$이면
>
>$$
>R(0) = \sigma^2\sum_{j=1}^p\frac{1}{\lambda_j} = E(L^2)
>$$
>
>이다. 각 항을 미분하면
>
>$$
>\frac{d}{dk}
>\left[
>\frac{\sigma^2\lambda_j+k^2\alpha_j^2} {(\lambda_j+k)^2}
>\right] = \frac{2\lambda_j(k\alpha_j^2-\sigma^2)} {(\lambda_j+k)^3}.
>$$
>
>따라서
>
>$$
>R'(k)
>= 2\sum_{j=1}^p \frac{\lambda_j(k\alpha_j^2-\sigma^2)} {(\lambda_j+k)^3}.
>$$
>
>특히 $k=0$에서
>
>$$
>\begin{aligned}
>R'(0)
>&= 2\sum_{j=1}^p \frac{\lambda_j(0-\sigma^2)} {\lambda_j^3}\\
>&= -2\sigma^2\sum_{j=1}^p\frac{1}{\lambda_j^2}\\
>&<0.
>\end{aligned}
>$$
>
>$R'(0)<0$이므로 $k=0$의 오른쪽에서 $R(k)$는 감소한다. 따라서 충분히 작은 어떤 $k>0$에 대해 $R(k)<R(0)$ 가 성립한다. 즉,
>
>$$
>\boxed{E[L^2(k)]<E(L^2)}
>$$
>
>인 $k>0$가 반드시 존재한다.


### 13.2.1 능형회귀추정량의 성질

능형회귀추정량의 성질과 k값을 선택하는 방법을 알아보자. 행렬 $X^TX$의 고유값들을 대각원소로 갖는 행렬을 $\Lambda$, 그에 대응하는 고유벡터들을 열로 갖는 행렬을 $P$라 하자. 즉

$$
\Lambda= \begin{pmatrix}
\lambda_1 & 0 & \cdots & 0\\
0 & \lambda_2 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & \lambda_p
\end{pmatrix},
\qquad
P=[\mathbf p_1, \mathbf p_2,\cdots, \mathbf p_p]
$$

이고,

$$(X^TX-\lambda_j I_p)\mathbf p_j=0_p,\qquad j=1,2,\cdots,p$$

또한 $P^TP=PP^T=I_p$ 이며, $X^TX=P\Lambda P^T$ 를 만족한다.  
이제 선형모형 $y=X\beta+\epsilon$ 에 대하여 $Z=XP,\quad \alpha=P^T\beta$ 라는 변환을 적용하면,

$$
\begin{aligned}
y
&=X\beta+\epsilon \\
&=XPP^T\beta+\epsilon \\
&=Z\alpha+\epsilon,\qquad \epsilon\sim N(0_n,\sigma^2I_n)
\end{aligned}
\tag{13.14}
$$

이때 $Z^TZ=P^TX^TXP=\Lambda$ 이므로, $\alpha$의 최소제곱추정량은

$$\hat{\alpha}=(Z^TZ)^{-1}Z^Ty=\Lambda^{-1}Z^Ty$$

이다. 또한 적합된 모형 $y=Z\hat{\alpha}$의 잔차제곱합은

$$
\begin{aligned}
SSE(\hat{\alpha})
&=(y-Z\hat{\alpha})^T(y-Z\hat{\alpha}) \\
&=y^Ty-2y^TZ\hat{\alpha}+\hat{\alpha}^T\Lambda\hat{\alpha}
\end{aligned}
\tag{13.15}
$$

이고, $\hat{\alpha}$의 분산은

$$
\begin{aligned}
\text{Var}(\hat{\alpha})
&=\text{Var}(\Lambda^{-1}Z^Ty) \\
&=\Lambda^{-1}Z^T\text{Var}(y)Z\Lambda^{-1} \\
&=\sigma^2\Lambda^{-1}
\end{aligned}
$$

앞에서 정의한 $\alpha$의 능형회귀추정량은 $\hat{\alpha}(k)=(Z^TZ+kI)^{-1}Z^Ty$ 이며,

$$
\begin{aligned}
\hat{\alpha}(k)
&=(\Lambda+kI)^{-1}Z^Ty \\
&=(\Lambda+kI)^{-1}\Lambda\hat{\alpha}
\end{aligned}
$$

가 된다. 따라서 각 성분(component)에 대해

$$\hat{\alpha}_j(k)=\frac{\lambda_j}{\lambda_j+k}\hat{\alpha}_j,\qquad j=1,2,\cdots,p$$

이다. 이 식은 매우 중요하다. $\lambda_j$가 $k$에 비해 충분히 크면 $\hat{\alpha}_j(k)$는 $\hat{\alpha}_j$와 거의 차이가 없다. 반면 $\lambda_j$가 작으면 $\hat{\alpha}_j(k)$는 크게 축소된다. 즉, 능형회귀는 작은 고유값에 대응하는 불안정한 방향의 추정을 강하게 줄인다.  
한편 $\hat{\alpha}(k)$의 분산-공분산행렬(variance-covariance matrix)은

$$
\begin{aligned}
Var[\hat{\alpha}(k)]
&=Var[(\Lambda+kI)^{-1}Z^Ty] \\
&=(\Lambda+kI)^{-1}Z^T\text{Var}(y)Z(\Lambda+kI)^{-1} \\
&=\sigma^2\Lambda(\Lambda+kI)^{-2}
\end{aligned}
\tag{13.16}
$$

이고, 이는 대각행렬이다.  
또한 편의의 크기는

$$E[\hat{\alpha}(k)-\alpha] = [(\Lambda+kI_p)^{-1}\Lambda-I_p]\alpha \tag{13.17}$$

따라서 $\hat{\alpha}(k)$의 평균제곱오차는

$$
\begin{aligned}
MSE(\hat{\alpha}(k))
&=E[L^2(k)]
=E\left[(\hat{\alpha}(k)-\alpha)^T(\hat{\alpha}(k)-\alpha)\right] \\
&=\text{tr}(Var[\hat{\alpha}(k)])+[\hat{\alpha}(k)\text{의 편의}]^2 \\
&=\sigma^2\text{tr}[\Lambda(\Lambda+kI)^{-2}]
+\alpha^T[(\Lambda+kI)^{-1}\Lambda-I_p]^T[(\Lambda+kI)^{-1}\Lambda-I_p]\alpha
\end{aligned} 
$$

$$
=\sigma^2\sum_{j=1}^{p}\frac{\lambda_j}{(\lambda_j+k)^2}
+\sum_{j=1}^{p}\frac{k^2\alpha_j^2}{(\lambda_j+k)^2}
\tag{13.18}
$$

이다. 이 식은 모형 $y=Z\alpha+\epsilon$에 대한 능형회귀추정량의 평균제곱오차이며, 앞의 식 (13.12)와 동일한 내용을 고유값 분해(eigendecomposition)를 이용하여 다시 표현한 것이다.

#### $k$가 0 근처에서 평균제곱오차를 줄이는 이유

식 (13.18)을 $k$로 미분하면

$$
\frac{dE[L^2(k)]}{dk}
= -2\sigma^2\sum_{j=1}^{p}\frac{\lambda_j}{(\lambda_j+k)^3}
+ 2k\sum_{j=1}^{p}\frac{\lambda_j\alpha_j^2}{(\lambda_j+k)^3}
$$

이다. 또한 $k=0$일 때 $E[L^2(0)]=E[L^2]$이고, $L^2(k)$는 $k$의 연속함수다. 따라서

$$
\lim_{k\to 0+}\frac{dE[L^2(k)]}{dk}
= -2\sigma^2\sum_{j=1}^{p}\frac{1}{\lambda_j^2}<0
$$

가 성립한다. 즉 $k$를 0에서 아주 조금 증가시키면 평균제곱오차가 감소한다. 그러므로 $E[L^2(k)]<E(L^2)$가 되는 $k$가 존재한다.

식(13.18)을 최소로하는 $k$의 값을 간단한 공식으로 구할수는 없지만, $X^TX=I_p$인 특수한 경우 최소해를 구할 수 있다: 만약 $X^TX=I_p$이면 평균제곱오차를 최소로 하는 $k$는

$$
k=\frac{p\sigma^2}{\boldsymbol{\alpha}^T\boldsymbol{\alpha}}
\tag{13.19}
$$

가 된다. (참고문헌 13.7)

#### 서로 다른 능형모수의 허용

능형회귀 추정량의 식을 더 일반화하여 하나의 $k$ 대신 대각행렬

$$
K= \begin{pmatrix}
k_1 & 0 & \cdots & 0\\
0 & k_2 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & k_p
\end{pmatrix}
\tag{13.20}
$$

를 도입하여 서로다른 $p$개의 $k$값을 허용할수도 있다. 이 경우

$$
\hat{\boldsymbol{\alpha}}(K)=(\Lambda+K)^{-1}\Lambda\hat{\boldsymbol{\alpha}}
\tag{13.21}
$$

로 정의한다. 그리고 평균제곱오차는

$$
MSE(\hat{\boldsymbol{\alpha}}(K))
= E[L^2(K)]
= \sum_{j=1}^{p}\frac{\sigma^2\lambda_j+k_j^2\alpha_j^2}{(\lambda_j+k_j)^2}
$$

가 된다. 각 $j$에 대하여 $k_j=\frac{\sigma^2}{\alpha_j^2},\quad j=1,2,\cdots,p$ 일 때 $MSE[\hat{\alpha}(K)]$가 최소가 됨을 쉽게 보일 수 있다.

>**증명**
>
>평균제곱오차가 각 $k_j$에 관한 항의 합으로 분리되므로, 각 항을 개별적으로 최소화하면 전체 MSE가 최소가 된다.
>
>$MSE\bigl(\hat{\boldsymbol{\alpha}}(K)\bigr)$의 $j$번째 항을 미분하면,
>
>$$
>\frac{2k_j\alpha_j^2(\lambda_j+k_j) - 2(\sigma^2\lambda_j+k_j^2\alpha_j^2)}{(\lambda_j+k_j)^3} 
>= \frac{2\lambda_j(k_j\alpha_j^2-\sigma^2)}{(\lambda_j+k_j)^3}
>$$
>
>$\lambda_j>0$이고 $\lambda_j+k_j>0$이므로 $R_j'(k_j)=0$ 이 되기 위한 조건은 $k_j\alpha_j^2-\sigma^2=0$ 이다. 따라서 $\alpha_j\neq0$이면
>
>$$
>\boxed{k_j^*=\frac{\sigma^2}{\alpha_j^2}}
>$$

### 13.2.2 $k$값의 선택

능형회귀를 실제로 사용하려면 조절모수(tuning parameter) $k$를 정해야 한다. 이를 위하여 몇 가지 선택 방법을 사용한다.

[첫째 방법]

$\hat{\boldsymbol{\beta}}(k)=(X^TX+kI_p)^{-1}X^T\mathbf{y}$에서 $k$를 0에서 1까지 변화시키며 각 성분

$$
\hat{\boldsymbol{\beta}}(k)=\bigl(\hat{\beta}_1(k),\hat{\beta}_2(k),\cdots,\hat{\beta}_p(k)\bigr)^T
$$

의 변화를 관찰하는 것이다. 이때 $\hat{\beta}_j(k)$들이 급격히 변하지 않고 안정화(stabilized)되는 $k$를 선택한다. 이러한 $\hat{\beta}_j(k)$들의 궤적(trajectory)을 **능형트레이스(ridge trace)** 라 한다.  
이 방법으로 선택된 $k$는 $X^TX+kI_p$를 사용하여 다중공선성을 상당히 완화할 수 있다. 다만 계산량이 많고, 선택이 주관적일 수 있다는 단점이 있다.

[둘째 방법]

식 (13.19)를 이용하는 것이다. 즉, 

$$
k=\frac{p\hat{\sigma}^2}{\hat{\boldsymbol{\alpha}}^T\hat{\boldsymbol{\alpha}}}
\tag{13.22}
$$

로 선택한다. 여기서 $\sigma^2$ 대신 불편추정량(unbiased estimator) $\hat{\sigma}^2$를 사용하고 ($\mathbf{y}=X\boldsymbol{\beta}+\boldsymbol{\epsilon}$에서의 MSE로 추정), $\boldsymbol{\alpha}$ 대신 $\hat{\boldsymbol{\alpha}}=P^T\hat{\boldsymbol{\beta}}=P^T(X^TX)^{-1}X^T\mathbf{y}$ 를 대입한다. 이 방법은 최소제곱추정량을 이용하여 한 번에 $k$를 정할 수도 있고, 다음과 같은 반복적(iterative) 방법으로 선택할 수도 있다.

먼저 식 (13.22)로 얻은 값을 $k_1$이라 하고, $\hat{\boldsymbol{\beta}}(k_1)=(X^TX+k_1I_p)^{-1}X^T\mathbf{y}$ 를 구한 뒤, $\hat{\boldsymbol{\alpha}}(k_1)=P^T\hat{\boldsymbol{\beta}}(k_1)$ 를 계산한다. 다음으로 이것을 이용하여

$$k_2=\frac{p\hat{\sigma}^2}{\hat{\boldsymbol{\alpha}}(k_1)^T\hat{\boldsymbol{\alpha}}(k_1)}$$

를 구하고, $\hat{\boldsymbol{\beta}}(k_2)=(X^TX+k_2I_p)^{-1}X^T\mathbf{y},\quad \hat{\boldsymbol{\alpha}}(k_2)=P^T\hat{\boldsymbol{\beta}}(k_2)$ 를 계산한다. 다시

$$k_3=\frac{p\hat{\sigma}^2}{\hat{\boldsymbol{\alpha}}(k_2)^T\hat{\boldsymbol{\alpha}}(k_2)}$$

를 구하는 식으로 반복한다. 이렇게 $k_1,k_2,\cdots,k_q$를 차례로 구할 때, $k_{q-1}$와 $k_q$의 차이가 유의하지 않으면 $k_{q-1}$를 최종 $k$로 선택한다.

[셋째 방법]

**예측오차제곱합(PRESS; predicted residual error sum of squares)** 을 이용하는 것이다. 주어진 $k$에 대한 능형회귀모형의 예측오차제곱합은

$$
PRESS(k)
= \sum_{i=1}^{n}(y_i-\hat{y}^{(i)})^2
= \sum_{i=1}^{n}(y_i-\mathbf{x}_i^T\hat{\boldsymbol{\beta}}^{(i)}(k))^2
\tag{13.23}
$$

로 정의한다. 여기서 $\hat{\boldsymbol{\beta}}^{(i)}(k)$는 $i$번째 관측값과 그 설명변수벡터 $(y_i,\mathbf{x}_i)$를 제외하고 계산한 능형회귀추정량으로서, 행렬 $A=X(X^TX+kI_p)^{-1}X^T$의 $i$번째 대각원소(diagonal element)를 $a_{ii}$라 하면,

$$\hat{\boldsymbol{\beta}}^{(i)}(k)
= \hat{\boldsymbol{\beta}}(k)-\frac{(X^TX+kI_p)^{-1}\mathbf{x}_ie_i}{1-a_{ii}}
\tag{13.24}$$

로 계산된다. 여기서 $e_i=y_i-\mathbf{x}_i^T\hat{\boldsymbol{\beta}}(k)$ 는 전체 자료(full data)를 이용한 능형회귀모형의 잔차다.  
이 결과를 식 (13.23)에 대입하여 정리하면

$a_{ii} = \mathbf{x}_i^T(X^TX+kI_p)^{-1}\mathbf{x}_i$ 이므로 

$$
\mathbf{x}_i^T\hat{\boldsymbol{\beta}}^{(i)}(k) = \mathbf{x}_i^T\hat{\boldsymbol{\beta}}(k) - \frac{\mathbf{x}_i^T(X^TX+kI_p)^{-1}\mathbf{x}_ie_i}{1-a_{ii}} = \mathbf{x}_i^T\hat{\boldsymbol{\beta}}(k) -\frac{a_{ii}e_i}{1-a_{ii}}.
$$

따라서 

$$
\begin{aligned}
PRESS(k) &= \sum_{i=1}^{n}(y_i-\mathbf{x}_i^T\hat{\boldsymbol{\beta}}^{(i)}(k))^2 \\
&= \sum_{i=1}^{n} (y_i-\mathbf{x}_i^T\hat{\boldsymbol{\beta}}(k) + \frac{a_{ii}e_i}{1-a_{ii}})^2\\
&= \sum_{i=1}^{n}(e_i+\frac{a_{ii}e_i}{1-a_{ii}})^2 \\

&=\sum_{i=1}^{n}(\frac{e_i}{1-a_{ii}})^2 
\end{aligned}
\tag{13.25}$$


가 된다. 여기서 조절모수 $k$는 $PRESS(k)$를 최소로 하는 값으로 선택할 수 있다.  

>**식 13.24 증명**
>
>먼저 표기를 다음과 같이 두자. $M_k=X^TX+kI_p, \quad \hat{\boldsymbol{\beta}}(k)=M_k^{-1}X^Ty.$
>
>$i$번째 관측값을 제거한 설계행렬과 반응벡터를 각각 $X_{(i)}$, $y_{(i)}$라고 하면 $X_{(i)}^TX_{(i)}=X^TX-\mathbf{x}_i\mathbf{x}_i^T$ 이고 $X_{(i)}^Ty_{(i)}=X^Ty-\mathbf{x}_iy_i$ 이다.  
>따라서 $i$번째 관측값을 제거한 능형회귀추정량은 $\hat{\boldsymbol{\beta}}^{(i)}(k) = \left(X^TX-\mathbf{x}_i\mathbf{x}_i^T+kI_p\right)^{-1} \left(X^Ty-\mathbf{x}_iy_i\right)$ 이다.
>
>1. 두 정규방정식을 비교하는 증명
>
>전체 자료를 이용한 능형회귀 정규방정식은 $M_k\hat{\boldsymbol{\beta}}(k)=X^Ty$ 이다. 반면 $i$번째 관측값을 제외한 정규방정식은 $\left(M_k-\mathbf{x}_i\mathbf{x}_i^T\right) \hat{\boldsymbol{\beta}}^{(i)}(k) = X^Ty-\mathbf{x}_iy_i$ 이므로 $M_k\hat{\boldsymbol{\beta}}^{(i)}(k) - \mathbf{x}_i\mathbf{x}_i^T\hat{\boldsymbol{\beta}}^{(i)}(k) = X^Ty-\mathbf{x}_iy_i.$ 여기서 $X^Ty=M_k\hat{\boldsymbol{\beta}}(k)$를 대입하면
>
>$$
>M_k\hat{\boldsymbol{\beta}}^{(i)}(k) - \mathbf{x}_i\mathbf{x}_i^T\hat{\boldsymbol{\beta}}^{(i)}(k)
>= M_k\hat{\boldsymbol{\beta}}(k)-\mathbf{x}_iy_i.
>$$
>
>항을 정리하면
>
>$$
>M_k
>\left[\hat{\boldsymbol{\beta}}^{(i)}(k) - \hat{\boldsymbol{\beta}}(k)
>\right] =
>\mathbf{x}_i \left[ \mathbf{x}_i^T\hat{\boldsymbol{\beta}}^{(i)}(k)-y_i \right].
>$$
>
>$i$번째 관측값의 leave-one-out 잔차를 $e_i^{(i)} = y_i-\mathbf{x}_i^T\hat{\boldsymbol{\beta}}^{(i)}(k)$ 라고 정의하면 $\mathbf{x}_i^T\hat{\boldsymbol{\beta}}^{(i)}(k)-y_i = -e_i^{(i)}$ 이다. 따라서
>
>$$
>M_k
>\left[
>\hat{\boldsymbol{\beta}}^{(i)}(k)-\hat{\boldsymbol{\beta}}(k)
>\right]
>=-\mathbf{x}_ie_i^{(i)}.
>$$
>
>양변에 $M_k^{-1}$을 곱하면
>
>$$
>\hat{\boldsymbol{\beta}}^{(i)}(k)
>= \hat{\boldsymbol{\beta}}(k) - M_k^{-1}\mathbf{x}_ie_i^{(i)}.
>$$
>
>아직 오른쪽에 leave-one-out 잔차 $e_i^{(i)}$가 있으므로 이를 전체 자료 잔차 $e_i$로 바꾸어야 한다.
>
>2. Leave-one-out 잔차를 전체 자료 잔차로 표현
>
>전체 자료로 적합한 능형회귀 잔차는 $e_i=y_i-\mathbf{x}_i^T\hat{\boldsymbol{\beta}}(k)$ 이다.위 $\hat{\boldsymbol{\beta}}^{(i)}(k)$의 양변에 $\mathbf{x}_i^T$를 곱하면
>
>$$
>\mathbf{x}_i^T\hat{\boldsymbol{\beta}}^{(i)}(k)
>= \mathbf{x}_i^T\hat{\boldsymbol{\beta}}(k) - \mathbf{x}_i^TM_k^{-1}\mathbf{x}_i e_i^{(i)}.
>$$
>
>행렬 $A=X(X^TX+kI_p)^{-1}X^T =XM_k^{-1}X^T$ 의 $i$번째 대각원소는 $a_{ii}=\mathbf{x}_i^TM_k^{-1}\mathbf{x}_i$이다. 따라서
>
>$$
>\mathbf{x}_i^T\hat{\boldsymbol{\beta}}^{(i)}(k)
>= \mathbf{x}_i^T\hat{\boldsymbol{\beta}}(k) -a_{ii}e_i^{(i)}.
>$$
>
>양변을 $y_i$에서 빼면
>
>$$
>y_i-\mathbf{x}_i^T\hat{\boldsymbol{\beta}}^{(i)}(k) = y_i-\mathbf{x}_i^T\hat{\boldsymbol{\beta}}(k) +a_{ii}e_i^{(i)}.
>$$
>
>왼쪽은 $e_i^{(i)}$이고, 오른쪽 첫 두 항은 $e_i$이므로 $e_i^{(i)} = e_i+a_{ii}e_i^{(i)}.$
>
>따라서 $(1-a_{ii})e_i^{(i)}=e_i$ 이고 $e_i^{(i)} = \frac{e_i}{1-a_{ii}}$ 를 얻는다. 따라서
>
>$$
>\begin{aligned}
>\hat{\boldsymbol{\beta}}^{(i)}(k)
>&= \hat{\boldsymbol{\beta}}(k) - M_k^{-1}\mathbf{x}_i\frac{e_i}{1-a_{ii}}\\
>&= \hat{\boldsymbol{\beta}}(k) - \frac{(X^TX+kI_p)^{-1}\mathbf{x}_ie_i}{1-a_{ii}}.
>\end{aligned}
>$$
>
>따라서 식 (13.24)가 성립한다.


[네번째 방법]

최근에는 **교차검증(cross validation)** 방법으로도 $k$를 선택한다. 여기서 $PRESS$는 교차검증의 특별한 형태인 **하나빼기-교차검증(LOOCV; leave-one-out cross validation)** 에 해당한다. 


## 13.3 주성분회귀 (Principal Component Regression, PCR)

주성분회귀(principal component regression)는 다변량 자료분석의 차원축소방법에 기초한 회귀분석 방법이다. 자세한 내용은 참고문헌(13.1, 13.9)을 참조하고, 여기서는 간단히 핵심 구조를 정리한다.

[13.2.1 능형회귀추정량의 성질]과 같이 $\Lambda$와 $P$를 각각 $X^TX$ 행렬의 고유값들을 대각원소로 갖는 행렬을 $\Lambda$, 그에 대응하는 고유벡터들을 열로 갖는 행렬을 $P$라고 하자. 그러면 식 (13.14)와 같이 선형모형을

$$\mathbf{y}=X\boldsymbol{\beta}+\boldsymbol{\epsilon} = XPP^T\boldsymbol{\beta}+\boldsymbol{\epsilon} = Z\boldsymbol{\alpha}+\boldsymbol{\epsilon}
\tag{13.26}$$

로 바꾸어 쓸 수 있다. 여기서 $Z=XP,\quad \boldsymbol{\alpha}=P^T\boldsymbol{\beta}$ 또한 $Z^TZ=P^TX^TXP=\Lambda$ 이므로 $\boldsymbol{\alpha}$의 최소제곱추정량(least squares estimator)은

$$\Lambda \hat{\boldsymbol{\alpha}}=Z^T\mathbf{y}
\tag{13.27}$$

를 만족한다. 그리고 $\boldsymbol{\beta}$의 최소제곱추정량은 $\boldsymbol{\alpha}=P^T\boldsymbol{\beta}$, $\boldsymbol{\beta}=P\boldsymbol{\alpha}$이므로

$$\hat{\boldsymbol{\beta}}=P\hat{\boldsymbol{\alpha}}
\tag{13.28}$$

이제 $p$개의 고유값 $\lambda_j$들 중에서 $s$개가 0이라고 하자. 그러면 $P$의 열들 가운데 고유값 0에 대응되는 고유벡터들은 $Z=XP$에서 해당 열이 영벡터가 된다. 따라서 모형 (13.26)에서 $Z$의 $s$개 열은 모두 0이 되고, 식 (13.27)은 $p$개의 방정식이 아니라 $g=p-s$개의 방정식만을 포함하게 된다.

이제 $P,\Lambda,\boldsymbol{\alpha}$를 다음과 같이 나눈다.

$$
P=(P_g,\;P_s),\qquad \boldsymbol{\alpha}^T=(\boldsymbol{\alpha}_g^T,\;\boldsymbol{\alpha}_s^T),\qquad
\Lambda= \begin{pmatrix}
\Lambda_g & 0\\
0 & \Lambda_s
\end{pmatrix}
$$

여기서 $\lambda_1\ge \lambda_2\ge \cdots \ge \lambda_p\ge 0$로 가정하고, $\Lambda_g$는 처음 $g$개의 고유값을 포함하는 대각행렬, $\Lambda_s$는 나머지 고유값들로 이루어진 대각행렬이다. 그러면 식 (13.27)은

$$\Lambda_g\hat{\boldsymbol{\alpha}}_g=P_g^TX^T\mathbf{y}
\tag{13.29}$$

가 되며, $\hat{\boldsymbol{\beta}}_g=P_g\hat{\boldsymbol{\alpha}}_g$로 표현된다. 이 식 (13.29)에 의하여 추정하는 방법을 주성분회귀라 한다.

고유값이 정확히 0이 아니더라도 매우 작아서 거의 0에 가까우면, 그에 대응하는 성분은 사실상 정보가 거의 없는 방향으로 볼 수 있다. 

예를 들어 $\lambda_j$가 0에 가까운 값이라 하자. 그러면 $\Lambda=P^TX^TXP=(XP)^T(XP)$ 이므로 $\lambda_j=p_j^TX^TXp_j=(Xp_j)^T(Xp_j)$로 쓸 수 있다. 따라서 $Xp_j\approx 0_n$이 되어 거의 영벡터(zero vector)에 가까워진다. 이 경우 $p_j$를 영벡터에 대응하는 방향으로 보고 $\lambda_j=0$으로 두어 회귀추정량을 계산하는 것이 주성분회귀분석의 기본 생각이다.  
또한 $Xp_j\approx 0_n$이면 $X$의 열들 사이에는 완전에 가까운 다중공선성이 존재한다고 볼 수 있다. 즉 $X^TX$의 $p$개 고유값 중 $s$개가 거의 0에 가까우면, $\boldsymbol{\beta}$의 주성분회귀추정량은

$$\hat{\boldsymbol{\beta}}_g=P_g\hat{\boldsymbol{\alpha}}_g=P_g\Lambda_g^{-1}P_g^TX^T\mathbf{y}
\tag{13.30}$$

가 된다. 한편 최소제곱추정량 $\hat{\boldsymbol{\beta}}$와는

$$\hat{\boldsymbol{\beta}}_g=(I-P_sP_s^T)\hat{\boldsymbol{\beta}}
\tag{13.31}$$

의 관계가 있다. 따라서 $\hat{\boldsymbol{\beta}}_g$는 $\boldsymbol{\beta}$의 편의된 추정량(biased estimator)이다.  

>**증명**  
>최소제곱추정량을 주성분 좌표로 표현하면 $\hat{\boldsymbol{\beta}} = P\hat{\boldsymbol{\alpha}} = P_g\hat{\boldsymbol{\alpha}}_g + P_s\hat{\boldsymbol{\alpha}}_s$ 이다. 주성분회귀추정량은 작은 고유값에 대응하는 $P_s$ 방향을 제거하므로 $\hat{\boldsymbol{\beta}}_g = P_g\hat{\boldsymbol{\alpha}}_g.$
>
>한편 $P_gP_g^T=I_p-P_sP_s^T$ 이므로
>
>$$
>\begin{aligned}
>(I_p-P_sP_s^T)\hat{\boldsymbol{\beta}}
>&= P_gP_g^T
>\left( P_g\hat{\boldsymbol{\alpha}}_g + P_s\hat{\boldsymbol{\alpha}}_s \right)\\
>&= P_g(P_g^TP_g)\hat{\boldsymbol{\alpha}}_g + P_g(P_g^TP_s)\hat{\boldsymbol{\alpha}}_s\\
>&= P_gI_g\hat{\boldsymbol{\alpha}}_g+P_g0\\
>&= P_g\hat{\boldsymbol{\alpha}}_g\\
>&= \hat{\boldsymbol{\beta}}_g.
>\end{aligned}
>$$
>
>따라서
>
>$$
>\hat{\boldsymbol{\beta}}_g = (I_p-P_sP_s^T)\hat{\boldsymbol{\beta}}
>$$
>
>$P_sP_s^T$는 제거된 고유벡터들이 생성하는 부분공간으로의 직교사영행렬이다. 따라서 식 (13.31)은 최소제곱추정량에서 불안정한 $P_s$ 방향의 성분을 제거한다는 의미이다.


기대값(expectation)과 분산은

$$E(\hat{\boldsymbol{\beta}}_g)=\boldsymbol{\beta}-\sum_{j=g+1}^{p}(\mathbf p_j^T\boldsymbol{\beta})\mathbf p_j
\tag{13.32}$$

>**증명**  
>주성분회귀추정량은 $\hat{\boldsymbol{\beta}}_g=P_g\Lambda_g^{-1}P_g^TX^Ty$이다. 선형모형에 의해 $E(y)=X\boldsymbol{\beta}$ 이다. 따라서
>
>$$
>E(\hat{\boldsymbol{\beta}}_g)  = P_g\Lambda_g^{-1}P_g^TX^TE(y) = P_g\Lambda_g^{-1}P_g^TX^TX\boldsymbol{\beta}.
>$$
>
>고유값분해 $X^TX=P\Lambda P^T$ 를 이용하면 $E(\hat{\boldsymbol{\beta}}_g) = P_g\Lambda_g^{-1}P_g^TP\Lambda P^T \boldsymbol{\beta}.$
>
>또한 $P_g^TP=(I_g,\;0)$이므로 $P_g^TP\Lambda = (\Lambda_g,\;0).$ 따라서 
>
>$$E(\hat{\boldsymbol{\beta}}_g) = P_g\Lambda_g^{-1} (\Lambda_g,\;0)P^T\boldsymbol{\beta} = P_gP_g^T\boldsymbol{\beta}.$$
>
>그런데 $P_gP_g^T=I_p-P_sP_s^T$ 이므로
>
>$$
>E(\hat{\boldsymbol{\beta}}_g) = (I_p-P_sP_s^T)\boldsymbol{\beta} = \boldsymbol{\beta}-P_sP_s^T\boldsymbol{\beta}.
>$$
>
>$P_s=(\mathbf p_{g+1},\ldots,\mathbf p_p)$이므로 $P_sP_s^T =\sum_{j=g+1}^p\mathbf p_j\mathbf p_j^T.$
>
>따라서
>
>$$
>P_sP_s^T\boldsymbol{\beta} = \sum_{j=g+1}^p \mathbf p_j\mathbf p_j^T\boldsymbol{\beta} = \sum_{j=g+1}^p (\mathbf p_j^T\boldsymbol{\beta})\mathbf p_j.
>$$
>
>결국
>
>$$
>\boxed{
>E(\hat{\boldsymbol{\beta}}_g)
>= \boldsymbol{\beta} - \sum_{j=g+1}^p (\mathbf p_j^T\boldsymbol{\beta})\mathbf p_j
>}
>$$


$$\text{Var}(\hat{\boldsymbol{\beta}}_g)=\sigma^2\sum_{j=1}^{g}\lambda_j^{-1}\mathbf p_j \mathbf p_j^T
\tag{13.33}$$

>**증명**
>
>$\text{Var}(\hat{\boldsymbol{\beta}}_g) = P_g\Lambda_g^{-1}P_g^TX^T
>\text{Var}(y)  XP_g\Lambda_g^{-1}P_g^T$이다. 선형모형의 가정에 의해 $\text{Var}(y)=\sigma^2I_n$ 이므로 $\text{Var}(\hat{\boldsymbol{\beta}}_g) =\sigma^2 P_g\Lambda_g^{-1}P_g^TX^TXP_g\Lambda_g^{-1}P_g^T$.
>
>각 $\mathbf p_j$가 $X^TX$의 고유벡터이므로 $X^TXP_g=P_g\Lambda_g.$ 따라서 $P_g^TX^TXP_g = P_g^TP_g\Lambda_g = \Lambda_g.$  
>이를 대입하면
>
>$$
>\text{Var}(\hat{\boldsymbol{\beta}}_g) = \sigma^2P_g\Lambda_g^{-1} \Lambda_g \Lambda_g^{-1}P_g^T = \sigma^2P_g\Lambda_g^{-1}P_g^T.
>$$
>
>$\Lambda_g^{-1} = \text{diag} \left(\frac1{\lambda_1},\ldots,\frac1{\lambda_g}\right)$ 이므로 $P_g\Lambda_g^{-1}P_g^T =\sum_{j=1}^g \lambda_j^{-1}\mathbf p_j\mathbf p_j^T.$
>
>따라서
>
>$$
>\boxed{
>\text{Var}(\hat{\boldsymbol{\beta}}_g)
>= \sigma^2 \sum_{j=1}^g \lambda_j^{-1}\mathbf p_j\mathbf p_j^T
>}
>$$

따라서 평균제곱오차(mean squared error)는

$$
MSE(\hat{\boldsymbol{\beta}}_g)
= E\left[(\hat{\boldsymbol{\beta}}_g-\boldsymbol{\beta})^T(\hat{\boldsymbol{\beta}}_g-\boldsymbol{\beta})\right]
= \sigma^2\sum_{j=1}^{g}\lambda_j^{-1}
+\sum_{j=g+1}^{p}(\mathbf p_j^T\boldsymbol{\beta})^2
\tag{13.34}
$$

>**증명**  
>벡터 추정량의 스칼라 평균제곱오차는 편의-분산 분해에 의해
>
>$$
>MSE(\hat{\boldsymbol{\beta}}_g)
>= E\left[ (\hat{\boldsymbol{\beta}}_g-\boldsymbol{\beta})^T (\hat{\boldsymbol{\beta}}_g-\boldsymbol{\beta}) \right] 
>= \text{tr} \left[\text{Var}(\hat{\boldsymbol{\beta}}_g)\right] +
>\left\|\text{Bias}(\hat{\boldsymbol{\beta}}_g)\right\|^2
>$$
>
>분산항: 식 (13.33)에 의해 $\text{tr}\left[\text{Var}(\hat{\boldsymbol{\beta}}_g)\right]
>= \sigma^2\sum_{j=1}^g \lambda_j^{-1} \text{tr}(\mathbf p_j\mathbf p_j^T).$  
>trace의 순환성에 의해 $\text{tr}(\mathbf p_j\mathbf p_j^T) = \text{tr}(\mathbf p_j^T\mathbf p_j) = \mathbf p_j^T\mathbf p_j = 1$ 이다. 따라서
>
>$$
>\text{tr} \left[ \text{Var}(\hat{\boldsymbol{\beta}}_g) \right]
>= \sigma^2\sum_{j=1}^g\lambda_j^{-1}
>$$
>
>편의제곱항: 식 (13.32)에 의해 편의는 $\text{Bias}(\hat{\boldsymbol{\beta}}_g) = -\sum_{j=g+1}^p (\mathbf p_j^T\boldsymbol{\beta})\mathbf p_j$ 이다. 그러므로
>
>$$
>\begin{aligned}
>\left\| \text{Bias}(\hat{\boldsymbol{\beta}}_g) \right\|^2
>&= \left[ \sum_{j=g+1}^p (\mathbf p_j^T\boldsymbol{\beta})\mathbf p_j
>\right]^T
>\left[ \sum_{\ell=g+1}^p (\mathbf p_\ell^T\boldsymbol{\beta})\mathbf p_\ell \right]\\
>&= \sum_{j=g+1}^p\sum_{\ell=g+1}^p (\mathbf p_j^T\boldsymbol{\beta})
>(\mathbf p_\ell^T\boldsymbol{\beta}) \mathbf p_j^T\mathbf p_\ell.
>\end{aligned}
>$$
>
>고유벡터들이 서로 직교하므로
>
>$$
>\mathbf p_j^T\mathbf p_\ell
>= \begin{cases}
>1,&j=\ell,\\
>0,&j\neq\ell.
>\end{cases}
>$$
>
>이다. 따라서 교차항이 모두 사라져
>
>$$
>\boxed{
>\left\| \text{Bias}(\hat{\boldsymbol{\beta}}_g) \right\|^2
>= \sum_{j=g+1}^p (\mathbf p_j^T\boldsymbol{\beta})^2
>}
>$$
>
>따라서 분산항과 편의제곱항을 더하면
>
>$$
>\boxed{
>MSE(\hat{\boldsymbol{\beta}}_g)
>= \sigma^2\sum_{j=1}^g\lambda_j^{-1} +
>\sum_{j=g+1}^p (\mathbf p_j^T\boldsymbol{\beta})^2
>}
>$$
>
>(정확히 $0$인 고유값에 관한 주의점)  
>식 (13.31)과 식 (13.35)의 일반적인 역행렬 표현은 모든 $\lambda_j>0$, 즉 $X$가 완전 열계수일 때 직접 성립한다. 어떤 $\lambda_j$가 정확히 $0$이면 $(X^TX)^{-1}$가 존재하지 않으므로 통상적인 최소제곱추정량은 유일하지 않다.  
>이 경우에는 무어–펜로즈 일반화역행렬을 사용한 최소노름 최소제곱추정량 $\hat{\boldsymbol{\beta}}^{+} = P_g\Lambda_g^{-1}P_g^TX^Ty$ 을 사용해야 한다. 따라서 PCR의 MSE 비교는 일반적으로 고유값이 정확히 $0$인 상황보다는 $0<\lambda_p\leq\cdots\leq\lambda_{g+1}\ll1$ 처럼 고유값들이 양수이지만 매우 작은 상황에서 해석하는 것이 정확하다.

이에 비해 최소제곱추정량 $\hat{\boldsymbol{\beta}}$의 평균제곱오차는 식 (13.8)에서 본 바와 같이

$$
MSE(\hat{\boldsymbol{\beta}})
= E\left[(\hat{\boldsymbol{\beta}}-\boldsymbol{\beta})^T(\hat{\boldsymbol{\beta}}-\boldsymbol{\beta})\right]
= \sigma^2\sum_{j=1}^{p}\lambda_j^{-1}
\tag{13.35}
$$

그러므로 다중공선성의 관계가 존재하여 $\lambda_p,\lambda_{p-1},\cdots,\lambda_{g+1}$이 0에 가까운 값이면

$$MSE(\hat{\boldsymbol{\beta}}_g)<MSE(\hat{\boldsymbol{\beta}})$$

를 만족하는 $1\le g<p$ 값이 존재하게 된다. 즉 주성분을 일부만 사용하여 편의를 도입하더라도, 분산 감소 효과가 더 커지면 평균제곱오차 기준으로 더 좋은 추정량이 될 수 있다.

### 13.3.1 $g$값의 선택

이제 자료로부터 $g$를 선택하는 문제를 생각한다. 여기서 $g$는 고유값이 0이 아닌 것들 가운데 큰 고유값을 갖는 개수이며, 동시에 사용할 주성분(principal components)의 개수를 의미한다.

일반적으로 주성분은 자료의 특징을 잘 이해하기 위해 필요한 최소한의 개수만 사용한다. 이를 위해 능형회귀(ridge regression)에서 능형트레이스(ridge trace)가 안정화되는 $k$를 선택한 것과 비슷하게, 주성분회귀에서는 $\lambda_1\ge \lambda_2\ge \cdots \ge \lambda_p$ 를 주성분 개수에 대해 그려 보고 고유값이 크게 떨어지는 지점에서 주성분 개수 $g$를 선택한다. 이것을 **산비탈 그림(scree plot)** 이라 한다. 고유값이 급격히 떨어지는 지점은 흔히 **팔꿈치(elbow)** 지점이라고 부른다.

그러나 이러한 방법으로 선택한 $g$는 주관적일 수 있다. 또한 이렇게 선택된 주성분이 반드시 반응변수를 잘 설명하는 설명변수가 된다는 보장이 없다는 문제가 있다. 주성분회귀는 설명변수의 전체 변동을 잘 설명하는 방향을 우선시하지만, 그 방향이 반응변수와 가장 관련이 큰 방향이라고는 할 수 없기 때문이다.

> 주석: 고윳값 대신 전체 자료의 분산 중 각 주성분이 차지하는 분산의 비율을 그리기도 한다. 전체자료의 분산은 $\sum_{j=1}^{p}\widehat{\
Var}(X_j) = \sum_{j=1}^{p}1/n\sum_{i=1}^{n}X^2_{ij}$ 이고, $g$번째 주성분에 의해 설명되는 분산은 $\sum_{j=1}^p 1/n\sum_{i=1}^{n}Z^2_{ij}$ 이다.

이 문제를 보완하는 또 다른 편의회귀추정방법이 다음 절의 부분최소제곱 회귀(partial least squares regression)이다.

또한 $g$를 선택하는 다른 방법으로는 능형회귀에서 언급한 교차검증(cross validation) 방법을 사용할 수 있다.


## 13.4 부분최소제곱 회귀 (Partial Least Squares Regression, PLS)

주성분회귀모형에서는 설명변수 $X_1,X_2,\cdots,X_p$의 전체 변동을 가장 잘 나타내는 선형결합인 주성분을 찾고, 이를 새로운 설명변수로 사용하여 선형회귀모형을 고려한다. 이때 주성분의 개수는 $p$보다 작지만 설명변수의 전체 변동을 잘 설명할 수 있도록 선택한다.

그러나 이러한 회귀모형의 설명력이 항상 좋은 것은 아니다. 주성분회귀에서는 주성분을 결정할 때 반응변수 $Y$를 고려하지 않기 때문이다. 따라서 설명변수의 전체 변동을 가장 잘 요약하는 주성분이 반응변수를 가장 잘 설명한다고 기대하기는 어렵다. 즉, 주성분회귀에서 처음 $g(<p)$ 개의 주성분보다 설명변수로 사용하지 않은 $s(=p-g)$개의 주성분들이 반응변수와 더 강한 연관성을 가질 수도 있다.

이러한 단점을 보완하기 위하여 부분최소제곱 회귀(partial least squares regression)를 사용한다.

주성분회귀와 달리 부분최소제곱 회귀에서는 설명변수들의 변동뿐 아니라 동시에 반응변수를 잘 설명하는, 서로 직교하는 성질을 가진 소수의 설명변수들의 선형결합을 찾아 새로운 설명변수 $T_1,T_2,\cdots,T_p$로 정의한다. 그리고 새로운 변수 $T_l\;(l=1,\cdots,p)$과 반응변수 사이의 선형회귀모형을 가정한다. 즉,

$$\mathbf{y}=\mathbf{T}\boldsymbol{\alpha}+\epsilon \tag{13.36}$$

이다. 여기서 $\mathbf{T}=[\mathbf{T}_1,\cdots,\mathbf{T}_p]$, $\boldsymbol{\alpha}=(\alpha_1,\cdots,\alpha_p)^T$이고, $\boldsymbol{\alpha}$를 최소제곱법으로 추정한다. 여기서 $\mathbf{T}_l$은 **설명변수의 변동을 잘 설명할 뿐 아니라 반응변수도 잘 설명하는 설명변수의 선형결합으로 결정**된다. 따라서 설명변수에 대한 차원축소는 점에서는 주성분회귀와 비슷하지만, 선형결합의 방향을 정하는 방식에서 차이가 있다.

부분최소제곱 회귀에서는 $T_1,\cdots,T_p$를 서로 직교하도록 다음 절차에 의해 순차적으로 생성한다. 

먼저 계산을 단순하게 하기 위해 반응변수 $\mathbf{Y}$와 모든 설명변수 $\mathbf{X}$에 대해 중심 변환을 한다.

$$\mathbf{U}_1=\mathbf{Y}-\bar{\mathbf{Y}},\qquad \mathbf{V}_{1j}=\mathbf{X}_j-\bar{\mathbf{X}}_j,\quad (j=1,\cdots,p)$$

그리고 각 평균이 0이라고 가정하면, 이들 확률변수에 대한 $n$개의 관측값은

$$\mathbf{u}_1=\mathbf{y}-\bar{\mathbf{y}}\cdot 1_n,\qquad \mathbf{v}_{1j}=\mathbf{x}_j-\bar{\mathbf{x}}_j\cdot \mathbf{1}_n$$

로 쓸 수 있다. 즉,

$$
\mathbf{u}_1= \begin{pmatrix}
y_1-\bar{y}\\
y_2-\bar{y}\\
\vdots\\
y_n-\bar{y}
\end{pmatrix},
\qquad
\mathbf{v}_{1j}= \begin{pmatrix}
x_{1j}-\bar{x}_j\\
x_{2j}-\bar{x}_j\\
\vdots\\
x_{nj}-\bar{x}_j
\end{pmatrix}
$$

이 변환 후 첫 번째 새로운 설명변수 $T_1$은 설명변수 $V_{1j},\;(j=1,\cdots,p)$의 선형결합으로 결정된다. 이때 각 설명변수에 대한 가중치(weight)는 설명변수 $V_{1j}$와 반응변수 $U_1$에 대한 $p$개의 단순회귀모형 

$$U_1=\phi_{1j}V_{1j}+\epsilon_j,\qquad j=1,\cdots,p \tag{13.37}$$

으로부터 구한 회귀계수의 추정값 $\hat{\phi}_{1j}=\frac{\mathbf{v}_{1j}^T\mathbf{u}_1}{\mathbf{v}_{1j}^T\mathbf{v}_{1j}}$ 을 사용한다.

그러나 각 설명변수에 대해 단순회귀모형을 적합하는 방식은 설명변수들 사이의 연관성을 고려하지 못한다. 따라서 각 회귀계수 추정값에 다시 가중치 $w_{1j}$를 부여하여 최종적으로 첫 번째 성분 $T_1$을 다음과 같이 정한다.

$$
T_1=\sum_{j=1}^{p}w_{1j}\hat{\phi}_{1j}V_{1j} \tag{13.38}
$$

여기서 $\sum_{j=1}^{p}w_{1j}=1$ 이다. 이와 같이 정의된 $T_1$은 설명변수 $V_{1j}$들의 가중평균(weighted mean)으로서 $U_1$, 즉 $Y$에 대해 높은 설명력을 나타내도록 구성된다.

다음으로 $T_1$이 주어진 상태에서 $T_1$에 의해 설명되지 않는 $X_1,\cdots,X_p$의 변동과 반응변수 $Y$의 변동을 잘 설명하는 선형결합 $T_2$를 찾는다. 이를 위해 먼저 $T_1$이 설명하지 못하는 $X_j\;(j=1,\cdots,p)$들의 변동을 계산하고, 동시에 $T_1$이 설명하지 못하는 $Y$의 변동도 계산한다.

즉 $X_j$를 반응변수로 하고 $T_1$을 설명변수로 하는 회귀모형의 잔차를 $V_{2j}$라 하고, $U_1$을 반응변수로 하고 $T_1$을 설명변수로 하는 회귀모형의 잔차를 $U_2$라 한다. 그러면 $V_{2j}$와 $U_2$를 이용하여 두 번째 성분 $T_2$를 $V_{2j}$들의 선형결합으로 정한다. 이때의 가중치는 $T_1$을 정할 때와 마찬가지로 $V_{2j}$와 $U_2$로부터 구한 단순회귀계수 추정값으로 정한다.

이 절차를 반복하여 $T_2,\cdots,T_p$를 순차적으로 결정한다. 일반적으로 $T_l$이 주어지면, $V_{lj}$를 반응변수로 하고 $T_l$을 설명변수로 하는 회귀모형의 잔차 $V_{(l+1)j},\;(j=1,\cdots,p)$와 $U_l$을 반응변수로 하고 $T_l$을 설명변수로 하는 회귀모형의 잔차 $U_{l+1}$를 만든 뒤, 이들로부터 $T_{l+1}$를 구성한다.

#### 부분최소제곱 알고리즘
위 설명을 알고리즘으로 표현하면 아래와 같다.

1. 설명변수 $X_j$와 반응변수 $Y$에 대하여 표본평균을 0으로 하는 새로운 변수를 만든다. ($j=1,\cdots,p$)

    $$
    V_{1j}=X_j-\bar{X}_j,\quad U_1=Y-\bar{Y}
    $$

2. 설명변수를 $V_{1j}$, 반응변수를 $U_1$로 하는 단순회귀모형의 회귀계수 추정량은

    $$
    \hat{\phi}_{1j}=\frac{\mathbf{v}_{1j}^T\mathbf{u}_1}{\mathbf{v}_{1j}^T\mathbf{v}_{1j}}
    $$

3. 첫 번째 새로운 변수 $T_1$은

    $$
    T_1=\sum_{j=1}^{p}w_{1j}\hat{\phi}_{1j}V_{1j}
    $$

4. $l=1,\cdots,m-1$에 대하여 다음을 반복한다.

[4.1]  
순차적으로 $T_{l+1}$을 생성하기 위해 $T_l$로 설명되지 않는 설명변수 $V_{lj}$와 반응변수 $U_l$를 각각 단순회귀모형의 잔차로부터 구하여 $V_{(l+1)j}$와 $U_{l+1}$라 하고, 다음과 같이 정한다.

$$
\mathbf{V}_{(l+1)j}=\mathbf{V}_{lj}-\left[\frac{\mathbf{t}_l^T\mathbf{v}_{lj}}{\mathbf{t}_l^T\mathbf{t}_l}\right]\mathbf{T}_l,\qquad j=1,\cdots,p \\
\mathbf{U}_{l+1}=\mathbf{U}_l-\left[\frac{\mathbf{t}_l^T\mathbf{u}_l}{\mathbf{t}_l^T\mathbf{t}_l}\right]\mathbf{T}_l$$

여기서 $\mathbf{t}_l,\mathbf{u}_l,\mathbf{v}_{lj}$는 각각 $\mathbf{T}_l,\mathbf{U}_l,\mathbf{V}_{lj}$의 관측벡터이다.

[4.2]  
다음으로 설명변수를 $\mathbf{V}_{(l+1)j}$, 반응변수를 $\mathbf{U}_{l+1}$로 하는 단순회귀모형의 회귀계수는

$$\hat{\phi}_{(l+1)j} = \frac{\mathbf{V}_{(l+1)j}^T\mathbf{U}_{l+1}}{\mathbf{V}_{(l+1)j}^T\mathbf{V}_{(l+1)j}}$$

로 추정한다.

[4.3]  
위 식으로부터 

$$\mathbf{T}_{l+1}=\sum_{j=1}^{p}w_{(l+1)j}\hat{\phi}_{(l+1)j}\mathbf{V}_{(l+1)j}$$

로 정한다.

이 알고리즘은 주성분회귀와 마찬가지로 새로운 설명변수 $T_l\;(l=1,\cdots,p)$들의 표본상관계수(sample correlation coefficient)를 0으로 한다. 즉 새로운 성분들 사이가 서로 직교하도록 만든다. 따라서 모형에 다른 주성분이 더해지거나 빠져도 이미 구한 성분의 역할이 바뀌지 않는 구조를 가진다.

이 알고리즘을 완성하려면 가중치 $w_{lj}$를 정해야 한다. 가중치를 정하는 첫 번째 방법은 부분최소제곱 회귀에서 흔히 쓰이는 방식으로 $w_{lj}\propto \text{Var}(\mathbf{V}_{lj})$로 놓는다. 이 경우 $\hat{\phi}_{lj}$의 분산은 $\mathbf{V}_{lj}$의 분산에 반비례한다. 이 경우

$$w_{1j}\left[\frac{\mathbf{v}_{1j}^T\mathbf{u}_1}{\mathbf{v}_{1j}^T\mathbf{v}_{1j}}\right]=\mathbf{v}_{1j}^T\mathbf{u}_1 \\
\mathbf{T}_1=\sum_{j=1}^{p}(\mathbf{v}_{1j}^T\mathbf{u}_1)\mathbf{v}_{1j}\propto \sum_{j}\widehat{Cov}(\mathbf{v}_{1j},\mathbf{u}_1)\mathbf{v}_{1j}$$

가 되어, 새로운 변수 $\mathbf{T}_1$의 결정이 설명변수와 반응변수의 공분산(covariance)에 의해 이루어진다는 점을 알 수 있다.

두 번째 방법은 가중치를 $w_{lj}=\frac{1}{p}$ 로 두는 것이다.  
가중치 선택에 따라 부분최소제곱 회귀는 서로 다른 불변성(invariance) 성질을 가진다. 첫 번째 가중치는 설명변수 $X$의 직교변환(orthogonal transformations)에 대하여 반응변수 $Y$의 예측이 불변이고, 두 번째 방법은 설명변수 $X$의 척도변환(scale transformations)에 대하여 반응변수 $Y$의 예측값이 불변이다. (더 자세한것은 참고문헌 13.13 참고)

한편 부분최소제곱 회귀에서 $T_l$은 반복적으로 생성되므로, 실제로는 적당한 개수의 주성분 $T_1,\cdots,T_g$만 사용하기 위하여 $g$를 정해야 한다. 이 값은 교차검증(cross validation) 방법을 사용하여 정한다.


## 13.5 제한회귀 (Restricted Regression)

중회귀모형(multiple regression model) $\mathbf{y}=X\boldsymbol{\beta}+\boldsymbol{\epsilon},\quad \boldsymbol{\epsilon}\sim N(0_n,\sigma^2I_n)$ 에서 회귀계수벡터 $\boldsymbol{\beta}$에 제한조건

$$C\boldsymbol{\beta}=\mathbf{0}_k \tag{13.39}$$

이 성립한다고 가정하자. 문제를 간단히 하기 위하여 $\mathbf{m}=\mathbf{0}_k$인 경우를 다룬다. 여기서 $X$는 $n\times (p+1)$ 행렬이고 계수는 $p+1$이며, $C$는 $k\times (p+1)$ 행렬이고 계수는 $k(\le p+1)$이다.  
이 제한조건 아래에서 $\boldsymbol{\beta}$의 최소제곱추정량은

$$
\tilde{\boldsymbol{\beta}}
= \left[ I_{p+1}-(X^TX)^{-1}C^T\{C(X^TX)^{-1}C^T\}^{-1}C \right]\hat{\boldsymbol{\beta}}
$$

임을 식 (5.29)에서 보였다. 만약 $C\boldsymbol{\beta}=\mathbf{0}_k$가 사실이라면 $\tilde{\boldsymbol{\beta}}$는 $\boldsymbol{\beta}$의 불편추정량이다. 그러나 $C\boldsymbol{\beta}=\mathbf{0}_k$가 사실이 아니면 $\tilde{\boldsymbol{\beta}}$는 $\boldsymbol{\beta}$의 편의추정량이 된다. 이와 같은 제한추정량(restricted estimator) $\tilde{\boldsymbol{\beta}}$의 분산은

$$
\text{Var}(\tilde{\boldsymbol{\beta}})
= \sigma^2(X^TX)^{-1}-\sigma^2(X^TX)^{-1}C^T[C(X^TX)^{-1}C^T]^{-1}C(X^TX)^{-1}
\tag{13.40}
$$

여기서 $(X^TX)^{-1}C^T[C(X^TX)^{-1}C^T]^{-1}C(X^TX)^{-1}$ 은 양정치(positive definite) 행렬이므로 $\text{Var}(\tilde{\boldsymbol{\beta}}_j)<\text{Var}(\hat{\boldsymbol{\beta}}_j)$가 된다. 즉 제한을 가하면 분산은 줄어든다.  
그러나 $C\boldsymbol{\beta}=\mathbf{0}_k$가 사실이 아니면 $\tilde{\boldsymbol{\beta}}$는 편의추정량이 되며, 그 편의(bias)는

$$E(\tilde{\boldsymbol{\beta}})-\boldsymbol{\beta}
= -(X^TX)^{-1}C^T[C(X^TX)^{-1}C^T]^{-1}C\boldsymbol{\beta}
\tag{13.41}$$

이제 최소제곱추정량 $\hat{\boldsymbol{\beta}}=(X^TX)^{-1}X^T\mathbf{y}$와 제한회귀추정량 $\tilde{\boldsymbol{\beta}}$의 평균제곱오차를 비교한다.  
최소제곱추정량은 불편추정량이므로

$$
MSE(\hat{\boldsymbol{\beta}})
= E\left[(\hat{\boldsymbol{\beta}}-\boldsymbol{\beta})^T(\hat{\boldsymbol{\beta}}-\boldsymbol{\beta})\right]
= \sigma^2\,tr[(X^TX)^{-1}]
$$

한편 $\tilde{\boldsymbol{\beta}}$는 편의추정량이므로 Bias-Variance분해와 식 (13.40), (13.41)로부터

$$
\begin{aligned}
MSE(\tilde{\boldsymbol{\beta}})
&= E\left[(\tilde{\boldsymbol{\beta}}-\boldsymbol{\beta})^T(\tilde{\boldsymbol{\beta}}-\boldsymbol{\beta})\right] \\
&= tr[\text{Var}(\tilde{\boldsymbol{\beta}})]+[E(\tilde{\boldsymbol{\beta}})-\boldsymbol{\beta}]^T[E(\tilde{\boldsymbol{\beta}})-\boldsymbol{\beta}] \\
&= \sigma^2\,tr[(X^TX)^{-1}] -\sigma^2\,tr[(X^TX)^{-1}C^T\{C(X^TX)^{-1}C^T\}^{-1}C(X^TX)^{-1}] \\
&\quad +\boldsymbol{\beta}^TC^T[C(X^TX)^{-1}C^T]^{-1}C(X^TX)^{-2}C^T[C(X^TX)^{-1}C^T]^{-1}C\boldsymbol{\beta}
\end{aligned}
$$

가 된다. 이제 $B=(X^TX)^{-1}C^T[C(X^TX)^{-1}C^T]^{-1}C$ 로 놓으면 평균제곱오차의 차이는

$$
MSE(\hat{\boldsymbol{\beta}})-MSE(\tilde{\boldsymbol{\beta}})
= \sigma^2\,tr[B(X^TX)^{-1}]-(B\boldsymbol{\beta})^T(B\boldsymbol{\beta})
\tag{13.42}
$$

여기서 $B(X^TX)^{-1}$은 양정치행렬이므로 $tr[B(X^TX)^{-1}]>0$ 이고, $(B\boldsymbol{\beta})^T(B\boldsymbol{\beta})$는 제곱합 형태이므로 음이 아닌 값이다. 따라서 $(C\boldsymbol{\beta}=\mathbf{0}_k)$이 정확히 성립하지 않더라도 거의 성립하는 경우에는 $(B\boldsymbol{\beta})^T(B\boldsymbol{\beta})$가 작아져 $(\tilde{\boldsymbol{\beta}})$가 $(\hat{\boldsymbol{\beta}})$보다 평균제곱오차 기준에서 더 우수한 추정량이 될 수 있다. 반대로 $(C\boldsymbol{\beta}=\mathbf{0}_k)$가 사실과 많이 다르면 $(B\boldsymbol{\beta})^T(B\boldsymbol{\beta})$가 커져 $(\hat{\boldsymbol{\beta}})$가 더 나은 추정량이 된다.

따라서 두 추정량 가운데 어느 것이 더 적절한지 판단하는 하나의 기준으로

$$D=\sigma^2tr[B(X^TX)^{-1}]-(B\boldsymbol{\beta})^T(B\boldsymbol{\beta}) \tag{13.43}$$

를 사용한다. 실제 적용에서는 최소제곱법에 의하여 $\sigma^2$와 $\boldsymbol{\beta}$를 각각 $\text{MSE}$와 $\hat{\boldsymbol{\beta}}$로 추정한 후 이를 식 (13.43)에 대입하여 $D$를 계산한다. 만약 $D>0$이면 $\tilde{\boldsymbol{\beta}}$가 더 적절하고, $D<0$이면 $\hat{\boldsymbol{\beta}}$를 사용하는 것이 바람직하다.

---

[추가]

제한조건이 $C\boldsymbol{\beta}=\mathbf m$ 인 일반적인 경우에는 $\mathbf m=\mathbf0_k$인 경우에서 제한 위반량 $C\hat{\boldsymbol{\beta}}$를 $C\hat{\boldsymbol{\beta}}-\mathbf m$ 으로 바꾸면 된다.

다음을 가정하자. $Q=(X^TX)^{-1}, \quad S=CQC^T.$ $X$는 완전열계수이고 $C$는 완전행계수라고 가정하면 $Q$와 $S^{-1}$가 존재한다.

**1. 제한최소제곱추정량**

다음 목적함수를 최소화한다. $(\mathbf y-X\boldsymbol{\beta})^T (\mathbf y-X\boldsymbol{\beta})$. 단, 제한조건은 $C\boldsymbol{\beta}=\mathbf m$이다.

라그랑주함수를

$$
\mathcal L(\boldsymbol{\beta},\boldsymbol{\lambda})
= (\mathbf y-X\boldsymbol{\beta})^T (\mathbf y-X\boldsymbol{\beta}) + 2\boldsymbol{\lambda}^T(C\boldsymbol{\beta}-\mathbf m)
$$

로 놓는다. $\boldsymbol{\beta}$에 관하여 미분하면

$$
\frac{\partial\mathcal L}{\partial\boldsymbol{\beta}}
= -2X^T(\mathbf y-X\boldsymbol{\beta}) + 2C^T\boldsymbol{\lambda}.
$$

이를 $\mathbf0$으로 놓으면 $X^TX\boldsymbol{\beta} = X^T\mathbf y-C^T\boldsymbol{\lambda}.$  따라서 $\boldsymbol{\beta} = \hat{\boldsymbol{\beta}} - QC^T\boldsymbol{\lambda}.$  
제한조건을 대입하면 $C\hat{\boldsymbol{\beta}} - CQC^T\boldsymbol{\lambda} = \mathbf m.$

따라서 $\boldsymbol{\lambda} = (CQC^T)^{-1} (C\hat{\boldsymbol{\beta}}-\mathbf m)$ 이다. 그러므로 일반적인 제한회귀추정량은

$$
\boxed{
\tilde{\boldsymbol{\beta}} = \hat{\boldsymbol{\beta}} - QC^T(CQC^T)^{-1} (C\hat{\boldsymbol{\beta}}-\mathbf m)
}
$$

이다. 원래 행렬을 모두 표시하면

$$
\boxed{
\begin{aligned}
\tilde{\boldsymbol{\beta}}
&= \hat{\boldsymbol{\beta}} - (X^TX)^{-1}C^T \left[ C(X^TX)^{-1}C^T \right]^{-1} (C\hat{\boldsymbol{\beta}}-\mathbf m).
\end{aligned}
}
\tag{13.39$'$}
$$

**2. 기대값과 편의**

다음을 정의하자. $L=QC^T(CQC^T)^{-1}.$ 그러면 제한추정량은 $\tilde{\boldsymbol{\beta}} = \hat{\boldsymbol{\beta}}-L(C\hat{\boldsymbol{\beta}}-\mathbf m)$ 이다.  
최소제곱추정량이 불편추정량이므로 $E(\hat{\boldsymbol{\beta}})=\boldsymbol{\beta}.$

따라서

$$
E(\tilde{\boldsymbol{\beta}}) = E(\hat{\boldsymbol{\beta}}) - L\left[ CE(\hat{\boldsymbol{\beta}})-\mathbf m\right] = \boldsymbol{\beta} - L(C\boldsymbol{\beta}-\mathbf m).
$$

즉,

$$
\boxed{
E(\tilde{\boldsymbol{\beta}})
= \boldsymbol{\beta} - (X^TX)^{-1}C^T \left[C(X^TX)^{-1}C^T\right]^{-1} (C\boldsymbol{\beta}-\mathbf m)
}
$$

따라서 편의벡터는

$$
\boxed{
\begin{aligned}
\text{Bias}(\tilde{\boldsymbol{\beta}})
&= E(\tilde{\boldsymbol{\beta}}) -\boldsymbol{\beta}\\
&= -(X^TX)^{-1}C^T
\left[ C(X^TX)^{-1}C^T \right]^{-1}
(C\boldsymbol{\beta}-\mathbf m).
\end{aligned}
}
\tag{13.41$'$}
$$

실제 모수가 제한조건을 정확히 만족한다면 $C\boldsymbol{\beta}-\mathbf m=\mathbf0_k$ 이므로 $E(\tilde{\boldsymbol{\beta}}) = \boldsymbol{\beta}.$  
따라서 제한조건이 정확하면 $\tilde{\boldsymbol{\beta}}$는 불편추정량이다.

**3. 분산·공분산행렬**

제한추정량은 $\tilde{\boldsymbol{\beta}} = (I_{p+1}-LC)\hat{\boldsymbol{\beta}}+L\mathbf m$ 으로 쓸 수 있다. $L\mathbf m$은 확률변수가 아닌 상수이므로 분산에는 영향을 주지 않는다.

따라서 $\text{Var}(\tilde{\boldsymbol{\beta}}) = (I_{p+1}-LC) \text{Var}(\hat{\boldsymbol{\beta}}) (I_{p+1}-LC)^T.$

최소제곱추정량의 분산은 $\text{Var}(\hat{\boldsymbol{\beta}})= \sigma^2Q$ 이므로

$$
\text{Var}(\tilde{\boldsymbol{\beta}}) = \sigma^2(I_{p+1}-LC)Q(I_{p+1}-LC)^T.
$$

이를 전개하면

$$
(I_{p+1}-LC)Q(I_{p+1}-LC)^T = Q-LCQ-QC^TL^T +LCQC^TL^T.
$$

그런데 $LCQ = QC^T(CQC^T)^{-1}CQ$ 이고 $QC^TL^T = QC^T(CQC^T)^{-1}CQ$ 이다. 또한

$$
\begin{aligned}
LCQC^TL^T
&= [QC^T(CQC^T)^{-1} C][QC^T (CQC^T)^{-1}CQ]\\
&= QC^T(CQC^T)^{-1}CQ.
\end{aligned}
$$

따라서 세 항을 정리하면

$$
\boxed{
\text{Var}(\tilde{\boldsymbol{\beta}}) = \sigma^2Q - \sigma^2QC^T(CQC^T)^{-1}CQ
}
\tag{13.40$'$}
$$

이다. 즉,

$$
\boxed{
\begin{aligned}
\text{Var}(\tilde{\boldsymbol{\beta}})
&= \sigma^2(X^TX)^{-1} - \sigma^2(X^TX)^{-1}C^T
\left[C(X^TX)^{-1}C^T\right]^{-1} C(X^TX)^{-1}.
\end{aligned}
}
$$

분산 공식은 $\mathbf m=\mathbf0_k$인 경우와 동일하다. $\mathbf m$은 상수항으로만 들어가기 때문이다.

**4. 제한추정량의 분포**

오차항이 정규분포를 따르므로 최소제곱추정량은 $\hat{\boldsymbol{\beta}} \sim N\left( \boldsymbol{\beta}, \sigma^2Q \right)$ 이다. 따라서 제한추정량도 정규분포를 따르며

$$
\boxed{
\tilde{\boldsymbol{\beta}}
\sim
N\left(
\boldsymbol{\beta}-L(C\boldsymbol{\beta}-\mathbf m),
\;
\sigma^2
\left[Q-QC^T(CQC^T)^{-1}CQ\right]
\right)
}
$$

제한조건이 정확하면 $C\boldsymbol{\beta}=\mathbf m$이므로

$$
\tilde{\boldsymbol{\beta}}
\sim
N\left(
\boldsymbol{\beta},
\;
\sigma^2
\left[Q-QC^T(CQC^T)^{-1}CQ\right]
\right).
$$

**5. 평균제곱오차**

편의-분산 분해를 사용하면 $MSE(\tilde{\boldsymbol{\beta}})
= \text{tr} \left[\text{Var}(\tilde{\boldsymbol{\beta}})\right] + \left\|\text{Bias}(\tilde{\boldsymbol{\beta}})\right\|^2.$  
제한조건의 실제 위반 정도를 $\boldsymbol{\delta} =C\boldsymbol{\beta}-\mathbf m$ 이라고 하자. 그러면 $\text{Bias}(\tilde{\boldsymbol{\beta}}) = -L\boldsymbol{\delta}.$

따라서 편의제곱은

$$
\begin{aligned}
\left\|
\text{Bias}(\tilde{\boldsymbol{\beta}})
\right\|^2
&= \boldsymbol{\delta}^TL^TL\boldsymbol{\delta}\\
&= \boldsymbol{\delta}^T
(CQC^T)^{-1}
CQ^2C^T
(CQC^T)^{-1}
\boldsymbol{\delta}.
\end{aligned}
$$

그러므로

$$
\boxed{
\begin{aligned}
MSE(\tilde{\boldsymbol{\beta}})
&=
\sigma^2\text{tr}(Q)\\
&\quad-
\sigma^2\text{tr}
\left[
QC^T(CQC^T)^{-1}CQ
\right]\\
&\quad+
(C\boldsymbol{\beta}-\mathbf m)^T
(CQC^T)^{-1}CQ^2C^T(CQC^T)^{-1}
(C\boldsymbol{\beta}-\mathbf m).
\end{aligned}
}
$$

원래 행렬을 복원하면

$$
\begin{aligned}
MSE(\tilde{\boldsymbol{\beta}})
&=
\sigma^2\text{tr}[(X^TX)^{-1}]\\
&\quad-
\sigma^2\text{tr}\left[
(X^TX)^{-1}C^T
\{C(X^TX)^{-1}C^T\}^{-1}
C(X^TX)^{-1}
\right]\\
&\quad+
(C\boldsymbol{\beta}-\mathbf m)^T
\{C(X^TX)^{-1}C^T\}^{-1}\\
&\qquad\times
C(X^TX)^{-2}C^T
\{C(X^TX)^{-1}C^T\}^{-1}
(C\boldsymbol{\beta}-\mathbf m).
\end{aligned}
$$

제한조건이 정확하면 $\boldsymbol{\delta}=\mathbf0_k$이므로 편의제곱항은 사라지고

$$
\boxed{
\begin{aligned}
MSE(\tilde{\boldsymbol{\beta}})
&=
\sigma^2\text{tr}(Q)\\
&\quad-
\sigma^2\text{tr}
\left[
QC^T(CQC^T)^{-1}CQ
\right].
\end{aligned}
}
$$

**6. 최소제곱추정량과의 MSE 차이**

다음과 같이 정의하자. $B=QC^T(CQC^T)^{-1}C, \quad L=QC^T(CQC^T)^{-1}.$

이때 $B=LC$ 이고 $BQ=QC^T(CQC^T)^{-1}CQ.$

최소제곱추정량의 MSE는 $MSE(\hat{\boldsymbol{\beta}}) = \sigma^2\text{tr}(Q)$ 이므로 두 MSE의 차이는

$$
\boxed{
\begin{aligned}
D
&= MSE(\hat{\boldsymbol{\beta}}) - MSE(\tilde{\boldsymbol{\beta}})\\
&= \sigma^2\text{tr}(BQ) -
\left[ L(C\boldsymbol{\beta}-\mathbf m)\right]^T
\left[L(C\boldsymbol{\beta}-\mathbf m)\right].
\end{aligned}
}
\tag{13.42$'$}
$$

또는

$$
\boxed{
D = \sigma^2\text{tr}
\left[B(X^TX)^{-1}\right] -
\left\|
(X^TX)^{-1}C^T
\{C(X^TX)^{-1}C^T\}^{-1}
(C\boldsymbol{\beta}-\mathbf m)
\right\|^2
}
\tag{13.43$'$}
$$

이다.

따라서 판단 기준은 다음과 같다.

$$
D>0
\quad\Longrightarrow\quad
MSE(\tilde{\boldsymbol{\beta}}) < MSE(\hat{\boldsymbol{\beta}})
$$

이므로 제한추정량이 유리하다.

$$
D<0 \quad\Longrightarrow\quad MSE(\tilde{\boldsymbol{\beta}})> MSE(\hat{\boldsymbol{\beta}})
$$

이므로 최소제곱추정량이 유리하다.

실제 적용에서는 $\sigma^2\longrightarrow\hat{\sigma}^2, \quad \boldsymbol{\beta}\longrightarrow \hat{\boldsymbol{\beta}}$ 로 대체하여

$$
\boxed{
\hat D
= \hat{\sigma}^2\text{tr}(BQ) -
\left\|L(C\hat{\boldsymbol{\beta}}-\mathbf m)\right\|^2
}
$$

를 계산할 수 있다.

마지막으로, 본문의 “$B(X^TX)^{-1}$은 양정치행렬”이라는 표현은 일반적으로는 “양의 준정부호행렬”이라고 해야 정확하다. 실제로

$$
BQ=QC^T(CQC^T)^{-1}CQ
$$

는 대칭 양의 준정부호행렬이다. 따라서 각 계수의 분산은 증가하지 않지만, 모든 계수에서 반드시 엄격하게 감소한다고 단정할 수는 없다.

