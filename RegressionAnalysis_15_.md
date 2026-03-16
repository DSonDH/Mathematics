# Chapter 15 차원축소방법 (Dimension Reduction Methods)

최근 여러 분야에서 고차원 자료 (high-dimensional data) 또는 대용량 자료가 빠르게 생성되고 있다. 고차원 자료란 일반적으로 설명변수의 차원 $p$가 자료의 개수 $n$보다 커지는 경우를 의미한다. 한편 $p<n$이라 하더라도 설명변수의 차원이 증가하면 의미 있는 결과 분석에 필요한 자료 수가 기하급수적으로 증가하게 되는데, 이를 차원의 저주 (curse of dimensionality)라고 한다. 이러한 문제를 해결하기 위한 대표적 방법으로 차원축소 (dimension reduction)와 변수선택 (variable selection)이 사용된다.

고차원 선형회귀모형은 다음과 같이 둔다.

$$y_i=\beta_1x_{i1}+\cdots+\beta_px_{ip}+\varepsilon_i,\qquad \varepsilon_i\sim N(0,\sigma^2),\quad i=1,\dots,n \tag{17.1}$$

여기서

$$\beta=(\beta_1,\dots,\beta_p)^T$$

는 $p$차원의 회귀계수벡터 (regression coefficient vector)이다. 또한 모든 변수는 표준화 (standardization)되어

$$\sum_{i=1}^n y_i=0,\qquad \sum_{i=1}^n x_{ij}=0,\qquad \sum_{i=1}^n x_{ij}^2=1\quad (j=1,\dots,p)$$

를 만족한다고 가정한다.

고차원 회귀분석의 주요 목적은 일반적인 회귀분석과 마찬가지로 다음 두 가지이다.

1. 추정된 모형의 용이한 해석
2. 새로운 자료에 대한 예측의 정확성

그러나 설명변수의 차원이 커지면 최소제곱추정량 (least squares estimator)은 해석력과 예측력 측면에서 여러 문제를 일으킨다. 실제 고차원 자료에서는 반응변수와 직접적으로 관련되지 않은 설명변수가 대부분을 차지하는 경우가 많다. 예를 들어 유전체 자료에서는 소수의 변수만이 실제로 반응변수에 의미 있는 영향을 주고, 나머지 변수는 사실상 무의미할 수 있다. 이런 변수들을 그대로 포함하면 모형은 불필요하게 복잡해지고 해석도 어려워진다.

또한 예측 정확성 측면에서도 차원이 커질수록 최소제곱추정량은 불안정해진다. $n>p$라 하더라도 $n$이 $p$보다 충분히 크지 않으면 추정량의 변동이 커져 과적합 (overfitting)이 발생하기 쉽다. 더 나아가 $p>n$이면
$X^TX$ 가 비정칙행렬 (singular matrix)이 되어 최소제곱추정량 $(X^TX)^{-1}X^Ty$ 이 존재하지 않는다. 정규방정식 (normal equation)

$X^TX\beta=X^Ty$ 를 생각하면, 방정식의 수는 $n$개인데 미지수의 수는 $p$개이므로 $p>n$일 때 해가 유일하게 결정되지 않는다.

이러한 문제를 완화하기 위하여 능형회귀 (ridge regression), 주성분 회귀 (principal component regression), 부분최소제곱회귀 (partial least squares regression) 등이 사용된다. 이 가운데 능형회귀는 회귀계수를 직접 축소 (shrinkage)시켜 분산을 낮추는 대표적 방법이다.

## 15.1 능형회귀 (Ridge Regression)

15장에서 다룬 편의추정 (biased estimation)의 관점에서 능형회귀는 최소제곱추정량 대신 약간의 편의 (bias)를 허용하는 대신 분산 (variance)을 감소시켜 전체적인 평균제곱오차 (mean squared error, MSE)를 줄이는 방법이다. 여기서는 설명변수 행렬 $X$의 특잇값 분해 (singular value decomposition, SVD)를 이용하여 능형회귀추정량의 구조와 성질을 분석한다.

### 15.1.1 특잇값 분해와 최소제곱추정량 (SVD and Least Squares Estimator)

먼저 $n>p$인 경우를 생각한다. $n\times p$ 설명변수 행렬 $X$의 특잇값 분해는
$$X=UDP^T=\sum_{l=1}^p d_lu_lp_l^T \tag{17.2} $$
로 쓸 수 있다.

여기서

* $U$는 $n\times n$ 직교행렬 (orthogonal matrix)이다.
* $P$는 $p\times p$ 직교행렬이다.
* $u_l$와 $p_l$은 각각 $U$와 $P$의 $l$번째 열벡터이다.
* $D$는 블록행렬 (block matrix)이며, $n>p$일 때
  $$
  D=
  \begin{bmatrix}
  D_{p\times p}\\
  0_{(n-p)\times p}
  \end{bmatrix}
  \tag{17.3}
  $$
  와 같이 쓸 수 있다.
* $D_{p\times p}$의 대각원소 $d_1,\dots,d_p$는 $X$의 특잇값 (singular values)이다.

직교성에 의해

$$UU^T=U^TU=I_n,\qquad PP^T=P^TP=I_p$$

가 성립한다.

식 (17.2)로부터
$$ X^TX=PD^TU^TUDP^T=PD^TDP^T $$

이므로, $X^TX$의 고유값 분해 (eigenvalue decomposition)와의 관계도 알 수 있다. 즉 $X^TX$의 고유값은 $d_j^2$이고, $P$는 고유벡터행렬 (eigenvector matrix)이 된다.

따라서 최소제곱추정량은
$$ \hat{\beta}=(X^TX)^{-1}X^Ty$$

이며, 이를 특잇값 분해로 표현하면

$$\hat{\beta} = (PD^TDP^T)^{-1}PD^TU^Ty =P(D^TD)^{-1}D^TU^Ty \tag{17.4}$$

가 된다.

이 식은 최소제곱추정량이 특잇값이 작은 방향에서 불안정해질 수 있음을 보여 준다. 왜냐하면 $(D^TD)^{-1}$의 대각원소는 $1/d_j^2$이므로, 작은 $d_j$는 계수 추정량의 변동을 크게 증폭시키기 때문이다.

### 15.1.2 능형회귀추정량의 정의 (Definition of Ridge Estimator)

능형회귀추정량은 최소제곱추정량의 불안정성을 줄이기 위하여
$X^TX$ 에 양의 상수 $k$를 곱한 항등행렬을 더한 다음 역행렬을 취하여 정의한다. 즉 능형회귀추정량은

$$
\hat{\beta}(k)=(X^TX+kI_p)^{-1}X^Ty
\tag{17.5}
$$

이다. 여기서 $k>0$는 벌점모수 (penalty parameter)이다. 이를 특잇값 분해로 표현하면

$$ \hat{\beta}(k) = (PD^TDP^T+kI_p)^{-1}PD^TU^Ty$$

이고, $PP^T=I_p$를 이용하면

$$ \hat{\beta}(k) = P(D^TD+kI_p)^{-1}D^TU^Ty \tag{17.6}$$

이 식은 능형회귀가 최소제곱추정량과 비교하여 각 방향별 계수를 축소시키는 구조를 명확히 보여 준다.

#### 증명: 능형회귀의 축소 성질 (Shrinkage Property)

식 (17.4)와 식 (17.6)을 비교하면 최소제곱추정량은 각 방향에 대하여 $\frac{d_j}{d_j^2}$ 의 계수를 가지는 반면, 능형회귀추정량은 $\frac{d_j}{d_j^2+k}$
의 계수를 가진다. $k>0$이면 

$$
\frac{d_j}{d_j^2}>\frac{d_j}{d_j^2+k}
$$

이므로 능형회귀추정량은 최소제곱추정량보다 더 작은 크기로 회귀계수를 줄인다. 따라서 능형회귀는 최소제곱추정량을 축소하는 추정량이다.

### 15.1.3 고차원 자료 $(p>n)$에서의 능형회귀 (Ridge Regression for High-Dimensional Data)

이제 $p>n$인 고차원 자료를 생각한다. 이 경우 $X^TX$의 최대 계수(rank)는 $n$이므로, 정규방정식 $X^TX\beta=X^Ty$ 의 해는 유일하지 않고 무수히 많다. 다시 말해 최소제곱추정량은 일반적으로 존재하지 않거나 유일하지 않다.

이때 $r(X)=n(<p)$라 가정하면 $D$ 행렬은

$$
D=\begin{bmatrix}
D_{n\times n} & 0_{n\times (p-n)}
\end{bmatrix}
\tag{17.7}
$$

과 같은 구조를 갖는다.

그러나 능형회귀에서는

$$
(X^TX+kI_p)^{-1} =
\sum_{j=1}^p(d_j^2+k)^{-1}p_jp_j^T
$$

로 표현되며, $j>n$이면 $d_j=0$이지만 $k>0$이므로 $d_j^2+k=k>0$ 가 되어 역행렬이 존재한다. 따라서 최소제곱추정량과 달리 능형회귀추정량은 $p>n$인 경우에도 계산 가능하다. 이것이 능형회귀가 고차원 자료에서 특히 유용한 핵심 이유이다.

### 15.1.4 벌점모수 $k$에 따른 극한 성질 (Limiting Behavior with Respect to $k$)

능형회귀추정량의 성질은 벌점모수 $k$에 크게 의존한다. 각 성분에 대해

$$
\lim_{k\to 0}\frac{d_j}{d_j^2+k} =
\begin{cases}
d_j^{-1}, & d_j\neq 0 \\[4pt]
0, & d_j=0
\end{cases}
$$

가 성립한다. 따라서 $k\to 0$이면 $(X^TX+kI_p)^{-1}$ 은 Moore-Penrose 역행렬 (Moore-Penrose inverse)에 대응하는 형태로 수렴하며,

$$
\lim_{k\to 0}\hat{\beta}(k)=(X^TX)^{-}X^Ty
$$

가 된다. 여기서 $(X^TX)^{-}$는 일반역행렬 (generalized inverse)이다.

반대로

$$
\lim_{k\to\infty}\frac{d_j}{d_j^2+k}=0
$$

이므로

$$
\lim_{k\to\infty}\hat{\beta}(k)=0_{p\times 1}
$$

가 된다. 즉 $k$가 무한히 커지면 모든 회귀계수가 0으로 축소된다.

다만 $k$가 커진다고 해서 항상 추정량의 노름(norm)이 단조적으로 감소하는 것은 아니다. 즉 $k_1>k_2$라고 하더라도 반드시

$$
|\hat{\beta}(k_1)|<|\hat{\beta}(k_2)|
$$

가 성립하는 것은 아니다. 따라서 벌점모수의 선택은 단순히 “크면 더 많이 축소된다”는 직관만으로 다룰 수 없으며, 예측오차의 관점에서 결정되어야 한다.

### 15.1.5 능형회귀의 편의 (Bias) (Bias of Ridge Estimator)

능형회귀는 편의를 허용하는 대신 분산을 줄이는 방법이다. 기대값을 계산하면

$$
E[\hat{\beta}(k)]
= (X^TX+kI_p)^{-1}(X^TX)\beta
$$

이고, 이를 정리하면

$$
E[\hat{\beta}(k)]
= \beta-k(X^TX+kI_p)^{-1}\beta
$$

또는 특잇값 분해를 이용하여

$$
E[\hat{\beta}(k)]
=\beta-kP(D^TD+kI_p)^{-1}P^T\beta
$$

로 쓸 수 있다. 따라서 편의는

$$
E[\hat{\beta}(k)]-\beta
=-kP(D^TD+kI_p)^{-1}P^T\beta
$$

이며, 편의의 제곱은

$$(E[\hat{\beta}(k)]-\beta)^T(E[\hat{\beta}(k)]-\beta)
=k^2\beta^TP(D^TD+kI_p)^{-2}P^T\beta
$$

가 된다.

#### 증명: 능형회귀의 기대값 계산 (Derivation of Expectation)

선형모형 $y=X\beta+\varepsilon,\qquad E(\varepsilon)=0$ 에서 식 (17.5)에 기대값을 취하면

$$
E[\hat{\beta}(k)] =
(X^TX+kI_p)^{-1}X^TE(y)
= (X^TX+kI_p)^{-1}X^TX\beta
$$

가 된다. 여기에 $X^TX=(X^TX+kI_p)-kI_p$ 를 대입하면

$$
E[\hat{\beta}(k)]
= (X^TX+kI_p)^{-1}\{(X^TX+kI_p)-kI_p\}\beta
= \beta-k(X^TX+kI_p)^{-1}\beta
$$

를 얻는다. 따라서 능형회귀는 일반적으로 불편추정량 (unbiased estimator)이 아니며, $k>0$일 때 편의를 가진다.

### 15.1.6 능형회귀의 분산과 평균제곱오차 (Variance and Mean Squared Error)

식 (17.6)으로부터 분산을 계산하면

$$
\mathrm{Var}[\hat{\beta}(k)]
= \sigma^2(X^TX+kI_p)^{-1}X^TX(X^TX+kI_p)^{-1}
$$

이고, 특잇값 분해를 이용하면

$$
\mathrm{Var}[\hat{\beta}(k)]
= \sigma^2P(D^TD+kI_p)^{-1}D^TD(D^TD+kI_p)^{-1}P^T
$$

가 된다.

$p>n$인 경우 $j=n+1,\dots,p$에 대하여 $(D^TD)_j=0$이므로, 해당 방향에서는 분산 기여가 0이 된다.

능형회귀추정량의 평균제곱오차 (mean squared error, MSE)는
$$
\mathrm{MSE}[\hat{\beta}(k)]
= E\big[(\hat{\beta}(k)-\beta)^T(\hat{\beta}(k)-\beta)\big]
$$

이며, 이를 전개하면

$$
\mathrm{MSE}[\hat{\beta}(k)]
= \sigma^2\sum_{j=1}^{n}\frac{d_j^2}{(d_j^2+k)^2}
+ k^2\sum_{j=1}^{n}\frac{(p_j^T\beta)^2}{(d_j^2+k)^2}
$$

이 식은 능형회귀의 본질을 잘 보여 준다.

* 첫 번째 항은 분산 항 (variance term)이다.
* 두 번째 항은 편의 제곱 항 (squared bias term)이다.

$k$가 커질수록 첫 번째 항은 작아지지만 두 번째 항은 커진다. 따라서 적절한 $k$를 선택하면 전체 MSE를 줄일 수 있다. 능형회귀가 실무에서 유용한 이유는 바로 이 편의-분산 절충 (bias-variance trade-off)에 있다.

### 15.1.7 고차원 자료에서의 계산 효율성 (Computational Efficiency in High Dimensions)

실제 고차원 자료에서는 능형회귀를 계산할 때 직접 $(X^TX+kI_p)^{-1}$ 를 계산하는 것이 매우 비효율적일 수 있다. 예를 들어 $n=100$, $p=4000$인 마이크로어레이 자료 (microarray data)를 생각하자. 이 경우 $p\times p$ 행렬은 $4000\times 4000$이므로 역행렬 계산 비용이 매우 크다.

이때 특잇값 분해를 이용하면 더 효율적인 계산이 가능하다. 식 (17.7)에서 특잇값이 0인 부분을 제거하여 $D$와 $P$를 각각 $n\times n$, $p\times n$ 형태로 쓰면, 능형회귀추정량은

$$
\hat{\beta}(k)
= (X^TX+kI_p)^{-1}X^Ty
= (PD^TDP^T+kI_p)^{-1}PD^TU^Ty
= P(D^TD+kI_n)^{-1}D^TU^Ty
$$

로 계산할 수 있다.

이 표현을 사용하면 $p\times p$ 행렬이 아니라 $n\times n$ 행렬만 역행렬 계산에 필요하므로 계산량이 크게 줄어든다. 예를 들어 $n=100$이면 $100\times100$ 행렬의 계산만으로 충분하다. 따라서 고차원 자료에서는 SVD 기반 표현이 계산상 매우 중요하다.

#### 예제: 마이크로어레이 자료의 계산 부담 완화 (Microarray Example)

자료가 $n=100$, $p=4000$인 경우를 생각하자. 직접 능형회귀를 계산하면 $4000\times4000$ 행렬의 역행렬이 필요하므로 일반적인 계산 환경에서는 부담이 매우 크다. 그러나 특잇값 분해를 이용하여 $n\times n$ 차원으로 계산을 바꾸면, 실질적으로 $100\times100$ 행렬의 연산만 수행하면 된다. 따라서 계산 복잡도가 크게 감소하며, 고차원 자료에 대한 능형회귀의 실제 적용이 가능해진다.


## 15.2 주성분회귀 (Principal Component Regression)

설명변수들 사이에 높은 상관관계가 존재하면 다중공선성 (multicollinearity)으로 인해 최소제곱추정량 (least squares estimator)을 안정적으로 구하기 어렵거나, 구하더라도 분산이 매우 커지는 문제가 발생한다. 이러한 상황에서는 차원축소 (dimension reduction)를 이용하여 회귀모형을 다시 구성하는 방법이 효과적이다. 주성분회귀 (principal component regression, PCR)는 이러한 목적을 위해 사용되는 대표적 방법이다.

주성분회귀는 설명변수 행렬 $X$의 전체 변동을 잘 설명하는 주성분 (principal components)을 먼저 구한 뒤, 그 중 일부만을 사용하여 반응변수 $y$를 회귀하는 방법이다. 즉, 원래의 $p$개 설명변수를 직접 사용하는 대신, 이들 변수의 선형결합으로 이루어진 소수의 주성분을 이용하여 회귀모형을 구성한다.

고차원 자료에서는 다중공선성이 더욱 자주 발생하므로, 앞에서 다룬 차원축소방법을 그대로 활용할 수 있다. 주성분회귀와 부분최소제곱회귀 (partial least squares regression)는 모두 약간의 편의 (bias)를 허용하는 대신 분산을 줄여 평균제곱오차 (mean squared error, MSE) 측면에서 더 효율적인 추정량을 얻으려는 접근이다.

이 절에서는 능형회귀와 마찬가지로 설명변수 행렬 $X$의 특잇값 분해 (singular value decomposition, SVD)를 이용하여 주성분회귀추정량을 정리한다.

### 15.2.1 주성분과 주성분회귀모형의 구성 (Construction of Principal Components and PCR Model)

설명변수 행렬 $X$의 특잇값 분해에서 직교행렬 $P=(p_1,\dots,p_p)$를 생각하자. 이때 첫 번째 주성분 (first principal component)은

$$z_1=Xp_1$$

일반적으로 $g$개의 주성분을 사용할 때, $P$를 $P=(P_g,\;P_{p-g})$ 로 나누면 처음 $g$개의 주성분은 $Z_g=XP_g$ 로 주어진다. 여기서

* $P_g$는 처음 $g$개의 고유벡터 (eigenvectors)로 이루어진 행렬이다.
* $Z_g=(z_1,z_2,\dots,z_g)$는 처음 $g$개의 주성분점수 (principal component scores)를 열로 갖는 행렬이다.

원래의 선형회귀모형 $y=X\beta+\varepsilon$ 에 대하여 직교변환을 이용하면

$$
y=XPP^T\beta+\varepsilon=Za+\varepsilon
$$
와 같이 표현할 수 있다. 여기서 $a$는 주성분 좌표계에서의 회귀계수벡터이다.

이제 $g$개의 주성분만을 사용하는 회귀모형 $y=Z_ga+\varepsilon$ 을 가정한다. 즉, 설명변수 전체가 아니라 $X$의 변동을 가장 잘 설명하는 상위 $g$개의 방향만을 선택하여 회귀하는 것이다.

### 15.2.2 저차원 자료 $(n>p)$에서의 주성분회귀추정량 (PCR Estimator for Low-Dimensional Data)

$n>p$인 경우, 주성분회귀모형
$y=Z_ga+\varepsilon$ 에서 $a$에 대한 최소제곱추정량은

$$
\hat{a}_g=(Z_g^TZ_g)^{-1}Z_g^Ty
$$

이를 특잇값 분해로 정리하면

$$
\hat{a}_g
= (P_g^TX^TXP_g)^{-1}P_g^TX^Ty
$$

이고, 다시 전개하면

$$
\hat{a}_g
=(D_g^TD_g)^{-1}D_g^TU^Ty
$$

가 된다. 여기서 $D_g$는 $D$ 행렬에서 왼쪽 앞의 $g$개 열만 남기고 나머지 $p-g$개 열은 제거한 행렬이다.

따라서 $g$개의 주성분을 사용한 회귀추정량은

$$\hat{\beta}_g=P_g\hat{a}_g$$

이므로

$$
\hat{\beta}_g = P_g(D_g^TD_g)^{-1}D_g^TU^Ty \tag{17.8}
$$

이 식을 최소제곱추정량
$$\hat{\beta}=P(D^TD)^{-1}D^TU^Ty$$

와 비교하면, 주성분회귀는 전체 $p$개의 방향을 사용하는 대신 그 중 상위 $g$개 방향만을 사용한다는 차이가 있다. 즉, 고유값이 큰 방향만을 남기고 작은 방향은 완전히 제거하는 방법이다.

특히 $g=p$이면 모든 주성분을 사용하게 되므로
$$\hat{\beta}_g=\hat{\beta}$$

가 되어 주성분회귀추정량은 최소제곱추정량과 같아진다.

#### 증명: 주성분회귀추정량의 유도 (Derivation of PCR Estimator)

$Z_g=XP_g$이므로

$$Z_g^TZ_g=P_g^TX^TXP_g,\qquad Z_g^Ty=P_g^TX^Ty$$

이다. 따라서
$$
\hat{a}_g=(P_g^TX^TXP_g)^{-1}P_g^TX^Ty
$$

이제 특잇값 분해 $X=UDP^T$ 를 대입하면 $X^TX=PD^TDP^T$ 이고, $P_g$는 $P$의 앞쪽 $g$개 열벡터로 이루어지므로

$$
P_g^TX^TXP_g=D_g^TD_g
$$

가 된다. 또한

$$P_g^TX^Ty=D_g^TU^Ty$$

이므로

$$\hat{a}_g=(D_g^TD_g)^{-1}D_g^TU^Ty$$

이고,

$$\hat{\beta}_g=P_g\hat{a}_g=P_g(D_g^TD_g)^{-1}D_g^TU^Ty$$

를 얻는다.

### 15.2.3 고차원 자료 $(p>n)$에서의 주성분회귀 (PCR for High-Dimensional Data)

이제 $p>n$인 고차원 자료를 생각한다. 이 경우 설명변수 행렬 $X$의 계수(rank)는 최대 $n$이고, 이에 따라 특잇값 분해는 0이 아닌 특잇값만 남긴 형태로 간단히 쓸 수 있다.

$$X=UDP^T=\sum_{l=1}^n d_lu_lp_l^T \tag{17.9}$$

* $U^TU=U^TU=P^TP=I_n$이다.
* $PP^T\neq I_p$이다.
* 대각행렬 $D$에서 0이 아닌 대각원소는 최대 $n$개이다.

이때 $X^TX$의 고유값 분해는

$$ X^TX=P\Lambda P^T =
(P_g,\;P_s)
\begin{pmatrix}
\Lambda_g & 0\\
0 & \Lambda_s
\end{pmatrix}
\binom{P_g^T}{P_s^T}
\tag{17.10}$$

로 표현된다. 여기서 $\Lambda=D^2$이며, 0이 아닌 고유값은 최대 $n$개이다. 또한 $s=n-g$가 된다.

따라서 고차원 자료에 대한 주성분회귀추정량은

$$ \hat{\beta}_g=P_g\Lambda_g^{-1}P_g^TX^Ty \tag{17.11} $$

가 된다. 여기서 $g<n$이다.  
즉, 저차원 자료에서는 주성분 개수 $g$가 $p$보다 작을 수 있었고, 고차원 자료에서는 주성분 개수 $g$가 $n$보다 작을 수 있을 뿐이라는 차이만 있을 뿐, 추정의 핵심 구조는 동일하다.

### 15.2.4 주성분회귀추정량의 기대값 (Expectation of PCR Estimator)

식 (17.11)의 기대값을 계산하면
$$ E(\hat{\beta}_g) = P_g\Lambda_g^{-1}P_g^TX^TX\beta$$

이고, 이를 전개하면

$$ E(\hat{\beta}_g) =\left(\sum_{j=1}^g p_jp_j^T\right)\beta $$

가 된다. 따라서

$$ E(\hat{\beta}_g) = \beta-\sum_{j=g+1}^{n}(p_j^T\beta)p_j \tag{17.12} $$

이 식은 주성분회귀가 불편추정량 (unbiased estimator)이 아님을 보여 준다. 즉, 상위 $g$개 주성분 방향만 남기고 나머지 방향을 제거하므로, 제거된 방향에 투영된 $\beta$ 성분만큼 편의가 발생한다.

#### 증명: 기대값의 계산 (Derivation of Expectation)

선형모형

$$y=X\beta+\varepsilon,\qquad E(\varepsilon)=0$$

에서

$$E(\hat{\beta}_g)=P_g\Lambda_g^{-1}P_g^TX^TX\beta$$

이다. 그런데

$$X^TX=\sum_{k=1}^{n}\lambda_kp_kp_k^T$$

이므로
$$ E(\hat{\beta}_g) =
\left(\sum_{j=1}^g\frac{1}{\lambda_j}p_jp_j^T\right)
\left(\sum_{k=1}^{n}\lambda_kp_kp_k^T\right)\beta
$$

이고, 직교성 때문에

$$E(\hat{\beta}_g)=\left(\sum_{j=1}^g p_jp_j^T\right)\beta$$

가 된다. 마지막으로 전체 직교기저를 기준으로 분해하면

$$\beta=\sum_{j=1}^{n}(p_j^T\beta)p_j+\sum_{j=n+1}^{p}(p_j^T\beta)p_j$$

이지만 주성분회귀는 앞의 $g$개 방향만 남기므로

$$E(\hat{\beta}_g)=\beta-\sum_{j=g+1}^{n}(p_j^T\beta)p_j$$

를 얻는다.

### 15.2.5 주성분회귀추정량의 분산 (Variance of PCR Estimator)

추정량의 분산을 계산하면

$$ \mathrm{Var}(\hat{\beta}_g) = \sigma^2P_g\Lambda_g^{-1}P_g^TX^TXP_g\Lambda_g^{-1}P_g^T$$

이고, 이를 정리하면
$$
\mathrm{Var}(\hat{\beta}_g)
= \sigma^2\left(\sum_{j=1}^{g}\frac{1}{\lambda_j}p_jp_j^T\right)
\tag{17.13}\tag{17.14}
$$

이 식은 상위 $g$개 주성분 방향에 대해서만 분산이 존재하고, 제거된 나머지 방향에 대해서는 더 이상 분산 기여가 없음을 뜻한다. 즉, 작은 고유값에 해당하는 불안정한 방향을 제거함으로써 분산을 감소시키는 효과가 생긴다.

### 15.2.6 평균제곱오차 (Mean Squared Error)와 해석 (Interpretation)

주성분회귀추정량의 평균제곱오차는
$$
\mathrm{MSE}(\hat{\beta}_g)
= E\big[(\hat{\beta}_g-\beta)^T(\hat{\beta}_g-\beta)\big]
$$

이며, 식 (17.12)와 식 (17.14)를 이용하면
$$ \mathrm{MSE}(\hat{\beta}_g) = \sigma^2\sum_{j=1}^{g}\lambda_j^{-1} + \sum_{j=g+1}^{n}(p_j^T\beta)^2 \tag{17.15}
$$
가 된다.

이 식은 주성분회귀의 편의-분산 절충 (bias-variance trade-off)을 명확히 보여 준다.

* $g$가 커지면 더 많은 주성분을 포함하므로 편의는 감소한다.
* 그러나 더 많은 방향이 추정에 포함되므로 분산은 증가한다.
* $g$가 작아지면 분산은 감소하지만 제거된 방향만큼 편의가 커진다.

따라서 적절한 $g$를 선택하여 전체 MSE를 최소화하는 것이 중요하다.

### 15.2.7 주성분회귀의 해석과 한계 (Interpretation and Limitation of PCR)

주성분회귀는 설명변수들에 대한 선형결합으로 설명변수들의 전체 변동을 가장 잘 설명하는 방향을 먼저 찾는다. 이때 주성분분석 (principal component analysis, PCA)의 목적은 기본적으로 차원축소이다. 따라서 주성분은 반응변수 $y$를 고려하지 않고 오직 설명변수 행렬 $X$만으로 결정된다.

이 점에서 주성분회귀는 다음과 같은 특징을 가진다.

1. 설명변수의 전체 변동을 잘 설명하는 주성분을 사용한다.
2. 그러나 반응변수와의 관련성이 큰 방향을 반드시 먼저 선택하는 것은 아니다.
3. 따라서 설명변수의 변동을 설명하는 데는 좋지만, 반응변수 예측에 가장 유리한 방향을 선택한다는 보장은 없다.

즉, 주성분회귀는 반응변수를 “예측”하는 데보다 반응변수를 “설명”하는 데 상대적으로 더 적합한 모형으로 이해할 수 있다. 주성분을 해석 가능한 방향으로 이해할 수 있다면, 원래 변수들보다 오히려 더 간결하게 반응변수와의 관계를 설명할 수 있다.


## 15.3 부분최소제곱회귀 (Partial Least Squares Regression)

부분최소제곱회귀 (partial least squares regression, PLS)는 주성분회귀와 함께 대표적인 차원축소방법이다. 두 방법은 모두 설명변수들을 새로운 저차원 축으로 변환한다는 공통점을 가지지만, 주성분의 선택 원리가 다르다.

주성분회귀에서는 설명변수들만을 이용하여 전체 변동이 큰 순서대로 주성분을 결정한다. 따라서 낮은 순위의 주성분이 오히려 반응변수 예측에 더 중요할 수도 있다는 한계가 있다. 반면 부분최소제곱회귀에서는 주성분과 유사한 잠재성분 (latent components)을 결정할 때 반응변수와의 연관성까지 함께 고려한다. 따라서 같은 개수의 성분을 사용하더라도 주성분회귀보다 부분최소제곱회귀가 더 높은 예측력을 보이는 경향이 있다.

또한 주성분회귀에서는 주성분들이 $X^TX$의 고유값 분해 또는 $X$의 특잇값 분해를 통해 한 번에 결정된다. 그러나 부분최소제곱회귀에서는 첫 번째 성분이 결정된 후, 그 정보를 반영한 상태에서 두 번째 성분이 결정되는 식으로 성분이 순차적 (sequential)으로 구성된다. 이러한 순차적 구조 때문에 부분최소제곱회귀는 최소제곱추정량이나 주성분회귀추정량보다 더 유연한 장점을 가진다.

### 15.3.1 부분최소제곱회귀의 기본 아이디어 (Basic Idea of PLS)

부분최소제곱회귀는 설명변수의 선형결합을 이용하여 새로운 축을 만들되, 그 축이 설명변수의 변동을 설명하는 동시에 반응변수와도 강하게 관련되도록 만든다. 즉, 단순히 $X$의 분산이 큰 방향만을 찾는 것이 아니라, $y$를 잘 설명하는 방향을 함께 고려한다.

이러한 특징으로 인해 부분최소제곱회귀는 다음과 같은 상황에서 특히 유용하다.

* 설명변수의 수가 많을 때
* 설명변수들 사이에 상관관계가 클 때
* 반응변수 예측이 주된 목적일 때
* 고차원 자료 ($p>n$)를 다룰 때

부분최소제곱회귀는 $n>p$라는 조건을 특별히 요구하지 않으므로, 고차원 자료에도 쉽게 적용할 수 있다.

### 15.3.2 주성분회귀와의 차이 (Difference from PCR)

부분최소제곱회귀와 주성분회귀의 차이는 크게 두 가지이다.

첫째, 성분 선택 기준이 다르다.

* 주성분회귀는 설명변수들의 전체 변동을 가장 크게 설명하는 방향을 선택한다.
* 부분최소제곱회귀는 반응변수와의 관련성을 함께 고려하여 성분을 선택한다.

둘째, 성분 결정 방식이 다르다.

* 주성분회귀는 $X^TX$의 고유값 분해를 통해 주성분을 한 번에 결정한다.
* 부분최소제곱회귀는 첫 번째 성분을 정한 뒤, 그 다음 성분을 순차적으로 구성한다.

이 때문에 부분최소제곱회귀는 예측문제에서 더 직접적인 목적 적합성 (goal alignment)을 가진다.

### 15.3.3 왜 부분최소제곱회귀가 필요한가 (Why PLS is Needed)

주성분회귀에서는 $g(<p)$개의 주성분을 사용하여 설명변수들의 변동 대부분을 설명하는 선형조합을 먼저 찾는다. 그러나 만약 예측변수의 모든 선형조합, 즉 $p$개의 주성분을 모두 사용해야만 반응변수를 충분히 설명할 수 있다면, 차원축소가 이루어졌다고 보기 어렵다. 또한 주성분은 반응변수를 고려하지 않고 선택되므로, 선택된 주성분이 실제로 반응변수를 잘 예측한다는 보장이 없다.

이에 비해 부분최소제곱회귀는 반응변수를 가장 잘 설명하는 설명변수의 선형조합을 찾는 데 초점을 둔다. 따라서 고려해야 할 설명변수가 많은 고차원 자료에서 보다 효과적으로 사용될 수 있다.

### 15.3.4 해석상의 차이 (Difference in Interpretation)

주성분회귀는 반응변수보다 설명변수 구조를 먼저 요약하므로, 주성분을 적절히 해석할 수 있다면 반응변수와의 관계를 간단히 설명하는 데 유리하다. 다시 말해 주성분회귀는 반응변수를 예측하기보다 반응변수를 설명하는 데 상대적으로 더 적합하다.

반대로 부분최소제곱회귀는 반응변수를 가장 잘 설명하는 방향을 찾는 데 중점을 둔다. 따라서 설명변수가 매우 많고 예측이 핵심 목표인 경우에 더 효과적으로 활용된다. 다만 주성분회귀와 달리 부분최소제곱회귀는 반응변수 값 자체를 직접 설명하는 해석 도구로 사용되기보다는, 예측을 위한 차원축소 도구로 이해하는 것이 적절하다.

### 15.3.5 부분최소제곱회귀의 장단점 정리 (Summary of Strengths and Weaknesses)

부분최소제곱회귀의 장점은 다음과 같다.

1. 반응변수와의 연관성을 반영하여 성분을 구성한다.
2. 예측력이 주성분회귀보다 더 높게 나타나는 경우가 많다.
3. 고차원 자료에서도 쉽게 적용할 수 있다.
4. $n>p$ 조건이 필요하지 않다.

반면 주의할 점은 다음과 같다.

1. 성분이 순차적으로 정의되므로 구조가 주성분회귀보다 덜 단순하다.
2. 설명변수의 전체 변동을 정리하는 해석력은 주성분회귀보다 약할 수 있다.
3. 주된 목적은 설명보다는 예측이다.
