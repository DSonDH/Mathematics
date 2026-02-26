# Chapter 5 추정과 가설검정 I (Estimation and Hypothesis Testing I)
우리가 다루는 중회귀모형은
$$y_i = \beta_0 + \sum_{j=1}^p \beta_j x_{ij} + \varepsilon_i$$

행렬형:
$$\mathbf{y} = \mathbf{X}\beta + \mathbf{\varepsilon}$$
* $\mathbf{y}$: $n \times 1$ 반응벡터
* $\mathbf{X}$: $n \times (p+1)$ 설계행렬(design matrix)
  - rank($\mathbf{X}$) = $p+1$ (full rank)
  - $\mathbf{X}^T\mathbf{X}$는 가역행렬(invertible matrix, 정칙행렬, non-singular matrix)
* $\beta$: $(p+1) \times 1$ 모수벡터
* $\mathbf{\varepsilon}$: 오차벡터

가정:
1. $E(\mathbf{\varepsilon})=0$
2. $Var(\mathbf{\varepsilon})=\sigma^2 I_n$
3. (정규성 가정) $\mathbf{\varepsilon} \sim N(0,\sigma^2 I_n)$

위와 같은 성질을 가진 모형을 완전계수의 중선형회귀모형(multiple linear regression model of full rank)이라고 하며, 간단히 중회귀모형하면 이 모형을 의미한다.

이번 챕터에서는 중회귀모형의 모수 $\beta$와 $\sigma^2$에 대한 점추정(point estimation), 구간추정(interval estimation), 가설검정(hypothesis testing)을 다룬다.

## 5.1 점추정 (Point Estimation)

### 5.1.1 오차의 정규성 가정 하의 최대가능도추정 (Maximum Likelihood Estimation)
정규성 가정: $\mathbf{\varepsilon} \sim N(0,\sigma^2 I)$
모수 벡터 $\theta = (\beta^T, \sigma^2)^T$에 대한 추정에 있어서 최대가능도법을 사용해보자.

가능도함수(likelihood function):
$$f(\beta,\sigma^2) = (2\pi\sigma^2)^{-n/2} \exp\left[-\frac{(\mathbf{y}-\mathbf{X}\beta)^T(\mathbf{y}-\mathbf{X}\beta)}{2\sigma^2}\right]$$
로그가능도(log-likelihood)를 $\beta$, $\sigma^2$에 대해 미분하면
$$\frac{\partial \log f}{\partial \beta} = \frac{1}{\sigma^2} \mathbf{X}^T(\mathbf{y}-\mathbf{X}\beta) \\
\frac{\partial \log f}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4} (\mathbf{y}-\mathbf{X}\beta)^T(\mathbf{y}-\mathbf{X}\beta)$$

$\tilde{\beta}$, $\tilde{\sigma}^2$가 최대가능도추정량이라면
$$\mathbf{X}^T(\mathbf{y}-\mathbf{X}\tilde{\beta})=0 \\
\tilde{\sigma}^2 = \frac{1}{n} (\mathbf{y}-\mathbf{X}\tilde{\beta})^T(\mathbf{y}-\mathbf{X}\tilde{\beta})$$

#### (1) $\beta$의 MLE
정규방정식(normal equations): $\mathbf{X}^T\mathbf{X}\tilde{\beta}=\mathbf{X}^T\mathbf{y}$ 이므로
$$\tilde{\beta}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$
이는 최소제곱추정량(OLS estimator)과 동일하고, 불편추정량(unbiased estimator)이다.

#### (2) $\sigma^2$의 MLE
$$\tilde{\sigma}^2 = \frac{1}{n} (\mathbf{y}-\mathbf{X}\tilde{\beta})^T(\mathbf{y}-\mathbf{X}\tilde{\beta}) = \frac{SSE}{n}$$
그러나 $E(SSE)=(n-p-1)\sigma^2$ 이므로
$$E(\tilde{\sigma}^2) = \frac{n-p-1}{n}\sigma^2$$
즉, 편의(biased) 추정량이다.  
불편추정량이 되기위해서는, $\hat{\sigma}^2 = \frac{SSE}{n-p-1} = MSE$를 사용해야한다.

### 5.1.2 오차의 정규성 가정이 없는 경우
정규성가정이 없으면, 최대가능도법을 사용할 수 없다. 하지만 최소제곱법(least squares)은 사용 가능하다.
$$\hat{\beta}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$
은 여전히 불편추정량인데, 이를 이용한 $\sigma^2$의 추정량은 
$$\hat{\sigma}^2 = \frac{1}{n-p-1} (\mathbf{y}-\mathbf{X}\hat{\beta})^T(\mathbf{y}-\mathbf{X}\hat{\beta})$$
로 정의할 수 있다. 이는 $\sigma^2$의 불편추정량이다.

### 5.1.3 가우스–마르코프 정리 (Gauss–Markov Theorem)
최소제곱추정량 $\hat{\beta}$의 특수한 성질인 최적성(optimality)을 보이는 정리이다.

가정:
* $E(\mathbf{\varepsilon})=0$
* $Var(\mathbf{\varepsilon})=\sigma^2 I$
* $\mathbf{X}$는 full rank

이면
$$\hat{\beta}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$
는
**최소분산 선형 불편추정량(Best Linear Unbiased Estimator, BLUE)** 즉, 모든 선형(linear)이고 불편(unbiased)인 추정량 중에서 분산이 최소이다 (minimum variance linear unbiased estimator, MVLUE).

#### 증명
$\beta$의 임의의 선형 불편추정량을 $\tilde{\beta}=A\mathbf{y}$라 하자. 여기서 $A$는 $(p+1) \times n$ 행렬이다.  
$\tilde{\beta}$가 불편추정량이면
$$E(\tilde{\beta})=AE(\mathbf{y})=A\mathbf{X}\beta=\beta\\
\therefore A\mathbf{X}=I$$
$\hat{\beta}$와의 차이를 고려하면
$$\tilde{\beta}-\hat{\beta}=A\mathbf{y}-(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}\\
=[A-(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T]\mathbf{y}$$

$B=A-(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$라 정의하면
$$B\mathbf{X}=A\mathbf{X}-I=0$$

따라서
$$Var(\tilde{\beta})=Var(\hat{\beta}+B\mathbf{y})\\
=Var(\hat{\beta})+Var(B\mathbf{y})+2Cov(\hat{\beta},B\mathbf{y})\\
=\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}+\sigma^2 BB^T+2\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^TB^T$$

$(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^TB^T=0$ (왜냐하면 $B\mathbf{X}=0$이므로)이고,

$$Var(\tilde{\beta})=\sigma^2[(\mathbf{X}^T\mathbf{X})^{-1}+BB^T]$$

$BB^T$는 비음정치(non-negative definite) 행렬이므로
$$Var(\tilde{\beta})-Var(\hat{\beta})=\sigma^2 BB^T \geq 0$$

등호는 $B=0$, 즉 $A=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$일 때만 성립한다.


## 5.2 구간추정 (Interval Estimation)
중회귀모형에서 $E(y|x)=\beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p$의 형태로 평균반응(mean response)을 추정할 수 있다. 또한, 개별 관측값 $y$에 대한 예측(prediction)도 가능하다. 이때, 평균반응과 개별 관측값에 대한 구간추정(interval estimation)을 다룬다.

### 5.2.1 평균반응의 구간추정 (Confidence Interval for Mean Response)
$y$의 평균반응 $E(y|x)$에 대한 구간추정을 고려하자. 이 점추정량은 $\hat{y} = \mathbb{x}^T \mathbb{\hat{\beta}}$이다. $\hat{y}=\mathbf{x}^T\hat{\beta}$의 분산:
$$Var(\hat{y}) = Var(\mathbf{x}^T\hat{\beta}) = \mathbf{x}^T Var(\hat{\beta}) \mathbf{x} = \mathbf{x}^T (\sigma^2 (\mathbf{X}^T\mathbf{X})^{-1}) \mathbf{x} = \sigma^2 \mathbf{x}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}$$

**(1) $\sigma^2$가 알려진 경우**  
$$\hat{y} \pm z_{\alpha/2} \sqrt{\mathbf{x}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x} \sigma^2}$$

**(2) $\sigma^2$가 미지인 경우**  
MSE로 추정하여 대입 
$$\hat{y} \pm t_{\alpha/2}(n-p-1) \sqrt{\mathbf{x}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x} MSE}$$

### 5.2.2 개별관측값 예측구간 (Prediction Interval)
새로운 관측값 $y_s$에 대해 $Var(y_s) = [1+\mathbf{x}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}]\sigma^2$ 이므로, $y_s$와 $\hat{y}$의 차이의 분산은
$$Var(y_s-\hat{y}) = \sigma^2\left[1 + \mathbf{x}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}\right]$$

예측구간:
$$\hat{y} \pm z_{\alpha/2} \sqrt{\left[1 + \mathbf{x}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}\right]\sigma^2} \\
\hat{y} \pm t_{\alpha/2}(n-p-1) \sqrt{\left[1 + \mathbf{x}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}\right]MSE}$$

### 5.2.3 회귀계수의 구간추정
$$Var(\hat{\beta}_j)=c_{jj}\sigma^2$$

**(1) $\sigma^2$가 알려진 경우**  
$$\hat{\beta}_j \pm z_{\alpha/2} \sqrt{c_{jj}\sigma^2}$$

**(2) $\sigma^2$가 미지인 경우**  
$$\hat{\beta}_j \pm t_{\alpha/2}(n-p-1) \sqrt{c_{jj}MSE}$$

### 5.2.4 선형결합 $(q^T\beta)$의 구간추정
임의 벡터 $q$에 대해 $q^T\hat{\beta}$ 은 $q^T\beta$의 불편추정량이다.  
분산:
$$Var(q^T\hat{\beta}) = \sigma^2 q^T(\mathbf{X}^T\mathbf{X})^{-1}q$$

**신뢰구간:**  
$$q^T\hat{\beta} \pm t_{\alpha/2}(n-p-1) \sqrt{q^T(\mathbf{X}^T\mathbf{X})^{-1}q MSE}$$

예를들어, $\beta_1 - \beta_2$이면 $q^T=(0,1,-1,0,\dots,0)$, $q^T(\mathbf{X}^T\mathbf{X})^{-1}q = c_{11} + c_{22} - 2c_{12}$ 이다.


## 5.3 $(q^T\beta)$의 가설검정 (Hypothesis Testing)

### 5.3.1 평균반응에 대한 가설검정
주어진 $\mathbf{x}$에서 $H_0: E(y|\mathbf{x})=\eta$, $H_1: E(y|\mathbf{x}) \neq \eta$ 라는 가설을 검정하자.

**(1) 분산이 알려진 경우**  
$$Z_0=\frac{\hat y_0-\eta}{\sqrt{\mathbf{x}^T (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{x} \sigma^2}} \sim N(0,1)$$

**(2) 분산이 미지인 경우**  
$$t_0=\frac{\hat y_0-\eta}{\sqrt{\mathbf{x}^T (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{x} MSE}} \sim t(n-p-1)$$

자세한 검정절차:
1. 검정통계량의 관측값 $z_0$ 또는 $t_0$ 계산
2. 유의수준 $\alpha$를 정하고, 표준정규분포표와 t-분포표에 기각치 $z_{\alpha/2}$ 또는 $t_{\alpha/2}(n-p-1)$를 찾는다.
3. $\sigma^2$가 알려진 경우 $|z_0| > z_{\alpha/2}$이면 귀무가설 기각, 그렇지 않으면 채택한다. $\sigma^2$가 미지인 경우 $|t_0| > t_{\alpha/2}(n-p-1)$이면 귀무가설 기각, 그렇지 않으면 채택한다.

TODO:
## 5.4 가설 $(\mathbf{C}\mathbf{\beta}=\mathbf{m})$의 검정 (Test of Linear Hypothesis $(\mathbf{C}\mathbf{\beta}=\mathbf{m})$)
여러 개의 선형제약을 동시에 검정한다.
$$\mathbf{C}\mathbf{\beta} = \mathbf{m}$$

* $\mathbf{C}$: $k\times(p+1)$ 행렬
* $\mathbf{m}$: $k\times1$ 벡터
* $rank(\mathbf{C})=k$
  - k개의 선형제약이 서로 독립적임을 의미한다.

가설: $H_0:\mathbf{C}\mathbf{\beta}=\mathbf{m}, \quad H_1: \mathbf{C}\mathbf{\beta} \neq \mathbf{m}$

예시: $H_0: \beta_1 = \beta_2 = 0$ 라는 가설은 $\mathbf{C}=\begin{bmatrix}0 & 1 & 0 & \cdots & 0 \\ 0 & 0 & 1 & \cdots & 0\end{bmatrix}$, $\mathbf{m}=\begin{bmatrix}0 \\ 0\end{bmatrix}$로 표현할 수 있다.

예시2: $H_0: \beta_1-\beta_2 = 0, \beta_3-2\beta_4 = 0$ 라는 가설은 $\mathbf{C}=\begin{bmatrix}0 & 1 & -1 & 0 & \cdots & 0 \\ 0 & 0 & 0 & 1 & -2 & \cdots & 0\end{bmatrix}$, $\mathbf{m}=\begin{bmatrix}0 \\ 0\end{bmatrix}$로 표현할 수 있다.

### 5.4.1 방법 I: 제한최소제곱법 (Restricted Least Squares via Lagrange Multipliers)
제한조건 $\mathbf{C}\mathbf{\beta}=\mathbf{m}$을 만족하는 $\beta$ 중에서 잔차제곱합이 최소가 되는 $\beta$를 구한다.

문제:
$$\min_\beta (\mathbf{y}-\mathbf{X}\beta)^T(\mathbf{y}-\mathbf{X}\beta) \quad \text{s.t. } \mathbf{C}\mathbf{\beta}=\mathbf{m}$$

라그랑지안:
$$L=(\mathbf{y}-\mathbf{X}\beta)^T(\mathbf{y}-\mathbf{X}\beta)+2\theta^T(\mathbf{C}\mathbf{\beta}-\mathbf{m})$$

정리하면 제한추정량:
$$\tilde\beta = \hat\beta - (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}(\mathbf{C}\hat\beta-\mathbf{m})$$

**SSE 증가량**  
$$Q = (\mathbf{C}\hat\beta-\mathbf{m})^T[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}(\mathbf{C}\hat\beta-\mathbf{m})$$

**분포**  
$$\frac{Q}{\sigma^2} \sim \chi^2(k,\lambda)$$
귀무가설 하에서 $\lambda=0$.

또한
$$\frac{SSE}{\sigma^2} \sim \chi^2(n-p-1)$$
이며 서로 독립이다.

**F-통계량**  
$$F_0 = \frac{Q/k}{MSE} \sim F(k,n-p-1)$$

기각규칙:
$$F_0 > F_\alpha(k,n-p-1)$$

### 5.4.2 방법 II: 축소모형 접근 (Reduced Model Approach)
제한조건 $\mathbf{C}\mathbf{\beta}=\mathbf{m}$을 만족하는 $\beta$로 모수를 재정의하여 축소모형을 만든다.

귀무가설을 반영하여 모수를 재정의한 축소모형을 만든다.
완전모형(full model)과 축소모형(reduced model)의 잔차제곱합 비교:
$$F_0 = \frac{SSE(R)-SSE(F)}{k} \Big/ MSE(F)$$
이는 방법 I의 통계량과 동일하다.

### 절편 없는 경우
절편이 없는 모형:
$$\mathbf{y}=\mathbf{X}\beta+\mathbf{\varepsilon}$$
이면 자유도는 $n-p$가 된다.
$$F_0 \sim F(k,n-p)$$


## 5.5 적합결여검정 (Lack-of-Fit Test)
모형이 실제 평균구조를 충분히 설명하는지 검정한다.

**잔차 분해**  
잔차:
$$e_i=y_i-\hat y_i$$

이를
$$e_i=\underbrace{y_i-E(y_i|x_i)}_{\text{순오차 (pure error)}}+\underbrace{E(y_i|x_i)-\hat y_i}_{\text{적합결여오차 (lack-of-fit error)}}$$
로 분해할 수 있다.

**반복관측이 있을 때**  
같은 $x$값에서 반복측정이 존재하면
총잔차제곱합을
$$SSE = SS_{PE}+SS_{LOF}$$
로 분해 가능하다.

* $SS_{PE}$: 순오차제곱합
* $SS_{LOF}$: 적합결여제곱합

**검정통계량**  
$$F_0 = \frac{SS_{LOF}/(g-p-1)}{SS_{PE}/(n-g)} \sim F(g-p-1,n-g)$$
* $g$: 서로 다른 설계점의 수

**해석**  
* $F_0$가 크면 모형의 구조적 부적합 존재
* 작으면 모형 적합성 유지


## 5.5 적합결여검정 (Lack-of-Fit Test)
중회귀모형이 자료의 평균구조를 충분히 설명하는지를 검정하는 절차이다.
핵심 아이디어는 잔차제곱합을
* 순오차(pure error)
* 적합결여오차(lack-of-fit error)
로 분해하는 것이다.

### 5.5.1 잔차의 분해
모형:
$$y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip} + \varepsilon_i$$

적합값:
$$\hat y_i = x_i^T \hat\beta$$

잔차:
$$e_i = y_i - \hat y_i$$

이를 다음과 같이 분해할 수 있다.
$$e_i=\underbrace{y_i - E(y_i|x_i)}_{\text{순오차}}+\underbrace{E(y_i|x_i) - \hat y_i}_{\text{적합결여오차}}$$
* 첫 번째 항: 실험적 변동
* 두 번째 항: 모형의 구조적 부적합

### 5.5.2 반복관측이 있는 경우
같은 설계점 $x_i$에서 $n_i$번 반복측정이 존재한다고 하자.
$$\bar y_i = \frac{1}{n_i}\sum_{j=1}^{n_i} y_{ij}$$
총 잔차제곱합:
$$SSE = \sum_{i=1}^k \sum_{j=1}^{n_i} (y_{ij}-\hat y_{ij})^2$$
이를 다음과 같이 분해한다.
$$SSE = SSPE + SSLF$$

**(1) 순오차제곱합**  
$$SSPE = \sum_{i=1}^k \sum_{j=1}^{n_i} (y_{ij}-\bar y_i)^2$$

자유도:
$$df_{PE} = \sum (n_i-1) = n-k$$

**(2) 적합결여제곱합**  
$$SSLF = \sum_{i=1}^k n_i(\bar y_i-\hat y_i)^2$$
자유도:
$$df_{LF} = (n-p-1)-(n-k) = k-p-1$$

### 5.5.3 F-검정
평균제곱:
$$MSPE = \frac{SSPE}{n-k}\\
MSLF = \frac{SSLF}{k-p-1}$$

검정통계량:
$$F_0 = \frac{MSLF}{MSPE} \sim F(k-p-1, n-k)$$

#### 해석
* $F_0$가 크면 → 모형 구조가 잘못되었을 가능성
* 작으면 → 현재 모형 유지 가능
즉, 적합결여오차가 순오차에 비해 유의하게 크면 모형 부적합이다.


## 5.6 잔차의 검토 (Residual Analysis)
잔차분석은 회귀가정의 타당성을 진단하는 절차이다.

### 5.6.1 잔차의 기본 성질
정규방정식:
$$X^TX\hat\beta = X^Ty$$
로부터 다음이 성립한다.

**(1) 잔차의 합**  
$$\sum e_i = 0$$

**(2) 설명변수와의 직교성**  
$$\sum x_{ij} e_i = 0, \quad j=1,\dots,p$$
즉, 잔차는 설계행렬 $X$의 열공간과 직교한다.

**(3) 적합값과도 직교**  
$$\sum \hat y_i e_i = 0$$

### 5.6.2 잔차의 분산–공분산 구조
$$e = y-\hat y = [I - X(X^TX)^{-1}X^T]y$$
따라서
$$E(e)=0\\
Var(e)=\sigma^2[I - X(X^TX)^{-1}X^T]$$
이는 일반적으로 대각행렬이 아니므로 잔차들 사이에는 상관이 존재한다.

잔차 상관계수:
$$\rho_{ij} = \frac{Cov(e_i,e_j)}{\sqrt{Var(e_i)Var(e_j)}}$$
이 값은 $\sigma^2$에 의존하지 않고 설계행렬 $X$에 의해 결정된다.

### 5.6.3 잔차 산점도 해석
잔차를
* $\hat y$에 대해
* 각 $x_j$에 대해
* 시간에 대해 (시계열의 경우)
그려서 모형가정을 점검한다.

**(a) 무작위 분포**  
→ 가정 위반 없음

**(b) 부채꼴 모양**  
→ 이분산성 (heteroscedasticity)

**(c) 선형 패턴**  
→ 절편 누락 가능

**(d) 곡선 패턴**  
→ 비선형항(제곱항 등) 필요
