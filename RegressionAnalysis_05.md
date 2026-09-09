# Chapter 5 추정과 가설검정 I (Estimation and Hypothesis Testing I)
우리가 다루는 중회귀모형은 $y_i = \beta_0 + \sum_{j=1}^p \beta_j x_{ij} + \varepsilon_i$ 또는 $\mathbf{y} = \mathbf{X}\beta + \mathbf{\varepsilon}$ 형태이다.

* $\mathbf{y}$: $n \times 1$ 반응벡터
* $\mathbf{X}$: $n \times (p+1)$ 설계행렬(design matrix)
  - rank($\mathbf{X}$) = $p+1$ (full rank)M
  - $\mathbf{X}^T\mathbf{X}$는 가역행렬(invertible matrix, 정칙행렬, non-singular matrix)
* $\beta$: $(p+1) \times 1$ 모수벡터
* $\mathbf{\varepsilon}$: 오차벡터

위와 같은 성질을 가진 모형을 완전계수의 중선형회귀모형(multiple linear regression model of full rank)이라고 하며, 간단히 중회귀모형하면 이 모형을 의미한다.

이번 챕터에서는 중회귀모형의 모수 $\beta$와 $\sigma^2$에 대한 점추정(point estimation), 구간추정(interval estimation), 가설검정(hypothesis testing)을 다룬다.

분포 가정: $\mathbf{\varepsilon} \sim N(0,\sigma^2 I)$ 또는 평균이 0이고 공분산이 $\sigma^2 I$인 임의의분포를 따른다고 가정할 수 있다.  
점추정은 두가지 가정을 모두 살펴보고, 구간추정과 가설검정은 전자의 분포 가정만 다룬다.

## 5.1 점추정 (Point Estimation)

### 5.1.1 오차의 정규성 가정 하의 최대가능도추정 (Maximum Likelihood Estimation)

$\mathbf{\varepsilon} \sim N(0,\sigma^2 I)$ 가정 하에, 모수 벡터 $\theta = (\beta^T, \sigma^2)^T$에 대한 추정에 있어서 최대가능도법을 사용해보자.

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

### 정리 5.1 가우스–마르코프 정리 (Gauss–Markov Theorem)
최소제곱추정량 $\hat{\beta}$의 특수한 성질인 최적성(optimality)을 보이는 정리이다.

중회귀 모형에서, $E(\mathbf{\varepsilon})=0$ 이고 $Var(\mathbf{\varepsilon})=\sigma^2 I$ 이며 $\mathbf{X}$는 full rank를 가진다면,

$$\hat{\beta}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

는 **최소분산 선형 불편추정량(Best Linear Unbiased Estimator, BLUE)** 이다.  
즉, 모든 선형(linear)이고 불편(unbiased)인 추정량 중에서 분산이 최소이다 (minimum variance linear unbiased estimator, MVLUE).
  - 최량(bets)라는 표현을 써서, 최량 선형 불편추정량이라고도 함.

>**증명**
>
>먼저 $ C=(\mathbf X^T\mathbf X)^{-1}\mathbf X^T $ 라고 두면 $\hat\beta=C\mathbf y$이다.
>
>임의의 선형 불편추정량을 $ \tilde\beta=A\mathbf y $ 라고 하자. 불편성은 모든 $\beta$에 대해 성립해야 하므로 $ E(\tilde\beta) =A\mathbf X\beta =\beta $ 에서 $ A\mathbf X=I_k $ 를 얻는다.
>
>이제 $B=A-C$ 라고 하면 $B\mathbf X =A\mathbf X-C\mathbf X =I_k-I_k =0 $ 이므로 $B\mathbf X=0$이다.  
>따라서
>
>$$ \begin{aligned} \text{Var}(\tilde\beta) &=\text{Var}(A\mathbf y)\\ 
>&=A\text{Var}(\mathbf y)A^T\\ 
>&=\sigma^2AA^T\\ 
>&=\sigma^2(C+B)(C+B)^T\\ 
>&=\sigma^2 \left(CC^T+CB^T+BC^T+BB^T\right). \end{aligned} $$
>
>그런데 $CB^T =(\mathbf X^T\mathbf X)^{-1}\mathbf X^TB^T =(\mathbf X^T\mathbf X)^{-1}(B\mathbf X)^T =0$ 이고 $BC^T=(CB^T)^T=0$이다.  
>또한 $CC^T =(\mathbf X^T\mathbf X)^{-1} \mathbf X^T\mathbf X (\mathbf X^T\mathbf X)^{-1} =(\mathbf X^T\mathbf X)^{-1}$ 이므로
>
>$$ \text{Var}(\tilde\beta) = \sigma^2(\mathbf X^T\mathbf X)^{-1} +\sigma^2BB^T. $$
>
>한편 $ \text{Var}(\hat\beta) = \sigma^2(\mathbf X^T\mathbf X)^{-1} $ 이므로
>
>$$ \boxed{ \text{Var}(\tilde\beta) -\text{Var}(\hat\beta) =\sigma^2BB^T\succeq0 } $$
>
>이다. 따라서 $\hat\beta$는 BLUE이다.
>
>더 자세히는, 등호 $\text{Var}(\tilde\beta)=\text{Var}(\hat\beta)$ 가 성립한다고 하자. $\sigma^2>0$이므로 $\sigma^2BB^T=0 \Rightarrow BB^T=0$  
>$B$의 $i$번째 행을 $b_i^T$라고 하면 $BB^T$의 $i$번째 대각성분은 $(BB^T)_{ii}=b_i^Tb_i=\|b_i\|^2$ 이다. 따라서 $BB^T=0$이면 모든 $i$에 대해 $\|b_i\|^2=0$ 즉 $b_i=0$이다. 그러므로
>
>$$ B=0. $$
>
>따라서 $ A=C $ 이고, 모든 $\mathbf y$에 대하여
>
>$$ \tilde\beta=A\mathbf y =C\mathbf y =\hat\beta $$
>
>이다. 그러므로 $\hat\beta$는 유일한 BLUE이다.


## 5.2 구간추정 (Interval Estimation)
중회귀모형에서 $E(y|x)=\beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p$의 형태로 평균반응(mean response)을 추정할 수 있다. 또한, 개별 관측값 $y$에 대한 예측(prediction)도 가능하다. 이때, 평균반응과 개별 관측값에 대한 구간추정(interval estimation)을 다룬다.

### 5.2.1 평균반응의 구간추정 (Confidence Interval for Mean Response)
$y$의 평균반응 $E(y|x)$에 대한 구간추정을 고려하자. 이 점추정량 $\hat{y}=\mathbf{x}^T\hat{\beta}$의 분산:

$$Var(\hat{y}) = Var(\mathbf{x}^T\hat{\beta}) = \mathbf{x}^T Var(\hat{\beta}) \mathbf{x} = \mathbf{x}^T (\sigma^2 (\mathbf{X}^T\mathbf{X})^{-1}) \mathbf{x} = \sigma^2 \mathbf{x}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}$$

**(1) $\sigma^2$가 알려진 경우** 
 
$$\hat{y} \pm z_{\alpha/2} \sqrt{\mathbf{x}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x} \sigma^2}$$

**(2) $\sigma^2$가 미지인 경우**  
MSE로 추정하여 대입 

$$\hat{y} \pm t_{\alpha/2}(n-p-1) \sqrt{\mathbf{x}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x} MSE}$$

### 5.2.2 개별관측값 예측구간 (Prediction Interval)
새로운 관측값 $y_s$에 대해 $Var(y_s) = [1+\mathbf{x}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}]\sigma^2$ 이므로

예측구간:

$$\hat{y} \pm z_{\alpha/2} \sqrt{\left[1 + \mathbf{x}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}\right]\sigma^2} \\
\hat{y} \pm t_{\alpha/2}(n-p-1) \sqrt{\left[1 + \mathbf{x}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}\right]MSE}$$

### 5.2.3 회귀계수의 구간추정
$Var(\hat{\beta}_j)=c_{jj}\sigma^2$ 이고 $c_{jj}$는 $(\mathbf{X}^T\mathbf{X})^{-1}$의 $j+1$번째 대각성분이다. 따라서 회귀계수 $\beta_j$에 대한 구간추정은 다음과 같다.

**(1) $\sigma^2$가 알려진 경우** 
 
$$\hat{\beta}_j \pm z_{\alpha/2} \sqrt{c_{jj}\sigma^2}$$

**(2) $\sigma^2$가 미지인 경우** 
 
$$\hat{\beta}_j \pm t_{\alpha/2}(n-p-1) \sqrt{c_{jj}MSE}$$

### 5.2.4 선형결합 $(q^T\beta)$의 구간추정
임의 벡터 $q$에 대해 $q^T\hat{\beta}$ 은 $q^T\beta$의 불편추정량이다.  

분산:

$$Var(q^T\hat{\beta}) = \sigma^2 q^T(\mathbf{X}^T\mathbf{X})^{-1}q$$

**신뢰구간:** 

$\sigma^2$가 알려진 경우

$$q^T\hat{\beta} \pm z_{\alpha/2} \sqrt{q^T(\mathbf{X}^T\mathbf{X})^{-1}q \sigma^2}$$

$\sigma^2$가 알려지지 않은 경우

$$q^T\hat{\beta} \pm t_{\alpha/2}(n-p-1) \sqrt{q^T(\mathbf{X}^T\mathbf{X})^{-1}q MSE}$$

예를들어, $\beta_1 - \beta_2$의 신뢰구간을 구하고싶으면 $q^T=(0,1,-1,0,\dots,0)$ 이 되고, $q^T(\mathbf{X}^T\mathbf{X})^{-1}q = c_{11} + c_{22} - 2c_{12}$ 이다.


## 5.3 가설검정 (Hypothesis Testing)

### 5.3.1 평균반응에 대한 가설검정
주어진 $\mathbf{x}$에서 $H_0: E(y|\mathbf{x})=\eta$, 그리고 $H_1: E(y|\mathbf{x}) \neq \eta$ 라는 가설을 검정하자.

$E(y|\mathbf{x})$의 점추정량 $\hat{y}_0 = \mathbf{x}^T\hat{\beta}$ 이며 $\text{Var}(\hat{y}_0) = \sigma^2 \mathbf{x}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}$ 이므로 귀무가설하에서 검정통계량은 표준정규분포를 따른다.

**(1) 분산이 알려진 경우** 
 
$$Z_0
= \frac{\hat y_0 -E(y|\mathbf{x})}{\sqrt{\text{Var}(\hat{y}_0)}}
=\frac{\hat y_0-\eta}{\sqrt{\mathbf{x}^T (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{x} \sigma^2}} \sim N(0,1)$$

**(2) 분산이 미지인 경우** 
$\text{Var}(\hat{y}_0)$를 MSE로 추정하여 대입하면, 검정통계량은 t-분포를 따른다.

$$t_0=\frac{\hat y_0-\eta}{\sqrt{\mathbf{x}^T (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{x} MSE}} \sim t(n-p-1)$$

자세한 검정절차:
1. 검정통계량의 관측값 $z_0$ 또는 $t_0$ 계산
2. 유의수준 $\alpha$를 정하고, 표준정규분포표와 t-분포표에 기각치 $z_{\alpha/2}$ 또는 $t_{\alpha/2}(n-p-1)$를 찾는다.
3. $\sigma^2$가 알려진 경우 $|z_0| > z_{\alpha/2}$이면 귀무가설 기각, 그렇지 않으면 채택한다. $\sigma^2$가 미지인 경우 $|t_0| > t_{\alpha/2}(n-p-1)$이면 귀무가설 기각, 그렇지 않으면 채택한다.


## 5.4 가설 $(\mathbf{C}\mathbf{\beta}=\mathbf{m})$의 검정 (Test of Linear Hypothesis $(\mathbf{C}\mathbf{\beta}=\mathbf{m})$)
지금까지는 $\beta_j$들의 선형조합에 대한 가설검정을 보았는데, 이 절에서는 $k$개의 선형제약을 동시에 검정한다.

$$\mathbf{C}\mathbf{\beta} = \mathbf{m}$$

* $\mathbf{C}$: $k\times(p+1)$ 행렬
* $\mathbf{m}$: $k\times1$ 벡터
* 가정: $rank(\mathbf{C})=k$
  - k개의 선형제약이 서로 독립적임을 의미한다.

가설: 

$$H_0:\mathbf{C}\mathbf{\beta}=\mathbf{m}, \quad H_1: \mathbf{C}\mathbf{\beta} \neq \mathbf{m}$$

예시: $H_0: \beta_1 = \beta_2 = 0$ 라는 가설은 $\mathbf{C}=\begin{bmatrix}0 & 1 & 0 & \cdots & 0 \\ 0 & 0 & 1 & \cdots & 0\end{bmatrix}$, $\mathbf{m}=\begin{bmatrix}0 \\ 0\end{bmatrix}$로 표현할 수 있다.

예시2: $H_0: \beta_1-\beta_2 = 0, \beta_3-2\beta_4 = 0$ 라는 가설은 $\mathbf{C}=\begin{bmatrix}0 & 1 & -1 & 0 & \cdots & 0 \\ 0 & 0 & 0 & 1 & -2 & \cdots & 0\end{bmatrix}$, $\mathbf{m}=\begin{bmatrix}0 \\ 0\end{bmatrix}$로 표현할 수 있다.

검정하는 방법에는 크게 두 가지가 있다 (둘 다 매우 중요!)
- 제한최소제곱법 (Restricted Least Squares via Lagrange Multipliers)
- 축소모형 접근 (Reduced Model Approach)

### 5.4.1 방법 I: 제한최소제곱법 (Restricted Least Squares via Lagrange Multipliers)

제한조건 $\mathbf{C}\mathbf{\beta}=\mathbf{m}$을 만족하는 $\mathbf{\tilde\beta}$ 중에서 잔차제곱합이 최소가 되는 $\mathbf{\tilde\beta}$를 구한다.

$$\min_\beta (\mathbf{y}-\mathbf{X} \mathbf{\tilde\beta})^T(\mathbf{y}-\mathbf{X}\mathbf{\tilde\beta}) \quad \text{s.t. } \mathbf{C}\mathbf{\tilde\beta}=\mathbf{m}$$

라그랑지안 (Lagrange 배수법, Lagrange multipliers)를 이용하여 다음과 같이 문제를 풀 수 있다:

$$L=(\mathbf{y}-\mathbf{X}\mathbf{\tilde\beta})^T(\mathbf{y}-\mathbf{X}\mathbf{\tilde\beta})+2\mathbf{\theta}^T(\mathbf{C}\mathbf{\tilde\beta}-\mathbf{m})$$

- $\mathbf{\theta}$는 $k \times 1$ 벡터로, 라그랑지 배수다.
- 함수 $L$을 $\mathbf{\tilde\beta}$, $\mathbf{\theta}$에 대해 편미분하여 0으로 놓는다.

$$ 
\frac{\partial L}{\partial \mathbf{\tilde\beta}} = -2\mathbf{X}^T(\mathbf{y}-\mathbf{X}\mathbf{\tilde\beta}) + 2\mathbf{C}^T \mathbf{\theta} = 0 \\
\frac{\partial L}{\partial \mathbf{\theta}} = 2(\mathbf{C}\mathbf{\tilde\beta}-\mathbf{m}) = 0 \\
$$

$$
\therefore \mathbf{X}^T\mathbf{X}\mathbf{\tilde\beta} + \mathbf{C}^T \mathbf{\theta} = \mathbf{X}^T\mathbf{y} \\
\mathbf{C}\mathbf{\tilde\beta} = \mathbf{m}
$$

정리하면 

$C\mathbf{\tilde\beta} = C[\mathbf{\hat\beta}- (X^TX)^{-1}C^T\theta] = \mathbf{m}$ 이고, $\theta = [C(X^TX)^{-1}C^T]^{-1}(C\hat\beta - \mathbf{m})$ 이므로, 이를 $\tilde\beta$에 대입하면 제한추정량은:

$$\mathbf{\tilde\beta} = \mathbf{\hat\beta} - (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}(\mathbf{C}\mathbf{\hat\beta}-\mathbf{m})$$

**SSE 증가량**  
$\mathbf{\tilde\beta}$로 추정한 잔차제곱합 $SSE$와 $\mathbf{\hat\beta}$로 추정한 잔차제곱합 $SSE$의 차이를 구해보자.

$$SSE(\tilde\beta) = (\mathbf{y}-\mathbf{X}\mathbf{\tilde\beta})^T(\mathbf{y}-\mathbf{X}\mathbf{\tilde\beta}) \\
= [\mathbf{y}-\mathbf{X}\mathbf{\hat\beta} + (\mathbf{X}\mathbf{\hat\beta} - \mathbf{X}\mathbf{\tilde\beta})]^T[\mathbf{y}-\mathbf{X}\mathbf{\hat\beta} + (\mathbf{X}\mathbf{\hat\beta} - \mathbf{X}\mathbf{\tilde\beta})] \\
= (\mathbf{y}-\mathbf{X}\mathbf{\hat\beta})^T(\mathbf{y}-\mathbf{X}\mathbf{\hat\beta}) + 2(\mathbf{X}\mathbf{\hat\beta}-\mathbf{X}\mathbf{\tilde\beta})^T(\mathbf{y}-\mathbf{X}\mathbf{\hat\beta}) + (\mathbf{\hat\beta}-\mathbf{\tilde\beta})^T\mathbf{X}^T\mathbf{X}(\mathbf{\hat\beta}-\mathbf{\tilde\beta}) 
$$

여기서 첫번째 항은 $SSE$이다.  
그리고 제한조건없는 경우의 정규방정식에서 $\mathbf{X}^T(\mathbf{y}-\mathbf{X}\mathbf{\hat\beta})=0$ 이므로, 두번째 항은 0이 된다.

또한, 위에서 $\tilde\beta = \hat\beta - (X^TX)^{-1}C^T[C(X^TX)^{-1}C^T]^{-1}(C\hat\beta - m)$ 이므로, 세번째 항은

$$(\mathbf{\hat\beta}-\mathbf{\tilde\beta})^T\mathbf{X}^T\mathbf{X}(\mathbf{\hat\beta}-\mathbf{\tilde\beta}) = (\mathbf{C}\mathbf{\hat\beta}-\mathbf{m})^T[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}(\mathbf{C}\mathbf{\hat\beta}-\mathbf{m})$$

따라서 

$$ SSE(\tilde\beta) = SSE + (\mathbf{C}\mathbf{\hat\beta}-\mathbf{m})^T[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}(\mathbf{C}\mathbf{\hat\beta}-\mathbf{m}) $$

제한조건을 가하여 얻은것이 $SSE(\tilde\beta)$이고, 제한조건을 가하지 않은 것이 $SSE(\hat\beta)$이므로, $SSE$가 더 작거나 같다. 그 차이를 $Q$ 로 정의하면,

$$Q = SSE(\tilde\beta) - SSE(\hat\beta)
= (\mathbf{C}\mathbf{\hat\beta}-\mathbf{m})^T[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}(\mathbf{C}\mathbf{\hat\beta}-\mathbf{m})$$

이 $Q$에 대해 알아보자.

**$Q$의 분포**  
중회귀모형 $\mathbf{y}=\mathbf{X}\beta + \varepsilon$에서 $\varepsilon \sim N(0,\sigma^2 I)$이므로, $\mathbf{y}, \mathbf{\hat{\beta}}, \mathbf{C}\mathbf{\hat{\beta}}-\mathbf{m}$의 확률분포는

$$
\mathbf{y} \sim N(\mathbf{X}\beta, \sigma^2 I)\\
\mathbf{\hat{\beta}} \sim N(\beta, \sigma^2 (\mathbf{X}^T\mathbf{X})^{-1})\\
\mathbf{C}\mathbf{\hat{\beta}}-\mathbf{m} \sim N(\mathbf{C}\beta-\mathbf{m}, \sigma^2 \mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T)
$$

따라서 정리3.3을 이용하면

$$\frac{Q}{\sigma^2} = (\mathbf{C}\mathbf{\hat\beta}-\mathbf{m})^T[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}(\mathbf{C}\mathbf{\hat\beta}-\mathbf{m}) \
 \sim \chi^2(k,\lambda)$$

- $\lambda = \frac{1}{2\sigma^2}(\mathbf{C}\beta - \mathbf{m})^T[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}(\mathbf{C}\beta - \mathbf{m})$는 비중심성 매개변수(non-centrality parameter)이다.
- $k=rank([\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1})=rank([\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]) = rank(\mathbf{C})$

다음으로 $Q$와 $SSE$가 서로 독립임을 보이자. 

$$SSE = \mathbf{y}^T(I_n - \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T)\mathbf{y}$$

$$Q = (\mathbf{C}\mathbf{\hat\beta}-\mathbf{m}) 
[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}(\mathbf{C}\mathbf{\hat\beta}-\mathbf{m}) \\
= \mathbf{y}^T \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \mathbf{y} \\- 2\mathbf{m}^T[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \mathbf{y} + \mathbf{m}^T[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}\mathbf{m}$$

이며 $[I_n - \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T]\mathbf{X} = 0$이므로 정리3.5, 정리3.6을 이용해서 $SSE$와 $Q$는 서로 독립임을 보일 수 있다. 따라서

$$F_0 = \frac{Q/{k\sigma^2}}{SSE/(n-p-1)\sigma^2} = \frac{Q/k}{SSE/(n-p-1)} \sim F(k,n-p-1,\lambda)$$

- $\lambda= \frac{1}{2\sigma^2}(\mathbf{C}\beta - \mathbf{m})^T[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}(\mathbf{C}\beta - \mathbf{m})$는 F-분포의 비중심성 매개변수이다.

**F-통계량**  
귀무가설이 참일 때 $\lambda=0$이므로, 귀무가설이 참인 경우의 검정통계량은 다음과 같다.

$$F_0 = \frac{Q/k}{SSE/(n-p-1)} = \frac{Q/k}{MSE} \sim F(k,n-p-1)$$

기각규칙:

$$F_0 > F_\alpha(k,n-p-1)$$

### 5.4.2 방법 II: 축소모형 접근 (Reduced Model Approach)
귀무가설을 만족하는 제한조건 $\mathbf{C}\mathbf{\beta}=\mathbf{m}$을 만족하는 $\beta$로 모수를 재정의하여 축소모형을 만든다. $\mathbf{\beta_j}$간의 종속관계를 반영하여 모수의 재조정을 통해 축소모형을 만든다.  

예1, $H_0: \beta_1 = \beta_2 = 0$ 라는 가설은 $\beta_1$과 $\beta_2$가 0이 되도록 모수를 재정의하여 축소모형을 만든다.

예2, $\beta_1 = \beta_2, \beta_3 = 2\beta_4$ 라는 가설은 $\beta_1$과 $\beta_3$를 각각 $\beta_2$와 $\beta_4$로 표현하여 모수를 재정의하여 축소모형을 만든다:

$$
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \beta_3 x_{i3} + \beta_4 x_{i4} + \cdots + \varepsilon_i \\
\text{제한조건: } \beta_1 = \beta_2, \beta_3 = 2\beta_4 \\
\text{축소모형: } y_i = \beta_0 + \beta_1 x_{i1} + \beta_1 x_{i2} + 2\beta_4 x_{i3} + \beta_4 x_{i4} + \cdots + \varepsilon_i \\
\text{즉, } y_i = \beta_0 + \beta_1 (x_{i1} + x_{i2}) + \beta_4 (2x_{i3} + x_{i4}) + \cdots + \varepsilon_i $$

귀무가설을 반영하여 모수를 재정의한 축소모형을 만든다.
완전모형(full model)과 축소모형(reduced model)의 잔차제곱합 비교:
  - 완전모형의 잔차제곱합: $SSE_F$, 자유도 $df_F = n-p-1$
  - 축소모형의 잔차제곱합: $SSE_R$, 자유도 $df_R = n-p-1+k$ (k개의 제약조건이 추가되었으므로 자유도가 k만큼 감소한다.)
  - $SSE_R - SSE_F$는 0이상의 양수일 것이고, 귀무가설이 맞는 경우에는 0에 가까울 것이고, 귀무가설이 틀린 경우에는 양수로 커질 것이다.

$$F_0 = \frac{SSE_R-SSE_F}{k} \Big/ \frac{SSE_F}{n-p-1} = \frac{SSE_R-SSE_F}{k \cdot MSE_F} $$

위 검정통계량을 만들면 귀무가설이 성립할 때 $F_0 \sim F(k,n-p-1)$의 분포를 하게 된다. 따라서 $F_0 > F_\alpha(k,n-p-1)$이면 귀무가설을 기각하고, 그렇지 않으면 귀무가설을 채택한다.

위 식의 검정통계량 분자는 실제로 $SSE_R - SSE_F = Q$와 동일하다. 왜냐하면 $SSE_F = SSE$이고, $SSE_R = SSE(\tilde\beta)$이므로 (방법 I에서 구한 제한추정량 $\tilde\beta$를 이용한 잔차제곱합), 

$$SSE_R = \min_{\tilde\beta} (\mathbf{y}-\mathbf{X}\tilde\beta)^T(\mathbf{y}-\mathbf{X}\tilde\beta) \text{ subject to } \mathbf{C}\tilde\beta=\mathbf{m}$$

이고, 이는 방법 I에서 구한 제한추정량 $\tilde\beta$를 이용한 잔차제곱합과 동일하다. 따라서

$$SSE_R - SSE_F = SSE(\tilde\beta) - SSE = Q$$

즉, $Q$ 구하는 계산방법 차이만 있지, 검정통계량 $F_0$는 방법 I과 방법 II가 동일하다.

### 절편 없는 경우
방법I, II의 모든 절차는 절편이 있는 경우와 같고, 자유도만 유의하면 된다.

절편이 없는 모형: $\mathbf{y}=\mathbf{X}\beta+\mathbf{\varepsilon}$ 이면 자유도는 $n-p$가 된다.

$$F_0 \sim F(k,n-p)$$


## 5.5 적합결여검정 (Lack-of-Fit Test)

중회귀모형이 실제 평균구조를 충분히 설명하는지 검정한다.

- $y_i$: $i$번째 관측값
- $E(y_i \mid x_{i1}, \dots, x_{ip}) = \eta_i$

최소제곱법으로 얻은 $\beta$의 추정량 $\hat\beta$를 이용하여 적합된 $\eta_i$의 추정값은

$$
\hat y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip} = \mathbf{x}_i^T \mathbf{\hat\beta}
$$

$y_i$와 $\hat y_i$의 차이인 잔차 $e_i$는
 
$$e_i=y_i-\hat y_i =\underbrace{y_i-E(y_i|x_i)}_{\text{순오차 (pure error)}}+\underbrace{E(y_i|x_i)-\hat y_i}_{\text{적합결여오차 (lack-of-fit error)}}$$

로 분해할 수 있다.

- 순오차: 실험적 변동을 나타내며, 같은 $x$값에서 반복측정이 존재할 때 측정값들 사이의 변동을 나타낸다.
- 적합결여오차: 모형의 구조적 부적합을 나타내며, 모형이 실제 평균구조를 충분히 설명하지 못할 때 발생한다.
  - 즉, $\eta_i$값이 $\mathbf{x}_i^T \beta$와 다를 때 발생한다.

**반복관측이 있을 때**  
순오차를 데이터로부터 직접 추정하기 위해서는 같은 $x$값에서 $y$값의 반복측정이 존재해야 한다.
- $i$: 서로 다른 $\mathbf{x}$값, $1, \dots, k$
- $p$: 설명변수의 개수
- $n_i$: $i$번째 $\mathbf{x}$값에서의 $y_i$ 반복측정 개수
  - $y_{ij}$: $i$번째 $\mathbf{x}$값에서의 $j$번째 반복측정값, $j=1, \dots, n_i$

$$
(x_11, \dots, x_{1p}) \to y_{11}, \dots, y_{1n_1} \\
(x_21, \dots, x_{2p}) \to y_{21}, \dots, y_{2n_2} \\
\vdots \\
(x_{k1}, \dots, x_{kp}) \to y_{k1}, \dots, y_{kn_k} \\
$$

회귀모형 추정식을

$$
\hat y_{ij} = \hat{\beta}_0+\hat{\beta}_1 x_{i1} + \cdots + \hat{\beta}_p x_{ip} = \mathbf{x}_i^T \hat{\beta}, \quad i = 1, \dots, k, \quad j = 1, \dots, n_i
$$

라 하면 잔차제곱합은

$$SSE = \sum_{i=1}^k \sum_{j=1}^{n_i} (y_{ij}-\hat y_{ij})^2$$

이고, $\bar y_i = \frac{1}{n_i} \sum_{j=1}^{n_i} y_{ij}$ 라 하면 SSE는 둘로 분해할 수 있다.

$$SSE = \sum_{i=1}^k \sum_{j=1}^{n_i} (y_{ij}-\bar y_i)^2 + \sum_{i=1}^k n_i (\bar y_i - \hat y_i)^2 = SS_{PE} + SS_{LF}$$

- $SS_{PE}$: 순오차제곱합
  - 같은 $x$값에서 반복측정이 존재할 때 측정값들 사이의 변동을 나타낸다. 
  - 자유도는 $df_{PE} = n-k$이다.
- $SS_{LF}$: 적합결여제곱합
  - 모형의 구조적 부적합을 나타낸다.
  - 자유도는 $SSE$의 자유도에서 $SS_{PE}$의 자유도를 뺀 값으로, $df_{LF} = k-p-1$이다.

적합결여오차가 커진다는것은 $SS_{LF}$가 커진다는 것이고, 이는 모형이 실제 평균구조를 충분히 설명하지 못한다는 것을 의미한다. 따라서 적합결여검정은 $SS_{LF}$와 $SS_{PE}$를 비교하여 수행한다.

### 5.5.3 F-검정
평균제곱:

$$MS_{PE} = \frac{SS_{PE}}{n-k}, \quad MS_{LF} = \frac{SS_{LF}}{k-p-1}$$

검정통계량:

$$F_0 = \frac{MS_{LF}}{MS_{PE}} \sim F(k-p-1, n-k)$$

**분산분석표 (ANOVA Table for Lack-of-Fit Test)**

| 요인 | 제곱합 | 자유도 | 평균제곱 | $F_0$ |
|------|--------|--------|---------|-------|
| 회귀 | $SSR$ | $p$ | $MSR=\frac{SSR}{p}$ | |
| 잔차 | $SSE$ | $n-p-1$ | $MSE=\frac{SSE}{n-p-1}$ | |
| 　 순오차 | $SS_{PE}$ | $n-k$ | $MS_{PE}=\frac{SS_{PE}}{n-k}$ | |
| 　 적합결여 | $SS_{LF}$ | $k-p-1$ | $MS_{LF}=\frac{SS_{LF}}{k-p-1}$ | $F_L=\frac{MS_{LF}}{MS_{PE}}$ |
| 계 | $SST$ | $n-1$ | | |

- 가정된 회귀모형이 적합한가의 검정은 위 표의 $F_L$를 이용하여 검정한다.
  - 분산분석표의 $F_L$값과 $F_L$의 기각치$F_\alpha(k-p-1, n-k)$를 비교하여 검정한다.
  - $F_L$가 크면 → 모형 구조가 잘못되었을 가능성
  - 작으면 → 현재 모형 유지 가능
  - 즉, 적합결여오차가 순오차에 비해 유의하게 크면 모형 부적합이다.


## 5.6 잔차의 검토 (Residual Analysis)
잔차분석은 회귀가정의 타당성을 진단하는 절차이다.

### 5.6.1 잔차의 기본 성질
정규방정식: $X^TX\hat\beta = X^Ty$로부터 다음이 성립한다.

**(1) 잔차의 합** 
 
$$\sum e_i = 0$$

$$
\sum e_i = \sum (y_i - \hat y_i) = \sum y_i - \sum (\hat\beta_0 + \hat\beta_1 x_{i1} + \cdots + \hat\beta_p x_{ip}) \\
= \sum y_i - \hat\beta_0 n - \sum \hat\beta_j \sum x_{ij} = 0
$$

**(2) 잔차들의 $x_{ij}$에 대한 가중합은 0 (잔차와 설명변수의 직교성)**

$$\sum x_{ij} e_i = 0, \quad j=1,\dots,p$$

즉, 잔차는 설계행렬 $X$의 열공간과 직교한다.

$$
\sum x_{ij} e_i = \sum x_{ij} (y_i - \hat \beta_0 - \hat\beta_1 x_{i1} - \cdots - \hat\beta_p x_{ip}) \\ = \sum x_{ij} y_i - \hat\beta_0 \sum x_{ij} - \sum \hat\beta_k \sum x_{ij} x_{ik} = 0
$$

**(3) 잔차들의 $\hat y_i$에 대한 가중합은 0 (잔차와 예측값의 직교성)**

$$\sum \hat y_i e_i = 0$$

$$
\sum \hat y_i e_i = \sum (\hat\beta_0 + \hat\beta_1 x_{i1} + \cdots + \hat\beta_p x_{ip}) (e_i) \\ = \sum \hat\beta_0 e_i + \sum_{j=1}^p \hat\beta_j \sum_{i=1}^n x_{ij} e_i = 0
$$

**(4) 잔차 $\varepsilon_i$간 상관관계 존재**  

$\mathbf{e} = \mathbf{y} - \mathbf{X}\hat{\beta} = [I - X(X^TX)^{-1}X^T]\mathbf{y}$이므로, 

$$ E(\mathbf{e}) = 0, \quad Var(\mathbf{e}) = \sigma^2 [I - X(X^TX)^{-1}X^T]$$

이므로 $Var(\mathbf{e})$는 일반적으로 대각행렬이 아니어서 잔차들 사이에는 공분산이 존재한다.
- 이 상관계수는 $\rho_{ij} = \frac{Cov(e_i,e_j)}{\sqrt{Var(e_i)Var(e_j)}}$로 정의할 수 있다. 이 값은 $\sigma^2$에 의존하지 않고 설계행렬 $X$에 의해 결정된다.

- 잔차의 산점도를 그려봄으로써 중회귀모형의 가정을 점검할 수 있다.

### 5.6.3 잔차 산점도 해석
잔차를
* $\hat y$에 대해
* 각 $x_j$에 대해
* 시간에 대해 (시계열의 경우)
그려서 모형가정을 점검한다.

* $\hat y$에 대해
  - 무작위로 흩어져 있으면 → 등분산성 가정 만족
  - 사다리꼴로 분산이 점점 커지는 패턴을 보이면 → 이분산성 가능성
  - 동일분산으로 상승/하강하는 패턴을 보이면 → 절편이 필요한데 절편이 없는 모형을 사용했을 가능성
  - 특정한 패턴(y=-x^2같은)을 보이면 → 모형의 구조적 부적합 가능성
    - 설명변수의 제곱항이나 교호작용항이 필요할 수 있다.

* 각 $x_j$에 대해
  - 무작위로 흩어져 있으면 → 선형성 가정 만족
  - 사다리꼴로 분산이 커지는 패턴을 보이면 → 가중회귀를 쓰거나 $y_i$에 대한 변환이 필요할 수 있다.
  - 동일분산으로 상승/하강하는 패턴을 보이면 → $x_{ij}$의 선형효과가 적절히 취급되지 않음. x항을 빼먹은 실수를 하거나 계산상 실수
  - 특정한 패턴(y=-x^2같은)을 보이면 → 모형의 구조적 부적합 가능성
    - 설명변수의 제곱항이나 교호작용항이 필요할 수 있다.


---
[연습문제 7.6]

중회귀모형을 $\mathbf y=\mathbf X\beta+\varepsilon,\quad E(\varepsilon)=0,\quad \text{Var}(\varepsilon)=\sigma^2I_n$ 이라고 하자. 다음 행렬을 정의한다.

$$
H=\mathbf X(\mathbf X^T\mathbf X)^{-1}\mathbf X^T,
\quad
M=I_n-H.
$$

그러면 적합값과 잔차벡터는 각각

$$
\hat{\mathbf y}=H\mathbf y,\quad
\mathbf e=\mathbf y-\hat{\mathbf y}=M\mathbf y
$$

$H$와 $M$은 대칭 멱등행렬이며

$$
H^T=H,\quad M^T=M,\quad
H^2=H,\quad M^2=M,\quad
MH=HM=0
$$

을 만족한다. 또한 $M\mathbf X=0,\quad \mathbf X^TM=0$ 이다.

(1) $\text{Cov}(\mathbf e,\mathbf y)$

$\mathbf e=M\mathbf y$ 이므로

$$
\begin{aligned}
\text{Cov}(\mathbf e,\mathbf y)
&=\text{Cov}(M\mathbf y,\mathbf y)\\
&=M\text{Var}(\mathbf y)\\
&=M(\sigma^2I_n)\\
&=\sigma^2M.
\end{aligned}
$$

따라서

$$
\boxed{
\text{Cov}(\mathbf e,\mathbf y) = \sigma^2
\left[
I_n-\mathbf X(\mathbf X^T\mathbf X)^{-1}\mathbf X^T
\right]
}
$$

(2) $\text{Cov}(\mathbf e,\hat{\mathbf y})$

$\mathbf e=M\mathbf y,\quad \hat{\mathbf y}=H\mathbf y$ 이므로

$$
\begin{aligned}
\text{Cov}(\mathbf e,\hat{\mathbf y})
&=\text{Cov}(M\mathbf y,H\mathbf y)\\
&=M\text{Var}(\mathbf y)H^T\\
&=\sigma^2MH\\
&=0.
\end{aligned}
$$

따라서

$$
\boxed{
\text{Cov}(\mathbf e,\hat{\mathbf y})=O_n
}
$$

정규오차를 가정하면 $\mathbf e$와 $\hat{\mathbf y}$는 결합정규분포를 따르므로 무상관일 뿐만 아니라 서로 독립이다.

(3) $\text{Cov}(\mathbf e,\hat\beta)$

$\hat\beta =(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf y$ 이므로

$$
\begin{aligned}
\text{Cov}(\mathbf e,\hat\beta)
&=
\text{Cov}
\left(
M\mathbf y,\,
(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf y
\right)\\
&=
M\text{Var}(\mathbf y)
\left[
(\mathbf X^T\mathbf X)^{-1}\mathbf X^T
\right]^T\\
&=
\sigma^2M\mathbf X(\mathbf X^T\mathbf X)^{-1}\\
&=0.
\end{aligned}
$$

따라서

$$
\boxed{
\text{Cov}(\mathbf e,\hat\beta) = O_{n\times(p+1)}
}
$$

마찬가지로 정규오차를 가정하면 $\mathbf e\perp\!\!\!\perp\hat\beta$ 도 성립한다.

(4) $\text{Cov}(\varepsilon,\hat\beta)$

먼저 $\hat\beta =(\mathbf X^T\mathbf X)^{-1}\mathbf X^T (\mathbf X\beta+\varepsilon) =\beta+(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\varepsilon.$ 이므로 

$$
\begin{aligned}
\text{Cov}(\varepsilon,\hat\beta)
&=
\text{Cov}
\left(
\varepsilon,\,
(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\varepsilon
\right)\\
&=
\text{Var}(\varepsilon)
\left[
(\mathbf X^T\mathbf X)^{-1}\mathbf X^T
\right]^T\\
&=
\sigma^2I_n\mathbf X(\mathbf X^T\mathbf X)^{-1}.
\end{aligned}
$$

따라서

$$
\boxed{
\text{Cov}(\varepsilon,\hat\beta) = \sigma^2\mathbf X(\mathbf X^T\mathbf X)^{-1}
}
$$

주의할 점은 잔차 $\mathbf e$와 참오차 $\varepsilon$가 다르다는 것이다.  
$\text{Cov}(\mathbf e,\hat\beta)=0$ 이지만 일반적으로 $\text{Cov}(\varepsilon,\hat\beta)\neq0$ 이다. 추정량 $\hat\beta$가 오차 $\varepsilon$를 이용하여 계산되기 때문이다.

(5) $\displaystyle \sum_{i=1}^n e_i y_i=SSE$

벡터로 나타내면

$$
\sum_{i=1}^n e_i y_i
=\mathbf e^T\mathbf y.
$$

그런데 $\mathbf y=\hat{\mathbf y}+\mathbf e$ 이므로

$$
\mathbf e^T\mathbf y =\mathbf e^T(\hat{\mathbf y}+\mathbf e) =\mathbf e^T\hat{\mathbf y}+\mathbf e^T\mathbf e.
$$

잔차와 적합값은 직교하므로 $\mathbf e^T\hat{\mathbf y}=0.$ 또한

$$
\mathbf e^T\mathbf e
=\sum_{i=1}^ne_i^2
=SSE.
$$

따라서

$$
\boxed{
\sum_{i=1}^ne_i y_i=SSE
}
$$

행렬로 직접 계산해도 된다.

$$
\begin{aligned}
\mathbf e^T\mathbf y
&=(M\mathbf y)^T\mathbf y\\
&=\mathbf y^TM\mathbf y\\
&=\mathbf y^TM^2\mathbf y\\
&=(M\mathbf y)^T(M\mathbf y)\\
&=\mathbf e^T\mathbf e=SSE.
\end{aligned}
$$

여기서는 $M^T=M$, $M^2=M$을 이용했다.

(6) $\displaystyle \sum_{i=1}^ne_i\hat y_i=0$

벡터로 나타내면

$$
\sum_{i=1}^ne_i\hat y_i
=\mathbf e^T\hat{\mathbf y}.
$$

$$
\mathbf e=M\mathbf y,\quad
\hat{\mathbf y}=H\mathbf y
$$

이므로

$$
\begin{aligned}
\mathbf e^T\hat{\mathbf y}
&=(M\mathbf y)^TH\mathbf y\\
&=\mathbf y^TM^TH\mathbf y\\
&=\mathbf y^TMH\mathbf y\\
&=0.
\end{aligned}
$$

따라서

$$
\boxed{
\sum_{i=1}^ne_i\hat y_i=0
}
$$

정규방정식으로도 증명할 수 있다. $\hat{\mathbf y}=\mathbf X\hat\beta$이므로 $\mathbf e^T\hat{\mathbf y} =\mathbf e^T\mathbf X\hat\beta =(\mathbf X^T\mathbf e)^T\hat\beta =0$ 이다. 여기서 OLS 정규방정식 $\mathbf X^T\mathbf e=0$ 을 사용했다.

---

**절편이 없는 중회귀모형**

절편이 없는 모형에서는 $\mathbf X$의 열에 모든 성분이 1인 벡터 $\mathbf 1_n$이 포함되지 않을 뿐이다.

투영행렬 $H=\mathbf X(\mathbf X^T\mathbf X)^{-1}\mathbf X^T$ 과 잔차생성행렬 $M=I_n-H$ 의 성질은 그대로 유지된다. 따라서 문제의 (1)∼(6)은 절편이 없어도 모두 그대로 성립한다.

다만 절편이 있는 모형에서 추가로 성립하는 $\sum_{i=1}^ne_i=0$ 은 절편이 없는 모형에서는 일반적으로 성립하지 않는다.

절편이 있으면 $\mathbf 1_n$이 $\mathbf X$의 열공간에 포함되므로 $\mathbf 1_n^T\mathbf e=0$ 이지만, 절편이 없으면 $\mathbf 1_n$이 $\mathbf X$의 열공간에 포함된다는 보장이 없기 때문이다.

---

[연습문제 7.7]

증명할 식은 다음과 같다.

$$
(\mathbf y-\mathbf X\beta)^T(\mathbf y-\mathbf X\beta) = (\mathbf y-\mathbf X\hat\beta)^T
(\mathbf y-\mathbf X\hat\beta) + (\beta-\hat\beta)^T
\mathbf X^T\mathbf X
(\beta-\hat\beta).
$$

여기서 식의 $\beta$는 참모수라기보다 최소화할 목적함수의 임의의 후보값을 의미한다. 혼동을 피하기 위해 임의의 후보값을 $b$라고 쓰겠다.

$$
SSE(b)=(\mathbf y-\mathbf Xb)^T(\mathbf y-\mathbf Xb)
$$

라고 하자.

**1단계: 잔차 분해**

$$
\begin{aligned}
\mathbf y-\mathbf Xb
&=\mathbf y-\mathbf X\hat\beta +\mathbf X\hat\beta-\mathbf Xb\\
&=(\mathbf y-\mathbf X\hat\beta) +\mathbf X(\hat\beta-b).
\end{aligned}
$$

따라서

$$
\begin{aligned}
SSE(b)
&=
\left[
(\mathbf y-\mathbf X\hat\beta) +\mathbf X(\hat\beta-b)
\right]^T \cdot
\left[
(\mathbf y-\mathbf X\hat\beta) +\mathbf X(\hat\beta-b)
\right] \\
&= (\mathbf y-\mathbf X\hat\beta)^T
(\mathbf y-\mathbf X\hat\beta) +2(\hat\beta-b)^T
\mathbf X^T(\mathbf y-\mathbf X\hat\beta) +(\hat\beta-b)^T
\mathbf X^T\mathbf X
(\hat\beta-b).
\end{aligned}
$$

**2단계: 교차항 제거**

OLS 추정량은 정규방정식 $\mathbf X^T(\mathbf y-\mathbf X\hat\beta)=0$ 을 만족한다. 따라서 교차항은 $2(\hat\beta-b)^T \mathbf X^T(\mathbf y-\mathbf X\hat\beta)=0$

그러므로

$$
SSE(b) = SSE(\hat\beta) + (\hat\beta-b)^T\mathbf X^T\mathbf X(\hat\beta-b).
$$

이차형식은 부호를 바꾸어도 같으므로

$$
(\hat\beta-b)^T\mathbf X^T\mathbf X(\hat\beta-b) = (b-\hat\beta)^T\mathbf X^T\mathbf X(b-\hat\beta).
$$

따라서

$$
\boxed{
SSE(b) = SSE(\hat\beta) + (b-\hat\beta)^T
\mathbf X^T\mathbf X
(b-\hat\beta)
}
$$

를 얻는다.

**3단계: 최소성**

두 번째 항은

$$
\begin{aligned}
(b-\hat\beta)^T
\mathbf X^T\mathbf X
(b-\hat\beta)
&=
[\mathbf X(b-\hat\beta)]^T
[\mathbf X(b-\hat\beta)]\\
&=
\|\mathbf X(b-\hat\beta)\|^2\\
&\geq0
\end{aligned}
$$

따라서 모든 $b$에 대하여 $SSE(b)\geq SSE(\hat\beta).$ 즉,

$$
\boxed{
\hat\beta = \arg\min_b
(\mathbf y-\mathbf Xb)^T(\mathbf y-\mathbf Xb)
}
$$

또한 $\mathbf X$가 full column rank이면 $\mathbf X^T\mathbf X$는 양의 정부호이므로

$$
(b-\hat\beta)^T\mathbf X^T\mathbf X(b-\hat\beta)=0
\iff b=\hat\beta.
$$

따라서 최소제곱해는 유일하다.

반대로 $\mathbf X$가 full column rank가 아니라면 $\mathbf X(b-\hat\beta)=0$ 을 만족하는 $b\neq\hat\beta$가 존재할 수 있으므로 최소제곱해가 유일하지 않을 수 있다.

이 식은 기하학적으로 잔차벡터 $\mathbf y-\mathbf X\hat\beta$ 와 추가 변화량 $\mathbf X(\hat\beta-b)$ 가 서로 직교하기 때문에 성립하는 피타고라스 정리이다.
