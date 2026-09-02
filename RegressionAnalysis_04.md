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

$$\mathbf y = X\beta + \varepsilon$$

* $\mathbf y$ : $n\times1$ 반응벡터(response vector)
* $X$ : $n\times(p+1)$ 설계행렬(design matrix)
  - p: 설명변수의 수
* $\beta$ : $(p+1)\times1$ 회귀계수 벡터
* $\varepsilon$ : 오차벡터(error vector)
  - $E(\varepsilon)=0_n$, $\text{Var}(\varepsilon)=\sigma^2 I_n$

두 설명변수의 경우

$$X=\begin{pmatrix}
1 & x_{11} & x_{12} \\
1 & x_{21} & x_{22} \\
\vdots & \vdots & \vdots \\
1 & x_{n1} & x_{n2}
\end{pmatrix}$$

여기서

$$X^T X = \begin{pmatrix}
\sum_{i=1}^n 1 & \sum_{i=1}^n x_{i1} & \sum_{i=1}^n x_{i2} \\
\sum_{i=1}^n x_{i1} & \sum_{i=1}^n x_{i1}^2 & \sum_{i=1}^n x_{i1}x_{i2} \\
\sum_{i=1}^n x_{i2} & \sum_{i=1}^n x_{i1}x_{i2} & \sum_{i=1}^n x_{i2}^2
\end{pmatrix}
= \begin{pmatrix}
 n & \sum x_{i1} & \sum x_{i2} \\
 \sum x_{i1} & \sum x_{i1}^2 & \sum x_{i1}x_{i2} \\
 \sum x_{i2} & \sum x_{i1}x_{i2} & \sum x_{i2}^2
\end{pmatrix}$$

그리고

$$X^T \mathbf y = \begin{pmatrix}
\sum_{i=1}^n y_i \\
\sum_{i=1}^n x_{i1}y_i \\
\sum_{i=1}^n x_{i2}y_i
\end{pmatrix}
= \begin{pmatrix}
\sum y_i \\
\sum x_{i1}y_i \\
\sum x_{i2}y_i
\end{pmatrix}$$

이므로, 정규방정식 $X^T X\hat\beta = X^T y$는

$$\begin{pmatrix}
 n & \sum x_{i1} & \sum x_{i2} \\
 \sum x_{i1} & \sum x_{i1}^2 & \sum x_{i1}x_{i2} \\
 \sum x_{i2} & \sum x_{i1}x_{i2} & \sum x_{i2}^2
\end{pmatrix}
\begin{pmatrix}
\hat\beta_0 \\
\hat\beta_1 \\
\hat\beta_2
\end{pmatrix}
= \begin{pmatrix}
\sum y_i \\
\sum x_{i1}y_i \\
\sum x_{i2}y_i
\end{pmatrix}$$

으로 쓸 수 있다.

**(2) 최소제곱해 (Least Squares Solution)**  
오차제곱합은 

$$S = (\mathbf y-X\beta)^T (\mathbf y-X\beta)
= y^Ty-2\beta^TX^Ty+\beta^TX^TX\beta$$

이다. 여기서 $y^Ty$는 $\beta$와 무관한 상수이고, $\beta$에 대한 미분은

$$\frac{\partial}{\partial \beta}(\beta^T a)=a, \qquad
\frac{\partial}{\partial \beta}(\beta^T A \beta)=2A\beta
$$

(여기서 $A$는 대칭행렬) 를 이용하면

$$\frac{\partial S}{\partial \beta}
= -2X^Ty + 2X^TX\beta.$$

최소제곱해는 이 미분값이 0이 되는 $\hat\beta$이므로

$$-2X^Ty + 2X^TX\hat\beta = 0$$

즉,

$$X^T X \hat\beta = X^T y.$$

이를 정규방정식(normal equation) 또는 행렬형 정규방정식(matrix normal equation)이라 한다.

이를 행렬형 정규방정식(matrix normal equation)이라 한다.

**(3) 해의 존재 조건**  
$X^T X$가 가역행렬(invertible matrix)일 때 (가역이 되기 위한 필요충분조건은 $\text{rank}(X)=p+1$)

$$\hat\beta = (X^T X)^{-1} X^T y$$

즉 설명변수들 사이에 완전한 선형종속(linear dependence)이 없어야 한다.

>참고: 설계행렬이 $X\in\mathbb R^{n\times(p+1)}$ 일 때 다음 두 조건은 동치이다.
>
>$$
>\boxed{\text{rank}(X)=p+1
>\iff X^\top X\text{가 가역이다}}
>$$
>
>[**증명**]
>
>임의의 $\mathbf a\in\mathbb R^{p+1}$에 대하여 $\mathbf a^\top X^\top X\mathbf a =(X\mathbf a)^\top(X\mathbf a) =\lVert X\mathbf a\rVert^2$  
>먼저 $X^\top X\mathbf a=\mathbf0$ 이라고 하자. 양변 왼쪽에 $\mathbf a^\top$을 곱하면 $\mathbf a^\top X^\top X\mathbf a=0$ 이고, 따라서 $\lVert X\mathbf a\rVert^2=0$ 이다. 벡터의 제곱노름이 $0$이므로 $X\mathbf a=\mathbf0$ 이다.
>
>반대로 $X\mathbf a=\mathbf0$이면 $X^\top X\mathbf a =X^\top\mathbf0 =\mathbf0$ 이다.
>
>따라서 $\boxed{\text{Null}(X^\top X)=\text{Null}(X)}$
>
>---
>
>1. $\text{rank}(X)=p+1$이면 $X^\top X$가 가역이다
>
>$\text{rank}(X)=p+1$이라는 것은 $X$의 $p+1$개 열벡터가 선형독립이라는 뜻이다.  
>따라서 $X\mathbf a=\mathbf0$ 의 유일한 해는 $\mathbf a=\mathbf0$ 이다. 즉, 
>$\text{Null}(X)=\{\mathbf0\}$ 이다.
>
>앞에서 두 영공간이 같음을 보였으므로 $\text{Null}(X^\top X)=\{\mathbf0\}$ 이다.
>
>$X^\top X$는 $(p+1)\times(p+1)$ 정사각행렬이다. 정사각행렬의 영공간에 영벡터만 존재하면 그 행렬은 가역이다. 따라서
>
>$$
>\boxed{
>\text{rank}(X)=p+1 \Longrightarrow X^\top X\text{가 가역이다}
>}
>$$
>
>---
>
>2. $X^\top X$가 가역이면 $\text{rank}(X)=p+1$이다
>
>$X^\top X$가 가역이라고 하자. 그러면 $X^\top X\mathbf a=\mathbf0$ 의 유일한 해는 $\mathbf a=\mathbf0$이다.
>
>이제 $X\mathbf a=\mathbf0$ 이라고 하자. 양변에 $X^\top$을 곱하면 $X^\top X\mathbf a=\mathbf0 $ 이다. $X^\top X$가 가역이므로 반드시 $\mathbf a=\mathbf0$ 이다. 따라서 $X\mathbf a=\mathbf0$의 해가 영벡터뿐이므로 $X$의 열벡터들이 선형독립이다. 즉,
>
>$$
>\boxed{\text{rank}(X)=p+1}
>$$
>
>---
>
>**양의 정부호를 이용한 설명**
>
>임의의 $\mathbf a$에 대하여 $\mathbf a^\top X^\top X\mathbf a =\lVert X\mathbf a\rVert^2\ge0$ 이므로 $X^\top X$는 항상 양의 준정부호이다.
>
>만약 $\text{rank}(X)=p+1$이면 $\mathbf a\ne\mathbf0$에 대하여 $X\mathbf a\ne\mathbf0$ 이다. 따라서
>
>$$
>\mathbf a^\top X^\top X\mathbf a
>=\lVert X\mathbf a\rVert^2>0
>$$
>
>즉, $X^\top X$는 양의 정부호이고, 양의 정부호 행렬은 가역이다. 따라서 다음 세 조건이 동치이다.
>
>$$
>\boxed{
>\text{rank}(X)=p+1
>\iff X^\top X\text{가 양의 정부호}
>\iff X^\top X\text{가 가역}
>}
>$$
>
>---
>
>**통계적 의미**
>
>다음 조건들은 모두 동치이다.
>
>1. $\text{rank}(X)=p+1$이다.
>2. $X$의 모든 열벡터가 선형독립이다.
>3. 완전 다중공선성이 존재하지 않는다.
>4. $X^\top X$가 가역이다.
>5. 최소제곱추정량 $\hat{\boldsymbol\beta}$가 유일하다.
>
>반대로 $\text{rank}(X)<p+1$이면 어떤 $\mathbf a\ne\mathbf0$에 대하여 $X\mathbf a=\mathbf0$ 이 성립한다. 따라서
>
>$$
>X(\boldsymbol\beta+\mathbf a)
>=X\boldsymbol\beta+X\mathbf a
>=X\boldsymbol\beta
>$$
>
>이다. 즉, 서로 다른 계수벡터 $\boldsymbol\beta$와 $\boldsymbol\beta+\mathbf a$가 동일한 회귀평균을 만든다. 따라서 회귀계수는 유일하게 식별되지 않는다.
>
>**$XX^\top$와의 차이**
>
>$X$가 $n\times(p+1)$ 행렬이고 $n>p+1$이면 $XX^\top\in\mathbb R^{n\times n}$ 이다. 그러나 $\text{rank}(XX^\top) =\text{rank}(X) =p+1<n$ 이므로 $XX^\top$는 가역이 아니다.
>

## 4.4 분산분석 (Analysis of Variance, ANOVA)
회귀모형에서의 분산분석은 반응변수의 총변동(total variation)을
* 회귀식에 의해 설명되는 변동 (variation due to regression)
* 잔차에 의한 변동 (variation due to residuals)

으로 분해하는 과정이다.

중회귀모형에서 이들 변동이 어떻게 표현되는지 살펴보자.

$$y = X\beta + \varepsilon, \qquad \varepsilon \sim N(0_n, \sigma^2 I_n)$$

### 4.4.1 총변동의 분해 (Decomposition of Total Variation)

**(1) 총제곱합 (Total Sum of Squares, SST)**  
총변동 $SST = \sum_{i=1}^n (y_i - \bar y)^2$

행렬형으로는

$$SST = \mathbf{y}^T \mathbf{y} - n(\bar y)^2 = \mathbf{y}^T\left(I_n - \frac{J_n}{n}\right)\mathbf{y}$$

* $I_n$ : 단위행렬 (identity matrix)
* $J_n$ : 모든 원소가 1인 행렬

>자유도에 대한 고찰  
>자유도(degree of freedom)는 $n-1$이다. 이는 $\sum (y_i-\bar y)=0$이라는 하나의 선형 제약이 존재하기 때문이다.  
>
> 통계학에서 자유도란 근원적으로 카이제곱 분포와 관련있다. 즉, 정규성 가정하에 어떤 통계량(t 또는 F 통계량)이 자요도 $\nu$인 카이제곱 분포를 따르는 확률변수를 포함하고 있음을 나타낸다.  
>
>회귀분석에서 다뤄지는 모든 통계량은 3장에서 배운 이차형식으로 나타낼 수 있고, 거기서 배운 내용들로 분포를 계산할 수 있다. 또한, 위 식은 이차형식으로 표현되므로, 3.5절의 정리로부터 자유도는 $n-1$의 카이제곱분포를 따른다. 이처럼 자유도를 정규분포를 가정하고 이를 이용한 카이제곱분포와 연결하여 생각하면 이해도 쉽고 계산도 쉬워진다.

**(2) 잔차제곱합 (Sum of Squares due to Error, SSE)**  
예측값 벡터는 $\hat{\mathbf{y}} = X\hat\beta$  
잔차는 $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$  
잔차제곱합은

$$SSE = (\mathbf{y}-\hat{\mathbf{y}})^T (\mathbf{y}-\hat{\mathbf{y}})\\ 
= \mathbf{y}^T\mathbf{y} - 2\hat\beta^T X^T \mathbf{y} + \hat\beta^T X^T X \hat\beta \\
= \mathbf{y}^T\left[I_n - X(X^T X)^{-1}X^T\right]\mathbf{y} \\
= \mathbf{y}^T \mathbf{y} - \hat\beta^T X^T \mathbf{y}$$

자유도는 $\mathbf{y}^T \mathbf{y}$에서 $\hat\beta^T X^T \mathbf{y}$를 빼는 과정에서 $p+1$개의 선형 제약이 추가되므로 $n - p - 1$  

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
F-검정의 통계적 의미를 고찰하기 위해  MSR, MSE의 기댓값을 계산해보자.


$$E(y) = X\beta, \qquad \text{Var}(y) = \sigma^2 I_n$$

(i) SSE의 기대값

정리 3.1을 사용하면,

$$E(SSE) = E\left[y^T\left(I_n - X(X^T X)^{-1}X^T\right)y\right] \\
= \sigma^2\text{tr}(I_n - X(X^T X)^{-1}X^T)+\beta^TX^T(I_n - X(X^T X)^{-1}X^T)X\beta$$

그런데 $\text{tr}\left[\left(I_n - X(X^T X)^{-1}X^T\right)\right] = \text{tr}(I_n)- \text{tr}(X(X^T X)^{-1}X^T) = n-(p+1)$ 이고,

$X^T(I_n - X(X^T X)^{-1}X^T)X = O_{p+1}$ 이므로

$$E(SSE) = \sigma^2(n-p-1)$$

$$E(MSE) = \frac{E(SSE)}{n-p-1} = \sigma^2$$

즉, MSE는 오차분산 $\sigma^2$의 불편추정량(unbiased estimator)이다.

(ii) SSR의 기대값
  
$$
\begin{aligned}
E(SSR)
&=
E\left[\mathbf y^T X(X^TX)^{-1}X^T\mathbf y- n\bar y^2\right]\\
&=
E\left[\mathbf y^T X(X^TX)^{-1}X^T\mathbf y -\mathbf y^T\left(\frac{J_n}{n}\right)\mathbf y\right]\\
&=
E\left[\mathbf y^T
\{ X(X^TX)^{-1}X^T-\frac{J_n}{n} \}
\mathbf y\right].
\end{aligned}
$$

여기서

$$
A=X(X^TX)^{-1}X^T-\frac{J_n}{n}
$$

라고 놓고 정리 3.1을 적용하면

$$
\begin{aligned}
E(SSR)
&=
E(\mathbf y^TA\mathbf y)\\
&=
\text{tr}\left[A\text{Var}(\mathbf y)\right] + E(\mathbf y)^TAE(\mathbf y)\\
&=
\text{tr}(A\sigma^2I_n) + (X\boldsymbol\beta)^TA(X\boldsymbol\beta)\\
&=
\sigma^2\text{tr}(A) + \boldsymbol\beta^TX^TAX\boldsymbol\beta.
\end{aligned}
$$

먼저 $\text{tr}(A)$를 계산하면

$$
\begin{aligned}
\text{tr}(A)
&=
\text{tr}\left[X(X^TX)^{-1}X^T\right] - \text{tr}\left(\frac{J_n}{n}\right)\\
&=
\text{tr}\left[(X^TX)^{-1}X^TX\right] - \frac{1}{n}\text{tr}(J_n)\\
&=
\text{tr}(I_{p+1})-\frac{n}{n}\\
&=
(p+1)-1\\
&=p.
\end{aligned}
$$

다음으로 $X^TAX$를 계산한다.

$$
\begin{aligned}
X^TAX
&=
X^T
\left[
X(X^TX)^{-1}X^T-\frac{J_n}{n}
\right]X\\
&=
X^TX(X^TX)^{-1}X^TX - X^T\frac{J_n}{n}X\\
&=
X^TX-X^T\frac{J_n}{n}X\\
&=
X^T\left(I_n-\frac{J_n}{n}\right)X.
\end{aligned}
$$

따라서

$$
\boxed{
E(SSR)
=
p\sigma^2
+
\boldsymbol\beta^TX^T
\left(I_n-\frac{J_n}{n}\right)
X\boldsymbol\beta
}
$$

따라서

$$E(MSR) = \frac{E(SSR)}{p} = \sigma^2 + \frac{1}{p} \beta^T X^T \left(I_n - \frac{J_n}{n}\right) X\beta$$

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

$$\text{Var}(\hat y)
=\mathbf{x}^T\text{Var}(\hat\beta)\mathbf{x}
=\sigma^2 \mathbf{x}^T(X^TX)^{-1}\mathbf{x}$$

즉 예측값 $\hat y$의 분산은 설명변수 벡터 $x$와 설계행렬 $X$의 구조에 의해 결정된다.

개별 관측값 예측의 경우, 새로운 관측값 $y_{new} = \mathbf{x}^T\hat\beta + \varepsilon_{new}$의 분산은 추정된 회귀식의 오차와 새로운 관측값의 오차 모두를 포함해야 한다.

$$\text{Var}(\hat y_{new})
=\text{Var}(\mathbf{x}^T\hat\beta) + \text{Var}(\varepsilon_{new}) = \sigma^2\mathbf{x}^T(X^TX)^{-1}\mathbf{x} + \sigma^2 = \sigma^2\left[1+\mathbf{x}^T(X^TX)^{-1}\mathbf{x}\right]$$

따라서 개별 관측값의 예측분산은 평균반응의 추정분산보다 항상 크다. 이는 개별값 예측이 평균값 추정보다 불확실성이 크다는 의미이며, 이를 반영하여 더 넓은 예측구간(prediction interval)을 구성하게 된다.

단순회귀의 경우 이는

$$\text{Var}(\hat y)=\sigma^2\left[1+\frac{1}{n}+\frac{(x-\bar x)^2}{\sum (x_i-\bar x)^2}\right]$$

와 일치한다.


## 4.6 절편 없는 중회귀모형 (Regression without Intercept)
일반 중회귀모형은 절편(intercept term) $\beta_0$을 포함한다. 그러나 설명변수가 0일 때 반드시 $y=0$이어야 하는 구조적 제약이 있으면 절편을 제거한다.

**4.6.1 모형** 
 
$$y_i=\beta_1x_{i1}+\cdots+\beta_px_{ip}+\varepsilon_i =X\beta+\varepsilon$$

단, 여기서 $X$는 첫 열이 1이 아닌 $n\times p$ 행렬이다.

**4.6.2 제곱합**  

절편 없는 모형에서는 원점 $0$을 기준으로 한 비중심 총제곱합을 사용한다.

$$
\boxed{
SST_U
=
\sum_{i=1}^n y_i^2
=
\mathbf y^T\mathbf y
}
$$

여기서 아래첨자 $U$는 uncorrected 또는 uncentered를 의미한다.

예측값과 잔차 사이에는 $\mathbf y=\hat{\mathbf y}+\mathbf e$ 가 성립하므로

$$
\begin{aligned}
\mathbf y^T\mathbf y
&=
(\hat{\mathbf y}+\mathbf e)^T
(\hat{\mathbf y}+\mathbf e)\\
&=
\hat{\mathbf y}^T\hat{\mathbf y}
+2\hat{\mathbf y}^T\mathbf e
+\mathbf e^T\mathbf e.
\end{aligned}
$$

그런데 $\hat{\mathbf y}^T\mathbf e=0$이므로

$$
\boxed{
\mathbf y^T\mathbf y
=
\hat{\mathbf y}^T\hat{\mathbf y}
+
\mathbf e^T\mathbf e
}
$$

이다. 따라서

$$
\boxed{SST_U=SSR_U+SSE}
$$

가 성립한다.

**회귀제곱합 $SSR$**

회귀제곱합은 $SSR_U = \hat{\mathbf y}^T\hat{\mathbf y}$ 로 정의한다. 그런데 $\hat{\mathbf y}=X\hat{\boldsymbol\beta}$ 이므로 $SSR_U = (X\hat{\boldsymbol\beta})^T (X\hat{\boldsymbol\beta})= \hat{\boldsymbol\beta}^TX^TX \hat{\boldsymbol\beta}.$

정규방정식 $X^TX\hat{\boldsymbol\beta}=X^T\mathbf y$ 을 이용하면

$$
\boxed{
SSR_U
=
\hat{\boldsymbol\beta}^TX^T\mathbf y
}
$$

또한 $H=X(X^TX)^{-1}X^T, \quad H^T=H$이고 $H^2=H$이므로

$$
\begin{aligned}
SSR_U
&=
\hat{\mathbf y}^T\hat{\mathbf y}\\
&=
(H\mathbf y)^TH\mathbf y\\
&=
\mathbf y^TH^TH\mathbf y\\
&=
\mathbf y^TH\mathbf y.
\end{aligned}
$$

따라서 다음 식들은 모두 같다.

$$
\boxed{
SSR_U
=
\hat{\mathbf y}^T\hat{\mathbf y}
=
\hat{\boldsymbol\beta}^TX^T\mathbf y
=
\mathbf y^TH\mathbf y
}
$$

**잔차제곱합 $SSE$**

잔차제곱합은 $SSE = \mathbf e^T\mathbf e$ 이다. 제곱합 분해로부터 $SSE = SST_U-SSR_U$ 이므로

$$
SSE = \mathbf y^T\mathbf y - \hat{\boldsymbol\beta}^TX^T\mathbf y
$$

행렬 형태로 직접 유도할 수도 있다.

$$
\begin{aligned}
SSE
&=
\mathbf e^T\mathbf e\\
&=
\mathbf y^T(I_n-H)^T(I_n-H)\mathbf y.
\end{aligned}
$$

$H$는 대칭이고 멱등이므로 $(I_n-H)^T(I_n-H) = (I_n-H)^2 = I_n-H$ 이다. 따라서

$$
\boxed{
SSE
=
\mathbf y^T(I_n-H)\mathbf y
}
$$

이다.


**전체 자유도**:  
비중심 총제곱합은 $SST_U=\sum_{i=1}^ny_i^2$ 이다. 중심화된 총제곱합에서는 $\bar y$ 하나를 추정하므로 자유도 하나가 소모되어 $n-1$이 된다. 하지만 비중심 총제곱합에서는 평균을 추정하거나 중심화하지 않는다.  
따라서 $\boxed{\text{df}(SST_U)=n}$

**회귀 자유도**:  
$H$는 $X$의 열공간으로 투영하는 행렬이다. $X$의 계수가 $p$개이고 $\text{rank}(X)=p$이므로 $\text{rank}(H)=p.$  
투영행렬은 고윳값이 $0$ 또는 $1$이므로 $\text{tr}(H)=\text{rank}(H)=p.$  
따라서 $\boxed{\text{df}(SSR_U)=p}$

**잔차 자유도**:  
잔차는 $I_n-H$에 의해 만들어진다. 따라서 $\text{rank}(I_n-H) = \text{tr}(I_n-H) = n-\text{tr}(H) = n-p.$  
그러므로 $\boxed{\text{df}(SSE)=n-p}$

결국 자유도 역시

$$
\boxed{n=p+(n-p)}
$$

로 분해된다.

**4.6.3 분산분석표**  

| 변동 요인 |                                                             제곱합 |   자유도 |            평균제곱 |
| ----- | --------------------------------------------------------------: | ----: | --------------: |
| 회귀    |                    $SSR_U=\hat{\boldsymbol\beta}^TX^T\mathbf y$ |   $p$ |   $MSR=SSR_U/p$ |
| 잔차    | $SSE=\mathbf y^T\mathbf y-\hat{\boldsymbol\beta}^TX^T\mathbf y$ | $n-p$ | $MSE=SSE/(n-p)$ |
| 전체    |                                    $SST_U=\mathbf y^T\mathbf y$ |   $n$ |               — |

**$MSE$ 식 유도**  
$SSE = \mathbf y^T(I_n-H)\mathbf y$ 이다. 이차형식의 기대값 공식에 따라 $E(\mathbf y^TA\mathbf y) =\sigma^2\text{tr}(A)+E(\mathbf y)^TAE(\mathbf y)$ 이다.

여기서 $A=I_n-H, \quad E(\mathbf y)=X\boldsymbol\beta$ 로 놓으면

$$
\begin{aligned}
E(SSE)
&= \sigma^2\text{tr}(I_n-H) + \boldsymbol\beta^TX^T(I_n-H)X\boldsymbol\beta.
\end{aligned}
$$

그런데 $HX = X(X^TX)^{-1}X^TX = X$ 이므로 $(I_n-H)X=0.$ 따라서 $X^T(I_n-H)X=0$ 이고,  
$E(SSE) = \sigma^2\text{tr}(I_n-H) = (n-p)\sigma^2.$

그러므로

$$
\boxed{
E(MSE) = E\left(\frac{SSE}{n-p}\right) = \sigma^2}
$$

**$F$ 검정통계량의 유도**  
전체 회귀 유의성 검정의 귀무가설은 $H_0:\beta_1=\cdots=\beta_p=0$ 이다. 절편이 없으므로 이는 벡터 형태로 $H_0:\boldsymbol\beta=\mathbf0$ 과 같다.

추가로 정규오차를 가정한다: $\boldsymbol\varepsilon \sim N_n(\mathbf0,\sigma^2I_n).$

귀무가설 아래에서는 $\mathbf y=\boldsymbol\varepsilon \sim N_n(\mathbf0,\sigma^2I_n).$

그리고 $SSR_U=\mathbf y^TH\mathbf y, \quad SSE=\mathbf y^T(I_n-H)\mathbf y.$

$H$와 $I_n-H$는 각각 계수가 $p$, $n-p$인 대칭 멱등행렬이다. 따라서 코크런 정리에 의해

$$
\frac{SSR_U}{\sigma^2} \sim\chi_p^2
$$

이고

$$
\frac{SSE}{\sigma^2}\sim\chi_{n-p}^2
$$

이다.

또한 $H(I_n-H)=H-H^2=0$ 이므로 두 이차형식은 서로 독립이다. 따라서

$$
\frac{SSR_U/(p\sigma^2)}
{SSE/\{(n-p)\sigma^2\}}
\sim F(p,n-p).
$$

즉,

$$
\boxed{
F_0
=
\frac{MSR}{MSE}
=
\frac{SSR_U/p}{SSE/(n-p)}
\sim F(p,n-p)
}
$$

이다.

단, 이 분포는 반드시 다음 두 조건 아래에서 성립한다.

$$
H_0:\boldsymbol\beta=\mathbf0,
\qquad
\boldsymbol\varepsilon\sim N_n(\mathbf0,\sigma^2I_n).
$$

대립가설 아래에서는 중심 $F$분포가 아니라 비중심 $F$분포를 따른다.

**4.6.4 결정계수**  

절편 없는 모형의 비중심 결정계수는

$$
R_U^2 = \frac{SSR_U}{SST_U}
$$

로 정의된다. 따라서

$$
\boxed{
R_U^2
= \frac{\hat{\boldsymbol\beta}^TX^T\mathbf y} {\mathbf y^T\mathbf y}
= 1-\frac{SSE}{\mathbf y^T\mathbf y}
}
$$


**4.6.4 예측값의 분산**  

새로운 설명변수 벡터를

$$
\mathbf x_0=
\begin{pmatrix}
x_{01}\\
\vdots\\
x_{0p}
\end{pmatrix}
$$

라고 하자. 절편이 없으므로 $\mathbf x_0$에는 첫 번째 성분 $1$이 들어가지 않는다.

새로운 지점에서 조건부 평균은 $E(Y_0\mid\mathbf x_0) = \mathbf x_0^T\boldsymbol\beta$ 이고, 그 추정값은 $\hat y_0 = \mathbf x_0^T\hat{\boldsymbol\beta}$ 이다.

먼저 $\hat{\boldsymbol\beta} = \boldsymbol\beta + (X^TX)^{-1}X^T\boldsymbol\varepsilon$ 이므로

$$
\begin{aligned}
\text{Var}(\hat{\boldsymbol\beta})
&=
(X^TX)^{-1}X^T
\text{Var}(\boldsymbol\varepsilon)
X(X^TX)^{-1}\\
&=
(X^TX)^{-1}X^T
(\sigma^2I_n)
X(X^TX)^{-1}\\
&=
\sigma^2(X^TX)^{-1}.
\end{aligned}
$$

따라서 평균반응 추정값의 분산은

$$
\begin{aligned}
\text{Var}(\hat y_0)
&=
\text{Var}
(\mathbf x_0^T\hat{\boldsymbol\beta})\\
&=
\mathbf x_0^T
\text{Var}(\hat{\boldsymbol\beta})
\mathbf x_0\\
&=\boxed{
\sigma^2\mathbf x_0^T(X^TX)^{-1}\mathbf x_0
}
\end{aligned}
$$

이는 새로운 지점의 평균반응을 추정할 때의 분산이다.

반면 새로운 실제 관측값 $Y_0=\mathbf x_0^T\boldsymbol\beta+\varepsilon_0$ 을 예측할 때는 새로운 오차 $\varepsilon_0$의 분산도 포함해야 한다. 따라서 예측오차의 분산은

$$
\boxed{
\text{Var}(Y_0-\hat y_0)
=
\sigma^2
\left[
1+\mathbf x_0^T(X^TX)^{-1}\mathbf x_0
\right]
}
$$

이다.

>결론적으로 절편 없는 중회귀모형의 모든 공식은 단순회귀에서 OLS를 구한 과정, 즉
>
>$$
>\text{최소제곱화}
>\longrightarrow
>\text{정규방정식}
>\longrightarrow
>\text{직교투영}
>\longrightarrow
>\text{제곱합 분해}
>$$
>
>를 $p$차원 행렬로 확장한 결과이다. 다만 절편이 없기 때문에 총제곱합의 기준점이 표본평균 $\bar y$가 아니라 원점 $0$으로 바뀌며, 이에 따라 전체 자유도가 $n-1$에서 $n$으로 바뀐다.

## 4.7 제곱합의 분포 (Distribution of Sum of Squares)
분산분석표에 나타나는 제곱합(SST, SSR, SSE)의 분포를 이론적으로 분석하고, 왜 $F_0$가 $F$분포를 따르는지 설명한다.

**기본 가정**  
중회귀모형 $y = X\beta + \varepsilon, \quad \varepsilon \sim N(0_n, \sigma^2 I_n)$ 즉,

$$y \sim N(X\beta, \sigma^2 I_n)$$

**SST 분포** 

정리 3.3과 밑에 '단순회귀 예시' 참고.  

$SST = y^T\left(I_n - \frac{J_n}{n}\right)y$에서 행렬 $A = I_n - \frac{J_n}{n}$는 대칭이고 멱등이며 계수(rank) = $n-1$이므로, 

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
두 $y$의 이차형식인 $\frac{SSR}{\sigma^2}$과 $\frac{SSE}{\sigma^2}$가 서로 독립인지 확인해보자. 정리 3.6을 사용한다.

$$\mathbf{y} \sim N(X\beta, \sigma^2 I_n), \quad
\begin{pmatrix}\frac{SSR}{\sigma^2} \\ \frac{SSE}{\sigma^2}\end{pmatrix} = \begin{pmatrix}\mathbf{y}^T A \mathbf{y} \\ \mathbf{y}^T B \mathbf{y}\end{pmatrix} \\
$$

여기서

$$A=\frac{X(X^TX)^{-1}X^T - J/n}{\sigma^2}, \quad B=\frac{I - X(X^TX)^{-1}X^T}{\sigma^2}$$

라 하면 필요충분조건인

$$AVB = A\sigma^2 I B = \mathbf{0}_{n \times n}$$

이 성립한다. 따라서 SSR과 SSE는 독립이다.

**7. F-통계량의 분포**  
noncentral F distrubution이론에서 논의된 바와 같이 $\frac{SSR}{\sigma^2} \sim \chi^2(p,\lambda)$이고 $\frac{SSE}{\sigma^2} \sim \chi^2(n-p-1)$이며 두 카이제곱이 서로 독립이므로,

$$F_0 = \frac{MSR}{MSE} = \frac{(SSR/p)}{(SSE/(n-p-1))}$$

는 비중심 F분포(noncentral F distribution)를 따르고,

$$F_0 \sim F(p,n-p-1,\lambda)$$

여기서

$$\lambda = \frac{\beta^T X^T (I-\frac{J}{n}) X\beta}{2\sigma^2}$$

**귀무가설 하에서** 

설명변수들과 반응변수 간 관계를 설명하는데 중회귀모형이 의미가 없다: 

$$H_0:\beta_1=\cdots=\beta_p=0$$

귀무가설이 성립하면 $\lambda=0$이므로

$$F_0\sim F(p,n-p-1)$$

이 된다. 이것이 분산분석 F-검정의 이론적 근거이다.

## 4.8 변수의 직교화 (Orthogonalization of Variables)

변수의 직교화는 중회귀모형에서 특정 설명변수의 부분효과를 이해하기 위한 방법이다. 핵심은 이미 고려한 변수 $X_1$에 의해 설명되는 부분을 $\mathbf y$와 $X_2$ 양쪽에서 제거한 뒤, 남은 부분끼리 회귀하는 것이다.

이 결과를 Frisch–Waugh–Lovell 정리라고 한다.

단, 여기서 말하는 “효과”는 회귀계수가 나타내는 선형 조건부 관계이다. 별도의 인과적 가정이 없다면 인과효과를 의미하지 않는다.

### 4.8.1 단순회귀와 중회귀의 차이

절편이 포함된 단순선형회귀모형 $y_i=\beta_0+\beta_1x_i+\varepsilon_i$ 에서 기울기의 최소제곱추정량은 $\hat\beta_1 =r_{xy}\frac{s_y}{s_x}$. 즉, 단순회귀에서는 $\hat\beta_1$의 부호와 표본상관계수 $r_{xy}$의 부호가 같으며, 두 값은 표준편차의 비율을 통해 직접 연결된다.

하지만 중회귀에서는 다른 설명변수들을 통제해야 하므로, $\hat\beta_j$는 일반적으로 $x_j$와 $y$ 사이의 단순상관계수만으로 결정되지 않는다. 중회귀계수는 다른 변수들의 선형효과를 제거한 후 남는 관계를 나타낸다.

**1. 모형의 분할**

절편과 $p$개의 설명변수를 포함하는 중회귀모형을 다음과 같이 두 부분으로 나눈다: 

$$\mathbf y=X_1\boldsymbol\beta_1 + X_2\boldsymbol\beta_2\boldsymbol\varepsilon \\
\mathbf y=
\begin{pmatrix}
y_1\\
\vdots\\
y_n
\end{pmatrix}
\in\mathbb R^n, \quad
X_1\in\mathbb R^{n\times(q+1)}, \quad
X_2\in\mathbb R^{n\times(p-q)}
$$

$X_1$은 절편과 이미 고려된 $q$개의 설명변수를 포함한다.

$$
X_1=
\begin{pmatrix}
1&x_{11}&\cdots&x_{1q}\\
1&x_{21}&\cdots&x_{2q}\\
\vdots&\vdots&&\vdots\\
1&x_{n1}&\cdots&x_{nq}
\end{pmatrix}
$$

$X_2$는 추가로 고려하려는 $p-q$개의 설명변수를 포함한다.

$$
X_2=
\begin{pmatrix}
x_{1,q+1}&\cdots&x_{1p}\\
x_{2,q+1}&\cdots&x_{2p}\\
\vdots&&\vdots\\
x_{n,q+1}&\cdots&x_{np}
\end{pmatrix}
$$

$i$번째 관측값에 대해서는

$$
y_i = \mathbf x_{1i}^T\boldsymbol\beta_1 + \mathbf x_{2i}^T\boldsymbol\beta_2 + \varepsilon_i \\
\mathbf x_{1i}^T
=
(1,x_{i1},\ldots,x_{iq}),
\qquad
\mathbf x_{2i}^T
=
(x_{i,q+1},\ldots,x_{ip})
$$

이때, $X_1^TX_1$ 및 $X_2^T(I_n-H_1)X_2$ 가 가역행렬이라고 가정한다. 이는 $X_1$ 자체에 완전한 다중공선성이 없고, $X_1$의 영향을 제거한 후에도 $X_2$에 독립적인 정보가 남아 있음을 의미한다.

그리고 $X_1$에 대한 투영행렬과 잔차생성행렬을 정의한다: 

- $H_1 = X_1(X_1^TX_1)^{-1}X_1^T$
  - $X_1$의 열공간으로 투영하는 행렬로 정의
  - 대칭행렬이자 멱등행렬
- $M_1=I_n-H_1$
  - $X_1$의 선형효과를 제거하는 잔차생성행렬 이라고 정의
  - $M_1$도 대칭행렬이자 멱등행렬
  - $M_1X_1 = (I_n-H_1)X_1 = X_1-H_1X_1 = X_1-X_1 = O$. 즉, $M_1$을 곱하면 $X_1$의 열공간에 속하는 성분이 제거된다.

이 가정하에 다음의 단계적 절차를 생각해보자.

**2. 단계별 직교화**

- [1단계: $\mathbf y$에서 $X_1$의 선형효과 제거]

먼저 $\mathbf y$를 $X_1$에 회귀한다: $\mathbf y=X_1\boldsymbol\alpha_1+\mathbf e_1$  
OLS 추정량은 $\hat{\boldsymbol\alpha}_1 = (X_1^TX_1)^{-1}X_1^T\mathbf y$ 이고, 예측값은 $\hat{\mathbf y}_1 = X_1\hat{\boldsymbol\alpha}_1 = H_1\mathbf y$ 이다. 따라서 잔차는 $\boxed{\mathbf e_1 = \mathbf y-\hat{\mathbf y}_1 = (I_n-H_1)\mathbf y = M_1\mathbf y }$

$\mathbf e_1$은 $\mathbf y$에서 $X_1$로 설명할 수 있는 선형성분을 제거하고 남은 부분이다.  
또한 $X_1^T\mathbf e_1 = X_1^TM_1\mathbf y = O$ 이므로 $\mathbf e_1$은 $X_1$의 모든 열과 직교한다.

- [2단계: $X_2$에서 $X_1$의 선형효과 제거]

이번에는 $X_2$의 각 열을 $X_1$에 회귀한다: $X_2=X_1\Gamma+U$

$\Gamma$의 OLS 추정량은 $\hat\Gamma=(X_1^TX_1)^{-1}X_1^TX_2$ 이다.  
따라서 $X_1$에 의해 설명되는 $X_2$의 부분은 $\widehat X_2=X_1\hat\Gamma=H_1X_2$ 이다.

$X_1$의 영향을 제거한 $X_2$의 잔차행렬을 $X_{2\cdot1}=X_2-\widehat X_2$ 라고 정의하면 (즉, $X_1$을 통제한 후의 $X_2$)

$$
\boxed{X_{2\cdot1}= (I_n-H_1)X_2=M_1X_2}
$$

$X_1^TX_{2\cdot1} = X_1^TM_1X_2=O$ 이다. 따라서 $X_{2\cdot1}$의 모든 열은 $X_1$의 모든 열과 직교한다.

- [3단계: 잔차를 잔차화된 설명변수에 회귀]

이제 1단계 결과인 잔차 $\mathbf e_1$을, 2단계 결과인 $X_{2\cdot1}$에 회귀한다: $\mathbf e_1 = X_{2\cdot1}\boldsymbol\alpha_2+\mathbf u$

OLS 추정량은 $\hat{\boldsymbol\alpha}_2 = (X_{2\cdot1}^TX_{2\cdot1})^{-1} X_{2\cdot1}^T\mathbf e_1$ 이다.  
$\mathbf e_1=M_1\mathbf y$와 $X_{2\cdot1}=M_1X_2$를 대입하면 $\hat{\boldsymbol\alpha}_2 = (X_2^TM_1^TM_1X_2)^{-1} X_2^TM_1^TM_1\mathbf y$  
$M_1$은 대칭이고 멱등이므로

$$
\boxed{
\hat{\boldsymbol\alpha}_2 = (X_2^TM_1X_2)^{-1} X_2^TM_1\mathbf y
}
$$

**3. $\hat{\boldsymbol\alpha}_2=\hat{\boldsymbol\beta}_2$의 증명**

전체 중회귀모형의 잔차제곱합은

$$
SSE(\boldsymbol\beta_1,\boldsymbol\beta_2)
=
\left(\mathbf y-X_1\boldsymbol\beta_1-X_2\boldsymbol\beta_2\right)^T
\left(\mathbf y-X_1\boldsymbol\beta_1-X_2\boldsymbol\beta_2\right)
$$

이를 $\boldsymbol\beta_1$과 $\boldsymbol\beta_2$에 관해 미분하면 블록 정규방정식을 얻는다.

$$
X_1^T \left(\mathbf y-X_1\hat{\boldsymbol\beta}_1 -X_2\hat{\boldsymbol\beta}_2 \right) = \mathbf0 \\
X_2^T
\left(\mathbf y-X_1\hat{\boldsymbol\beta}_1 -X_2\hat{\boldsymbol\beta}_2 \right) = \mathbf0
$$

첫 번째 정규방정식에서 $X_1^T X_1\hat{\boldsymbol\beta}_1 = X_1^T \left(\mathbf y-X_2\hat{\boldsymbol\beta}_2\right)$ 이므로 $\hat{\boldsymbol\beta}_1 = (X_1^TX_1)^{-1}X_1^T \left(\mathbf y-X_2\hat{\boldsymbol\beta}_2\right)$. 이를 두 번째 정규방정식에 대입한다.

$$
X_2^T
\left[
\mathbf y - X_1(X_1^TX_1)^{-1}X_1^T \left(\mathbf y-X_2\hat{\boldsymbol\beta}_2\right) - X_2\hat{\boldsymbol\beta}_2
\right]
= \mathbf0
$$

$H_1=X_1(X_1^TX_1)^{-1}X_1^T$를 사용하면

$$
X_2^T
\left[
\mathbf y-H_1\mathbf y +H_1X_2\hat{\boldsymbol\beta}_2 -X_2\hat{\boldsymbol\beta}_2
\right]
= \mathbf0
$$

이다. 괄호 안을 정리하면 $X_2^T \left[ M_1\mathbf y-M_1X_2\hat{\boldsymbol\beta}_2 \right] = \mathbf0$ 이므로 $X_2^TM_1X_2\hat{\boldsymbol\beta}_2 = X_2^TM_1\mathbf y$ 이다. 따라서

$$
\boxed{
\hat{\boldsymbol\beta}_2 = (X_2^TM_1X_2)^{-1} X_2^TM_1\mathbf y
}
$$

앞에서 구한 $\hat{\boldsymbol\alpha}_2$와 비교하면 같음을 확인할 수 있다!  
이것이 Frisch–Waugh–Lovell 정리이다.

**4. 전체 모형과 단계별 회귀의 잔차가 같은 이유**

전체 모형의 예측값은 $\hat{\mathbf y} = X_1\hat{\boldsymbol\beta}_1 + X_2\hat{\boldsymbol\beta}_2$ 이다. 앞에서 구한 $\hat{\boldsymbol\beta}_1 = (X_1^TX_1)^{-1}X_1^T (\mathbf y-X_2\hat{\boldsymbol\beta}_2)$ 를 이용하면

$$
X_1\hat{\boldsymbol\beta}_1 = H_1\mathbf y-H_1X_2\hat{\boldsymbol\beta}_2
$$

이다. 따라서

$$
\begin{aligned}
\hat{\mathbf y}
&= H_1\mathbf y-H_1X_2\hat{\boldsymbol\beta}_2 +X_2\hat{\boldsymbol\beta}_2\\
&= H_1\mathbf y + (I_n-H_1)X_2\hat{\boldsymbol\beta}_2\\
&= H_1\mathbf y +  X_{2\cdot1}\hat{\boldsymbol\beta}_2.
\end{aligned}
$$

따라서 전체 모형의 잔차는

$$
\begin{aligned}
\hat{\boldsymbol\varepsilon}
&= \mathbf y-\hat{\mathbf y}\\
&= \mathbf y-H_1\mathbf y -X_{2\cdot1}\hat{\boldsymbol\beta}_2\\
&= \mathbf e_1 -X_{2\cdot1}\hat{\boldsymbol\alpha}_2.
\end{aligned}
$$

즉,

$$
\boxed{
\mathbf y-X_1\hat{\boldsymbol\beta}_1-X_2\hat{\boldsymbol\beta}_2
= \mathbf e_1-X_{2\cdot1}\hat{\boldsymbol\alpha}_2
}
$$

따라서 전체 모형을 한 번에 적합하든, 직교화한 후 단계적으로 적합하든 다음이 모두 같다.

* $X_2$에 대한 회귀계수
* 최종 예측값
* 최종 잔차
* 잔차제곱합

단, $X_1$의 계수 $\hat{\boldsymbol\beta}_1$은 1단계에서 $\mathbf y$를 $X_1$에만 회귀하여 얻은 $\hat{\boldsymbol\alpha}_1$과 일반적으로 같지 않다.

$$
\hat{\boldsymbol\alpha}_1 = (X_1^TX_1)^{-1}X_1^T\mathbf y \\
\hat{\boldsymbol\beta}_1 = (X_1^TX_1)^{-1}X_1^T \left(\mathbf y-X_2\hat{\boldsymbol\beta}_2\right)
$$

**5. 추가 설명변수가 하나인 경우**

$X_2$가 하나의 설명변수 $\mathbf x_2$만 포함한다고 하자. 그러면 $\mathbf x_{2\cdot1}= M_1\mathbf x_2$ 이고, $\mathbf y_{\cdot1} = M_1\mathbf y$ 라고 쓸 수 있다.

이때 $\beta_2$의 OLS 추정량은

$$
\boxed{
\hat\beta_2
= \frac{\mathbf x_{2\cdot1}^T\mathbf y_{\cdot1}}{\mathbf x_{2\cdot1}^T\mathbf x_{2\cdot1}}
}
$$

성분별로 쓰면

$$
\hat\beta_2
=
\frac{\sum_{i=1}^n x_{2\cdot1,i}y_{\cdot1,i}}
{\sum_{i=1}^n x_{2\cdot1,i}^2}
$$

이는 절편 없는 단순회귀의 기울기 공식과 정확히 같은 형태이다. 다만 원래 변수 $\mathbf x_2$와 $\mathbf y$를 사용하는 것이 아니라, 양쪽에서 $X_1$의 선형효과를 제거한 잔차들을 사용한다.

$X_1$에 절편이 포함되어 있으므로 잔차들의 평균은 $0$이다.

$$
\sum_{i=1}^n x_{2\cdot1,i}=0, \quad \sum_{i=1}^n y_{\cdot1,i}=0
$$

따라서 부분상관계수를

$$
r_{y2\cdot1}
=
\frac{
\sum_{i=1}^n
y_{\cdot1,i}x_{2\cdot1,i}
}{
\sqrt{
\sum_{i=1}^n y_{\cdot1,i}^2
\sum_{i=1}^n x_{2\cdot1,i}^2
}
}
$$

라고 하면

$$
\boxed{
\hat\beta_2 = r_{y2\cdot1} \frac{s_{y\cdot1}}{s_{x_2\cdot1}}
}
$$

즉, 중회귀계수는 원래 변수 사이의 단순상관계수와 연결되는 것이 아니라, 다른 변수들의 효과를 제거한 잔차 사이의 부분상관관계와 연결된다.

**6. 다중공선성과의 관계**

오차분산이 $\text{Var}(\boldsymbol\varepsilon) = \sigma^2I_n$ 일 때 전체 모형의 $\hat{\boldsymbol\beta}_2$의 분산은

$$
\boxed{
\text{Var}(\hat{\boldsymbol\beta}_2)
= \sigma^2 \left[ X_2^TM_1X_2 \right]^{-1}
}
$$

이다. 그런데 $X_2^TM_1X_2 = X_{2\cdot1}^TX_{2\cdot1}$ 이므로 $\text{Var}(\hat{\boldsymbol\beta}_2) = \sigma^2 \left[ X_{2\cdot1}^TX_{2\cdot1} \right]^{-1}$ 이다.

만약 $X_2$가 $X_1$에 의해 거의 완전히 설명된다면 $X_{2\cdot1}=M_1X_2$ 가 거의 영행렬이 된다. 그러면 $X_{2\cdot1}^TX_{2\cdot1}$가 특이행렬에 가까워지고 그 역행렬의 성분이 커진다. 따라서 $\hat{\boldsymbol\beta}_2$의 분산과 표준오차가 커진다.

하나의 추가 설명변수 $\mathbf x_2$만 있는 경우에는

$$
\text{Var}(\hat\beta_2)
=
\frac{\sigma^2}
{\mathbf x_{2\cdot1}^T\mathbf x_{2\cdot1}}
$$

이다.

$\mathbf x_2$를 $X_1$에 회귀했을 때의 결정계수를 $R_2^2$라고 하면

$$
\mathbf x_{2\cdot1}^T\mathbf x_{2\cdot1}
=
(1-R_2^2)
\sum_{i=1}^n(x_{i2}-\bar x_2)^2
$$

이므로

$$
\boxed{
\text{Var}(\hat\beta_2)
=
\frac{\sigma^2}
{
(1-R_2^2)
\sum_{i=1}^n(x_{i2}-\bar x_2)^2
}
}
$$

$R_2^2$가 $1$에 가까울수록 $\mathbf x_2$가 $X_1$에 의해 거의 설명되므로 분모가 작아지고 $\hat\beta_2$의 분산은 커진다. 이때 $\frac{1}{1-R_2^2}$ 가 분산팽창계수인 VIF이다.

**7. 최종 해석**

Frisch–Waugh–Lovell 정리에 따라

$$
\boxed{
\hat{\boldsymbol\beta}_2
=
\left[
X_2^T(I_n-H_1)X_2
\right]^{-1}
X_2^T(I_n-H_1)\mathbf y
}
$$

이를 단계적으로 해석하면 다음과 같다.

1. $\mathbf y$에서 $X_1$로 선형적으로 설명되는 부분을 제거한다.

$$
\mathbf y_{\cdot1}=M_1\mathbf y
$$

2. $X_2$에서도 $X_1$로 선형적으로 설명되는 부분을 제거한다.

$$
X_{2\cdot1}=M_1X_2
$$

3. 남은 두 부분을 서로 회귀한다.

$$
\mathbf y_{\cdot1}
=
X_{2\cdot1}\boldsymbol\beta_2+\text{잔차}
$$

그 결과 얻는 회귀계수는 원래 전체 중회귀모형의 $X_2$ 계수와 정확히 같다.

$$
\boxed{
\hat{\boldsymbol\alpha}_2
=
\hat{\boldsymbol\beta}_2
}
$$

따라서 $\hat{\boldsymbol\beta}_2$는 $X_1$을 일정하게 통제했을 때 $X_2$와 $\mathbf y$ 사이에 남아 있는 선형관계를 나타낸다.

다만 $X_2$가 여러 개의 변수를 포함하는 경우, $\hat{\boldsymbol\beta}_2$는 $X_2$ 변수들을 하나씩 독립적으로 회귀한 결과가 아니라 $X_2$의 모든 변수를 동시에 포함하여 얻은 계수벡터이다. 따라서 $\boldsymbol\beta_2$의 각 성분은 $X_1$뿐 아니라 $X_2$에 포함된 다른 변수들도 통제한 부분회귀계수로 해석해야 한다.
