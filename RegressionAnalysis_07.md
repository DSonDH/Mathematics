# Chapter 7 다항회귀 (Polynomial Regression)

## 7.1 다항회귀모형 (Polynomial Regression Model)
설명변수와 반응변수 사이의 관계가 직선(linear relationship)이 아니라 **곡선(curvilinear relationship)** 형태일 때 사용하는 회귀모형이다.
단순회귀(simple regression)가 직선 관계를 가정한다면, 다항회귀(polynomial regression)는 설명변수의 거듭제곱 항을 포함하여 곡선 형태의 관계를 설명한다.  예를 들어, 산점도(scatter diagram)를 확인했을 때 다음과 같은 특징이 나타날 수 있다.
* 증가 후 감소하는 포물선 형태
* 감소하다가 완만해지는 곡선 형태

이 경우 단순선형회귀(simple linear regression)는 적절하지 않으며 **다항항(polynomial term)** 을 포함한 회귀모형을 고려해야 한다. (quadratic regression, cubic regression 등)

### 7.1.1 1변수 다항회귀모형 (Polynomial Regression with One Independent Variable)

설명변수가 하나인 경우 가장 기본적인 다항회귀모형은 다음과 같다.

**이차 다항회귀모형 (Second-Order Polynomial Regression)**

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \epsilon$$

* $\beta_0$ : 절편(intercept)
* $\beta_1$ : 1차 효과(linear effect)
* $\beta_2$ : 곡률(curvature)을 나타내는 2차 효과
* $\epsilon$ : 오차항(error term)

이 모형은 **포물선(parabola)** 형태의 관계를 설명한다.

**삼차 다항회귀모형 (Third-Order Polynomial Regression)**

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \epsilon$$

일반적으로 설명변수가 하나일 때 $k$차 다항회귀모형은 다음과 같이 표현한다.

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_k x^k + \epsilon$$

이를 **k차 다항회귀모형 (k-th order polynomial regression model)** 이라고 한다.

### 7.1.2 다항회귀의 선형회귀 표현 (Linear Representation of Polynomial Regression)
다항회귀는 이름과 달리 **모수(parameter)에 대해 선형(linear in parameters)** 이다.
따라서 **중회귀분석(multiple regression)** 형태로 변환하여 분석할 수 있다.  
예를 들어 $y = \beta_0 + \beta_1 x + \beta_2 x^2 + \epsilon$ 에서

$$x_1 = x,\quad x_2 = x^2$$

라고 두면

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \epsilon$$

가 되어 **2개의 설명변수를 가진 중회귀모형**이 된다.  

$$X = \begin{pmatrix}
1 & x_1 & x_1^2 \\
1 & x_2 & x_2^2 \\
\vdots & \vdots & \vdots \\
1 & x_n & x_n^2
\end{pmatrix}$$

이고, $X\top X, X\top y$는 각각

$$
X\top X = \begin{pmatrix}
n & \sum x_i & \sum x_i^2 \\
\sum x_i & \sum x_i^2 & \sum x_i^3 \\
\sum x_i^2 & \sum x_i^3 & \sum x_i^4
\end{pmatrix}, \quad
X\top y = \begin{pmatrix}
\sum y_i \\
\sum x_i y_i \\
\sum x_i^2 y_i
\end{pmatrix}$$

이 되고, 회귀계수는 최소제곱법(least squares method)으로 추정한다.

$$\hat{\beta} = \begin{pmatrix}
\hat{\beta_0}\\
\hat{\beta_1}\\
\hat{\beta_2}
\end{pmatrix} = (X^TX)^{-1}X^Ty$$

#### 일반적인 k차 다항회귀

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_k x^k + \epsilon$$

을 다음과 같이 정의한다.

$$x_1 = x,\quad x_2 = x^2,\quad \cdots,\quad x_k = x^k$$

그러면

$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_k x_{ik} + \epsilon_i$$

이 되며 **일반적인 중회귀분석 방법을 그대로 적용**할 수 있다.

## 7.2 모형의 타당성 (Model Adequacy)
다항회귀모형에서 중요한 문제는 **적절한 차수(order) $k$** 를 결정하는 것이다.
차수가 너무 낮으면 실제 곡선 관계를 설명하지 못하고, 차수가 너무 높으면 **과적합(overfitting)** 문제가 발생할 수 있다.  
따라서 데이터에 대해 **가장 적절한 다항회귀 차수 $k$** 를 선택하는 과정이 필요하다.

### 7.2.1 설명변수가 하나인 경우 (One Independent Variable)
설명변수가 하나일 때 $k$차 다항회귀모형은 다음과 같다.

$$y = \beta_0 + \sum_{j=1}^{k} \beta_j x^j + \epsilon$$

* $k$ : 다항식의 차수(order)
* $x^j$ : $j$차 다항항
* $\beta_j$ : 회귀계수
* $\epsilon$ : 오차항

#### 차수 선택 방법
적절한 차수 $k$는 다음 방법을 통해 판단한다.
1. **산점도 분석 (Scatter Diagram Inspection)**
2. **잔차 분석 (Residual Analysis)**
3. **순차 F-검정 (Sequential F-test)**

여기서
#### 산점도를 이용한 판단
데이터 $(x_i, y_i)$의 산점도를 먼저 그려 보면 곡선 형태를 어느 정도 추정할 수 있다. 일반적으로 다음 기준을 따른다.
* 직선 형태 → $k = 1$
* 포물선 형태 → $k = 2$
* S형 곡선 → $k = 3$

하지만 산점도만으로 판단하기 어려운 경우에는 통계적 검정을 사용한다.
#### 잔차 분석 (Residual Analysis)
회귀모형을 적합한 후 **잔차(residuals)** 를 분석한다. 잔차 패턴이
* 랜덤 → 모형 적절
* 곡선 패턴 존재 → 더 높은 차수 필요
* 즉, 잔차에 구조가 보이면 모형이 부적절하다.

#### 순차 F-검정 (Sequential F-test)
차수를 하나씩 증가시키며 검정을 수행한다. 절차는 다음과 같다.

**1단계: 1차 모형 적합** 
 
$y_i = \beta_0 + \beta_1 x_i + \epsilon_i , \quad H_0 : \beta_1 = 0$

이 검정은 **회귀 ANOVA의 F-검정**과 동일하다.  
귀무가설이 기각되면 2단계로 넘어간다.  

**2단계: 2차 모형 적합** 
 
$y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \epsilon_i \quad H_0 : \beta_2 = 0$

이때 사용하는 통계량은 **추가제곱합(extra sum of squares)** 이다. $SS(\beta_2 | \beta_0,\beta_1)$ 이 값이 유의하면→ 이차항 필요  
귀무가설이 기각되면 3단계로 넘어간다.  

**3단계: 3차 모형** 
 
$y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \beta_3 x_i^3 + \epsilon_i \quad H_0 : \beta_3 = 0$

추가제곱합 $SS(\beta_3|\beta_0,\beta_1,\beta_2)$

**순차 F-test의 판단 기준**  
절차는 다음과 같다.
1. $k=1$ 모형 적합
2. $k=2$ 모형으로 확장
3. 추가항의 유의성 검정
4. 유의하면 다음 차수 고려
5. 유의하지 않으면 이전 차수 채택

즉,
* 어떤 단계에서 $H_0 : \beta_r = 0$을 **기각하지 못하면**
* **적절한 차수는 $r-1$** 이 된다.

### 7.2.2 설명변수가 둘 이상인 경우 (Multiple Independent Variables)
설명변수가 $p$개일 때 다항회귀모형은 다음과 같이 구성한다.

**1차 모형 (First-Order Model)** 
 
$$y = \beta_0 + \sum_{j=1}^{p} \beta_j x_j + \epsilon$$

**1차 모형의 평가**  
다음 기준으로 모형의 타당성을 평가한다.
1. **결정계수 $(R^2)$**
2. **ANOVA F-검정**
3. **잔차 분석**
4. **적합결여 검정 (Lack-of-fit test)**

이 검정을 통해
* 모형이 충분한지
* 더 높은 차수가 필요한지
판단한다.

**2차 모형 (Second-Order Model)**  
1차 모형이 충분하지 않으면 **이차모형(second-order model)** 을 고려한다.

$$y = \beta_0 + \sum_{j=1}^{p} \beta_j x_j + \sum_{j \le l} \beta_{jl} x_j x_l + \epsilon$$

이 모형에는 다음 항이 포함된다.
* 선형항(linear terms)
* 제곱항(quadratic terms)
* 교호작용항(interaction terms)

예를 들어 $p=2$이면

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_{11}x_1^2 + \beta_{22}x_2^2 + \beta_{12}x_1x_2 + \epsilon$$

**실제 분석에서의 차수 선택**  
실제 분석에서는 **1차 모형**, **2차 모형** 이 두 가지가 가장 많이 사용된다. 그 이유는,
1. 차수가 증가하면 변수 수가 급격히 증가한다.
2. 모형 해석이 어려워진다.
3. 다중공선성(multicollinearity)이 증가한다.

따라서 **3차 이상 모형은 거의 사용되지 않는다.**

**항 선택 문제 (Term Selection)**  
이차모형이 적합하다고 판단되어도 **모든 항을 반드시 포함할 필요는 없다.**  
예를 들어

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_{11}x_1^2 + \beta_{22}x_2^2 + \beta_{12}x_1x_2 + \epsilon$$

에서 부분 F-검정(partial F-test, ch06 참고)을 통해 $H_0 : \beta_{12}=0$을 기각하지 못하면  
→ **교호작용항 $(x_1x_2)$** 는 제거할 수 있다. 즉, $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_{11}x_1^2 + \beta_{22}x_2^2 + \epsilon$ 같이 더 단순한 모형이 더 적절할 수 있다.

>**변수 선택 문제**  
>어떤 항을 포함할지 결정하는 문제는 **변수 선택 (Selection of Variables)** 문제이다. 대표적인 방법은 다음과 같다.
>* 부분 F-검정
>* 단계적 회귀(stepwise regression)
>* 전진 선택(forward selection)
>* 후진 제거(backward elimination)
>
>이 주제는 이후 장에서 다시 다룬다.


## 7.3 직교다항회귀 (Orthogonal Polynomial Regression)
설명변수가 하나인 다항회귀모형은 일반적으로 다음과 같이 표현된다.

$$y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \cdots + \beta_k x_i^k + \epsilon_i, \quad i = 1,2,\ldots,n$$

이 모형은 **중회귀모형(multiple regression model)** 으로 취급하여 분석할 수 있다.
그러나 설명변수 $x$의 수준(level)이 **등간격(equally spaced)** 으로 배치되어 있을 경우에는 **직교다항식(orthogonal polynomials)** 을 사용하면 계산이 매우 단순해진다.

예를 들어 $x$ 값이 일정 간격 $d$로 배치되어 있다고 하자.

$$x_2 = x_1 + d, \quad x_3 = x_2 + d, \quad \ldots, \quad x_n = x_{n-1}+d$$

이때 기존의 $x^j$ 대신 **차수 $q$의 직교다항식 $(p_q(x))$** 을 사용하여 모형을 다음과 같이 표현한다.

$$y_i = \beta_0 + \beta_1 p_1(x_i) + \beta_2 p_2(x_i) + \cdots + \beta_k p_k(x_i) + \epsilon_i$$

여기서 $k < n$이다.

### 직교다항식의 조건 (Orthogonality Conditions)
직교다항식 $p_q(x)$는 다음 조건을 만족해야 한다.

**(1) 평균이 0**

$$\sum_{i=1}^{n} p_q(x_i) = 0, \quad q = 1,2,\ldots,k$$

**(2) 서로 직교**

$$\sum_{i=1}^{n} p_q(x_i)p_r(x_i) = 0, \quad q,r=1,2,\ldots,k, \quad q\neq r$$

즉 서로 다른 차수의 다항식은 **상관이 없다(orthogonal)**.

**(3) 제곱합은 0이 아님**

$$\sum_{i=1}^{n} p_q^2(x_i) \neq 0$$

### 행렬 표현 (Matrix Representation)
직교다항식을 사용한 회귀모형의 설계행렬 $(X)$는 다음과 같다.

$$X = \begin{pmatrix}
1 & p_1(x_1) & p_2(x_1) & \cdots & p_k(x_1) \\
1 & p_1(x_2) & p_2(x_2) & \cdots & p_k(x_2) \\
\vdots & \vdots & \vdots & & \vdots \\
1 & p_1(x_n) & p_2(x_n) & \cdots & p_k(x_n)
\end{pmatrix}$$

직교조건 때문에

$$X^TX = \begin{pmatrix}
n & 0 & 0 & \cdots & 0 \\
0 & \sum p_1^2(x_i) & 0 & \cdots & 0 \\
0 & 0 & \sum p_2^2(x_i) & \cdots & 0 \\
\vdots & \vdots & \vdots & & \vdots \\
0 & 0 & 0 & \cdots & \sum p_k^2(x_i)
\end{pmatrix}$$

이 되어 **대각행렬(diagonal matrix)** 이 된다.
이 때문에 계산이 매우 단순해진다.

### 최소제곱추정량 (Least Squares Estimator)
정규방정식 $X^TX\hat{\beta}=X^Ty$에서

$$X^Ty = \begin{pmatrix}
\sum y_i \\
\sum p_1(x_i)y_i \\
\sum p_2(x_i)y_i \\
\vdots \\
\sum p_k(x_i)y_i
\end{pmatrix}$$

이므로 최소제곱추정량은

$$\hat{\beta} = \begin{pmatrix}
\hat{\beta_0}\\
\hat{\beta_1}\\
\hat{\beta_2}\\
\vdots\\
\hat{\beta_k}
\end{pmatrix}$$

이며 각 계수는 다음과 같이 계산된다.

$$\hat{\beta_0} = \frac{\sum y_i}{n}\\
\hat{\beta_j} = \frac{\sum p_j(x_i)y_i}{\sum p_j^2(x_i)}, \quad j=1,2,\ldots,k$$

이러한 단순한 계산이 가능한 이유는 $X^TX$가 **대각행렬**이기 때문이다.

### 직교다항식의 형태 (Form of Orthogonal Polynomials)
설명변수 평균을 $\bar{x} = \frac{1}{n}\sum x_i$ 라 하고 $x$의 수준간 간격을 $d$라 하면 다음과 같은 직교다항식이 만들어진다.

**0차**

$$p_0(x) = 1$$

**1차**

$$p_1(x) = \frac{x-\bar{x}}{d}$$

**2차**

$$p_2(x) = \left(\frac{x-\bar{x}}{d}\right)^2 - \frac{n^2-1}{12}$$

**3차**

$$p_3(x) = \left(\frac{x-\bar{x}}{d}\right)^3 - \frac{3n^2-7}{20}\left(\frac{x-\bar{x}}{d}\right)$$

**일반식**

$$p_{r+1}(x) = p_r(x)p_1(x) - \frac{r^2(n^2-r^2)}{4(4r^2-1)}p_{r-1}(x), \quad r\ge1$$

### 분산분석 (ANOVA for Orthogonal Polynomial Regression)
총제곱합 $SST = y^Ty - n\bar{y}^2 = \sum y_i^2 - n(\frac{\sum y_i}{d})^2$  
회귀제곱합$SSR = \hat{\beta}^TX^Ty - n\bar{y}^2$  
직교성 때문에

$$SSR  = \sum_{j=1}^k \left[\frac{\sum_{i=1}^n p_j(x_i)y_i}{\sum_{i=1}^n p_j^2(x_i)}\cdot \sum_{i=1}^n p_j(x_i)y_i \right]$$

$X^TX$가 대각행렬이므로 각 column간에는 직교하는 성질이 있다. 따라서 추가제곱합에 관해서 $SS(\hat\beta_j \mid \hat\beta_0, \dots, \hat\beta_k)= SS(\hat\beta_j), \quad j=1,2, \dots, k$ 이다.  
또한 직교다항회귀의 $X^Ty, \hat\beta$ 행렬 원소로부터 

$$
SS(\hat\beta_0) = n(\bar y)^2\\
SS(\hat{\beta_j}) = \frac{\left(\sum p_j(x_i)y_i\right)^2}{\sum p_j^2(x_i)}, \quad j=1,2, \dots, k
$$

이므로

$$SSR= \sum_{j=1}^{k} SS(\hat{\beta_j})$$

**직교다항회귀의 ANOVA표**
| 요인  | 제곱합           | 자유도     | 평균제곱          | F                 | $F_{\alpha}$ |
| --- | ------------- | ------- | ------------- | ----------------- | ----------- |
| 회귀  | SSR           | k       | MSR           | MSR/MSE     | $F_\alpha(k, n-k-1)$         |
| 1차  | $SS(\hat{\beta_1})$ | 1       | $MS(\hat{\beta_1})$ | $MS(\hat{\beta_1})/MSE$ |     $F_\alpha(1, n-k-1)$         |
| 2차  | $SS(\hat{\beta_2})$ | 1       | $MS(\hat{\beta_2})$ | $MS(\hat{\beta_2})/MSE$ |     $F_\alpha(1, n-k-1)$         |
| ... | ...           | ...     | ...           | ...               |               |
| k차  | $SS(\hat{\beta_k})$ | 1       | $MS(\hat{\beta_k})$ | $MS(\hat{\beta_k})/MSE$ |     $F_\alpha(1, n-k-1)$         |
| 잔차  | SSE           | $n-k-1$ | MSE           |                   |               |
| 전체  | SST           | $n-1$   |               |                   |               |

### 차수 검정 (Order Testing)
각 차수에 대해 $F_0 = \frac{MS(\beta_j)}{MSE}$ 를 계산하여 $F_{(1,n-k-1)}$ 과 비교한다.
* $F_0$가 임계값보다 크면 해당 차수는 유의
* 작으면 해당 차수는 필요 없음

따라서
**가장 높은 유의한 차수까지 포함하는 모형을 선택한다.**

### 회귀계수의 분산 (Variance of Regression Coefficients)
중회귀모형의 일반 결과로부터 $Var(\hat{\beta}) = (X^TX)^{-1}\sigma^2$ 이므로,

$$\widehat{Var}(\hat{\beta_j}) = \frac{MSE}{\sum p_j^2(x_i)}$$

### 예측값의 분산 (Variance of Prediction)
예측값 $\hat{y} = (p_0(x),p_1(x),\ldots,p_k(x))\hat{\beta}$ 이므로, 직교다항식의 경우

$$Var(\hat{y}) = p(x)^T(X^TX)^{-1}p(x)\sigma^2 = \sum_{j=0}^{k} \frac{p_j^2(x)}{\sum_{i=1}^n p_j^2(x_i)}\sigma^2 \\
\widehat{Var}(\hat{y})  = \sum_{j=1}^{k} \frac{p_j^2(x)}{\sum_{i=1}^n p_j^2(x_i)}\cdot MSE$$

실제 계산에서는 $\sigma^2$ 대신 $MSE$를 사용한다.


## 7.4 조각다항회귀 (Piecewise Polynomial Regression)
설명변수 $(x)$의 변화에 따라 반응변수 $(y(x))$의 평균을

$$E[y(x)] = \eta(x)$$

라 하면 일반적인 다항회귀모형은 다음과 같다.

$$\eta(x) = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_k x^k \tag{7.16}$$

이 모형의 특징은 다음과 같다.
* $\eta(x)$ 는 **연속함수 (continuous function)** 이다.
* 회귀계수 $\beta_j$ 에 대해 **선형(linear)** 이다.
* 미분 $d\eta(x)/dx$ 역시 $x$에 대해 연속이다.

그러나 실제 자료에서는 전체 구간

$$a \le x \le b$$

에서 하나의 다항식이 항상 적절하지 않을 수 있다.
이 경우 구간을 여러 개로 나누어 각 구간에 다른 다항식을 적용하는 **조각다항회귀 (piecewise polynomial regression)** 를 사용할 수 있다.

### 조각다항회귀모형 (Piecewise Polynomial Regression Model)
설명변수 $x$의 변화에 따른 반응변수 $y(x)$를 연구할 때에는 전체 정의역에 대한 평균 $E[y(x)] = \eta(x)$를 가정하는 다항회귀모형이 많이 쓰인다. 하지만 실제 자료에서는 전체 구간에서 하나의 다항식이 적절하지 않을 수 있다. 즉 구간을 나눠서 각각 회귀모형을 적합하는 것이 더 적절할 수 있다.

구간을 다음과 같이 나눈다고 하자.

$$a \le x \le a_1,\quad a_1 \le x \le a_2,\quad \cdots,\quad a_{r-1} \le x \le b$$

이때 조각다항회귀모형은

$$\eta(x) = \begin{cases}
\beta_{10} + \beta_{11}x + \cdots + \beta_{1q_1}x^{q_1}, & a \le x \le a_1 \\
\beta_{20} + \beta_{21}x + \cdots + \beta_{2q_2}x^{q_2}, & a_1 \le x \le a_2 \\
\vdots \\
\beta_{r0} + \beta_{r1}x + \cdots + \beta_{rq_r}x^{q_r}, & a_{r-1} \le x \le b
\end{cases} \tag{7.17}$$

반응변수는

$$y = \eta(x) + \epsilon, \quad \epsilon \sim N(0,\sigma^2)$$

를 따른다고 가정한다.

**조각다항회귀가 필요한 경우 예시**  
예를 들어 어떤 회사의 **수출액 데이터**가 다음과 같다고 하자.

| 연도             | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 |
| -------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| $(x)$            | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    |
| $(y)$ (단위:100만불) | 1.0  | 1.2  | 1.3  | 1.5  | 1.7  | 2.5  | 3.4  | 4.2  |

산점도를 보면
* 2011–2015 : 완만한 증가
* 2015 이후 : 급격한 증가

**조각 1차 다항회귀**  
경계점 $x=4$ 라 하면

$$\eta(x)= \begin{cases}
\beta_{10} + \beta_{11}x, & 0 \le x \le 4 \\
\beta_{20} + \beta_{21}x, & 4 < x \le 8
\end{cases}$$

이때 $x=4$는 **불연속점 (disconnected point)** 이라고 한다.

>**스플라인 함수 (Spline Function)**  
>* 함수 $\eta(x)$
>* 미분 $d\eta(x)/dx$
>* 고차 미분
>
>이 모두 **연속**이면 이를 **스플라인 함수 (spline function)** 라고 한다.

#### 조각다항식을 하나의 식으로 표현

조각다항식은 다음 함수를 이용하여 하나의 식으로 표현할 수 있다.

- 기본 함수 $1,\ x,\ x^2,\ \ldots,\ x^q$  
- 추가 함수 $T_0(x-\alpha_l),\ T_1(x-\alpha_l),\ \ldots,\ T_q(x-\alpha_l)$
  - $q = \max(q_l)$

$$
T_j(x-\alpha_l) = \begin{cases}
0, & x \le \alpha_l \\
(x-\alpha_l)^j, & x>\alpha_l
\end{cases} \tag{7.18}
$$

**예: 불연속점이 하나인 경우**

구간 $a \le x \le b$ 에서 불연속점이 $x=\alpha$라 하고 2차 다항식을 사용하면

$$\eta(x)= \begin{cases}
\beta_{10} + \beta_{11}x + \beta_{12}x^2, & a \le x \le \alpha \\
\beta_{20} + \beta_{21}x + \beta_{22}x^2, & \alpha < x \le b
\end{cases} \tag{7.19}$$

이를 하나의 식으로 쓰면

$$\eta(x) = \beta_0 + \beta_1x + \beta_2x^2 + \beta_3T_0(x-\alpha) + \beta_4T_1(x-\alpha) + \beta_5T_2(x-\alpha) \tag{7.20}$$

- $\beta_0 = \beta_{10}$
- $\beta_1 = \beta_{11}$
- $\beta_2 = \beta_{12}$
- $\beta_3 = \alpha^2(\beta_{22} - \beta_{12}) + \alpha(\beta_{21} - \beta_{11}) + (\beta_{20} - \beta_{10})$
- $\beta_4 = 2\alpha(\beta_{22} - \beta_{12}) + (\beta_{21} - \beta_{11})$
- $\beta_5 = \beta_{22} - \beta_{12}$

**연속 조건**  
만약 $x=\alpha$에서 함수가 연속이면

$$\beta_{10} + \beta_{11}\alpha + \beta_{12}\alpha^2 = \beta_{20} + \beta_{21}\alpha + \beta_{22}\alpha^2$$

이 되어 $\beta_3 = 0$이 된다. 따라서 모형은

$$\eta(x)= \beta_0 + \beta_1x + \beta_2x^2 + \beta_4T_1(x-\alpha) + \beta_5T_2(x-\alpha)$$

**미분 연속 조건**  
만약 $\frac{d\eta(x)}{dx}$도 연속이면 $2a(\beta_{22}-\beta_{12}) + (\beta_{21}-\beta_{11}) = 0$이 되어$\beta_4 = 0$이 된다.  
따라서 모형은

$$\eta(x)=\beta_0 + \beta_1x + \beta_2x^2 + \beta_5T_2(x-\alpha)$$

이 모형을 **회귀 스플라인 (regression spline)** 모형이라고 한다.

**일반적인 표현**

$$\eta(x) = \sum_{i=0}^{q_1}\beta_{1i}x^i + \sum_{j=0}^{q_2}\beta_{2j}T_j(x-\alpha_1) + \cdots + \sum_{k=0}^{q_r}\beta_{rk}T_k(x-\alpha_{r-1}) \tag{7.21}$$

연속 조건을 적용하면
* $T_0(x-\alpha)$ 항 제거
* 1차 미분 연속이면 $T_1(x-\alpha)$ 제거
* 2차 미분 연속이면 $T_2(x-\alpha)$ 제거

경계점 $\alpha_i$를 **연결점 (junction point)** 라고 한다.
만약 $\alpha_i$가 알려진 상수(known constants, '기지 상수' 라고도 함)이면
* 모형은 **선형회귀모형**
* 계수 $\beta$는 **최소제곱법**으로 추정 가능

하지만 $\alpha_i$가 미지이면
* 모형이 **비선형회귀 (nonlinear regression)** 가 된다.

**예제 7.4**  

수출액 데이터를 이용하여 $x=4$ 에서 조각다항회귀를 추정한다.  
두 직선은 $x=4$에서 **연속**이라고 가정한다.  
모형 $\eta(x)=\beta_0 + \beta_1x + \beta_2T_1(x-4)$
$$T_1(x-4)= \begin{cases}
0, & x\le4 \\
x-4, & x>4
\end{cases}$$

**행렬표현**  
설계행렬

$$X = \begin{pmatrix}
1 & 0 & 0 \\
1 & 1 & 0 \\
1 & 2 & 0 \\
1 & 3 & 0 \\
1 & 4 & 0 \\
1 & 5 & 1 \\
1 & 6 & 2 \\
1 & 7 & 3
\end{pmatrix}$$

반응벡터

$$y=(1.0,1.2,1.3,1.5,1.7,2.5,3.4,4.2)^T$$

**최소제곱추정** 

$$\hat{\beta}=(X^TX)^{-1}X^Ty = \begin{pmatrix}
1.001 \\
0.169 \\
0.676
\end{pmatrix}$$

**회귀식** $\hat{y}=1.001 + 0.169x + 0.676T_1(x-4)$  
즉 $x \le 4$이면 $\hat{y}=1.001 + 0.169x$, $x > 4$ 이면 $\hat{y}=-1.703 + 0.845x$  

**예측**
2019년 $x=8$ 일 때 $\hat{y}=5.057$(단위: 백만 불)

**결정계수** $\bar{y}=2.1, SST = 9.440, SSR = 9.4218, R^2 = 0.998$  
따라서 모형 적합도가 매우 높다.


## 7.5 최적실험계획 (Optimal Experimental Design)
지금까지 우리는 $(x_i,y_i),\ i=1,\ldots,n$의 데이터가 **이미 존재한다고 가정**하였다. 그러나 실제 연구에서는 어떤 $x$ 값에서 몇 번 측정할 것인지를 **실험계획 (design of experiments)** 을 통해 결정할 수 있다.  

설명변수의 **관심 영역(region of interest)** 을 $a \le x \le b$ 라고 하자. 데이터를 $n$개 측정한다고 해도 반드시 **서로 다른 $n$개의 $x$** 를 사용할 필요는 없다. 예를 들어 $k < n$개의 수준을 선택하고 각 수준에서 $n_1,n_2,\ldots,n_k$번 측정하여 $n=n_1+n_2+\cdots+n_k$이 되도록 할 수도 있다.

실험계획을 하기위해서는 판정기준(criteria)을 먼저 정해야 한다. 일반적으로는 회귀계수의 분산을 최소화하는 것을 목표로 한다.

>### 추가: 실험계획법의 기본원리와 주요 개념
>
>실험계획법의 고전적인 기본원리는 일반적으로 **반복, 랜덤화, 블록화의 세 가지**로 정리한다. 직교성은 효율적인 실험계획이 갖추어야 할 중요한 성질이며, 교락은 제한된 실험 횟수 안에서 여러 인자를 다룰 때 발생하거나 의도적으로 활용하는 설계 개념이다.  
>[JMP의 실험계획 기본원리](https://www.jmp.com/en/statistics-knowledge-portal/design-of-experiments/key-design-of-experiments-concepts/key-principles-of-experimental-design), [Penn State STAT 503](https://online.stat.psu.edu/stat503/Lesson01.html)
>
>요약하면, 반복은 실험오차를 추정하고, 랜덤화는 체계적 편향을 방지하며, 블록화는 알려진 잡음인자의 영향을 분리한다. 직교성은 효과들을 효율적으로 추정할 수 있게 하는 설계의 성질이고, 교락(Confounding)은 제한된 실험 자원과 효과의 식별 가능성 사이에서 발생하는 절충관계다.
>1) 반복의 원리(Principle of Replication)
>
>반복이란 동일한 처리조건을 서로 독립적인 여러 실험단위에 적용하는 것을 말한다. 반복실험을 수행하면 동일한 처리조건에서도 나타나는 실험단위 사이의 변동을 측정할 수 있으며, 이를 이용하여 실험오차의 크기를 추정할 수 있다.
>
>실험오차를 추정해야 처리효과의 표준오차를 계산하고, 처리 간 차이가 우연한 변동으로 설명될 수 있는지를 통계적으로 검정할 수 있다. 일반적으로 반복 횟수가 증가하면 처리효과 추정량의 분산이 감소하므로 실험의 정밀도와 검정력이 높아진다.
>
>다만 동일한 실험단위를 여러 번 측정하는 것은 독립적인 반복이 아닐 수 있다. 예를 들어 한 사람의 혈압을 열 번 측정했다고 해서 표본크기가 10인 독립 반복실험이 되는 것은 아니다. 같은 실험단위에서 얻은 반복측정이나 하나의 시료를 여러 부분으로 나누어 측정한 결과를 독립적인 반복으로 간주하면 의사반복(pseudoreplication) 문제가 발생한다.
>
>2) 랜덤화의 원리(Principle of Randomization)
>
>랜덤화란 처리조건을 실험단위에 무작위로 배정하고, 필요한 경우 실험의 수행 순서도 무작위로 정하는 것을 말한다.
>
>실험 결과에는 연구자가 알고 있거나 통제할 수 있는 요인뿐 아니라 시간의 흐름, 장비 상태, 작업자의 숙련도, 실험단위의 특성처럼 완전히 파악하기 어려운 요인도 영향을 미친다. 랜덤화를 실시하면 이러한 잠재적 교란요인이 특정 처리조건에 체계적으로 집중되는 것을 방지할 수 있다.
>
>랜덤화가 모든 교란요인을 실제 표본에서 완벽하게 균형화하는 것은 아니다. 다만 각 처리집단이 관측되지 않은 요인의 영향을 받을 기회를 확률적으로 동일하게 만들어 처리효과의 편향을 줄이고, 통계적 검정과 인과적 해석의 근거를 제공한다.
>
>예를 들어 신약의 효과를 평가할 때 참가자를 실험군과 대조군에 무작위로 배정하면 체력, 생활습관, 질병의 중증도와 같은 특성이 어느 한 집단에 체계적으로 몰리는 것을 방지할 수 있다.
>
>3) 블록화의 원리(Principle of Blocking)
>
>블록화란 반응에 영향을 미칠 것으로 예상되는 외생변수 또는 잡음인자의 값이 비슷한 실험단위끼리 묶은 다음, 각 블록 안에서 처리조건을 무작위로 배정하는 방법이다.
>
>예를 들어 토양 상태가 다른 여러 지역에서 비료의 효과를 비교한다면 토양이 유사한 구역끼리 블록을 구성하고, 각 블록 안에 모든 비료 조건을 배정할 수 있다. 그러면 토양 차이로 인한 변동과 비료 처리로 인한 변동을 분리하여 분석할 수 있다.
>
>블록화를 적절히 실시하면 블록 간 변동을 오차에서 분리할 수 있으므로 오차제곱합과 오차분산이 감소한다. 이에 따라 처리효과의 추정 정밀도와 검정력이 높아진다. 반대로 반응과 관련이 거의 없는 변수를 기준으로 지나치게 세분화하면 자유도만 감소할 수 있으므로, 반응에 실질적으로 영향을 미치는 변수를 블록 기준으로 선택해야 한다. [Penn State의 블록화 설명](https://online.stat.psu.edu/stat503/Lesson04)
>
>4) 직교성(Orthogonality)
>
>직교성이란 실험계획행렬에서 서로 다른 효과에 해당하는 열벡터의 내적이 0이 되는 성질을 말한다. 두 열벡터 $\mathbf x_j$와 $\mathbf x_k$가 직교하려면 $\mathbf x_j^\mathsf{T}\mathbf x_k=0$ 이어야 한다.
>
>직교설계에서는 한 효과의 추정이 다른 효과의 추정에 영향을 받지 않는다. 등분산 선형모형 아래에서는 서로 직교하는 효과의 추정량이 서로 상관되지 않으며, 각 효과에 대응하는 제곱합도 명확하게 분리된다. 따라서 동일한 실험 횟수로도 각 효과를 효율적으로 추정할 수 있다.
>
>다만 직교성이 있다고 해서 모든 주효과와 교호작용을 항상 구분하여 추정할 수 있는 것은 아니다. 어떤 효과를 추정할 수 있는지는 실험계획에 포함된 처리조합과 교락구조에 따라 달라진다. 특히 부분요인실험에서는 설계가 직교하더라도 일부 효과가 서로 교락될 수 있다. 직교성은 효과를 독립적으로 추정할 수 있게 하는 중요한 성질이지만, 모든 효과의 식별 가능성을 자동으로 보장하지는 않는다. [NIST의 직교성과 블록설계 설명](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3333.htm)
>
>5) 교락(Confounding 또는 Aliasing)
>
>교락이란 둘 이상의 효과가 동일한 관측값의 대비에 반영되어 각각의 효과를 따로 추정할 수 없는 상태를 말한다. 예를 들어 인자 $A$와 $B$의 교호작용 $AB$가 인자 $C$의 주효과와 교락되어 있다면 자료에서 추정되는 효과는 $AB+C$ 와 같은 결합된 효과다. 따라서 전체적인 결합 효과는 추정할 수 있지만, 그 효과가 $AB$에서 발생했는지 $C$에서 발생했는지는 구분할 수 없다.
>
>교락은 일반적으로 피해야 하지만, 모든 처리조합을 한꺼번에 실험하기 어렵거나 실험 횟수를 줄여야 할 때 의도적으로 활용할 수 있다. 예를 들어 $2^k$ 요인실험을 여러 블록으로 나누면서 블록효과를 중요도가 낮다고 판단한 고차 교호작용과 교락시킬 수 있다. 이렇게 하면 주요한 주효과와 저차 교호작용을 추정할 수 있는 능력을 유지하면서 블록의 크기를 줄일 수 있다.
>
>부분요인실험에서도 전체 처리조합 중 일부만 선택하는 대신 특정 효과들을 서로 교락시킨다. 이때 일반적으로 주효과와 저차 교호작용은 중요하고 고차 교호작용은 상대적으로 작다는 **효과의 희소성 원리**를 전제로 한다. 이 가정이 성립하지 않으면 중요한 효과를 잘못 해석할 위험이 있다. [NIST의 교락 정의](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3343.htm), [Penn State의 요인실험 교락 설명](https://online.stat.psu.edu/stat503/book/export/html/663)



실험계획 판정기준으로 많이 쓰는 회귀계수 분산 최소화를 살펴보자: $y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip} + \epsilon_i \quad
\epsilon_i \sim N(0,\sigma^2)$


일반화분산 (Generalized Variance): 분산행렬의 행렬식 $|Var(\hat{\beta})| = |\sigma^2 (X^TX)^{-1}|$을 **일반화분산 (generalized variance)** 이라고 한다. 이 값을 작게 하려면 $|(X^TX)^{-1}|$이 작아야 하므로 $|X^TX|$가 커야 한다.  

그런데 $X^TX$는 완전히 변수 $x_1, \ldots, x_p$의 수준에 의해 결정된다. 따라서 실험계획은 $|X^TX|$를 최대화하는 $x$의 수준을 선택하는 문제로 귀결된다.  
$|X^TX|$ 를 최대화하는 실험계획을 **D-최적계획 (D-optimal design)** 이라고 한다.
* 회귀계수의 공분산을 최소화
* $\mathbf{\beta}$의공동신뢰영역의 부피 최소

한편, 분산행렬의 대각선상에 있는 $\hat\beta_0, \dots, \hat\beta_p$의 분산들이 잇고, 이들의 합 $tr(\sigma^2(X^TX)^{-1})$ 을 최소화하는 계획을 **A-최적계획 (A-optimal design)** 이라고 한다. 이는 $Var(\hat{\beta_0})+\cdots+Var(\hat{\beta_p})$ 의 합을 최소화한다.  
D-최적계획과 A-최적계획은 **항상 일치하지 않는다.**

> 기타 실험계획 형태들이 있으나, 이 책의 범위를 넘기므로 생략한다.

**예제 7.5**  

단순회귀모형 $y=\beta_0+\beta_1x+\epsilon$에서 D-최적계획과 A-최적계획을 구하라.

실험 영역 $-1 \le x \le 1$ 데이터 수 $n=3$

**설계행렬** 
 
$$X= \begin{pmatrix}
1 & x_1 \\
1 & x_2 \\
1 & x_3
\end{pmatrix}\\
X^TX= \begin{pmatrix}
3 & x_1+x_2+x_3 \\
x_1+x_2+x_3 & x_1^2+x_2^2+x_3^2
\end{pmatrix}$$

행렬식 $|X^TX| = 3(x_1^2+x_2^2+x_3^2)-(x_1+x_2+x_3)^2$  
이를 최대화하면 $x_1=x_2=-1,\quad x_3= \quad \text{or} \quad x_1=-1,\quad x_2=x_3=1$

따라서 **D-최적실험계획**: $x=-1,-1,1 \quad \text{or} \quad x=-1,1,1$ 

A-최적계획을 계산해도 동일한 결과가 나온다.  
하지만 일반적으로 **D-최적계획 (D-optimal design)**, **A-최적계획 (A-optimal design)** 은 항상 동일하지는 않다.
