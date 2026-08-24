# Chapter 11 가변수 (Dummy Variables)

앞에서 다룬 회귀분석모형에서는 모든 설명변수(explanatory variable)가 **양적변수(quantitative variable)** 라고 가정하였다. 양적변수란 어떤 구간에 속하는 모든 값을 관측값으로 취할 수 있는 변수이며, 예를 들어 온도, 압력, 습도, 무게 등과 같이 **수치적으로 비교 가능한 변수**를 의미한다.

그러나 실제 데이터 분석에서는 이러한 양적변수뿐만 아니라 **질적변수(qualitative variable)** 도 자주 사용된다. 질적변수는 값의 크기를 양적으로 비교할 수 없는 변수이며 **범주형 변수(categorical variable)** 라고도 한다. 예를 들어 성별(남, 여), 신용등급(좋음, 나쁨), 보험의 종류(생명보험, 손해보험) 등이 이에 해당한다.

질적변수가 두 개의 범주로 분류되는 경우에는 보통 값을 **0과 1**로 표현하는 것이 편리하다. 이러한 변수 표현 방식을 **가변수(dummy variable)** 또는 **이진변수(binary variable)** 라고 한다. 이 장에서는 이러한 가변수를 회귀모형에 포함시키는 방법을 설명한다.

또한 양적변수는 측정척도에 따라 다음과 같이 구분되기도 한다.

* **구간변수(interval variable)**
* **비율변수(ratio variable)**

구간변수는 두 값의 차이를 측정할 수 있는 변수이지만 절대적인 0의 의미가 없는 변수이다. 예를 들어 온도(섭씨온도), IQ 등이 이에 해당한다.

반면 비율변수는 절대적인 0을 가지며 값의 비율 비교가 가능한 변수이다. 예를 들어 온도를 Kelvin 단위로 측정한 값, 키, 몸무게, 인터넷 사용시간 등이 이에 해당한다. 실제 데이터 분석에서는 대부분의 양적변수가 비율변수이다.

이 장에서는 질적변수가 **설명변수(explanatory variable)** 인 경우와 **반응변수(response variable)** 인 경우를 모두 다룬다. 먼저 설명변수가 가변수인 경우를 설명하고, 이후 반응변수가 가변수인 경우를 설명한다.

## 11.1 설명변수로 한 개의 가변수를 갖는 경우 (Single Dummy Variable)

### 11.1.1 기본적인 모형 (Basic Model)

중회귀모형(multiple regression model)에서 하나의 설명변수가 0과 1의 값을 가지는 가변수인 경우를 고려한다.

어떤 회사에서 새로운 프로그램을 도입하고 직원들에게 교육을 실시하였다. 교육 이후 직원들이 프로그램을 익히는 데 걸린 시간을 측정하고, 교육 시작 전에 실시한 적성검사 점수와 성별 정보를 함께 수집하였다.  
수집된 데이터는 다음과 같은 변수들로 구성된다.

* $y$ : 프로그램을 익히는 데 걸린 **총 소요시간(total time)**
* $x_1$ : 교육 시작 전에 실시한 **적성검사 점수(aptitude test score)**
* $x_2$ : **성별(sex)**

성별 변수는 가변수로 다음과 같이 정의한다.

$$
x_2=
\begin{cases}
0, & \text{남자인 경우} \\
1, & \text{여자인 경우}
\end{cases}
$$

표본의 크기는 $n=20$  
이 데이터를 설명하기 위하여 다음과 같은 **중회귀모형(multiple linear regression model)** 을 가정한다.

$$y_i=\beta_0+\beta_1 x_{i1}+\beta_2 x_{i2}+\varepsilon_i$$

* $\varepsilon_i \sim N(0,\sigma^2)$
* $\operatorname{Cov}(\varepsilon_i,\varepsilon_j)=0 \quad (i\ne j)$

이 모형에서의 **반응함수(response function)** 는 다음과 같다.

$$E(y)=\beta_0+\beta_1 x_1+\beta_2 x_2$$

이제 성별에 따라 반응함수를 구해 보면  
**남자인 경우**: $x_2=0$이므로 

$$E(y)=\beta_0+\beta_1 x_1$$

**여자인 경우**: $x_2=1$이므로

$$E(y)=\beta_0+\beta_1 x_1+\beta_2 =(\beta_0+\beta_2)+\beta_1 x_1$$

따라서 남자와 여자에 대한 두 개의 회귀직선(regression line)은 **기울기(slope)** 는 동일하지만 **절편(intercept)** 이 서로 다른 두 직선이 된다.

* 남자 직선의 절편 : $\beta_0$
* 여자 직선의 절편 : $\beta_0+\beta_2$

따라서 $\beta_2$는 **남녀 간 평균 차이(mean difference)** 를 나타내는 계수이다.

**최소제곱추정 (Least Squares Estimation)**  
주어진 데이터에 대해 **최소제곱추정법(least squares method)** 을 적용하면 회귀계수의 추정치는 다음과 같이 계산된다.

$$
\begin{bmatrix}
\hat{\beta}_0 \\
\hat{\beta}_1 \\
\hat{\beta}_2
\end{bmatrix}
= (X^TX)^{-1}X^Ty
= \begin{bmatrix}
33.8741 \\
-0.1017 \\
8.0555
\end{bmatrix}
$$

따라서 **추정된 회귀식(fitted regression function)** 은 $\hat{y}=33.8741-0.1017x_1+8.0555x_2$ 이를 성별에 따라 분리하면 다음과 같다.  
**남자**: $\hat{y}=33.8741-0.1017x_1$  
**여자**: $\hat{y}=(33.8741+8.0555)-0.1017x_1$  
즉 $\hat{y}=41.9296-0.1017x_1$

**(1) $\beta_1$**  
적성검사 점수 $x_1$이 증가할 때 프로그램을 익히는 데 걸리는 평균 시간이 어떻게 변하는지를 나타낸다. $\hat{\beta}_1=-0.1017$ 이므로 적성검사 점수가 높을수록 프로그램을 더 빠르게 익힌다고 해석할 수 있다. 예를 들어 점수가 **100점 높아질 경우 평균 소요시간은 약 10시간 감소**한다.

**(2) $\beta_2$**  
성별에 따른 평균 차이를 나타낸다. $\hat{\beta}_2=8.0555$ 이므로 평균적으로 남자가 여자보다 약 **8시간 정도 더 빨리 프로그램을 익힌다**고 해석할 수 있다.

또한 두 회귀직선의 기울기가 동일하므로 적성검사 점수에 관계없이 남녀 간 평균 차이는 일정하다.

**분산분석 (Analysis of Variance, ANOVA)**  

| 요인 | 제곱합 | 자유도 | 평균제곱 | F |
|---|---:|---:|---:|---:|
| 회귀 (Regression) | $SSR=1504.4$ | 2 | $MSR=752.2$ | 72.3 |
| 잔차 (Error) | $SSE=176.4$ | 17 | $MSE=10.4$ |  |
| 전체 (Total) | $SST=1680.8$ | 19 |  |  |

유의수준 $\alpha=0.05$에서의 임계값은 $F_{0.05}(2,17)=3.59$ 따라서
$F_0=72.3>3.59$ 이므로 모형은 통계적으로 유의하다고 판단할 수 있다.

### 11.1.2 교호작용효과를 포함한 모형 (Model with Interaction Effect)

앞 절의 모형에서는 적성검사 점수와 소요시간 사이의 관계가 남녀 모두 동일하다고 가정하였다. 즉 두 직선의 기울기가 같다고 가정하였다.

그러나 만약 **적성검사 점수와 소요시간 사이의 관계가 성별에 따라 달라진다면**, 두 변수 사이에는 **교호작용(interaction)** 이 존재한다고 한다. 이 경우 다음과 같은 모형을 고려할 수 있다.

$$y_i=\beta_0+\beta_1 x_{i1}+\beta_2 x_{i2}+\beta_3 x_{i1}x_{i2}+\varepsilon_i$$

반응함수는

$$E(y)=\beta_0+\beta_1 x_1+\beta_2 x_2+\beta_3 x_1x_2$$

성별에 따른 반응함수는 다음과 같다.

**남자**: $E(y)=\beta_0+\beta_1 x_1$  
**여자**: $E(y)=\beta_0+\beta_1 x_1+\beta_2+\beta_3 x_1 =(\beta_0+\beta_2)+(\beta_1+\beta_3)x_1$

따라서

* 절편 차이 : $\beta_2$
* 기울기 차이 : $\beta_3$

**최소제곱추정 (Least Squares Estimation)**  
최소제곱추정을 수행하면 다음과 같은 추정치를 얻는다.

$$
\begin{bmatrix}
\hat{\beta}_0 \\
\hat{\beta}_1 \\
\hat{\beta}_2 \\
\hat{\beta}_3
\end{bmatrix}
= \begin{bmatrix}
33.8384 \\
-0.1015 \\
8.1313 \\
-0.0004
\end{bmatrix}
$$

따라서 추정된 회귀식은 $\hat{y}=33.8384-0.1015x_1+8.1313x_2-0.0004x_1x_2$

**교호작용 검정 (Interaction Test)**  
교호작용 효과가 존재하는지 검정하는 가설: $H_0:\beta_3=0$  
검정통계량은 

$$t_0=\frac{\hat{\beta}_3}{\sqrt{\operatorname{Var}(\hat{\beta}_3)}}$$

여기서 $\operatorname{Var}(\hat{\beta}_3)=c_{33}\sigma^2$ 이고 그 추정값은
 
$$\widehat{\operatorname{Var}(\hat{\beta}_3)} = c_{33}MSE = (0.00003049)(11.02) = 0.000336$$

따라서

$$t_0=\frac{-0.0004}{\sqrt{0.000336}}=-0.022$$

유의수준 $\alpha=0.05$에서 $t_{0.025}(16)=2.120$ 이므로 $|t_0|<2.120$

따라서 **귀무가설 $H_0:\beta_3=0$을 기각할 수 없다.**  
즉 이 데이터에서는 **교호작용 효과가 존재하지 않는 것으로 판단된다.**


## 11.2 설명변수로 여러 개의 가변수를 갖는 경우 (Multiple Dummy Variables)

앞 절에서는 하나의 설명변수가 가변수(dummy variable)인 경우를 다루었다. 이 절에서는 **범주가 세 개 이상인 질적변수(qualitative variable)** 또는 **두 개 이상의 질적 설명변수**가 존재하는 경우 회귀모형(regression model)에 여러 개의 가변수를 포함시키는 방법을 설명한다.

### 11.2.1 세 개 이상의 범주를 갖는 질적변수 (Qualitative Variable with Three or More Categories)

앞에서 살펴본 예제에서 새로운 프로그램을 익히는 데 직원들의 **학력** 이 중요한 요인이라고 판단한다고 하자. 이 경우 학력을 회귀분석모형에 포함시킬 수 있다.

직원의 학력은 다음 세 가지 범주 중 하나로 관측된다고 가정한다.

* $E_1$: 고등학교 졸업 (High school graduate)
* $E_2$: 대학교 졸업 (College graduate)
* $E_3$: 대학원 이상 (Graduate school or higher)

학력은 범주가 세 개인 **질적변수(categorical variable)** 이므로 이를 직접 회귀모형에 사용할 수 없다. 따라서 이를 **가변수(dummy variable)** 로 변환하여 사용한다.

직관적으로 다음과 같이 세 개의 가변수를 생각할 수 있다.

$$
x_2 =
\begin{cases}
1, & \text{학력이 }E_1 \text{인 경우} \\
0, & \text{기타}
\end{cases} \\
x_3 =
\begin{cases}
1, & \text{학력이 }E_2 \text{인 경우} \\
0, & \text{기타}
\end{cases} \\
x_4 =
\begin{cases}
1, & \text{학력이 }E_3 \text{인 경우} \\
0, & \text{기타}
\end{cases}
$$

앞 절에서 고려한 성별에 대한 가변수는 생략하고, 적성검사점수 $x_1$과 학력을 나타내는 가변수들을 포함하는 **중회귀모형(multiple linear regression model)** 을 다음과 같이 설정할 수 있다.

$$
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \beta_3 x_{i3} + \beta_4 x_{i4} + \varepsilon_i
\quad (i=1,2,\dots,n)
$$

이 모형은 직관적으로 이해하기 쉬우나 중요한 문제가 존재한다. 이를 설명하기 위하여 간단히 표본크기를 $n=5$라고 하고 다음과 같이 학력 분포가 있다고 하자.

* 처음 두 명 : $E_1$
* 다음 두 명 : $E_2$
* 마지막 한 명 : $E_3$

이 경우 설계행렬(design matrix) $X$는 다음과 같이 된다.

$$
X= \begin{bmatrix}
1 & x_{11} & 1 & 0 & 0 \\
1 & x_{21} & 1 & 0 & 0 \\
1 & x_{31} & 0 & 1 & 0 \\
1 & x_{41} & 0 & 1 & 0 \\
1 & x_{51} & 0 & 0 & 1
\end{bmatrix}
$$

여기서 중요한 점은 $ x_2 + x_3 + x_4 = 1$ 이 항상 성립한다는 것이다. 즉 세 개의 열은 서로 **선형종속(linearly dependent)** 관계를 가진다.

이와 같은 상황을 **완전 다중공선성(perfect multicollinearity)** 이라고 하며 이 경우

* 행렬 $X$의 계수(rank)가 감소하고
* $X^TX$가 **특이행렬(singular matrix)** 이 된다.

따라서 최소제곱추정량 $\hat{\beta} = (X^TX)^{-1}X^Ty$이 정의되지 않는다.

이 문제를 해결하기 위해서는 **가변수 하나를 제거**해야 한다. 일반적으로 범주가 $q$개이면 **$q-1$개의 가변수**만을 사용한다.

예를 들어 $x_4$를 제거하면 모형은 다음과 같이 된다.

$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \beta_3 x_{i3} + \varepsilon_i $$

이때 평균 반응함수(mean response function)는

$$E(y)=\beta_0+\beta_1x_1+\beta_2x_2+\beta_3x_3$$

각 학력 수준에 대한 반응함수는 다음과 같다.

**고등학교 졸업 ($E_1$)**: 이 경우 $x_2=1, x_3=0$이므로 $E(y)=\beta_0+\beta_1x_1+\beta_2$ 즉 $E(y)=(\beta_0+\beta_2)+\beta_1x_1$  
**대학교 졸업 ($E_2$)**: 이 경우 $x_2=0, x_3=1$이므로 $E(y)=\beta_0+\beta_1x_1+\beta_3$ 즉 $E(y)=(\beta_0+\beta_3)+\beta_1x_1$  
**대학원 이상 ($E_3$)**: 이 경우 $x_2=0, x_3=0$이므로 $E(y)=\beta_0+\beta_1x_1$ 

따라서 세 학력 집단의 회귀직선(regression line)은 **기울기는 동일하지만 절편이 서로 다르다.**
* $E_1$: $(\beta_0+\beta_2)$
* $E_2$: $(\beta_0+\beta_3)$
* $E_3$: $(\beta_0)$

따라서

* $E_1$과 $E_3$의 평균 차이 → $\beta_2$
* $E_2$와 $E_3$의 평균 차이 → $\beta_3$
* $E_1$과 $E_2$의 평균 차이 → $\beta_2-\beta_3$

로 해석할 수 있다.

**교호작용이 존재하는 경우 (Interaction Model)**  
학력과 적성검사점수 사이에 **교호작용(interaction)** 이 존재할 수도 있다. 즉 적성검사점수와 소요시간 사이의 관계가 학력에 따라 달라질 수 있다. 이 경우 모형은 다음과 같이 확장된다.

$$y_i =\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \beta_3 x_{i3} + \beta_4 x_{i1}x_{i2} + \beta_5 x_{i1}x_{i3} + \varepsilon_i$$

각 학력 집단에 대한 반응함수는 다음과 같다.  
**고등학교 졸업 ($E_1$)**: $E(y)=(\beta_0+\beta_2)+(\beta_1+\beta_4)x_1$  
**대학교 졸업 ($E_2$)**: $E(y)=(\beta_0+\beta_3)+(\beta_1+\beta_5)x_1$  
**대학원 이상 ($E_3$)**: $E(y)=\beta_0+\beta_1x_1$  

따라서 교호작용을 포함하면 **학력에 따라 절편(intercept)과 기울기(slope)가 모두 달라질 수 있다.**

### 11.2.2 두 개 이상의 질적변수 (Two or More Qualitative Variables)

앞 절에서는 하나의 질적변수만을 고려하였다. 그러나 실제 분석에서는 **여러 개의 질적 설명변수(multiple qualitative explanatory variables)** 가 동시에 포함될 수 있다. 예를 들어 프로그램 학습시간 데이터가 다음 변수들과 함께 수집되었다고 하자.

* $x_1$ : 적성검사점수
* $x_2$ : 성별 (남/여)
* $x_3, x_4$ : 학력 (고등학교, 대학교, 대학원)

이 경우 여러 개의 가변수를 이용하여 다음과 같은 **중회귀모형(multiple regression model)** 을 구성할 수 있다.

$$ y_i = \beta_0  + \beta_1 x_{i1} + \beta_2 x_{i2} + \beta_3 x_{i3} + \beta_4 x_{i4} + \varepsilon_i $$

필요한 경우 다음과 같은 **교호작용항(interaction term)** 도 포함할 수 있다.

* 성별 × 적성검사점수
* 학력 × 적성검사점수
* 성별 × 학력

이와 같이 여러 질적변수와 가변수를 포함하는 회귀모형은 실제 사회과학 및 경제학 데이터 분석에서 매우 널리 사용된다.


## 11.3 구간별 선형회귀 (Segmented Regression)

양적변수 $x$를 여러 구간으로 나누어 각 구간에서 서로 다른 **선형회귀모형(linear regression model)** 을 적합하는 방법을 **구간별 선형회귀(segmented regression)** 또는 **조각별 선형회귀(piecewise linear regression)** 라고 한다.

이는 앞 장에서 설명한 **조각다항회귀(piecewise polynomial regression)** 모형과 동일한 개념이며, 회귀모형을 1차식으로 제한하면 각 구간의 차수는 모두 1이 된다.

앞 장에서는 절단 거듭제곱 기저함수(truncated power basis function)

$$T_j(x-a_j)$$

를 이용하여 구간별 회귀모형을 설명하였다. 여기서는 **가변수(dummy variable)** 를 이용하여 같은 모형을 표현하는 방법을 설명한다.

예를 들어 설명변수 $x$가 어떤 기준점 $x_0$을 기준으로 두 구간으로 나누어진다고 하자.
* $x < x_0,\quad x \ge x_0$

이 경우 다음과 같은 가변수를 정의한다.

$$
z =
\begin{cases}
0, & x < x_0 \\
1, & x \ge x_0
\end{cases}
$$

이 가변수를 이용하면 회귀모형을 다음과 같이 표현할 수 있다.

$$y = \beta_0 + \beta_1 x + \beta_2 z + \beta_3 xz + \varepsilon$$

이때 두 구간에서의 반응함수는  
**첫 번째 구간**: $E(y)=\beta_0+\beta_1x$  
**두 번째 구간**: $E(y)=(\beta_0+\beta_2)+(\beta_1+\beta_3)x$

따라서
* $\beta_2$ → 절편 변화(intercept shift)
* $\beta_3$ → 기울기 변화(slope change)

이와 같이 가변수를 이용하면 구간별 회귀모형을 매우 간단하게 표현할 수 있으며, 구조적 변화(structural change)나 정책 변화(policy intervention) 분석에 자주 사용된다.


## 11.4 반응변수가 가변수인 경우 (Binary Response Case)

앞 절들에서는 **설명변수(explanatory variable)** 가 가변수(dummy variable)인 경우를 다루었다. 그러나 실제 데이터 분석에서는 **반응변수(response variable)** 자체가 가변수인 경우도 자주 발생한다. 예를들어, 
* 제조업체에서 **통계적 공정관리(statistical process control; SPC)** 활동을 실시하고 있는지 여부
* 개인이 **생명보험(life insurance)** 에 가입했는지 여부  

이와 같은 경우 반응변수는 두 가지 결과만 가질 수 있으므로 **이진형 변수(binary variable)** 가 된다. 이러한 경우 반응변수는 **0 또는 1의 값만을 가지는 가변수**가 된다.

$$
y = \begin{cases}
1, & \text{SPC 활동을 하는 경우} \\
0, & \text{SPC 활동을 하지 않는 경우}
\end{cases} \\

y = \begin{cases}
1, & \text{생명보험에 가입한 경우} \\
0, & \text{생명보험에 가입하지 않은 경우}
\end{cases}
$$

### 11.4.1 반응함수의 의미 (Meaning of the Response Function)

단순회귀모형(simple regression model)에서 반응변수가 가변수라고 하자. 그러면 모형은 다음과 같이 표현된다.

$$y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \quad y_i = 0,1$$

여기서 $E(\varepsilon_i)=0$ 이라고 하면 평균 반응함수(mean response function)는 $E(y_i)=\beta_0+\beta_1 x_i$ 로 표현된다. 반응변수 $y_i$는 0 또는 1의 값을 가지므로 **베르누이 확률분포(Bernoulli distribution)** 를 따른다고 가정할 수 있다. $P(y_i=1)=p_i, P(y_i=0)=1-p_i=q_i$ 따라서 $p_i+q_i=1$ 이 성립한다. 이때 기대값은 $E(y_i)=p_i$ 이므로 평균 반응함수는 

$$E(y_i)=\beta_0+\beta_1 x_i=p_i$$

즉 설명변수 $x_i$가 주어졌을 때 반응변수의 기대값은 **사건이 발생할 확률(probability)** 과 같다. 따라서 $\hat{y}= \hat{p}$ 은 사건이 발생할 확률의 추정값이 된다.

예를 들어 개인의 소득 $x$와 생명보험 가입 여부 $y$ 사이의 관계를 고려하면

* $x$ : 개인의 소득
* $y$ : 보험 가입 여부

이고 평균 반응함수는 다음과 같은 **확률함수(probability function)** 로 해석된다.

$$E(y)=\beta_0+\beta_1 x$$

이는 소득이 증가함에 따라 생명보험에 가입할 확률이 어떻게 변화하는지를 나타낸다.

### 11.4.2 반응변수가 가변수인 경우의 문제점 (Problems with Binary Response in Linear Regression)

설명변수가 범주형인 경우와 달리 반응변수가 범주형일 때 **선형회귀모형(linear regression model)** 을 그대로 적용하면 몇 가지 문제가 발생한다.

특히 다음 세 가지 문제가 중요한 통계적 문제를 일으킨다.  
1. **오차의 비정규성 (Non-normality of errors)**
2. **오차의 이분산성 (Heteroscedasticity; unequal variance of errors)**
3. **반응함수의 제약성 (Constraint on response function)**

#### (1) 오차의 비정규성 (Non-normality of errors)

모형 $y_i=\beta_0+\beta_1 x_i+\varepsilon_i$ 에서 오차는 $\varepsilon_i=y_i-(\beta_0+\beta_1 x_i) $ 그러나 $y_i$는 0 또는 1만을 가지므로 오차는 다음 두 값만을 취한다.

$$
\varepsilon_i=
\begin{cases}
1-\beta_0-\beta_1 x_i, & y_i=1 \\
-\beta_0-\beta_1 x_i, & y_i=0
\end{cases}
$$

즉 오차항은 **두 개의 값만을 가지는 이산분포(discrete distribution)** 를 따르며 **정규분포(normal distribution)** 를 따르지 않는다. 따라서 선형회귀분석에서 사용하는 t 검정, F 검정 (F-test) 등의 통계적 검정이 이론적으로 성립하지 않는다.

#### (2) 오차의 이분산성 (Heteroscedasticity)

오차의 분산을 계산하면 $Var(\varepsilon_i)=Var(y_i)$ 이고 $Var(y_i)=E[(y_i-E(y_i))^2]$ 베르누이 분포의 성질에 의해 $Var(y_i)=p_i(1-p_i)$
따라서 $Var(\varepsilon_i)=p_i(1-p_i)$ 단순회귀모형에서는 $p_i=\beta_0+\beta_1 x_i$ 이므로

$$ Var(\varepsilon_i)=(\beta_0+\beta_1 x_i)(1-\beta_0-\beta_1 x_i)$$

따라서 오차의 분산은 $x_i$에 의존하며 **이분산(heteroscedasticity)** 을 가진다. 즉 $Var(\varepsilon_i) \neq Var(\varepsilon_j)$ 이므로 오차는 **등분산성(homoscedasticity)** 을 만족하지 않는다.

#### (3) 반응함수의 제약성 (Constraint on Response Function)

평균 반응함수는 $E(y)=p$ 이며 이는 **확률(probability)** 이므로 $0 \le p \le 1$ 의 범위를 가져야 한다. 그러나 선형회귀모형 $E(y)=\beta_0+\beta_1 x$ 은 이 범위를 벗어날 수 있다.  
예를 들어 $x$가 충분히 커지면 $E(y)>1$ 이 될 수 있으며 이는 확률로 해석할 수 없다. 따라서 반응변수가 가변수인 경우 선형회귀모형을 그대로 사용하는 것은 이론적으로 타당하지 않다.  
이 문제는 이후 **로지스틱 회귀모형(logistic regression model)** 에서 보다 적절하게 해결된다.

### 11.4.3 추정방법 (Estimation Method)

비록 오차항이 정규분포를 따르지 않지만 **최소제곱추정량(ordinary least squares estimator)** $\hat{\beta}=(X^TX)^{-1}X^Ty$ 은 여전히 **불편추정량(unbiased estimator)** 의 성질을 유지한다. 또한 표본의 크기가 충분히 크면 $\hat{\beta}$ 는 **점근적으로 정규분포(asymptotically normal distribution)** 를 따른다.  
그러나 오차의 분산이 일정하지 않기 때문에 이론적으로는 **일반화 최소제곱추정법(generalized least squares; GLS)** 을 사용하는 것이 더 적절하다.

오차의 분산-공분산행렬은 다음과 같이 표현된다.

$$
Var(\varepsilon)=
\begin{bmatrix}
p_1(1-p_1) & 0 & \cdots & 0 \\
0 & p_2(1-p_2) & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & p_n(1-p_n)
\end{bmatrix}
=V
$$

따라서 이론적인 **GLS 추정량** 은

$$\hat{\beta}^*=(X^T V^{-1} X)^{-1}X^T V^{-1}y$$

그러나 실제로는 $p_i$가 알려져 있지 않기 때문에 다음과 같은 절차를 사용한다.

1. 최소제곱추정(OLS)으로 $(\hat{\beta}_0,\hat{\beta}_1)$을 구한다.

2. $\hat{p}_i=\hat{\beta}_0+\hat{\beta}_1 x_i$ 를 계산한다.

3. $\hat{V}=\operatorname{diag}(\hat{p}_1(1-\hat{p}_1),\dots,\hat{p}_n(1-\hat{p}_n))$ 을 계산한다.

4. $\hat{\beta}^*=(X^T \hat{V}^{-1}X)^{-1}X^T \hat{V}^{-1}y$ 를 계산한다.

$\hat{\beta}^*$ 를 구한후에 $Var(\hat{\beta}^*)=(X^T \hat{V}^{-1}X)^{-1}$ 이므로 행렬 $V$의 $p_i$에 $\hat{\beta}_0^*+\hat{\beta}_1^* x_i$ 를 대입하여 행렬$V$의 대각원소를 구한다.

$$
\hat{V}^*= \begin{bmatrix}
\hat{p}_1^*(1-\hat{p}_1^*) & 0 & \cdots & 0 \\
0 & \hat{p}_2^*(1-\hat{p}_2^*) & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\ 
0 & 0 & \cdots & \hat{p}_n^*(1-\hat{p}_n^*)
\end{bmatrix}
$$

를 계산하고, $\widehat{Var}(\hat{\beta}^*)=(X^T \hat{V}^{*-1}X)^{-1}$ 을 계산하여 추정된 회귀계수의 분산을 구한다.


#### 예제 11.1

다음 데이터는 임의로 선정된 25개의 중소기업에 대하여

* 연간 총 매출액 $x$
* SPC 활동 여부 $y$

를 조사한 결과이다.

SPC 활동이 활발한 기업에는 $y=1$을 부여하고 그렇지 않은 기업에는 $y=0$ 을 부여하였다. 먼저 최소제곱추정에 의해 회귀계수를 추정하면 $\hat{\beta}_0=-0.0922,\quad \hat{\beta}_1=0.0315$ 가 된다. 따라서 추정된 반응함수는 $\hat{E}(y)=-0.0922+0.0315x$

예를 들어 매출액이 1000억 원인 기업의 경우 $x=10$이므로 $\hat{p}=-0.0922+0.0315(10)=0.2228$ 즉 해당 기업이 SPC 활동을 수행할 확률은 약 **0.2228**이다.

그러나 매출액이 매우 큰 경우 $\hat{p}>1$ 이 될 수 있어 확률로 해석할 수 없는 문제가 발생한다. 이를 개선하기 위하여 일반화 최소제곱추정(GLS)을 적용하면 $\hat{\beta}_0^*=-0.1171,\quad \hat{\beta}_1^*=0.0327 $  
따라서 추정된 반응함수는 $\hat{E}(y)=-0.1171+0.0327x$

마지막으로 최소제곱추정과 일반화최소제곱추정의 분산을 비교하면 다음과 같다.

| 추정방법            | $\beta_0$      | $\beta_1$     |
| --------------- | ------- | ------ |
| 최소제곱추정 (OLS)    | -0.0922 | 0.0315 |
| 일반화최소제곱추정 (GLS) | -0.1171 | 0.0327 |

분산 비교 결과 **GLS 추정량의 분산이 더 작다**는 것을 확인할 수 있다.  
따라서 반응변수가 가변수인 경우에는 **가중회귀(weighted regression)** 또는 **일반화 최소제곱법(GLS)** 을 사용하는 것이 보다 효율적인 추정방법이 된다.
