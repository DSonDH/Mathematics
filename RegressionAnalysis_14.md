# Chapter 14 로지스틱 회귀분석 (Logistic Regression)

반응변수 $y$가 0 또는 1의 값을 가지는 경우에는 선형회귀보다 확률 자체를 직접 모형화하는 접근이 필요하다. 이때 가장 널리 사용되는 모형이 로지스틱 회귀모형(logistic regression model)이다. 로지스틱 회귀는 선형회귀모형의 개념을 반응변수가 범주형인 경우로 확장한 것으로, 설명변수의 변화에 따라 사건 발생확률이 어떻게 달라지는지를 다룬다.

반응변수가 가변수일 때 조건부 평균 $E(y \mid x)$는 곧 $y=1$일 확률이 된다. 따라서

$$E(y \mid x)=P(y=1 \mid x)$$

이며, 이 값은 반드시 0과 1 사이에 있어야 한다. 실제 자료에서는 설명변수 $x$가 증가함에 따라 이 확률이 0에서 1로 서서히 수렴하거나, 반대로 1에서 0으로 감소하는 S자형 곡선이 자주 나타난다. 이러한 형태를 자연스럽게 표현하는 함수가 로지스틱 함수(logistic function)이다.  
로지스틱 회귀의 핵심은 확률을 직접 선형식으로 두지 않고, 확률을 적절히 변환한 값이 설명변수에 대해 선형이 되도록 만드는 데 있다. 이때 사용되는 변환이 로짓 변환(logit transformation)이며, 이를 통해 **해석 가능한** 회귀계수와 확률모형을 동시에 얻는다.

## 14.1 단순 로지스틱 회귀 (Simple Logistic Regression)

단순 로지스틱 회귀는 하나의 설명변수 $x$와 이진 반응변수(binary response variable) $y$ 사이의 관계를 다루는 모형이다. 여기서 $y$는 0 또는 1의 값을 가지며, 보통 $y=1$은 성공, 사건 발생, 질병 유무 중 "있음"에 해당하는 상태를 뜻한다.

### 14.1.1 로지스틱 반응함수 (Logistic Response Function)

설명변수 $x$에 대한 반응변수의 조건부 평균은

$$
E(y\mid x)=\frac{\exp(\beta_0+\beta_1x)}{1+\exp(\beta_0+\beta_1x)}
\tag{16.1}
$$

이 식은 로지스틱 함수(logistic function)의 형태이며, 다음과 같은 성질을 가진다.

1. $E(y\mid x)$는 항상 0과 1 사이의 값을 가진다.
2. $x$가 증가할 때 $\beta_1$의 부호에 따라 확률이 증가하거나 감소한다.
3. 함수의 모양은 S자형 곡선(S-shaped curve)이다.
4. $\beta_0$와 $\beta_1$의 값에 따라 곡선의 위치와 기울기가 달라진다.

따라서 선형회귀처럼 예측값이 $[0,1]$ 범위를 벗어나는 문제가 없고, 확률을 직접 설명하는 데 적합하다.  
또한 조건부 평균은 곧 사건확률이므로

$$
E(y\mid x)=P(y=1\mid x)=\pi(x) \tag{16.2}
$$

로 쓸 수 있다. 여기서 $\pi(x)$는 설명변수 값이 $x$일 때의 성공확률이다.

### 14.1.2 로짓 변환과 선형화 (Logit Transformation and Linearization)

식 (16.1)은 $(\beta_0,\beta_1)$에 대해 비선형함수처럼 보이지만, 확률 $\pi(x)$에 로짓 변환을 적용하면 선형형태로 바뀐다. 로짓(logit)은 다음과 같이 정의된다.

$$
\operatorname{logit}(\pi(x))
=\ln\left(\frac{\pi(x)}{1-\pi(x)}\right)
=\beta_0+\beta_1x
\tag{16.3}
$$

이 식이 단순 로지스틱 회귀모형의 기본식이다. 즉, **확률 자체가 선형인 것이 아니라 확률의 로그 오즈(log-odds)가 선형**이다.

- $\pi(x)$는 사건이 일어날 확률이다.
- $1-\pi(x)$는 사건이 일어나지 않을 확률이다.
- $\pi(x)/(1-\pi(x))$는 오즈(odds)이다.
- 그 오즈에 로그를 취한 값이 설명변수 $x$에 대해 선형관계를 가진다.

이러한 선형화 덕분에 확률모형을 다루면서도 회귀계수 $(\beta_0,\beta_1)$에 대해 명확한 해석을 할 수 있다.

### 14.1.3 연결함수 (Link Function)와 다른 이항반응 모형

확률 $\pi(x)$를 어떤 단조함수(monotone function) $g$로 변환하여 $g(\pi(x))=\beta_0+\beta_1x$ 와 같이 모형화하는 것이 일반적인 이항반응 회귀의 기본 개념이다. 여기서 $g$를 연결함수(link function)라고 한다.  
연결함수의 선택에 따라 여러 모형이 나온다.

- 로지스틱 회귀모형(logistic regression model)
- 겜벨 모형(Gumbel model)
- 프로빗 모형(probit model)

그 중 로지스틱 모형은 결과 해석이 상대적으로 쉽기 때문에 가장 널리 사용된다. 특히 회귀계수를 오즈와 오즈비 관점에서 직접 해석할 수 있다는 장점이 있다.

### 14.1.4 확률모형의 형태 (Probability Model Form)

로짓 식 (16.3)을 다시 확률 형태로 풀면

$$
\pi(x)=\frac{\exp(\beta_0+\beta_1x)}{1+\exp(\beta_0+\beta_1x)} \tag{16.4}
$$

이 식은 설명변수 $x$가 변함에 따라 성공확률이 어떻게 바뀌는지를 직접 보여준다. $\beta_1>0$이면 $x$가 증가할수록 $\pi(x)$가 증가하고, $\beta_1<0$이면 감소한다.

### 14.1.5 오즈 (Odds)와 오즈비 (Odds Ratio)

로지스틱 회귀에서 회귀계수의 해석은 오즈와 오즈비를 통해 이루어진다.  
먼저 성공의 오즈는

$$
\frac{\pi(x)}{1-\pi(x)}
=\exp(\beta_0+\beta_1x)
=e^{\beta_0}(e^{\beta_1})^x
\tag{16.5}
$$

이 식은 $x$가 한 단위 증가할 때마다 오즈가 $e^{\beta_1}$배가 됨을 의미한다. 따라서 $e^{\beta_1}$는 오즈비(odds ratio)로 해석된다.  
오즈비를 식으로 쓰면

$$
\frac{P(y=1\mid x+1)/P(y=0\mid x+1)}
{P(y=1\mid x)/P(y=0\mid x)}
=\exp(\beta_1)
\tag{16.6}
$$

즉, 설명변수 $x$가 1 증가할 때 성공 오즈가 얼마나 배수적으로 변하는지를 나타내는 값이 $\exp(\beta_1)$이다.

#### 예제: 회귀계수의 오즈비 해석 (Interpretation of Odds Ratio)

$x$를 소득, $y$를 어떤 상품의 구입 여부라 하자. 여기서 $y=1$은 구입, $y=0$은 미구입이다. 만약 $\beta_1=3.72$이면 $\exp(3.72)\approx 42$ 이므로, 소득이 한 단위 증가할 때 그 상품을 구매할 오즈가 약 42배 증가한다고 해석한다.

이 예제는 로지스틱 회귀가 단순히 확률을 맞추는 모형이 아니라, 설명변수와 사건 발생 사이의 강도를 오즈비라는 형태로 정량화한다는 점을 잘 보여준다.

### 14.1.6 선형확률모형과의 차이 (Difference from Linear Probability Model)

이진 반응변수에 대해 단순선형회귀모형 $y=\beta_0+\beta_1x+\varepsilon$ 을 그대로 적용하면 다음과 같은 문제가 발생한다.

1. $\beta_0+\beta_1x$가 $[0,1]$ 범위를 벗어날 수 있다.
2. 오차항 $\varepsilon$가 정규분포를 따른다고 보기 어렵다.
3. 분산이 일정하지 않을 가능성이 크다.

로지스틱 회귀는 이러한 문제를 피하기 위해, 확률을 직접 선형으로 두지 않고 로짓을 선형으로 둔다. 따라서 이항반응 자료(binomial response data)에 더 적합한 모형이 된다.


## 14.2 로지스틱 모형의 추정 (Estimation of Logistic Model)

다중 로지스틱 회귀모형에서는 설명변수 벡터를 $\mathbf{x}^T=(x_0,x_1,\cdots,x_p), \quad x_0=1$ 로 두고, 성공확률을

$$
P(y=1\mid \mathbf{x})=\pi(\mathbf{x}^T\boldsymbol{\beta})
=\frac{\exp(\mathbf{x}^T\boldsymbol{\beta})}{1+\exp(\mathbf{x}^T\boldsymbol{\beta})}
\tag{16.7}
$$

로 가정한다. 여기서 $\boldsymbol{\beta}=(\beta_0,\beta_1,\cdots,\beta_p)^T$는 추정해야 할 회귀계수 벡터이다.

### 14.2.1 최대가능도추정법 (Maximum Likelihood Estimation)

로지스틱 회귀에서는 선형회귀에서와 같은 최소제곱법(least squares method)을 사용하지 않고, 최대가능도추정법(maximum likelihood estimation, MLE)을 사용한다. 이유는 반응변수 $y_i$가 연속형 정규오차를 가진 형태가 아니라, 각 관측치가 0 또는 1인 베르누이형 확률구조를 가지기 때문이다.  
관측값 $\{(y_i,\mathbf{x}_i), i=1,2,\dots,n\}$가 주어졌을 때 가능도함수(likelihood function)는

$$
L(\boldsymbol{\beta})
=\prod_{i=1}^{n}
\pi(\mathbf{x}_i^T\boldsymbol{\beta})^{y_i}
\{1-\pi(\mathbf{x}_i^T\boldsymbol{\beta})\}^{1-y_i}
\tag{16.8}
$$

이 식의 의미는 각 관측치가 실제 관측된 값 $y_i$를 가질 확률을 모두 곱한 것이다. $y_i=1$이면 해당 항은 $\pi(\mathbf{x}_i^T\boldsymbol{\beta})$가 되고, $y_i=0$이면 $1-\pi(\mathbf{x}_i^T\boldsymbol{\beta})$가 된다.  
가능도를 직접 최적화하기보다 보통 로그가능도함수(log-likelihood function)를 사용한다.

$$
\ln L(\boldsymbol{\beta})
=\sum_{i=1}^{n}
\Big[
y_i\ln \pi(\mathbf{x}_i^T\boldsymbol{\beta})
+ (1-y_i)\log(1-\pi(\mathbf{x}_i^T\boldsymbol{\beta}))
\Big]
\tag{16.9}
$$

로그를 취하면 곱이 합으로 바뀌어 계산이 쉬워지고, 최대화하는 해는 동일하게 유지된다.

### 14.2.2 로그가능도함수의 도함수 (Derivatives of Log-Likelihood)

로그가능도함수를 $\boldsymbol{\beta}$로 미분하면 점수함수(score function)는

$$
\frac{\partial \ln L(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}}
=\sum_{i=1}^{n}\mathbf{x}_i\,[y_i-\pi(\mathbf{x}_i^T\boldsymbol{\beta})]
=\mathbf{X}^T[\mathbf{y}-\pi(\boldsymbol{\beta})]
$$

또한 이차미분, 즉 헤시안 행렬(Hessian matrix)은

$$
\frac{\partial^2 \ln L(\boldsymbol{\beta})}{\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}^T}
= -\sum_{i=1}^{n}
\mathbf{x}_i\mathbf{x}_i^T\,\pi(\mathbf{x}_i^T\boldsymbol{\beta})\{1-\pi(\mathbf{x}_i^T\boldsymbol{\beta})\}
= -\mathbf{X}^T W(\boldsymbol{\beta}) \mathbf{X}
$$

- $\mathbf{X}$는 설계행렬(design matrix)이다.
- $\pi(\boldsymbol{\beta})=(\pi(\mathbf{x}_1^T\boldsymbol{\beta}),\dots,\pi(\mathbf{x}_n^T\boldsymbol{\beta}))^T$이다.
- $W(\boldsymbol{\beta})$는 $i$번째 대각원소가 $\pi(\mathbf{x}_i^T\boldsymbol{\beta})\{1-\pi(\mathbf{x}_i^T\boldsymbol{\beta})\}$인 대각행렬(diagonal matrix)이다.

이 구조는 로지스틱 회귀 추정이 반복적 수치해법을 필요로 한다는 점을 보여준다. 선형회귀처럼 닫힌형태 해(closed-form solution)를 즉시 얻을 수 없다. 이런 반복적 방법 중 하나가 다음에 설명할 Newton-Raphson 알고리즘이다.

### 14.2.3 Newton-Raphson 알고리즘 (Newton-Raphson Algorithm)

최대가능도추정량 $\hat{\boldsymbol{\beta}}$는 수치적 방법으로 구한다. 대표적인 방법이 Newton-Raphson 알고리즘 (Newton-Raphson algorithm)이다.  
업데이트 식은

$$
\hat{\boldsymbol{\beta}} = \boldsymbol{\beta}
- \left[
\frac{\partial^2 \ln L(\boldsymbol{\beta})}{\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}^T}
\right]^{-1}
\frac{\partial \ln L(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}}
$$

이며, 이를 이용하면 반복추정식은

$$
\hat{\boldsymbol{\beta}}^{(t+1)}
= \boldsymbol{\beta}^{(t)}
+ \left[\mathbf{X}^T W(\boldsymbol{\beta}^{(t)}) \mathbf{X}\right]^{-1}
\mathbf{X}^T\{\mathbf{y}-\pi(\boldsymbol{\beta}^{(t)})\} \tag{16.10}
$$

또는 같은 식을 다음의 보조변수 $z(\boldsymbol{\beta}^{(t)})$를 이용하여 표현할 수 있다.

$$
z(\boldsymbol{\beta}^{(t)})
= \mathbf{X}\boldsymbol{\beta}^{(t)}
+ W^{-1}(\boldsymbol{\beta}^{(t)})(\mathbf{y}-\pi(\boldsymbol{\beta}^{(t)}))
\tag{16.11}
$$

적절한 초기값, 예를 들어 $\boldsymbol{\beta}^{(0)}=0_{p+1}$을 두고 식 (16.10), (16.11)을 반복 적용하면 수렴할 때까지 회귀계수를 갱신할 수 있다.

### 14.2.4 반복가중최소제곱 (Iteratively Reweighted Least Squares, IRLS)

Newton-Raphson 갱신은 각 반복에서 다음 문제를 푸는 것과 동등하다.

$$
\boldsymbol{\beta}^{(t+1)}
\leftarrow
\arg\min_{\boldsymbol{\beta}}
\left\{
[z(\boldsymbol{\beta}^{(t)})-\mathbf{X}\boldsymbol{\beta}]^T
W(\boldsymbol{\beta}^{(t)})
[z(\boldsymbol{\beta}^{(t)})-\mathbf{X}\boldsymbol{\beta}]
\right\}
$$

즉, 매 반복마다 가중최소제곱문제 (weighted least squares problem)를 푸는 형태가 되므로, 이를 반복가중최소제곱 (iteratively reweighted least squares, IRLS) 추정이라고 부른다. IRLS의 의미는 다음과 같다.
1. 현재 추정값 $\boldsymbol{\beta}^{(t)}$를 기준으로 확률 $\pi(\boldsymbol{\beta}^{(t)})$를 계산한다.
2. 그 확률로부터 가중치행렬 $W(\boldsymbol{\beta}^{(t)})$를 만든다.
3. 보조반응변수 $z(\boldsymbol{\beta}^{(t)})$를 구성한다.
4. $z$를 반응변수로 하는 가중최소제곱 문제를 풀어 새로운 $\boldsymbol{\beta}^{(t+1)}$를 구한다.
5. 수렴할 때까지 반복한다.

이 방법은 로지스틱 회귀의 최대가능도추정을 계산적으로 구현하는 표준적인 절차이다.

### 14.2.5 추정 절차의 요약 (Summary of Estimation Procedure)

로지스틱 모형의 추정 흐름은 다음과 같이 정리된다.
1. 성공확률 설정:

   $$
   \pi(x^T\beta)=\frac{\exp(x^T\beta)}{1+\exp(x^T\beta)}
   $$

2. 관측자료로부터 가능도함수 $L(\beta)$를 구성한다.
3. 로그가능도함수 $\ln L(\beta)$를 만든다.
4. 점수함수와 헤시안 행렬을 구한다.
5. Newton-Raphson 또는 IRLS를 이용하여 $\beta$를 반복적으로 갱신한다.
6. 수렴한 값을 최대가능도추정량 $\hat{\beta}$로 채택한다.


## 14.3 2×2 분할표의 로지스틱 회귀 (Logistic Regression for 2×2 Tables)

자료의 수집은 연구자의 간섭 유무에 따라 크게 **관측연구 (observational study)** 와 **실험연구 (experimental study)** 로 나눌 수 있다. 실제로는 많은 자료가 관측연구에서 수집되며, 특히 임상역학 (clinical epidemiology)에서는 질병과 위험요인의 관련성을 평가하기 위하여 2×2 분할표 (2×2 contingency table)가 가장 자주 사용되는 자료 형식을 이룬다. 이러한 분할표는 질병 발생 여부와 위험인자 노출 여부처럼 두 개의 이진변수 (binary variables)로 구성된다.

임상역학에서 대표적인 연구 설계는 **코호트 연구 (cohort study)**, **사례-대조군 연구 (case-control study)**, **단면조사 연구 (cross-sectional study)** 로 구분된다. 로지스틱 회귀는 이들 서로 다른 설계에서 얻어진 2×2 분할표를 하나의 통일된 확률모형으로 설명할 수 있게 해 준다. 특히 **오즈비 (odds ratio, OR)** 를 중심으로 해석이 가능하다는 점에서 매우 중요하다.

### 14.3.1 관측연구의 세 가지 형태 (Three Types of Observational Studies)

코호트 연구 (cohort study)는 특정 위험요인에 노출된 집단과 노출되지 않은 집단을 일정 기간 동안 전향적 (prospective)으로 관찰하여 질병 발생률을 비교함으로써 위험요인과 질병 사이의 관련성을 조사하는 연구이다.

사례-대조군 연구 (case-control study)는 환자군과 대조군 각각에 대하여 과거 위험요인에 노출된 정도를 후향적 (retrospective)으로 비교하여 질병과 위험요인의 관련성을 밝히려는 연구이다.

단면조사 연구 (cross-sectional study)는 특정 시점 또는 특정 기간에 질병의 유병상태와 위험요인의 노출 상태를 동시에 조사하여 두 변수의 연관성을 파악하는 연구이다.

2×2 분할표는 이러한 연구들에서 공통적으로 등장하며, 연구설계가 다르더라도 오즈비를 통해 관련성의 강도를 비교할 수 있다.

#### 예제: 코호트 연구 (Cohort Study Example)

Pauling(1971)의 연구에서는 환자들을 임의로 두 집단으로 나누어 비타민 C와 위약 (placebo)을 투여하고, 각 집단에서 감기환자가 얼마나 발생하는지를 조사하였다. 이를 통해 비타민 C가 감기 예방에 도움이 되는가를 평가하고자 하였다.

표 16.1의 교차분할표는 다음과 같다.

* 비타민 C군: 감기 발생 예 17, 아니오 122, 총합 139
* 위약군: 감기 발생 예 31, 아니오 109, 총합 140

이 표는 처리변수와 질병 발생 여부로 이루어진 전형적인 2×2 분할표이다. 여기서 관심사는 비타민 C 처리가 감기 발생확률을 낮추는지 여부이다.

#### 예제: 사례-대조군 연구 (Case-Control Study Example)

Keller(1965)는 구강암과 흡연의 관계를 조사하기 위하여 구강암 환자와 정상인을 구분한 뒤 흡연 여부를 조사하였다. 이를 통해 흡연이 구강암과 관련이 있는지, 관련이 있다면 흡연자의 구강암 발생 가능성이 비흡연자에 비하여 얼마나 큰지를 알고자 하였다.

표 16.2의 교차분할표는 다음과 같다.

* 사례군: 흡연 예 484, 아니오 27, 총합 511
* 대조군: 흡연 예 385, 아니오 90, 총합 475

이 자료에서는 질병 발생 여부가 먼저 고정되어 있고, 그 안에서 노출 여부를 조사한다는 점이 코호트 연구와 다르다. 그럼에도 오즈비를 통해 관련성을 측정할 수 있다.

### 왜 분할표만이 아니라 로지스틱 회귀를 사용하는가 (Why Use Logistic Regression Beyond a Simple Table)

표본비율과 오즈비만으로도 2×2 분할표의 관련성을 설명할 수 있으므로, 굳이 로지스틱 회귀모형을 사용할 필요가 있는지 의문이 생길 수 있다. 그러나 실제 역학연구에서는 관심 위험요인 외에도 결과에 영향을 미칠 수 있는 여러 다른 요인들이 존재한다.

실험연구에서 랜덤화 (randomization)를 수행하더라도 이미 알려진 위험요인이나 관심 밖의 요인의 효과를 완전히 무시할 수는 없다. 따라서 이들 교란요인 (confounding factors)의 효과를 통제해야 한다. 로지스틱 회귀는 관련 변수들을 동시에 모형에 포함시켜, 관심 있는 위험요인의 순수한 효과를 추정할 수 있게 한다.

즉, 2×2 분할표는 단순 관련성 파악에는 유용하지만, 다수의 설명변수를 함께 고려하고 조정된 오즈비 (adjusted odds ratio)를 구하려면 로지스틱 회귀가 필요하다.


## 14.4 로지스틱 모형을 이용한 분류분석 (Classification using Logistic Model)

분류분석 (classification analysis)은 다변량 자료분석에서 매우 중요한 문제로, 모집단이 $K$개의 범주로 나누어져 있을 때 각 개체의 특성값을 이용하여 해당 개체가 어느 범주에 속하는지를 예측하는 절차이다. 예를 들면 의료자료를 이용하여 환자의 질병 단계를 예측하거나, 금융자료를 이용하여 대출 신청자의 부도 여부를 예측하는 문제 등이 이에 해당한다.

반응범주가 $K=2$인 경우에는 반응변수 $y$가 0 또는 1의 이항변수 (binary variable)가 되며, 이때 로지스틱 회귀모형을 이용하여 분류문제를 다룰 수 있다.

### 14.4.1 이진 분류문제로서의 로지스틱 회귀 (Logistic Regression as a Binary Classifier)

설명변수벡터 $\mathbf{x}=(x_1,x_2,\dots,x_p)^T$ 가 주어졌을 때, 로지스틱 회귀는 사건 발생확률 $P(y=1\mid \mathbf{x})$을 추정한다. 이 추정확률을 그대로 해석에 사용할 수도 있고, 어떤 절단값 (cut-off) $c$와 비교하여 범주를 결정할 수도 있다. 분류규칙은 다음과 같다.

* $P(y=1\mid \mathbf{x})>c$이면 관측치를 범주 $y=1$로 분류한다.
* $P(y=1\mid \mathbf{x})<c$이면 관측치를 범주 $y=0$로 분류한다.

즉, 로지스틱 회귀는 확률을 추정하는 회귀모형이면서 동시에 그 확률에 기반한 분류기 (classifier)로도 작동한다.

### 14.4.2 절단값의 선택 (Choice of Cut-off)

절단값 $c$는 단순히 0.5로 고정되는 것이 아니라, 문제의 성격에 따라 신중하게 선택하여야 한다. 절단값 결정 시 고려할 사항은 다음과 같다.

첫째, 사전정보 (prior information)를 고려한다. 만약 $y=1$인 자료가 상대적으로 많이 나타나는 상황이라면 절단값을 더 작게 잡을 수 있다. 반대로 $y=1$이 매우 드문 사건이라면 절단값을 더 크게 설정할 수도 있다.

둘째, 손실함수 (loss function)를 고려한다. 두 종류의 오분류가 동일한 손실을 가지지 않을 수 있기 때문이다. 예를 들어 $y=1$ 자료를 잘못 분류하는 손실이 $y=0$ 자료를 잘못 분류하는 손실보다 훨씬 크다면, $y=1$을 놓치지 않기 위해 절단값을 작게 정할 수 있다.

셋째, 문제영역의 전문가 판단, 민감도 (sensitivity), 특이도 (specificity) 같은 성능 기준도 함께 고려할 수 있다.

#### 예제: 스팸 메일 분류에서의 절단값 (Cut-off in Spam Classification)

이메일을 정상 메일과 스팸 메일로 분류하는 문제를 생각하자. 정상 메일을 스팸으로 잘못 분류하면 중요한 메일이 필터링되어 사용자에게 큰 불편을 줄 수 있다. 반면 스팸 메일을 정상으로 분류하는 경우의 손실은 상대적으로 작을 수 있다. 이 경우에는 정상 메일을 스팸으로 오분류하는 손실을 크게 보아, 절단값을 조정함으로써 그러한 오류를 줄이도록 설계할 수 있다.

즉, 절단값은 단순한 수치가 아니라 의사결정의 비용 구조를 반영하는 매개변수이다.

### 14.4.3 로지스틱 회귀 기반 분류기의 선형성 (Linearity of Logistic Regression Classifier)

위와 같이 정의한 로지스틱 회귀 기반 분류자는 선형분류자 (linear classifier)이다. 이를 보이기 위해 $P(y=1\mid \mathbf{x})>c$ 라는 분류규칙을 로짓 형태로 바꾸어 본다. 먼저

$$ \log\left[\frac{P(y=1\mid \mathbf{x})}{1-P(y=1\mid \mathbf{x})}\right]
> \ln\left(\frac{c}{1-c}\right)
$$

이면 범주 1로 분류하는 것과 동일하다. 로지스틱 회귀모형에서

$$\log\left[\frac{P(y=1\mid \mathbf{x})}{1-P(y=1\mid \mathbf{x})}\right]
= \beta_0+\beta_1x_1+\cdots+\beta_px_p$$

이므로, 위 규칙은

$$ \beta_0+\beta_1x_1+\cdots+\beta_px_p > \ln\left(\frac{c}{1-c}\right) $$

와 같아진다.  
따라서 분류경계 (decision boundary)는 설명변수에 대한 선형식으로 주어진다. 즉, 로지스틱 회귀는 확률모형으로는 비선형적인 S자형 반응함수를 사용하지만, 실제 분류규칙의 경계는 선형이다.

이미지에 제시된 단일 설명변수 표기에서는

$$\beta_0+\beta_1x>\log(c^*)$$

의 형태로 정리되며, 여기서 $c^*=\frac{c}{1-c}$ 이다. 이 역시 분류경계가 선형임을 보여 준다.

#### 증명: 로지스틱 분류경계의 선형성 (Proof of Linear Decision Boundary)

분류규칙 $P(y=1\mid \mathbf{x})>c$에서 출발한다.

$$ P(y=1\mid \mathbf{x})>c \iff \frac{P(y=1\mid \mathbf{x})}{1-P(y=1\mid \mathbf{x})}>\frac{c}{1-c}$$

이고, 양변에 로그를 취하면

$$ \log\left[\frac{P(y=1\mid \mathbf{x})}{1-P(y=1\mid \mathbf{x})}\right] >\log\left(\frac{c}{1-c}\right)$$

을 얻는다. 그런데 로지스틱 회귀에서 좌변은 설명변수의 선형식이므로, 최종 분류규칙도 설명변수의 선형부등식으로 표현된다. 따라서 로지스틱 회귀 기반 분류자는 선형분류자이다.

### 14.4.4 분류분석에서의 해석 (Interpretation in Classification)

로지스틱 회귀를 분류분석에 사용하는 경우, 핵심은 단순히 "어느 범주에 넣을 것인가"만이 아니다. 먼저 각 개체가 범주 1에 속할 확률을 추정하고, 그 다음 문제 상황에 맞는 절단값을 적용하여 범주를 결정한다는 점이 중요하다. 이 점에서 로지스틱 회귀는 결과를 확률적으로 해석할 수 있는 분류모형이다.

또한 설명변수의 효과가 회귀계수와 오즈비의 형태로 해석 가능하므로, 단순 예측뿐 아니라 어떤 변수들이 분류에 크게 기여하는가를 이해하는 데도 유용하다. 따라서 로지스틱 회귀는 예측모형 (predictive model)이면서 동시에 설명모형 (explanatory model)의 성격도 가진다.

### 14.4.5 로지스틱 회귀와 판별분석의 관련성 (Relation to Discriminant Analysis)

분류분석의 대표적인 다른 방법으로 Fisher의 선형판별분석 (linear discriminant analysis, LDA)이 있다. 로지스틱 회귀 역시 이항분류 문제에서 널리 사용되며, 특히 반응변수의 조건부 확률 $P(y=1\mid \mathbf{x})$를 직접 추정할 수 있다는 점에서 실용적이다. 따라서 로지스틱 회귀는 판별분석과 더불어 이진 분류문제의 기본 도구로 자리 잡는다.

   
## 핵심 정리 (Key Takeaways)

* 2×2 분할표 (2×2 contingency table)는 관측연구에서 질병과 위험요인 간 관련성을 표현하는 가장 기본적인 자료형식이다.
* 코호트 연구 (cohort study), 사례-대조군 연구 (case-control study), 단면조사 연구 (cross-sectional study)는 모두 오즈비 (odds ratio, OR)를 통해 관련성을 평가할 수 있다.
* 2×2 분할표는 설명변수가 0과 1만 갖는 가장 단순한 로지스틱 회귀모형으로 해석할 수 있다.
* 분할표의 표본비율은 로지스틱 회귀의 최대가능도추정량 (MLE)과 일치한다.
* 로지스틱 회귀는 교란요인 (confounding factors)을 통제하기 위하여 여러 설명변수를 동시에 포함할 수 있으므로, 단순 분할표보다 훨씬 강력하다.
* 로지스틱 회귀는 $P(y=1\mid \mathbf{x})$를 추정한 뒤 절단값 (cut-off)을 적용하여 분류하는 확률기반 분류기 (probability-based classifier)이다.
* 절단값은 사전정보 (prior information), 손실함수 (loss function), 민감도 (sensitivity), 특이도 (specificity) 등을 고려하여 정한다.
* 로지스틱 회귀의 분류경계는 선형식으로 표현되므로 선형분류자 (linear classifier)가 된다.
