# 다차원 확률변수의 확률분포 *(Multivariate Random Variables and Their Distributions)*

## 두 확률변수의 분포 *(Joint Distribution of Two Random Variables)*

### 2.1 두 확률변수의 분포
일반적으로 랜덤한 실험에서 관심의 대상이 되는 확률변수는 여러 개다. 이 절에서는 두 확률변수의 분포를 나타내는 방법을 정리한다.

#### 예시 2.1.1 (공 추출: 이산형 결합분포)
두 개의 흰 공, 세 개의 검은 공, 네 개의 빨간 공이 들어있는 상자에서 공 3개를 함께 꺼낸다.  
추출되는 흰 공의 개수를 $X$, 검은 공의 개수를 $Y$라고 하면, 결합확률질량함수는 다음과 같다.

- 전체 공 개수: $9$
- 추출 개수: $3$
- 빨간 공 개수: $3-x-y$

$P(X=x, Y=y)=\dfrac{\binom{2}{x}\binom{3}{y}\binom{4}{3-x-y}}{\binom{9}{3}},\quad x=0,1,2,\ y=0,1,2,3,\ x+y\le 3.$

#### 표 2.1.1 두 확률변수 $X,Y$의 확률분포표
(각 셀은 $P(X=x,Y=y)$이며, 오른쪽/아래 합계는 주변확률이다.)

| $X\backslash Y$ | $Y=0$ | $Y=1$ | $Y=2$ | $Y=3$ | 합계 |
|---:|---:|---:|---:|---:|---:|
| $X=0$ | $4/84$ | $18/84$ | $12/84$ | $1/84$ | $35/84$ |
| $X=1$ | $12/84$ | $24/84$ | $6/84$ | $0$ | $42/84$ |
| $X=2$ | $4/84$ | $3/84$ | $0$ | $0$ | $7/84$ |
| 합계 | $20/84$ | $45/84$ | $18/84$ | $1/84$ | $1$ |

### 이산형 이변량 확률벡터의 확률질량함수(결합확률질량함수)
두 확률변수 $(X,Y)$가 가질 수 있는 순서쌍들의 집합이 $\{ (x_j,y_k)\mid j=1,2,\ldots,\ k=1,2,\ldots \}$일 때, 각 순서쌍에 확률을 대응시키는 함수 $f$를  
$f(x_j,y_k)=P(X=x_j, Y=y_k)$로 정의한다. 이를 (이산형) 결합확률질량함수라 한다. 성질은 다음과 같다.

- 비음성 및 지지집합 밖에서 0
  - $f(x,y)\ge 0\ \forall x,y,$ 그리고 $(x,y)\notin\{(x_j,y_k)\}$이면 $f(x,y)=0$이다.
- 전체 확률은 1
  - $\sum_x\sum_y f(x,y)=\sum_{j=1}^\infty\sum_{k=1}^\infty f(x_j,y_k)=1.$
- 직사각형 영역 확률은 합으로 계산
  - $\sum_{x: a\le x\le b}\ \sum_{y: c\le y\le d} f(x,y)=P(a\le X\le b,\ c\le Y\le d).$

> 주: 이차원 확률변수를 이변량(bivariate) 확률벡터(random vector)라고도 한다.

### 연속형 이차원 확률변수의 확률밀도함수(결합확률밀도함수)
$X,Y$가 모두 실수 구간의 값을 가지며 확률이 적분으로 주어질 때, $(X,Y)$를 연속형 이차원 확률변수(이변량 확률벡터)라 한다.  
이때 함수 $f$가  $$\int_{y:c\le y\le d}\int_{x:a\le x\le b} f(x,y)\,dx\,dy = P(a\le X\le b,\ c\le Y\le d)\ (a<b,\ c<d)$$  

를 만족하면 $f$를 $(X,Y)$의 결합확률밀도함수라 한다. 성질은 다음과 같다.
- 비음성
  - $f(x,y)\ge 0\ \forall x,y.$
- 전체 적분은 1
  - $\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty} f(x,y)\,dy\,dx=1.$
- 직사각형 영역 확률은 이중적분
  - $\int_a^b\int_c^d f(x,y)\,dy\,dx = P(a\le X\le b,\ c\le Y\le d)\ (a<b,\ c<d).$

#### 예시 2.1.2 (연속형: 정규화 상수 및 영역확률)
다음 함수가 결합확률밀도함수가 되기 위한 상수 $c$를 구하고, 이 함수가 $(X,Y)$의 결합확률밀도함수일 때 $P(X\ge 2, Y\ge 3)$을 구한다.

$f(x,y)=c e^{-x-y}\,\mathbf{1}_{\{0\le x\le y\}}.$

**1) 상수 $c$ 결정**  
전체 확률이 1이어야 하므로  
$\int_0^{+\infty}\int_{y=x}^{+\infty} c e^{-x-y}\,dy\,dx=1.$

안쪽 적분:  
$\int_x^{+\infty} c e^{-x-y}\,dy = c e^{-x}\int_x^{+\infty} e^{-y}\,dy = c e^{-x}e^{-x}=c e^{-2x}.$

따라서  
$\int_0^{+\infty} c e^{-2x}\,dx = c\left[-\dfrac{e^{-2x}}{2}\right]_0^{+\infty}=\dfrac{c}{2}=1,$  
이므로 $c=2$이다.

**2) $P(X\ge 2, Y\ge 3)$ 계산**  
$P(X\ge 2, Y\ge 3)=\int_{x=2}^{+\infty}\int_{y=3}^{+\infty} 2e^{-x-y}\mathbf{1}_{\{0\le x\le y\}}\,dy\,dx.$

조건 $y\ge 3$과 $y\ge x$를 합치면 $y\ge \max(3,x)$이므로  
$\int_{y=3}^{+\infty} 2e^{-x-y}\mathbf{1}_{\{x\le y\}}\,dy = \int_{\max(3,x)}^{+\infty} 2e^{-x-y}\,dy = 2e^{-x}e^{-\max(3,x)}.$
> 주: $a\vee b$는 $\max(a,b)$, $a\wedge b$는 $\min(a,b)$를 뜻한다.

따라서 $x\in[2,3)$와 $x\in[3,\infty)$로 나누어  
$P(X\ge 2, Y\ge 3)=\int_2^3 2e^{-x}e^{-3}\,dx + \int_3^{+\infty} 2e^{-x}e^{-x}\,dx.$

계산하면  
$\int_2^3 2e^{-x-3}\,dx = 2e^{-3}[-e^{-x}]_2^3 = 2(e^{-5}-e^{-6}),$  
$\int_3^{+\infty} 2e^{-2x}\,dx = [-e^{-2x}]_3^{+\infty}=e^{-6}.$

따라서  
$P(X\ge 2, Y\ge 3)=2e^{-5}-e^{-6}\approx 0.011.$

### 주변분포와 주변확률밀도함수(주변확률질량함수) *(Marginal Distribution and Marginal Probability Density/Mass Function)*
두 확률변수 $(X,Y)$가 주어지면 결합분포(joint probability)뿐 아니라 $X$ 단독, $Y$ 단독의 분포(marginal probability)도 고려한다.  
$P(a\le X\le b)=P(a\le X\le b, -\infty<Y<+\infty)$이고,

- $(X,Y)$가 이산형이면 $P(a\le X\le b)=\sum_{x:a\le x\le b}\sum_y f(x,y)$,
- $(X,Y)$가 연속형이면 $P(a\le X\le b)=\int_a^b\int_{-\infty}^{+\infty} f(x,y)\,dy\,dx$ 이다.

따라서 주변함수는  
- 이산형: $f_1(x)=\sum_y f(x,y)$, $f_2(y)=\sum_x f(x,y)$
- 연속형: $f_1(x)=\int_{-\infty}^{+\infty} f(x,y)\,dy$, $f_2(y)=\int_{-\infty}^{+\infty} f(x,y)\,dx$

로 정의된다.

#### 예시 2.1.3 (예시 2.1.1의 주변분포)
예시 2.1.1에서  
$f(x,y)=\dfrac{\binom{2}{x}\binom{3}{y}\binom{4}{3-x-y}}{\binom{9}{3}}$ 이다.

**1) $X$의 주변확률질량함수**  
$f_1(x)=\sum_y f(x,y)=\dfrac{\binom{2}{x}}{\binom{9}{3}}\sum_{y=0}^{3-x}\binom{3}{y}\binom{4}{3-x-y}.$

조합항등식 $\sum_{y=0}^{\ell}\binom{m}{y}\binom{n}{\ell-y}=\binom{m+n}{\ell}$을 쓰면  
$\sum_{y=0}^{3-x}\binom{3}{y}\binom{4}{3-x-y}=\binom{7}{3-x}$ 이다.

따라서 $f_1(x)=\dfrac{\binom{2}{x}\binom{7}{3-x}}{\binom{9}{3}},\ x=0,1,2.$

>**참고: 조합항등식**  
>조합항등식 $\sum_{k=0}^{\ell}\binom{m}{k}\binom{n}{\ell-k}=\binom{m+n}{\ell}$는 **Vandermonde's identity(반데르몽드 항등식)** 로 알려져 있다.  
>이는 $m$개와 $n$개로 구성된 총 $m+n$개의 객체에서 $\ell$개를 선택하는  방법의 수가,  
>첫 번째 그룹에서 $k$개, 두 번째 그룹에서 $\ell-k$개를 선택하는 모든 경우의 합과 같다는 것을 의미한다.  
>
>대수적 설명: $(1+t)^m(1+t)^n = (1+t)^{m+n}$의 양변에서 $t^\ell$의 계수를 비교하면,  
>좌변은 $\sum_{k=0}^{\ell}\binom{m}{k}\binom{n}{\ell-k}$이고, 우변은 $\binom{m+n}{\ell}$이므로 항등식이 성립한다.

**2) $Y$의 주변확률질량함수**  
$f_2(y)=\sum_x f(x,y)=\dfrac{\binom{3}{y}}{\binom{9}{3}}\sum_{x=0}^{3-y}\binom{2}{x}\binom{4}{3-y-x}.$

같은 항등식으로  
$\sum_{x=0}^{3-y}\binom{2}{x}\binom{4}{3-y-x}=\binom{6}{3-y}$ 이다.

따라서 $f_2(y)=\dfrac{\binom{3}{y}\binom{6}{3-y}}{\binom{9}{3}},\ y=0,1,2,3.$

#### 예시 2.1.4 (예시 2.1.2의 주변확률밀도함수)
예시 2.1.2에서 $f(x,y)=2e^{-x-y}\mathbf{1}_{\{0\le x\le y\}}$ 이다.  
**1) $X$의 주변확률밀도함수**  
$f_1(x)=\int_{-\infty}^{+\infty}2e^{-x-y}\mathbf{1}_{\{0\le x\le y\}}\,dy = \int_{y=x}^{+\infty}2e^{-x-y}\,dy\,\mathbf{1}_{\{x\ge 0\}}$  
$=2e^{-2x}\mathbf{1}_{\{x\ge 0\}}.$  
**2) $Y$의 주변확률밀도함수**  
$f_2(y)=\int_{-\infty}^{+\infty}2e^{-x-y}\mathbf{1}_{\{0\le x\le y\}}\,dx = \int_0^{y}2e^{-x-y}\,dx\,\mathbf{1}_{\{y\ge 0\}}$  
$=2e^{-y}(1-e^{-y})\mathbf{1}_{\{y\ge 0\}}.$

### 결합누적분포함수(Joint CDF)와 결합확률함수의 관계
결합누적분포함수는 $F(x,y)=P(X\le x, Y\le y)$로 정의된다.

#### (a) 이산형인 경우
가능한 값들이 $x_1<x_2<\cdots$, $y_1<y_2<\cdots$일 때  
$F(x_m,y_n)=\sum_{j=1}^m\sum_{k=1}^n f(x_j,y_k).$

또한  
$f(x_j,y_k)=P(X=x_j, Y=y_k) = \{F(x_j,y_k)-F(x_{j-1},y_k)\}-\{F(x_j,y_{k-1})-F(x_{j-1},y_{k-1})\}.$
> **설명:**  
> 이산형 확률변수의 경우, 점확률 $P(X=x_j, Y=y_k)$는 누적분포함수의 차분으로 표현된다.  
> 직사각형 영역 $\{x_{j-1}<X\le x_j,\ y_{k-1}<Y\le y_k\}$의 확률이 바로 $P(X=x_j, Y=y_k)$이므로,  
> 포함-배제 원리(inclusion-exclusion principle)를 적용하면  
> $$P(x_{j-1}<X\le x_j,\ y_{k-1}<Y\le y_k) = F(x_j,y_k)-F(x_{j-1},y_k)-F(x_j,y_{k-1})+F(x_{j-1},y_{k-1})$$
> 이 성립한다. 이는 큰 직사각형 $\{X\le x_j, Y\le y_k\}$에서 왼쪽 영역 $\{X\le x_{j-1}, Y\le y_k\}$와  
> 아래쪽 영역 $\{X\le x_j, Y\le y_{k-1}\}$을 빼고, 중복으로 뺀 왼쪽 아래 영역 $\{X\le x_{j-1}, Y\le y_{k-1}\}$을 더하는 것이다.

#### (b) 연속형인 경우
$F(x,y)=\int_{-\infty}^{x}\int_{-\infty}^{y} f(t,u)\,du\,dt.$  
$f$가 $(x,y)$에서 연속이면 $f(x,y)=\dfrac{\partial^2}{\partial x\,\partial y}F(x,y)$ 이다.

#### 예시 2.1.5 (예시 2.1.2의 결합누적분포함수)
$f(t,u)=2e^{-t-u}\mathbf{1}_{\{0\le t\le u\}}$일 때  
$F(x,y)=\int_{-\infty}^{x}\int_{-\infty}^{y}2e^{-t-u}\mathbf{1}_{\{0\le t\le u\}}\,du\,dt.$

먼저 $u$에 대해 적분하면 ($y\ge t\ge 0$에서만 기여)  
$\int_{-\infty}^{y}2e^{-t-u}\mathbf{1}_{\{0\le t\le u\}}\,du = 2e^{-t}\int_t^{y}e^{-u}\,du\,\mathbf{1}_{\{y\ge t\ge 0\}}$  
$=2e^{-t}(e^{-t}-e^{-y})\mathbf{1}_{\{y\ge t\ge 0\}}.$

따라서  
$F(x,y)=\int_{0}^{x\wedge y}2\left(e^{-2t}-e^{-t-y}\right)dt\,\mathbf{1}_{\{x\wedge y\ge 0\}}.$

계산하면  
$F(x,y)=\left\{1-e^{-2(x\wedge y)}+2e^{-y}\left(e^{-(x\wedge y)}-1\right)\right\}\mathbf{1}_{\{x\ge 0,\ y\ge 0\}}.$

### 주변누적분포함수(Marginal CDF)
X의 주변누적분포함수: $F_1(x)=P(X\le x)$,  
Y의 주변누적분포함수: $F_2(y)=P(Y\le y)$로 정의된다.  
확률측도의 연속성에 의해  
$F_1(x)=\lim_{y\to +\infty}F(x,y)$, $F_2(y)=\lim_{x\to +\infty}F(x,y)$ 이다.

#### 예시 2.1.6 (예시 2.1.5로부터 주변누적분포함수 구하기)
$F(x,y)=\left\{1-e^{-2(x\wedge y)}+2e^{-y}\left(e^{-(x\wedge y)}-1\right)\right\}\mathbf{1}_{\{x\ge 0,\ y\ge 0\}}$로부터

- $F_1(x)=\lim_{y\to +\infty}F(x,y)=(1-e^{-2x})\mathbf{1}_{\{x\ge 0\}}$,
- $F_2(y)=\lim_{x\to +\infty}F(x,y)=(1+e^{-2y}-2e^{-y})\mathbf{1}_{\{y\ge 0\}}$.

또한 이를 미분하면  
$f_1(x)=2e^{-2x}\mathbf{1}_{\{x\ge 0\}}$, $f_2(y)=2e^{-y}(1-e^{-y})\mathbf{1}_{\{y\ge 0\}}$로서 예시 2.1.4와 일치한다.

### 정리 2.1.1 결합누적분포함수의 성질
$F(x,y)=P(X\le x, Y\le y)$는 다음을 만족한다.  

**(a) 증가성**  
(i) 모든 $h>0,k>0$에 대해  
$F(x,y)\le F(x+h,y)$, $F(x,y)\le F(x,y+k)$.  
(ii) $x_1<x_2$, $y_1<y_2$이면  
$F(x_2,y_2)-F(x_1,y_2)-F(x_2,y_1)+F(x_1,y_1)\ge 0$.

**(b) 전체변동(극한값)**  
모든 $a,b$에 대해  
$\lim_{x\to-\infty}F(x,b)=0$,  
$\lim_{y\to-\infty}F(a,y)=0$,  
$\lim_{x\to+\infty, y\to+\infty}F(x,y)=1$.

**(c) 오른쪽 연속성**  
모든 $x,y$에 대해  
$\lim_{h\downarrow 0}F(x+h,y)=F(x,y)$,  
$\lim_{k\downarrow 0}F(x,y+k)=F(x,y)$.
> **참고: 극한 표기 $\lim_{h\downarrow 0}$와 $\lim_{h\to 0}$의 차이**  
> - $\lim_{h\downarrow 0} f(x+h)$: $h$가 **양수 값에서** 0으로 접근함을 명시 (오른쪽 극한, right-hand limit)
> - $\lim_{h\to 0} f(x+h)$: $h$가 양수 또는 음수 값에서 0으로 접근 (양방향 극한, two-sided limit)
>
> 누적분포함수의 **오른쪽 연속성**을 강조하기 위해 $h\downarrow 0$ 표기를 사용한다.  
> 반대로 $h\uparrow 0$는 $h$가 음수 값에서 0으로 접근함을 의미하며(왼쪽 극한, left-hand limit),  
> 이는 $F(x-,y)=\lim_{h\downarrow 0}F(x-h,y)$와 같이 왼쪽 극한값을 나타낼 때 사용된다.

또한 결합누적분포함수는 왼쪽 연속이 아닐 수 있으며, 왼쪽 극한을  
$F(x-,y)=\lim_{h\downarrow 0}F(x-h,y)=P(X<x, Y\le y)$  
로 두면, 불연속의 크기는  
$P(X=x, Y\le y)=F(x,y)-F(x-,y)$  
로 주어진다.

## 결합확률분포의 특성치 *(Characteristics of Joint Distributions)*

### 결합확률분포에 대한 기대값 *(Expectation with Respect to a Joint Distribution)*
이변량 확률변수 $(X,Y)$의 결합분포는 각 주변분포가 제공하지 못하는
**두 변수 사이의 관계**를 포함한다. 
이를 정량화하기 위한 기본 도구가 **결합기대값(joint expectation)** 이다.

#### 정의 2.2.1 (결합확률분포에 대한 기대값, Joint Expectation)
$(X,Y)$의 결합확률질량함수 또는 결합확률밀도함수를 $f(x,y)$라 하고,
실수값 함수 $g:\mathbb R^2\to\mathbb R$에 대하여 다음이 유한하게 정의되면 이를 $g(X,Y)$의 기대값이라 한다.

* **이산형(discrete case)**
  $$
  E[g(X,Y)] = \sum_x\sum_y g(x,y)f(x,y)
  $$

* **연속형(continuous case)**
  $$
  E[g(X,Y)] = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} g(x,y)f(x,y)dydx
  $$
이는 측도론적으로 $g(X,Y)$의 **Lebesgue 적분**에 해당한다.

#### 예제 2.2.1 (결합기대값 계산)
결합확률밀도함수가
$$
f(x,y)=120xy(1-x-y)\mathbf 1_{{x\ge0;y\ge0;x+y\le1}}
$$
일 때,
$$
E(XY)=\iint xy f(x,y)dxdy =\frac{2}{21}
$$
이는 **단순한 곱 $E(X)E(Y)$와 일반적으로 다름**을 보여주는 대표적 예다.

### 결합기대값의 성질 *(Properties of Joint Expectation)*
### 정리 2.2.1 (선형성과 단조성, Linearity and Monotonicity)
결합기대값이 실수로 존재한다고 가정하면 다음이 성립한다.

1. **선형성(Linearity)**
   $$
   E[c_1 g_1(X,Y)+c_2 g_2(X,Y)]
   =c_1E[g_1(X,Y)]+c_2E[g_2(X,Y)]
   $$
2. **단조성(Monotonicity)**
   $$
   g_1(X,Y)\le g_2(X,Y);\text{a.s.} \Rightarrow E[g_1(X,Y)]\le E[g_2(X,Y)]
   $$

증명: 이중합의 선형성, 이중적분의 선형성□

### 주변분포를 이용한 기대값 *(Expectation via Marginal Distributions)*
$g(X)$처럼 **하나의 변수에만 의존**하는 함수의 경우,
결합분포 대신 **주변분포(marginal distribution)** 만으로 기대값을 계산할 수 있다.  
$X$의 주변확률밀도함수를 $f_1(x)=\int f(x,y)dy$라 하면,
$$
E[g(X)] = \int_{-\infty}^{\infty} g(x)f_1(x)dx
$$

### 공분산과 상관계수 *(Covariance and Correlation Coefficient)*
**정의 2.2.2 (공분산, Covariance)**  
확률변수 $X,Y$의 평균을 $\mu_X,\mu_Y$라 할 때,
$$
\mathrm{Cov}(X,Y)=E[(X-\mu_X)(Y-\mu_Y)]
$$
이는 두 변수의 **동시 변동 방향(direction of joint variation)** 을 측정한다.  
그리고 각 변수의 측정단위에 의존하므로, Cov가 측정 단위에 기인한건지 변화량의 크기에 기인한건지 구분하기 위해 상관계수를 고안한다.  
#### 정의 2.2.3 (상관계수, Correlation Coefficient)
$\mathrm{Var}(X),\mathrm{Var}(Y)>0$일 때,
$$
\mathrm{Corr}(X,Y)
=\frac{\mathrm{Cov}(X,Y)}{\sqrt{\mathrm{Var}(X)}\sqrt{\mathrm{Var}(Y)}}
$$
이는 **무차원(dimensionless)** 이며 선형 의존성의 크기를 나타낸다.
* $\rho<0$: 음의 선형 관계
* $\rho=0$: 선형 상관 없음 (독립과는 다름, 독립이면 $\rho=0$이지만, $\rho=0$이라고 해서 독립인 것은 아님)
* $\rho>0$: 양의 선형 관계
* $|\rho|\to1$: 분포가 직선 주변으로 집중

### 정리 2.2.2  공분산의 성질 *(Properties of Covariance)*
(우변의 기댓값, 공분산이 모두 실수라는 전제 하에 성립)
1. $\mathrm{Cov}(X,Y)=\mathrm{Cov}(Y,X)$
2. $\mathrm{Cov}(X,X)=\mathrm{Var}(X)$
3. $\mathrm{Cov}(aX+b,cY+d)=ac\mathrm{Cov}(X,Y)$
4. $\mathrm{Cov}(X,Y)=E(XY)-E(X)E(Y)$
  - 식 전개하면 바로 나옴: $E[(X-\mu_X)(Y-\mu_Y)]=E(XY)-\mu_X\mu_Y$ □

### 정리 2.2.3 상관계수의 성질 *(Properties of Correlation)*
상관계수 $\rho=\mathrm{Corr}(X,Y)$에 대하여
1. $\mathrm{Var}\left(\frac{Y-\mu_Y}{\sigma_Y}-\rho\frac{X-\mu_X}{\sigma_X}\right)=1-\rho^2$
2. $-1\le\rho\le1$
3. $|\rho|=1$ ⇔ $Y$는 $X$의 **선형함수(linear function)** a.s.  
다르게 표현하면,  
$|\rho|=1$ ⇔ $P\left(\frac{Y-\mu_Y}{\sigma_Y}=\rho\frac{X-\mu_X}{\sigma_X}\right) = 1$
    - $\rho=1$ ⇔ $P\left(\frac{Y-\mu_Y}{\sigma_Y}=\frac{X-\mu_X}{\sigma_X}\right) = 1$
    - $\rho=-1$ ⇔ $P\left(\frac{Y-\mu_Y}{\sigma_Y}=-\frac{X-\mu_X}{\sigma_X}\right) = 1$
    - 직선관계를 나타내는 특성치: 상관계수 절댓값이 커질수록 $(X, Y)$ 분포는 직선에 가깝게 분포한다.
  
#### 증명
**(1) 증명**  
분산의 정의와 공분산의 성질을 이용하면

$$
\begin{align}
\mathrm{Var}\left(\frac{Y-\mu_Y}{\sigma_Y}-\rho\frac{X-\mu_X}{\sigma_X}\right)
&= E\left[(\frac{Y-\mu_Y}{\sigma_Y}-\rho\frac{X-\mu_X}{\sigma_X})^2\right]\\
&= \mathrm{Var}\left(\frac{Y-\mu_Y}{\sigma_Y}\right) + \rho^2\mathrm{Var}\left(\frac{X-\mu_X}{\sigma_X}\right) - 2\rho\mathrm{Cov}\left(\frac{Y-\mu_Y}{\sigma_Y}, \frac{X-\mu_X}{\sigma_X}\right)\\
&= 1 + \rho^2 - 2\rho\cdot\frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y}\\
&= 1 + \rho^2 - 2\rho^2\\
&= 1 - \rho^2.
\end{align}
$$

**(2) 증명**  
(1)에서 분산은 항상 비음이므로 $1-\rho^2\ge 0$, 즉 $\rho^2\le 1$이다.  
따라서 $-1\le\rho\le 1$이다.

**(3) 증명**  
($\Leftarrow$) $Y=aX+b$ (a.s.)이면  
$$\mathrm{Cov}(X,Y)=a\mathrm{Var}(X), \quad \mathrm{Var}(Y)=a^2\mathrm{Var}(X)$$
이므로
$$\rho=\frac{a\mathrm{Var}(X)}{|a|\mathrm{Var}(X)}=\mathrm{sign}(a)=\pm 1.$$

($\Rightarrow$) $|\rho|=1$이면 (1)에서 $1-\rho^2=0$이므로
$$\mathrm{Var}\left(\frac{Y-\mu_Y}{\sigma_Y}-\rho\frac{X-\mu_X}{\sigma_X}\right)=0.$$
$Z = \frac{Y-\mu_Y}{\sigma_Y}-\rho\frac{X-\mu_X}{\sigma_X}$라 하면, 
$Z=\mathrm{E}\left[Z\right]=0 \quad\text{a.s.}$  
따라서
$$\frac{Y-\mu_Y}{\sigma_Y}-\rho\frac{X-\mu_X}{\sigma_X}=0 \quad\text{a.s.}$$
따라서
$$Y=\mu_Y+\rho\sigma_Y\frac{X-\mu_X}{\sigma_X}=\left(\rho\frac{\sigma_Y}{\sigma_X}\right)X+\left(\mu_Y-\rho\frac{\sigma_Y}{\sigma_X}\mu_X\right) \quad\text{a.s.}$$
즉, $Y$는 $X$의 선형함수이다. □

>참고: Cauchy–Schwarz 부등식
>$$
>|E(VW)|\le\sqrt{E(V^2)}\sqrt{E(W^2)}
>$$
>여기서 $V=\frac{X-\mu_X}{\sigma_X}$, $W=\frac{Y-\mu_Y}{\sigma_Y}$로 두면
>$$
>\left|E\left(\frac{X-\mu_X}{\sigma_X}\cdot\frac{Y-\mu_Y}{\sigma_Y}\right)\right|\le\sqrt{E\left[(\frac{X-\mu_X}{\sigma_X})^2\right]}\sqrt{E\left[(\frac{Y-\mu_Y}{\sigma_Y})^2\right]}=1
>$$
>이 되고, 이는 $|\rho|\le 1$과 동일하다.  
>즉, 정리(2)의 상관계수 부등식은 코시-슈바르츠 부등식으로도 유도할 수 있다.          

### 결합적률 *(Joint Moments)*
상관계수, 공분산처럼 결합확률분포의 특성을 나타내는 또 다른 개념이다.  
$r,s\in\mathbb N$에 대하여
$$
m_{r,s}=E(X^rY^s)
$$
를 $(X,Y)$의 $(r+s)$차의 $(r,s)$번째 결합적률(joint moment)이라 한다. (단, $E(|X|^r|Y|^s)<\infty$로서 결합적률이 존재해야 한다.)

### 결합적률생성함수 *(Joint Moment Generating Function)*
0을 포함하는 어떤 열린집합(근방) $U\subset\mathbb R^2$에서
$$
M(t_1,t_2)=E\left(e^{t_1X+t_2Y}\right)
$$
가 유한하면 이를 **결합적률생성함수(joint moment generating function, joint MGF)** 라 한다.

* 존재영역(domain of finiteness)은 일반적으로 ${(t_1,t_2): M(t_1,t_2)<\infty}$로 정의한다.
* $M$이 0 근방에서 유한하면, $M$은 그 근방에서 미분 가능하며(정당화는 지배수렴/미분-적분 교환 조건 필요), 계수로 결합적률을 생성한다.

### 결합누율생성함수 *(Joint Cumulant Generating Function)*
결합적률은 생성할 뿐만 아니라(조건 하에) 결합확률분포를 결정하는 성질을 갖는다.
$$
C(t_1,t_2)=\log M(t_1,t_2)
$$
를 **결합누율생성함수(joint cumulant generating function, joint CGF)** 라 한다.

결합누율생성함수 $C(t_1,t_2)$가 $(0,0)$ 근방에서 충분히 매끄럽다면, 다변수 Taylor 전개를 통해 다음과 같이 쓸 수 있다:

$$
C(t_1,t_2) = \sum_{r=0}^{\infty}\sum_{s=0}^{\infty} \frac{c_{r,s}}{r!s!}t_1^r t_2^s
$$

여기서 **결합누율(joint cumulant)** $c_{r,s}$는

$$
c_{r,s} = \frac{\partial^{r+s}}{\partial t_1^r \partial t_2^s}C(t_1,t_2)\Big|_{(0,0)}
$$

로 정의된다.

**결합누율, 결합적률 관계식**  
$M(t_1,t_2) = e^{C(t_1,t_2)}$이므로

$$
\sum_{r=0}^{\infty}\sum_{s=0}^{\infty} \frac{m_{r,s}}{r!s!}t_1^r t_2^s = \exp\left(\sum_{r=0}^{\infty}\sum_{s=0}^{\infty} \frac{c_{r,s}}{r!s!}t_1^r t_2^s\right)
$$

또는 역으로

$$
\log\left(1 + \sum_{\substack{r+s\ge 1}} \frac{m_{r,s}}{r!s!}t_1^r t_2^s\right) = \sum_{r=0}^{\infty}\sum_{s=0}^{\infty} \frac{c_{r,s}}{r!s!}t_1^r t_2^s
$$

여기서 $m_{0,0}=1$을 분리하였다.

낮은 차수의 결합누율은 확률분포의 주요 특성치와 직접 대응된다:
* **1차 누율(1st order cumulants)**: 평균(mean)
  **결합누율, 결합적률 관계식**  
  로그함수의 멱급수 전개식 $\log(1+A) = A - \frac{A^2}{2} + \frac{A^3}{3} - \cdots$를 이용하면,

  $$
  A = \sum_{\substack{r+s\ge 1}} \frac{m_{r,s}}{r!s!}t_1^r t_2^s = \frac{m_{1,0}}{1!}t_1 + \frac{m_{0,1}}{1!}t_2 + \frac{m_{2,0}}{2!}t_1^2 + \frac{m_{1,1}}{1!1!}t_1t_2 + \frac{m_{0,2}}{2!}t_2^2 + \cdots
  $$

  로 놓고, 이를 대입하여 $t_1, t_2$의 오름차순으로 정리하면 다음을 얻는다:

  $$
  \begin{align}
  \log(1+A) &= A - \frac{A^2}{2} + \cdots \\
  &= \left(\frac{m_{1,0}}{1!}t_1 + \frac{m_{0,1}}{1!}t_2 + \cdots\right) - \frac{1}{2}\left(\frac{m_{1,0}}{1!}t_1 + \frac{m_{0,1}}{1!}t_2 + \cdots\right)^2 + \cdots \\
  &= m_{1,0}t_1 + m_{0,1}t_2 + \frac{1}{2}(m_{2,0} - m_{1,0}^2)t_1^2 + (m_{1,1} - m_{1,0}m_{0,1})t_1t_2 \\
  &\quad + \frac{1}{2}(m_{0,2} - m_{0,1}^2)t_2^2 + \cdots
  \end{align}
  $$

  이를 누율생성함수의 전개식 $C(t_1,t_2) = \sum_{r,s} \frac{c_{r,s}}{r!s!}t_1^r t_2^s$와 계수를 비교하면,

  낮은 차수의 결합누율은 확률분포의 주요 특성치와 직접 대응된다:
  * **1차 누율(1st order cumulants)**: 평균(mean)
    $$
    c_{1,0} = m_{1,0} = E[X], \quad c_{0,1} = m_{0,1} = E[Y]
    $$

  * **2차 누율(2nd order cumulants)**: 분산(variance), 공분산(covariance)
    $$
    c_{2,0} = m_{2,0} - m_{1,0}^2 = E[X^2] - (E[X])^2 = \mathrm{Var}(X)
    $$
    $$
    c_{0,2} = m_{0,2} - m_{0,1}^2 = E[Y^2] - (E[Y])^2 = \mathrm{Var}(Y)
    $$
    $$
    c_{1,1} = m_{1,1} - m_{1,0}m_{0,1} = E[XY] - E[X]E[Y] = \mathrm{Cov}(X,Y)
    $$
  
  **일반적인 CGF 테일러 전개**  
  결합누율생성함수 $C(t_1,t_2)$를 $(0,0)$ 근방에서 테일러 전개하면
  $$
  C(t_1,t_2) = E[X]t_1 + E[Y]t_2 + \frac{1}{2}\left(\mathrm{Var}(X)t_1^2 + 2\mathrm{Cov}(X,Y)t_1t_2 + \mathrm{Var}(Y)t_2^2\right) + \cdots
  $$
  이 성립한다. 즉, 1차 계수는 평균, 2차 계수는 분산과 공분산으로 직접 주어진다.

### 정리 2.2.4 결합적률생성함수와 결합누율생성함수의 성질 *(Properties of Joint MGF and Joint CGF)*

확률벡터 $(X,Y)$의 결합적률생성함수를
$$
M(t_1,t_2)=E\!\left(e^{t_1X+t_2Y}\right)
$$
라 하자.

**(1) 결합적률 생성성 (Moment generation)**  
$M(t_1,t_2)$가 $(0,0)$를 포함하는 어떤 열린근방에서 유한하며, 해당 근방에서 필요한 차수만큼 편미분 가능하다고 가정하자.
그러면 모든 $r,s\in\mathbb N$에 대하여
$$
\frac{\partial^{r+s}}{\partial t_1^r\partial t_2^s}M(t_1,t_2)\Big|_{(t_1,t_2)=(0,0)}=E(X^rY^s)=:m_{r,s}
$$
가 성립한다.

즉, 결합적률생성함수의 $(0,0)$에서의 편미분 계수는 $(X,Y)$의 결합적률을 생성한다.
- 결합적률생성함수가 결합적률의 생성함수(generating function) 임을 의미한다.

단, 분포 결정성은 결합적률생성함수가 $(0,0)$의 근방에서 존재한다는 가정 하에서만 성립한다.  

**(2) 분포 결정성 (Determination of distribution)**  
$(0,0)$를 포함하는 어떤 열린집합 $U\subset\mathbb R^2$에서 두 확률벡터 $(X,Y)$와 $(X',Y')$의 결합적률생성함수가 모두 존재한다고 하자.
만약
$$
M_{X,Y}(t_1,t_2)=M_{X',Y'}(t_1,t_2)\quad\forall (t_1,t_2)\in U
$$
이면, $(X,Y)$와 $(X',Y')$는 동일한 결합확률분포를 갖는다.

즉, 결합적률생성함수는 (존재하는 경우) 결합확률분포를 유일하게 결정한다.
- 결합적률생성함수가 결합분포의 완전한 특성치(characterization) 임을 의미한다.

#### 증명 (개요)
(1)은 $e^{t_1X+t_2Y}$를 거듭제곱급수로 전개하고 미분-기대값 교환을 정당화(지배수렴 또는 균등적분가능성 조건)하면 얻어진다.

(2)는 0 근방에서 MGF가 존재하면 라플라스 변환이 근방에서 일치하고, 해석적 연장(analytic continuation) 또는 특성함수(characteristic function)로의 연결을 통해 분포가 유일하게 결정됨을 사용한다. □

#### 예시 2.2.5 (예시 2.1.2의 결합 MGF, CGF 및 1·2차 특성치)
예시 2.1.2에서
$$
f(x,y)=2e^{-x-y}\mathbf 1_{{0\le x\le y}}
$$
이다.

**1) 결합적률생성함수 $M(t_1,t_2)$ 계산**
정의에 의해
$$
M(t_1,t_2)=E(e^{t_1X+t_2Y})
=\int_0^\infty\int_{y=x}^\infty 2e^{-x-y}e^{t_1x+t_2y},dy,dx.
$$
내적분이 수렴하려면 $t_2<1$이 필요하고,
$$
\int_{y=x}^\infty 2e^{(t_1-1)x+(t_2-1)y},dy
=\frac{2}{1-t_2},e^{(t_1+t_2-2)x}.
$$
바깥 적분이 수렴하려면 $t_1+t_2<2$가 필요하다. 따라서
$$
M(t_1,t_2)=\frac{2}{(1-t_2)(2-t_1-t_2)},
\quad (t_1,t_2)\ \text{가}\ t_2<1,\ t_1+t_2<2\ \text{를 만족할 때}.
$$

**2) 결합누율생성함수 $C(t_1,t_2)$**
$$
C(t_1,t_2)=\log 2-\log(1-t_2)-\log(2-t_1-t_2).
$$

**3) 평균, 분산, 공분산, 상관계수**
$M$의 편도함수로 적률을 구하면
$$
E[X]=\frac{\partial}{\partial t_1}M(t_1,t_2)\Big|*{(0,0)}=\frac12,\qquad
E[Y]=\frac{\partial}{\partial t_2}M(t_1,t_2)\Big|*{(0,0)}=\frac32.
$$
또한
$$
E[X^2]=\frac{\partial^2}{\partial t_1^2}M\Big|*{(0,0)}=\frac12,\quad
E[Y^2]=\frac{\partial^2}{\partial t_2^2}M\Big|*{(0,0)}=\frac72,\quad
E[XY]=\frac{\partial^2}{\partial t_1\partial t_2}M\Big|_{(0,0)}=1.
$$
따라서
$$
\mathrm{Var}(X)=E[X^2]-E[X]^2=\frac12-\frac14=\frac14,
$$
$$
\mathrm{Var}(Y)=E[Y^2]-E[Y]^2=\frac72-\left(\frac32\right)^2=\frac72-\frac94=\frac54,
$$
$$
\mathrm{Cov}(X,Y)=E[XY]-E[X]E[Y]=1-\frac12\cdot\frac32=\frac14.
$$
상관계수는
$$
\mathrm{Corr}(X,Y)=\frac{\mathrm{Cov}(X,Y)}{\sqrt{\mathrm{Var}(X)\mathrm{Var}(Y)}}
=\frac{\frac14}{\sqrt{\frac14\cdot\frac54}}
=\frac{1}{\sqrt5}
$$
이다.

(참고) 위 2차량은 CGF에서도 바로 얻어진다:
$$
\frac{\partial^2}{\partial t_1^2}C\Big|*{(0,0)}=\mathrm{Var}(X)=\frac14,\quad
\frac{\partial^2}{\partial t_2^2}C\Big|*{(0,0)}=\mathrm{Var}(Y)=\frac54,\quad
\frac{\partial^2}{\partial t_1\partial t_2}C\Big|_{(0,0)}=\mathrm{Cov}(X,Y)=\frac14.
$$

### 주변적률생성함수, 주변누율생성함수 *(Marginal MGF and Marginal CGF)*
결합 MGF와 각 확률변수의 MGF를 구분하기 위해 $X$와 $Y$의 MGF를 각각 **주변적률생성함수(marginal MGF)** 라 한다.

* $X$의 주변적률생성함수(marginal MGF):
  $$
  M_X(s)=E(e^{sX})
  $$
* $Y$의 주변적률생성함수(marginal MGF):
  $$
  M_Y(t)=E(e^{tY})
  $$
  각각의 주변누율생성함수(marginal CGF)는
  $$
  C_X(s)=\log M_X(s),\qquad C_Y(t)=\log M_Y(t)
  $$
  로 정의한다.

또한 결합 MGF로부터 즉시
$$
M_X(s)=M(s,0),\qquad M_Y(t)=M(0,t)
$$
가 성립한다. (정의에서 $t_2=0$ 또는 $t_1=0$을 대입한 특수화다.)

#### 예시 2.2.6 (예시 2.1.2의 주변 MGF, CGF)
앞의 결합 MGF
$$
M(t_1,t_2)=\frac{2}{(1-t_2)(2-t_1-t_2)}
$$
로부터

* $X$의 주변 MGF:
  $$
  M_X(s)=M(s,0)=\frac{2}{2-s},\quad s<2.
  $$
  따라서 주변 CGF는
  $$
  C_X(s)=\log 2-\log(2-s).
  $$

* $Y$의 주변 MGF:
  $$
  M_Y(t)=M(0,t)=\frac{2}{(1-t)(2-t)},\quad t<1.
  $$
  따라서 주변 CGF는
  $$
  C_Y(t)=\log 2-\log(1-t)-\log(2-t).
  $$

(확인) 예시 2.1.4에서 구한 주변밀도

* $f_X(x)=2e^{-2x}\mathbf 1_{{x\ge0}}$ (지수분포 $\mathrm{Exp}(2)$)는 $M_X(s)=\frac{2}{2-s}$와 일치한다.
* $f_Y(y)=2e^{-y}(1-e^{-y})\mathbf 1_{{y\ge0}}=2(e^{-y}-e^{-2y})\mathbf 1_{{y\ge0}}$도 적분하면 $M_Y(t)=\frac{2}{(1-t)(2-t)}$와 일치한다.


## 조건부분포와 조건부기댓값 *(Conditional Distributions and Conditional Expectations)*

### 조건부확률질량함수 *(Conditional Probability Mass Function)*
두 이산형 확률변수 $X, Y$가 결합확률질량함수 $f(x,y)$를 가질 때,  
$X=x$가 주어진 조건 하에서 $Y$의 조건부확률질량함수는

$$
f_{Y|X}(y|x) = P(Y=y \mid X=x) = \frac{P(X=x, Y=y)}{P(X=x)} = \frac{f(x,y)}{f_1(x)}
$$
로 정의된다. 단, $f_1(x) = P(X=x) > 0$이어야 한다.

이산형 확률변수의 경우, $X$가 실제로 가질 수 있는 값(즉, $f_1(x)>0$인 $x$)에 대해서는 조건부확률질량함수가 항상 잘 정의된다.  
마찬가지로 $Y=y$가 주어진 조건 하에서 $X$의 조건부확률질량함수는

$$
f_{1|2}(x\mid y)=f_{X|Y}(x|y) = P(X=x \mid Y=y) = \frac{f(x,y)}{f_2(y)}, \quad f_2(y) > 0
$$
로 정의된다.

### 예 2.3.1 조건부확률의 분포표 *(A Conditional Probability Table)*
두 개의 흰 공, 세 개의 검은 공, 네 개의 빨간 공이 들어있는 상자에서 3개의 공을 동시에 꺼낸다. 흰 공의 개수를 $X$, 검은 공의 개수를 $Y$라 하자.

이미 결합분포표(결합확률질량함수의 표)가 다음과 같이 주어져 있다고 하자(교재 표).

* 예: $P(X=0,Y=0)=4/84$ 등

이때 “흰 공이 추출되지 않았다($X=0$)”는 조건 하에서 검은 공의 개수 $Y$의 분포를 구하고자 한다.

조건부확률 정의에 의해
$$
P(Y=y\mid X=0)=\frac{P(X=0,Y=y)}{P(X=0)}\quad (y=0,1,2,3)
$$
이다.

여기서 분모는
$$
P(X=0)=\sum_{y=0}^3 P(X=0,Y=y)=\frac{35}{84}
$$
이고, 따라서

* $P(Y=0\mid X=0)=\frac{4/84}{35/84}=\frac{4}{35}$
* $P(Y=1\mid X=0)=\frac{18}{35}$
* $P(Y=2\mid X=0)=\frac{12}{35}$
* $P(Y=3\mid X=0)=\frac{1}{35}$

로 조건부확률의 분포표를 얻는다.

이 계산은 “조건부확률은 결합확률을 조건 사건의 확률로 나눈 뒤 재정규화(normalization)한 것”이라는 의미를 분명히 보여준다.

### 이산형 조건부확률질량함수의 성질 *(Properties of Conditional PMF)*
$X=x$가 고정되었을 때, $y\mapsto f_{2|1}(y\mid x)$는 $Y$의 확률질량함수와 같은 성질을 만족한다.
1. 비음성 *(Nonnegativity)*
   $$
   f_{2|1}(y\mid x)\ge 0,\ \forall y
   $$

2. 전체합 1 *(Normalization)*
   $$
   \sum_y f_{2|1}(y\mid x)=1
   $$

(이 성질은 단순히 확률의 기본성질과 정의에서 즉시 따른다.)

### 연속형 조건부확률밀도함수 *(Conditional Probability Density Function, Conditional PDF)*
연속형인 경우에는 $P(X=x)=0$이므로 단순히 $\frac{P(X=x,Y=y)}{P(X=x)}$ 같은 형태로는 정의할 수 없다.  
그래서 “$X$가 $x$ 근처의 작은 구간에 들어갔다”는 조건을 걸고, 구간 폭을 0으로 보내는 극한으로 정의한다.

구체적으로, 양수 $h$에 대하여
$$
P(c\le Y\le d\mid x\le X\le x+h)
=\frac{P(c\le Y\le d,\ x\le X\le x+h)}{P(x\le X\le x+h)}
$$
을 생각한다.

이때 결합밀도 $f_{1,2}$, 주변밀도 $f_1$가 연속이면

* 분자에서
  $$
  P(c\le Y\le d,\ x\le X\le x+h)
  =\int_x^{x+h}\int_c^d f_{1,2}(s,y),dy,ds
  $$
* 분모에서
  $$
  P(x\le X\le x+h)=\int_x^{x+h} f_1(s),ds
  $$

가 되고, $h\downarrow 0$ 극한에서 비율이 잘 정의되도록 만들 수 있다.

그 결과, $f_1(x)>0$일 때
$$
f_{2|1}(y\mid x)=\frac{f_{1,2}(x,y)}{f_1(x)}
$$
로 정의하며 이를 **조건부확률밀도함수(conditional PDF)** 라 한다.  
($X=x$인 조건에서 $Y$에 관한 조건부 확률)
### 연속형 조건부확률밀도함수의 성질 *(Properties of Conditional PDF)*
$f_1(x)>0$인 고정된 $x$에 대해 다음이 성립한다.
1. 비음성
   $$
   f_{2|1}(y\mid x)\ge 0,\ \forall y
   $$
2. 적분이 1
   $$
   \int_{-\infty}^{\infty} f_{2|1}(y\mid x)dy=1
   $$
3. 구간확률
   $$
   P(c\le Y\le d\mid X=x)=\int_c^d f_{2|1}(y\mid x)dy
   $$

### 예 2.3.2 
(예 2.1.2, 2.1.4에서)
$$
f_{1,2}(x,y)=2e^{-x-y}\mathbf 1_{(0\le x\le y)} \\
f_1(x)=2e^{-2x}\mathbf 1_{(x\ge 0)}
$$
임을 알고 있다.

그러면 $x\ge 0$에서
$$
f_{2|1}(y\mid x)=\frac{2e^{-x-y}\mathbf 1_{(0\le x\le y)}}{2e^{-2x}}
=e^{-(y-x)}\mathbf 1_{(y\ge x)}
$$
이다.

또한 임의의 $c$에 대해
$$
P(Y\ge c\mid X=x)=\int_c^\infty f_{2|1}(y\mid x)dy
=\int_{\max(c,x)}^\infty e^{-(y-x)}dy
=e^{-(\max(c,x)-x)}
$$
이므로 특히 $c\ge x$이면 $e^{-(c-x)}$가 된다.

### 2.3.3 조건부확률을 이용한 결합확률 계산 *(Computing Joint Probabilities via Conditioning)*

### 정리 2.3.1 조건부확률의 성질
임의의 구간 $[a,b]$, $[c,d]$에 대해
* 이산형의 경우
  $$
  P(a\le X\le b,\ c\le Y\le d)
  =
  \sum_{a\le x\le b} P(c\le Y\le d\mid X=x)f_1(x)
  $$
* 연속형의 경우
  $$
  P(a\le X\le b,\ c\le Y\le d)
  =
  \int_a^b P(c\le Y\le d\mid X=x)f_1(x)dx
  $$
가 성립한다.

#### 증명 (연속형)
연속형에서
$$
P(c\le Y\le d\mid X=x)=\int_c^d f_{2|1}(y\mid x)dy
$$
이고 $f_{2|1}(y\mid x)=\frac{f_{1,2}(x,y)}{f_1(x)}$이므로,
$$
\int_a^b P(c\le Y\le d\mid X=x)f_1(x)dx
=\int_a^b\left(\int_c^d \frac{f_{1,2}(x,y)}{f_1(x)}dy\right)f_1(x)dx
$$
$$
=\int_a^b\int_c^d f_{1,2}(x,y)dydx
=P(a\le X\le b,\ c\le Y\le d)
$$
이다. □  
이산형도 같은 방법으로 증명 가능.  

#### 예 2.3.3 *(Consistency check using conditioning)*
앞의 예들과 동일한 상황에서
$$
f_1(x)=2e^{-2x}\mathbf 1_{(x\ge 0)},\quad
P(Y\ge 3\mid X=x)=e^{-(3-x)}\ (x\le 3),\ 1\ (x\ge 3)
$$
와 같은 형태를 이용하여
$$
P(X\ge 2,\ Y\ge 3)=\int_2^\infty P(Y\ge 3\mid X=x)f_1(x)dx
$$
로 계산하면, 교재에서 이전 절(직접 적분)로 구한 값과 일치함을 확인한다.

(이 예제의 목적은 “조건부분포를 알면 결합확률도 쉽게 재구성된다”는 점을 보여주는 데 있다.)

### 2.3.4 조건부평균 *(Conditional Mean)*
조건부분포는 $X=x$가 주어졌을 때 $Y$의 분포이므로, 그 분포에 대한 평균을 정의할 수 있다.

$X=x$에서의 $Y$의 조건부평균 *(conditional mean of $Y$ given $X=x$)* 을
$$
\mu_{2|1}(x)=E(Y\mid X=x)
$$
로 표기한다.
* 이산형:
  $$
  E(Y\mid X=x)=\sum_y yf_{2|1}(y\mid x)
  $$
* 연속형:
  $$
  E(Y\mid X=x)=\int_{-\infty}^{\infty} yf_{2|1}(y\mid x)dy
  $$

### 2.3.5 조건부기대값 *(Conditional Expectation)*
조건부평균은 $Y$ 자체의 함수 $g(Y)=Y$에 대한 조건부기대값의 특수한 경우다.  
$X=x$인 조건에서 $Y$의 조건부밀도/질량함수가 $f_{2|1}(y\mid x)$일 때, 임의의 실값 함수 $g$에 대해
* 이산형:
  $$
  E(g(Y)\mid X=x)=\sum_y g(y)f_{2|1}(y\mid x)
  $$

* 연속형:
  $$
  E(g(Y)\mid X=x)=\int_{-\infty}^{\infty} g(y)f_{2|1}(y\mid x)dy
  $$
로 정의한다.

#### 예 2.3.4 *(Conditional mean from a conditional PDF)*
(교재의 예 2.2.1–2.2.2 설정)
$$
f_{1,2}(x,y)=120xy(1-x-y)\mathbf 1_{(x\ge 0,\ y\ge 0,\ x+y\le 1)}
$$
이고
$$
f_1(x)=20x(1-x)^3\mathbf 1_{(0\le x\le 1)}
$$
이다.

따라서
$$
f_{2|1}(y\mid x)=\frac{f_{1,2}(x,y)}{f_1(x)}
=\frac{120xy(1-x-y)}{20x(1-x)^3}\mathbf 1_{(0\le y\le 1-x)}
=\frac{6y(1-x-y)}{(1-x)^3}\mathbf 1_{(0\le y\le 1-x)}
$$

이므로 조건부평균은
$$
\mu_{2|1}(x)=E(Y\mid X=x)=\int_0^{1-x} y\cdot \frac{6y(1-x-y)}{(1-x)^3}dy
=\frac{1-x}{2}
$$
가 된다.

### 정리 2.3.2 조건부기댓값의 성질
다음 성질들이 성립한다 (단, 필요한 조건부기대값이 모두 존재한다고 가정).  

**(1) 선형성 (Linearity)**  
임의의 상수 $a, b$와 함수 $g_1(Y), g_2(Y)$에 대하여
$$
E[ag_1(Y) + bg_2(Y) \mid X=x] = aE[g_1(Y) \mid X=x] + bE[g_2(Y) \mid X=x]
$$

**(2) $X$의 함수 빼내기 (Taking out what is known)**  
$X$의 함수 $v(X)$에 대하여
$$
E[v(X)g(Y) \mid X=x] = v(x)E[g(Y) \mid X=x]
$$

**(3) 반복기대값의 법칙 (Law of iterated expectations)**  
$$
E[E(g(Y) \mid X)] = E[g(Y)]
$$

**(4) 단조성 (Monotonicity)**  
$g_1(Y) \le g_2(Y)$ a.s.이면
$$
E[g_1(Y) \mid X=x] \le E[g_2(Y) \mid X=x]
$$

#### 증명
**(1)** 조건부확률밀도함수(또는 확률질량함수)에 대한 적분(또는 합)의 선형성에서 직접 따른다.

**(2)** $X=x$가 주어졌을 때, $v(x)$는 상수이므로
$$
E[v(x)g(Y) \mid X=x] = v(x)E[g(Y) \mid X=x]
$$

**(3)** 연속형의 경우,
$$
E[E(g(Y) \mid X)] = \int_{-\infty}^{\infty} E[g(Y) \mid X=x]f_1(x)\,dx
$$
$$
= \int_{-\infty}^{\infty}\left(\int_{-\infty}^{\infty} g(y)f_{2|1}(y \mid x)\,dy\right)f_1(x)\,dx
$$
$$
= \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} g(y)f_{1,2}(x,y)\,dy\,dx
$$
$$
= \int_{-\infty}^{\infty} g(y)\left(\int_{-\infty}^{\infty} f_{1,2}(x,y)\,dx\right)dy \quad \text{(Fubini 정리에 의해 적분 순서 교환)}
$$
$$
= \int_{-\infty}^{\infty} g(y)f_2(y)\,dy = E[g(Y)]
$$
이산형의 경우도 동일한 방식으로 증명된다. □

**(4)** $g_1(Y) \le g_2(Y)$ a.s.이면 $g_2(Y) - g_1(Y) \ge 0$ a.s.이므로

연속형의 경우,
$$
E[g_2(Y) - g_1(Y) \mid X=x] = \int_{-\infty}^{\infty} (g_2(y) - g_1(y))f_{2|1}(y \mid x)\,dy \ge 0
$$

이산형의 경우,
$$
E[g_2(Y) - g_1(Y) \mid X=x] = \sum_y (g_2(y) - g_1(y))f_{2|1}(y \mid x) \ge 0
$$

선형성 **(1)** 을 적용하면
$$
E[g_2(Y) \mid X=x] - E[g_1(Y) \mid X=x] \ge 0 \\
\therefore E[g_1(Y) \mid X=x] \le E[g_2(Y) \mid X=x]
$$
□

### 2.3.6 조건부분산 *(Conditional Variance)*
조건부기대값이 있으면, 조건부분산도 같은 방식으로 정의한다.  
$X=x$에서 $Y$의 조건부분산 *(conditional variance of $Y$ given $X=x$)* 은
$$
\mathrm{Var}(Y\mid X=x)=E\big[(Y-\mu_{2|1}(x))^2\mid X=x\big]
$$
로 정의한다. 이를 $\sigma^2_{2|1}(x)$로 표기하기도 한다.

### 정리 2.3.3 조건부분산의 계산공식 *(Computational Formula)*
$$
\mathrm{Var}(Y\mid X=x)=E(Y^2\mid X=x)-\{E(Y\mid X=x)\}^2
$$

#### 증명
정의에서
$$
\mathrm{Var}(Y\mid X=x)=E[(Y-\mu_{2|1}(x))^2\mid X=x]
$$
이므로 전개하면
$$
E[Y^2-2\mu_{2|1}(x)Y+\mu_{2|1}(x)^2\mid X=x]
$$
$$
=E(Y^2\mid X=x)-2\mu_{2|1}(x)E(Y\mid X=x)+\mu_{2|1}(x)^2
$$
인데 $E(Y\mid X=x)=\mu_{2|1}(x)$이므로
$$
=E(Y^2\mid X=x)-\mu_{2|1}(x)^2
$$
이다. □

#### 예 2.3.5 *(Conditional variance computation)*
예 2.3.4에서
$$
E(Y\mid X=x)=\frac{1-x}{2}
$$
이고, 같은 조건부밀도 $f_{2|1}$를 이용하여
$$
E(Y^2\mid X=x)=\int_0^{1-x} y^2\cdot \frac{6y(1-x-y)}{(1-x)^3},dy
$$
를 계산하면
$$
\mathrm{Var}(Y\mid X=x)=E(Y^2\mid X=x)-\left(\frac{1-x}{2}\right)^2
=\frac{(1-x)^2}{20}
$$
이 된다.

### 2.3.7 확률변수로서의 조건부기대값 *(Conditional Expectation as a Random Variable)*
함수
$$
h(x)=E(g(Y)\mid X=x)
$$
를 정의하면, $X$는 확률변수이므로 $h(X)$도 확률변수가 된다.

이를
$$
E(g(Y)\mid X)=h(X)
$$
로 표기하며, 이것이 **확률변수로서의 조건부기대값(conditional expectation as a random variable)** 이다.

### 정리 2.3.4 *(Properties of $E(Y\mid X)$)*
(가정: 적절한 적분가능성 조건이 필요.)  
1. 전체기대값의 법칙 *(Law of Total Expectation)*
   $$
   E[E(Y\mid X)]=E(Y)
   $$

2. 타워 성질 *(Tower Property)*
    $$
    E[E(Y\mid X)\mid X]=E(Y\mid X)
    $$

3. 직교성/비상관 성질 *(Orthogonality / Uncorrelatedness)*  
   $$
   \forall v(X),\ \mathrm{Cov}(Y-E(Y\mid X),\ v(X))=0
   $$

#### 증명 
(1) (연속형)
$$
E[E(Y\mid X)]
=\int\left(\int y f_{2|1}(y\mid x)dy\right)f_1(x)dx
=\int\int yf_{1,2}(x,y)dydx
=E(Y)
$$

(2)  
$E(Y\mid X)$는 $X$의 함수이므로, $X=x$가 주어진 조건에서 $E(Y\mid X)$는 상수 $E(Y\mid x)$가 된다.

따라서
$$
E[E(Y\mid X)\mid X=x]=E(Y\mid x)\cdot 1=E(Y\mid x)
$$

즉, 확률변수로서
$$
E[E(Y\mid X)\mid X]=E(Y\mid X)
$$
가 성립한다. □

(3) $Z=Y-E(Y\mid X)$라 두면, 조건부기대값의 정의상
$$
E(Z\mid X)=E(Y\mid X)-E(E(Y\mid X)\mid X)=E(Y\mid X)-E(Y\mid X)=0
$$
이므로
$$
E(Z)=E(E(Z\mid X))=0
$$
이다. 따라서
$$
\mathrm{Cov}(Z,v(X))=E[Zv(X)]-E(Z)E(v(X))=E[Zv(X)]
$$
인데, 조건부기대값을 한 번 더 쓰면
$$
E[Zv(X)]=E(E[Zv(X)\mid X])=E(v(X)E[Z\mid X])=E(v(X)\cdot 0)=0
$$
이다. □

### 정리 2.3.5 최소제곱예측자 *(Least Squares Predictor)*
조건부평균 $E(Y\mid X)$는 $Y$를 예측하는 "가장 좋은" $X$의 함수라는 뜻을 가지고 있고, 흔히 회귀함수(Regression Function)이라 한다.  

확률변수 $X$의 함수 $u(X)$들 중에서 $E[(Y-u(X))^2]$를 최소로 만드는 함수는
$$
u(X)=E(Y\mid X)
$$
이다. 즉
$$
E[(Y-E(Y\mid X))^2]\le E[(Y-u(X))^2],\quad \forall u(X)
$$
가 성립한다.

또한 최소값(최소 평균제곱예측오차, *mean squared prediction error*)은
$$
E[(Y-E(Y\mid X))^2]=E[\mathrm{Var}(Y\mid X)]
$$
이다.

> 앞으로 확률변수의 합을 $X \oplus Y$로 표기할 때에는,  
> 확률변수 $X$와 $Y$가 서로 **비상관(uncorrelated)**, 즉
> $$\mathrm{Corr}(X,Y) = 0 \quad (\text{동치로 } \mathrm{Cov}(X,Y) = 0)$$
> 임을 전제로 한다.
> 
> $\oplus$는 단순한 대수적 덧셈 $X+Y$가 아니라,  
>   **분산이 교차항 없이 분해되는 합**임을 명시적으로 나타내는 기호다.
> $$ X \oplus Y \implies \mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$$
> 
> 이는 일반적인 분산 공식
> $$ \mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\mathrm{Cov}(X,Y)$$
> 에서 $\mathrm{Cov}(X,Y)=0$인 경우다.

> **참고:**  
> - 독립(independence)이면 항상 비상관이지만, 역은 성립하지 않는다.  
> - $\oplus$ 표기는 독립성을 요구하지 않으며, 오직 비상관성만을 전제로 한다.  
> - 이 표기는 특히 분산의 가법성(additivity of variance)이 중요한 맥락에서 유용하다.

#### 증명
임의의 $X$의 함수 $u(X)$에 대하여, 다음과 같이 분해한다.
$$
Y-u(X)=(Y-E(Y\mid X))\oplus(E(Y\mid X)-u(X)).
$$
이 분해가 $\oplus$로 가능함을 확인하자.  
$E(Y\mid X)-u(X)$는 $X$의 함수이므로, 정리 2.3.4(직교성 성질)에 의해
$$
\mathrm{Cov}(Y-E(Y\mid X),\ E(Y\mid X)-u(X))=0
$$
가 성립한다. 따라서 두 항은 서로 비상관이며, $\oplus$ 표기가 정당하다.

이제 양변의 제곱의 기댓값을 취하면, $\oplus$의 정의에 의해 분산이 가법적으로 분해되어
$$
E[(Y-u(X))^2]=E[(Y-E(Y\mid X))^2]+E[(E(Y\mid X)-u(X))^2] \\
+ 2E[(Y-E(Y\mid X))(E(Y\mid X)-u(X))] \quad (\text{이 마지막 항은 0임})
$$
가 된다.

두 번째 항은 항상 비음이므로
$$
E[(Y-u(X))^2]\ge E[(Y-E(Y\mid X))^2]
$$
가 성립하며, 등호는
$$
E(Y\mid X)-u(X)=0\quad\text{a.s.}
$$

즉 $u(X)=E(Y\mid X)$일 때, 그리고 그때에만 성립한다.  
따라서 $E[(Y-u(X))^2]$를 최소화하는 $X$의 함수는
$$
u(X)=E(Y\mid X)
$$
이다.

마지막으로 최소값은 조건부기대값의 정의를 이용하여
$$
E[(Y-E(Y\mid X))^2]=E[E((Y-E(Y\mid X))^2\mid X)]=E[\mathrm{Var}(Y\mid X)]
$$
로 주어진다. □

### 2.3.10 분산의 분해 *(Decomposition of Variance / Law of Total Variance)*

### 정리 2.3.6 분산의 분해 *(Variance Decomposition)*
$$
\mathrm{Var}(Y)=E[\mathrm{Var}(Y\mid X)]+\mathrm{Var}(E(Y\mid X))
$$

#### 증명
$\mu = E(Y)$라 하자.  
$Y-\mu$를 다음과 같이 분해한다:
$$
Y-\mu=(Y-E(Y\mid X))+(E(Y\mid X)-\mu)
$$

양변을 제곱하면
$$
(Y-\mu)^2=(Y-E(Y\mid X))^2+(E(Y\mid X)-\mu)^2+2(Y-E(Y\mid X))(E(Y\mid X)-\mu)
$$

양변의 기댓값을 취하면
$$
\mathrm{Var}(Y)=E[(Y-E(Y\mid X))^2]+E[(E(Y\mid X)-\mu)^2]+2E[(Y-E(Y\mid X))(E(Y\mid X)-\mu)]
$$

정리 2.3.4(3)의 직교성 성질에 의해 교차항은
$$
E[(Y-E(Y\mid X))(E(Y\mid X)-\mu)]=\mathrm{Cov}(Y-E(Y\mid X),\ E(Y\mid X)-\mu)=0
$$

또한 정리 2.3.4(1)에 의해 $E(E(Y\mid X))=E(Y)=\mu$이므로
$$
E[(E(Y\mid X)-\mu)^2]=\mathrm{Var}(E(Y\mid X))
$$

그리고
$$
E[(Y-E(Y\mid X))^2]=E[\mathrm{Var}(Y\mid X)]
$$

따라서
$$
\mathrm{Var}(Y)=E[\mathrm{Var}(Y\mid X)]+\mathrm{Var}(E(Y\mid X))
$$

### 예 2.3.8 *(Verification for the Beta-like example)*
예 2.3.6에서
$$
E(Y\mid X)=\frac{1-X}{2},\qquad \mathrm{Var}(Y\mid X)=\frac{(1-X)^2}{20}
$$
이고, 앞 절들에서
$$
E(X)=\frac{1}{3},\quad E(X^2)=\frac{1}{7},\quad \mathrm{Var}(X)=\frac{2}{63}
$$
를 알고 있다.

따라서
$$
\mathrm{Var}(E(Y\mid X))=\mathrm{Var}\left(\frac{1-X}{2}\right)=\frac{1}{4}\mathrm{Var}(X)=\frac{1}{4}\cdot\frac{2}{63}=\frac{1}{126}
$$
또한
$$
E[\mathrm{Var}(Y\mid X)]=E\left(\frac{(1-X)^2}{20}\right)
=\frac{1}{20}E(1-2X+X^2)
=\frac{1}{20}\left(1-2\cdot\frac{1}{3}+\frac{1}{7}\right)
$$
이므로 합을 계산하면 $\mathrm{Var}(Y)$와 일치함을 확인한다(교재 결론과 동일하다).

**참고: 편차의 분해 (Decomposition of Deviation)**
$$
Y - E[Y] = (Y - E[Y \mid X]) \oplus (E[Y \mid X] - E[Y])
$$
이런 편차 $Y - E[Y]$의 분해는 반응변수 $Y$가 여러 다른 값을 갖게 되는 원인을 설명변수 $X$에 따른 부분 $(E[Y \mid X] - E[Y])$과 그 이외의 부분 $(Y - E[Y \mid X])$으로 나눠 생각하는 것을 수식으로 나타내고 있는 것이다.  
정리 2.3.4(3)의 직교성 성질에 의해 이 두 항은 서로 비상관이므로 $\oplus$ 표기가 정당하다.  
- $(E[Y \mid X] - E[Y])$: $X$에 의해 **설명되는(explained)** 부분
- $(Y - E[Y \mid X])$: $X$로 **설명되지 않는(unexplained)** 잔차(residual) 부분  

이 아이디어는 10장에서 소개되는 회귀분석과 분산분석의 기초가 된다.

### 정리 2.3.7 조건부분산과 전체 분산의 관계 *(Relationship between Conditional and Total Variance)*
$$
\mathrm{Var}(X-E(X\mid Y))=E[\mathrm{Var}(X\mid Y)]
$$

#### 증명
$\mu = E(X)$라 하자. 정리 2.3.6의 분산의 분해에 의해
$$
\mathrm{Var}(X)=E[\mathrm{Var}(X\mid Y)]+\mathrm{Var}(E(X\mid Y))
$$
가 성립한다.

한편, $Z=X-E(X\mid Y)$라 두면, 조건부기대값의 성질에 의해
$$
E(Z\mid Y)=E(X\mid Y)-E(E(X\mid Y)\mid Y)=E(X\mid Y)-E(X\mid Y)=0
$$
이므로
$$
E(Z)=E(E(Z\mid Y))=0
$$
이다.

따라서
$$
\mathrm{Var}(Z)=E(Z^2)-[E(Z)]^2=E(Z^2)
$$
$$
=E[(X-E(X\mid Y))^2]
$$

이제 조건부기대값을 이용하면
$$
E[(X-E(X\mid Y))^2]=E[E((X-E(X\mid Y))^2\mid Y)]
$$

조건부분산의 정의에 의해
$$
E[(X-E(X\mid Y))^2\mid Y]=\mathrm{Var}(X\mid Y)
$$
이므로
$$
\mathrm{Var}(X-E(X\mid Y))=E[\mathrm{Var}(X\mid Y)]
$$
가 성립한다. □

#### 예시
$$
\mathrm{Var}(Y+X-E(X\mid Y))=\mathrm{Var}(Y)+E[\mathrm{Var}(X\mid Y)]
$$

$Z=X-E(X\mid Y)$라 하자. 그러면
$$
Y+X-E(X\mid Y)=Y+Z
$$
이다.

정리 2.3.4(3)의 직교성 성질에 의해
$$
\mathrm{Cov}(Y,Z)=\mathrm{Cov}(Y,X-E(X\mid Y))=0
$$
이므로 $Y$와 $Z$는 비상관이다.

따라서 정리 2.4.5에 의해
$$
\mathrm{Var}(Y+Z)=\mathrm{Var}(Y)+\mathrm{Var}(Z)
$$

여기서 정리 2.3.7에 의해
$$
\mathrm{Var}(Z)=\mathrm{Var}(X-E(X\mid Y))=E[\mathrm{Var}(X\mid Y)]
$$

따라서
$$
\mathrm{Var}(Y+X-E(X\mid Y))=\mathrm{Var}(Y)+E[\mathrm{Var}(X\mid Y)]
$$
□

## 확률변수의 독립성 *(Independence of Random Variables)*
확률변수의 독립성은 “한 확률변수에 대한 정보가 다른 확률변수의 분포에 아무런 영향을 주지 않는다”는 개념을 수학적으로 정식화한 것이다.
이는 조건부확률, 결합분포, 적률, 분산 구조 등 이후의 모든 통계 이론의 핵심 가정으로 작용한다.

### 정의 2.4.1 (확률변수의 독립성)
두 확률변수 $X, Y$에 대하여, 임의의 실수 $a<b$, $c<d$에 대해
$$
P(a \le X \le b,\ c \le Y \le d)
= P(a \le X \le b),P(c \le Y \le d)
$$
가 성립하면, $X$와 $Y$는 **서로 독립(mutually independent)** 이라고 한다.

이는 $X$에 관한 어떤 사건도 $Y$에 관한 사건의 확률을 변화시키지 않음을 의미한다.

### 정리 2.4.1 (독립성의 동치조건)
다음 조건들은 서로 동치이며, 이 중 하나라도 성립하면 $X$와 $Y$는 독립이다.

(a) 누적분포함수(CDF)
$$
F_{X,Y}(x,y)=F_X(x)F_Y(y),
\quad \forall x,y \in \mathbb{R}
$$

(b) 확률밀도함수(PDF) 또는 확률질량함수(PMF)
$$
f_{X,Y}(x,y)=f_X(x)f_Y(y),
\quad \forall x,y
$$

(c) 적률생성함수(MGF) (존재하는 경우)
$$
M_{X,Y}(t_1,t_2)=M_X(t_1)M_Y(t_2),
\quad (t_1,t_2)\text{ in a neighborhood of }(0,0)
$$

(d) 확률측도 표현
$$
P(X\in A,\ Y\in B)=P(X\in A)P(Y\in B),
\quad \forall A,B \text{ (Borel 집합)}
$$

**참고**  
확률밀도함수로 독립을 증명하는 과정에서 반드시 주변확률밀도함수를 구할 필요는 없음.  

**(b') 인수분해 기준 (Factorization Criterion)**  
결합확률밀도함수(또는 결합확률질량함수)가 $x$만의 함수와 $y$만의 함수의 곱으로 인수분해되면, 즉
$$
f_{X,Y}(x,y) = g_1(x) \cdot g_2(y), \quad \forall x,y
$$
를 만족하는 비음함수 $g_1(x)$와 $g_2(y)$가 존재하면, $X$와 $Y$는 독립이다.

**증명 (연속형의 경우)**  
항등식 양변을 모든 $x, y$에 대해 적분하면
$$
\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} f_{X,Y}(x,y)\,dy\,dx 
= \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} g_1(x)g_2(y)\,dy\,dx = 1
$$
이므로
$$
\left(\int_{-\infty}^{\infty} g_1(x)\,dx\right)\left(\int_{-\infty}^{\infty} g_2(y)\,dy\right) = 1.
$$

따라서
$$
f_{X,Y}(x,y) = g_1(x)g_2(y) 
= \left\{\frac{g_1(x)}{\int g_1(x)\,dx}\right\}\left\{\frac{g_2(y)}{\int g_2(y)\,dy}\right\}
$$
로 쓸 수 있고, 각 중괄호 안의 함수는 적분이 1인 확률밀도함수이므로
$$
f_{X,Y}(x,y) = f_X(x)f_Y(y)
$$
가 성립하여 (b)가 성립한다. □

마찬가지로,  
누적분포함수나 적률생성함수를 이용할 때도, (a), (c)의 항등식 오른쪽 변에 나오는 주변누적분포함수 $F_X(x), F_Y(y)$나 주변적률생성함수 $M_X(t_1), M_Y(t_2)$를 명시적으로 구하지 않고도 독립성 여부를 판단할 수 있다.  
즉, 결합함수가 $x$의 함수와 $y$의 함수의 곱으로 분해되는지만 확인하면 된다.

#### 독립성 동치조건 증명 (개요)
* (a) ⇔ (b): CDF와 PDF의 미분 관계로부터 즉시 성립한다.
* (b) ⇔ (c): MGF는 PDF의 적분 변환이므로 곱 구조가 보존된다.
* (b) ⇔ (d): 확률측도와 밀도함수는 일대일 대응 관계에 있다.
* (d) ⇒ 정의: 구간 사건을 대입하면 독립성의 정의를 얻는다.
* 정의 ⇒ (a): 확률측도의 연속성과 극한을 이용한다. □

#### 예시 2.4.1 (독립인 경우)
결합확률밀도함수가
$$
f_{X,Y}(x,y)=2e^{-x-2y}\mathbf{1}_{{x\ge0,y\ge0}}
$$
로 주어졌다고 하자.

주변확률밀도함수는
$$
f_X(x)=\int_0^\infty 2e^{-x-2y}dy=e^{-x}\mathbf{1}*{{x\ge0}},
$$
$$
f_Y(y)=\int_0^\infty 2e^{-x-2y}dx=2e^{-2y}\mathbf{1}*{{y\ge0}}.
$$

따라서
$$
f_{X,Y}(x,y)=f_X(x)f_Y(y)
$$
이므로 $X$와 $Y$는 서로 독립이다. □

#### 예시 2.4.2 (독립이 아닌 경우)
결합확률밀도함수가
$$
f_{X,Y}(x,y)=2e^{-x-y}\mathbf{1}_{{0\le x\le y}}
$$
로 주어졌다고 하자.

주변밀도 $f_X, f_Y$는 각각 존재하지만,
$$
f_{X,Y}(x,y)\neq f_X(x)f_Y(y)
$$
이므로 $X$와 $Y$는 독립이 아니다. □

### 정리 2.4.2 (독립 확률변수의 함수)
$X$와 $Y$가 독립이면, 임의의 함수 $g_1, g_2$에 대하여
확률변수 $g_1(X)$와 $g_2(Y)$도 서로 독립이다.

#### 증명
임의의 집합 $A,B$에 대해
$$
{g_1(X)\in A,\ g_2(Y)\in B}
= {X\in g_1^{-1}(A),\ Y\in g_2^{-1}(B)}.
$$

$X,Y$가 독립이므로
$$
P(X\in g_1^{-1}(A),Y\in g_2^{-1}(B))
= P(X\in g_1^{-1}(A))P(Y\in g_2^{-1}(B)).
$$

이는 곧
$$
P(g_1(X)\in A)P(g_2(Y)\in B)
$$
이므로 $g_1(X)$와 $g_2(Y)$는 독립이다. □

### 정리 2.4.3 (독립 확률변수의 곱의 기댓값)
$X$와 $Y$가 독립이고 기댓값이 존재하면,
$$
E[g_1(X)g_2(Y)]=E[g_1(X)]E[g_2(Y)]
$$
가 성립한다.

### 증명 (연속형)
$$
E[g_1(X)g_2(Y)]
= \int\int g_1(x)g_2(y)f_{X,Y}(x,y)dydx
$$

독립성이므로 $f_{X,Y}(x,y)=f_X(x)f_Y(y)$이고,
Fubini 정리에 의해
$$
=\left(\int g_1(x)f_X(x),dx\right)
\left(\int g_2(y)f_Y(y),dy\right).
$$

즉 $$E[g_1(X)]E[g_2(Y)]$$
□

### 정리 2.4.4 (독립성과 상관관계)
$X$와 $Y$가 독립이고 공분산이 존재하면
$$
\mathrm{Cov}(X,Y)=0
$$
이다.

#### 주의 (역은 성립하지 않음)
$\mathrm{Cov}(X,Y)=0$이라고 해서 $X,Y$가 독립일 필요는 없다.

#### 예시 2.4.3 (상관계수는 0이나 독립이 아님)

결합확률밀도함수가
$$
f_{X,Y}(x,y)=\frac{9}{40}(1+x^2y^2)\mathbf{1}_{{-1\le x\le1,-1\le y\le1}}
$$
일 때,
$$
E(X)=E(Y)=E(XY)=0 \Rightarrow \mathrm{Cov}(X,Y)=0.
$$

그러나
$$
f_{X,Y}(x,y)\neq f_X(x)f_Y(y)
$$
이므로 $X,Y$는 독립이 아니다. □

### 정리 2.4.5 (확률변수의 합의 분산)
임의의 확률변수 $X,Y$에 대해
$$
\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)+2\mathrm{Cov}(X,Y).
$$

특히 $X,Y$가 독립이면
$$
\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y).
$$
#### 증명
분산의 정의와 전개를 이용하면
$$
\begin{align}
\mathrm{Var}(X+Y) &= E[(X+Y)^2] - [E(X+Y)]^2 \\
&= E(X^2 + 2XY + Y^2) - [E(X) + E(Y)]^2 \\
&= E(X^2) + 2E(XY) + E(Y^2) - [E(X)]^2 - 2E(X)E(Y) - [E(Y)]^2 \\
&= \mathrm{Var}(X) + \mathrm{Var}(Y) + 2[E(XY) - E(X)E(Y)] \\
&= \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\mathrm{Cov}(X,Y)
\end{align}
$$
특히 $X$와 $Y$가 독립이면 $\mathrm{Cov}(X,Y) = 0$이므로
$$
\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y).
$$
□


## 다차원 확률변수의 분포 *(Distributions of Multivariate Random Variables)*
확률변수 $X_1, X_2, \dots, X_k$를 각 성분으로 하는 벡터
$$
\mathbf{X}=(X_1, X_2, \dots, X_k)^T
$$
를 **$k$차원 확률변수**, 또는 **$k$변량 확률벡터($k$-variate random vector)**라 한다.

이차원 확률변수의 경우와 마찬가지로, 다차원 확률변수도
**이산형(discrete)**과 **연속형(continuous)**으로 나누어 다룬다.

### 정의 2.5.1 (이산형 다차원 확률변수의 결합확률질량함수)
이산형 확률벡터 $\mathbf{X}=(X_1,\dots,X_k)^T$의 **결합확률질량함수(joint pmf)**
$f(x_1,\dots,x_k)$는 다음 성질을 만족한다.

1. **비음성**
   $$
   f(x_1,\dots,x_k)\ge 0,\quad \forall x_i\in\mathbb{R}
   $$

2. **정규화**
   $$
   \int_{-\infty}^{\infty}\cdots\int_{-\infty}^{\infty}
   f(x_1,\dots,x_k),dx_k\cdots dx_1=1
   $$

3. **확률 계산**
   $$
   P(a_1\le X_1\le b_1,\dots,a_k\le X_k\le b_k)
   =
   \int_{a_1}^{b_1}\cdots\int_{a_k}^{b_k}
   f(x_1,\dots,x_k)dx_k\cdots dx_1
   $$

#### 예시 2.5.1
두 개의 흰 공, 세 개의 검은 공, 네 개의 빨간 공, 다섯 개의 노란 공이 들어 있는 상자에서 네 개의 공을 동시에 꺼낸다.  
흰 공의 개수: $X_1$, 검은 공의 개수: $X_2$, 빨간 공의 개수: $X_3$
라고 하면,
$$
f(x_1,x_2,x_3)
=
\frac{\binom{2}{x_1}\binom{3}{x_2}\binom{4}{x_3}\binom{5}{4-x_1-x_2-x_3}}
{\binom{14}{4}}
$$
이며,
$$
x_1=0,1,2,\quad x_2=0,1,2,3,\quad x_3=0,1,2,3,4,\quad x_1+x_2+x_3\le4
$$
이다. □

### 정의 2.5.2 (연속형 다차원 확률변수의 결합확률밀도함수)
연속형 확률벡터 $\mathbf{X}=(X_1,\dots,X_k)^T$의
**결합확률밀도함수(joint pdf)** $f(x_1,\dots,x_k)$는 다음을 만족한다.

1. **비음성**
   $$
   f(x_1,\dots,x_k)\ge 0
   $$

2. **정규화**
   $$
   \int_{-\infty}^{\infty}\cdots\int_{-\infty}^{\infty}
   f(x_1,\dots,x_k),dx_k\cdots dx_1=1
   $$

3. **확률 계산**
   $$
   P(a_1\le X_1\le b_1,\dots,a_k\le X_k\le b_k)
   =
   \int_{a_1}^{b_1}\cdots\int_{a_k}^{b_k}
   f(x_1,\dots,x_k),dx_k\cdots dx_1
   $$

#### 예시 2.5.2
다음 함수가 결합확률밀도함수가 되도록 상수 $c$를 구한다.
$$
f(x_1,x_2,x_3)
=
c e^{-x_1-x_2-x_3}\mathbf{1}_{(0\le x_1\le x_2\le x_3)}
$$

**풀이**
전체 확률이 1이어야 하므로
$$
\int_0^\infty\int_{x_1}^\infty\int_{x_2}^\infty
c e^{-x_1-x_2-x_3}dx_3dx_2dx_1=1
$$
계산하면 $c=6$이다. □

### 주변확률밀도함수
확률벡터 $\mathbf{X}=(X_1,\dots,X_k)^T$의 결합확률밀도함수가 $f$일 때,
* **이산형**
  $$
  f_1(x_1)=\sum_{x_2}\cdots\sum_{x_k} f(x, x_2, \dots,x_k)
  $$
* **연속형**
  $$
  f_1(x_1)=\int_{-\infty}^{\infty}\cdots\int_{-\infty}^{\infty}
  f(x, x_2,\dots,x_k)dx_k\cdots dx_2
  $$

일반적으로 $(X_1,\dots,X_j)$의 주변확률밀도함수는
* **이산형**
  $$
  f_{1,\dots,j}(x_1,\dots,x_j)=\sum_{x_{j+1}}\cdots\sum_{x_k} f(x_1,\dots,x_k)
  $$
* **연속형**
  $$
  f_{1,\dots,j}(x_1,\dots,x_j)=\int_{-\infty}^{\infty}\cdots\int_{-\infty}^{\infty}
  f(x_1,\dots,x_k)dx_k\cdots dx_{j+1}
  $$
로 정의한다. (1~k까지 변수취급, 나머지 $dx_?$로 빠짐)

#### 예시 2.5.3
예시 2.5.1에서 $(X_1,X_2)$의 주변분포는
$$
f_{1,2}(x,y)
= \sum_{x_3} f(x,y,x_3)
\frac{\binom{2}{x}\binom{3}{y}\binom{9}{4-x-y}}{\binom{14}{4}},
\quad x+y\le4
$$
이다. □

### 누적분포함수
다차원의 경우에도 확률밀도함수, 누적분포함수는 일대일로 대응한다.  
연속형의 경우
$$
F(x_1,\dots,x_k)
= P(X_1\le x_1,\dots,X_k\le x_k)
= \int_{-\infty}^{x_1}\cdots\int_{-\infty}^{x_k}
f(t_1,\dots,t_k),dt_k\cdots dt_1
$$

$f$가 연속인 점에서
$$
f(x_1,\dots,x_k)
=\frac{\partial^k}{\partial x_1\cdots\partial x_k}
F(x_1,\dots,x_k)
$$
가 성립한다.

### 확률벡터의 기댓값
실수값 함수 $g(x_1,\dots,x_k)$에 대해

* **이산형**
  $$
  E[g(X_1,\dots,X_k)]
  =
  \sum_{x_1}\cdots\sum_{x_k}
  g(x_1,\dots,x_k)f(x_1,\dots,x_k)
  $$

* **연속형**
  $$
  E[g(X_1,\dots,X_k)]
  =
  \int\cdots\int
  g(x_1,\dots,x_k)f(x_1,\dots,x_k),dx_k\cdots dx_1
  $$

#### 예시 2.5.5
$$
f(x,y,z)=120x(1-x-y-z)\mathbf{1}_{(x,y,z\ge0,\ x+y+z\le1)}
$$
일 때,
$$
E(X_1X_2)=\iiint xy\cdot120x(1-x-y-z)\,dz\,dy\,dx
$$
적분 영역은 $x,y,z\ge0$, $x+y+z\le1$이므로
$$
=\int_0^1\int_0^{1-x}\int_0^{1-x-y} 120x^2y(1-x-y-z)\,dz\,dy\,dx
$$
먼저 $z$에 대해 적분하면
$$
\int_0^{1-x-y} (1-x-y-z)\,dz = (1-x-y)z - \frac{z^2}{2}\Big|_0^{1-x-y} = \frac{(1-x-y)^2}{2}
$$
따라서
$$
=\int_0^1\int_0^{1-x} 120x^2y\cdot\frac{(1-x-y)^2}{2}\,dy\,dx
=\int_0^1\int_0^{1-x} 60x^2y(1-x-y)^2\,dy\,dx
$$
$y$에 대해 적분하면($u=1-x-y$로 치환하거나 직접 계산)
$$
\int_0^{1-x} y(1-x-y)^2\,dy = \frac{(1-x)^4}{12}
$$
따라서
$$
=\int_0^1 60x^2\cdot\frac{(1-x)^4}{12}\,dx = 5\int_0^1 x^2(1-x)^4\,dx
$$
베타함수 $B(3,5) = \frac{\Gamma(3)\Gamma(5)}{\Gamma(8)} = \frac{2!\cdot4!}{7!} = \frac{1}{105}$를 이용하면
$$
=5\cdot\frac{1}{105} = \frac{1}{21}
$$
이다. □

### 정리 2.5.1 (기댓값의 성질)
1차원에서의 기댓값 성질이 다차원에서도 동일하게 성립함.  
확률벡터 $\mathbf{X}=(X_1,\dots,X_k)^T$와 실수값 함수 $g_1(\mathbf{X}), g_2(\mathbf{X})$에 대하여 기댓값이 존재한다고 하자.

(a) **선형성**  
임의의 상수 $c_1,c_2$에 대하여
$$
E[c_1 g_1(\mathbf{X})+c_2 g_2(\mathbf{X})]
= c_1 E[g_1(\mathbf{X})]+c_2 E[g_2(\mathbf{X})]
$$

(b) **단조성**
$$
g_1(\mathbf{X})\le g_2(\mathbf{X}) \ \text{이면}\
E[g_1(\mathbf{X})]\le E[g_2(\mathbf{X})]
$$

### 평균벡터와 공분산행렬 *(Mean Vector and Covariance Matrix)*
확률벡터 $\mathbf{X}=(X_1,\dots,X_k)^T$에 대하여, 각 성분의 평균을
$$
\mu_i=E(X_i),\quad i=1,\dots,k
$$
라 하면, **평균벡터(mean vector)** 는
$$
\mu=E(\mathbf{X})=(\mu_1,\dots,\mu_k)^T
$$
로 정의된다.

**공분산행렬(covariance matrix)** 또는 **분산–공분산행렬(variance-covariance matrix)** 은
$$
\mathrm{Var}(\mathbf{X})
=(\sigma_{ij})_{1\le i,j\le k},\quad
\sigma_{ij}=\mathrm{Cov}(X_i,X_j)
$$
로 정의되며, 공분산의 정의를 이용하면
$$
\mathrm{Var}(\mathbf{X})
= E[(\mathbf{X}-\mu)(\mathbf{X}-\mu)^T]
$$
로도 표현할 수 있다.
- 대각성분 $\sigma_{ii}=\mathrm{Cov}(X_i,X_i)=\mathrm{Var}(X_i)$는 각 확률변수의 분산이다.
- 비대각성분 $\sigma_{ij}=\mathrm{Cov}(X_i,X_j)\ (i\neq j)$는 두 확률변수 간의 공분산이다.
- 공분산행렬은 대칭행렬이며($\sigma_{ij}=\sigma_{ji}$), 항상 양의 준정부호(positive semidefinite) 성질을 갖는다.

> **참고: 분산행렬 표기**  
> 공분산행렬 $\mathrm{Var}(\mathbf{X})$를 **분산행렬(variance matrix)** 이라고도 부르며,  
> $\Sigma$ 또는 $V(\mathbf{X})$ 등의 기호로 표기하기도 한다. $\mathrm{Var}(\mathbf{X})$ 표기를 주로 사용한다.

### 정리 2.5.2 (확률변수 행렬의 기댓값의 성질)
확률변수 행렬 $W = (W_{ij})$의 기댓값은 각 성분의 기댓값을 같은 위치에 배치한 행렬로 정의된다:

$$
E(W) = E\begin{pmatrix}
W_{11} & W_{12} & \cdots & W_{1n} \\
W_{21} & W_{22} & \cdots & W_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
W_{m1} & W_{m2} & \cdots & W_{mn}
\end{pmatrix}
=
\begin{pmatrix}
E(W_{11}) & E(W_{12}) & \cdots & E(W_{1n}) \\
E(W_{21}) & E(W_{22}) & \cdots & E(W_{2n}) \\
\vdots & \vdots & \ddots & \vdots \\
E(W_{m1}) & E(W_{m2}) & \cdots & E(W_{mn})
\end{pmatrix}
$$
즉, $(E(W))_{ij} = E(W_{ij})$이다.

확률변수 행렬 $V=(V_{ij})$, $W=(W_{ij})$와 원소가 상수인 행렬 $C,D$에 대하여 기댓값이 존재한다고 하자. 그러면 다음과 같은 선형성을 가진다.  
(a)
$$
E(CWD)=CE(W)D
$$

(b)
$$
E(V+W)=E(V)+E(W)
$$

**설명**
행렬의 기댓값은 성분별로 정의한다. 즉,
$$
E(W)=(E(W_{ij}))
$$
로 둔다. 그러면 (a), (b)는 기댓값의 선형성을 성분별로 적용하여 얻는다.

### 분산행렬과 공분산행렬의 표현 방법 정리
확률벡터 $\mathbf{X}=(X_1,\dots,X_k)^T$에 대하여 공분산행렬(또는 분산행렬)을 표현하는 방법은 다음과 같다.

#### (1) 성분별 정의
공분산행렬의 $(i,j)$ 성분은
$$
(\mathrm{Var}(\mathbf{X}))_{ij} = \mathrm{Cov}(X_i,X_j) = E[(X_i-\mu_i)(X_j-\mu_j)]
$$
로 정의된다. 여기서 $\mu_i=E(X_i)$이다.

#### (2) 행렬 표현
평균벡터를 $\mu=E(\mathbf{X})$라 하면,
$$
\mathrm{Var}(\mathbf{X}) = E[(\mathbf{X}-\mu)(\mathbf{X}-\mu)^T]
$$
로 표현할 수 있다.

**전개하면**
$$
\mathrm{Var}(\mathbf{X}) = E[\mathbf{X}\mathbf{X}^T] - E[\mathbf{X}]E[\mathbf{X}]^T = E[\mathbf{X}\mathbf{X}^T] - \mu\mu^T
$$

#### (3) 행렬 형태
공분산행렬은 다음과 같은 대칭행렬이다.
$$
\mathrm{Var}(\mathbf{X}) = 
\begin{pmatrix}
\mathrm{Var}(X_1) & \mathrm{Cov}(X_1,X_2) & \cdots & \mathrm{Cov}(X_1,X_k) \\
\mathrm{Cov}(X_2,X_1) & \mathrm{Var}(X_2) & \cdots & \mathrm{Cov}(X_2,X_k) \\
\vdots & \vdots & \ddots & \vdots \\
\mathrm{Cov}(X_k,X_1) & \mathrm{Cov}(X_k,X_2) & \cdots & \mathrm{Var}(X_k)
\end{pmatrix}
$$

#### (4) 간략 표기
문헌에 따라 다음과 같은 표기를 사용하기도 한다.
- $\Sigma = \mathrm{Var}(\mathbf{X})$
- $V(\mathbf{X}) = \mathrm{Var}(\mathbf{X})$
- $\mathrm{Cov}(\mathbf{X}) = \mathrm{Var}(\mathbf{X})$ (단, $\mathbf{X}$ 자신과의 공분산)

#### (5) 공분산행렬의 일반화
두 확률벡터 $\mathbf{X}\in\mathbb{R}^k$, $\mathbf{Y}\in\mathbb{R}^\ell$에 대하여
$$
\mathrm{Cov}(\mathbf{X},\mathbf{Y}) = E[(\mathbf{X}-E\mathbf{X})(\mathbf{Y}-E\mathbf{Y})^T]
$$
는 $k\times\ell$ 행렬이며,
$$
(\mathrm{Cov}(\mathbf{X},\mathbf{Y}))_{ij} = \mathrm{Cov}(X_i,Y_j)
$$
이다.

특히 $\mathbf{X}=\mathbf{Y}$인 경우
$$
\mathrm{Var}(\mathbf{X}) = \mathrm{Cov}(\mathbf{X},\mathbf{X})
$$
가 성립한다.

#### 예시 2.5.6
확률벡터 $\mathbf{X}=(X_1,X_2,X_3)^T$가
$$
E(X_1)=1,\quad E(X_2)=2,\quad E(X_3)=3
$$
$$
\mathrm{Var}(X_1)=4,\quad \mathrm{Var}(X_2)=5,\quad \mathrm{Var}(X_3)=6
$$
$$
\mathrm{Cov}(X_1,X_2)=1,\quad \mathrm{Cov}(X_1,X_3)=2,\quad \mathrm{Cov}(X_2,X_3)=3
$$
를 만족할 때, 평균벡터와 공분산행렬은 다음과 같다.
$$
E(\mathbf{X})=
\begin{pmatrix}
1\\
2\\
3
\end{pmatrix},
\qquad
\mathrm{Var}(\mathbf{X})=
\begin{pmatrix}
4 & 1 & 2\\
1 & 5 & 3\\
2 & 3 & 6
\end{pmatrix}
$$

#### 예시 2.5.7
확률벡터 $\mathbf{X}=(X_1,X_2)^T$의 결합확률밀도함수가
$$
f(x,y)=2e^{-x-y}\mathbf{1}_{(0\le x\le y)}
$$
일 때(교재 앞 예제들의 계산 결과를 이용하면),
$$
E(\mathbf{X})=
\begin{pmatrix}
1/2\\
3/2
\end{pmatrix},
\qquad
\mathrm{Var}(\mathbf{X})=
\begin{pmatrix}
1/4 & 1/4\\
1/4 & 5/4
\end{pmatrix}
$$

### 정리 2.5.3 (평균벡터와 공분산행렬의 성질)
확률벡터 $\mathbf{X}\in\mathbb{R}^k$, $\mathbf{Y}\in\mathbb{R}^\ell$과 상수행렬 $A,C$, 상수벡터 $b,d$에 대하여(필요한 기댓값이 존재한다고 가정한다)

(a)
$$
E(A\mathbf{X}+b)=AE(\mathbf{X})+b
$$

(b)
$$
\mathrm{Var}(A\mathbf{X}+b)=A\mathrm{Var}(\mathbf{X})A^T
$$

(c)
$$
\mathrm{Cov}(A\mathbf{X}+b,\ C\mathbf{Y}+d)=A\mathrm{Cov}(\mathbf{X},\mathbf{Y})C^T
$$

(d)
$$
\mathrm{Cov}(\mathbf{X}+\mathbf{Y},\ \mathbf{Z})
=\mathrm{Cov}(\mathbf{X},\mathbf{Z})+\mathrm{Cov}(\mathbf{Y},\mathbf{Z})
$$
또한
$$
\mathrm{Cov}(\mathbf{X},\ \mathbf{Z}+\mathbf{W})
=\mathrm{Cov}(\mathbf{X},\mathbf{Z})+\mathrm{Cov}(\mathbf{X},\mathbf{W})
$$

(e)
$$
\mathrm{Cov}(\mathbf{Y},\mathbf{X})=\mathrm{Cov}(\mathbf{X},\mathbf{Y})^T,
\qquad
\mathrm{Var}(\mathbf{X})=\mathrm{Cov}(\mathbf{X},\mathbf{X})
$$

(f)
$$
\mathrm{Var}(\mathbf{X}+\mathbf{Y})
=
\mathrm{Var}(\mathbf{X})+\mathrm{Var}(\mathbf{Y})
+\mathrm{Cov}(\mathbf{X},\mathbf{Y})+\mathrm{Cov}(\mathbf{Y},\mathbf{X})
$$

#### 증명
공분산행렬의 정의를
$$
\mathrm{Cov}(\mathbf{X},\mathbf{Y})
=E\big[(\mathbf{X}-E\mathbf{X})(\mathbf{Y}-E\mathbf{Y})^T\big]
$$
로 두고, 정리 2.5.1의 선형성을 성분별로 적용하면 된다.
예를 들어 (c)는
$$
\mathrm{Cov}(A\mathbf{X}+b,\ C\mathbf{Y}+d)
=E[(A\mathbf{X}-AE\mathbf{X})(C\mathbf{Y}-CE\mathbf{Y})^T]
$$
$$
=E[A(\mathbf{X}-E\mathbf{X})(\mathbf{Y}-E\mathbf{Y})^T C^T]
=A\mathrm{Cov}(\mathbf{X},\mathbf{Y})C^T
$$
로 얻는다. 나머지도 동일한 방식이다. □

#### 예시 2.5.8
예시 2.5.7에서
$$
E(\mathbf{X})=
\begin{pmatrix}
1/2\\
3/2
\end{pmatrix},
\qquad
\mathrm{Var}(\mathbf{X})=
\begin{pmatrix}
1/4 & 1/4\\
1/4 & 5/4
\end{pmatrix}
$$
라고 하자.

여기서 $\mathbf{Y}=(Y_1,Y_2)^T=(2X_1,\ X_2-X_1)^T$라고 두면
$$
\mathbf{Y}=A\mathbf{X},\quad
A=
\begin{pmatrix}
2&0\\
-1&1
\end{pmatrix}
$$
이다.

정리 2.5.3에 의해
$$
E(\mathbf{Y})=AE(\mathbf{X})
=
\begin{pmatrix}
2&0\\
-1&1
\end{pmatrix}
\begin{pmatrix}
1/2\\
3/2
\end{pmatrix}
=
\begin{pmatrix}
1\\
1
\end{pmatrix}
$$
또한
$$
\mathrm{Var}(\mathbf{Y})=A\mathrm{Var}(\mathbf{X})A^T
=
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix}
$$
이다. □

### 정리 2.5.4 (분산행렬의 성질)
확률벡터 $\mathbf{X}$의 분산행렬 $\mathrm{Var}(\mathbf{X})$는 대칭이며 음이 아닌 정부호(non-negative definite)이다. 즉
$$
\mathrm{Var}(\mathbf{X})=\mathrm{Var}(\mathbf{X})^T,\qquad
a^T\mathrm{Var}(\mathbf{X})a\ge0\ \ (\forall a\in\mathbb{R}^k)
$$

#### 증명
대칭성은
$$
\mathrm{Var}(\mathbf{X})=(\mathrm{Cov}(X_i,X_j))
$$
에서 $\mathrm{Cov}(X_i,X_j)=\mathrm{Cov}(X_j,X_i)$이므로 즉시 따른다.

이제 임의의 $a\in\mathbb{R}^k$에 대해 $a^T\mathbf{X}$는 실수값 확률변수이므로
$$
\mathrm{Var}(a^T\mathbf{X})\ge0
$$

한편, 분산은 항상 비음이므로 정리 2.5.3(b)에 의해 $$\mathrm{Var(a^T\mathbf{X})} = a^T\mathrm{Var}(\mathbf{X})a \ge0$$  

**참고: Statistics_02_분산행렬의 스펙트럼 분해와 기하학적 해석.md** 

### 다차원 결합적률과 결합적률생성함수 *(Multivariate Joint Moments and Joint Moment Generating Functions)*
다차원 확률벡터 $\mathbf{X} = (X_1,\dots,X_k)^T$에 대하여, 자연수 $r_1,\dots,r_k \in \mathbb{N}$가 주어졌을 때
$$
E(X_1^{r_1}\cdots X_k^{r_k})
$$
를 $(r_1,\dots,r_k)$차 **결합적률(joint moment)** 이라 한다.

이는 단변량 확률변수의 적률(moment)을 다차원으로 확장한 개념이며, 확률벡터의 분포적 특성을 요약하는 기본적인 수치이다.

### 결합적률생성함수 *(Joint Moment Generating Function, Joint MGF)*
다차원 확률벡터 $\mathbf{X}=(X_1,\dots,X_k)^T$에 대하여,
$\mathbf{0}$을 포함하는 어떤 열린집합에서
$$
M(t_1,\dots,t_k)
=E\left(e^{t_1X_1+\cdots+t_kX_k}\right)
$$
가 유한하면 이를 $\mathbf{X}$의 **결합적률생성함수(joint moment generating function)** 라 한다.  
즉, 어떤 $h_i>0$가 존재하여 $-h_i<t_i<h_i\quad (i=1,\dots,k)$인 모든 $(t_1,\dots,t_k)$에 대해 $M(t_1,\dots,t_k)<\infty$가 성립하는 경우이다.

### 정리 2.5.5 결합적률생성함수의 성질 *(Properties of the Joint Moment Generating Function)*

#### (a) 결합적률 생성 성질 *(Generation of Joint Moments)*
다차원 확률벡터 $\mathbf{X}=(X_1,\dots,X_k)^T$의 결합적률생성함수
$M(t_1,\dots,t_k)=E(e^{t_1X_1+\cdots+t_kX_k})$가 원점을 포함하는 열린집합에서 존재한다고 가정한다.

이때 $\mathbf{X}$의 모든 결합적률이 존재하며,
$$
E(X_1^{r_1}\cdots X_k^{r_k})
=\left[
\frac{\partial^{r_1+\cdots+r_k}}
{\partial t_1^{r_1}\cdots \partial t_k^{r_k}}
M(t_1,\dots,t_k)
\right]_{t_1=\cdots=t_k=0}
$$
가 성립한다.

또한 결합적률생성함수는 다음의 멱급수(power series)로 전개된다.
$$
M(t_1,\dots,t_k)
=\sum_{r_1=0}^{\infty}\cdots\sum_{r_k=0}^{\infty}
\frac{E(X_1^{r_1}\cdots X_k^{r_k})}
{r_1!\cdots r_k!}
t_1^{r_1}\cdots t_k^{r_k}.
$$

이는 지수함수의 멱급수 전개와 적분–미분 교환의 정당화(예: 지배수렴정리)에 근거한다.

#### (b) 분포 결정성 *(Distribution Determination / Uniqueness)*
다차원 확률벡터
$\mathbf{X}=(X_1,\dots,X_k)^T,\ \mathbf{Y}=(Y_1,\dots,Y_k)^T$에 대하여 결합적률생성함수
$$
M_\mathbf{X}(t_1,\dots,t_k),\quad M_\mathbf{Y}(t_1,\dots,t_k)
$$
가 원점을 포함하는 열린집합에서 존재하고,
$$
M_\mathbf{X}(t_1,\dots,t_k)=M_\mathbf{Y}(t_1,\dots,t_k)
$$
가 성립하면 $\mathbf{X}$와 $\mathbf{Y}$의 확률분포는 동일하다.

즉, 결합확률밀도함수(joint probability density function)와 결합누적분포함수(joint cumulative distribution function)가 일치한다.

### 결합누율생성함수 *(Cumulant Generating Function, CGF)*
결합적률생성함수 $M(t_1,\dots,t_k)$가 존재할 때
$$
C(t_1,\dots,t_k) =\log M(t_1,\dots,t_k)
=\log E\left(e^{t_1X_1+\cdots+t_kX_k}\right)
$$
를 **결합누율생성함수(cumulant generating function)** 라 한다.

### 결합누율생성함수와 평균, 분산 *(Mean and Covariance via CGF)*
결합누율생성함수 $C(\mathbf{t})$에 대하여 다음이 성립한다.
$$
\left.
\frac{\partial C(\mathbf{t})}{\partial t_i}
\right|_{t_1=\cdots=t_k=0}
=E(X_i),
$$

$$
\left.
\frac{\partial^2 C(\mathbf{t})}{\partial t_i\partial t_j}
\right|_{t_1=\cdots=t_k=0}
=\mathrm{Cov}(X_i,X_j).
$$

일차 편도함수 벡터
$$
\nabla C(\mathbf{t}) 
=\dot C(\mathbf{\mathbf{t}})
=\left(
\frac{\partial C}{\partial t_1},\dots,
\frac{\partial C}{\partial t_k}
\right)^T
$$
를 **기울기 벡터(gradient vector)** 라 하며,

이차 편도함수 행렬
$$
\nabla^2 C(\mathbf{t})
=\ddot C(\mathbf{\mathbf{t}})
=\left(
\frac{\partial^2 C}{\partial t_i\partial t_j}
\right)
$$
를 **헤시안 행렬(Hessian matrix)** 이라 한다.

### 예시 2.5.9 *(Computation via CGF)*
확률벡터 $\mathbf{X}=(X_1,X_2,X_3)^T$의 결합확률밀도함수는
$$
f(x_1,x_2,x_3)
=6e^{-x_1-x_2-x_3}
\mathbf{1}_{\{0\le x_1\le x_2\le x_3\}}
$$
이때 결합적률생성함수는
$$
M_\mathbf{X}(t_1,t_2,t_3)
=\int_0^\infty\int_{x_1}^\infty\int_{x_2}^\infty e^{t_1x_1+t_2x_2+t_3x_3}6e^{-x_1-x_2-x_3}dx_3\,dx_2\,dx_1
$$
$$
= 6\int_0^\infty e^{(t_1-1)x_1}\left[\int_{x_1}^\infty e^{(t_2-1)x_2}\left(\int_{x_2}^\infty e^{(t_3-1)x_3}dx_3\right)dx_2\right]dx_1
$$
$$
=\frac{6}{(3-t_1-t_2-t_3)(2-t_2-t_3)(1-t_3)}
$$
이며,$t_1+t_2+t_3<3,\quad t_2+t_3<2,\quad t_3<1$ 에서 정의된다.

따라서 결합누율생성함수는
$$
C(\mathbf{t})
=-\log\left(1-\frac{t_1+t_2+t_3}{3}\right)
-\log\left(1-\frac{t_2+t_3}{2}\right)
-\log(1-t_3).
$$

로그함수의 멱급수 전개
$$
-\log(1-A)=A+\frac{A^2}{2}+\frac{A^3}{3}+\cdots
\quad(-1<A<1)
$$
를 이용하여 $C(\mathbf{t})$를 전개하면,
일차항과 이차항의 계수로부터
$$
E(\mathbf{X})
= 
\begin{pmatrix}
1/3\\
5/6\\
11/6
\end{pmatrix},
$$
$$
\mathrm{Var}(\mathbf{X})
= \frac{1}{36}
\begin{pmatrix}
4&4&4\\
4&13&13\\
4&13&49
\end{pmatrix}
$$
을 얻는다.

### 다차원 확률변수의 조건부확률밀도함수 *(Conditional Probability Density Function)*
다차원 확률벡터 $\mathbf{X}=(X_1,\dots,X_k)^T$의 결합확률밀도함수가
$f_{1,\dots,k}(x_1,\dots,x_k)$이고,
$X_1$의 주변확률밀도함수가 $f_1(x_1)$일 때,
$X_1=x_1$이라는 조건 하에서
$(X_2,\dots,X_k)^T$의 조건부확률밀도함수는
$$
f_{2,\dots,k|1}(x_2,\dots,x_k\mid x_1)
= \frac{f_{1,\dots,k}(x_1,\dots,x_k)}{f_1(x_1)},
\quad f_1(x_1)>0
$$
로 정의된다.  

일반적으로, 확률벡터를 두 부분 $\mathbf{X}=(X_1,\dots,X_j)^T$와 $\mathbf{Y}=(X_{j+1},\dots,X_k)^T$로 나누었을 때,
$\mathbf{X}=\mathbf{x}$가 주어진 조건 하에서 $\mathbf{Y}$의 조건부확률밀도함수는
$$
f_{\mathbf{Y}|\mathbf{X}}(\mathbf{y}|\mathbf{x})
= \frac{f_{1,\dots,k}(x_1,\dots,x_j,y_1,\dots,y_{k-j})}{f_{1,\dots,j}(x_1,\dots,x_j)},
\quad f_{1,\dots,j}(\mathbf{x})>0
$$
로 정의된다.

이는 이차원 확률변수에서의 조건부확률밀도함수 정의를 그대로 다차원으로 확장한 것이다.

### 다차원 확률변수의 조건부기댓값 *(Conditional Expectation)*
확률벡터
$$
\mathbf{X}=(X_1,\dots,X_k)^T,\quad
\mathbf{Y}=(Y_1,\dots,Y_\ell)^T
$$
에 대하여 $\mathbf{X}=\mathbf{x}$일 때 $\mathbf{Y}$의 조건부확률밀도함수가 $f_{\mathbf{Y}|\mathbf{X}}(\mathbf{y}\mid \mathbf{x})$이면,
실수값 함수 $g(\mathbf{X},\mathbf{Y})$의 조건부기댓값은
$$
E(g(\mathbf{X},\mathbf{Y})\mid \mathbf{X}=\mathbf{x}) = \int\cdots\int g(\mathbf{x},\mathbf{y})\,f_{\mathbf{Y}|\mathbf{X}}(\mathbf{y}\mid \mathbf{x})\,d\mathbf{y}
$$
로 정의된다(연속형의 경우).

### 정리 2.5.6 조건부기댓값의 성질 *(Properties of Conditional Expectation)*

#### (a) 반복기댓값의 법칙 *(Law of Iterated Expectations)*
$$
E[E(\mathbf{Y}\mid \mathbf{X})] = E(\mathbf{Y}).
$$

#### (b) 직교성 *(Orthogonality Property)*
$$
\forall v(\mathbf{X}), \ \mathrm{Cov}(\mathbf{Y}-E(\mathbf{Y}\mid \mathbf{X}),\,v(\mathbf{X}))=\mathbf{0}
$$
가 성립한다.  
이는 조건부기댓값이 $L^2$ 공간에서 $\mathbf{Y}$의 $\mathbf{X}$-가측 함수 공간으로의 정사영(projection)임을 의미한다.

### 정리 2.5.7 분산행렬의 분해 *(Variance Decomposition / Law of Total Variance)*
$$
\mathrm{Var}(\mathbf{Y})
= E[\mathrm{Var}(\mathbf{Y}\mid \mathbf{X})]
+ \mathrm{Var}(E(\mathbf{Y}\mid \mathbf{X})).
$$
#### 증명
$\mu = E(\mathbf{Y})$라 하자. $\mathbf{Y}-\mu$를 다음과 같이 분해한다:
$$
\mathbf{Y}-\mu=(\mathbf{Y}-E(\mathbf{Y}\mid \mathbf{X}))\oplus(E(\mathbf{Y}\mid \mathbf{X})-\mu)
$$

이 분해가 $\oplus$로 가능함을 확인하자.  
$E(\mathbf{Y}\mid \mathbf{X})-\mu$는 $\mathbf{X}$의 함수이므로, 정리 2.5.6(b)의 직교성 성질에 의해
$$
\mathrm{Cov}(\mathbf{Y}-E(\mathbf{Y}\mid \mathbf{X}),\ E(\mathbf{Y}\mid \mathbf{X})-\mu)=\mathbf{0}
$$
가 성립한다. 따라서 두 항은 서로 비상관이며, $\oplus$ 표기가 정당하다.

이제 양변에서 분산을 구하면, $\oplus$의 정의에 의해 분산이 가법적으로 분해되어
$$
\mathrm{Var}(\mathbf{Y})=\mathrm{Var}(\mathbf{Y}-E(\mathbf{Y}\mid \mathbf{X}))\oplus\mathrm{Var}(E(\mathbf{Y}\mid \mathbf{X})-\mu)
$$
$$
=\mathrm{Var}(\mathbf{Y}-E(\mathbf{Y}\mid \mathbf{X}))+\mathrm{Var}(E(\mathbf{Y}\mid \mathbf{X}))
$$

여기서
$$
\mathrm{Var}(\mathbf{Y}-E(\mathbf{Y}\mid \mathbf{X}))
=E[(\mathbf{Y}-E(\mathbf{Y}\mid \mathbf{X}))(\mathbf{Y}-E(\mathbf{Y}\mid \mathbf{X}))^T]
$$

조건부기댓값의 타워 성질과 정리 2.5.6(a)를 이용하면
$$
E[(\mathbf{Y}-E(\mathbf{Y}\mid \mathbf{X}))(\mathbf{Y}-E(\mathbf{Y}\mid \mathbf{X}))^T]
=E[E((\mathbf{Y}-E(\mathbf{Y}\mid \mathbf{X}))(\mathbf{Y}-E(\mathbf{Y}\mid \mathbf{X}))^T\mid \mathbf{X})]
$$
$$
=E[\mathrm{Var}(\mathbf{Y}\mid \mathbf{X})]
$$

따라서
$$
\mathrm{Var}(\mathbf{Y})
=E[\mathrm{Var}(\mathbf{Y}\mid \mathbf{X})]+\mathrm{Var}(E(\mathbf{Y}\mid \mathbf{X}))
$$
□

### 정리 2.5.8 다차원 확률변수의 최소제곱예측자 *(Minimum Mean Square Error Estimator)*
벡터값 함수 $u(\mathbf{X})$에 대하여
$$
E\|\mathbf{Y}-u(\mathbf{X})\|^2 = E[(Y_1-u_1(\mathbf{X}))^2+\cdots+(Y_\ell-u_\ell(\mathbf{X}))^2]
$$
를 최소로 하는 함수는
$$
u(\mathbf{X})=E(\mathbf{Y}\mid \mathbf{X})
$$
이다.

즉, 조건부기댓값은 평균제곱오차(mean squared error) 기준에서 최적 예측자이다.

#### 증명
임의의 $\mathbf{X}$의 벡터값 함수 $u(\mathbf{X})=(u_1(\mathbf{X}),\dots,u_\ell(\mathbf{X}))^T$에 대하여, 다음과 같이 분해한다.
$$
\mathbf{Y}-u(\mathbf{X})=(\mathbf{Y}-E(\mathbf{Y}\mid \mathbf{X}))\oplus(E(\mathbf{Y}\mid \mathbf{X})-u(\mathbf{X}))
$$
이 분해는 정리 2.5.6(b)의 직교성 성질에 의해 $\oplus$로 가능하다.

이제 양변의 제곱 노름의 기댓값을 취하면
$$
E\|\mathbf{Y}-u(\mathbf{X})\|^2
=E\|(\mathbf{Y}-E(\mathbf{Y}\mid \mathbf{X}))\oplus(E(\mathbf{Y}\mid \mathbf{X})-u(\mathbf{X}))\|^2 \\
=E\|\mathbf{Y}-E(\mathbf{Y}\mid \mathbf{X})\|^2+E\|E(\mathbf{Y}\mid \mathbf{X})-u(\mathbf{X})\|^2
$$

두 번째 항은 항상 비음이므로
$$
E\|\mathbf{Y}-u(\mathbf{X})\|^2\ge E\|\mathbf{Y}-E(\mathbf{Y}\mid \mathbf{X})\|^2
$$

등호는
$$
E(\mathbf{Y}\mid \mathbf{X})-u(\mathbf{X})=\mathbf{0}\quad\text{a.s.}
$$
즉 $u(\mathbf{X})=E(\mathbf{Y}\mid \mathbf{X})$일 때, 그리고 그때에만 성립한다. □

### 정리 2.5.9 다차원 확률변수의 독립성 *(Independence of Multivariate Random Variables)*
다차원 확률벡터 $\mathbf{X}_1,\dots,\mathbf{X}_n$이 서로 독립일 필요충분조건은 다음과 같다.

#### (a) 확률밀도함수에 의한 조건
$$
f_{1,\dots,n}(\mathbf{x}_1,\dots,\mathbf{x}_n)
= f_1(\mathbf{x}_1)\cdots f_n(\mathbf{x}_n).
$$

#### (b) 적률생성함수에 의한 조건
$$
M_{1,\dots,n}(\mathbf{t}_1,\dots,\mathbf{t}_n)
= M_1(\mathbf{t}_1)\cdots M_n(\mathbf{t}_n)
$$
(각 MGF가 원점 근방에서 존재함을 가정).

**참고: 독립성 판정 시 인수분해 기준**  
독립성을 판정할 때 반드시 주변확률밀도함수나 주변적률생성함수를 명시적으로 구할 필요는 없다.

결합확률밀도함수가 각 확률벡터에만 의존하는 함수들의 곱으로 인수분해되면, 즉
$$
f_{1,\dots,n}(\mathbf{x}_1,\dots,\mathbf{x}_n) = g_1(\mathbf{x}_1) \cdot g_2(\mathbf{x}_2) \cdots g_n(\mathbf{x}_n)
$$
를 만족하는 비음함수 $g_1,\dots,g_n$이 존재하면, $\mathbf{X}_1,\dots,\mathbf{X}_n$은 서로 독립이다.

이는 2차원의 경우(정리 2.4.1의 인수분해 기준)와 동일한 원리이며, 각 $g_i$를 적절히 정규화하면 주변확률밀도함수를 얻을 수 있다.

마찬가지로 결합적률생성함수가
$$
M_{1,\dots,n}(\mathbf{t}_1,\dots,\mathbf{t}_n) = h_1(\mathbf{t}_1) \cdot h_2(\mathbf{t}_2) \cdots h_n(\mathbf{t}_n)
$$
형태로 인수분해되면, 각 $h_i$가 주변적률생성함수가 되므로 독립성이 성립한다.

### 정리 2.5.10 독립인 다차원 확률변수들의 성질 *(Properties of Independent Random Variables)*
* 독립인 확률벡터에 함수 $g_i$를 적용해도 독립성은 유지된다.
* 곱의 기댓값은 기댓값의 곱으로 분리된다.
* 서로 다른 성분 사이의 공분산은 0이다.

### 정리 2.5.11 서로 독립인 확률변수의 합 *(Sum of Independent Random Variables)*
독립인 확률변수 $X_1,\dots,X_n$에 대하여
$$
\mathrm{Var}(X_1+\cdots+X_n)
=\mathrm{Var}(X_1)+\cdots+\mathrm{Var}(X_n),
$$

$$
M_{X_1+\cdots+X_n}(t)
= M_{X_1}(t)\cdots M_{X_n}(t).
$$

#### 증명
**분산의 경우**  
분산은 공분산의 특수한 경우이므로
$\mathrm{Var}(Y)=\mathrm{Cov}(Y,Y)$
를 이용한다. 따라서
$$
\mathrm{Var}(X_1+\cdots+X_n)
=\mathrm{Cov}\!\left(\sum_{i=1}^n X_i,\,\sum_{j=1}^n X_j\right).
$$

공분산의 이중 선형성(bilinearity)에 의해
$$
\mathrm{Cov}\!\left(\sum_{i=1}^n X_i,\,\sum_{j=1}^n X_j\right)
=\sum_{i=1}^n\sum_{j=1}^n \mathrm{Cov}(X_i,X_j).
$$

독립성에 의해 $i\neq j$이면 $\mathrm{Cov}(X_i,X_j)=0$
이고, $i=j$이면 $\mathrm{Cov}(X_i,X_i)=\mathrm{Var}(X_i).$

따라서
$$
\mathrm{Var}(X_1+\cdots+X_n)
=\sum_{i=1}^n \mathrm{Var}(X_i).
$$

**적률생성함수의 경우**  
적률생성함수의 정의에 의해
$$
M_{X_1+\cdots+X_n}(t)
=E\!\left[e^{t(X_1+\cdots+X_n)}\right]
=E\!\left[\prod_{i=1}^n e^{tX_i}\right].
$$

확률변수 $X_1,\dots,X_n$이 서로 독립이므로
함수 $e^{tX_1},\dots,e^{tX_n}$ 역시 서로 독립이고,
따라서
$$
E\!\left[\prod_{i=1}^n e^{tX_i}\right]
=\prod_{i=1}^n E[e^{tX_i}]
=\prod_{i=1}^n M_{X_i}(t).
$$
□
