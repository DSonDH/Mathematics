# Chapter 16 벌점화 회귀분석 방법 (Penalized Regression Methods)

벌점화 회귀분석 (penalized regression analysis)은 고차원 회귀분석에서 중요한 도구다. 기본 아이디어는 잔차제곱합에 벌점항 (penalty term)을 추가하여 회귀계수의 크기와 복잡성을 동시에 제어하는 데 있다. 이를 통해 예측 안정성 (prediction stability)을 높이고, 경우에 따라 변수선택 (variable selection)까지 수행할 수 있다.

이 장에서는 설명변수를 먼저 표준화 (standardization)하여 다음 조건을 가정한다.

$$\sum_{i=1}^n x_{ij}=0,\qquad \sum_{i=1}^n x_{ij}^2=n,\qquad j=1,2,\dots,p$$

즉, 각 설명변수는 평균이 0이고 제곱합이 $n$이 되도록 조정된다. 또한 절편이 없는 선형회귀모형을 가정한다. 이때 회귀계수 추정은 일반적으로 다음 벌점화 오차제곱합 (penalized sum of squares)을 최소로 하는 해로 정의된다.

$$\frac{1}{n}\sum_{i=1}^n\left(y_i-\sum_{j=1}^p \beta_j x_{ij}\right)^2+\sum_{j=1}^p J_\lambda(\beta_j) = \frac{1}{n}SSE+\sum_{j=1}^p J_\lambda(\beta_j)$$

여기서 $J_\lambda(\beta_j)$는 회귀계수 $\beta_j$에 대한 벌점함수 (penalty function)이고, $\lambda>0$는 벌점의 강도를 조절하는 조절모수 (tuning parameter)다. $\lambda$가 커질수록 회귀계수는 더 강하게 0 방향으로 축소 (shrinkage)된다.


## 16.1 라쏘회귀분석 (Lasso Regression)

라쏘회귀분석 (lasso regression)은 벌점화 회귀분석 가운데 가장 대표적인 방법 중 하나다. 라쏘는 $\ell_1$-노름 ($\ell_1$-norm)에 기반한 벌점항을 사용하며, 회귀계수를 0으로 만드는 효과가 있어 변수선택과 축소추정을 동시에 수행한다.

### 16.1.1 벌점화 회귀분석의 기본 구조

벌점화 회귀분석에서는 손실함수 (loss function)와 벌점함수 (penalty function)를 결합하여 추정량을 정의한다. 기본 구조는 다음과 같다.

$$\text{손실함수}+\text{벌점함수}$$

회귀문제에서는 손실함수로 제곱합 손실 (squared loss)을 사용하는 경우가 기본적이며, 벌점함수는 회귀계수의 복잡도를 측정하는 역할을 한다. 벌점항을 도입하면 단순히 데이터 적합도만 최대화하는 것이 아니라, 복잡도가 낮은 모형을 선호하게 된다.

이때 벌점은 다음 두 기능을 가진다.

1. 회귀계수의 크기를 줄이는 축소 (shrinkage)
2. 불필요한 변수를 제거하는 변수선택 (variable selection)

### 16.1.2 대표적 벌점함수

대표적인 벌점함수는 다음 세 가지다.

#### $\ell_0$-노름 ($\ell_0$-norm)

$$J_\lambda(z)=\lambda I(z\neq 0)$$

이는 0이 아닌 회귀계수의 개수에 벌점을 부과한다. 따라서 추정된 $\beta$에서 0이 아닌 계수의 개수는 모형의 복잡성 (model complexity)에 대한 직접적인 척도가 된다. 가장 직관적인 변수선택 기준이지만, 계산량이 매우 커진다.

#### $\ell_1$-노름 ($\ell_1$-norm)

$$J_\lambda(z)=\lambda |z|$$

이는 Tibshirani가 회귀분석의 변수선택 방법으로 제안한 벌점이다. 이 벌점함수 또는 이를 이용한 회귀분석을 라쏘 (lasso)라고 부른다. $\ell_1$-벌점은 볼록성 (convexity)을 유지하면서도 회귀계수를 정확히 0으로 만들 수 있다는 점이 핵심이다.

#### $\ell_2$-노름 ($\ell_2$-norm)

$$J_\lambda(z)=\lambda z^2$$

이 경우의 추정량은 능형회귀추정량 (ridge estimator)에 해당한다. 이는 예측의 안정성과 정확성 측면에서 장점을 가지지만, 일반적으로 변수선택 기능은 없다.

### 16.1.3 조절모수의 역할

조절모수 $\lambda$의 선택은 벌점화 회귀분석의 핵심이다.

* $\lambda=0$이면 벌점이 사라져 최소제곱법 (ordinary least squares)으로 돌아간다.
* $p\le n$인 경우에는 일반적으로 0이 아닌 해가 유일하게 존재하며, 모든 설명변수가 모형에 포함된다.
* 반대로 $p>n$인 고차원 회귀분석 (high-dimensional regression)에서는 최소제곱해가 유일하게 정해지지 않을 수 있다.
* $\lambda=\infty$에 가까워지면 벌점이 지나치게 커져 $\beta_1=\beta_2=\cdots=\beta_p=0$이 되며, 표준화 이전 기준으로는 절편만 남는 모형에 대응한다.

즉, $\lambda=0$은 가장 복잡한 모형, $\lambda=\infty$는 가장 단순한 모형에 해당한다. 따라서 적절한 $\lambda$ 선택은 모형선택 (model selection)의 핵심 단계다.

### 16.1.4 손실함수와 벌점의 확장

벌점화 회귀분석은 두 방향으로 확장될 수 있다.

첫째, 손실함수를 바꿀 수 있다. 제곱합 손실 대신 다음과 같은 손실함수를 사용할 수 있다.

* 힌지 손실 (hinge loss)
* 로지스틱 손실 (logistic loss)
* 허버 손실 (Huber loss)

이러한 손실함수들은 계산 편의를 위해 대체로 볼록성 (convexity)을 유지한다.

둘째, 벌점함수를 확장할 수 있다. $\ell_1$-벌점을 변형한 방법으로 다음과 같은 확장들이 있다.

* 적응 라쏘 (adaptive lasso)
* 그룹 라쏘 (group lasso)
* fused 라쏘 (fused lasso)

이들 방법은 변수들 사이의 구조를 반영하거나, 특정한 형태의 희소성 (sparsity)을 유도하기 위해 사용된다.

### 16.1.5 분해가능성

벌점함수의 중요한 구조적 성질 중 하나가 분해가능성 (decomposability)이다. 설명변수 전체가 $p$차원 벡터공간을 이루고, 실제로 유의한 회귀계수가 $s$개뿐이라고 하자. 즉,

$$\beta_j\neq 0\quad (j=1,\dots,s),\qquad \beta_j=0\quad (j=s+1,\dots,p)$$

라고 하자. $\boldsymbol{\beta}=(\beta_1,\dots,\beta_p)$를 유효한 부분과 나머지 부분으로 나누어

$$\boldsymbol{\beta}_S=(\beta_1,\dots,\beta_s),\qquad \boldsymbol{\beta}_{S^\perp}=(\beta_{s+1},\dots,\beta_p)$$

라고 하면, $\ell_1$-노름은

$$|\boldsymbol{\beta}|_1=\sum_{j=1}^p |\beta_j| = \sum_{j=1}^s |\beta_j|+\sum_{j=s+1}^p |\beta_j| = |\boldsymbol{\beta}_S|_1+|\boldsymbol{\beta}_{S^\perp}|_1$$

로 분해된다.

이 성질 때문에 $\ell_1$-벌점은 희소한 구조를 다루는 데 특히 적합하다. 즉, 유의한 계수와 그렇지 않은 계수를 분리하여 해석할 수 있게 해준다.

#### 예제: $\ell_0$-벌점과 모든 가능한 회귀 (All Possible Regressions)

$\ell_0$-벌점을 사용하면 $|\boldsymbol{\beta}|_0=s$라는 제약 아래에서 잔차제곱합 (SSE)를 최소로 하는 문제를 풀게 된다. 이는 크기가 $s$인 설명변수 부분집합 중에서 가장 좋은 조합을 찾는 문제와 동일하다.

각 $s$에 대해 가능한 모형의 수는 $\binom{p}{s}$개이고, 전체적으로는 $s=1,\dots,p$를 모두 고려해야 하므로 계산량은 사실상 $O(2^p)$ 수준으로 커진다. 따라서 $p$가 큰 고차원 문제에서는 현실적으로 사용하기 어렵다.

이 계산문제를 완화하기 위해 $\ell_0$-벌점 대신 개념적으로 유사하면서도 목적함수를 볼록하게 만드는 $\ell_1$-벌점, 즉 라쏘가 도입된다.


## 16.2 라쏘추정량 정의 (Definition of Lasso Estimator)

라쏘는 $\ell_0$-벌점이 가지는 계산상의 어려움을 해결하기 위해 제안된 방법이다. $\ell_1$-벌점은 $\ell_0$-벌점에 개념적으로 가장 가까우면서도, 최적화 문제를 볼록하게 만든다.

라쏘 추정량 (lasso estimator)은 다음 목적함수를 최소화하는 해로 정의된다.

$$\frac{1}{n}\sum_{i=1}^n\left(y_i-\sum_{j=1}^p {\beta}_j x_{ij}\right)^2+\lambda \sum_{j=1}^p |{\beta}_j| = \frac{1}{n}SSE+\lambda\sum_{j=1}^p |{\beta}_j|$$

이를 최소화하는 $\hat{\boldsymbol{\beta}}^{lasso}$를 라쏘추정량이라 한다.

### 16.2.1 직교설계 (Orthogonal Design)에서의 라쏘추정량

라쏘의 구조를 이해하기 위해 설명변수 행렬이 직교 (orthogonal)한다고 가정하자. $X^TX=nI_p$ 이 경우 $\lambda=0$이면 라쏘추정량은 최소제곱추정량 (least squares estimator)과 동일하다.

$$\hat{\boldsymbol{\beta}}^{lse}=(X^TX)^{-1}X^T\mathbf{y}=\frac{1}{n}X^T\mathbf{y}$$

이제 $\lambda>0$일 때 라쏘추정량은 각 좌표별로 다음과 같이 주어진다.

$$\hat{\boldsymbol{\beta}}^{lasso}_j = \operatorname{sgn}(\hat{\boldsymbol{\beta}}^{lse}_j)\max\left(|\hat{\boldsymbol{\beta}}^{lse}_j|-\lambda,0\right)$$

이 식은 라쏘가 최소제곱추정량을 일정 문턱값 (thresholding value) $\lambda$만큼 0 방향으로 줄이고, 그 절댓값이 $\lambda$ 이하이면 정확히 0으로 만든다는 것을 보여준다. 이를 소프트 임계화 (soft-thresholding)라고 한다.

#### 증명: KKT 조건 (Karush-Kuhn-Tucker Conditions)에 의한 유도

라쏘의 목적함수를 계산 편의를 위해 다음과 같이 쓰자.

$$Q_\lambda(\boldsymbol{\beta})=\frac{1}{2n}(y-X\boldsymbol{\beta})^T(y-X\boldsymbol{\beta})+\lambda |\boldsymbol{\beta}|_1$$

라쏘추정량에서 0이 아닌 계수의 인덱스 집합을

$$A=\{j:\hat{\boldsymbol{\beta}}^{lasso}_j\neq 0\}$$

라고 하자. 그러면 $j\in A$에 대해서는 목적함수가 미분 가능하므로 KKT 조건은

$$-\frac{1}{n}x_j^T(y-X\hat{\boldsymbol{\beta}}^{lasso})+\lambda \operatorname{sgn}(\hat{\boldsymbol{\beta}}^{lasso}_j)=0$$

이 된다.  
반면 $j\notin A$에서는 $\beta_j=0$에서의 부분도함수 (subgradient)를 고려해야 하므로

$$\left| -x_j^T(y-X\hat{\boldsymbol{\beta}}^{lasso}) \right| \le n\lambda$$

가 성립한다.  
이제 직교설계 ($X^TX=nI_p$)를 사용하면, $j\in A$에 대해

$$-\hat{\boldsymbol{\beta}}^{lse}_j+\hat{\boldsymbol{\beta}}^{lasso}_j+\lambda \operatorname{sgn}(\hat{\boldsymbol{\beta}}^{lasso}_j)=0$$

를 얻는다. 또한 $j\notin A$에 대해서는

$$|\hat{\boldsymbol{\beta}}^{lse}_j|\le \lambda$$

가 된다.  
라쏘해와 최소제곱해는 부호가 같으므로,

$$\hat{\boldsymbol{\beta}}^{lasso}_j = \operatorname{sgn}(\hat{\boldsymbol{\beta}}^{lse}_j)\max\left(|\hat{\boldsymbol{\beta}}^{lse}_j|-\lambda,0\right)$$

가 도출된다. 이것이 라쏘의 소프트 임계화 공식이다.

> KKT조건은 블록최적화 문제의 해에 대한 필요충분조건으로, 라쏘회귀 뿐만 아니라 제약조건이 있는 최대우도추정량의 계상 등 여러 통계문제들의 추정량을 계산할 떄 사용된다. 자세한 조건은 Boyd, Vandenberghe (참고문헌 16.1)이나 다른 Convex Optimization 교재 참고.

### 16.2.2 라쏘의 핵심 작동원리

위 정의식에서 확인할 수 있는 라쏘의 핵심은 다음 두 가지다.

1. 큰 계수는 0 방향으로 연속적으로 축소된다.
2. 작은 계수는 정확히 0이 되어 변수선택이 발생한다.

즉, 라쏘는 단순한 축소추정량 (shrinkage estimator)이면서 동시에 희소추정량 (sparse estimator)이다.


## 16.3 라쏘추정량의 이해 (Understanding Lasso Estimator)

라쏘추정량을 직관적으로 이해하려면 두 가지 경우를 구분하는 것이 좋다.

1. 설명변수 행렬이 직교하는 경우
2. 설명변수 행렬이 직교하지 않는 경우

### 16.3.1 직교설계에서의 해석

직교설계에서는 각 회귀계수가 서로 독립적으로 처리된다. 이 경우 라쏘는 각 최소제곱추정량에 대해 개별적으로 소프트 임계화를 적용한다.

$$\hat{\boldsymbol{\beta}}^{lasso}_j = \operatorname{sgn}(\hat{\boldsymbol{\beta}}^{lse}_j)\max\left(|\hat{\boldsymbol{\beta}}^{lse}_j|-\lambda,0\right)$$

이 식은 다음 의미를 가진다.

* $|\hat{\boldsymbol{\beta}}^{lse}_j|\le \lambda$이면 $\hat{\boldsymbol{\beta}}^{lasso}_j=0$이다.
* $|\hat{\boldsymbol{\beta}}^{lse}_j|>\lambda$이면 $\hat{\boldsymbol{\beta}}^{lasso}_j$는 원래 값과 같은 부호를 유지한 채 0 방향으로 $\lambda$만큼 줄어든다.

따라서 라쏘는 작은 계수는 제거하고 큰 계수는 축소한다. 이 때문에 라쏘추정량을 축소추정량 (shrinkage estimator)이라 부른다.

#### 예제: $p=2$인 직교설계

$p=2$일 때 $\lambda$가 증가하면 두 회귀계수의 절댓값은 점차 감소한다. 어느 계수의 최소제곱추정량 절댓값이 $\lambda$ 이하가 되는 순간, 그 계수는 즉시 0이 된다. 이후 해당 변수는 회귀모형에서 제외된다.

즉, 계수 경로 (coefficient path)는 연속적으로 줄어들다가 특정 문턱에서 정확히 0에 도달한다. 이것이 라쏘의 변수선택 메커니즘이다.

### 16.3.2 비직교설계 (Non-Orthogonal Design)에서의 해석

설명변수 행렬이 직교하지 않는 $p=2$의 경우를 생각하자. 제곱합 손실함수는 최소제곱추정량 $\hat{\boldsymbol{\beta}}^{lse}$를 사용하여 다음과 같이 다시 쓸 수 있다.

$$L(\boldsymbol{\beta})=\frac{1}{n}(y-X\boldsymbol{\beta})^T(y-X\boldsymbol{\beta}) = \frac{1}{n}(\boldsymbol{\beta}-\hat{\boldsymbol{\beta}}^{lse})^T(X^TX)(\boldsymbol{\beta}-\hat{\boldsymbol{\beta}}^{lse})+\text{상수}$$

따라서 라쏘추정량은 어떤 양의 상수 $t$에 대해 다음 제약최적화 문제와 동치다.

$$\min_{\boldsymbol{\beta}} \frac{1}{n}(\boldsymbol{\beta}-\hat{\boldsymbol{\beta}}^{lse})^T(X^TX)(\boldsymbol{\beta}-\hat{\boldsymbol{\beta}}^{lse}) \quad \text{subject to} \quad |\boldsymbol{\beta}|_1=\sum_{j=1}^p |\boldsymbol{\beta}_j|\le t$$

여기서 $t$는 조절모수 $\lambda$에 대응하는 상수다.

이 문제를 기하학적으로 보면,

* 손실함수의 등고선 (contour)은 타원형 (ellipse)이다.
* $\ell_1$-제약집합 ($|\boldsymbol{\beta}|_1\le t$)는 마름모꼴 (diamond)이다.

최적해는 보통 타원형 등고선이 마름모와 처음 접하는 점에서 결정된다. 마름모는 꼭짓점 (corner)을 가지므로, 최적점이 축 위의 꼭짓점에서 발생하기 쉽다. 이 경우 한 좌표가 정확히 0이 된다.

이것이 라쏘가 희소성 (sparsity)을 유도하는 기하학적 이유다.

#### 예제: $\ell_1$-제약집합의 기하학적 해석

$$|\boldsymbol{\beta}_1|+|\boldsymbol{\beta}_2|\le t$$

는 $(\boldsymbol{\beta}_1,\boldsymbol{\beta}_2)$ 평면에서 마름모 영역을 만든다. 타원형 손실함수의 중심은 최소제곱추정량 $\hat{\boldsymbol{\beta}}^{lse}$에 위치한다. 벌점이 없으면 해는 $\hat{\boldsymbol{\beta}}^{lse}$다. 그러나 $\ell_1$-제약이 도입되면 해는 마름모 내부 또는 경계로 이동해야 한다.

마름모의 경계는 축과 만나는 뾰족한 꼭짓점을 가지므로, 최적해가 이 꼭짓점에 도달할 가능성이 높다. 따라서 $\boldsymbol{\beta}_1=0$ 또는 $\boldsymbol{\beta}_2=0$ 같은 해가 자연스럽게 발생한다. 이것이 라쏘의 희소성 유도 원리다.


## 16.4 라쏘추정량의 이론적 성질 (Theoretical Properties of Lasso)

선형회귀모형을 따르는 데이터에 대하여 식 (16.3)을 최소로 만드는 라쏘추정량 $\hat{\boldsymbol{\beta}}^{lasso}$의 이론적 성질은 크게 두 가지로 나누어 볼 수 있다.

1. 추정오차 (estimation error), 즉 $|\hat{\boldsymbol{\beta}}^{lasso}-\boldsymbol{\beta}|_2$의 상한에 대한 성질
2. 변수선택 (variable selection)에 대한 일치성 (consistency)

이 두 성질은 서로 관련되어 있으나, 요구하는 조건은 서로 다르다. 먼저 추정오차의 성질을 살피기 위해 두 가지 가정을 둔다.

### 16.4.1 추정오차에 대한 성질

첫째, $\boldsymbol{\beta}$의 서포트 (support)를 $S=\{ j\mid \boldsymbol{\beta}_j\neq 0,; j=1,2,\dots,p \}$로 두고, 그 크기를 $|S|=s$라 하자. 즉, 실제로 0이 아닌 회귀계수의 개수는 $s$개라고 가정한다.

둘째, 설명변수 행렬 $X$가 $S$에 대하여 고윳값 제한 (restricted eigenvalue) 조건을 만족한다고 가정한다. 이를 위해 상수 $\alpha\ge 1$에 대하여 다음 원뿔집합 (cone set)을 정의한다.

$$C_\alpha(S):={\Delta\in \mathbb{R}^p \mid |\Delta_{S^c}|_1 \le \alpha|\Delta_S|_1} \tag{16.12}$$

여기서 $\Delta=(\delta_1,\dots,\delta_p)$라 하면, $\Delta_S$는 $j\in S$인 성분들로만 이루어진 $|S|$차원 벡터이고, $\Delta_{S^c}$는 나머지 성분들로 이루어진 벡터다. 이때 설명변수 행렬 $X$가 어떤 $(\kappa,\alpha)$에 대하여

$$\frac{1}{n}|X\Delta|_2^2 \ge \kappa|\Delta|_2^2,\qquad \forall \Delta\in C_\alpha(S) \tag{16.13}$$

를 만족한다고 하자. 그러면 $X$는 $S$ 위에서 모수 $(\kappa,\alpha)$에 대한 고윳값 제한 조건을 만족한다고 말한다.

이 조건의 의미는, 희소한 방향 (sparse direction) 또는 거의 희소한 방향에서 설명변수 행렬 $X$가 지나치게 납작해지지 않는다는 데 있다. 즉, 특정 방향으로 정보가 소실되어 $\boldsymbol{\beta}$를 구별할 수 없게 되는 상황을 막아준다.

위의 두 조건이 만족되면 추정오차 $|\hat{\boldsymbol{\beta}}^{lasso}-\boldsymbol{\beta}|_2$ 는 빠른 속도로 0에 수렴한다. 즉, 확률수렴 (convergence in probability)의 의미에서 라쏘추정량은 진짜 회귀계수 $\boldsymbol{\beta}$를 잘 근사한다.

### 16.4.2 변수선택의 일치성

라쏘추정량이 단지 계수를 잘 근사하는 것에 그치지 않고, 실제로 0이 아닌 변수들을 정확히 골라내려면 더 강한 조건이 필요하다. 이를 위해 다음 두 조건을 추가로 둔다.

첫째, 서포트에 해당하는 설명변수들만 모은 행렬을 $X_S$라 하자. 그러면 이들의 표본공분산행렬이 아래로 유계여야 한다.   즉, $\frac{1}{n}X_S^TX_S$ 의 가장 작은 고윳값이

$$c_{\min}>0 \tag{16.14}$$

로 유계라고 가정한다. 이는 진짜로 중요한 변수들끼리 심하게 선형종속되지 않아야 함을 뜻한다.

둘째, 상호 일관성 (mutual incoherence) 조건을 둔다. 어떤 $\alpha\in[0,1)$에 대하여

$$\max_{j\in S^c}\left|(X_S^TX_S)^{-1}X_S^TX_j\right|_1\le \alpha \tag{16.15}$$

가 성립한다고 하자. 이 조건은 진짜 변수 집합 $S$에 속하지 않는 변수들이, 진짜 변수들과 지나치게 강한 상관관계를 가져서는 안 된다는 뜻이다. 다시 말해, 잡음 변수 (noise variable)가 진짜 변수의 선형결합처럼 행동하지 않아야 한다.

이 두 조건이 만족되면, 라쏘추정량의 비영 서포트 (estimated support)

$$\hat{S}=\{j\mid \hat{\boldsymbol{\beta}}^{lasso}_j\neq 0,; j=1,2,\dots,p\} \tag{16.16}$$

가 진짜 서포트 $S$와 같아지는 확률이 1로 수렴한다. 즉, 라쏘는 변수선택에 대해 일치적 (selection consistent)일 수 있다.

정리하면:
* 고윳값 제한 (restricted eigenvalue) 조건은 주로 추정오차의 제어에 필요하다.
* 최소 고윳값 조건과 상호 일관성 (mutual incoherence) 조건은 변수선택 일치성에 필요하다.
* 따라서 계수를 잘 추정하는 것과 변수를 정확히 선택하는 것은 서로 다른 난이도와 조건을 가진 문제다.

#### 예제: 추정 일치성과 선택 일치성의 차이

라쏘가 $|\hat{\beta}^{lasso}-\beta|_2\to 0$를 만족한다고 해서 자동으로 $\hat{S}=S$가 되는 것은 아니다. 예를 들어 실제로는 0인 변수가 진짜 변수와 매우 강하게 상관되어 있으면, 추정오차는 작더라도 해당 변수가 잘못 선택될 수 있다. 따라서 추정의 정확성 (estimation accuracy)과 선택의 정확성 (selection accuracy)은 구분해서 보아야 한다.


## 16.5 라쏘추정량의 계산 (Computation of Lasso Estimator)

라쏘추정량을 실제로 계산하는 알고리즘 가운데 가장 널리 알려진 방법은 LARS (least angle regression)를 이용하는 알고리즘이다. 이 알고리즘은 전진 선택법 (forward selection)을 개량한 방법으로 이해할 수 있으며, $p>n$인 경우에도 적용 가능하다. 또한 약간의 수정만 가하면 라쏘 추정량의 조절모수 $\lambda$ 변화에 따른 해의 자취 (solution path)를 계산해 준다.

중요한 점은 라쏘 추정량의 해 자취가 $\lambda$의 변화에 따라 조각별 선형 (piecewise linear)의 형태를 가진다는 점이다. 이 성질 덕분에 전체 경로를 효율적으로 추적할 수 있다.

LARS 알고리즘은 최소제곱추정량을 구하는 경우와 같은 차수의 계산 복잡도 ($O(np^2)$) 수준으로 라쏘의 전체 해 경로를 구할 수 있으며, 추가적으로 $\lambda$에 따라 최적해가 생성, 성장, 감소, 소멸하는 현상까지 자세히 보여준다.

이제 알고리즘 설명을 위해 몇 가지 기호를 정의한다.

* 현재 회귀계수 추정값을 $\hat{\boldsymbol{\beta}}$라 하자.
* 현재 예측값을 $\hat{\boldsymbol{\mu}}=X\hat{\boldsymbol{\beta}}$라 하자.
* 현재 잔차벡터를 $\mathbf{r}=\mathbf{y}-\hat{\boldsymbol{\mu}}$라 하자.
* $j$번째 설명변수 벡터를 $\mathbf{x}_j=(x_{1j},x_{2j},\dots,x_{nj})^T$라 하자.
* 현재 잔차벡터 $\mathbf{r}$과 $\mathbf{x}_j$의 상관계수를 $\hat{c}_j$라 하자.
* 전체 상관계수벡터를 $\hat{\mathbf{c}}=(\hat{c}_1,\dots,\hat{c}_p)^T$라 하자.

### 16.5.1 LARS 알고리즘의 기본 단계

**단계 1**  

현재 추정값을 $\hat{\boldsymbol{\beta}}=0$로 초기화한다. 따라서 $\hat{\boldsymbol{\mu}}=0_n,\quad \mathbf{r}=\mathbf{y}$ 가 되고, 현재 활성 설명변수의 집합 $A$는 공집합이다.

**단계 2**

설명변수들 중 현재 잔차 $\mathbf{r}$과 가장 높은 상관관계를 갖는 변수를 선택한다. 즉,

$$j_1=\arg\max_j |\hat{c}_j|$$

로 두고, 이 변수를 활성집합 (active set)에 추가한다.

$$A\leftarrow A\cup\{j_1\}$$

즉, 가장 유망한 변수 하나를 먼저 모형에 넣는다.

**단계 3**

선택된 변수 $j_1$의 계수를 상관계수의 부호 방향으로 조금씩 증가시킨다. 즉, 사전에 정한 작은 상수 $\alpha$에 대하여

$$\hat{\boldsymbol{\beta}}_{j_1}\leftarrow \hat{\boldsymbol{\beta}}_{j_1}+\alpha\operatorname{sign}(\hat{c}_{j_1})$$

의 방향으로 움직인다. 이 과정을 계속하여, 아직 선택되지 않은 변수들 가운데 어떤 변수 하나가 현재 활성 변수들과 동일한 수준의 상관관계를 갖게 될 때까지 진행한다. 즉,

$$\max_{k\in A^c} |\operatorname{corr}(x_k,\mathbf{r}(\alpha))| \ge \max_{j\in A} |\operatorname{corr}(x_j,\mathbf{r}(\alpha))| \tag{16.17}$$

가 처음 성립하는 시점까지 이동한다. 이때 위 조건을 만족시키는 새로운 변수를 $j_2$라 하면,

$$A\leftarrow A\cup\{j_2\}$$

로 업데이트한다.

**단계 4**

이제 집합 $A$에 속한 모든 설명변수에 대하여 회귀계수를 동시에 조정한다. 조정 방향은

$$\delta_A=(X_A^TX_A)^{-1}X_A^T\mathbf{r}$$

로 두고,

$$\hat{\boldsymbol{\beta}}_A \leftarrow \hat{\boldsymbol{\beta}}_A+\alpha\delta_A \tag{16.18}$$

방향으로 움직인다. 여기서 $X_A$는 집합 $A$에 속한 변수들로 만든 설명변수행렬이고, $\hat{\boldsymbol{\beta}}_A$는 $j\in A$인 계수들만 모은 벡터다.

이 과정을 계속하여 또 다른 새로운 변수가 식 (16.17)을 처음 만족시키는 순간을 찾고, 그 변수를 다시 활성집합에 추가하며, 잔차 $\mathbf{r}$도 새롭게 계산한다.

**단계 5**

위 과정을 모든 $p$개의 변수가 선택될 때까지 반복한다.

이 알고리즘의 핵심은 매 순간 가장 관련성이 큰 변수들을 골라 활성집합에 넣고, 활성집합 내부에서는 공동으로 조정하면서 해의 경로를 따라간다는 점이다.

### 16.5.2 라쏘에 맞춘 수정

위 LARS 알고리즘을 그대로 적용하면 LARS 해를 얻는다. 이를 라쏘 회귀추정량 계산에 맞추려면 한 가지 중요한 수정을 추가한다.

**단계 4a**

단계 4에서 계수를 조정하며 이동하는 도중, 어떤 $j\in A$에 대하여 $\hat{\boldsymbol{\beta}}_j=0$이 되면 그 변수를 활성집합 $A$에서 제거하고 절차를 계속한다.

이 수정은 라쏘의 핵심 성질인 희소성 (sparsity)을 반영한다. 즉, 어떤 변수가 한때 선택되었더라도 경로를 따라가며 다시 0이 되면 모형에서 빠질 수 있어야 한다. 이 점이 일반 LARS와 라쏘 경로 알고리즘을 구분하는 핵심이다.

### 16.5.3 계산적 의미

LARS 기반 알고리즘의 장점은 다음과 같다.

* $p>n$ 상황에서도 적용 가능하다.
* $\lambda$ 값 하나마다 별도로 최적화를 반복하지 않고, 전체 해 경로를 한 번에 계산할 수 있다.
* 해 경로가 조각별 선형 (piecewise linear)이므로 계산이 효율적이다.
* 변수의 진입 (entry)과 이탈 (drop-out)을 동시에 추적할 수 있다.

또한 LARS 알고리즘은 라쏘뿐 아니라 forward-stage 회귀추정량 등 다른 변형에도 적용 가능하며, 실제 구현은 R의 `"lars"` 패키지 등에서 제공된다.

#### 예제: 변수의 진입과 이탈

초기에는 가장 상관이 큰 변수 하나만 선택된다. 이후 다른 변수들이 잔차와의 상관이 비슷해지면 차례로 모형에 들어온다. 그러나 라쏘에서는 계수 경로를 따라 이동하는 중 이미 들어온 변수의 계수가 다시 0이 될 수 있으므로, 그 변수는 활성집합에서 제거된다. 따라서 라쏘의 해 경로는 단순한 단조 증가 구조가 아니라, 변수의 진입과 이탈이 모두 가능한 동적 구조를 가진다.


## 16.6 여러가지 축소추정량 (Other Shrinkage Estimators)

라쏘 회귀추정량은 추정 성능을 높이거나, 자료가 가진 고유한 구조를 반영하기 위해 여러 형태로 확장되어 왔다. 여기서는 대표적인 변형 몇 가지를 정리한다.

### 16.6.1 적응라쏘 회귀추정량 (Adaptive Lasso Estimator)

적응라쏘 (adaptive lasso)는 라쏘 회귀추정량의 편의 (bias)를 줄이기 위해 제안된 방법이다. 기본 아이디어는 모든 계수에 동일한 $\ell_1$-벌점을 주는 대신, 계수마다 다른 가중치 (weight)를 부여하는 것이다.

가중치는 보통 초기추정량 $\hat{\beta}^{init}$ 또는 $\hat{\beta}^{lse}$를 이용하여

$$w_j=\frac{1}{|\hat{\beta}^{lse}_j|^\nu},\qquad \nu>0,\quad j=1,\dots,p$$

와 같이 둔다. 그리고 벌점함수는 가중 $\ell_1$-노름 (weighted $\ell_1$-norm)

$$\sum_{j=1}^p w_j|\beta_j|$$

을 사용한다. 이 벌점의 의미는:

* 초기추정량이 0에 가까운 계수는 큰 벌점을 받는다.
* 초기추정량이 충분히 큰 계수는 작은 벌점을 받는다.

따라서 중요하지 않아 보이는 변수는 더 강하게 줄이고, 중요한 변수는 덜 줄이게 된다. 이로써 라쏘의 과도한 축소로 인한 편의 (shrinkage bias)를 완화할 수 있다.

적응라쏘는 라쏘의 볼록성 (convexity)을 유지하면서도, 동시에 변수선택의 일치성 (selection consistency)을 보장하는 것으로 알려져 있다.

#### 예제: 적응적 벌점의 효과

어떤 두 변수의 초기추정량이 $|\hat{\beta}^{lse}_1| \gg |\hat{\beta}^{lse}_2|$라고 하자. 그러면

$$w_1 \ll w_2$$

가 되어 두 번째 변수에 더 큰 벌점이 부과된다. 따라서 첫 번째 변수는 유지되고, 두 번째 변수는 더 쉽게 0으로 줄어든다. 이것이 적응라쏘의 핵심 작동원리다.

### 16.6.2 엘라스틱넷 회귀추정량 (Elastic-Net Estimator)

엘라스틱넷 (elastic-net)은 $\ell_1$-노름과 $\ell_2$-노름을 결합한 벌점화 회귀추정량이다. Zou와 Hastie가 제안한 방법으로, 벌점함수는 다음과 같다.

$$\lambda \sum_{j=1}^p \left[(1-\alpha)|\beta_j|+\alpha \beta_j^2\right],\qquad \alpha\in[0,1] \tag{16.19}$$

즉, $\ell_1$-벌점과 $\ell_2$-벌점의 볼록결합 (convex combination)을 사용한다.

라쏘는 상관관계가 높은 유의미한 변수들이 있을 때, 흔히 그중 하나만 선택하는 성질을 가진다. 예를 들어 $p=2$이고 $X_1\approx X_2$인 상황에서

$$Y=\beta_1X_1+\beta_2X_2+\varepsilon$$

와 같은 회귀모형을 생각하면, 라쏘는 흔히 $\hat{\beta}^{lasso}_1=0$ 또는 $\hat{\beta}^{lasso}_2=0$ 가운데 하나를 주는 경향이 있다. 물론 다른 회귀계수는 0이 아닐 수 있다. 즉, 강하게 상관된 변수들을 함께 살리지 못하고 하나만 선택하는 경향이 있다.

이에 비해 엘라스틱넷은 $\ell_2$-벌점을 함께 넣음으로써 상관관계가 있는 유의한 변수들을 함께 선택하는 경향을 가진다. 위 예에서는 $(\beta_1,\beta_2)$를 모두 0이 아닌 값으로 추정하거나, 또는 함께 0으로 보낼 수 있다. 이 성질을 집단화 효과 (grouping effect)라고 이해할 수 있다.

설명변수 두 개인 경우, $(\alpha,\lambda)$가 주어졌을 때 엘라스틱넷 추정계수를 $\hat{\beta}_1(\alpha,\lambda)$, $\hat{\beta}_2(\alpha,\lambda)$라 하면, 어떤 상수 $M>0$에 대하여 다음 부등식이 성립한다.

$$|\hat{\beta}_1(\alpha,\lambda)-\hat{\beta}_2(\alpha,\lambda)| < \frac{\sqrt{n}M}{\alpha\lambda}\sqrt{2(1-r_{12})} \tag{16.20}$$

여기서 $r_{12}$는 두 설명변수의 표본 상관계수 (sample correlation coefficient)다. 따라서 $r_{12}$가 1에 가까울수록 두 회귀계수의 차이는 0에 가까워진다. 즉, 강한 상관을 가진 변수일수록 거의 같은 계수값을 갖게 된다.

마지막으로 엘라스틱넷의 계산은 반응변수 벡터와 설명변수 행렬을 적절히 변형하여 라쏘 회귀추정 문제로 바꾸어 풀 수 있다.

#### 예제: 상관된 변수에 대한 라쏘와 엘라스틱넷의 차이

$X_1$과 $X_2$가 거의 같은 정보를 담고 있을 때, 라쏘는 둘 중 하나만 남기고 다른 하나를 0으로 만드는 경향이 강하다. 반면 엘라스틱넷은 두 변수에 유사한 계수를 부여하면서 함께 선택하는 경향이 강하다. 따라서 강한 다중공선성 (multicollinearity)이 존재하는 경우 엘라스틱넷이 더 자연스러운 해를 줄 수 있다.

### 16.6.3 그룹라쏘 회귀추정량 (Group Lasso Regression Estimator)

그룹라쏘 (group lasso)는 변수들이 사전에 정해진 그룹구조 (group structure)를 가진다는 정보를 반영한 방법이다. Yuan과 Lin이 제안하였으며, 개별 변수를 선택하는 대신 변수의 그룹을 선택하는 데 초점을 둔다.

이러한 그룹정보가 존재하는 대표적 예는 다음과 같다.

* 하나의 범주형 변수 (categorical variable)를 더미변수 (dummy variable)들로 변환한 경우
* 마이크로어레이 (microarray) 자료에서 특정 생물학적 경로 (biological pathway)에 속하는 유전자들을 하나의 그룹으로 보는 경우

이 경우에는 변수를 하나씩 고르는 것보다, 변수들의 그룹 단위로 선택하는 것이 더 의미가 있다.

설명변수들이 $G$개의 그룹으로 나누어지고, $p_g$를 $g$번째 그룹에 속하는 변수 수라 하자. 그러면 총 변수 수는

$$p=\sum_{g=1}^G p_g$$

가 된다. 이때 그룹라쏘의 목적함수는 다음과 같은 벌점함수를 사용한다.

$$\min_{\boldsymbol{\beta}\in \mathbb{R}^p} \sum_{i=1}^n \left(y_i-\sum_{g=1}^G \mathbf{x}_{ig}^T\boldsymbol{\beta}_g\right)^2 + \lambda\sum_{g=1}^G \sqrt{p_g}|\boldsymbol{\beta}_g|_2 \tag{16.21}$$

* $\mathbf{x}_{ig}=(x_{i1},\dots,x_{ip_g})^T$는 $i$번째 관측치의 $g$번째 그룹 변수 벡터
* $\boldsymbol{\beta}_g=(\beta_{g1},\dots,\beta_{gp_g})^T$는 $g$번째 그룹의 회귀모수 벡터다.

즉, 그룹별로 $\ell_2$-노름을 계산하고, 그룹들 전체에 대해서는 $\ell_1$-형식의 합을 취하는 구조다. 이 때문에 그룹 내 변수들은 함께 살아남거나 함께 제거되는 경향을 가진다.

#### 예제: 범주형 변수의 더미코딩

하나의 범주형 변수를 여러 개의 더미변수로 바꾸면, 이 변수들은 사실상 하나의 원래 변수에서 파생된 것이다. 이들을 각각 따로 선택하면 해석이 어색해질 수 있다. 그룹라쏘는 이 더미변수 집합 전체를 하나의 그룹으로 취급하여, 해당 범주형 변수 전체를 선택하거나 제거하도록 만든다.

### 16.6.4 퓨즈드라쏘 회귀추정량 (Fused Lasso Regression Estimator)

퓨즈드라쏘 (fused lasso)는 설명변수들 사이에 순서 (ordering)가 존재하고, 인접한 변수들 사이에 강한 상관관계가 있을 때 사용되는 방법이다. Tibshirani 등이 제안하였다.

대표적인 예로, 단백질의 $m/z$ 변수를 측정한 SELDI-TOF MS (surface-enhanced laser desorption/ionization time-of-flight mass spectrometry) 자료를 생각할 수 있다. 이 경우 순서화된 인접 $m/z$ 변수들 사이에 강한 양의 상관성이 존재하며, 인접한 변수들의 회귀계수도 비슷할 것으로 기대된다.

이때 fused 라쏘의 벌점함수는 다음과 같이 정의된다.

$$\lambda\left((1-\alpha)\sum_{j=1}^p |\beta_j| + \alpha\sum_{j=2}^p |\beta_j-\beta_{j-1}|\right),\qquad \alpha\in[0,1]$$

이 벌점은 두 부분으로 이루어진다.

1. 계수 자체의 희소성 (sparsity)을 유도하는 $\sum |\beta_j|$
2. 인접한 계수들 사이의 차이를 작게 만들어 계수의 평활성 (smoothness) 또는 구간별 상수성 (piecewise constancy)을 유도하는 $\sum |\beta_j-\beta_{j-1}|$

따라서 fused 라쏘는 elastic-net과 달리 변수들 사이의 관계를 직접적으로 강제한다. 특히 순서가 있는 설명변수에서 이웃한 계수들이 서로 비슷해지도록 만드는 점이 핵심이다.

#### 예제: 순서가 있는 신호자료

설명변수들이 시간축 또는 위치축을 따라 정렬되어 있다고 하자. 이때 인접 구간에서는 영향력이 비슷할 가능성이 크다. fused 라쏘를 적용하면 개별 계수를 독립적으로 추정하는 대신, 인접 계수들이 같은 값 또는 비슷한 값을 갖도록 유도하여 더 해석 가능한 구조를 얻을 수 있다.


### 16.6.5 정리

라쏘의 여러 확장형은 각기 다른 목적을 가진다.

* 적응라쏘 (adaptive lasso): 계수별 가중 벌점을 통해 편의 (bias)를 줄이고 선택 일치성 (selection consistency)을 개선한다.
* 엘라스틱넷 (elastic-net): 상관된 변수들을 함께 선택하는 성질을 강화한다.
* 그룹라쏘 (group lasso): 변수의 그룹구조를 반영하여 그룹 단위 선택을 수행한다.
* 퓨즈드라쏘 (fused lasso): 순서가 있는 변수들에 대해 인접 계수의 유사성을 강제한다.

즉, 벌점화 회귀분석은 단일한 방법이 아니라, 자료의 구조와 분석 목적에 맞추어 벌점함수 (penalty function)를 설계하는 하나의 큰 틀로 이해해야 한다.

