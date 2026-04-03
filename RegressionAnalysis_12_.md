# Chapter 12 분산분석의 응용 (Applications of ANOVA)

분산분석(analysis of variance, **ANOVA**)은 회귀분석(regression analysis)과 함께 널리 사용되는 통계적 분석 방법이다. 분산분석은 관심 있는 반응변수(response variable)의 총변동(total variation)을 설명할 수 있는 요인과 설명할 수 없는 요인으로 분해(decomposition)하고, 어떤 변동의 원인(source of variation)이 총변동을 설명하는 데 중요한 역할을 하는지를 분석하는 방법이다.

일반적인 회귀분석에서는 설명변수(explanatory variable)로 양적 변수와 질적 변수 모두를 포함할 수 있다. 그러나 분산분석 모형에서는 설명변수가 주로 질적 변수(categorical variable)로 구성된다. 따라서 회귀모형이 관측 연구(observational study)에서 수집된 데이터 분석에 많이 활용되는 것과 달리, 분산분석 모형은 실험 연구(experimental study)에서 수집된 데이터의 분석에 자주 활용된다.  
분산분석 모형과 회귀모형은 서로 다른 모형으로 생각될 수 있지만, 분산분석에서 사용되는 분산분석표(analysis of variance table)는 회귀분석 모형의 결과로도 작성될 수 있다. 넓은 의미에서 분산분석은 회귀분석의 특수한 경우로 해석될 수 있다.

## 12.1 일원배치법 (One-Way ANOVA)

분산분석 모형에서는 질적인 설명변수를 인자(factor)라고 한다. 하나의 인자 $A$가 있고, 이 인자가 $l$개의 수준(level) $A_1, A_2, \dots, A_l$을 가지고 있다고 하자.

이 인자의 각 수준이 주는 효과(effect) $a_i$는 **고정효과(fixed effect)** 라고 가정한다. 각 수준에서 $m$개의 관측값이 있다고 하면 데이터의 배열은 다음과 같다.

|   | $A_1$    | $A_2$    | ... | $A_l$    |
| - | -------- | -------- | --- | -------- |
|   | $y_{11}$ | $y_{21}$ | ... | $y_{l1}$ |
|   | $y_{12}$ | $y_{22}$ | ... | $y_{l2}$ |
|   | ...      | ...      | ... | ...      |
|   | $y_{1m}$ | $y_{2m}$ | ... | $y_{lm}$ |

각 수준에서의 합과 평균은 다음과 같다.

$$T_i = \sum_{j=1}^{m} y_{ij}, \qquad \bar{y}_i = \frac{T_i}{m}$$

전체 데이터의 합과 평균은 다음과 같다.

$$T = \sum_{i=1}^{l} T_i = \sum_{i=1}^{l}\sum_{j=1}^{m} y_{ij} \\
\bar{y} = \frac{1}{lm}\sum_{i=1}^{l}\sum_{j=1}^{m} y_{ij}$$

이와 같은 데이터를 얻는 실험계획을 **일원배치법(one-factor design of experiment)** 이라고 한다.

**모형 (Model)**  
하나의 인자를 가진 분산분석 모형은 다음과 같이 표현된다. 

$$y_{ij} = \mu_i + \epsilon_{ij} \quad \text{또는} \quad y_{ij} = \mu + \alpha_i + \epsilon_{ij}$$

* $i = 1,2,\dots,l$
* $j = 1,2,\dots,m$
* $\epsilon_{ij} \sim N(0,\sigma^2)$
* $\mu_i$: $i$번째 수준의 평균
* $\mu$: 전체 평균
* $\alpha_i$: 인자의 효과

전체 평균은 $\mu = \frac{1}{l}\sum_{i=1}^{l}\mu_i$ 이고 인자의 효과는 $\alpha_i = \mu_i - \mu$  
이때 다음 제약조건이 성립한다. $\sum_{i=1}^{l}\alpha_i = 0$

**일원배치법의 분산분석표 (ANOVA Table)**  
| 요인    | 제곱합   | 자유도      | 평균제곱             | F             |
| ----- | ----- | -------- | ---------------- | ------------- |
| A(요인) | $S_A$ | $l-1$    | $V_A=S_A/(l-1)$  | $F_0=V_A/V_E$ |
| 오차    | $S_E$ | $l(m-1)$ | $V_E=S_E/l(m-1)$ |               |
| 전체    | $S_T$ | $lm-1$   |                  |               |

**선형회귀모형 표현 (Linear Regression Representation)**  
분산분석 모형은 선형회귀모형(linear regression model)으로 다음과 같이 표현할 수 있다.

$$y_{ij} = \mu + \alpha_1 x_1 + \alpha_2 x_2 + \cdots + \alpha_l x_l + \epsilon_{ij}$$

여기서 가변수(dummy variable) $x_k$는 다음과 같이 정의된다.

$$
x_k =
\begin{cases}
1 & k=i \\
0 & k\neq i
\end{cases}
$$

행렬 형태로 표현하면 

$$\mathbf{y} = \mathbf{X}\beta + \boldsymbol{\varepsilon}$$

정규방정식(normal equations)은 

$$\mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^T \mathbf{y} \\

\mathbf{y} = \begin{pmatrix}y_{11} \\ y_{12} \\ \vdots \\ y_{1m} \\ y_{21} \\ y_{22} \\ \vdots \\ y_{2m} \\ \vdots \\ y_{l1} \\ y_{l2} \\ \vdots \\ y_{lm}\end{pmatrix},\qquad
\mathbf{X} = \begin{pmatrix}1 & 1 & 0 & \cdots & 0 \\ 1 & 1 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & 1 & 0 & \cdots & 0 \\ 1 & 0 & 1 & \cdots & 0 \\ 1 & 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & 0 & 0 & \cdots & 1 \\ 1 & 0 & 0 & \cdots & 1 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & 0 & 0 & \cdots & 1\end{pmatrix},\qquad
\boldsymbol{\beta} = \begin{pmatrix}\mu \\ \alpha_1 \\ \alpha_2 \\ \vdots \\ \alpha_l\end{pmatrix},\qquad
\boldsymbol{\varepsilon} = \begin{pmatrix}\epsilon_{11} \\ \epsilon_{12} \\ \vdots \\ \epsilon_{1m} \\ \epsilon_{21} \\ \epsilon_{22} \\ \vdots \\ \epsilon_{2m} \\ \vdots \\ \epsilon_{l1} \\ \epsilon_{l2} \\ \vdots \\ \epsilon_{lm}\end{pmatrix}
$$
  
이를 풀면 다음의 연립방정식을 얻는다.

$$ lm\mu + m\sum_{i=1}^{l}\alpha_i = T \\ m\mu + m\alpha_i = T_i$$

$X^T X$와 $X^T y$는 다음과 같다.

$$X^T X = \begin{pmatrix} lm & m & m & \cdots & m \\ m & m & 0 & \cdots & 0 \\ m & 0 & m & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ m & 0 & 0 & \cdots & m\end{pmatrix},\qquad
X^T y = \begin{pmatrix} lm\bar{y} \\ m\bar{y}_1 \\ m\bar{y}_2 \\ \vdots \\ m\bar{y}_l\end{pmatrix} = \begin{pmatrix} T \\ T_1 \\ T_2 \\ \vdots \\ T_l\end{pmatrix}$$

따라서 정규방정식을 풀어서 써보면, $lm\mu + m\sum_{i=1}^{l}\alpha_i = T$ 이고 $m\mu + m\alpha_i = T_i$ 이다.  
$X^T X$는 두번째 열에서 마지막 열까지 모두 더하여 첫번째 열이 되므로 정칙(nonsingular)행렬이 아니다. 따라서, 정규방정식의 해는 무수히 많다.  
이는 11장에서 언급했듯이 $l$개의 수준을 갖는 변수에 대해 $l$개의 가변수를 생성해서 생기는 문제다. 분산분석모형에서는 $l-1$개의 가변수를 생성하는 대신에, 제약조건에 의해 유일한 해를 구할 수 있게 된다.  
이 해를 $\hat\mu, \hat\alpha_i$라고 하면, 제약조건 $\sum_{i=1}^{l}\alpha_i = 0$ 을 이용하면 추정값은 다음과 같다.

$$\hat{\mu} = \bar{y} \\ \hat{\alpha}_i = \bar{y}_i - \bar{y}$$

**변동의 분해 (Decomposition of Variation)**  
회귀분석에 의한 총변동, 회귀변동과 잔차변동을 구해보면 재밌는 결과를 얻을 수 있다.  

총변동(total variation)은 앞의 일원배치법의 분산분석표의 총제곱합 $S_T$와 같다.  

$$SST = \sum_{i=1}^{l}\sum_{j=1}^{m}(y_{ij}-\bar{y})^2$$

회귀변동 (regression variation)은 분산분석표의 $A$의 변동 $S_A$와 동일하다

$$SSR = \hat{\boldsymbol{\beta}}^T X^T \mathbf{y} - n(\bar{y})^2 \\
= lm(\bar{y})^2 + m\sum_{i=1}^{l} \hat{\alpha}_i \bar{y}_i - lm(\bar{y})^2 \\
= m\sum_{i=1}^{l}(\bar{y}_i-\bar{y})^2$$

마찬가지로 $SSE = S_E$도 확인할 수 있다. 잔차변동(error variation)은 $SSE = SST - SSA$

$$SSA = m\sum_{i=1}^{l}(\bar{y}_i-\bar{y})^2$$

>즉, 분산분석모형으로 작성한 분산분석표는 회귀모형으로 계산한 결과와 일치한다.

**가설검정 (Hypothesis Testing)**  
분산분석에서의 가설은 다음과 같다. ($A$의 변동이 유의하지 않다 vs 유의하다 검정과 동일하다)

$$
H_0 : \alpha_1 = \alpha_2 = \cdots = \alpha_l = 0 \\
H_1 : \text{적어도 하나의 } \alpha_i \neq 0
$$

검정통계량은 $F = \frac{V_A}{V_E} = \frac{SSR}{SSE} \cdot \frac{df_E}{df_A} = \frac{SSR/(l-1)}{SSE/(l(m-1))} = \frac{MSR}{MSE}$  
임계값 $F_\alpha(l-1,\ l(m-1))$ 보다 크면 귀무가설을 기각한다.

**각 수준에서 반복수가 다른 경우 (Unequal Replication)**  
각 수준에서의 반복수가 서로 다를 경우 $m_i$개의 관측값이 있다고 하자.  
각 수준의 마지막 측정값 표기가 $y_{1m1}, y_{2m2}, \dots, y_{lm_l}$로 바뀐다.

이때 평균은 

$$\mu = \frac{\sum_{i=1}^{l} m_i \mu_i}{\sum_{i=1}^{l} m_i}$$

이며 제약조건은 $\sum_{i=1}^{l} m_i \alpha_i = 0$

**반복수가 다른 경우의 분산분석표**
| 요인    | 제곱합   | 자유도      | 평균제곱             | F             |
| ----- | ----- | -------- | ---------------- | ------------- |
| A(요인) | $S_A = \sum_{i=1}^{l} m_i (\bar{y}_i-\bar{y})^2$ | $l-1$    | $V_A=S_A/(l-1)$  | $F_0=V_A/V_E$ |
| 오차    | $S_E = S_T - S_A$ | $\sum_{i=1}^{l} (m_i-1)$ | $V_E=S_E/\sum_{i=1}^{l} (m_i-1)$ |               |
| 전체    | $S_T = \sum_{i=1}^{l} \sum_{j=1}^{m_i} (y_{ij}-\bar{y})^2$ | $\sum_{i=1}^{l} m_i - 1$   |                  |               |

$$
X^T X = \begin{pmatrix} \sum_{i=1}^{l} m_i & m_1 & m_2 & \cdots & m_l \\ m_1 & m_1 & 0 & \cdots & 0 \\ m_2 & 0 & m_2 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ m_l & 0 & 0 & \cdots & m_l\end{pmatrix},\qquad
X^T y = \begin{pmatrix} \sum_{i=1}^{l} m_i \bar{y}_i \\ m_1 \bar{y}_1 \\ m_2 \bar{y}_2 \\ \vdots \\ m_l \bar{y}_l\end{pmatrix}
$$

정규방정식을 풀어 써 보면

$$\mu\sum_{i=1}^{l} m_i + \sum_{i=1}^{l} m_i \alpha_i = \sum_{i=1}^{l} m_i \bar{y}_i = T\\
m_i (\mu + \alpha_i) = m_i \bar{y}_i = T_i$$

여기에 가정 $\sum_{i=1}^{l} m_i \alpha_i = 0$ 을 이용하면 다음과 같은 추정값을 얻는다.

$$\hat{\mu} = \bar{y} = \frac{1}{\sum_{i=1}^{l} m_i} \sum_{i=1}^{l} m_i \bar{y}_i = \frac{T}{\sum_{i=1}^{l} m_i} = \bar{y} \\
\hat{\alpha}_i = \bar{y}_i - \bar{y}$$

즉, 반복수가 동일한 경우와 동일한 해의 형태를 갖는다!  
총제곱합($S_R$), 회귀제곱합($S_A$), 잔차제곱합($S_E$)도 동일한 형태를 갖는다.

총변동은

$$
SST = \sum_{i=1}^{l}\sum_{j=1}^{m_i}(y_{ij}-\bar{y})^2
$$

요인변동은

$$
SSA = \sum_{i=1}^{l} m_i (\bar{y}_i-\bar{y})^2
$$

잔차변동은 $SSE = SST - SSA$

#### 예제 12.1 (Example)

네 가지 다이어트 방법 $A_1, A_2, A_3, A_4$을 비교하는 실험을 수행하였다. 총 10명을 대상으로 실험을 수행하였다.

|   | A1 | A2 | A3 | A4 |
| - | -- | -- | -- | -- |
|   | 12 | 14 | 19 | 24 |
|   | 18 | 12 | 17 | 30 |
|   |    | 13 | 21 |    |

평균은 다음과 같다.

$$
\bar{y}_1=15,\quad
\bar{y}_2=13,\quad
\bar{y}_3=19,\quad
\bar{y}_4=27
$$

전체 평균은 $\bar{y}=18$  
총제곱합은 $SST = 304$  
요인제곱합은 $SSA = 258$  
잔차제곱합은 $SSE = 46$  

분산분석표는 다음과 같다.

| 요인 | 제곱합 | 자유도 | 평균제곱 | F     |
| -- | --- | --- | ---- | ----- |
| A  | 258 | 3   | 86   | 11.21 |
| 오차 | 46  | 6   | 7.67 |       |
| 전체 | 304 | 9   |      |       |

유의수준 $0.05$에서 $F_{0.05}(3,6)=4.76$ 이므로 $F_0=11.21>4.76$  
따라서 귀무가설을 기각한다. 즉, 다이어트 방법에 따라 체중 감소량에 유의한 차이가 있다고 결론 내린다.

TODO: FIXME: 관련 개념 찾아서 없으면 추가하기. 시험에 나왔음
``` 
**수준조합(level combination)**은 이원배치(two-way layout)에서 매우 기본적인 개념이다. 정의를 명확히 정리하면 다음과 같다.

## 1. 수준(level)의 의미
모형:

$$
y_{ij} = \mu + \alpha_i + \beta_j + \epsilon_{ij}
$$

- $\alpha_i$: 첫 번째 요인(factor A)의 **$i$번째 수준(level)** 효과  
- $\beta_j$: 두 번째 요인(factor B)의 **$j$번째 수준(level)** 효과

즉,

- factor A: $(A_1, A_2, \dots, A_\ell)$  
- factor B: $(B_1, B_2, \dots, B_m)$

## 2. 수준조합(level combination)

**수준조합이란:**

> 두 요인의 특정 수준을 하나씩 선택하여 만든 조합

즉,

$$
(A_i, B_j)
$$

이 하나의 수준조합이다.

## 3. 예: $(A_1, B_1)$ 의미

$$
\mu(A_1, B_1) = \mu + \alpha_1 + \beta_1
$$

이는

> factor A의 1번째 수준 + factor B의 1번째 수준이 동시에 적용된 경우의 평균

을 의미한다.

## 4. 직관적 예시

예를 들어:

- A: 비료 종류  
    $\rightarrow$ $A_1$: 비료1, $A_2$: 비료2
- B: 물의 양  
    $\rightarrow$ $B_1$: 적게, $B_2$: 많이

그러면 수준조합은 다음과 같다.

| 수준조합      | 의미         |
| ------------- | ------------ |
| $(A_1, B_1)$ | 비료1 + 물 적게 |
| $(A_1, B_2)$ | 비료1 + 물 많이 |
| $(A_2, B_1)$ | 비료2 + 물 적게 |
| $(A_2, B_2)$ | 비료2 + 물 많이 |

## 5. 수학적 의미 (중요)

각 수준조합은 하나의 평균을 가진다.

$$
E[y_{ij}] = \mu + \alpha_i + \beta_j
$$

> **수준조합은 이원배치에서 하나의 셀(cell)이며, 해당 셀의 평균이 $\mu + \alpha_i + \beta_j$이다.**

```

## 12.2 반복이 없는 이원배치법 (Two-Way ANOVA without Replication)

앞 절에서는 인자가 하나인 경우에 대하여 회귀분석 방법에 의한 분산분석표를 작성하는 방법을 살펴보았다. 이제 두 개의 인자 $A$, $B$가 있는 **이원배치법(two-factor design of experiment)**에 대하여 생각해 보자.

인자 $A$의 $i$수준 $(A_i)$이 주는 효과를 $\alpha_i$, 인자 $B$의 $j$수준 $(B_j)$이 주는 효과를 $\beta_j$라 하자. 여기서 $\alpha_i$와 $\beta_j$는 **고정효과(fixed effect)** 라고 가정한다. 또한 $A_i$와 $B_j$의 조건에서 측정값에 반복이 없다고 하자.

이때 분산분석 모형(ANOVA model)은 다음과 같이 표현된다.

$$y_{ij} = \mu + \alpha_i + \beta_j + \epsilon_{ij},\qquad \epsilon_{ij} \sim N(0,\sigma^2)\\
i = 1,2,\dots,l, \qquad j = 1,2,\dots,m$$

오차항 $\epsilon_{ij}$는 서로 **독립(independent)** 이라고 가정한다.

또한 $\alpha_i$와 $\beta_j$에 대해서는 다음의 제약조건(constraint)이 널리 사용된다.

$$\sum_{i=1}^{l} \alpha_i = 0,\qquad \sum_{j=1}^{m} \beta_j = 0$$

### 선형회귀모형 표현 (Linear Regression Representation)

모형을 **선형회귀모형(linear regression model)** 으로 표현하면 다음과 같다.

$$y_{ij} = \mu + \sum_{i=1}^{l} \alpha_i x_i + \sum_{j=1}^{m} \beta_j x_{l+j} + \epsilon_{ij}$$

여기서 **가변수(dummy variable)** 는 다음과 같이 정의된다.

$$x_k =
\begin{cases}
1 & k=i \\
0 & k\neq i
\end{cases}
\qquad (k=1,2,\dots,l) \\
x_{l+v} =
\begin{cases}
1 & v=j \\
0 & v\neq j
\end{cases}
\qquad (v=1,2,\dots,m)$$

데이터 $y_{ij}$를 이원배치법의 배열표로 나타내면 다음과 같다.

|       | $A_1$          | $A_2$          | … | $A_l$          | 합     | 평균             |
| ----- | -------------- | -------------- | - | -------------- | ----- | -------------- |
| $B_1$ | $y_{11}$       | $y_{21}$       | … | $y_{l1}$       | $T_1$ | $\bar{y}_{.1}$ |
| $B_2$ | $y_{12}$       | $y_{22}$       | … | $y_{l2}$       | $T_2$ | $\bar{y}_{.2}$ |
| …     | …              | …              | … | …              | …     | …              |
| $B_m$ | $y_{1m}$       | $y_{2m}$       | … | $y_{lm}$       | $T_m$ | $\bar{y}_{.m}$ |
| 합     | $T_1$          | $T_2$          | … | $T_l$          | $T$   |                |
| 평균    | $\bar{y}_{1.}$ | $\bar{y}_{2.}$ | … | $\bar{y}_{l.}$ |       | $\bar{y}$      |

$$T = \sum_{i=1}^{l}\sum_{j=1}^{m} y_{ij}, \quad \bar{y} = \frac{1}{lm}\sum_{i=1}^{l}\sum_{j=1}^{m} y_{ij}$$

### 분산분석표 (ANOVA Table)

이원배치법에서 반복이 없는 경우의 분산분석표는 다음과 같다.

| 요인    | 제곱합                                                    | 자유도          | 평균제곱                           | $F_0$     |
| ----- | ------------------------------------------------------ | ------------ | ------------------------------ | --------- |
| A(인자) | $S_A = m\sum_{i=1}^{l}(\bar{y}_{i.}-\bar{y})^2$        | $l-1$        | $V_A = \frac{S_A}{l-1}$        | $V_A/V_E$ |
| B(인자) | $S_B = l\sum_{j=1}^{m}(\bar{y}_{.j}-\bar{y})^2$        | $m-1$        | $V_B = \frac{S_B}{m-1}$        | $V_B/V_E$ |
| E(잔차) | $S_E = S_T - S_A - S_B$                                | $(l-1)(m-1)$ | $V_E = \frac{S_E}{(l-1)(m-1)}$ |           |
| T(계)  | $S_T = \sum_{i=1}^{l}\sum_{j=1}^{m}(y_{ij}-\bar{y})^2$ | $lm-1$       |                                |           |

### 회귀분석 관점에서의 제곱합 계산

분산분석표의 제곱합을 **회귀분석(regression analysis)** 의 입장에서 구할 수 있다. 예를 들어 $l=2$, $m=3$이라고 하면 모형은 행렬 형태로 다음과 같이 표현된다.

$$y = X\beta + \epsilon \\ \beta = (\mu, \alpha_1, \alpha_2, \beta_1, \beta_2, \beta_3)^T \\
\begin{pmatrix} y_{11} \\ y_{12} \\ y_{13} \\ y_{21} \\ y_{22} \\ y_{23} \end{pmatrix} =
\begin{pmatrix} 1 & 1 & 0 & 1 & 0 & 0 \\ 1 & 1 & 0 & 0 & 1 & 0 \\ 1 & 1 & 0 & 0 & 0 & 1 \\ 1 & 0 & 1 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 & 1 & 0 \\ 1 & 0 & 1 & 0 & 0 & 1 \end{pmatrix}
\begin{pmatrix} \mu \\ \alpha_1 \\ \alpha_2 \\ \beta_1 \\ \beta_2 \\ \beta_3 \end{pmatrix} +
\begin{pmatrix} \epsilon_{11} \\ \epsilon_{12} \\ \epsilon_{13} \\ \epsilon_{21} \\ \epsilon_{22} \\ \epsilon_{23} \end{pmatrix}
$$

정규방정식(normal equations)은 $X^T X \beta = X^T y$, 이를 풀어 쓰면 다음과 같은 연립방정식을 얻는다.

$$6\mu + 3(\alpha_1+\alpha_2) + 2(\beta_1+\beta_2+\beta_3) = T\\
3\mu + 3\alpha_1 + (\beta_1+\beta_2+\beta_3) = T_1\\
3\mu + 3\alpha_2 + (\beta_1+\beta_2+\beta_3) = T_2\\
2\mu + (\alpha_1+\alpha_2) + 2\beta_1 = T_{.1}\\
2\mu + (\alpha_1+\alpha_2) + 2\beta_2 = T_{.2}\\
2\mu + (\alpha_1+\alpha_2) + 2\beta_3 = T_{.3}$$

여기서 제약조건 $\sum_{i=1}^{l}\alpha_i = 0,\quad \sum_{j=1}^{m}\beta_j = 0$ 을 사용하면 다음의 추정값을 얻는다.

$$\hat{\mu} = \bar{y}\\
\hat{\alpha}_i = \bar{y}_{i.}-\bar{y}\\
\hat{\beta}_j = \bar{y}_{.j}-\bar{y}$$

또한 위 제약조건(가정)하에서는 언제나 성립한다. 즉,

$$\sum_{i=1}^{l}\hat{\alpha}_i=0,\qquad \sum_{j=1}^{m}\hat{\beta}_j=0$$

### 회귀변동 (Regression Sum of Squares)

회귀변동(regression variation)은 다음과 같이 계산된다.

$$SSR = \hat{\boldsymbol{\beta}}^T X^T \mathbf{y} - n(\bar{y})^2\\
= lm(\bar{y})^2 + m\sum_{i=1}^{l} \hat{\alpha}_i \bar{y}_{i.} + l\sum_{j=1}^{m} \hat{\beta}_j \bar{y}_{.j} - lm(\bar{y})^2\\
= m\sum_{i=1}^{l}(\bar{y}_{i.}-\bar{y})^2 + l\sum_{j=1}^{m}(\bar{y}_{.j}-\bar{y})^2 = S_A + S_B$$

따라서 $SSR = S_A + S_B$ 임을 알 수 있다.

### 축소모형 (Reduced Model)

인자 $A$의 변동 $S_A$를 구하기 위해서는 $B$의 효과가 없다고 가정한 **축소모형(reduced model)**

$$y_{ij}=\mu+\sum_{i=1}^{l}\alpha_i x_i+\epsilon_{ij}$$

을 고려하면 된다. 마찬가지로 $B$의 변동 $S_B$를 구하기 위해서는

$$y_{ij}=\mu+\sum_{j=1}^{m}\beta_j x_{l+j}+\epsilon_{ij}$$

을 고려하면 된다.

총변동(total variation)은

$$SST = \mathbf{y}^T \mathbf{y} - n(\bar{y})^2 = \sum_{i=1}^{l}\sum_{j=1}^{m}y_{ij}^2 - lm(\bar{y})^2 = \sum_{i=1}^{l}\sum_{j=1}^{m}(y_{ij}-\bar{y})^2 = S_T$$

잔차변동(error variation)은 $SSE = SST - SSR = S_E$  
따라서 이원배치법의 분산분석표도 회귀분석의 입장에서 계산할 수 있다.


## 12.3 반복이 있는 이원배치법 (Two-Way ANOVA with Replication)

인자가 둘인 **이원배치법(two-factor design of experiment)** 에서는 인자 $A$와 $B$의 수준 조합에서 여러 번 반복하여 측정값을 얻는 경우가 흔히 있다. 이 반복수를 $r$이라 하자. 반복이 있는 이원배치법에서는 두 인자 $A$, $B$ 간의 **교호작용(interaction)** 의 효과를 측정할 수 있으므로, 분산분석의 모형은 일반적으로 다음과 같이 표현된다.

$$y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}\\
\epsilon_{ijk} \sim N(0, \sigma^2),\qquad
i=1,2,\cdots,l,\quad j=1,2,\cdots,m,\quad k=1,2,\cdots,r$$

여기서 $(\alpha\beta)_{ij}$는 $A_i$ 수준과 $B_j$ 수준에서 발생되는 두 인자 $A$, $B$ 간의 교호작용의 효과이다.

데이터의 합과 평균은 다음과 같이 둔다.

$$T_{i..} = \sum_{j=1}^{m}\sum_{k=1}^{r} y_{ijk},\qquad
T_{.j.} = \sum_{i=1}^{l}\sum_{k=1}^{r} y_{ijk},\qquad
T_{ij.} = \sum_{k=1}^{r} y_{ijk}\\
\bar y_{i..} = \frac{T_{i..}}{mr},\qquad
\bar y_{.j.} = \frac{T_{.j.}}{lr},\qquad
\bar y_{ij.} = \frac{T_{ij.}}{r}$$

또한 전체 평균은

$$\bar y = \frac{1}{lmr} \sum_{i=1}^{l}\sum_{j=1}^{m}\sum_{k=1}^{r} y_{ijk}$$

모형에서 많이 사용되는 가정은 다음과 같다. $\alpha_i$와 $\beta_j$는 각각 고정된 효과(fixed effect)라고 간주한다. 그리고 다음의 제약조건을 둔다.

$$\sum_{i=1}^{l} \alpha_i = 0,\qquad
\sum_{j=1}^{m} \beta_j = 0\\
\sum_{i=1}^{l} (\alpha\beta)_{ij} = 0,\qquad j=1,2,\cdots,m\\
\sum_{j=1}^{m} (\alpha\beta)_{ij} = 0,\qquad i=1,2,\cdots,l$$

이 제약조건은 모형의 모수를 유일하게 식별하기 위하여 필요한 조건이다. 즉, 전체 평균 $\mu$, 주효과(main effect) $\alpha_i,\beta_j$, 교호작용 효과 $(\alpha\beta)_{ij}$를 서로 중복되지 않게 분리하기 위한 조건이다.

### 회귀모형으로의 표현 (Regression Representation)

분산분석모형을 가변수(dummy variable)의 사용을 도입하여 회귀분석모형으로 바꾸면 다음과 같이 표현할 수 있다.

$$y_{ijk} = \mu
+ \sum_{i=1}^{l} \alpha_i x_i
+ \sum_{j=1}^{m} \beta_j x_{l+j}
+ \sum_{i=1}^{l} \sum_{j=1}^{m} (\alpha\beta)_{ij} x_i x_{l+j}
+ \epsilon_{ijk}$$

여기서 가변수는

$$x_u =
\begin{cases}
1, & u = i \\
0, & u \ne i
\end{cases}
\qquad u = 1,2,\cdots,l\\
x_{l+u} =
\begin{cases}
1, & u = j \\
0, & u \ne j
\end{cases}
\qquad u = 1,2,\cdots,m$$

이 표현의 의미는 다음과 같다.

* $x_i$는 관측값이 인자 $A$의 $i$번째 수준에서 왔는지를 나타낸다.
* $x_{l+j}$는 관측값이 인자 $B$의 $j$번째 수준에서 왔는지를 나타낸다.
* 곱 $x_i x_{l+j}$는 관측값이 $(A_i,B_j)$ 조합에서 왔는지를 나타낸다.

따라서 모형은 하나의 절편(intercept), $A$에 대한 효과, $B$에 대한 효과, 그리고 $A\times B$ 교호작용 효과를 모두 포함한 선형회귀모형이다.

### 분산분석표 (ANOVA Table)

| 요인                 | 제곱합 | 자유도 | 평균제곱 | $F_0$ |
| ------------------ | ------------------------------------------------------ | ----------- | ------------------------------ | --------------------------- |
| $A$ (인자)           | $S_A = mr\sum_{i=1}^{l}(\bar y_{i..}-\bar y)^2$        | $l-1$        | $V_A = \frac{S_A}{l-1}$        | $V_A/V_E$ |
| $B$ (인자)           | $S_B = lr\sum_{j=1}^{m}(\bar y_{.j.}-\bar y)^2$        | $m-1$        | $V_B = \frac{S_B}{m-1}$        | $V_B/V_E$ |
| $A\times B$ (교호작용) | $S_{A\times B}=r\sum_{i=1}^{l}\sum_{j=1}^{m}(\bar y_{ij.}-\bar y_{i..}-\bar y_{.j.}+\bar y)^2$ | $(l-1)(m-1)$ | $V_{A\times B} = \frac{S_{A\times B}}{(l-1)(m-1)}$ | $V_{A\times B}/V_E$ |
| $E$ (잔차)           | $S_E = S_T - S_A - S_B - S_{A\times B}$                | $lm(r-1)$    | $V_E = \frac{S_E}{lm(r-1)}$    |                              |
| $T$ (계)            | $S_T = \sum_{i=1}^{l}\sum_{j=1}^{m}\sum_{k=1}^{r}(y_{ijk}-\bar y)^2$ | $lmr-1$      |                                |                              |

* $S_A$: 인자 $A$의 수준 차이만으로 설명되는 변동
* $S_B$: 인자 $B$의 수준 차이만으로 설명되는 변동
* $S_{A\times B}$: 두 인자가 함께 작용할 때 추가로 생기는 변동
* $S_E$: 위 세 가지로 설명되지 않는 나머지 변동
* $S_T$: 전체 관측치의 총변동(total variation)

특히 교호작용의 제곱합 $S_{A\times B}$에 들어 있는

$$\bar y_{ij.} - \bar y_{i..} - \bar y_{.j.} + \bar y$$

는 셀 평균(cell mean)이 단순한 주효과의 합으로 설명되지 않고 따로 남는 부분이다. 즉, 교호작용의 크기를 직접 나타내는 항이다.

### 회귀분석에 의한 제곱합 계산

회귀분석에 의하여 위 분산분석표의 제곱합들을 구할 수 있다. 예를 들어 $l=2$, $m=2$, $r=2$이면 회귀모형은 다음과 같이 표현된다.

$$\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

여기서

$$\boldsymbol{\beta} = (\mu, \alpha_1, \alpha_2, \beta_1, \beta_2, (\alpha\beta)_{11}, (\alpha\beta)_{12}, (\alpha\beta)_{21}, (\alpha\beta)_{22})^T$$

정규방정식 $X^TX\boldsymbol{\beta} = X^T\mathbf{y}$를 만들고, 제약조건을 대입하여 $\mu, \alpha_i, \beta_j, (\alpha\beta)_{ij}$를 추정하면 다음의 결과를 얻는다.

$$\hat\mu = \bar y\\
\hat\alpha_i = \bar y_{i..} - \bar y,\qquad
\hat\beta_j = \bar y_{.j.} - \bar y\\
\widehat{(\alpha\beta)}_{ij} = \bar y_{ij.} - \bar y_{i..} - \bar y_{.j.} + \bar y$$

즉,

* 전체 평균의 추정값은 전체 표본평균
* $A$의 효과 추정값은 행 평균과 전체 평균의 차이
* $B$의 효과 추정값은 열 평균과 전체 평균의 차이
* 교호작용 효과 추정값은 셀 평균에서 행 효과와 열 효과를 제거하고 전체 평균을 다시 더한 값

회귀변동(regression sum of squares)은

$$SSR = \boldsymbol{\hat \beta}^T X^T \mathbf{y} - n(\bar y)^2 \\
= [\hat\mu, \hat\alpha_1, \hat\alpha_2, \hat\beta_1, \hat\beta_2, \widehat{(\alpha\beta)}_{11}, \widehat{(\alpha\beta)}_{12}, \widehat{(\alpha\beta)}_{21}, \widehat{(\alpha\beta)}_{22}] \begin{pmatrix} T \\ T_1 \\ T_2 \\ T_{.1} \\ T_{.2} \\ T_{11.} \\ T_{12.} \\ T_{21.} \\ T_{22.} \end{pmatrix} - lmr(\bar y)^2 \\
= lmr(\bar y)^2
+ \sum_{i=1}^{l} \hat\alpha_i T_{i..}
+ \sum_{j=1}^{m} \hat\beta_j T_{.j.}
+ \sum_{i=1}^{l} \sum_{j=1}^{m} \widehat{(\alpha\beta)}_{ij} T_{ij.}
- lmr(\bar y)^2 \\
= mr\sum_{i=1}^{l}(\bar y_{i..}-\bar y)^2
+ lr\sum_{j=1}^{m}(\bar y_{.j.}-\bar y)^2
+ r\sum_{i=1}^{l}\sum_{j=1}^{m}(\bar y_{ij.}-\bar y_{i..}-\bar y_{.j.}+\bar y)^2 \\
= S_A + S_B + S_{A\times B}$$

즉, 전체 회귀변동은 인자 $A$, 인자 $B$, 그리고 교호작용이 설명하는 변동의 합으로 분해된다.  

또한 $S_A, S_B, S_{A\times B}$를 하나하나 개별적으로 회귀분석에 의하여 구할 수 있는 또 하나의 방법은 축소모형(reduced model)을 사용하는 것이다.  
먼저 $\beta_j=0$, $(\alpha\beta)_{ij}=0$으로 놓고 축소모형

$$y_{ijk} = \mu + \sum_{i=1}^{l} \alpha_i x_i + \epsilon_{ijk}$$

를 만들어 회귀변동을 구하면 $S_A$를 얻을 수 있다. 또한 $\alpha_i=0$, $(\alpha\beta)_{ij}=0$으로 놓고

$$y_{ijk} = \mu + \sum_{j=1}^{m} \beta_j x_{l+j} + \epsilon_{ijk}$$

의 축소모형에 대한 회귀변동을 구하여 $S_B$를 얻는다. 같은 방법으로 $\alpha_i=0$, $\beta_j=0$으로 놓고 축소모형

$$y_{ijk} =
\mu + \sum_{i=1}^{l}\sum_{j=1}^{m} (\alpha\beta)_{ij} x_i x_{l+j} + \epsilon_{ijk}$$

를 만들어 회귀변동을 구하면

$$r\sum_{i=1}^{l}\sum_{j=1}^{m}(\bar y_{ij.}-\bar y)^2$$

이 되는데, 여기에서 $S_A$와 $S_B$를 빼면 $S_{A\times B}$를 얻을 수 있다. 총변동(total sum of squares)은 회귀분석에서 구한 공식으로부터

$$SST = \sum_{i=1}^{l}\sum_{j=1}^{m}\sum_{k=1}^{r} y_{ijk}^2 - lmr(\bar y)^2 \\
= \sum_{i=1}^{l}\sum_{j=1}^{m}\sum_{k=1}^{r}(y_{ijk}-\bar y)^2 = S_T$$

잔차변동(error sum of squares)은

$$SSE = SST - SSR = S_T - (S_A + S_B + S_{A\times B}) = S_E$$

TODO: 예제 데이터 형식이 어떻게 되있어야 원하는 수준의 분석을 할 수 있는지 따로 살펴보기
#### 예제 12.2

어떤 공기청정기 제조 회사에서는 새로 개발한 자동차 필터가 오염을 줄이는 데 효과적이라고 주장하였다. 그런데 새로 개발한 필터가 차량 소음에 미치는 영향은 없는지 의문이 생겨 실험을 계획하게 되었다. 새로 개발한 필터가 차량소음에 미치는 영향을 알아보기 위하여, 차량소음에 대한 중요 인자로서 차량의 크기와 필터의 종류를 선택하여 다음과 같이 인자의 수준을 선택하고 차량 소음을 반복 2회 측정하였다. 이때 차량은 같은 회사에서 생산한 것으로 크기만 다른 차량을 선택하였다.

차량 크기와 필터 종류의 수준은 다음과 같다.

| 인자/수준        | 1     | 2     | 3   | 4  |
| ------------ | ----- | ----- | --- | -- |
| $A$ (자동차 크기) | 대형차   | 중형차   | 소형차 | 경차 |
| $B$ (자동차 필터) | 기존형 1 | 기존형 2 | 신제품 |    |

자동차 소음측정 데이터는 다음과 같다.

**표 12.7 자동차 소음측정 데이터**  

|       |    $A_1$ |    $A_2$ |    $A_3$ |    $A_4$ |    합 |   평균 |
| ----- | -------: | -------: | -------: | -------: | ---: | ---: |
| $B_1$ | 2.5, 3.3 | 2.6, 3.0 | 2.9, 3.2 | 3.0, 3.4 | 23.9 | 2.99 |
| $B_2$ | 2.6, 3.2 | 2.5, 3.1 | 3.1, 2.8 | 2.9, 3.5 | 23.7 | 2.96 |
| $B_3$ | 1.9, 2.1 | 2.0, 2.3 | 2.7, 3.2 | 4.2, 4.0 | 22.4 | 2.80 |
| 합     |     15.6 |     15.5 |     17.9 |     21.0 | 70.0 |      |
| 평균    |     2.60 |     2.58 |     2.98 |     3.50 |      | 2.92 |

위 데이터에 회귀분석모형을 가정하면 다음과 같다.

$$y_{ijk} =
\mu + \sum_{i=1}^{4} \alpha_i x_i
+ \sum_{j=1}^{3} \beta_j x_{4+j}
+ \sum_{i=1}^{4} \sum_{j=1}^{3} (\alpha\beta)_{ij} x_i x_{4+j}
+ \epsilon_{ijk}$$

이 모형에 부수되는 가정은

$$\sum_{i=1}^{4} \alpha_i = 0,\qquad
\sum_{j=1}^{3} \beta_j = 0\\
\sum_{i=1}^{4} (\alpha\beta)_{ij} = 0,\quad j=1,2,3\\
\sum_{j=1}^{3} (\alpha\beta)_{ij} = 0,\quad i=1,2,3,4$$

정규방정식의 해를 구하면 다음과 같은 추정값을 얻는다.

$$\hat\mu = \bar y = 2.92\\
\hat\alpha_1 = \bar y_{1..} - \bar y = 2.60 - 2.92 = -0.32\\
\hat\alpha_2 = \bar y_{2..} - \bar y = 2.58 - 2.92 = -0.34\\
\hat\alpha_3 = \bar y_{3..} - \bar y = 2.98 - 2.92 = 0.06\\
\hat\alpha_4 = \bar y_{4..} - \bar y = 3.50 - 2.92 = 0.58\\
\hat\beta_1 = \bar y_{.1.} - \bar y = 2.99 - 2.92 = 0.07\\
\hat\beta_2 = \bar y_{.2.} - \bar y = 2.96 - 2.92 = 0.04\\
\hat\beta_3 = \bar y_{.3.} - \bar y = 2.80 - 2.92 = -0.12$$

그리고 교호작용 효과의 추정값은 다음과 같다.

$$\widehat{(\alpha\beta)}_{11} = \bar y_{11.} - \bar y_{1..} - \bar y_{.1.} + \bar y
= \frac{2.5+3.3}{2} - 2.60 - 2.99 + 2.92 = 0.23\\
\widehat{(\alpha\beta)}_{12} = \frac{2.6+3.2}{2} - 2.60 - 2.96 + 2.92 = 0.26\\
\widehat{(\alpha\beta)}_{13} = \frac{1.9+2.1}{2} - 2.60 - 2.80 + 2.92 = -0.48\\
\widehat{(\alpha\beta)}_{21} = \frac{2.6+3.0}{2} - 2.58 - 2.99 + 2.92 = 0.15\\
\widehat{(\alpha\beta)}_{22} = \frac{2.5+3.1}{2} - 2.58 - 2.96 + 2.92 = 0.18\\
\widehat{(\alpha\beta)}_{23} = \frac{2.0+2.3}{2} - 2.58 - 2.80 + 2.92 = -0.31\\
\widehat{(\alpha\beta)}_{31} = \frac{2.9+3.2}{2} - 2.98 - 2.99 + 2.92 = 0\\
\widehat{(\alpha\beta)}_{32} = \frac{3.1+2.8}{2} - 2.98 - 2.96 + 2.92 = -0.07\\
\widehat{(\alpha\beta)}_{33} = \frac{2.7+3.2}{2} - 2.98 - 2.80 + 2.92 = 0.09\\
\widehat{(\alpha\beta)}_{41} = \frac{3.0+3.4}{2} - 3.50 - 2.99 + 2.92 = -0.37\\
\widehat{(\alpha\beta)}_{42} = \frac{2.9+3.5}{2} - 3.50 - 2.96 + 2.92 = -0.34\\
\widehat{(\alpha\beta)}_{43} = \frac{4.2+4.0}{2} - 3.50 - 2.80 + 2.92 = 0.72$$

이 추정값들을 이용하여 회귀변동을 구하면 $SSR = \hat\beta^T X^T y - n(\bar y)^2$ 즉,

$$SSR = (-0.32)(15.6) + (-0.34)(15.5) + (0.06)(17.9) + (0.58)(21.0)
+ (0.07)(23.9) + (0.04)(23.7) + (-0.12)(22.4)
+ (0.23)(2.5+3.3) + \cdots + (0.72)(4.2+4.0)\\
= 3.33 + 0.16 + 2.58\\
S_A = 3.33,\qquad S_B = 0.16,\qquad S_{A\times B} = 2.58$$

을 얻는다. 여기서 소수점의 반올림 때문에 통계 소프트웨어를 통한 계산값과 정확하게 일치하지 않을 수 있다.

다음으로 총변동을 구하면 $SST = y^T y - n(\bar y)^2$ 즉,

$$S_T = (2.5)^2 + (3.3)^2 + \cdots + (4.2)^2 + (4.0)^2 - 24(2.92)^2 = 7.39$$

이 되고, 따라서 잔차변동은 $SSE = SST - SSR = 7.39 - 6.07 = 1.32$ 이들을 종합하여 분산분석표를 만들면 다음과 같다.

**표 12.8 예제 12.2의 분산분석표**  

| 요인          |  제곱합 | 자유도 | 평균제곱 | $F_0$ | $F_{0.05}$ |
| ----------- | ---: | --: | ---: | ----: | ---------: |
| $A$         | 3.33 |   3 | 1.11 | 10.09 |       3.49 |
| $B$         | 0.16 |   2 | 0.08 |  0.73 |       3.89 |
| $A\times B$ | 2.58 |   6 | 0.43 |  3.91 |       3.00 |
| $E$         | 1.32 |  12 | 0.11 |       |            |
| 계           | 7.39 |  23 |      |       |            |

각 요인의 변동이 유의한지 $F$-검정을 하여 보면 $A$ (자동차 크기)와 $A\times B$ (자동차 크기와 필터 종류)는 유의한 영향을 미치고 있다. 반면 $B$ (필터 종류)의 주효과는 유의하지 않다.

이제 $A$인자와 $B$인자의 어떤 수준 조합에서 가장 작은 차량소음이 발생했는지 살펴보자. $A_i$와 $B_j$ 수준에서 차량소음의 모평균을 $\mu(A_iB_j)$라고 하면,

$$\mu(A_iB_j) = E(y_{ij.}) = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij}$$

가 된다. 그런데 $B$의 변동이 유의하지 않다는 결론이 나왔으므로 $\beta_j=0$, $j=1,2,3$으로 가정하고, 모평균 $\mu(A_iB_j)$의 추정값을 구하면 $\hat\mu(A_iB_j) = \mu + \alpha_i + (\alpha\beta)_{ij}$ 즉,

$$\hat\mu + \hat\alpha_i + \widehat{(\alpha\beta)}_{ij}\\
\bar y + (\bar y_{i..} - \bar y) + (\bar y_{ij.} - \bar y_{i..} - \bar y_{.j.} + \bar y)\\
\bar y_{ij.} - \bar y_{.j.} + \bar y$$

이 모평균의 점추정값(point estimate)을 모든 $i,j$에 대하여 구하면 다음과 같다.

**표 12.9 $\mu(A_iB_j)$의 점추정값**  
|       | $A_1$ | $A_2$ | $A_3$ | $A_4$ |
| ----- | ----: | ----: | ----: | ----: |
| $B_1$ |  2.83 |  2.73 |  2.98 |  3.13 |
| $B_2$ |  2.86 |  2.76 |  2.91 |  3.16 |
| $B_3$ |  2.12 |  2.27 |  3.27 |  4.22 |

따라서 가장 작은 소음은 $A_1B_3$에서 얻어지며, 이 조건이 최적조건(optimal combination)이 된다. 즉, 차량소음이 가장 작았던 경우는 **대형차에 신제품 필터를 장착한 경우**가 된다.

>**보충 설명**
>
>이 절의 핵심은 반복이 없을 때와 달리, 반복이 있으면 셀 내부(cell within variation)의 변동을 따로 관측할 수 있으므로 교호작용을 분리하여 검정할 수 있다는 점이다. 즉,
>
>* 반복이 없으면 $A$, $B$, 잔차만 분리 가능하다.
>* 반복이 있으면 $A$, $B$, $A\times B$, 잔차를 모두 분리할 수 있다.
>
>또한 예제의 결론은 단순히 "필터 종류 $B$는 유의하지 않다"로 끝나지 않는다. 교호작용 $A\times B$가 유의하므로, 필터의 효과는 자동차 크기에 따라 달라진다고 해석해야 한다. 따라서 주효과 $B$만 따로 떼어 "필터는 영향이 없다"고 단정하는 것은 부적절하다. 이 점이 반복이 있는 이원배치법에서 가장 중요한 해석 포인트이다.
>
>마지막 각주에서는 이러한 해석과 관련하여 **효과계층의 원칙(principle of effect heredity)** 을 설명한다. 즉, 교호작용효과와 같은 고차항이 유의한 경우, 그 교호작용을 구성하는 주효과와 같은 저차항을 모형에 포함할 것을 권한다. 관련하여
>
>* 유의한 교호작용과 관련된 모든 저차항을 포함시키는 것을 **강-효과계층의 원칙(strong effect heredity)**
>* 관련된 저차항들 중 적어도 하나를 포함시키는 것을 **약-효과계층의 원칙(weak effect heredity)**
>
>이라 부른다. 위 예제의 경우 주효과 $B$는 모형에서 제외되었기 때문에 약-효과계층의 원칙을 만족하는 모형이라고 할 수 있다.


## 12.4 모수의 재조정법 (Reparameterization)

앞에서 기본적인 몇 가지의 실험계획법에 의하여 얻어진 데이터를 분석하는 방법으로 회귀분석의 적용을 검토하였다. 이때 공통적인 점은 $X^TX$가 **비정칙행렬(singular matrix)** 이 되어 $X^TX$ 행렬의 역행렬(inverse matrix)을 구할 수 없으므로, 최소제곱추정량(least squares estimator)

$$\hat{\beta}=(X^TX)^{-1}X^Ty$$

를 그대로 사용할 수 없었다는 점이다. 이러한 문제를 극복하기 위하여 모수들 사이에 가정을 도입하여 정규방정식(normal equations)을 풀어서 해를 얻었다. 여기서 모수(parameters)란 회귀계수가 되는 $(\mu,\alpha_i,\beta_j)$ 등을 말한다.  

이 모수들을 추정하는 데에 위의 방법을 사용하지 않고, 모수들 간의 가정을 직접 회귀모형에 도입시켜서 **모수의 재조정(reparameterization of parameters)** 을 통하여 $X^TX$ 행렬을 **정칙행렬(non-singular matrix)** 로 만들어 모수들을 추정하는 방법이 있다. 이 재조정법은 앞장에서 가정 $C\beta=m$의 검정에서 간단히 논의된 바 있다. 이제 이 방법을 상세히 살펴본다.

### 12.4.1 일원배치법 (방법 I)

먼저 반복수가 다른 일원배치법(one-way layout with unequal replications)에 대한 회귀모형을 살펴보자. 앞 절의 가정

$$m_1\alpha_1+m_2\alpha_2+\cdots+m_l\alpha_l=0$$

을 모형에 직접 대입하면

$$\alpha_l=-\frac{1}{m_l}(m_1\alpha_1+m_2\alpha_2+\cdots+m_{l-1}\alpha_{l-1})
\tag{12.27}$$

이므로

$$y_{ij}=\mu+\alpha_1x_1+\alpha_2x_2+\cdots+\alpha_lx_l+\epsilon_{ij} \\
=\mu+\alpha_1x_1+\alpha_2x_2+\cdots+\frac{1}{m_l}(-m_1\alpha_1-\cdots-m_{l-1}\alpha_{l-1})x_l+\epsilon_{ij}\\
=\mu+\alpha_1\left(x_1-\frac{m_1}{m_l}x_l\right)+\cdots+\alpha_{l-1}\left(x_{l-1}-\frac{m_{l-1}}{m_l}x_l\right)+\epsilon_{ij}\\
=\mu+\alpha_1w_1+\cdots+\alpha_{l-1}w_{l-1}+\epsilon_{ij}
\tag{12.28}$$

가 되어, 설명변수(explanatory variable)의 수가 하나 줄어들게 된다. 여기서

$$w_i=x_i-\frac{m_i}{m_l}x_l,\qquad i=1,2,\cdots,l-1
\tag{12.29}$$

이 $w_i$의 값은 수준에 따라 다음과 같이 해석된다.

* $A_i$ 수준의 데이터에 대해서는 $x_i=1$, $x_l=0$이므로 $w_i=1$이다.
* $A_l$ 수준의 데이터에 대해서는 $x_i=0$, $x_l=1$이므로 $w_i=-\frac{m_i}{m_l}$
* 그 이외의 수준에 대해서는 $w_i=0$이다.

만약 $A$의 각 수준에서 같은 반복수를 갖는다면

$$m_1=m_2=\cdots=m_l=m
$$이므로 $w_i$는 $-1$, $0$, 또는 $1$의 값을 갖게 된다.

#### 예제

예제 12.1에 대하여 모수 재조정법에 의한 모형을 쓰면 $y_{ij}=\mu+\alpha_1w_1+\alpha_2w_2+\alpha_3w_3+\epsilon_{ij}$ 이다. 여기서
$m_1=m_4=2,\quad m_2=m_3=3$ 이므로

$$w_1=x_1-\frac{2}{2}x_4=x_1-x_4,\quad w_2=x_2-\frac{3}{2}x_4,\quad 
w_3=x_3-\frac{3}{2}x_4$$

선형회귀모형(linear regression model)의 행렬표현(matrix representation)은 다음과 같다.

$$y=X\beta+\epsilon\\
\begin{bmatrix}
12\\ 18\\ 14\\ 12\\ 13\\ 19\\ 17\\ 21\\ 24\\ 30
\end{bmatrix}
= \begin{bmatrix}
1&1&0&0\\
1&1&0&0\\
1&0&1&0\\
1&0&1&0\\
1&0&1&0\\
1&0&0&1\\
1&0&0&1\\
1&0&0&1\\
1&-1&-1.5&-1.5\\
1&-1&-1.5&-1.5
\end{bmatrix}
\begin{bmatrix}
\mu\\
\alpha_1\\
\alpha_2\\
\alpha_3
\end{bmatrix}
+\epsilon$$

여기서 얻어지는 행렬 $X$는 정칙행렬을 이루므로 $(\mu,\alpha_1,\alpha_2,\alpha_3)$의 추정이 공식 $\hat{\beta}=(X^TX)^{-1}X^Ty$ 로부터 직접 얻어진다. 즉,

$$\hat{\beta} =
\begin{bmatrix}
\hat{\mu}\\
\hat{\alpha}_1\\
\hat{\alpha}_2\\
\hat{\alpha}_3
\end{bmatrix}
= (X^TX)^{-1}X^Ty\\
\begin{bmatrix}
10&0&0&0\\
0&4&3&3\\
0&3&7.5&4.5\\
0&3&4.5&7.5
\end{bmatrix}^{-1}
\begin{bmatrix}
12+18+\cdots+30\\
12+18-24-30\\
14+12+13-1.5(24+30)\\
19+17+21-1.5(24+30)
\end{bmatrix}\\
\begin{bmatrix}
0.1&0&0&0\\
0&0.4&-0.1&-0.1\\
0&-0.1&0.2333&-0.1\\
0&-0.1&-0.1&0.2333
\end{bmatrix}
\begin{bmatrix}
180\\
-24\\
-42\\
-24
\end{bmatrix}
= \begin{bmatrix}
18\\
-3\\
-5\\
1
\end{bmatrix}$$

그리고 $\alpha_4$의 추정은 식 (12.27)로부터

$$\hat{\alpha}_4 =-\frac{1}{2}(2\hat{\alpha}_1+3\hat{\alpha}_2+3\hat{\alpha}_3) = -\frac{1}{2}{2(-3)+3(-5)+3(1)} =9$$

가 되어 $(\mu,\alpha_1,\alpha_2,\alpha_3,\alpha_4)$의 추정값이 모두 예제 12.1에서 구한 결과와 동일하다.

### 12.4.2 일원배치법 (방법 II)

앞의 방법 I에서는 가정 $\alpha_l=-\frac{1}{m_l}\sum_{i=1}^{l-1}m_i\alpha_i$ 를 회귀모형에 대입시켜서 변수를 하나 줄이는 방법을 선택하였다. 이 방법 외에 다음과 같은 모수의 재조정법을 사용할 수도 있다. 회귀모형 (12.3)에서 마지막 설명변수 $x_l$을 제거하고, 새로운 모수로 표현하면

$$y_{ij}=\gamma_0+\gamma_1x_1+\gamma_2x_2+\cdots+\gamma_{l-1}x_{l-1}+\epsilon_{ij}
\tag{12.30}$$

가 된다. 
>각주: 마지막 설명변수 $x_l$이 아니라 임의의 수준에 대한 설명변수 하나를 제거하여도 된다. $x_l$을 제거하는 것은 $\alpha_l=0$을 가정한 것과 동일하며, 실제로 이 방법은 범주형 변수(categorical variable)에 대하여 $(\text{수준 수}-1)$개의 가변수만 생성하는 방법과 같다.

이제 새로운 모수 $\gamma_i$와 $\alpha_i$들 간의 관계를 살펴보자. 
$A_l$ 수준의 데이터에 대해서는 원래의 모형 (12.3)이

$$y_{lj}=\mu+\alpha_l+\epsilon_{lj}$$

이나, 새로운 모형 (12.30)은

$$y_{lj}=\gamma_0+\epsilon_{lj}$$

이므로

$$\gamma_0=\mu+\alpha_l
\tag{12.31}$$

의 관계가 성립한다. 다음으로 $A_1$ 수준의 데이터에 대해서는 모형 (12.3)이

$$y_{1j}=\mu+\alpha_1+\epsilon_{1j}$$

이고, 모형 (12.30)은

$$y_{1j}=\gamma_0+\gamma_1+\epsilon_{1j}
=\mu+\alpha_l+\gamma_1+\epsilon_{1j}$$

이므로, 이 두 모형의 비교로부터

$$\gamma_1=\alpha_1-\alpha_l$$

를 얻는다. 일반적으로

$$\gamma_i=\alpha_i-\alpha_l,\qquad i=1,2,\cdots,l-1
\tag{12.32}$$

가 된다. 이는 $i$ 수준과 마지막 수준 $l$과의 처리효과(treatment effect)의 차를 의미한다. 

>각주: R에서는 보통 $\gamma_i=\alpha a_i-\alpha_1,\quad i=2,\cdots,l$ 를 디폴트(default)로 생성하며, 기준이 되는 수준(reference level)은 다시 설정할 수 있다. 다만, 각 수준별 평균의 추정값이나 분산분석표는 이러한 선형제약조건(linear constraints)에 영향을 받지 않는다.

따라서 일반적으로 마지막 수준을 **기본범주(reference category)** 로 하여 가변수를 생략하는 경우, 회귀계수의 의미는 기본범주에 대한 $i$ 수준에서의 처리효과를 나타낸다.

#### 예제

예제 12.1에 대하여 이 방법에 따라 회귀분석모형을 적어 보면

$$y_{ij}=\gamma_0+\gamma_1x_1+\gamma_2x_2+\gamma_3x_3+\epsilon_{ij}$$

이 된다. 행렬표현은 다음과 같다.

$$y=X\beta+\epsilon\\
\begin{bmatrix}
12\\ 18\\ 14\\ 12\\ 13\\ 19\\ 17\\ 21\\ 24\\ 30
\end{bmatrix}
= \begin{bmatrix}
1&1&0&0\\
1&1&0&0\\
1&0&1&0\\
1&0&1&0\\
1&0&1&0\\
1&0&0&1\\
1&0&0&1\\
1&0&0&1\\
1&0&0&0\\
1&0&0&0
\end{bmatrix}
\begin{bmatrix}
\gamma_0\\
\gamma_1\\
\gamma_2\\
\gamma_3
\end{bmatrix}
+\epsilon$$

공식을 사용하여 $\gamma_i$들의 추정값을 구하면

$$\hat{\beta}
= \begin{bmatrix}
\hat{\gamma}_0\\
\hat{\gamma}_1\\
\hat{\gamma}_2\\
\hat{\gamma}_3
\end{bmatrix}
= (X^TX)^{-1}X^Ty\\
\begin{bmatrix}
10&2&3&3\\
2&2&0&0\\
3&0&3&0\\
3&0&0&3
\end{bmatrix}^{-1}
\begin{bmatrix}
12+18+\cdots+24+30\\
12+18\\
14+12+13\\
19+17+21
\end{bmatrix}\\
\begin{bmatrix}
0.5&-0.5&-0.5&-0.5\\
-0.5&1&-0.5&0.5\\
-0.5&0.5&0.8333&0.5\\
-0.5&0.5&0.5&0.8333
\end{bmatrix}
\begin{bmatrix}
180\\ 30\\ 39\\ 57
\end{bmatrix}
=\begin{bmatrix}
27\\ -12\\ -14\\ -8
\end{bmatrix}$$

이 얻어진다. 이 추정값들을 검토하여 보면 다음과 같은 사실을 알 수 있다.

$$\hat{\gamma}_0=\hat{\mu}+\hat{\alpha}_4=\bar{y}+(\bar{y}_4-\bar{y})=\bar{y}_4\\
\hat{\gamma}_i=\hat{\alpha}_i-\hat{\alpha}_4
=(\bar{y}_i-\bar{y})-(\bar{y}_4-\bar{y})
=\bar{y}_i-\bar{y}_4,\qquad i=1,2,3$$

모형이 재조정되어 회귀계수들이 다른 의미를 갖더라도 얻어지는 회귀변동은 변함이 없다. 즉, 적합된 모형

$$\hat{y}=\hat{\gamma}_0+\sum_{i=1}^{3}\hat{\gamma}_ix_i$$

에 대한 회귀변동은

$$SSR=\hat{\beta}^TX^Ty-n(\bar{y})^2\\
=(\hat{\gamma}_0,\hat{\gamma}_1,\hat{\gamma}_2,\hat{\gamma}_3)
\begin{bmatrix}
180\\ 30\\ 39\\ 57
\end{bmatrix}
-10(\bar{y})^2\\
=(27)(180)+(-12)(30)+(-14)(39)+(-8)(57)-10(18)^2
=258$$

로 예제 12.1에서 얻은 회귀변동과 동일하다. 따라서 분산분석표를 작성하는 데에는 재조정된 모형을 쓰거나 원래의 모형을 그대로 쓰거나 아무런 차이가 없다.

### 12.4.3 반복이 없는 이원배치법

앞에서 검토된 두 가지의 재조정법을 이원배치법(two-way layout)에서도 그대로 적용할 수 있다. 먼저 반복이 없는 이원배치법에 대하여 재조정법 I을 알아보자. 가정이

$$\alpha_l=-(\alpha_1+\alpha_2+\cdots+\alpha_{l-1})\\
\beta_m=-(\beta_1+\beta_2+\cdots+\beta_{m-1})
\tag{12.33}$$

이므로, 이를 모형 (12.17)에 대입시키면

$$y_{ij}
= \mu+\alpha_1(x_1-x_l)+\alpha_2(x_2-x_l)+\cdots+\alpha_{l-1}(x_{l-1}-x_l)\\
\qquad
+\beta_1(x_{l+1}-x_{l+m})+\cdots+\beta_{m-1}(x_{l+m-1}-x_{l+m})
+\epsilon_{ij}\\
=\mu+\sum_{i=1}^{l-1}\alpha_iw_i+\sum_{j=1}^{m-1}\beta_jw_{l+j-1}+\epsilon_{ij}
\tag{12.34}$$

로 표현된다. 이때 새로운 변수 $w_k$ $(k=1,2,\cdots,l+m-2)$는 $-1,0,1$의 값을 갖는다. 예를 들어 $l=2$, $m=3$이라면

$$y_{ij} =
\mu+\alpha_1(x_1-x_2)+\beta_1(x_3-x_5)+\beta_2(x_4-x_5)+\epsilon_{ij} \\
=\mu+\alpha_1w_1+\beta_1w_2+\beta_2w_3+\epsilon_{ij}$$

가 되어 회귀분석모형은 다음과 같이 나타내어진다.

$$y=X\beta+\epsilon\\
\begin{bmatrix}
y_{11}\\ y_{12}\\ y_{13}\\ y_{21}\\ y_{22}\\ y_{23} \end{bmatrix}
= \begin{bmatrix}
1&1&0&0\\
1&1&0&1\\
1&1&-1&-1\\
1&-1&1&0\\
1&-1&0&1\\
1&-1&-1&-1
\end{bmatrix}
\begin{bmatrix}
\mu\\ \alpha_1\\ \beta_1\\ \beta_2 
\end{bmatrix}
+\epsilon$$

이 모형의 $X$행렬은 물론 정칙행렬이 된다. 따라서 $(\mu,\alpha_1,\beta_1,\beta_2)$는

$$\hat{\beta} =
\begin{bmatrix}
\hat{\mu}\\
\hat{\alpha}_1\\
\hat{\beta}_1\\
\hat{\beta}_2
\end{bmatrix}
=(X^TX)^{-1}X^Ty$$

에 의하여 추정되고, $\alpha_2$와 $\beta_3$는

$$\hat{\alpha}_2=-\hat{\alpha}_1\\ \hat{\beta}_3=-(\hat{\beta}_1+\hat{\beta}_2)$$

와 같이 추정된다.  
방법 II에 의한 재조정법을 사용하게 되면 이원배치법의 선형모형

$$y_{ij}=\mu+\sum_{i=1}^{l}\alpha_ix_i+\sum_{j=1}^{m}\beta_jx_{l+j}+\epsilon_{ij}$$

에서 $x_l$과 $x_{l+m}$을 제거시킨 축소모형(reduced model)이 되며, 이를

$$y_{ij} =
\gamma_0+\sum_{i=1}^{l-1}\gamma_ix_i+\sum_{j=1}^{m-1}\tau_jx_{l+j-1}+\epsilon_{ij}$$

로 나타낸다. 그러면 모수들 간의 관계는

$$\gamma_0=\mu+\alpha_l+\beta_m\\
\gamma_i=\alpha_i-\alpha_l,\qquad i=1,2,\cdots,l-1\\
\tau_j=\beta_j-\beta_m,\qquad j=1,2,\cdots,m-1$$

과 같아진다.

### 12.4.4 반복이 있는 이원배치법

$A_i$ 수준과 $B_j$ 수준에서 데이터의 반복측정이 있고, 그 반복수가 일정한 이원배치법의 경우를 살펴보자. 먼저 12.4.1절의 방법 I을 보면 가정 (12.20)으로부터

$$\alpha_l=-\sum_{i=1}^{l-1}\alpha_i\\
\beta_m=-\sum_{j=1}^{m-1}\beta_j\\
(\alpha\beta)_{lj}
=-\sum_{i=1}^{l-1}(\alpha\beta)_{ij},
\qquad j=1,2,\cdots,m\\
(\alpha\beta)_{im}
=-\sum_{j=1}^{m-1}(\alpha\beta)_{ij},
\qquad i=1,2,\cdots,l
\tag{12.35}$$

이므로, 이를 모형 (12.21)에 대입시키면

$$y_{ijk} =
\mu+\sum_{i=1}^{l-1}\alpha_i(x_i-x_l)
+\sum_{j=1}^{m-1}\beta_j(x_{l+j}-x_{l+m})\\
\qquad
+\sum_{i=1}^{l-1}\sum_{j=1}^{m-1}
(\alpha\beta)_{ij}(x_i-x_l)(x_{l+j}-x_{l+m})
+\epsilon_{ijk}
\tag{12.36}$$

$$= \mu+\sum_{i=1}^{l-1}\alpha_iw_i
+\sum_{j=1}^{m-1}\beta_jw_{l+j-1}
+\sum_{i=1}^{l-1}\sum_{j=1}^{m-1}(\alpha\beta)_{ij}w_iw_{l+j-1}
+\epsilon_{ijk}
\tag{12.37}$$

예제 12.2의 데이터에서는 $l=4$, $m=3$, $r=2$이므로 모형은

$$y_{ijk} =
\mu+\sum_{i=1}^{3}\alpha_iw_i+\sum_{j=1}^{2}\beta_jw_{3+j}
+\sum_{i=1}^{3}\sum_{j=1}^{2}(\alpha\beta)_{ij}w_iw_{3+j}
+\epsilon_{ijk}$$

가 된다. 이때 $X$와 $\beta$ 행렬을 만들면 다음과 같다.

$$\beta= \begin{bmatrix}
\mu\\ \alpha_1\\ \alpha_2\\ \alpha_3\\ \beta_1\\ \beta_2\\
(\alpha\beta)_{11}\\
(\alpha\beta)_{12}\\
(\alpha\beta)_{21}\\
(\alpha\beta)_{22}\\
(\alpha\beta)_{31}\\
(\alpha\beta)_{32}
\end{bmatrix}$$

해당 $X$행렬은 정칙행렬이므로 $(\mu,\alpha_i,\beta_j,(\alpha\beta)_{ij})$의 추정은 $(X^TX)^{-1}X^Ty$ 에 의하여 얻어지며, 모형 (12.37)에 나타나 있지 않은 $\alpha_l,\beta_m,(\alpha\beta)_{lj},(\alpha\beta)_{im}$ 등은 가정 (12.35)의 관계로부터 얻어진다.  
또한 12.4.2절의 방법 II에 의한 방법으로는 모형 (12.21)에서 변수 $x_l$, $x_{l+m}$, $x_ix_{l+j}\ (j=1,2,\cdots,m)$, $x_ix_{l+m}\ (i=1,2,\cdots,l)$를 모두 제거하고, 축소모형

$$y_{ijk} =
\gamma_0+\sum_{i=1}^{l-1}\gamma_ix_i+\sum_{j=1}^{m-1}\tau_jw_{l+j-1}
+\sum_{i=1}^{l-1}\sum_{j=1}^{m-1}(\gamma\tau)_{ij}x_iw_{l+j-1}
+\epsilon_{ijk}
\tag{12.38}$$

을 만들어 분석하면 된다. 상세한 내용은 생략하고 연습문제를 통하여 익히도록 한다.

> 이 절의 핵심은 "분산분석표 자체는 변하지 않지만, 회귀계수의 해석은 재조정 방식에 따라 달라진다"는 점이다. 따라서 각 방법의 **회귀계수의 의미**, **왜 $X^TX$가 정칙행렬이 되도록 만드는지**를 고찰해보면 된다.
