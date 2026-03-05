# Chapter 6 추정과 가설검정 II (Estimation and Hypothesis Testing II)

## 6.1 추가제곱합 (Extra Sum of Squares)
중회귀분석(multiple regression analysis)에서는 특정 설명변수(explanatory variable)를 모형에 포함하는 것이 통계적으로 유의한지를 판단해야 하는 경우가 빈번하다. 특정변수를 포함하지 않고 구한 회귀제곱합과 변수를 포함하여 구한 회귀제곱합의 차이를 이용하여 검정하는 방법이 바로 부분 F-검정(partial F-test)이다.  
이를 위해 사용되는 핵심 개념이 추가제곱합(extra sum of squares)으로, 추가로 증가한 제곱합을 의미한다.
$$ SS(X_2 \mid X_1) = SS(X_1, X_2) - SS(X_1) $$
  - $SS(X_1)$: $X_1$만 포함한 모형의 회귀제곱합
  - $SS(X_1, X_2)$: $X_1$과 $X_2$ 모두 포함한 모형의 회귀제곱합
  - $SS(X_2 \mid X_1)$: $X_1$이 이미 포함된 상태에서 $X_2$를 추가함으로써 증가하는 제곱합

### 6.1.1 기본 모형 설정
다음과 같은 중회귀모형을 고려한다.
$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \varepsilon,
\quad \varepsilon \sim N(0, \sigma^2)
$$
최소제곱법(least squares method)에 의해 $\hat{\beta}_0, \hat{\beta}_1, \dots, \hat{\beta}_p$를 추정하고, 오차분산 $\sigma^2$는 평균제곱오차(mean square error, MSE)로 추정한다.

기존 회귀제곱합(regression sum of squares)은
$$SSR = SS(\hat{\beta}_1, \dots, \hat{\beta}_p \mid \hat{\beta}_0) = \sum_{i=1}^n (\hat{y}_i - \bar{y})^2 = \hat{\beta}^T X^T \mathbf{y} - n\bar{y}^2$$
  - SST: $\sum (y_i - \bar{y})^2$: 총제곱합(total sum of squares), 총 편차
  - SSR: $\sum (\hat{y}_i - \bar{y})^2$: 회귀제곱합(regression sum of squares), 회귀로 설명되는 편차
  - SSE: $\sum (y_i - \hat{y}_i)^2$: 잔차제곱합(error sum of squares), 설명되지 않는 편차

절편(intercept)만 포함하는 모형 $y_i = \beta_0 + \varepsilon_i$의 회귀제곱합 $SS(\hat{\beta}_0)$는 절편항의 기여도를 나타내며, 이는 다음과 같이 유도된다.
>설계행렬은 $X = \mathbf{1} \in \mathbb{R}^{n \times 1}$ (모든 원소가 1인 열벡터)이다. 정규방정식 $X^T(y - X\hat{\beta}_0) = 0$에서
>$$
>\mathbf{1}^T(y - \mathbf{1}\hat{\beta}_0) = 0
>\quad \Rightarrow \quad
>\sum_{i=1}^n y_i - n\hat{\beta}_0 = 0 \\
>\therefore \hat{\beta}_0 = \bar{y}
>$$
>**회귀제곱합의 유도**  
>정의에 의해
>$$
>SS(\hat{\beta}_0) = \hat{\beta}_0^T X^T y = \hat{\beta}_0 \cdot \mathbf{1}^T y
>$$
>$\hat{\beta}_0 = \bar{y}$이고 스칼라이므로
>$$
>SS(\hat{\beta}_0) = \bar{y} \sum_{i=1}^n y_i = \bar{y} \cdot n\bar{y} = n\bar{y}^2
>$$
>한편, 예측값은 $\hat{y}_i = \hat{\beta}_0 = \bar{y}$ (모든 $i$에 대해 상수)이므로
>$$
>SS(\hat{\beta}_0) = \sum_{i=1}^n (\hat{y}_i - \bar{y})^2 = \sum_{i=1}^n (\bar{y} - \bar{y})^2 = 0
>$$
>이는 명백한 모순처럼 보이나, 실제로는 표기의 차이에서 비롯된다. 회귀제곱합 $SS(\hat{\beta}_0)$는 상수항의 기여도를 나타내며, 정규방정식의 형태로 정의될 때는 $n\bar{y}^2$이다. 이는 절편항이 표본평균을 중심으로 회귀를 수행할 때의 제곱합을 의미한다.
>
자유도는 1이다.

따라서 전체 회귀제곱합은
$$
SS(\hat{\beta}_1, \dots, \hat{\beta}_p \mid \hat{\beta}_0)
= SS(\hat{\beta}_0, \hat{\beta}_1, \dots, \hat{\beta}_p) - SS(\hat{\beta}_0) \\
(df=k) = (df=k+1) - (df=1)
$$

### 6.1.2 부분모형과 추가제곱합 (Reduced Model and Extra SS)
이제 $p$개의 변수 중 $q$개만 포함한 부분모형(reduced model)을 고려한다.
$$
y = \alpha_0 + \alpha_1 x_1 + \cdots + \alpha_q x_q + \varepsilon
$$
전체모형(full model)은
$$
y = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p + \varepsilon
$$
선택되지 않은 변수인 $x_{q+1}, \dots, x_p$를 추가함으로써 증가하는 회귀제곱합이며, 이는 $SS(\hat{\beta}) - SS(\hat{\alpha})$와 같다.  
  - $SS(\hat{\alpha_0}, \hat{\alpha}_1, \dots, \hat{\alpha}_q \mid \hat{\alpha}_0) = SS(\hat{\alpha}_0, \hat{\alpha}_1, \dots, \hat{\alpha}_q) - SS(\hat{\alpha}_0)= \hat{\alpha}^T X_1^T y - n\bar{y}^2$: 절편항만 포함한 모형과 부분모형의 회귀제곱합 차이, 자유도는 $q$
  - $SS(\hat{\alpha_0}, \hat{\alpha}_1, \dots, \hat{\alpha}_q)= \hat{\alpha}^T X_1^T y$: 부분모형의 회귀제곱합, 자유도는 $q+1$

$p-q$개의 변수를 추가함으로써 증가하는 제곱합은
$$
SS(\beta_{q+1}, \dots, \beta_p \mid \beta_0, \dots, \beta_q) \\
= SS(\hat{\beta}_0, \hat{\beta}_1, \dots, \hat{\beta}_p \mid \hat\beta_0) - SS(\hat{\alpha}_0, \hat{\alpha}_1, \dots, \hat{\alpha}_q \mid \hat{\alpha}_0) \\
= SS(\hat{\beta}_0, \hat{\beta}_1, \dots, \hat{\beta}_p) - SS(\hat{\alpha}_0, \hat{\alpha}_1, \dots, \hat{\alpha}_q) \\
= SS(\hat{\mathbf{\beta}}) - SS(\hat{\mathbf{\alpha}})$$

자유도는 $(p + 1) - (q + 1) = p - q$이다.

### 6.1.3 행렬표현과 기하학적 해석 (Matrix Form and Geometric Interpretation)
모형을 다음과 같이 표현한다.
$$
y = X_1 \beta_1 + \varepsilon\\
y = X_1 \beta_1 + X_2 \beta_2 + \varepsilon
$$
* $\mathbf{\beta}_1 = (\beta_0, \beta_1, \dots, \beta_q)^T$: 부분모형의 회귀계수 벡터
* $\mathbf{\beta}_2 = (\beta_{q+1}, \dots, \beta_p)^T$: 추가되는 변수들의 회귀계수 벡터
* $X_1$: 부분모형의 설계행렬(design matrix)
* $X_2$: 추가되는 변수들의 설계행렬
* 추가제곱합 $SS(\mathbf{\beta}_2 \mid \mathbf{\beta}_1)$: $X_1$이 이미 포함된 상태에서 $X_2$를 추가함으로써 증가하는 제곱합

해트행렬(hat matrix)을
$$
H_1 = X_1(X_1^T X_1)^{-1} X_1^T\\
H = X(X^T X)^{-1} X^T
$$
라 하면,
$$
SS(\mathbf{\beta}_2 \mid \mathbf{\beta}_1) = \mathbf{y}^T (H - H_1) \mathbf{y}
$$

### 정리 6.1
$(H - H_1)$은 멱등행렬(idempotent matrix)이며, 그 계수(rank)는 $p - q$이다.

>**증명**  
>1. 멱등성: $(H - H_1)^2 = H - H_1$  
>   $H$와 $H_1$은 모두 멱등행렬이므로,
>   $$
>   (H - H_1)^2 = H^2 - H H_1 - H_1 H + H_1^2 = H - H_1
>   $$
>2. 계수: $\operatorname{rank}(H - H_1) = p - q$  
>   $H$는 전체모형의 예측공간(prediction space)을 나타내며, $H_1$은 부분모형의 예측공간을 나타낸다. $H - H_1$은 부분모형의 예측공간에 직교하는 공간을 나타내며, 이 공간의 차원은 $p - q$이다.
>   따라서 $\operatorname{rank}(H - H_1) = p - q$이다.

### 정리 6.2 분포적 성질 (Distributional Properties)
정규성 가정 하에서
$$
\frac{1}{\sigma^2} SS(\beta_2 \mid \beta_1)
$$
은 자유도 $p - q$인 카이제곱분포(chi-square distribution)를 따른다.  
또한
$$
SSE = \mathbf{y}^T(I - H)\mathbf{y}
$$
와 서로 독립이다.

>**증명**
>1. $SS(\beta_2 \mid \beta_1) = \mathbf{y}^T (H - H_1) \mathbf{y}$
>2. $H - H_1$은 멱등행렬이며, 그 계수는 $p - q$이다.
>3. $\mathbf{y} \sim N(X\beta, \sigma^2 I)$이므로, $SS(\beta_2 \mid \beta_1)/\sigma^2$는 자유도 $p - q$인 카이제곱분포를 따른다.
>4. $SSE = \mathbf{y}^T(I - H)\mathbf{y}$는 자유도 $n - p - 1$인 카이제곱분포를 따른다.
>5. $H$와 $I - H$는 서로 직교하는 투영행렬(projection matrix)이므로, $SS(\beta_2 \mid \beta_1)$과 $SSE$는 서로 독립이다.

### 6.1.5 부분 F-검정 (Partial F-Test)
귀무가설
$$
H_0: \beta_{q+1} = \cdots = \beta_p = 0
$$

을 검정하기 위한 통계량은
$$
F_0 = \frac{SS(\hat{\mathbf{\beta}}_2 \mid \hat{\mathbf{\beta}}_1)/(p-q)}{MSE} \\
F_0 \sim F(p-q, n-p-1)
$$

$F_0 > F_\alpha(p-q, n-p-1)$이면 귀무가설을 기각한다.

### 6.1.6 직교성(Orthogonality)과 제곱합 분해
설계행렬을 $X = (X_1, X_2)$라 하자.

만약
$$
X_1^T X_2 = 0
$$
이면 두 부분공간은 직교한다.

이 경우 다음이 성립한다.
$$
SS(\hat{\beta}) = SS(\hat{\beta}_1) + SS(\hat{\beta}_2)\\
SS(\hat{\beta}_1 \mid \hat{\beta}_2) = SS(\hat{\beta}_1)\\
SS(\hat{\beta}_2 \mid \hat{\beta}_1) = SS(\hat{\beta}_2)
$$
즉, 변수의 포함 순서에 관계없이 동일한 제곱합을 갖는다.

### 6.1.7 일반적 분할 (General Partition)
$$
y = X_1\beta_1 + X_2\beta_2 + \cdots + X_q\beta_q + \varepsilon
$$
에서 모든 $i \ne j$에 대해
$$
X_i^T X_j = 0
$$
이면

$$
\hat{\beta}_j = (X_j^T X_j)^{-1} X_j^T y
$$
이고

$$
SS(\hat{\beta}) = \sum_{j=1}^q SS(\hat{\beta}_j)
$$
이다.

이 경우 각 회귀계수의 추정은 다른 변수 포함 여부에 영향을 받지 않는다.

### 6.1.8 예 6.1 (Example 6.1)
다음과 같은 자료에 대하여 중회귀모형
$$
y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \beta_3 x_{3i} + \varepsilon_i
$$
를 가정한다.

|       | 1  | 2  | 3  | 4  | 5  | 6  |
| ----- | -- | -- | -- | -- | -- | -- |
| $y$   | 7  | 8  | 10 | 15 | 18 | 26 |
| $x_1$ | 1  | 2  | 3  | 1  | 2  | 3  |
| $x_2$ | -1 | -1 | -1 | 1  | 1  | 1  |
| $x_3$ | -1 | 0  | 1  | 1  | 0  | -1 |

(1) 최소제곱추정과 분산분석표 (ANOVA Table)

설계행렬 $X$에 대하여
$$
X^T X =
\begin{pmatrix}
6 & 12 & 0 & 0 \\
12 & 28 & 0 & 0 \\
0 & 0 & 6 & 0 \\
0 & 0 & 0 & 4
\end{pmatrix}
$$

$$
(X^T X)^{-1} =
\begin{pmatrix}
1.16667 & -0.5 & 0 & 0 \\
-0.5 & 0.25 & 0 & 0 \\
0 & 0 & 0.16667 & 0 \\
0 & 0 & 0 & 0.25
\end{pmatrix}
$$

$$
X^T y =
\begin{pmatrix}
84 \\ 182 \\ 34 \\ -8
\end{pmatrix}
$$
따라서

$$
\hat{\beta} =
\begin{pmatrix}
7.0 \\ 3.5 \\ 5.667 \\ -2.0
\end{pmatrix}
$$

즉 추정된 회귀식은
$$
\hat{y} = 7.0 + 3.5x_1 + 5.667x_2 - 2.0x_3
$$


제곱합은
$$
SST = y^T y = 1438\\
SSR = \hat{\beta}^T X^T y = 1433.678\\
SSE = 4.322
$$

분산분석표는 다음과 같다.

| 요인 | 제곱합      | 자유도 | 평균제곱    | F       |
| -- | -------- | --- | ------- | ------- |
| 회귀 | 1433.678 | 4   | 358.420 | 165.858 |
| 잔차 | 4.322    | 2   | 2.161   |         |
| 계  | 1438     | 6   |         |         |

유의수준 0.05에서
$$
F_{0.05}(4,2) = 19.25
$$
이므로 회귀제곱합은 유의하다.

(2) $\beta_1$의 추가제곱합 (Extra SS for $\beta_1$)

축소모형:
$$
y = \beta_0 + \beta_2 x_2 + \beta_3 x_3 + \varepsilon
$$
에 대하여 계산하면
$$
SS(\hat{\beta}_0,\hat{\beta}_2,\hat{\beta}_3) = 1384.678
$$

따라서
$$
SS(\hat{\beta}_1 \mid \hat{\beta}_0,\hat{\beta}_2,\hat{\beta}_3)
= 1433.678 - 1384.678
= 49.0
$$

가설
$$
H_0 : \beta_1 = 0
$$
에 대한 검정통계량은
$$
F_0 = \frac{49.0}{2.161} = 22.67
$$
이며,
$$
F_{0.05}(1,2) = 18.51
$$
보다 크므로 $H_0$를 기각한다.

(3) 직교성 확인 (Orthogonality)

행렬을
$$
X = (X_1, X_2)
$$
로 분할하면
$$
X_1^T X_2 = 0
$$
이 성립한다.

따라서
$$
\hat{\beta}_1 = (X_1^T X_1)^{-1} X_1^T y\\
\hat{\beta}_2 = (X_2^T X_2)^{-1} X_2^T y
$$
이며 이는 전체모형에서 구한 추정치와 동일하다.

또한
$$
SS(\hat{\beta}) = SS(\hat{\beta}_1) + SS(\hat{\beta}_2)\\
SS(\hat{\beta}_1 \mid \hat{\beta}_2) = SS(\hat{\beta}_1)\\
SS(\hat{\beta}_2 \mid \hat{\beta}_1) = SS(\hat{\beta}_2)
$$
가 성립한다.


## 6.2 F-검정과 축차 F-검정 (Partial and Sequential F-Test)

### 6.2.1 F-검정 (Partial F-Test)
모형
$$
y = \beta_0 + \sum_{j=1}^{p} \beta_j x_j + \varepsilon
$$
에서 특정 변수 $x_j$의 필요성을 검정하고자 한다.

추가제곱합은
$$
SS(\hat{\beta}_j \mid \hat{\beta}_0,\hat{\beta}_1,\dots,\hat{\beta}_{j-1},\hat{\beta}_{j+1},\dots,\hat{\beta}_p)
$$
이며 자유도는 1이다.

가설은
$$
H_0 : \beta_j = 0
\quad
H_1 : \beta_j \ne 0
$$

검정통계량은
$$
F_0 =
\frac{
SS(\hat{\beta}_j \mid \text{others})
}{
MSE
}
$$

이고
$$
F_0 \sim F(1, n-p-1)
$$

이는 일반선형가설(general linear hypothesis)
$$
H_0 : C\beta = 0
$$
의 특수한 경우이다.

### 6.2.2 완전모형과 축소모형 비교 (Full vs Reduced Model)
완전모형(full model):
$$
y = \beta_0 + \sum_{j=1}^{p} \beta_j x_j + \varepsilon
$$

축소모형(reduced model):
$$
y = \alpha_0 + \sum_{k \ne j} \alpha_k x_k + \varepsilon
$$
이면
$$
SSR(F) - SSR(R)
= SS(\hat{\beta}_j \mid \text{others})
$$

따라서 부분 F-검정은 두 모형의 회귀제곱합 차이에 기반한다.

### 6.2.3 축차 F-검정 (Sequential F-Test)
변수를 하나씩 추가해 가며 검정하는 방법이다.

1단계:
$$
SS(\hat{\beta}_1 \mid \hat{\beta}_0)
$$

2단계:
$$
SS(\hat{\beta}_2 \mid \hat{\beta}_0,\hat{\beta}_1)
$$
…
이와 같이 순차적으로 추가한다.

검정통계량은
$$
F_0 =
\frac{
SS(\hat{\beta}_j \mid \text{previous})
}{
MSE
}
$$

특징
* 각 단계마다 완전모형이 달라진다.
* 따라서 MSE 값도 단계마다 변한다.
* 변수 선택(selection of variables)에 활용된다.

### 6.2.4 F-검정과 축차 F-검정의 차이
| 구분     | 부분 F        | 축차 F         |
| ------ | ----------- | ------------ |
| 기준모형   | 모든 다른 변수 포함 | 이전 단계 변수만 포함 |
| 순서 의존성 | 없음          | 있음           |
| MSE    | 고정          | 단계마다 변함      |

부분 F-검정은 전체모형이 참모형(true model)을 포함한다는 가정 하에서 유효하다.

이로써 추가제곱합을 이용한 부분 F-검정과 축차 F-검정의 이론적 구조가 완성된다.


## 6.3 변수의 표준화 (Standardization of Variables)
중회귀모형
$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \varepsilon
$$
에서 설명변수들의 단위가 서로 다를 경우 회귀계수의 상대적 크기를 직접 비교하기 어렵다. 이를 해결하기 위하여 변수의 표준화(standardization of variables)를 수행한다.

### 6.3.1 평균 중심화 (Centering)
먼저 각 설명변수의 평균을
$$
\bar{x}_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij}
$$
라 하자.

다음과 같이 모형을 변형한다.
$$
y = (\beta_0 + \beta_1 \bar{x}_1 + \cdots + \beta_p \bar{x}_p) + \beta_1(x_1 - \bar{x}_1) + \cdots + \beta_p(x_p - \bar{x}_p) + \varepsilon
$$

여기서
$$
\beta_0' = \beta_0 + \sum_{j=1}^{p} \beta_j \bar{x}_j,
\qquad
w_{ij} = x_{ij} - \bar{x}_j
$$
라 두면,
$$
y = \beta_0' + \beta_1 w_1 + \cdots + \beta_p w_p + \varepsilon
$$
이 된다.

정규방정식(normal equations)의 첫 번째 식을 보면
$$
n\hat{\beta}_0' + \hat{\beta}_1 \sum w_{i1} + \cdots + \hat{\beta}_p \sum w_{ip} = \sum y_i
$$
인데

$$
\bar{w}_j = \frac{1}{n} \sum_i (x_{ij} - \bar{x}_j) = 0
$$
이므로

$$
\hat{\beta}_0' = \bar{y}
$$
가 된다.

따라서 중심화된 모형은
$$
y - \bar{y} = \beta_1 w_1 + \cdots + \beta_p w_p + \varepsilon'
$$
로 쓸 수 있으며, 상수항 없이 회귀를 수행할 수 있다.

이 경우 설계행렬 $X$의 열 개수는 $p+1$에서 $p$로 줄어들어 계산량이 감소한다.

### 6.3.2 분산·공분산 행렬 표현 (Cross-Product Matrix)
중심화 변수에 대해
$$
\sum_i w_{ij} w_{il} = \sum_i (x_{ij} - \bar{x}_j)(x_{il} - \bar{x}_l) = S_{jl}
$$
라 하자.
* $S_{jj} = \sum (x_{ij} - \bar{x}_j)^2$
* $S_{jl} = \sum (x_{ij} - \bar{x}_j)(x_{il} - \bar{x}_l)$


정규방정식의 계수행렬은
$$
X^T X =
\begin{pmatrix}
S_{11} & S_{12} & \cdots & S_{1p} \\
S_{21} & S_{22} & \cdots & S_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
S_{p1} & S_{p2} & \cdots & S_{pp}
\end{pmatrix}
$$

### 6.3.3 표준화 변수의 정의 (Standardized Variables)
각 변수에 대해
$$
Z_{ij} = \frac{x_{ij} - \bar{x}_j}{\sqrt{S_{jj}}},
\qquad
j=1,\dots,p \\
y_i^* = \frac{y_i - \bar{y}}{\sqrt{S_{yy}}},
\qquad
S_{yy} = \sum (y_i - \bar{y})^2
$$
로 정의한다.

이때
$$
\sum_i Z_{ij}^2 = 1,
\qquad
\sum_i (y_i^*)^2 = 1
$$
이 성립한다.

표준화된 회귀모형은
$$
y^* = a_1 Z_1 + a_2 Z_2 + \cdots + a_p Z_p + \varepsilon^*
$$
로 쓸 수 있으며,
$$
a_j = \beta_j \sqrt{\frac{S_{jj}}{S_{yy}}}
$$

### 6.3.4 상관행렬 표현 (Correlation Matrix Form)
표준화 변수에 대해
$$
\sum_i Z_{ij} Z_{il} = \frac{S_{jl}}{\sqrt{S_{jj} S_{ll}}} = r_{jl}
$$
이 되며 이는 표본상관계수(sample correlation coefficient)이다.

따라서
$$
X^T X =
\begin{pmatrix}
1 & r_{12} & \cdots & r_{1p} \\
r_{21} & 1 & \cdots & r_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
r_{p1} & r_{p2} & \cdots & 1
\end{pmatrix}
$$
는 설명변수들의 상관행렬(correlation matrix)이 된다.

또한
$$
X^T y =
\begin{pmatrix}
r_{1y} \\
r_{2y} \\
\vdots \\
r_{py}
\end{pmatrix}
$$
가 된다.

따라서 정규방정식은
$$
R a = r_y
$$

형태가 되며,
* $R$: 설명변수 상관행렬
* $a$: 표준화 회귀계수 벡터
* $r_y$: 종속변수와 각 설명변수의 상관계수 벡터

### 6.3.5 두 설명변수의 경우 (Case of Two Predictors)
설명변수가 두 개인 경우
$$
\begin{pmatrix}
1 & r_{12} \\
r_{12} & 1
\end{pmatrix}
\begin{pmatrix}
a_1 \\
a_2
\end{pmatrix}
=
\begin{pmatrix}
r_{1y} \\
r_{2y}
\end{pmatrix}
$$

행렬식(determinant)
$$
D = 1 - r_{12}^2
$$
이며,

$$
a_1 = \frac{r_{1y} - r_{12} r_{2y}}{D},
\qquad
a_2 = \frac{r_{2y} - r_{12} r_{1y}}{D}
$$

분산
$$
\operatorname{Var}
\begin{pmatrix}
\hat{a}_1 \\
\hat{a}_2
\end{pmatrix}
=
\sigma^2
\begin{pmatrix}
1 & r_{12} \\
r_{12} & 1
\end{pmatrix}^{-1}
$$
이므로

$$
\operatorname{Var}(\hat{a}_1)
=\operatorname{Var}(\hat{a}_2)
=\frac{\sigma^2}{1 - r_{12}^2}
$$


#### 다중공선성(Multicollinearity)
만약 $|r_{12}| \to 1$이면
$$
D \to 0
$$
이 되어 분산이 급격히 증가한다. 이는 다중공선성(multicollinearity) 문제를 의미한다.

설명변수들 간 상관이 높을수록 회귀계수의 추정은 불안정해진다.

### 6.3.6 원래 계수로의 환원 (Back Transformation)
표준화 회귀계수로부터 원래 계수는
$$
\hat{\beta}_j
=
\hat{a}_j
\sqrt{\frac{S_{yy}}{S_{jj}}},
\quad j=1,\dots,p
$$
이며,

$$
\hat{\beta}_0
=\bar{y}-\sum_{j=1}^{p}
\hat{\beta}_j \bar{x}_j
$$
표준화 여부는 최종 회귀식의 추정값에 영향을 주지 않는다. 단지 계산 방식과 해석의 편의를 제공할 뿐이다.

### 6.3.7 해석
표준화 회귀계수 $a_j$는 각 설명변수가 1 표준편차 증가할 때 종속변수가 몇 표준편차 변화하는지를 의미한다.

따라서 서로 다른 단위를 가진 변수들 간 상대적 영향력을 비교할 수 있다.


## 6.4 공동신뢰영역 (Joint Confidence Region)
중회귀모형
$$
y = X\beta + \varepsilon, \qquad
\varepsilon \sim N(0, \sigma^2 I_n)
$$
을 가정한다. 여기서 $\beta = (\beta_0, \beta_1, \dots, \beta_p)^T$는 $(p+1)$차원 회귀계수 벡터이다.

이 절에서는 각 계수에 대한 개별 신뢰구간(individual confidence interval)이 아니라, 벡터 $\beta$ 전체를 동시에 포함하는 공동신뢰영역(joint confidence region)을 유도한다.

### 6.4.1 기본 분포이론 (Distributional Result)
최소제곱추정량은
$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$
이며,
$$
\hat{\beta} \sim N\bigl(\beta, \sigma^2 (X^T X)^{-1}\bigr)
$$
을 따른다.

또한,
$$
\hat{\beta} - \beta
=(X^T X)^{-1} X^T \varepsilon
$$
이므로
$$
\frac{1}{\sigma^2}
(\hat{\beta}-\beta)^T
X^T X
(\hat{\beta}-\beta)
=
\frac{1}{\sigma^2}
\varepsilon^T H \varepsilon
$$
여기서 $H = X(X^T X)^{-1}X^T$는 hat matrix이다.

정규성 가정과 이차형식의 성질에 의해
$$
\frac{1}{\sigma^2}
(\hat{\beta}-\beta)^T
X^T X
(\hat{\beta}-\beta)
\sim
\chi^2(p+1)
$$

한편,
$$
SSE = \varepsilon^T (I-H)\varepsilon
$$
이고,
$$
\frac{SSE}{\sigma^2}
\sim
\chi^2(n-p-1)
$$

또한 위 두 통계량은 서로 독립이다.

따라서
$$
\frac{
(\hat{\beta}-\beta)^T X^T X (\hat{\beta}-\beta)/(p+1)
}{SSE/(n-p-1)}
=\frac{
(\hat{\beta}-\beta)^T X^T X (\hat{\beta}-\beta)
}{(p+1)\,\text{MSE}}
\sim F(p+1, n-p-1)
$$

### 6.4.2 공동신뢰영역의 정의
위 결과로부터 $\beta$의 $100(1-\alpha)\%$ 공동신뢰영역은
$$
(\hat{\beta}-\beta)^T
X^T X
(\hat{\beta}-\beta)
\le
(p+1)\,\text{MSE}\,
F_{\alpha}(p+1, n-p-1)
$$
로 주어진다.

이는 $\beta$-공간에서 중심이 $\hat{\beta}$이고, 형태는 타원체(ellipsoid)인 영역이다.

### 6.4.3 단순회귀모형의 경우 (Simple Linear Regression Case)
단순선형회귀모형
$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i
$$
에서 $p=1$이므로 공동신뢰영역은 2차원 타원이다.

이때
$$
X^T X
=
\begin{pmatrix}
n & \sum x_i \\
\sum x_i & \sum x_i^2
\end{pmatrix}
$$
이고,
$$
(\hat{\beta}-\beta)^T X^T X (\hat{\beta}-\beta)
= n(\beta_0-\hat{\beta}_0)^2
+ (\sum x_i^2)(\beta_1-\hat{\beta}_1)^2
+ 2(\sum x_i)(\beta_0-\hat{\beta}_0)(\beta_1-\hat{\beta}_1)
$$
이다.

따라서 공동신뢰영역은
$$
n(\beta_0-\hat{\beta}_0)^2
+
(\sum x_i^2)(\beta_1-\hat{\beta}_1)^2
+
2(\sum x_i)(\beta_0-\hat{\beta}_0)(\beta_1-\hat{\beta}_1)
\le
2\,\text{MSE}\,F_{\alpha}(2, n-2)
$$
로 표현되는 타원이다.

### 6.4.4 분산·공분산과 타원 형태
공분산행렬은
$$
\operatorname{Var}(\hat{\beta})
= \sigma^2 (X^T X)^{-1}
$$
이다.

단순회귀의 경우
$$
(X^T X)^{-1}
= \frac{1}{D}
\begin{pmatrix}
\sum x_i^2 & -\sum x_i \\
-\sum x_i & n
\end{pmatrix}
$$
이며
$$
D = n\sum x_i^2 - (\sum x_i)^2
$$
이다.

따라서
$$
\operatorname{Var}(\hat{\beta}_0)
= \sigma^2 \frac{\sum x_i^2}{D} \\
\operatorname{Var}(\hat{\beta}_1)
= \sigma^2 \frac{n}{D} \\
\operatorname{Cov}(\hat{\beta}_0,\hat{\beta}_1)
= -\sigma^2 \frac{\sum x_i}{D}
$$
* 분산이 클수록 타원은 해당 축 방향으로 길어진다.
* 공분산이 0이면 타원의 축은 좌표축과 평행하다.
* 공분산이 양수이면 우상향 기울기,
* 음수이면 우하향 기울기를 갖는다.

### 6.4.5 동시신뢰구간 (Simultaneous Confidence Interval)
공동신뢰영역과 관련된 개념으로 동시신뢰구간(simultaneous confidence interval)이 있다.

이는 각 $\beta_j$에 대한 구간
$$
\hat{\beta}_j
\pm
c \sqrt{\operatorname{Var}(\hat{\beta}_j)}
$$
을 구성하되, 모든 구간이 동시에 참모수를 포함할 확률이 $1-\alpha$가 되도록 상수 $c$를 조정하는 방법이다.

#### (1) Bonferroni 방법
$$
\hat{\beta}_j
\pm
t_{\alpha/(2(p+1))}(n-p-1)
\sqrt{\operatorname{Var}(\hat{\beta}_j)}
$$
이다.

이 경우
$$
P\left(
\bigcap_{j=0}^{p}
A_j
\right)
\ge 1-\alpha
$$
가 된다.

#### (2) Scheffé 방법
모든 선형결합
$$
\psi = a^T \beta
$$
에 대해

$$
a^T\hat{\beta}
\pm
\sqrt{(p+1)F_{\alpha}(p+1,n-p-1)}
\sqrt{a^T \operatorname{Var}(\hat{\beta}) a}
$$
의 형태로 구간을 구성한다.

Scheffé 방법은 모든 가능한 선형결합에 대해 동시에 유효하다.

### 6.4.6 공동신뢰영역과 개별 신뢰구간의 비교
* 개별 신뢰구간은 각 모수에 대해 독립적으로 구성된다.
* 공동신뢰영역은 벡터 전체를 동시에 포함한다.
* 단순회귀에서 개별 신뢰구간은 직사각형 영역을 형성한다.
* 공동신뢰영역은 타원이며 일반적으로 직사각형보다 면적이 작다.
* 따라서 공동신뢰영역이 더 많은 정보를 제공한다.

### 6.4.7 해석
공동신뢰영역은 회귀계수 벡터가 존재할 수 있는 확률 $1-\alpha$의 타원체 영역을 의미한다.

이는 다변량 통계적 추론(multivariate inference)의 기본 구조이며, 일반선형가설 검정과 직접적으로 연결된다.

또한, 타원체의 형태는 설계행렬 $X$의 구조와 설명변수 간 상관구조에 의해 결정된다.


## 6.5 회귀모형의 비교검정 (Model Comparison Test)
이 절에서는 서로 다른 두 개 이상의 집단에서 적합된 회귀모형들이 동일한 회귀계수(regression coefficients)를 갖는지를 검정하는 방법을 다룬다. 이는 공정 변경 전후의 수율 비교, 서로 다른 지역·집단의 반응 비교 등에서 자주 등장하는 문제이다.

### 6.5.1 두 회귀모형의 비교 (Comparison of Two Regression Models)
두 집단에 대해 각각 다음과 같은 중회귀모형이 적합되었다고 하자.
$$
y_1 = X_1 \beta_1 + \varepsilon_1,
\qquad
y_2 = X_2 \beta_2 + \varepsilon_2
$$

* $y_1$: $n_1 \times 1$ 벡터
* $y_2$: $n_2 \times 1$ 벡터
* $X_1$: $n_1 \times (p+1)$ 행렬
* $X_2$: $n_2 \times (p+1)$ 행렬
* $\beta_1, \beta_2$: $(p+1) \times 1$ 벡터

검정하고자 하는 가설은
$$
H_0 : \beta_1 = \beta_2 = \beta_0\\
H_1 : \beta_1 \ne \beta_2
$$


### 6.5.2 완전모형과 축소모형 (Full and Reduced Models)
#### (1) 완전모형 (Full Model)
각 집단에 대해 별도의 회귀식을 적합한다.
$$
y_1 = X_1 \beta_1 + \varepsilon_1\\
y_2 = X_2 \beta_2 + \varepsilon_2
$$

잔차제곱합은
$$
SSE(F)
= (y_1 - X_1 \hat{\beta}_1)^T (y_1 - X_1 \hat{\beta}_1)
+ (y_2 - X_2 \hat{\beta}_2)^T (y_2 - X_2 \hat{\beta}_2)
$$
여기서
$$
\hat{\beta}_1 = (X_1^T X_1)^{-1} X_1^T y_1,
\quad
\hat{\beta}_2 = (X_2^T X_2)^{-1} X_2^T y_2
$$

자유도는
$$
n_1 - (p+1) + n_2 - (p+1)
= n - 2(p+1)
$$
이며 $n = n_1 + n_2$이다.

#### (2) 축소모형 (Reduced Model)
두 집단의 회귀계수가 동일하다고 가정한다.

벡터를 결합하여
$$
y = 
\begin{pmatrix}
y_1 \\
y_2
\end{pmatrix},
\quad
X =
\begin{pmatrix}
X_1 \\
X_2
\end{pmatrix}
$$
라 하면

$$
y = X \beta_0 + \varepsilon
$$


최소제곱추정량은
$$
\hat{\beta}_0 = (X^T X)^{-1} X^T y
$$

잔차제곱합은
$$
SSE(R)
= (y - X\hat{\beta}_0)^T (y - X\hat{\beta}_0)
$$

자유도는
$$
n - (p+1)
$$

### 6.5.3 검정통계량 (Test Statistic)
정규성 가정하에서
$$
\frac{SSE(R) - SSE(F)}{\sigma^2}
\sim
\chi^2(p+1)
$$
이고,

$$
\frac{SSE(F)}{\sigma^2}
\sim
\chi^2(n - 2(p+1))
$$
이며 두 통계량은 독립이다.

따라서 검정통계량은
$$
F_0
= \frac{
[SSE(R) - SSE(F)]/(p+1)
}{
SSE(F)/(n - 2(p+1))
}
$$
이며,
$$
F_0
\sim
F(p+1, n - 2(p+1))
$$
을 따른다.

### 6.5.4 판정기준
유의수준 $\alpha$에서
$$
F_0 > F_\alpha(p+1, n - 2(p+1))
$$
이면 귀무가설을 기각한다.

즉, 두 집단의 회귀계수는 동일하지 않다고 판단한다.

### 6.5.5 k개 집단의 비교 (Comparison of k Regression Models)
이제 $k$개의 집단을 고려하자.
$$
y_i = X_i \beta_i + \varepsilon_i,
\quad i=1,\dots,k
$$
각 집단의 표본크기는 $n_i$, 전체 표본크기는
$$
n = \sum_{i=1}^{k} n_i
$$

검정가설은
$$
H_0 :
\beta_1 = \beta_2 = \cdots = \beta_k = \beta_0 \\
H_1 :
\text{적어도 하나의 } \beta_i \ne \beta_0
$$

#### (1) 완전모형
$$
SSE(F)
=
\sum_{i=1}^{k}
(y_i - X_i \hat{\beta}_i)^T
(y_i - X_i \hat{\beta}_i)
$$

자유도는
$$
n - k(p+1)
$$

#### (2) 축소모형
$$
y =
\begin{pmatrix}
y_1 \\
y_2 \\
\vdots \\
y_k
\end{pmatrix},
\quad
X =
\begin{pmatrix}
X_1 \\
X_2 \\
\vdots \\
X_k
\end{pmatrix}
$$

$$
SSE(R)
=
(y - X\hat{\beta}_0)^T
(y - X\hat{\beta}_0)
$$

#### (3) 검정통계량
$$
F_0
=
\frac{
[SSE(R) - SSE(F)] / [(k-1)(p+1)]
}{
SSE(F) / [n - k(p+1)]
}
$$
이며,

$$
F_0
\sim
F\big((k-1)(p+1), n - k(p+1)\big)
$$

### 6.5.6 해석

* 이 검정은 회귀계수 벡터 전체의 동일성을 검정한다.
* 본질적으로 추가제곱합(extra sum of squares)에 기반한 일반선형가설 검정이다.
* 설계행렬을 확장하여 하나의 큰 모형으로 표현할 수 있다.
* 집단 간 차이는 상호작용항(interaction term) 검정과 동일한 구조를 가진다.
