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
은 자유도 $p - q$인 카이제곱분포(chi-square distribution)를 따르고,
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
귀무가설 $H_0: \beta_{q+1} = \cdots = \beta_p = 0$ 을 검정하기 위한 통계량은
$$
F_0 = \frac{SS(\hat{\mathbf{\beta}}_{q+1}, \dots, \hat{\mathbf{\beta}}_p \mid \hat{\mathbf{\beta}}_0, \dots, \hat{\mathbf{\beta}}_q)/(p-q)}{MSE} \\
F_0 \sim F(p-q, n-p-1)
$$
$F_0 > F_\alpha(p-q, n-p-1)$이면 귀무가설을 기각한다.

- $SSR = SS(\hat{\mathbf{\beta}}_0, \hat{\mathbf{\beta}}_1, \dots, \hat{\mathbf{\beta}}_p) = \mathbf{\hat{\beta}}^T X^T y$: 전체모형의 회귀제곱합
- $SST = \mathbf{y^T y} = \sum y_i^2$: 총제곱합

**수정항을 빼지 않는 분산분석표**  
| 요인 | 제곱합      | 자유도 | 평균제곱    | $F_0$       | $F(\alpha)$ |
| -- | -------- | --- | ----- | ------- | --------- |
| 회귀 | $SSR$ | $p+1$   | $$MSR = SSR/(p+1)$$ | $$F_0 = \frac{MSR}{MSE}$$ | $F_\alpha(p+1, n-p-1)$ |
| 잔차 | $SST - SSR$    | $n-p-1$   | $$MSE = (SST - SSR)/(n-p-1)$$   |         |           |
| 계  | $SST$     | $n$   |         |         |           |

### 6.1.6 직교성(Orthogonality)과 제곱합 분해
$$
y = X_1\beta_1 + X_2\beta_2 + \cdots + X_q\beta_q + \varepsilon
$$
행렬 X를 둘로 나눠서 생각해보자 
$$
X = 
\begin{pmatrix}
1 & x_{11} & \cdots & x_{1q} & \vdots & x_{1,q+1} & \cdots & x_{1p} \\
1 & x_{21} & \cdots & x_{2q} & \vdots & x_{2,q+1} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & \cdots & x_{nq} & \vdots & x_{n,q+1} & \cdots & x_{np}
\end{pmatrix}
= (X_1 \vdots X_2)
$$
$\mathbf{\hat{\beta}}_j = (X_j^T X_j)^{-1} X_j^T \mathbf{y}$를 구하고 이를 분할하여 $\mathbf{\hat{\beta}} = (\mathbf{\hat{\beta}}_1^T, \mathbf{\hat{\beta}}_2^T)^T$라 하자. $\mathbf{\hat{\beta}}_1 = (\hat{\beta}_0, \hat{\beta}_1, \dots, \hat{\beta}_q)^T$는 $X_1$에 대한 회귀계수 추정치이고, $\mathbf{\hat{\beta}}_2 = (\hat{\beta}_{q+1}, \dots, \hat{\beta}_p)^T$는 $X_2$에 대한 회귀계수 추정치이다.  
만약 $X_1^T X_2 = 0$이면, 즉 $X_1$과 $X_2$가 서로 직교(orthogonal)하면,
$$
SS(\mathbf{\hat{\beta}}_1, \mathbf{\hat{\beta}}_2) = SS(\mathbf{\hat{\beta}}_1) + SS(\mathbf{\hat{\beta}}_2) \\
SS(\mathbf{\hat{\beta}}_1 \mid \mathbf{\hat{\beta}}_2) = SS(\mathbf{\hat{\beta}}_1)\\
SS(\mathbf{\hat{\beta}}_2 \mid \mathbf{\hat{\beta}}_1) = SS(\mathbf{\hat{\beta}}_2)
$$
이다.

#### 증명
중회귀모형을 $\mathbf{y} = X\beta + \varepsilon$로 표현하자. $X$를 $X = (X_1 \vdots X_2)$로 분할하면, 
$$
\begin {pmatrix}
X_1^T X_1 & X_1^T X_2 \\
X_2^T X_1 & X_2^T X_2
\end{pmatrix} 
\begin {pmatrix}
\mathbf{\hat{\beta}_1} \\
\mathbf{\hat{\beta}_2}
\end{pmatrix}
= \begin {pmatrix}
X_1^T \mathbf{y} \\
X_2^T \mathbf{y}
\end{pmatrix}
$$
만약 $X_1^T X_2 = 0$이면, 정규방정식은 다음과 같이 분리된다.
$$
\begin {pmatrix}
\mathbf{\hat{\beta}_1} \\
\mathbf{\hat{\beta}_2}
\end{pmatrix}
= \begin {pmatrix}
(X_1^T X_1)^{-1} O_{(p-q)\times (q+1)} \\
O_{(q+1)\times (p-q)} (X_2^T X_2)^{-1}
\end{pmatrix}^{-1}
\begin {pmatrix}
X_1^T \mathbf{y} \\
X_2^T \mathbf{y}
\end{pmatrix}
= \begin {pmatrix}
(X_1^T X_1)^{-1} X_1^T \mathbf{y} \\
(X_2^T X_2)^{-1} X_2^T \mathbf{y}
\end{pmatrix}
$$
이 되어 $\mathbf{\hat{\beta}}_1$과 $\mathbf{\hat{\beta}}_2$가 서로 독립적으로 추정되고 이것은 회귀모형을 개별로 적합시켜 얻은 추정과 같다.  
이 결과로 $SS(\mathbf{\hat{\beta}}_1, \mathbf{\hat{\beta}}_2) = SS(\mathbf{\hat{\beta}}_1) + SS(\mathbf{\hat{\beta}}_2)$가 성립한다. 또한 $SS(\mathbf{\hat{\beta}}_1 \mid \mathbf{\hat{\beta}}_2) = SS(\mathbf{\hat{\beta}}_1)$와 $SS(\mathbf{\hat{\beta}}_2 \mid \mathbf{\hat{\beta}}_1) = SS(\mathbf{\hat{\beta}}_2)$도 성립한다.  
따라서 
$$ SS(\mathbf{\hat{\beta}}_1, \mathbf{\hat{\beta}}_2) 
= \mathbf{\hat{\beta}}^T X^T y = \mathbf{\hat{\beta}}_1^T X_1^T y + \mathbf{\hat{\beta}}_2^T X_2^T y = SS(\mathbf{\hat{\beta}}_1) + SS(\mathbf{\hat{\beta}}_2) $$

위 결과들은 $X_1$과 $X_2$가 서로 직교할 때만 성립한다. 만약 $X_1^T X_2 \ne 0$이면, 즉 $X_1$과 $X_2$가 서로 직교하지 않으면, $\mathbf{\hat{\beta}}_1$과 $\mathbf{\hat{\beta}}_2$는 서로 독립적으로 추정되지 않고, 회귀제곱합도 분해되지 않는다.

유의할 사항은 $X_1$과 $X_2$가 서로 직교하면 충분하지 $X_1$내의 열들끼리는 직교할 필요는 없다는 것이다. $X_1$과 $X_2$가 서로 직교하기만 하면, $X_1$내의 열들끼리는 서로 직교하지 않아도 된다. $X_2$내의 열들끼리도 서로 직교하지 않아도 된다. 

위 결과는 $X$를 둘로 분할한 것만 아니라, $q$개로 분할했을 때에도 성립한다.

#### 예 6.1
다음과 같은 자료에 대하여 중회귀모형 $y_i = \beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i} + \beta_3 x_{3i} + \varepsilon_i$를 가정한다.

| $y$   | 7  | 8  | 10 | 15 | 18 | 26 |
| ----- | -- | -- | -- | -- | -- | -- |
| $x_1$ | 1  | 2  | 3  | 1  | 2  | 3  |
| $x_2$ | -1 | -1 | -1 | 1  | 1  | 1  |
| $x_3$ | -1 | 0  | 1  | 1  | 0  | -1 |

**(1) 최소제곱추정과 분산분석표 (ANOVA Table)**  
설계행렬 $X$에 대하여
$$
X^T X =
\begin{pmatrix}
6 & 12 & 0 & 0 \\
12 & 28 & 0 & 0 \\
0 & 0 & 6 & 0 \\
0 & 0 & 0 & 4
\end{pmatrix} \\

(X^T X)^{-1} =
\begin{pmatrix}
1.16667 & -0.5 & 0 & 0 \\
-0.5 & 0.25 & 0 & 0 \\
0 & 0 & 0.16667 & 0 \\
0 & 0 & 0 & 0.25
\end{pmatrix} \\

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

유의수준 0.05에서 $F_{0.05}(4,2) = 19.25$ 이므로 회귀제곱합은 유의하다.

**(2) $\beta_1$의 추가제곱합 (Extra SS for $\beta_1$)**  
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
SS(\hat{\beta}_1 \mid \hat{\beta}_0,\hat{\beta}_2,\hat{\beta}_3) = 1433.678 - 1384.678 = 49.0
$$
가설 $H_0 : \beta_1 = 0$ 에 대한 검정통계량은
$$
F_0 = \frac{49.0}{2.161} = 22.67
$$
이며,
$$
F_{0.05}(1,2) = 18.51
$$
보다 크므로 $H_0$를 기각한다.

**(3) 직교성 확인 (Orthogonality)**  
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

### 6.2.1 부분 F-검정 (Partial F-Test)
모형 $y = \beta_0 + \sum_{j=1}^{p} \beta_j x_j + \varepsilon$에서 특정 변수 $x_j$의 필요성을 검정하고자 한다. $x_j$를 추가함으로써 증가하는 추가제곱합은
$$
SS(\hat{\beta}_j \mid \hat{\beta}_0,\hat{\beta}_1,\dots,\hat{\beta}_{j-1},\hat{\beta}_{j+1},\dots,\hat{\beta}_p)
$$
이며 자유도는 1이다. 가설은 $H_0 : \beta_j = 0, \quad  H_1 : \beta_j \ne 0$이며, 검정통계량은
$$
F_0 = \frac{SS(\hat{\beta}_j \mid \text{others})}{MSE} \\
$$
$F_0 > F(1, n-p-1)$이면 귀무가설은 기각되어 변수 $x_j$는 모형에 유의하게 기여한다는 결론이 나온다. 이와같은 검정을 **부분 $F$-검정(partial F-test)** 이라고 한다.

부분 $F$-검정은 특정 변수의 유의성을 검정하는 방법으로, 중회귀분석에 포함된 변수의 수가 과다하게 많다고 생각될 때 중요하지 않은 변수를 제거하는 방법으로 활용할 수 있다.
  - 1. 모든 $j$에 대해 부분 $F$-검정을 수행하여 유의하지 않은 변수를 제거한다.
  - 2. 제거된 변수들을 제외한 모형에 대하여 다시 부분 $F$-검정을 수행한다.
  - 3. 더 이상 제거할 변수가 없을 때까지 1과 2를 반복한다.
  - 더 자세한건 변수선택(selection of variables) 참고

이는 5장에서 소개한 일반선형가설(general linear hypothesis) $H_0 : C\beta = 0$의 특수한 경우이다.

### 6.2.2 완전모형과 축소모형 비교 (Full vs Reduced Model)
완전모형(full model): $y = \beta_0 + \sum_{j=1}^{p} \beta_j x_j + \varepsilon$  
축소모형(reduced model): $y = \alpha_0 + \sum_{k \ne j} \alpha_k x_k + \varepsilon$

$SST = SSR(F) + SSE(F) = SSR(R) + SSE(R)$이므로,
$$
SSR(F) - SSR(R) = SS(\hat{\beta}_j \mid \text{others})
$$
로서 추가제곱합과 일치한다.  
따라서 부분 F-검정은 두 모형의 회귀제곱합 차이에 기반한다.

### 6.2.3 축차 F-검정 (Sequential F-Test)
변수를 하나씩 추가해 가며 검정하는 방법이다.

**1단계:**  
$SS(\hat{\beta}_1 \mid \hat{\beta}_0)$를 구하고 부분F검정을 거쳐 유의한것 중 가장 큰 추가제곱합을 갖는 변수를 우선선택

**2단계:**  
$SS(\hat{\beta}_2 \mid \hat{\beta}_0,\hat{\beta}_1)$를 구하고 부분F검정을 거쳐 유의한것 중 가장 큰 추가제곱합을 갖는 변수를 선택  
이와 같이 순차적으로 추가한다.

$SS(\hat{\beta}_j \mid \hat{\beta}_0, \hat{\beta}_1)$의 F검정에서 추가제곱합은 
$$
SS(\hat{\beta}_j \mid \hat{\beta}_0, \hat{\beta}_1) = SS(\hat{\beta}_0, \hat{\beta}_1, \hat{\beta}_j) - SS(\hat{\beta}_0, \hat{\beta}_1)
$$
자유도는 1이고 MSE는 모형 $y_i = \beta_0 + \beta_1 x_{1i} + \varepsilon_i$에 대한 잔차제곱합(SSE)를 자유도 $n-3$으로 나눈 것이다. 따라서 검정통계량은
$$
F_0 = \frac{SS(\hat{\beta}_j \mid \text{previous})}{MSE}
$$
- $F$의 기각치는 $F_\alpha(1, n-3)$이다.
- 만약 $\beta_j$열과 $\beta_1, \beta_1$열이 서로 직교한다면 $SS(\hat\beta_j \mid \hat\beta_0, \hat\beta_1)$는 $SS(\hat\beta_j)$와 같아지므로 $SSR = SS(\hat\beta_1 \mid \hat\beta_0, \hat\beta_1)$가 된다.


특징
* 각 단계마다 완전모형이 달라지며 따라서 MSE 값도 단계마다 변한다.
* 그런데 부분 F검정이 F검정으로서 타당성을 지니기 위해서는 MSE가 오차항의 분산에 대한 불편추정량이 되어야 한다
  - 더 엄밀하게는 $1/\sigma^2 SSE$가 $\chi^2$분포를 따르고 이의 비중심도는 0이어야 한다
* 따라서 완전모형으로 사용하는 모형이 충분히 큰 모형이어서 자료의 true model을 포함하고 있지 않으면 축차F검정을 적용할 수 없다

**F-검정과 축차 F-검정의 차이**  
| 구분     | 부분 F        | 축차 F         |
| ------ | ----------- | ------------ |
| 기준모형   | 모든 다른 변수 포함 | 이전 단계 변수만 포함 |
| 순서 의존성 | 없음          | 있음           |
| MSE    | 고정          | 단계마다 변함      |


## 6.3 변수의 표준화 (Standardization of Variables)
중회귀모형
$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \varepsilon
$$
에서 설명변수들의 단위가 서로 다를 경우 회귀계수의 상대적 크기를 직접 비교하기 어렵다. 이를 해결하기 위하여 변수의 표준화(standardization of variables)를 수행한다.

### 6.3.1 평균 중심화 (Centering)
먼저 각 설명변수의 평균을 $\bar{x}_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij}$라 하고 다음과 같이 모형을 변형한다.
$$
y = (\beta_0 + \beta_1 \bar{x}_1 + \cdots + \beta_p \bar{x}_p) + \beta_1(x_1 - \bar{x}_1) + \cdots + \beta_p(x_p - i \bar{x}_p) + \varepsilon
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
가 된다. 이는 $\hat{\beta}_1, \hat{\beta}_2, \dots, \hat{\beta}_p$값이 뭐든지 간에 $\hat{\beta}_0'$는 항상 $\bar{y}$가 된다는 것을 의미한다.  
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
Z_{ij} = \frac{w_{ij}}{\sqrt{S_{jj}}} = \frac{x_{ij} - \bar{x}_j}{\sqrt{S_{jj}}}, \qquad j=1,\dots,p \\
y_i^* = \frac{y_i - \bar{y}}{\sqrt{S_{yy}}}, \qquad S_{yy} = \sum (y_i - \bar{y})^2
$$
로 정의한다. 이때
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
변수 $Z_j$는 $x_j$의 표준화된 버전이며, $y^*$는 $y$의 표준화된 버전이다. $a_j$는 표준화 회귀계수(standardized regression coefficient)라고 불린다.

### 6.3.4 상관행렬 표현 (Correlation Matrix Form)
표준화된 변수를 사용하는 회귀모형은 재미있는 성질을 가지고 있다.  
표준화 변수에 대해
$$
\sum_i Z_{ij} Z_{il} = \sum_i \frac{w_{ij}}{\sqrt{S_{jj}}} \frac{w_{il}}{\sqrt{S_{ll}}}
= \frac{S_{jl}}{\sqrt{S_{jj} S_{ll}}} = r_{jl}
$$
이 되며 이는 $Z_j, Z_l$의 표본상관계수(sample correlation coefficient)이며, 따라서
$$
X^T X =
\begin{pmatrix}
1 & r_{12} & \cdots & r_{1p} \\
r_{21} & 1 & \cdots & r_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
r_{p1} & r_{p2} & \cdots & 1
\end{pmatrix}
$$
는 설명변수들의 상관행렬(correlation matrix)이 되어 설명변수 $x_1, x_2, \dots, x_p$의 상관구조를 모두 알 수 있다.  
이와같은 $X^T X$를 변수$Z$의 상관행렬(correlation matrix)이라고도 한다.

또한
$$
X^T y = \begin{pmatrix}
\frac{\sum_i w_{i1} (y_i - \bar{y})}{\sqrt{S_{11} S_{yy}}} \\
\frac{\sum_i w_{i2} (y_i - \bar{y})}{\sqrt{S_{22} S_{yy}}} \\
\vdots \\
\frac{\sum_i w_{ip} (y_i - \bar{y})}{\sqrt{S_{pp} S_{yy}}}
\end{pmatrix}
= \begin{pmatrix}
r_{1y} \\
r_{2y} \\
\vdots \\
r_{py}
\end{pmatrix}
$$
으로 $y$와 각 설명변수 $x_j$의 상관계수 벡터가 된다.

따라서 정규방정식은
$$
R a = r_y \\
\begin{pmatrix}
1 & r_{12} & \cdots & r_{1p} \\
r_{21} & 1 & \cdots & r_{2p} \\ 
\vdots & \vdots & \ddots & \vdots \\
r_{p1} & r_{p2} & \cdots & 1
\end{pmatrix}
\begin{pmatrix}
\hat a_1 \\
\hat a_2 \\
\vdots \\
\hat a_p
\end{pmatrix}
= \begin{pmatrix}
r_{1y} \\
r_{2y} \\
\vdots \\
r_{py}
\end{pmatrix}
$$
형태가 되며,
* $R$: 설명변수 상관행렬
* $\hat{a}$: 표준화 회귀계수 벡터
* $r_y$: 종속변수와 각 설명변수의 상관계수 벡터

여기서 얻은 $\hat{a}_j$로 원래의 회귀모형의 계수인 $\hat\beta_j$를 구하려면 $\hat{\beta}_j = \hat{a}_j \sqrt{S_{yy}/S_{jj}}$를 사용한다.

### 6.3.5 두 설명변수의 경우 (Case of Two Predictors)
설명변수가 두 개인 경우 $y^* = a_1 Z_1 + a_2 Z_2 + \varepsilon^*$에 대하여 정규방정식은
$$
\begin{pmatrix}
1 & r_{12} \\
r_{12} & 1
\end{pmatrix}

\begin{pmatrix}
\hat a_1 \\
\hat a_2
\end{pmatrix}
= \begin{pmatrix}
r_{1y} \\
r_{2y}
\end{pmatrix}
$$

행렬식(determinant): $D = 1 - r_{12}^2$이며,
$$
\hat{a}_1 = \frac{r_{1y} - r_{12} r_{2y}}{D}, \qquad
\hat{a}_2 = \frac{r_{2y} - r_{12} r_{1y}}{D}
$$

분산
$$
\operatorname{Var}
\begin{pmatrix}
\hat{a}_1 \\
\hat{a}_2
\end{pmatrix}
= \sigma^2
\begin{pmatrix}
1 & r_{12} \\
r_{12} & 1
\end{pmatrix}^{-1} \\
$$
이므로
$$
\operatorname{Var}(\hat{a}_1)
=\operatorname{Var}(\hat{a}_2)
=\frac{\sigma^2}{1 - r_{12}^2}
$$

**다중공선성(Multicollinearity)**  
만약 $|r_{12}| \to 1$이면 $D \to 0$이 되어 분산이 급격히 증가한다. 이는 다중공선성(multicollinearity) 문제를 의미한다. 추정문제에서도, 설명변수들 간 상관이 높을수록 회귀계수의 추정은 불안정해진다.

### 6.3.6 원래 계수로의 환원 (Back Transformation)
표준화 회귀계수로부터 원래 계수는
$$
\hat{\beta}_j = \hat{a}_j
\sqrt{\frac{S_{yy}}{S_{jj}}},
\quad j=1,\dots,p \\
\hat{\beta}_0
=\bar{y}-\sum_{j=1}^{p}
\hat{\beta}_j \bar{x}_j
$$
표준화 여부는 최종 회귀식의 추정값에 영향을 주지 않는다. 단지 계산 방식과 해석의 편의를 제공할 뿐이다.

>**해석**  
>표준화 회귀계수 $a_j$는 각 설명변수가 1 표준편차 증가할 때 종속변수가 몇 표준편차 변화하는지를 의미한다.  
>따라서 서로 다른 단위를 가진 변수들 간 상대적 영향력을 비교할 수 있다.


## 6.4 공동신뢰영역 (Joint Confidence Region)
중회귀모형 $\mathbf{y} = \mathbf{X}\mathbf{\beta} + \varepsilon, \quad \varepsilon \sim N(0, \sigma^2 I_n)$을 가정한다. 여기서 $\mathbf{\beta} = (\beta_0, \beta_1, \dots, \beta_p)^T$는 $(p+1)$차원 회귀계수 벡터이다.  
이 절에서는 각 계수에 대한 개별 신뢰구간(individual confidence interval)이 아니라, 벡터 $\mathbf{\beta}$ 전체를 ($p+1$)차원 공간에서 동시에 포함할 확률이 $100(1-\alpha)\%$인 공동신뢰영역(joint confidence region)을 유도한다.

### 6.4.1 기본 분포이론 (Distributional Result)
최소제곱추정량은 $\hat{\mathbf{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$이며, $\hat{\mathbf{\beta}} \sim N\bigl(\mathbf{\beta}, \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}\bigr)$을 따른다.  
또한, $\hat{\mathbf{\beta}} - \mathbf{\beta}=(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \varepsilon$이므로
$$
\frac{1}{\sigma^2}
(\hat{\mathbf{\beta}}-\mathbf{\beta})^T
\mathbf{X}^T \mathbf{X}
(\hat{\mathbf{\beta}}-\mathbf{\beta})
= \frac{1}{\sigma^2}
\varepsilon^T \mathbf{H} \varepsilon
$$
여기서 $\mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T$는 hat matrix이다.

정규성 가정과 이차형식의 성질에 의해
$$
\frac{1}{\sigma^2}
(\hat{\mathbf{\beta}}-\mathbf{\beta})^T
\mathbf{X}^T \mathbf{X}
(\hat{\mathbf{\beta}}-\mathbf{\beta})
\sim
\chi^2(p+1)
$$
한편, $SSE = \varepsilon^T (I-\mathbf{H})\varepsilon$이고,
$$ 
\frac{SSE}{\sigma^2} \sim \chi^2(n-p-1)
$$
또한 위 두 통계량은 서로 독립이다.
$$
\therefore \frac{(\hat{\mathbf{\beta}}-\mathbf{\beta})^T \mathbf{X}^T \mathbf{X} (\hat{\mathbf{\beta}}-\mathbf{\beta})/(p+1)}{SSE/(n-p-1)}
=\frac{(\hat{\mathbf{\beta}}-\mathbf{\beta})^T \mathbf{X}^T \mathbf{X} (\hat{\mathbf{\beta}}-\mathbf{\beta})}{(p+1)\,\text{MSE}}
\sim F(p+1, n-p-1)
$$

### 6.4.2 공동신뢰영역의 정의
위 결과로부터 $\mathbf{\beta}$의 $100(1-\alpha)\%$ 공동신뢰영역은
$$
(\hat{\mathbf{\beta}}-\mathbf{\beta})^T \mathbf{X}^T \mathbf{X}(\hat{\mathbf{\beta}}-\mathbf{\beta}) \le (p+1)\cdot \text{MSE} \cdot F_{\alpha}(p+1, n-p-1)
$$
로 주어진다. 이는 $\mathbf{\beta}$-공간에서 중심이 $\hat{\mathbf{\beta}}$이고, 형태는 타원체(ellipsoid)인 영역이다.

### 6.4.3 단순회귀모형의 경우 (Simple Linear Regression Case)
단순선형회귀모형 ${y}_i = \beta_0 + \beta_1 x_i + \varepsilon_i$에서 $p=1$이므로 공동신뢰영역은 2차원 타원이다.
$$
\mathbf{X}^T \mathbf{X} =
\begin{pmatrix}
n & \sum x_i \\
\sum x_i & \sum x_i^2
\end{pmatrix}
$$
이고,
$$
(\hat{\mathbf{\beta}}-\mathbf{\beta})^T \mathbf{X}^T \mathbf{X} (\hat{\mathbf{\beta}}-\mathbf{\beta})
= n(\beta_0-\hat{\beta}_0)^2
+ (\sum x_i^2)(\beta_1-\hat{\beta}_1)^2
+ 2(\sum x_i)(\beta_0-\hat{\beta}_0)(\beta_1-\hat{\beta}_1)
$$
이다. 따라서 공동신뢰영역은
$$
n(\mathbf{\beta}_0-\hat{\mathbf{\beta}}_0)^2
+ (\sum x_i^2)(\mathbf{\beta}_1-\hat{\mathbf{\beta}}_1)^2
+ 2(\sum x_i)(\mathbf{\beta}_0-\hat{\mathbf{\beta}}_0)(\mathbf{\beta}_1-\hat{\mathbf{\beta}}_1)
\le 2\,\text{MSE}\,F_{\alpha}(2, n-2)
$$
로 표현되는 타원이다. 재밌는점은, 공분산행렬은 $\operatorname{Var}(\hat{\mathbf{\beta}}) = \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}$이고, 단순회귀의 경우
$$
(\mathbf{X}^T \mathbf{X})^{-1}
= \frac{1}{D}
\begin{pmatrix}
\sum x_i^2 & -\sum x_i \\
-\sum x_i & n
\end{pmatrix} \\
D = n\sum x_i^2 - (\sum x_i)^2
$$
따라서 공동신뢰영역 식은 아래처럼 변형된다
$$
Var(\hat{\mathbf{\beta}}_1)(\beta_0-\hat\beta_0)^2 + Var(\hat{\mathbf{\beta}}_0)(\beta_1-\hat\beta_1)^2 - 2 \operatorname{Cov}(\hat{\mathbf{\beta}}_0,\hat{\mathbf{\beta}}_1)(\beta_0-\hat\beta_0)(\beta_1-\hat\beta_1) \\
\le 2\cdot \text{MSE}\cdot F_{\alpha}(2, n-2) / D \cdot \sigma^2
\\
$$
이때 
$$
\operatorname{Var}(\hat{\mathbf{\beta}}_0) = \sigma^2 \frac{\sum x_i^2}{D} \\
\operatorname{Var}(\hat{\mathbf{\beta}}_1) = \sigma^2 \frac{n}{D} \\
\operatorname{Cov}(\hat{\mathbf{\beta}}_0,\hat{\mathbf{\beta}}_1) = -\sigma^2 \frac{\sum x_i}{D}
$$
* 분산이 클수록 타원은 해당 축 방향으로 길어진다.
* 공분산이 0이면 타원의 축은 좌표축과 평행하다.
* 공분산이 양수이면 우상향 기울기,
* 음수이면 우하향 기울기를 갖는다.

($\hat{\beta}_0$를 구하고 싶지 않으면 변수의 평균 중심화를 수행하여 $\hat{\beta}_0$가 $\bar{y}$가 되도록 하면 된다.)

### 6.4.5 동시신뢰구간 (Simultaneous Confidence Interval)
공동신뢰영역과 관련된 개념으로 동시신뢰구간(simultaneous confidence interval)이 있다.  
이는 각 $\beta_j$에 대한 구간
$$
\hat{\mathbf{\beta}}_j \pm c \sqrt{\operatorname{Var}(\hat{\mathbf{\beta}}_j)}
$$
을 구성하되, 모든 구간이 동시에 참모수를 포함할 확률이 $1-\alpha$가 되도록 상수 $c$를 조정하는 방법이다.
- 각 모수들에 대해 '같은 수준'의 신뢰구간들을 계산
- 이들의 cartesian product가 모수들을 동시에 포함할 확률이 $1-\alpha$가 되게 조정
- 따라서 동시신뢰구간은 항상 각 모수들의 신뢰구간들의 곱집합은 $p+1$차원 입방체의 형태로 표현됨
- 본페로니 방법, 쉐페 방법 등이 존재

#### (1) Bonferroni 방법
$$
A_j = \left\{\beta_j: \hat{\mathbf{\beta}}_j - t_{\alpha/(2(p+1))}(n-p-1) \sqrt{\operatorname{\widehat{Var}}(\hat{\mathbf{\beta}}_j)} \le \beta_j \le \hat{\mathbf{\beta}}_j + t_{\alpha/(2(p+1))}(n-p-1) \sqrt{\operatorname{\widehat{Var}}(\hat{\mathbf{\beta}}_j)} \right\} \\
= \left[ \hat{\mathbf{\beta}}_j - t_{\alpha/(2(p+1))}(n-p-1) \sqrt{\operatorname{\widehat{Var}}(\hat{\mathbf{\beta}}_j)},\ \hat{\mathbf{\beta}}_j + t_{\alpha/(2(p+1))}(n-p-1) \sqrt{\operatorname{\widehat{Var}}(\hat{\mathbf{\beta}}_j)} \right]
$$
이렇게 계산된 입방체의 확률은
$$
P(A_0 \times A_1 \times \cdots \times A_p) = P\left(\bigcap_{j=0}^{p}A_j\right) \\
\ge 1 - \sum_{j=0}^{p} P(A_j^c) = 1 - (p+1) \cdot \frac{\alpha}{p+1} =
1-\alpha
$$
를 만족한다.

#### (2) Scheffé 방법
모든 선형결합 $\psi = a^T \beta$ 에 대해
$$
\psi = \left\{\psi = \mathbf{a}^T \mathbf{\beta}, \forall \mathbf{a} \in \mathbb{R}^{p+1} \right\}
$$
의 형태로 구간을 구성한다.  
각 모수$\psi_j$에 대한 신뢰구간은
$$
\mathbf{a}^\top \hat{\mathbf{\beta}} \pm \sqrt{(p+1)F_{\alpha}(p+1,n-p-1)}
\sqrt{\mathbf{a}^T \operatorname{Var}(\hat{\mathbf{\beta}}) \mathbf{a}}
$$
여기서 각 모수의 신뢰수준은 $\mathbf{a}$의 선택과 무관하게 항상 일정하다. 이는 동시신뢰구간의 경우 '대전제'임을 기억할 필요가 있다. 다만 p가 커짐에 따라 두 방법등의 동시신뢰구간 추정방법 등은 신뢰구간이 크게 넓어진다.

### 6.4.6 공동신뢰영역과 개별 신뢰구간의 비교
* 개별 신뢰구간은 각 모수에 대해 독립적으로 구성된다.
* 공동신뢰영역은 벡터 전체를 동시에 포함한다.
* 단순회귀에서 개별 신뢰구간은 직사각형 영역을 형성한다.
* 공동신뢰영역은 타원이며 일반적으로 직사각형보다 면적이 작다.
* 따라서 공동신뢰영역이 더 많은 정보를 제공한다.

공동신뢰영역은 회귀계수 벡터가 존재할 수 있는 확률 $1-\alpha$의 타원체 영역을 의미한다.  
이는 다변량 통계적 추론(multivariate inference)의 기본 구조이며, 일반선형가설 검정과 직접적으로 연결된다.  
또한, 타원체의 형태는 설계행렬 $\mathbf{X}$의 구조와 설명변수 간 상관구조에 의해 결정된다.


## 6.5 회귀모형의 비교검정 (Model Comparison Test)
이 절에서는 서로 다른 두 개 이상의 집단에서 적합된 회귀모형들이 동일한 회귀계수(regression coefficients)를 갖는지를 검정하는 방법을 다룬다. 이는 공정 변경 전후의 수율 비교, 서로 다른 지역·집단의 반응 비교 등에서 자주 등장하는 문제이다.

### 6.5.1 두 회귀모형의 비교 (Comparison of Two Regression Models)
두 집단에 대해 각각 다음과 같은 중회귀모형이 적합되었다고 하자.
$$
\mathbf{y}_1 = \mathbf{X}_1 \mathbf{\beta}_1 + \varepsilon_1, \qquad \mathbf{y}_2 = \mathbf{X}_2 \mathbf{\beta}_2 + \varepsilon_2
$$

* $\mathbf{y}_1$: $n_1 \times 1$ 벡터
* $\mathbf{y}_2$: $n_2 \times 1$ 벡터
* $\mathbf{X}_1$: $n_1 \times (p+1)$ 행렬
* $\mathbf{X}_2$: $n_2 \times (p+1)$ 행렬
* $\mathbf{\beta}_1, \mathbf{\beta}_2$: $(p+1) \times 1$ 벡터

검정하고자 하는 가설은
$$
H_0 : \mathbf{\beta}_1 = \mathbf{\beta}_2 = \mathbf{\beta}_0\\
H_1 : \mathbf{\beta}_1 \ne \mathbf{\beta}_2
$$

### 6.5.2 완전모형과 축소모형 (Full and Reduced Models)
#### (1) 완전모형 (Full Model)
각 집단에 대해 별도의 회귀식을 적합한다.
$$
y_1 = \mathbf{X}_1 \mathbf{\beta}_1 + \varepsilon_1\\
y_2 = \mathbf{X}_2 \mathbf{\beta}_2 + \varepsilon_2
$$
잔차제곱합은
$$
SSE(F)
= (y_1 - \mathbf{X}_1 \hat{\mathbf{\beta}}_1)^T (y_1 - \mathbf{X}_1 \hat{\mathbf{\beta}}_1)
+ (y_2 - \mathbf{X}_2 \hat{\mathbf{\beta}}_2)^T (y_2 - \mathbf{X}_2 \hat{\mathbf{\beta}}_2) \\
\hat{\mathbf{\beta}}_1 = (\mathbf{X}_1^T \mathbf{X}_1)^{-1} \mathbf{X}_1^T y_1, \quad \hat{\mathbf{\beta}}_2 = (\mathbf{X}_2^T \mathbf{X}_2)^{-1} \mathbf{X}_2^T y_2
$$
자유도는
$$
n_1 - (p+1) + n_2 - (p+1) = n - 2(p+1) \\ n = n_1 + n_2
$$

#### (2) 축소모형 (Reduced Model)
두 집단의 회귀계수가 동일하다고 가정한다. 벡터를 결합하여
$$
y = \begin{pmatrix}
y_1 \\
y_2
\end{pmatrix},
\quad
\mathbf{X} =
\begin{pmatrix}
\mathbf{X}_1 \\
\mathbf{X}_2
\end{pmatrix} \\
y = \mathbf{X} \mathbf{\beta}_0 + \varepsilon
$$
최소제곱추정량은
$$
\hat{\mathbf{\beta}}_0 = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T y
$$
잔차제곱합은
$$SSE(R) = (y - \mathbf{X}\hat{\mathbf{\beta}}_0)^T (y - \mathbf{X}\hat{\mathbf{\beta}}_0)$$
자유도는
$n - (p+1)$

>3장에서 사용한 검정절차:  
>순서1: 완정모형 적합, 잔차제곱합 SSE(F) 계산  
>순서2: 축소모형 적합, 잔차제곱합 SSE(R) 계산  
>순서3: 검정통계량 계산 및 판정  
### 6.5.3 검정통계량 (Test Statistic)
정규성 가정하에서
$$
\frac{SSE(R) - SSE(F)}{\sigma^2} \sim \chi^2(p+1) \\
\frac{SSE(F)}{\sigma^2} \sim \chi^2(n - 2(p+1))
$$
이며 두 통계량은 독립이다. 따라서 검정통계량은
$$
F_0 = \frac{[SSE(R) - SSE(F)]/(p+1)}{SSE(F)/(n - 2(p+1))} \\
F_0 \sim F(p+1, n - 2(p+1))
$$

### 6.5.4 판정기준
유의수준 $\alpha$에서
$$
F_0 > F_\alpha(p+1, n - 2(p+1))
$$
이면 귀무가설을 기각한다. 즉, 두 집단의 회귀계수는 동일하지 않다고 판단한다.  

위에서 두 개의 표본(각각 표본크기가 n1, n2)을 사용하여 회귀모형을 적합한 후, 두 모형의 잔차제곱합을 비교하여 검정통계량을 계산했는데, k개의 표본에 대한 검정도 동일하게 할 수 있다.
### 6.5.5 k개 집단의 비교 (Comparison of k Regression Models)
이제 $k$개의 집단을 고려하자.
$$
\mathbf{y}_i = \mathbf{X}_i \mathbf{\beta}_i + \varepsilon_i, \quad i=1, \dots,k
$$
각 집단의 표본크기는 $n_i$, 전체 표본크기는 $n = \sum_{i=1}^{k} n_i$  

검정가설은
$$
H_0 :
\mathbf{\beta}_1 = \mathbf{\beta}_2 = \cdots = \mathbf{\beta}_k = \mathbf{\beta}_0 \\
H_1 :
\text{적어도 하나의 } \mathbf{\beta}_i \ne \mathbf{\beta}_0
$$

#### (1) 완전모형
$$
SSE(F)
=\sum_{i=1}^{k}
(\mathbf{y}_i - \mathbf{X}_i \hat{\mathbf{\beta}}_i)^T
(\mathbf{y}_i - \mathbf{X}_i \hat{\mathbf{\beta}}_i)
$$
자유도는$n - k(p+1)$

#### (2) 축소모형
$$
\mathbf{y} =\begin{pmatrix}
\mathbf{y}_1 \\
\mathbf{y}_2 \\
\vdots \\
\mathbf{y}_k
\end{pmatrix},
\quad
\mathbf{X} =
\begin{pmatrix}
\mathbf{X}_1 \\
\mathbf{X}_2 \\
\vdots \\
\mathbf{X}_k
\end{pmatrix} \\
SSE(R) = (\mathbf{y} - \mathbf{X}\hat{\mathbf{\beta}}_0)^T (\mathbf{y} - \mathbf{X}\hat{\mathbf{\beta}}_0)
$$

#### (3) 검정통계량
$$
F_0 = \frac{[SSE(R) - SSE(F)] / [(k-1)(p+1)]}{SSE(F) / [n - k(p+1)]} \\
F_0 \sim F\big((k-1)(p+1), n - k(p+1)\big)
$$

### 6.5.6 해석
* 이 검정은 회귀계수 벡터 전체의 동일성을 검정한다.
* 본질적으로 추가제곱합(extra sum of squares)에 기반한 일반선형가설 검정이다.
* 설계행렬을 확장하여 하나의 큰 모형으로 표현할 수 있다.
* 집단 간 차이는 상호작용항(interaction term) 검정과 동일한 구조를 가진다.
