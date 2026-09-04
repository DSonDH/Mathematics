# Chapter 6 추정과 가설검정 II (Estimation and Hypothesis Testing II)

## 6.1 추가제곱합 (Extra Sum of Squares)
중회귀분석(multiple regression analysis)에서는 특정 설명변수(explanatory variable)를 모형에 포함하는 것이 통계적으로 유의한지를 판단해야 하는 경우가 빈번하다. 특정변수를 포함하지 않고 구한 회귀제곱합과 변수를 포함하여 구한 회귀제곱합의 차이를 이용하여 검정하는 방법이 바로 부분 F-검정(partial F-test)이다.  
이를 위해 사용되는 핵심 개념이 추가제곱합(extra sum of squares)으로, 추가로 증가한 제곱합을 의미한다.

$$ SS(X_2 \mid X_1) = SS(X_1, X_2) - SS(X_1) $$

  - $SS(X_1)$: $X_1$만 포함한 모형의 회귀제곱합
  - $SS(X_1, X_2)$: $X_1$과 $X_2$ 모두 포함한 모형의 회귀제곱합
  - **(중요!!!!!)** $SS(X_2 \mid X_1)$: $X_1$이 이미 포함된 상태에서 $X_2$를 추가함으로써 증가하는 제곱합
    - $X_1$ 조건부도 아니고, $X_1$을 제외한것도 아님.

### 6.1.1 기본 모형 설정
다음과 같은 중회귀모형을 고려한다.

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \varepsilon,
\quad \varepsilon \sim N(0, \sigma^2)
$$

최소제곱법(least squares method)에 의해 $\hat{\beta}_0, \hat{\beta}_1, \dots, \hat{\beta}_p$를 추정하고, 오차분산 $\sigma^2$는 평균제곱오차(mean square error, MSE)로 추정한다.

기존 회귀제곱합(regression sum of squares)을 아래와 같이 표현하자.

$$SSR = SS(\hat{\beta}_1, \dots, \hat{\beta}_p \mid \hat{\beta}_0) = \sum_{i=1}^n (\hat{y}_i - \bar{y})^2 = \hat{\beta}^T X^T \mathbf{y} - n\bar{y}^2$$

  - SST: $\sum (y_i - \bar{y})^2$: 총제곱합(total sum of squares), 총 편차
  - SSR: $\sum (\hat{y}_i - \bar{y})^2$: 회귀제곱합(regression sum of squares), 회귀로 설명되는 편차
  - SSE: $\sum (y_i - \hat{y}_i)^2$: 잔차제곱합(error sum of squares), 설명되지 않는 편차

> 절편만 포함하는 모형 $y_i=\beta_0+\varepsilon_i,\quad i=1,\ldots,n$ 을 고려한다. 설계행렬을 $X_0=\mathbf1\in\mathbb R^{n\times1}$이라고 하면 정규방정식은 $X_0^T(\mathbf y-X_0\hat\beta_0)=0$ 이다. 따라서
>
> $\mathbf1^T(\mathbf y-\mathbf1\hat\beta_0)=0$ 이고, $\sum_{i=1}^ny_i-n\hat\beta_0=0$ 이므로
>
> $$
> \boxed{\hat\beta_0=\bar y}
> $$
>
> 이다. 따라서 이 모형의 적합벡터는 $\hat{\mathbf y}_0=X_0\hat\beta_0=\bar y\mathbf1$ 이다.
>
> $SS(\hat\beta_0)$로 표시하는 절편항의 비보정 제곱합, 즉 수정항(Corrected Factor)은
>
> $$
> \begin{aligned}
> CF
> :=SS_U(X_0)
> &=\hat{\mathbf y}_0^T\hat{\mathbf y}_0\\
> &=\hat\beta_0^2\mathbf1^T\mathbf1\\
> &=n\hat\beta_0^2\\
> &=n\bar y^2.
> \end{aligned}
> $$
>
> 정규방정식 $\mathbf1^T\mathbf y=n\hat\beta_0$을 이용하면 이를
>
> $$
> \begin{aligned}
> CF
> &=\hat\beta_0\mathbf1^T\mathbf y\\
> &=\hat\beta_0X_0^T\mathbf y\\
> &=n\bar y^2
> \end{aligned}
> $$
>
> 로도 나타낼 수 있다. 이 절편 방향의 비보정 제곱합은 자유도
>
> $$
> \text{rank}(H_0)=1
> $$
>
> 을 갖는다.
>
> 한편 통상적인 회귀제곱합 $SSR$은 표본평균을 기준으로 정의된다. 절편만 있는 모형에서는 모든 적합값이 $\bar y$이므로
>
> $$
> \begin{aligned}
> SSR_0
> &=\sum_{i=1}^n(\hat y_{0i}-\bar y)^2\\
> &=\sum_{i=1}^n(\bar y-\bar y)^2\\
> &=0.
> \end{aligned}
> $$
>
> 따라서
>
> $$
> \boxed{CF=SS_U(X_0)=n\bar y^2}
> $$
>
> 와
>
> $$
> \boxed{SSR_0=0}
> $$
>
> 을 구분해야 한다. 전자는 원점 기준 절편항의 비보정 제곱합이며 자유도는 1이고, 후자는 평균 기준 회귀제곱합이며 절편을 제외한 회귀 자유도는 0이다.

따라서 

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
>   $X=(X_1,X_2)$이므로 $X_1$의 각 열은 $X$의 열공간에 포함된다. 따라서 전체모형의 투영행렬 $H$는 $X_1$의 열공간에 속한 벡터를 그대로 두며,
>   
>   $$HX_1=X_1$$
>   
>   이다. 또한 $H$와 $H_1$은 대칭행렬이므로 위 식의 양변을 전치하면
>   
>   $$X_1^T H=X_1^T$$
>   
>   를 얻는다. 이제 $HH_1$을 계산하면,
>   
>   $$
>   HH_1=H X_1(X_1^T X_1)^{-1}X_1^T
>   =X_1(X_1^T X_1)^{-1}X_1^T=H_1.
>   $$
>   
>   같은 방식으로 $H_1H$를 계산하면,
>   
>   $$
>   H_1H=X_1(X_1^T X_1)^{-1}X_1^T H
>   =X_1(X_1^T X_1)^{-1}X_1^T=H_1.
>   $$
>   
>   즉, 부분모형의 예측공간이 전체모형의 예측공간에 포함되므로 $HH_1=H_1H=H_1$이다. 따라서 $H^2=H$, $H_1^2=H_1$ 및 이 두 관계를 전개식에 대입하면,
>   
>   $$
>   \begin{aligned}
>   (H-H_1)^2
>   &=H^2-HH_1-H_1H+H_1^2\\
>   &=H-H_1-H_1+H_1\\
>   &=H-H_1.
>   \end{aligned}
>   $$
>   
>2. 계수: $\text{rank}(H - H_1) = p - q$  
>   멱등행렬의 계수는 그 행렬의 대각합(trace)과 같으므로, $\text{rank}(H - H_1) = (p+1) - (q+1) = p - q$이다.

### 정리 6.2 분포적 성질 (Distributional Properties)

아래 모형의 가정 하에서,

$$
y = X_1 \beta_1 + \varepsilon\\
y = X_1 \beta_1 + X_2 \beta_2 + \varepsilon
$$


$\frac{1}{\sigma^2} SS(\beta_2 \mid \beta_1)$는 자유도가 $p$이고 비중심도가 $\lambda = \frac{1}{2\sigma^2} \beta_2^T X_2^T (I - H_1) X_2 \beta_2$ 인 카이제곱분포(chi-square distribution)를 따르고, 잔차제곱합 $SSE = \mathbf{y}^T(I - H)\mathbf{y}$ 와 서로 독립이다.

>**증명**

$X=(X_1,X_2)$라 놓고, $H_1$과 $H$를 각각 $\mathcal C(X_1)$과 $\mathcal C(X)$ 위로의 직교투영행렬이라고 하자.  
또한 $p=\text{rank}(X)-\text{rank}(X_1) =\text{rank}\bigl((I-H_1)X_2\bigr)$ 라고 하자.  
다음과 같이 행렬을 정의한다. $A=H-H_1,\quad B=I-H.$

1. $A=H-H_1$이 랭크 $p$인 대칭 멱등행렬임을 보인다

$\mathcal C(X_1)\subseteq\mathcal C(X)$이므로 중첩된 직교투영행렬의 성질에 따라 $HH_1=H_1H=H_1$ 이 성립한다 (정리 6.1 1번 증명내용 참고).  
$H$와 $H_1$이 대칭행렬이므로 $A=H-H_1$도 대칭행렬이다. 또한

$$
\begin{aligned}
A^2
&=(H-H_1)^2\\
&=H^2-HH_1-H_1H+H_1^2\\
&=H-H_1-H_1+H_1\\
&=A
\end{aligned}
$$

이므로 $A$는 멱등행렬이다.

또한 $\text{rank}(A) = \text{rank}(H)-\text{rank}(H_1) =p$ 이다.

한편, $AX_1=(H-H_1)X_1=X_1-X_1=0$ 이다. 그리고 $X_2$의 각 열은 $\mathcal C(X)$에 속하므로 $HX_2=X_2$ 이다. 따라서

$$
AX_2=(H-H_1)X_2=(I-H_1)X_2
$$

가 성립한다.

2. 추가제곱합의 분포를 구한다

완전모형은 $y=X_1\beta_1+X_2\beta_2+\varepsilon, \quad \varepsilon\sim N(0,\sigma^2I_n)$ 이므로 $y\sim N(\mu,\sigma^2I_n), \quad \mu=X_1\beta_1+X_2\beta_2$ 이다.

표준화된 확률벡터를 $z=\frac{y}{\sigma}$ 라고 놓으면 $z\sim N\left(\frac{\mu}{\sigma},I_n\right)$ 이다.

또한 $\frac{1}{\sigma^2}SS(\beta_2\mid\beta_1) =\frac{1}{\sigma^2}y^TAy =z^TAz$ 이다.

앞에서 $A$가 랭크 $p$인 대칭 멱등행렬임을 보였다. 따라서 정리 3.4의 3번에 의해

$$
z^TAz \sim \chi^2\left(
p,\frac{1}{2\sigma^2}\mu^TA\mu
\right)
$$

이제 비중심도를 계산한다. $AX_1=0$이므로 $A\mu =A(X_1\beta_1+X_2\beta_2) =(I-H_1)X_2\beta_2$  
$A$가 대칭 멱등행렬이므로 $\mu^TA\mu = \mu^TA^TA\mu = (A\mu)^T(A\mu)$ 이다. 따라서

$$
\begin{aligned}
\mu^TA\mu
&= \bigl((I-H_1)X_2\beta_2\bigr)^T \bigl((I-H_1)X_2\beta_2\bigr)\\
&= \beta_2^TX_2^T(I-H_1)^T(I-H_1)X_2\beta_2 \\
&= \beta_2^TX_2^T(I-H_1)X_2\beta_2
\end{aligned}
$$

따라서

$$
\boxed{
\frac{SS(\beta_2\mid\beta_1)}{\sigma^2}
\sim
\chi^2\left(
p,
\frac{1}{2\sigma^2}
\beta_2^TX_2^T(I-H_1)X_2\beta_2
\right)
}
$$

3. 추가제곱합과 잔차제곱합의 독립성을 보인다

잔차제곱합은 $SSE=y^T(I-H)y=y^TBy =(By)^T(By)$. 즉, $SSE$는 잔차벡터 $By$의 함수이다.

정리 3.5를 적용하기 위해 $B\text{Var}(y)A$를 계산한다. $\text{Var}(y)=\sigma^2I_n$이므로

$$
\begin{aligned}
B\text{Var}(y)A
&=(I-H)(\sigma^2I_n)(H-H_1)\\
&=\sigma^2(I-H)(H-H_1).
\end{aligned}
$$

그런데 $(I-H)H=H-H^2=0$ 이고, $(I-H)H_1=H_1-HH_1=0$ 이므로 $B\text{Var}(y)A =0.$ 따라서 정리 3.5에 의해 $y^TAy$ 와 $By$ 는 서로 독립이다.

$SSE=(By)^T(By)$는 $By$의 함수이므로 $y^TAy\;\perp\;SSE$ 이다. 양의 상수 $\sigma^2$로 나누어도 독립성이 유지되므로

$$
\boxed{
\frac{1}{\sigma^2}SS(\beta_2\mid\beta_1)
\;\perp\;
SSE
}
$$

특히 귀무가설 $H_0:\beta_2=0$ 아래에서는 $\lambda=0$이므로

$$
\frac{1}{\sigma^2}SS(\beta_2\mid\beta_1) \sim\chi_p^2
$$

가 된다. $\square$

### 6.1.5 부분 F-검정 (Partial F-Test)
귀무가설 $H_0: \beta_{q+1} = \cdots = \beta_p = 0$ 을 검정하기 위한 통계량은

$$
F_0 = \frac{SS(\hat{\mathbf{\beta}}_{q+1}, \dots, \hat{\mathbf{\beta}}_p \mid \hat{\mathbf{\beta}}_0, \dots, \hat{\mathbf{\beta}}_q)/(p-q)}{MSE} \\
F_0 \sim F(p-q, n-p-1)
$$

$F_0 > F_\alpha(p-q, n-p-1)$이면 귀무가설을 기각한다.

**수정항을 빼지 않는 분산분석표**  
| 요인 | 제곱합      | 자유도 | 평균제곱    | $F_0$       | $F(\alpha)$ |
| -- | -------- | --- | ----- | ------- | --------- |
| 회귀 | $SSR$ | $p+1$   | $$MSR = SSR/(p+1)$$ | $$F_0 = \frac{MSR}{MSE}$$ | $F_\alpha(p+1, n-p-1)$ |
| 잔차 | $SST - SSR$    | $n-p-1$   | $$MSE = (SST - SSR)/(n-p-1)$$   |         |           |
| 계  | $SST$     | $n$   |         |         |           |

- $SSR = SS(\hat{\mathbf{\beta}}_0, \hat{\mathbf{\beta}}_1, \dots, \hat{\mathbf{\beta}}_p) = \mathbf{\hat{\beta}}^T X^T y$: 전체모형의 회귀제곱합
- $SST = \mathbf{y^T y} = \sum y_i^2$: 총제곱합

### 추가제곱합에 관한 성질
세 개의 모수 $\beta_0, \beta_1, \beta_2$를 갖는 모형을 고려한다.

$$ 
\begin{aligned}
y &= \beta_1 x_1 + \varepsilon & \text{에서} SS(\hat\beta_1) \\
y &= \beta_0 + \beta_1 x_1 + \varepsilon & \text{에서} SS(\hat\beta_1 \mid \hat\beta_0) \\
y &= \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon & \text{에서} SS(\hat\beta_1 \mid \hat\beta_0, \hat\beta_2) \\
\end{aligned}
$$

여기에 $x_1, x_2$가 모두 중심화되어 $x_0^T x_1 = 0, \quad x_1^Tx_2=0$ 이어서 $x_1^T(I-H)x_2 = 0$이라 가정한다.

위 세개의 회귀제곱합은 일반적으로 값이 서로 다르지만, 같아지는 특수한 경우가 있다.

1. $x_0$ (절편항)과 $x_1$이 서로 직교(orthogonal)하면, 즉 $x_0^T x_1 = 0$이면 $SS(\hat\beta_1) = SS(\hat\beta_1 \mid \hat\beta_0)$
2. $x_1$과 $x_2$가 서로 직교하면,
   - $SS(\hat\beta_1 \mid \hat\beta_0) = SS(\hat\beta_1 \mid \hat\beta_0, \hat\beta_2)$
   - $SS(\hat\beta_2 \mid \hat\beta_0) = SS(\hat\beta_2 \mid \hat\beta_0, \hat\beta_1)$

이 성질이 시사하는 바는, 회귀제곱합이 설명변수의 투입 순서와 변수 사이의 상관관계에 따라 달라질 수 있다는 점이다. 따라서 개별 변수의 순수한 기여도를 해석할 때에는 이미 모형에 포함된 변수들의 효과를 통제한 추가제곱합을 사용해야 한다. 반면 설명변수들이 서로 직교하면 투입 순서와 관계없이 각 변수의 제곱합을 해석할 수 있고, 전체 회귀제곱합은 변수별 제곱합의 합으로 분해된다.

- 의의
  1. 완전모형의 전체 회귀제곱합은 변수 투입 순서와 무관하다.
  2. 상관된 설명변수 사이에서 개별 변수에 배분되는 순차적 추가제곱합은 투입 순서에 따라 달라질 수 있다.
  3. 추가제곱합은 다른 변수들을 통제한 조건부 설명량이며, 자동으로 인과적 또는 절대적인 기여도를 의미하지 않는다.
  3. 절편이 있는 모형에서는 설명변수의 직교성을 $ x_j^T(I-H_0)x_k=0 $ 으로 판단해야 한다.
  4. 설명변수들이 적절한 의미에서 서로 직교하면 변수별 제곱합은 투입 순서와 무관하고 전체 회귀제곱합은 변수별 제곱합으로 가법적으로 분해된다.
  5. 추가제곱합은 부분 $F$-검정, 중첩모형 비교 및 조건부 효과크기 계산에 유용하지만, 변수 선택의 단독 기준으로 사용하는 것은 적절하지 않다.

>**증명**
>
>1. $x_0^T x_1 = 0$
>
>직교하기 위해서는 $\sum x_i = 0$이어야 한다. 즉, $x_1$이 중심화(centering)되어야 한다. 1번모형은 절편없는 회귀 이므로
>
>$$ 
>SSR = SS(\hat\beta_1) = \hat\beta^T X^T y = \hat\beta_1\sum x_{i1} y_i \\
>= \frac{(\sum x_{i1} y_i)^2}{\sum x_{i1}^2}
>$$
>
>2번모형은 절편있는 모형의 회귀제곱합은 
>
>$$
>SSR = SS(\hat\beta_1 \mid \hat\beta_0) = \hat\beta^T X^T y - n\bar{y}^2 = \hat\beta_0\sum y_i +\hat\beta_1\sum x_{i1} y_i - n\bar{y}^2
>$$
>
>그런데 위에서 직교조건으로 $\sum x_{i1} = 0$이므로 $\bar x_1 = 0$이고  $\hat\beta_0 = \bar y$이고, $\hat\beta_1 = \frac{\sum x_{i1} y_i}{\sum x_{i1}^2}$이다. 따라서 절편있는 모형의 회귀제곱합을 다시 쓰면
>
>$$
>SS(\hat\beta_1 \mid \hat\beta_0) = \frac{(\sum x_{i1} y_i)^2}{\sum x_{i1}^2} = SS(\hat\beta_1)
>$$
>
>2. $x_1^T x_2 = 0$
>
>만약 $x_0^Tx_1=0, \quad x_0^Tx_2=0, \quad x_1^Tx_2=0$ 이라면 $x_0,x_1,x_2$가 서로 직교한다. 이때 완전모형의 최소제곱추정량은
>
>$$
>\hat\beta_0=\bar y, \qquad \hat\beta_1=\frac{x_1^Ty}{x_1^Tx_1}, \qquad \hat\beta_2=\frac{x_2^Ty}{x_2^Tx_2}
>$$
>
>절편을 제외한 회귀제곱합은
>
>$$
>\begin{aligned}
>SSR(x_1,x_2\mid x_0)
>&= \hat\beta_1x_1^Ty+\hat\beta_2x_2^Ty\\
>&= \frac{(x_1^Ty)^2}{x_1^Tx_1} + \frac{(x_2^Ty)^2}{x_2^Tx_2}.
>\end{aligned}
>$$
>
>따라서
>
>$$
>SS(\hat\beta_1\mid\hat\beta_0)
>=SS(\hat\beta_1\mid\hat\beta_0,\hat\beta_2)
>= \frac{(x_1^Ty)^2}{x_1^Tx_1} \\
>SS(\hat\beta_2\mid\hat\beta_0)
>= SS(\hat\beta_2\mid\hat\beta_0,\hat\beta_1)
>= \frac{(x_2^Ty)^2}{x_2^Tx_2}
>$$
>
>가 성립한다. 다만 앞의 중심화 조건 없이 $x_1^Tx_2=0$만 가정하면 이 결론은 일반적으로 성립하지 않는다.


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

$\mathbf{\hat{\beta}} = (X^T X)^{-1} X^T \mathbf{y}$를 구하고 이를 분할하여 $\mathbf{\hat{\beta}} = (\mathbf{\hat{\beta}}_1^T, \mathbf{\hat{\beta}}_2^T)^T$라 하자. $\mathbf{\hat{\beta}}_1 = (\hat{\beta}_0, \hat{\beta}_1, \dots, \hat{\beta}_q)^T$는 $X_1$에 대한 회귀계수 추정치이고, $\mathbf{\hat{\beta}}_2 = (\hat{\beta}_{q+1}, \dots, \hat{\beta}_p)^T$는 $X_2$에 대한 회귀계수 추정치이다.  

증명하려는 것은: 만약 $X_1^T X_2 = 0$이면, 즉 $X_1$의 모든 열과 $X_2$의 모든열이 서로 직교(orthogonal)하면,

$$
SS(\mathbf{\hat{\beta}}_1, \mathbf{\hat{\beta}}_2) = SS(\mathbf{\hat{\beta}}_1) + SS(\mathbf{\hat{\beta}}_2) \\
SS(\mathbf{\hat{\beta}}_1 \mid \mathbf{\hat{\beta}}_2) = SS(\mathbf{\hat{\beta}}_1)\\
SS(\mathbf{\hat{\beta}}_2 \mid \mathbf{\hat{\beta}}_1) = SS(\mathbf{\hat{\beta}}_2)
$$

이다.

>**증명**
>
>중회귀모형을 $\mathbf{y} = X\beta + \varepsilon$로 표현하자. 
>
>$$
>\begin {pmatrix}
>X_1^T X_1 & X_1^T X_2 \\
>X_2^T X_1 & X_2^T X_2
>\end{pmatrix} 
>\begin {pmatrix}
>\mathbf{\hat{\beta}_1} \\
>\mathbf{\hat{\beta}_2}
>\end{pmatrix}
>= \begin {pmatrix}
>X_1^T \mathbf{y} \\
>X_2^T \mathbf{y}
>\end{pmatrix}
>$$
>
>만약 $X_1^T X_2 = 0$이면, 정규방정식은 다음과 같이 분리된다.
>
>$$
>\begin {pmatrix}
>\mathbf{\hat{\beta}_1} \\
>\mathbf{\hat{\beta}_2}
>\end{pmatrix}
>= \begin {pmatrix}
>(X_1^T X_1)^{-1} O_{(p-q)\times (q+1)} \\
>O_{(q+1)\times (p-q)} (X_2^T X_2)^{-1}
>\end{pmatrix}^{-1}
>\begin {pmatrix}
>X_1^T \mathbf{y} \\
>X_2^T \mathbf{y}
>\end{pmatrix}
>= \begin {pmatrix}
>(X_1^T X_1)^{-1} X_1^T \mathbf{y} \\
>(X_2^T X_2)^{-1} X_2^T \mathbf{y}
>\end{pmatrix}
>$$
>
>이 되어 $\mathbf{\hat{\beta}}_1$과 $\mathbf{\hat{\beta}}_2$가 서로 독립적으로 추정되고 이것은 회귀모형을 개별로 적합시켜 얻은 추정과 같다.  
>
>또한 $SS(\mathbf{\hat{\beta}}_1) = \mathbf{\hat{\beta}}_1^T X_1^T y$ 이고, $SS(\mathbf{\hat{\beta}}_2) = \mathbf{\hat{\beta}}_2^T X_2^T y$ 임을 확인할 수 있다. 따라서
> 
>$$ SS(\mathbf{\hat{\beta}}) = SS(\mathbf{\hat{\beta}}_1, \mathbf{\hat{\beta}}_2) 
>= \mathbf{\hat{\beta}}^T X^T y = \mathbf{\hat{\beta}}_1^T X_1^T y + \mathbf{\hat{\beta}}_2^T X_2^T y = SS(\mathbf{\hat{\beta}}_1) + SS(\mathbf{\hat{\beta}}_2) $$
>
>이므로, 첫번째 식이 증명된다. 또한,
>
>$$SS(\mathbf{\hat{\beta}}_1 \mid \mathbf{\hat{\beta}}_2) =  SS(\mathbf{\hat{\beta}}_1, \mathbf{\hat{\beta}}_2) - SS(\mathbf{\hat{\beta}}_2) = SS(\mathbf{\hat{\beta}}_1) \\
>
>SS(\mathbf{\hat{\beta}}_2 \mid \mathbf{\hat{\beta}}_1) =  SS(\mathbf{\hat{\beta}}_1, \mathbf{\hat{\beta}}_2)-SS(\mathbf{\hat{\beta}}_1) = SS(\mathbf{\hat{\beta}}_2)$$
>
>로 나머지 식들도 성립한다.  

위 결과들은 $X_1$과 $X_2$가 서로 직교할 때만 성립한다. 만약 $X_1^T X_2 \ne 0$이면, 즉 $X_1$과 $X_2$가 서로 직교하지 않으면, $\mathbf{\hat{\beta}}_1$과 $\mathbf{\hat{\beta}}_2$는 서로 독립적으로 추정되지 않고, 회귀제곱합도 분해되지 않는다.

유의할 사항은 $X_1$과 $X_2$가 서로 직교하면 충분하지 $X_1$내의 열들끼리는 직교할 필요는 없다는 것이다. $X_1$과 $X_2$가 서로 직교하기만 하면, $X_1$내의 열들끼리는 서로 직교하지 않아도 된다. $X_2$내의 열들끼리도 서로 직교하지 않아도 된다. 

위 결과는 설계행렬을 두 블록이 아니라 $q$개의 블록으로 분할해도, 서로 다른 블록의 열공간이 쌍별로 직교하면 회귀제곱합과 추가제곱합이 블록별로 분해된다.

**증명**

전체모형의 정규방정식 $X^TX\hat\beta=X^Ty$ 이고, 쌍별 직교조건에 의해 $X_i^TX_j=0, \quad i\neq j$ 이므로

$$
X^TX =
\begin{pmatrix}
X_1^TX_1 & 0 & \cdots & 0\\
0 & X_2^TX_2 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & X_q^TX_q
\end{pmatrix}.
$$

따라서 정규방정식은 블록별로 분리된다: $X_j^TX_j\hat\beta_j=X_j^Ty, \quad j=1,\ldots,q.$ 그러므로 $\hat\beta_j = (X_j^TX_j)^{-1}X_j^Ty$ 이다.

따라서 각 블록의 회귀계수 추정량은 다른 블록의 포함 여부와 무관하다.

그리고 블록별 투영행렬의 직교성: $H_iH_j=H_jH_i=0, \quad i\neq j$ 이 성립함을 간단한 연산으로 보일 수 있다. 이는 각 투영행렬의 열공간이 서로 직교한다는 것을 의미한다.

전체 투영행렬의 분해:

다음 행렬을 생각하자. $H_\ast=\sum_{j=1}^qH_j.$ 각 $H_j$가 대칭행렬이므로

$$
H_\ast^T = \sum_{j=1}^qH_j^T = \sum_{j=1}^qH_j = H_\ast
$$

이다. 따라서 $H_\ast$는 대칭행렬이다.

또한, 각 $H_j$가 멱등행렬이고 $i\neq j$일 때 $H_iH_j=0$이므로

$$
H_\ast^2 = \left(\sum_{i=1}^qH_i\right) \left(\sum_{j=1}^qH_j\right) = \sum_{j=1}^qH_j^2 + \sum_{i\neq j}H_iH_j =H_\ast
$$

따라서 $H_\ast$는 대칭 멱등행렬이다.

한편, 쌍별 직교성에 의해 전체 설계행렬의 열공간은 직교직합으로 표현된다.

$$
\mathcal C(X) = \mathcal C(X_1) \oplus \mathcal C(X_2) \oplus\cdots\oplus \mathcal C(X_q).
$$

$H_\ast$는 바로 이 열공간 위로의 직교투영행렬이다. 직교투영행렬은 유일하므로

$$
\boxed{H = H_\ast = \sum_{j=1}^qH_j }
$$

한편, 전체모형의 수정하지 않은 회귀제곱합은 $SS(\hat\beta_1,\ldots,\hat\beta_q) = y^THy$ 이다.

앞에서 구한 투영행렬의 분해를 적용하면

$$
\begin{aligned}
SS(\hat\beta_1,\ldots,\hat\beta_q)
&=
y^T\left(\sum_{j=1}^qH_j\right)y\\
&=
\sum_{j=1}^qy^TH_jy.
\end{aligned}
$$

각 블록의 회귀제곱합을 $SS(\hat\beta_j)=y^TH_jy$ 라고 정의하면

$$
\begin{aligned}
SS(\hat\beta_j)
&=y^TH_jy\\
&=
y^TX_j(X_j^TX_j)^{-1}X_j^Ty\\
&=
\hat\beta_j^TX_j^Ty\\
&=
\hat\beta_j^TX_j^TX_j\hat\beta_j.
\end{aligned}
$$

따라서 전체 회귀제곱합은 각 블록에 의해 설명되는 제곱합의 합이다.

임의의 $j$를 고정하고, $X_j$를 제외한 나머지 모든 블록을

$$
X_{-j} =
\begin{pmatrix}
X_1&\cdots&X_{j-1}&X_{j+1}&\cdots&X_q
\end{pmatrix}
$$

라고 하자.

$X_{-j}$에 대한 투영행렬은 블록들의 직교성에 의해

$$
H_{-j} = \sum_{\substack{k=1\\k\neq j}}^qH_k
$$

이다. 전체모형의 투영행렬은 $H = \sum_{k=1}^qH_k$ 이므로

$$
H-H_{-j} = \sum_{k=1}^qH_k - \sum_{\substack{k=1\\k\neq j}}^qH_k =H_j.
$$

따라서 나머지 모든 블록이 이미 포함된 상태에서 $X_j$를 추가할 때의 추가제곱합은

$$
SS(\hat\beta_j\mid\hat\beta_{-j}) = y^T(H-H_{-j})y =y^TH_jy =SS(\hat\beta_j) 
$$

즉,

$$
\boxed{
SS(\hat\beta_j\mid
\hat\beta_1,\ldots,\hat\beta_{j-1},
\hat\beta_{j+1},\ldots,\hat\beta_q)
= SS(\hat\beta_j)
}
$$

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
  1. 모든 $j$에 대해 부분 $F$-검정을 수행하여 유의하지 않은 변수를 제거한다.
  2. 제거된 변수들을 제외한 모형에 대하여 다시 부분 $F$-검정을 수행한다.
  3. 더 이상 제거할 변수가 없을 때까지 1과 2를 반복한다.
  - 더 자세한건 변수선택(selection of variables) 참고

이는 5장에서 소개한 일반선형가설(general linear hypothesis) $H_0 : C\beta = 0$의 특수한 경우이다.

### 6.2.2 완전모형과 축소모형 비교 (Full vs Reduced Model)
완전모형(full model): $y = \beta_0 + \sum_{j=1}^{p} \beta_j x_j + \varepsilon$  
축소모형(reduced model): $y = \alpha_0 + \sum_{k \ne j} \alpha_k x_k + \varepsilon$

$SST = SSR_F + SSE_F = SSR_R + SSE_R$이므로,

$$
SSR_F - SSR_R = SSE_R - SSE_F = SS(\hat{\beta}_j \mid \text{others})
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

자유도는 1이고 MSE는 완전모형 $y_i = \beta_0 + \beta_1 x_{1i} + \beta_i x_{ij} + \varepsilon_i$에 대한 잔차제곱합(SSE)를 자유도 $n-3$으로 나눈 것이다. 따라서 검정통계량은

$$
F_0 = \frac{SS(\hat{\beta}_j \mid \text{previous})}{MSE}
$$

- $F$의 기각치는 $F_\alpha(1, n-3)$이다.

특징
- 만약 $\beta_j$열과 $\beta_0, \beta_1$열이 서로 직교한다면 $SS(\hat\beta_j \mid \hat\beta_0, \hat\beta_1)$는 $SS(\hat\beta_j)$와 같아지므로 $SSR = SS(\hat\beta_j \mid \hat\beta_0, \hat\beta_1)$가 된다.
  - $x_1$을 먼저 넣든 $x_j$를 먼저 넣든 각 변수의 제곱합이 변하지 않는다. 따라서 축차제곱합과 부분제곱합이 일치한다.
  - 다른 변수의 포함 여부가 회귀계수를 바꾸지 않는다
  - 제곱합을 변수별로 명확하게 분해할 수 있다
  - 직교하면 추가제곱합을 완전모형과 축소모형을 각각 적합하여 빼지 않고 직접 계산할 수 있다
* 각 단계마다 완전모형이 달라지며 따라서 MSE 값도 단계마다 변한다.
* 그런데 부분 F검정이 F검정으로서 타당성을 지니기 위해서는 MSE가 오차항의 분산에 대한 불편추정량이 되어야 한다
  - 더 엄밀하게는 $\frac1{\sigma^2} SSE$가 $\chi^2$분포를 따르고 이의 비중심도는 0이어야 한다
* 따라서 완전모형으로 사용하는 모형이 충분히 큰 모형이어서 자료의 true model을 포함하고 있지 않으면 축차F검정을 적용할 수 없다

## 6.3 변수의 표준화 (Standardization of Variables)
중회귀모형 $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \varepsilon$ 에서 설명변수들의 단위가 서로 다를 경우 회귀계수의 상대적 크기를 직접 비교하기 어렵다. 이를 해결하기 위하여 변수의 표준화(standardization of variables)를 수행한다.

### 6.3.1 평균 중심화 (Centering)
먼저 각 설명변수의 평균을 $\bar{x}_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij}$라 하고 다음과 같이 모형을 변형한다.

$$
y = (\beta_0 + \beta_1 \bar{x}_1 + \cdots + \beta_p \bar{x}_p) + \beta_1(x_1 - \bar{x}_1) + \cdots + \beta_p(x_p - \bar{x}_p) + \varepsilon
$$

여기서

$$
\beta_0' = \beta_0 + \sum_{j=1}^{p} \beta_j \bar{x}_j, \quad w_{ij} = x_{ij} - \bar{x}_j
$$

라 두면,

$$
y = \beta_0' + \beta_1 w_1 + \cdots + \beta_p w_p + \varepsilon
$$

이 된다. 이 식으로 정규방정식(normal equations)을 만들면,

$$
n\hat{\beta}_0' + \hat{\beta}_1 \sum w_{i1} + \cdots + \hat{\beta}_p \sum w_{ip} = \sum y_i
$$

인데 $\bar{w}_j = \frac{1}{n} \sum_i (x_{ij} - \bar{x}_j) = 0$ 이므로 다 0이 되어, $\hat{\beta}_0' = \bar{y}$ 가 된다. 이는 $\hat{\beta}_1, \hat{\beta}_2, \dots, \hat{\beta}_p$값이 뭐든지 간에 $\hat{\beta}_0'$는 항상 $\bar{y}$가 된다는 것을 의미한다. 따라서 중심화된 모형은

$$
y - \bar{y} = \beta_1 w_1 + \cdots + \beta_p w_p + \varepsilon'
$$

로 쓸 수 있으며, 상수항 없이 회귀를 수행할 수 있다. 이 경우 설계행렬 $X$의 열 개수는 $p+1$에서 $p$로 줄어들어 계산량이 감소한다.

### 6.3.2 분산·공분산 행렬 표현 (Cross-Product Matrix)
단순회귀에서는 $y = \beta_0' + \beta_1(x-\bar x) + \varepsilon$ 형태이고, 단순회귀 추정에서는 $\hat y - \bar y = \hat \beta_1(x-\bar x)$ 형태다.

중심화 변수에 대해

$$
\sum_i w_{ij} w_{il} = \sum_i (x_{ij} - \bar{x}_j)(x_{il} - \bar{x}_l) = S_{jl}
$$

라 하면, 정규방정식의 $X^TX$행렬은

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
각 변수에 대해 변수변환을 하자.

$$
Z_{ij} = \frac{w_{ij}}{\sqrt{S_{jj}}} = \frac{x_{ij} - \bar{x}_j}{\sqrt{S_{jj}}}, \qquad j=1,\dots,p \\
y_i^* = \frac{y_i - \bar{y}}{\sqrt{S_{yy}}}, \qquad S_{yy} = \sum (y_i - \bar{y})^2
$$

로 정의하면, 표준화된 회귀모형은 

$$
\begin{aligned}
&y - \bar{y} = \beta_1 w_1 + \cdots + \beta_p w_p + \varepsilon' \\
&\Rightarrow y_i^*\sqrt{S_{yy}} = \beta_1\sqrt{S_{11}} Z_1 + \beta_2\sqrt{S_{2}} Z_2 + \cdots + \beta_p\sqrt{S_{pp}} Z_p + \varepsilon^*\\
&\Rightarrow y_i^* = a_1 Z_{i1} + a_2 Z_{i2} + \cdots + a_p Z_{ip} + \varepsilon^*, \quad a_j = \beta_j \sqrt{\frac{S_{jj}}{S_{yy}}}\\
&\Rightarrow y^* = a_1 Z_1 + a_2 Z_2 + \cdots + a_p Z_p + \varepsilon^*
\end{aligned}
$$

로 쓸 수 있다.  
변수 $Z_j$는 $x_j$의 표준화된 버전이며, $y^*$는 $y$의 표준화된 버전이다. $a_j$는 표준화 회귀계수(standardized regression coefficient)라고 불린다.

이때 $\sum_i Z_{ij}^2 = 1, \ (j=1,2, \cdots, p)$ 이고, $\sum_i (y_i^*)^2 = 1$ 이다.

### 6.3.4 상관행렬 표현 (Correlation Matrix Form)
이 표준화된 회귀모형은 재미있는 성질을 가지고 있다.  

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

는 설명변수들의 상관행렬(correlation matrix)이 되어 설명변수 $x_1, x_2, \dots, x_p$의 상관구조를 모두 알 수 있다. 이와같은 $X^T X$를 변수$Z$의 상관행렬(correlation matrix)이라고도 한다.

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

는 $y$와 각 설명변수 $x_j$의 상관계수를 나타내 준다.

따라서 표준화된 방정식의 계수 추정값을 $\hat a_j$라 할때,

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

### 6.3.6 원래 계수로의 환원 (Back Transformation)

여기서 얻은 $\hat{a}_j$로 원래의 회귀모형의 계수인 $\hat\beta_j$를 구하려면 $\hat{\beta}_j = \hat{a}_j \sqrt{S_{yy}/S_{jj}}, \quad (j=1,2, \cdots, p)$를 사용한다.  

$\hat \beta_0$ 은 $\hat\beta_0 = \bar y-\hat\beta_1\bar x_1-\hat\beta_2\bar x_2 - \cdots -\hat\beta_p\bar x_p$ 로 구한다:  
$\hat{\beta}_0' = \bar{y}$ 이므로 $\beta_0' = \beta_0 + \sum_{j=1}^{p} \beta_j \bar{x}_j$ 로부터 유도됨.

**다중공선성(Multicollinearity)**  
상관행렬을 $D$라 할 때, 변수간의 상관관계가 1에 가까워지면 계수 추정의 분산이 매우 커진다. 이는 다중공선성(multicollinearity) 문제를 의미한다. 행렬식이 0에 가까우면 회귀계수 추정에 매우 유의해야 한다.


## 6.4 공동신뢰영역 (Joint Confidence Region)

이 절에서는 각 계수에 대한 개별 신뢰구간(individual confidence interval)이 아니라, 공동신뢰영역(joint confidence region)을 유도한다.  
> 공동신뢰영역: 크기가 $p+1$인 벡터 $\mathbf{\beta}$ 전체를 동시에 포함할 확률이 $100(1-\alpha)\%$ 인 영역

### 6.4.1 기본 분포이론 (Distributional Result)
중회귀모형 $\mathbf{y} = \mathbf{X}\mathbf{\beta} + \varepsilon, \quad \varepsilon \sim N(0, \sigma^2 I_n)$을 가정한다.

최소제곱추정량은 $\hat{\mathbf{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$이며, $\hat{\mathbf{\beta}} \sim N\bigl(\mathbf{\beta}, \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}\bigr)$을 따른다.  

또한, $\hat{\mathbf{\beta}} - \mathbf{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T (\mathbf{y} - \mathbf X\mathbf \beta)= (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \varepsilon$이므로

$$
\frac{1}{\sigma^2}
(\hat{\mathbf{\beta}}-\mathbf{\beta})^T
\mathbf{X}^T \mathbf{X}
(\hat{\mathbf{\beta}}-\mathbf{\beta})
= \frac{1}{\sigma^2}
\varepsilon^T \mathbf{H} \varepsilon
$$

여기서 $\mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T$는 hat matrix이다.

이는 정리 3.3에 의해

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

또한 위 두 통계량은 서로 독립이다 (정리 3.6로 증명 가능)

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
+ 2(\sum x_i)(\beta_0-\hat{\beta}_0)(\beta_1-\hat{\beta}_1) \\
\le 2\,\text{MSE}\,F_{\alpha}(2, n-2)
$$

이다. 이 식은 $\beta$로 표현되는 타원방정식으로, 타원형 속이 신뢰영역이다.  
재밌는점은, 공분산행렬은 $\text{Var}(\hat{\mathbf{\beta}}) = \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}$이고, 위 식을 $D = n\sum x_i^2 - (\sum x_i)^2$로 나누면,

$$
(\mathbf{X}^T \mathbf{X})^{-1}
= \frac{1}{D}
\begin{pmatrix}
\sum x_i^2 & -\sum x_i \\
-\sum x_i & n
\end{pmatrix}
$$

이므로, 타원방정식은 아래처럼 변형된다

$$
Var(\hat{\mathbf{\beta}}_1)(\beta_0-\hat\beta_0)^2 + Var(\hat{\mathbf{\beta}}_0)(\beta_1-\hat\beta_1)^2 - 2 \text{Cov}(\hat{\mathbf{\beta}}_0,\hat{\mathbf{\beta}}_1)(\beta_0-\hat\beta_0)(\beta_1-\hat\beta_1) \\
\le 2\cdot \text{MSE}\cdot F_{\alpha}(2, n-2) / D \cdot \sigma^2
$$

이때
 
$$
\text{Var}(\hat{\mathbf{\beta}}_0) = \sigma^2 \frac{\sum x_i^2}{D},\quad
\text{Var}(\hat{\mathbf{\beta}}_1) = \sigma^2 \frac{n}{D},\quad
\text{Cov}(\hat{\mathbf{\beta}}_0,\hat{\mathbf{\beta}}_1) = -\sigma^2 \frac{\sum x_i}{D}
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
\hat{\mathbf{\beta}}_j \pm c \sqrt{\text{Var}(\hat{\mathbf{\beta}}_j)}
$$

을 구성하되, 모든 구간이 동시에 참모수를 포함할 확률이 $1-\alpha$가 되도록 상수 $c$를 조정하는 방법이다.
- 각 모수들에 대해 '같은 수준'의 신뢰구간들을 계산
- 이들의 cartesian product가 모수들을 동시에 포함할 확률이 $1-\alpha$가 되게 조정
- 따라서 동시신뢰구간은 항상 각 모수들의 신뢰구간들의 곱집합은 $p+1$차원 입방체의 형태로 표현됨
- 본페로니 방법, 쉐페 방법 등이 존재

#### (1) Bonferroni 방법

$$

A_j = \left\{\beta_j: \hat{\mathbf{\beta}}_j - t_{\alpha/(2(p+1))}(n-p-1) \sqrt{\widehat{\text{Var}}(\hat{\mathbf{\beta}}_j)} \le \beta_j \le \hat{\mathbf{\beta}}_j + t_{\alpha/(2(p+1))}(n-p-1) \sqrt{\widehat{\text{Var}}(\hat{\mathbf{\beta}}_j)} \right\} \\
= \left[ \hat{\mathbf{\beta}}_j - t_{\alpha/(2(p+1))}(n-p-1) \sqrt{\widehat{\text{Var}}(\hat{\mathbf{\beta}}_j)},\ \hat{\mathbf{\beta}}_j + t_{\alpha/(2(p+1))}(n-p-1) \sqrt{\widehat{\text{Var}}(\hat{\mathbf{\beta}}_j)} \right]
$$

이렇게 계산된 입방체의 확률은

$$
P(A_0 \times A_1 \times \cdots \times A_p) = P\left(\bigcap_{j=0}^{p}A_j\right) \\
\ge 1 - \sum_{j=0}^{p} P(A_j^c) = 1 - (p+1) \cdot \frac{\alpha}{p+1} =
1-\alpha
$$

를 만족한다. 단순선형회귀의 경우, 동시신뢰구간은 $\beta_0$의 신과구간, $\beta_1$신뢰구간을 나타내는 사각형이 된다. 

#### (2) Scheffé 방법
모든 선형결합 $\psi = a^T \beta$ 에 대해

$$
\Psi = \left\{\psi = \mathbf{a}^T \mathbf{\beta}, \forall \mathbf{a} \in \mathbb{R}^{p+1} \right\}
$$

의 형태로 구간을 구성한다. $\mathbf a_i$는 $j$번째 성분만 1이고 나머지는 0인 벡터.

각 모수$\psi_j$에 대한 신뢰구간은

$$
\mathbf{a}^\top \hat{\mathbf{\beta}} \pm \sqrt{(p+1)F_{\alpha}(p+1,n-p-1)}
\sqrt{\mathbf{a}^T \text{Var}(\hat{\mathbf{\beta}}) \mathbf{a}}
$$

여기서 각 모수의 신뢰수준은 $\mathbf{a}_i$의 선택과 무관하게 항상 일정하다. 이는 동시신뢰구간의 경우 '대전제'임을 기억할 필요가 있다. 다만 p가 커짐에 따라 두 방법등의 동시신뢰구간 추정방법 등은 신뢰구간이 크게 넓어진다.

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
SSE_F
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

$$SSE_R = (y - \mathbf{X}\hat{\mathbf{\beta}}_0)^T (y - \mathbf{X}\hat{\mathbf{\beta}}_0)$$

자유도는
$n - (p+1)$

>3장에서 사용한 검정절차:  
>순서1: 완정모형 적합, 잔차제곱합 SSE_F 계산  
>순서2: 축소모형 적합, 잔차제곱합 SSE_R 계산  
>순서3: 검정통계량 계산 및 판정  
### 6.5.3 검정통계량 (Test Statistic)
정규성 가정하에서

$$
\frac{SSE_R - SSE_F}{\sigma^2} \sim \chi^2(p+1) \\
\frac{SSE_F}{\sigma^2} \sim \chi^2(n - 2(p+1))
$$

이며 두 통계량은 독립이다. 따라서 검정통계량은

$$
F_0 = \frac{[SSE_R - SSE_F]/(p+1)}{SSE_F/(n - 2(p+1))} \\
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
SSE_F
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
SSE_R = (\mathbf{y} - \mathbf{X}\hat{\mathbf{\beta}}_0)^T (\mathbf{y} - \mathbf{X}\hat{\mathbf{\beta}}_0)
$$

#### (3) 검정통계량

$$
F_0 = \frac{[SSE_R - SSE_F] / [(k-1)(p+1)]}{SSE_F / [n - k(p+1)]} \\
F_0 \sim F\big((k-1)(p+1), n - k(p+1)\big)
$$

### 6.5.6 해석
* 이 검정은 회귀계수 벡터 전체의 동일성을 검정한다.
* 본질적으로 추가제곱합(extra sum of squares)에 기반한 일반선형가설 검정이다.
* 설계행렬을 확장하여 하나의 큰 모형으로 표현할 수 있다.
* 집단 간 차이는 상호작용항(interaction term) 검정과 동일한 구조를 가진다.
