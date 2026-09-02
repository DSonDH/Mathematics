아래에서는 잔차를 $e_i$, 모형의 오차항을 $\varepsilon_i$로 구분한다. 가장 주의할 점은 **SSR이 교재마다 서로 다른 의미로 쓰일 수 있다는 점**이다.

## 1. 주요 용어

| 기호    | 영어                        | 의미           | 공식                                    |
| ----- | ------------------------- | ------------ | ------------------------------------- |
| $SST$ | Total Sum of Squares      | 전체 변동        | $\sum_i(y_i-\bar y)^2$                |
| $SSR$ | Regression Sum of Squares | 모형이 설명한 변동   | $\sum_i(\hat y_i-\bar y)^2$           |
| $SSE$ | Error Sum of Squares      | 잔차 변동        | $\sum_i(y_i-\hat y_i)^2=\sum_i e_i^2$ |
| $MSR$ | Mean Square Regression    | 자유도당 설명 변동   | $SSR/p$                               |
| $MSE$ | Mean Square Error         | 자유도당 잔차 변동   | $SSE/(n-p-1)$                         |
| $SSB$ | Sum of Squares Between    | 집단 간 변동      | $\sum_jn_j(\bar y_j-\bar y)^2$        |
| $SSW$ | Sum of Squares Within     | 집단 내 변동      | $\sum_j\sum_i(y_{ij}-\bar y_j)^2$     |
| $MSB$ | Mean Square Between       | 자유도당 집단 간 변동 | $SSB/(k-1)$                           |
| $MSW$ | Mean Square Within        | 자유도당 집단 내 변동 | $SSW/(n-k)$                           |

여기서 다음과 같이 정의한다.

* $n$: 전체 관측값 개수다.
* $p$: 절편을 제외한 설명변수 개수다.
* $k$: 집단 개수다.
* $e_i=y_i-\hat y_i$: 잔차다.
* $\varepsilon_i$: 모집단 회귀모형의 오차항이다.

---

## 2. 회귀분석에서의 변동 분해

절편을 포함한 OLS 회귀모형에서

$$
y_i=\hat y_i+e_i
$$

이므로

$$
y_i-\bar y=(\hat y_i-\bar y)+e_i
$$

이다. 양변을 제곱하여 더하면

$$
\begin{aligned}
\sum_{i=1}^n(y_i-\bar y)^2
&=
\sum_{i=1}^n(\hat y_i-\bar y)^2
+\sum_{i=1}^ne_i^2 \\
&\quad+
2\sum_{i=1}^n(\hat y_i-\bar y)e_i
\end{aligned}
$$

가 된다.

OLS에서는 적합값과 잔차가 직교하므로

$$
\sum_{i=1}^n(\hat y_i-\bar y)e_i=0
$$

이다. 따라서

$$
\boxed{SST=SSR+SSE}
$$

가 성립한다. 구체적으로는

$$
\boxed{
\sum_{i=1}^n(y_i-\bar y)^2 = \sum_{i=1}^n(\hat y_i-\bar y)^2
+
\sum_{i=1}^n(y_i-\hat y_i)^2
}
$$

이다.

각 항의 의미는 다음과 같다.

* $SST$: 관측값들이 전체 평균 $\bar y$로부터 얼마나 흩어져 있는지를 나타낸다.
* $SSR$: 전체 변동 중 회귀모형이 설명한 부분이다.
* $SSE$: 회귀모형이 설명하지 못하고 잔차로 남은 부분이다.

---

## 3. Cross term이 0이 되는 이유

다중회귀모형을 행렬로 나타내면

$$
\mathbf y=X\hat{\boldsymbol\beta}+\mathbf e
$$

이다. OLS 정규방정식에 의해

$$
X^\top\mathbf e=\mathbf 0
$$

가 성립한다.

적합값은

$$
\hat{\mathbf y}=X\hat{\boldsymbol\beta}
$$

이므로

$$
\begin{aligned}
\hat{\mathbf y}^{\top}\mathbf e
&=
(X\hat{\boldsymbol\beta})^\top\mathbf e\\
&=
\hat{\boldsymbol\beta}^{\top}X^\top\mathbf e\\
&=0
\end{aligned}
$$

이다.

또한 절편을 포함한 OLS에서는

$$
\sum_{i=1}^ne_i=0
$$

이므로

$$
\begin{aligned}
\sum_{i=1}^n(\hat y_i-\bar y)e_i
&=
\sum_{i=1}^n\hat y_ie_i
-\bar y\sum_{i=1}^ne_i\\
&=0-0\\
&=0
\end{aligned}
$$

이다.

이것은 오차항 $\varepsilon_i$에 대한 확률적 가정이 아니라, OLS로 계산된 **잔차 $e_i$의 대수적 성질**이다.

---

## 4. 회귀분석의 자유도와 평균제곱

설명변수가 $p$개이고 절편이 포함된 다중회귀모형에서는 다음과 같다.

| 변동 |   제곱합 |     자유도 |              평균제곱 |
| -- | ----: | ------: | ----------------: |
| 회귀 | $SSR$ |     $p$ |       $MSR=SSR/p$ |
| 오차 | $SSE$ | $n-p-1$ | $MSE=SSE/(n-p-1)$ |
| 전체 | $SST$ |   $n-1$ |       $SST/(n-1)$ |

자유도 역시

$$
\boxed{n-1=p+(n-p-1)}
$$

로 분해된다.

단순선형회귀에서는 설명변수가 하나이므로 $p=1$이다. 따라서

$$
MSR=\frac{SSR}{1}=SSR
$$

이고

$$
MSE=\frac{SSE}{n-2}
$$

이다.

---

## 5. MSE의 의미

모집단 회귀모형을

$$
y_i = \beta_0+\beta_1x_{i1}+\cdots+\beta_px_{ip}
+\varepsilon_i
$$

라고 하자. 일반적인 가정은

$$
E(\varepsilon_i\mid X)=0
$$

과

$$
\text{Var}(\varepsilon_i\mid X)=\sigma^2
$$

이다. 오차항들이 서로 비상관이라는 가정까지 만족하면

$$
\boxed{E(MSE\mid X)=\sigma^2}
$$

가 성립한다.

따라서

$$
\boxed{
MSE=\frac{SSE}{n-p-1}
=\frac{\sum_{i=1}^ne_i^2}{n-p-1}
}
$$

는 오차항 분산 $\sigma^2$의 불편추정량이다.

오차항의 표준편차는 다음과 같이 추정한다.

$$
\boxed{\hat\sigma=\sqrt{MSE}}
$$

이를 residual standard error 또는 regression standard error라고 부른다.

### 머신러닝의 MSE와 차이

머신러닝에서는 흔히

$$
MSE_{\mathrm{ML}} = \frac{1}{n}\sum_{i=1}^n(y_i-\hat y_i)^2 = \frac{SSE}{n}
$$

을 MSE라고 부른다.

반면 고전적 회귀분석의 ANOVA 표에서는 일반적으로

$$
\boxed{
MSE=\frac{SSE}{n-p-1}
}
$$

을 의미한다. 두 용어가 같은 이름을 사용하지만 분모가 다르므로 문맥을 확인해야 한다.

---

## 6. MSR과 회귀모형의 $F$-검정

다중회귀모형의 전체 유의성 검정은 다음 가설을 검정한다.

$$
H_0:\beta_1=\beta_2=\cdots=\beta_p=0
$$

대

$$
H_1:\text{적어도 하나의 }\beta_j\neq0
$$

이다.

검정통계량은

$$
\boxed{
F=\frac{MSR}{MSE} = \frac{SSR/p}{SSE/(n-p-1)}
}
$$

이다.

귀무가설과 정규성 가정 아래에서

$$
F\sim F_{p,n-p-1}
$$

이다.

귀무가설이 참이면

$$
E(MSR\mid X)=\sigma^2
$$

이고

$$
E(MSE\mid X)=\sigma^2
$$

이다. 따라서 귀무가설이 참일 때 $MSR$과 $MSE$는 모두 같은 오차분산 $\sigma^2$을 추정한다.

반면 회귀계수 중 하나라도 $0$이 아니라면 $MSR$에는 설명변수에 의한 신호가 추가된다. 따라서 $MSR/MSE$가 충분히 크면 귀무가설을 기각한다.

단순선형회귀에서는

$$
\boxed{F=t^2}
$$

이 성립한다. 여기서 $t$는 다음 가설을 검정하는 통계량이다.

$$
H_0:\beta_1=0
$$

---

## 7. 단순선형회귀에서 $E(SSR)$

단순선형회귀의 적합값은

$$
\hat y_i=\hat\beta_0+\hat\beta_1x_i
$$

이다. 절편을 포함한 OLS에서는

$$
\bar y=\overline{\hat y}
=\hat\beta_0+\hat\beta_1\bar x
$$

이므로

$$
\begin{aligned}
\hat y_i-\bar y
&=
(\hat\beta_0+\hat\beta_1x_i)
-(\hat\beta_0+\hat\beta_1\bar x)\\
&=
\hat\beta_1(x_i-\bar x)
\end{aligned}
$$

이다.

따라서

$$
\begin{aligned}
SSR
&=
\sum_{i=1}^n(\hat y_i-\bar y)^2\\
&=
\sum_{i=1}^n
\left\{\hat\beta_1(x_i-\bar x)\right\}^2\\
&=
\hat\beta_1^2
\sum_{i=1}^n(x_i-\bar x)^2
\end{aligned}
$$

이다.

다음과 같이 정의하면

$$
S_{xx}=\sum_{i=1}^n(x_i-\bar x)^2
$$

다음 관계를 얻는다.

$$
\boxed{SSR=\hat\beta_1^2S_{xx}}
$$

한편,

$$
E(\hat\beta_1\mid X)=\beta_1
$$

이고

$$
\text{Var}(\hat\beta_1\mid X) = \frac{\sigma^2}{S_{xx}}
$$

이다. 확률변수 $Z$에 대해

$$
E(Z^2)=\text{Var}(Z)+\{E(Z)\}^2
$$

이므로

$$
\begin{aligned}
E(\hat\beta_1^2\mid X)
&=
\text{Var}(\hat\beta_1\mid X)
+\{E(\hat\beta_1\mid X)\}^2\\
&=
\frac{\sigma^2}{S_{xx}}+\beta_1^2
\end{aligned}
$$

이다.

따라서

$$
\begin{aligned}
E(SSR\mid X)
&=
S_{xx}E(\hat\beta_1^2\mid X)\\
&=
S_{xx}
\left(
\frac{\sigma^2}{S_{xx}}+\beta_1^2
\right)\\
&=
\sigma^2+\beta_1^2S_{xx}
\end{aligned}
$$

이다. 즉,

$$
\boxed{
E(SSR\mid X) = \sigma^2+\beta_1^2S_{xx}
}
$$

이다.

단순선형회귀에서는 회귀 자유도가 $1$이므로

$$
MSR=SSR
$$

이다. 따라서

$$
\boxed{
E(MSR\mid X) = \sigma^2+\beta_1^2S_{xx}
}
$$

이다.

특히 귀무가설 $H_0:\beta_1=0$ 아래에서는

$$
\boxed{
E(MSR\mid X)=\sigma^2
}
$$

가 된다.

---

## 8. 결정계수 $R^2$

전체 변동 중 회귀모형이 설명한 비율은 결정계수로 나타낸다.

$$
\boxed{
R^2 = \frac{SSR}{SST} = 1-\frac{SSE}{SST}
}
$$

절편을 포함한 OLS에서는

$$
0\leq R^2\leq1
$$

이다.

* $R^2=0$이면 회귀모형이 단순히 $\bar y$로 예측하는 것보다 전체 변동을 더 설명하지 못했다는 뜻이다.
* $R^2=1$이면 모든 잔차가 $0$이어서 관측값을 완전히 적합했다는 뜻이다.

설명변수를 추가하면 $SSE$는 증가하지 않으므로 $R^2$도 감소하지 않는다. 불필요한 설명변수 추가를 보정하기 위해 수정 결정계수를 사용한다.

$$
\boxed{
R_{\mathrm{adj}}^2 = 1-
\frac{SSE/(n-p-1)}
{SST/(n-1)}
}
$$

즉,

$$
\boxed{
R_{\mathrm{adj}}^2 = 1-\frac{MSE}{SST/(n-1)}
}
$$

이다.

---

## 9. 일원분산분석에서 SSB와 SSW

집단이 $k$개이고, $j$번째 집단의 $i$번째 관측값을 $y_{ij}$라고 하자.

* $n_j$: $j$번째 집단의 표본 크기다.
* $\bar y_j$: $j$번째 집단의 표본평균이다.
* $\bar y$: 전체 표본평균이다.
* $n=\sum_{j=1}^kn_j$: 전체 표본 크기다.

각 관측값의 전체 평균으로부터의 편차는

$$
\boxed{
y_{ij}-\bar y = (\bar y_j-\bar y)
+
(y_{ij}-\bar y_j)
}
$$

로 분해된다.

여기에서

* $\bar y_j-\bar y$는 집단평균과 전체 평균의 차이다.
* $y_{ij}-\bar y_j$는 관측값과 해당 집단평균의 차이다.

제곱하여 모든 집단과 관측값에 대해 더하면

$$
\boxed{SST=SSB+SSW}
$$

가 성립한다.

전체제곱합은

$$
\boxed{
SST = \sum_{j=1}^k\sum_{i=1}^{n_j}
(y_{ij}-\bar y)^2
}
$$

이다.

집단간제곱합은

$$
\boxed{
SSB = \sum_{j=1}^k
n_j(\bar y_j-\bar y)^2
}
$$

이다.

집단내제곱합은

$$
\boxed{
SSW = \sum_{j=1}^k\sum_{i=1}^{n_j}
(y_{ij}-\bar y_j)^2
}
$$

이다.

---

## 10. SSB와 SSW의 의미

$SSB$는 각 집단평균이 전체 평균으로부터 얼마나 떨어져 있는지를 측정한다.

$$
SSB = \sum_{j=1}^k
n_j(\bar y_j-\bar y)^2
$$

집단평균들이 서로 비슷하면 $SSB$가 작아지고, 집단평균들이 크게 다르면 $SSB$가 커진다.

$SSW$는 같은 집단 안의 관측값들이 해당 집단평균으로부터 얼마나 흩어져 있는지를 측정한다.

$$
SSW = \sum_{j=1}^k\sum_{i=1}^{n_j}
(y_{ij}-\bar y_j)^2
$$

따라서 다음과 같이 해석할 수 있다.

$$
\boxed{
\text{전체 변동} = \text{집단 간 변동}
+
\text{집단 내 변동}
}
$$

---

## 11. 일원분산분석의 자유도와 평균제곱

| 변동   |   제곱합 |   자유도 |            평균제곱 |
| ---- | ----: | ----: | --------------: |
| 집단 간 | $SSB$ | $k-1$ | $MSB=SSB/(k-1)$ |
| 집단 내 | $SSW$ | $n-k$ | $MSW=SSW/(n-k)$ |
| 전체   | $SST$ | $n-1$ |     $SST/(n-1)$ |

자유도는

$$
\boxed{
n-1=(k-1)+(n-k)
}
$$

로 분해된다.

검정하려는 귀무가설은

$$
H_0:\mu_1=\mu_2=\cdots=\mu_k
$$

이다. 대립가설은

$$
H_1:\text{모든 집단평균이 같지는 않다}
$$

이다.

검정통계량은

$$
\boxed{
F=\frac{MSB}{MSW} = \frac{SSB/(k-1)}{SSW/(n-k)}
}
$$

이다.

귀무가설과 정규성, 등분산성, 독립성 가정 아래에서

$$
F\sim F_{k-1,n-k}
$$

이다.

귀무가설이 참이면

$$
E(MSB)=\sigma^2
$$

이고

$$
E(MSW)=\sigma^2
$$

이다. 집단평균들이 실제로 다르면 $MSB$가 커지는 경향이 있으므로 $F$도 커진다.

---

## 12. 회귀분석과 분산분석의 대응

일원분산분석은 집단을 더미변수로 표현한 회귀분석과 수학적으로 동일하다.

| 회귀분석  | 일원분산분석 | 의미                    |
| ----- | ------ | --------------------- |
| $SSR$ | $SSB$  | 모형 또는 집단 차이가 설명한 변동   |
| $SSE$ | $SSW$  | 설명되지 않은 잔차 또는 집단 내 변동 |
| $MSR$ | $MSB$  | 자유도당 설명 변동            |
| $MSE$ | $MSW$  | 자유도당 잔차 변동            |

따라서 일원분산분석을 회귀모형으로 나타내면

$$
\boxed{SSR=SSB}
$$

이고

$$
\boxed{SSE=SSW}
$$

이다. 또한

$$
\boxed{MSR=MSB}
$$

이고

$$
\boxed{MSE=MSW}
$$

이다.

따라서 두 분석의 $F$-통계량도 같다.

$$
\boxed{
\frac{MSR}{MSE} = \frac{MSB}{MSW}
}
$$

---

## 13. SSR 표기의 모호성

SSR은 교재에 따라 두 가지 의미로 사용된다.

### 관습 1: Regression Sum of Squares

$$
SSR = \sum_{i=1}^n(\hat y_i-\bar y)^2
$$

즉, 모형이 설명한 변동을 뜻한다. 이 관습에서는

$$
SST=SSR+SSE
$$

라고 쓴다.

### 관습 2: Sum of Squared Residuals

일부 교재에서는 SSR을 잔차제곱합이라는 의미로 사용한다.

$$
SSR = \sum_{i=1}^n(y_i-\hat y_i)^2 = \sum_{i=1}^ne_i^2
$$

이 경우 첫 번째 관습의 $SSE$와 같은 양이다.

따라서 SSR이라는 기호만 보고 의미를 판단하면 안 된다. 반드시 해당 교재의 정의를 확인해야 한다.

혼동을 피하려면 다음과 같이 표기하는 것이 안전하다.

$$
SS_{\mathrm{Reg}} = \sum_{i=1}^n(\hat y_i-\bar y)^2
$$

$$
SS_{\mathrm{Err}} = \sum_{i=1}^n(y_i-\hat y_i)^2
$$

---

## 14. 최종 관계 요약

### 회귀분석

$$
\boxed{SST=SSR+SSE}
$$

$$
\boxed{
MSR=\frac{SSR}{p}
}
$$

$$
\boxed{
MSE=\frac{SSE}{n-p-1}
}
$$

$$
\boxed{
F=\frac{MSR}{MSE}
}
$$

$$
\boxed{
R^2 = \frac{SSR}{SST} = 1-\frac{SSE}{SST}
}
$$

### 일원분산분석

$$
\boxed{SST=SSB+SSW}
$$

$$
\boxed{
MSB=\frac{SSB}{k-1}
}
$$

$$
\boxed{
MSW=\frac{SSW}{n-k}
}
$$

$$
\boxed{
F=\frac{MSB}{MSW}
}
$$

### 회귀분석과 일원분산분석의 대응

$$
\boxed{
SSR\longleftrightarrow SSB
}
$$

$$
\boxed{
SSE\longleftrightarrow SSW
}
$$

$$
\boxed{
MSR\longleftrightarrow MSB
}
$$

$$
\boxed{
MSE\longleftrightarrow MSW
}
$$

핵심적으로 회귀분석은 전체 변동을

$$
\text{전체 변동} = \text{모형이 설명한 변동}
+
\text{잔차 변동}
$$

으로 분해한다. 일원분산분석은 같은 원리를

$$
\text{전체 변동} = \text{집단 간 변동}
+
\text{집단 내 변동}
$$

으로 표현한 것이다.
