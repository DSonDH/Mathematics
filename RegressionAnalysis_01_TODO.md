필요한 요인 (Z)가 회귀모형에서 빠지면 가장 먼저 의심해야 하는 것은

[
E(\varepsilon\mid X)=0
]

이라는 조건부 평균 0 가정이다. 다만 누락변수가 있다고 해서 독립성·등분산성·정규성 등이 모두 자동으로 깨지는 것은 아니다. 누락된 변수가 (X), (Y), 관측치 구조와 어떤 관계를 가지는지가 중요하다.

## 1. 누락변수가 포함된 실제 모형

실제 데이터 생성과정이

[
Y_i=X_i^\top\beta+Z_i^\top\gamma+u_i
]

라고 하자.

* (X_i): 회귀모형에 포함한 설명변수
* (Z_i): 실제로 (Y_i)에 영향을 주지만 누락한 변수
* (u_i): (X_i,Z_i)를 고려하고도 남은 순수한 충격

그런데 (Z_i)를 빼고

[
Y_i=X_i^\top\beta+\varepsilon_i
]

로 추정하면 새로운 오차항은

[
\boxed{\varepsilon_i=Z_i^\top\gamma+u_i}
]

가 된다. 즉, 누락변수의 영향이 오차항 안으로 들어간다.

---

## 2. 조건부 평균 0 가정

실제 순수 오차에 대해

[
E(u_i\mid X,Z)=0
]

이라고 하자. 누락 후 오차의 조건부 평균은

[
\begin{aligned}
E(\varepsilon_i\mid X)
&=E(Z_i^\top\gamma+u_i\mid X)\
&=E(Z_i\mid X)^\top\gamma+E(u_i\mid X)\
&=E(Z_i\mid X)^\top\gamma.
\end{aligned}
]

따라서 일반적으로

[
E(Z_i\mid X)\neq 0
]

이면

[
\boxed{E(\varepsilon_i\mid X)\neq0}
]

가 되어 조건부 평균 0 가정이 깨진다.

정확히는 절편이 포함되어 있다면 상수인 (E(Z_i\mid X))는 절편에 흡수할 수 있다. 문제가 되는 것은 (E(Z_i\mid X))가 (X)에 따라 변하는 경우다.

---

## 3. 누락변수 편향이 발생하는 핵심 조건

누락변수 편향이 발생하려면 보통 다음 두 조건이 동시에 성립해야 한다.

[
\boxed{\gamma\neq0}
]

즉, 누락된 변수 (Z)가 실제로 (Y)에 영향을 주어야 한다.

그리고

[
\boxed{\operatorname{Cov}(X,Z)\neq0}
]

즉, 누락된 변수 (Z)가 포함된 설명변수 (X)와 관련되어야 한다.

단순회귀에서 실제 모형이

[
Y_i=\beta_0+\beta_1X_i+\gamma Z_i+u_i
]

라면 (Z)를 누락한 OLS 기울기의 확률극한은

[
\operatorname{plim}\hat\beta_1
==============================

\beta_1+
\gamma
\frac{\operatorname{Cov}(X,Z)}
{\operatorname{Var}(X)}.
]

따라서 누락변수 편향은

[
\boxed{
\text{Bias}
===========

\gamma
\frac{\operatorname{Cov}(X,Z)}
{\operatorname{Var}(X)}
}
]

로 나타난다.

### 편향 방향

| (Z)가 (Y)에 미치는 영향 (\gamma) | (\operatorname{Cov}(X,Z)) | 편향 방향 |
| ------------------------: | ------------------------: | ----- |
|                       (+) |                       (+) | 상향 편향 |
|                       (+) |                       (-) | 하향 편향 |
|                       (-) |                       (+) | 하향 편향 |
|                       (-) |                       (-) | 상향 편향 |

---

## 4. 예: 교육과 임금에서 능력을 누락

실제 모형이

[
\text{임금}_i
===========

\beta_0
+\beta_1\text{교육}_i
+\gamma\text{능력}_i
+u_i
]

라고 하자.

능력을 관찰하지 못해

[
\text{임금}_i
===========

\beta_0+\beta_1\text{교육}_i+\varepsilon_i
]

로 추정하면

[
\varepsilon_i
=============

\gamma\text{능력}_i+u_i.
]

능력이 높은 사람이 교육도 더 많이 받는다면

[
E(\text{능력}_i\mid\text{교육}_i)
]

가 교육 수준에 따라 증가한다. 따라서

[
E(\varepsilon_i\mid\text{교육}_i)\neq0
]

가 되고 교육계수는 순수한 교육의 인과효과와 능력의 효과를 함께 포함하게 된다.

그러나 이 예에서 (E(\varepsilon\mid X)=0)이 깨진다고 단정하려면 (\varepsilon)을 구조적 모형에서 누락된 능력까지 포함한 오차로 정의해야 한다. 만약 (\varepsilon=Y-E(Y\mid X))라는 통계적 오차로 다시 정의하면 조건부 평균 0은 정의상 성립할 수 있지만, 그 회귀계수는 교육의 인과효과가 아니라 관찰된 조건부 연관성을 나타낼 수 있다.

---

## 5. 누락변수가 (X)와 독립이면 어떻게 되는가

누락변수 (Z)가 (Y)에 영향을 주더라도 (X)와 독립이고 평균이 상수라면 상황이 다르다.

[
Z\perp!!!\perp X,
\qquad
E(Z)=\mu_Z.
]

절편이 포함된 모형에서는

[
Z_i=\mu_Z+(Z_i-\mu_Z)
]

로 분해할 수 있다. (\gamma\mu_Z)는 절편에 흡수되고 새로운 오차는

[
\varepsilon_i
=============

\gamma(Z_i-\mu_Z)+u_i
]

가 된다. 그러면

[
E(\varepsilon_i\mid X)=0
]

일 수 있다.

따라서 (Z)가 빠졌다는 사실만으로 OLS 계수가 편향되는 것은 아니다.

다만 오차분산은 증가한다.

[
\operatorname{Var}(\varepsilon_i\mid X)
=======================================

\gamma^2\operatorname{Var}(Z_i\mid X)
+\operatorname{Var}(u_i\mid X)
+2\gamma\operatorname{Cov}(Z_i,u_i\mid X).
]

즉, 계수는 불편일 수 있지만 추정의 정밀도가 낮아질 수 있다.

---

## 6. 등분산성은 언제 깨지는가

누락 후 오차는

[
\varepsilon_i=Z_i^\top\gamma+u_i
]

이므로

[
\operatorname{Var}(\varepsilon_i\mid X)
=======================================

\operatorname{Var}(Z_i^\top\gamma+u_i\mid X).
]

만약 누락변수의 조건부분산이 (X)에 따라 달라지면

[
\operatorname{Var}(Z_i\mid X)=h(X_i)
]

이고,

[
\operatorname{Var}(\varepsilon_i\mid X)
]

도 (X_i)에 따라 달라질 수 있다. 따라서 등분산성

[
\operatorname{Var}(\varepsilon_i\mid X)=\sigma^2
]

이 깨진다.

예를 들어 소득이 높을수록 누락된 소비성향의 개인차가 커진다면 소득 수준에 따라 오차분산이 증가할 수 있다.

하지만 누락변수의 분산이 (X)와 무관하다면 누락변수가 있어도 등분산성은 유지될 수 있다.

---

## 7. 오차항 간 독립성은 언제 깨지는가

누락변수가 여러 관측치에 공통으로 작용하면 오차항들이 의존하게 된다.

예를 들어 학생 (i)가 학교 (g)에 속하고, 학교효과 (A_g)를 누락했다고 하자.

[
Y_{ig}
======

X_{ig}^\top\beta+A_g+u_{ig}.
]

학교변수를 누락하면

[
\varepsilon_{ig}=A_g+u_{ig}.
]

같은 학교에 속한 학생 (i,j)에 대해

[
\begin{aligned}
\operatorname{Cov}
(\varepsilon_{ig},\varepsilon_{jg}\mid X)
&=
\operatorname{Cov}(A_g+u_{ig},A_g+u_{jg}\mid X)\
&\approx\operatorname{Var}(A_g\mid X)>0.
\end{aligned}
]

따라서

[
\varepsilon_{ig}\not!\perp!!!\perp
\varepsilon_{jg}\mid X.
]

대표적인 경우는 다음과 같다.

* 학교효과를 누락한 학생 데이터
* 기업효과를 누락한 근로자 데이터
* 가구효과를 누락한 가족 데이터
* 시간효과를 누락한 패널데이터
* 경기변동을 누락한 시계열
* 지역효과를 누락한 공간데이터

반면 누락변수 (Z_i)가 관측치별로 독립이면, 누락으로 조건부 평균 0은 깨져도 오차항 간 독립성은 유지될 수 있다.

즉,

[
\boxed{\text{누락변수}\not\Rightarrow\text{오차항 간 독립성 위반}}
]

이다. 공통 누락요인이 있을 때 독립성이 주로 깨진다.

---

## 8. 정규성은 언제 깨지는가

원래 순수 오차가

[
u_i\mid X,Z\sim N(0,\sigma^2)
]

라고 하더라도 누락 후 오차는

[
\varepsilon_i=Z_i^\top\gamma+u_i
]

이다.

(Z_i\mid X)가 비정규분포라면 (\varepsilon_i\mid X)도 일반적으로 정규분포가 아니다. 예를 들어 누락변수가 이산적인 집단변수이면 오차분포가 여러 정규분포의 혼합으로 나타날 수 있다.

그러나 (Z_i\mid X)와 (u_i\mid X)가 결합정규분포를 따르는 경우에는 합도 정규분포이므로 정규성이 유지될 수 있다.

따라서 정규성 위반 역시 자동적이지 않다.

---

## 9. 선형 함수형태 가정도 깨질 수 있다

누락된 요인이 새로운 변수가 아니라 비선형항일 수도 있다.

실제 모형이

[
Y_i=\beta_0+\beta_1X_i+\beta_2X_i^2+u_i
]

인데 (X_i^2)를 누락하여

[
Y_i=\alpha_0+\alpha_1X_i+\varepsilon_i
]

를 적합했다고 하자. 그러면

[
\varepsilon_i=\beta_2X_i^2+u_i
]

이고,

[
E(\varepsilon_i\mid X_i)=\beta_2X_i^2\neq0.
]

이 경우는 변수 누락인 동시에 함수형태 오류다.

다음 항들을 누락해도 같은 문제가 발생할 수 있다.

* (X^2,X^3) 등의 비선형항
* (X_1X_2) 같은 상호작용항
* 시간추세
* 계절효과
* 집단 고정효과
* 시차변수
* 구조변화 항

---

## 10. 어떤 가정이 자동으로 깨지는 것은 아닌가

누락변수가 생겨도 다음 조건들은 반드시 깨지는 것은 아니다.

### (X)의 완전계수 조건

[
\operatorname{rank}(X)=k
]

는 포함된 (X)열 사이의 선형관계에 관한 조건이다. 변수가 누락되었다고 이 조건이 자동으로 깨지지는 않는다.

### 무작위 표본 가정

[
(Y_i,X_i)\overset{\mathrm{iid}}{\sim}P
]

도 누락변수 때문에 자동으로 깨지지는 않는다. 동일한 모집단에서 독립 추출했다면 관측치들은 여전히 i.i.d.일 수 있다.

다만 누락된 집단·시간·공간 요인이 관측치 간 의존성을 만들었다면 독립 표본 가정이 깨질 수 있다.

---

## 11. 경우별 정리

| 누락된 요인의 성질                | 주로 깨지는 조건                | 계수에 미치는 영향      |
| ------------------------- | ------------------------ | --------------- |
| (Z)가 (Y)에 영향을 주고 (X)와 관련됨 | (E(\varepsilon\mid X)=0) | 편향·비일치 가능       |
| (Z)가 (Y)에 영향을 주지만 (X)와 독립 | 반드시 깨지는 핵심 외생성 조건 없음     | 보통 편향 없음, 분산 증가 |
| (Z)의 변동성이 (X)에 따라 달라짐     | 등분산성                     | 고전적 표준오차 오류     |
| (Z)가 여러 관측치의 공통요인         | 오차 비상관·독립성               | 고전적 표준오차 오류     |
| (Z\mid X)가 비정규·혼합분포       | 조건부 정규성                  | 소표본 (t,F) 추론 영향 |
| 누락항이 (X^2), 상호작용 등임       | 선형 조건부 평균                | 함수형태 오류·편향      |
| (Z)가 완전하게 무관하고 단순 잡음임     | 핵심 가정이 유지될 수 있음          | 설명력·효율성 저하      |

핵심은 다음과 같다.

[
\boxed{
\text{중요 변수를 누락했다고 모든 가정이 깨지는 것은 아니다}
}
]

가장 심각한 경우는 누락변수가 결과에 영향을 주면서 포함된 설명변수와 관련된 경우다.

[
\boxed{
Z\rightarrow Y
\quad\text{그리고}\quad
Z\not!\perp!!!\perp X
\quad\Longrightarrow\quad
E(\varepsilon\mid X)=0\text{이 일반적으로 깨진다}
}
]

공통 누락요인이면 오차 간 독립성도 깨질 수 있고, 누락요인의 변동성이 (X)에 따라 달라지면 등분산성도 깨질 수 있다. 따라서 누락변수 문제는 하나의 가정 위반이 아니라 누락변수의 구조에 따라 여러 가정 위반으로 나타나는 문제다.
