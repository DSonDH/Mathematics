7장의 검정 개념은 가설검정의 기본 형식 논리다. 검정법 비교기준은 그 기본 틀에서 유의수준을 고정하고 검정력을 비교하는 원리이며, 최강력 검정·UMP·UMPU는 그 비교 원리를 각각 단순 대립가설, 복합 대립가설, 불편검정류에 대해 최적화한 개념이다. 비모수 검정은 같은 검정 틀을 유지하되 분포가정을 약하게 둔 방법이고, 검정력 근사는 검정력 함수를 정확히 계산하기 어려울 때 점근분포 등을 이용해 근사하는 계산 기술이다. 7장의 LRT와 점근 LRT는 이 전체 구조 중 parametric 검정과 asymptotic approximation에 해당한다.

## 1. 7장에서 다루는 것은 무엇인가

7장은 다음을 중심으로 설명하고 있다.

* 검정은 귀무가설과 대립가설을 세우고, 자료가 귀무가설과 얼마나 부합하는지 보고 기각 여부를 결정하는 절차라는 점 
* 기각역은 유의수준 제약
  $$
  \sup_{\theta\in\Omega_0}P_\theta(X\in C_\alpha)\le \alpha
  $$
  을 만족해야 한다는 점 
* 같은 유의수준 안에서는 검정력을 크게 하는 방향이 바람직하다는 점 
* 검정력 함수
  $$
  \gamma_\phi(\theta)=E_\theta[\phi(X)]
  $$
  와 검정의 크기(size)를 정의하는 점 
* 최대가능도비 검정(LRT)은 귀무가설 하 최대가능도와 전체 모수공간 최대가능도를 비교하여 검정통계량을 만든다는 점 
* 큰 표본에서는 가능도비 통계량을 $\chi^2$로 근사하는 점근적 검정 아이디어가 나온다는 점 

즉 7장은 **검정의 형식 논리 + 검정력 + LRT + 점근 LRT**까지의 기본 틀을 제공한다.

## 2. “검정법 비교기준들”은 7장의 어디서 출발하는가

검정법 비교기준은 7장에 이미 씨앗이 들어 있다.
7장은 “유의수준을 만족시키면서 검정력을 최대화하는 것이 바람직하다”고 설명한다. 

여기서 바로 검정법 비교기준이 나온다. 보통 비교기준은 다음 순서로 정리된다.

### (1) 크기(size) 또는 유의수준(level)

먼저 같은 오류 통제를 만족해야 비교가 가능하다.

$$
\sup_{\theta\in\Omega_0}E_\theta[\phi(X)]\le \alpha
$$

7장도 바로 이 조건을 검정의 기본 제약으로 둔다. 

### (2) 검정력 함수(power function)

그 다음 대립가설 아래에서 기각확률이 큰 검정이 더 좋다.

$$
\gamma_\phi(\theta)=E_\theta[\phi(X)],\qquad \theta\in\Omega_1
$$

이 역시 7장의 핵심 정의다. 

### (3) 전역 비교 기준

대립가설이 한 점이 아니라 여러 값이면, 검정력 함수 전체를 비교해야 한다.
이때 “모든 $\theta\in\Omega_1$에서 더 큰가?”라는 질문이 생기고, 여기서 **최강력, UMP, UMPU**가 나온다.

즉, **검정법 비교기준은 7장의 ‘유의수준 + 검정력’ 정의를 한 단계 더 밀고 나간 것**이다.

## 3. 최강력 검정(Most Powerful test)은 무엇이며 7장과 무슨 관계인가

최강력 검정은 보통 **단순 귀무가설 vs 단순 대립가설**

$$
H_0:\theta=\theta_0
\quad\text{vs}\quad
H_1:\theta=\theta_1
$$

에서 정의된다.

유의수준 $\alpha$인 검정들 중에서

$$
E_{\theta_1}[\phi(X)]
$$

를 가장 크게 만드는 검정이 $\theta_1$에 대한 **최강력 검정**이다.

### 7장과의 관계

7장은 이미 “유의수준 제약 아래 검정력을 크게 하는 것이 목표”라고 말한다. 
최강력 검정은 바로 그 문장을 **가장 엄밀한 최적화 문제**로 만든 것이다.

즉,

* 7장: “좋은 검정은 같은 $\alpha$에서 검정력이 커야 한다.”
* 최강력 검정: “그러면 특정 대립가설 한 점 $\theta_1$에서 검정력이 가장 큰 검정을 정의하자.”

따라서 최강력 검정은 7장 개념의 **정밀화**다.

## 4. 전역최강력검정(UMP)은 무엇이며 왜 더 어려운가

UMP는 단순 대립가설 한 점이 아니라, **복합 대립가설 전체**에서 가장 좋은 검정이다.

예를 들어

$$
H_0:\theta\le \theta_0
\quad\text{vs}\quad
H_1:\theta>\theta_0
$$

에서 유의수준 $\alpha$ 검정 $\phi^*$가 모든 $\theta>\theta_0$에 대해

$$
E_\theta[\phi^*(X)] \ge E_\theta[\phi(X)]
$$

를 만족하면 UMP다.

### 7장과의 관계

7장은 복합가설과 검정력 함수를 이미 도입했다.
특히 검정의 크기를

$$
\sup_{\theta\in\Omega_0}E_\theta[\phi(X)]
$$

로 정의하므로, 귀무가설이 복합일 수 있다는 틀을 이미 제공한다. 

UMP는 여기서 한 걸음 더 나아가,

* 귀무가설은 복합이고,
* 대립가설도 보통 복합이며,
* 그 전체에서 uniformly, 즉 **전 구간에서 동시에** 가장 큰 검정력을 요구한다.

그래서 UMP는 7장의 “검정력 최대화”를 **복합 대립가설 전체에 대해 전역화한 개념**이다.

## 5. UMPU 검정은 왜 필요한가

양측가설에서 UMP가 존재하지 않는 경우가 많다.
대표적으로

$$
H_0:\mu=\mu_0
\quad\text{vs}\quad
H_1:\mu\ne\mu_0
$$

같은 양측 검정에서는 한쪽 꼬리에 유리한 검정이 다른 쪽 꼬리에서는 불리해질 수 있어서, 모든 대립가설 점에서 동시에 최고인 UMP가 대개 없다.

그래서 조건을 조금 바꾼다.
그냥 “전역적으로 제일 큰 검정력”이 아니라, **불편(unbiased)** 한 검정들만 후보로 제한한 뒤 그 안에서 최강력을 찾는다.

불편검정의 표준적 의미는 보통

$$
E_\theta[\phi(X)] \le \alpha \quad (\theta\in\Omega_0),\qquad
E_\theta[\phi(X)] \ge \alpha \quad (\theta\in\Omega_1)
$$

이다.

즉 대립가설 아래에서는 적어도 귀무가설 경계에서의 기각확률보다 작아지지 않아야 한다.

### 7장과의 관계

7장은 불편성 자체를 직접 다루지는 않지만,
이미 “같은 유의수준 하에서 검정력을 비교한다”는 관점을 제공한다. 
UMPU는 그 비교를 할 때 **비교대상을 무작정 전체 검정으로 두지 않고, 합리적인 검정군으로 제한한 것**이다.

정리하면

* MP: 한 점 대립가설에서 최고
* UMP: 대립가설 전체에서 최고
* UMPU: UMP가 없을 때, 불편검정 집합 안에서 최고

## 6. 비모수 검정은 7장의 “검정 개념”과 어떻게 다른가

비모수 검정은 **검정의 기본 논리 자체가 다른 것**이 아니라,
**모형 가정이 덜 강한 검정**이다.

7장의 많은 예시는 정규분포, 포아송분포, 지수분포처럼 **구체적 분포모형**을 놓고 검정을 만든다. 또한 LRT는 애초에 가능도함수 $f(x;\theta)$를 명시해야 한다. 

반면 비모수 검정은 보통

* 정확한 분포형태를 완전히 지정하지 않거나,
* 위치 대칭성, 연속성 같은 약한 가정만 두고,
* 순위(rank), 부호(sign), 순열(permutation) 같은 구조를 이용한다.

예를 들어 부호검정, Wilcoxon 부호순위검정, Mann–Whitney 검정 등이 여기에 속한다.

### 7장과의 관계

비모수 검정도 여전히

* 귀무가설을 세우고,
* 유의수준을 통제하고,
* 검정력을 논하고,
* 기각역을 정한다.

즉 **검정의 형식 논리는 7장과 완전히 같다.**
다만 차이는 **확률모형의 강도**다.

* 7장의 많은 예: 분포를 구체적으로 지정한 모수적(parametric) 검정
* 비모수 검정: 분포를 덜 가정한 검정

따라서 비모수 검정은 7장과 대립되는 개념이 아니라, **7장의 검정 틀 안에 들어오는 한 종류의 검정법**이다.

## 7. 검정력 근사는 7장의 무엇과 연결되는가

검정력 근사는 말 그대로 **검정력 함수를 정확히 계산하기 어려울 때 근사하는 방법**이다.

7장은 검정력 함수를 정의하고, 몇몇 단순 예에서는 정확식을 쓴다. 예를 들어 정규 평균 검정에서는 검정력 함수를 직접 적는다. 
또한 7.3에서는 큰 표본에서 가능도비 통계량 분포를 근사하는 점근 검정을 소개한다. 

검정력 근사는 바로 여기서 나온다.

### 대표적 상황

1. 검정통계량의 정확 분포는 너무 복잡하다.
2. 표본이 크므로 정규근사, $\chi^2$ 근사, 비중심분포 근사를 쓴다.
3. 이를 통해
   $$
   P_\theta(\text{기각})
   $$
   를 근사한다.

### 7장과의 관계

즉 검정력 근사는

* 7장의 **검정력 함수** 개념을 실제 계산 가능하게 만드는 기술이며, 
* 7장의 **점근 가능도비 검정**과 직접 연결된다. 

다시 말해, 7장에서 “검정력은 중요하다”, “큰 표본에서는 분포를 근사할 수 있다”고 했다면,
검정력 근사는 그 두 문장을 합쳐서 **실제 대안 모수값들에서 power curve를 계산하는 도구**다.

## 8. 한 장으로 정리하면: 서로의 관계

아래 구조로 보면 정리가 가장 쉽다.

### A. 7장의 기본 골격

* 귀무가설 / 대립가설
* 기각역 / 유의수준
* 제1종오류 / 제2종오류
* 검정력 함수
* 가능도비 검정
* 점근 근사 검정

### B. 그 위의 “비교 원리”

* 같은 유의수준이면 검정력을 더 크게 하는 검정이 더 좋다.
  이것이 검정법 비교기준이다. 

### C. 그 비교 원리를 최적화 개념으로 만든 것

* 한 점 대립가설에서 최고 → **최강력 검정**
* 대립가설 전체에서 최고 → **UMP**
* UMP가 없을 때 불편검정류 안에서 최고 → **UMPU**

### D. 모형 가정을 완화한 가지

* parametric 검정: 정규, 포아송, 지수 등 분포를 구체적으로 둔다. 7장의 주 예시가 여기에 가깝다. 
* nonparametric 검정: 순위, 부호, 순열 기반으로 더 약한 가정만 둔다.

### E. 계산 기술

* exact power가 가능하면 그대로 계산
* 어렵다면 asymptotic distribution을 써서 **검정력 근사**
  이것이 7장의 7.3과 이어진다. 

## 9. 사용자가 지금 공부하는 문맥에서 특히 중요한 연결

사용자가 최근 보고 있는 부호순위 검정 같은 내용까지 연결하면 다음처럼 보면 된다.

* **부호검정 / 부호순위검정**: 비모수 검정의 대표 예다.
* 이 검정도 여전히 유의수준과 검정력으로 평가한다.
* “이 검정이 한쪽 대립가설에서 단조성을 가지는가”, “유의수준 $\alpha$를 만족하는가”, “다른 검정보다 power가 어떤가” 같은 질문은 모두 7장의 기본 검정 프레임에서 나온다.
* 다만 LRT처럼 완전한 parametric likelihood를 쓰는 것이 아니라, 순위 구조나 대칭성 가정을 활용한다.

즉 **비모수 검정은 7장의 대체물이 아니라, 그 프레임 안에서 likelihood 대신 rank/sign structure를 쓰는 버전**이다.


# 제9장 검정의 비교 *(Comparison of Tests)*

## 검정법 비교의 기준 *(Criteria for Comparing Tests)*
**[단순 가설(simple hypothesis)]**: 귀무가설과 대립가설이 각각 하나의 확률밀도함수로 주어지므로, 검정 방법의 비교 기준을 이해하기 쉽다.  
- 모집단 분포: 확률밀도함수 $f(x;\theta)$
- 모수 공간: $\theta \in \Omega = \{\theta_0, \theta_1\}$
- 랜덤표본: $X = (X_1, \dots, X_n)$

검정 문제: $H_0: \theta = \theta_0 \quad \text{vs} \quad H_1: \theta = \theta_1$

**[복합 가설(composite hypothesis)]**  
단순 가설과 달리, 귀무가설 또는 대립가설이 하나의 확률분포가 아니라 여러 개의 모수값(즉, 여러 확률분포)으로 정의되는 경우를 **복합 가설(composite hypothesis)** 이라 한다.  
- 예시: $H_0: \theta \leq \theta_0$ 또는 $H_1: \theta > \theta_0$  
    이 경우 $H_0$는 $\theta$가 $\theta_0$ 이하인 모든 값을 포함하므로, 하나의 확률분포가 아니라 여러 분포의 집합이 된다.
- 일반적으로 실제 통계적 검정에서는 복합가설이 더 자주 등장한다.

**[검정과 오류 확률]**  
**(1) 비랜덤화 검정**  
기각역 $C \subset \mathcal{X}^n$에 대해

$$X \in C \Rightarrow H_0 \text{ 기각}$$

오류 확률 (검정이 실패할 확률):
- 제1종 오류: $P_{\theta_0}(X \in C)$
- 제2종 오류: $P_{\theta_1}(X \notin C) = 1 - P_{\theta_1}(X \in C)$

**(2) 랜덤화 검정**  
검정 함수:

$$\phi(X) = \phi(X_1,\dots,X_n), \quad 0 \le \phi(X) \le 1$$

- $\phi(X)$: 귀무가설을 기각할 확률

오류 확률 (검정력 함수, power function (검정함수랑 다름)):
- 제1종 오류: $E_{\theta_0}[\phi(X)]$
- 제2종 오류: $E_{\theta_1}[1 - \phi(X)]$

> **(추가) 용어설명: 검정함수 $\phi(x)$**  
> - **검정함수 $\phi(x)$**: 관측값 $x$에 대해 귀무가설 $H_0$를 기각할 확률을 나타내는 함수. $\phi(x) \in [0,1]$의 값을 가지며, $\phi(x)=1$이면 $H_0$를 반드시 기각, $\phi(x)=0$이면 $H_0$를 절대 기각하지 않음, $0<\phi(x)<1$이면 확률적으로 기각(랜덤화 검정).
> - **비랜덤화 검정**: 기각역 $C$에 대해 $\phi(x) = \mathbb{I}(x \in C)$로 정의할 수 있으며, 이때 $E_\theta[\phi(X)] = P_\theta(X \in C)$로 기존의 오류 확률 표현과 동일하다.
> - **랜덤화 검정**: $\phi(x)$가 $[0,1]$의 임의의 값을 가질 수 있어, 관측값에 따라 확률적으로 기각 여부를 결정한다.
> 
> 검정함수 $\phi(x)$는 비랜덤화/랜덤화 검정 모두에 적용되는 일반적인 표현이다.

1종오류, 2종오류 확률을 동시에 작게 만드는 검정이 이상적이지만, 한쪽 오류 확률을 줄이면 다른 쪽 오류 확률이 커지는 trade-off가 존재한다. 따라서 오류를 종합적으로 평가하는 기준이 필요하다.

### 최대오류확률과 베이지안 평균오류확률 *(Maximum Error Probability and Bayesian Average Error Probability)*

**[검정 비교의 기준들]**: 단순 가설에서 두 오류 확률을 동시에 작게 하는 기준이 필요한데, 대표적인 두 가지 기준은 다음과 같다.

**(a) 최대오류확률 기준 *(Maximum Error Probability Criterion)***

$$\max\left\{ E_{\theta_0}[\phi(X)],\; E_{\theta_1}[1-\phi(X)] \right\}$$

값이 작을수록 좋은 검정이다.

**(b) 베이지안 평균오류확률 기준 *(Bayesian Average Error Probability Criterion)***

$$\pi_0 E_{\theta_0}[\phi(X)] + \pi_1 E_{\theta_1}[1-\phi(X)]$$

- 가중치: $\pi_0 + \pi_1 = 1,\quad \pi_0 > 0,\; \pi_1 > 0$
- $(\pi_0, \pi_1)$: 사전확률(prior probability)

### 정리 9.1.1 단순 가설의 베이즈 검정
단순 가설 $H_0:\theta=\theta_0 ,\quad H_1:\theta=\theta_1$
에 대해, 다음 검정

$$
\phi^\pi(x) =
\begin{cases}
1, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} > \dfrac{\pi_0}{\pi_1} \\
0, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} < \dfrac{\pi_0}{\pi_1}
\end{cases}
$$

을 만족하는 검정은 베이지안 평균오류확률을 최소로 한다.
> 여기서 양수 나누기 0은, 양의 무한대값으로 취급하며, 어떤 실수보다도 큰 것으로 약속한다.

#### 증명
베이지안 평균오류확률은

$$
\pi_0 E_{\theta_0}[\phi(X)] + \pi_1 E_{\theta_1}[1-\phi(X)] \\
= \int \left[ \pi_0 \phi(x) pdf(x;\theta_0) + \pi_1 (1-\phi(x)) pdf(x;\theta_1) \right] dx
$$

이를 $\phi(x)$에 대해 최소화하려면, 각 $x$에 대해 integrand
$$
\pi_0 \phi(x) pdf(x;\theta_0) + \pi_1 (1-\phi(x)) pdf(x;\theta_1)

$$

을 최소화하는 $\phi(x)$를 선택하면 된다.

$\phi(x)$는 $[0,1]$ 사이의 값(랜덤화 검정)도 가능하지만, integrand가 $\phi(x)$에 대해 선형이므로, 극값은 $\phi(x)=0$ 또는 $\phi(x)=1$에서만 달성된다(비랜덤화 검정).

- $\phi(x)=1$일 때: 값은 $\pi_0 pdf(x;\theta_0)$
- $\phi(x)=0$일 때: 값은 $\pi_1 pdf(x;\theta_1)$

따라서 $\phi(x)=1$이 더 작으려면

$$
\pi_0 pdf(x;\theta_0) < \pi_1 pdf(x;\theta_1)
\implies
\frac{pdf(x;\theta_1)}{pdf(x;\theta_0)} > \frac{\pi_0}{\pi_1}
$$

이므로, 이 경우 $\phi(x)=1$로 하고, 그렇지 않으면 $\phi(x)=0$으로 한다. 즉,

$$
\phi^\pi(x) =
\begin{cases}
1, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} > \dfrac{\pi_0}{\pi_1} \\
0, & \text{otherwise}
\end{cases}$$

가 베이지안 평균오류확률을 최소로 한다.

#### 예 9.1.1 (정규분포)
표본: $X_1,\dots,X_n \sim N(\mu,1)$  
가설: $H_0:\mu=\mu_0 ,\quad H_1:\mu=\mu_1$

$$
\frac{pdf(x;\mu_1)}{pdf(x;\mu_0)} =
\exp\left[
n(\mu_1-\mu_0)
\left(
\bar{x} - \frac{\mu_1+\mu_0}{2}
\right)
\right]
$$

(a) $\mu_1 > \mu_0$일 때

$$
\phi^\pi(x)=
\begin{cases}
1, & \bar{x}-\mu_0 >
\frac{\mu_1-\mu_0}{2}
+ \frac{\log(\pi_0/\pi_1)}{n(\mu_1-\mu_0)} \\
0, & \text{otherwise}
\end{cases}
$$

(b) $\mu_1 < \mu_0$일 때

$$
\phi^\pi(x)=
\begin{cases}
1, & \bar{x}-\mu_0 <
-\frac{\mu_0-\mu_1}{2}
+ \frac{\log(\pi_0/\pi_1)}{n(\mu_0-\mu_1)} \\
0, & \text{otherwise}
\end{cases}
$$

베이지안 평균오류확률이나 최대오류확률 기준은 귀무가설과 대립가설을 대등하게 취급한다는 뜻을 내포하고 있으며, 이는 제7장에서 다룬 전통적 검정과 다르다.  
(단순가설 뿐만 아니라 일반적인 가설의 경우에도 적용가능하지만, 이 책의 수준을 넘으므로 생략한다.)

### 최강력 검정 *(Most Powerful Test)*

전통적 검정의 형식 논리에서는 다음 순서로 진행한다:

1. 제1종 오류 확률을 유의수준 $\alpha$ 이하로 제한  
2. 그 조건 하에서 검정력을 최대화: $\gamma_\phi(\theta_1) = E_{\theta_1}[\phi(X)]$ 를 최대화

**정의: 유의수준 $\alpha$의 최강력 검정**  
단순 가설 $H_0:\theta=\theta_0,\quad H_1:\theta=\theta_1$에서 검정 $\phi^{MP}_\alpha(X)$가 다음을 만족하면 **최강력 검정**(Most Powerful Test, MP test)이라 한다.

1. **(유의수준 조건)**
    $$E_{\theta_0}[\phi^{MP}_\alpha(X)] \le \alpha \quad (0<\alpha<1)$$

2. **(최대 검정력 조건)**
    $$E_{\theta_1}[\phi^{MP}_\alpha(X)] \ge E_{\theta_1}[\phi(X)] \quad \forall \phi: E_{\theta_0}[\phi(X)] \le \alpha$$

### 정리 9.1.2 단순 가설의 최강력 검정 (Neyman–Pearson Lemma, 네이만–피어슨 정리)

조건1: 단순 가설인 경우에, 다음과 같은 가능도비 검정꼴이고,

$$\phi^*(x)=\begin{cases}
1, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} > k \\
\gamma, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} = k \\
0, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} < k
\end{cases}$$

(즉, 우도비가 큰 순서대로 위에서 아래로 채운다. 단, $0 \le \gamma \le 1$, $k \ge 0$)  
조건2: 다음 조건을 만족할 때, 유의수준 $\alpha$의 **최강력 검정**이다.

$$E_{\theta_0}[\phi^*(X)] = \alpha$$

- $k$ 찾는법: $k$는 $pdf(x;\theta_1)/pdf(x;\theta_0)$의 분포에 따라 결정된다. 일반적으로 $k$를 증가시키면서 $E_{\theta_0}[\phi^*(X)]$가 $\alpha$보다 작아지는 지점을 찾는다.
- $\gamma$ 찾는법: $k$를 정한 후, $E_{\theta_0}[\phi^*(X)]$가 $\alpha$보다 작으면 $\gamma$를 1로, 크면 0으로 설정한다. 만약 $E_{\theta_0}[\phi^*(X)]$가 정확히 $\alpha$보다 크거나 작은 경우에는 $\gamma$를 적절히 조절하여 정확히 $\alpha$가 되도록 한다.

> **참고:** 이 기준은 제7장의 전통적 접근으로, 귀무가설에만 제약을 두고 대립가설에서 검정력을 최대화한다는 점에서 (a), (b)와 다르다.  
> frequentist 관점의 접근법이다.

>각주: 일반적으로 위 두 조건을 만족시키는 $k, \gamma$가 존재하는 것이 알려져 있다.

> 각주: 조건1에서, $\gamma$가 $x$에 의존해도 된다고 알려져 있다.

#### 증명
먼저, 임의의 검정 함수 $\phi(X)$에 대해 다음 식을 살펴보자.

$$E_{\theta_1}[\phi(X)] - k E_{\theta_0}[\phi(X)]$$

여기서 $E_{\theta_1}[\phi(X)]$는 대립가설 $\theta_1$ 하에서의 검정력, $E_{\theta_0}[\phi(X)]$는 귀무가설 $\theta_0$ 하에서의 제1종 오류 확률, $k$는 상수이다.

확률밀도함수 $pdf(X;\theta)$를 이용해 위 식을 다음과 같이 변형할 수 있다.

$$
E_{\theta_1}[\phi(X)] - k E_{\theta_0}[\phi(X)]
= \int \phi(x) [pdf(x;\theta_1) - k\, pdf(x;\theta_0)] dx
$$

이제 $pdf(x;\theta_0) > 0$인 부분과 $pdf(x;\theta_0) = 0$인 부분으로 나누어 생각하면,

$$= \int_{pdf(x;\theta_0)>0} \phi(x) [pdf(x;\theta_1) - k\, pdf(x;\theta_0)] dx + \int_{pdf(x;\theta_0)=0} \phi(x) pdf(x;\theta_1) dx$$

첫 번째 항은 $pdf(x;\theta_0)>0$인 부분에서 $pdf(x;\theta_1)/pdf(x;\theta_0) - k$로 쓸 수 있으므로,

$$= \int_{pdf(x;\theta_0)>0} \phi(x) pdf(x;\theta_0) \left( \frac{pdf(x;\theta_1)}{pdf(x;\theta_0)} - k \right) dx + \int_{pdf(x;\theta_0)=0} \phi(x) pdf(x;\theta_1) dx \\
= E_{\theta_0}\left[ \phi(X) \left( \frac{pdf(X;\theta_1)}{pdf(X;\theta_0)} - k \right) \right] + E_{\theta_1}\left[ \phi(X) I(pdf(X;\theta_0)=0) \right]$$

이제 최강력 검정 $\phi^*(X)$와 임의의 검정 $\phi(X)$의 차이를 비교해보자.

$$
\Delta = \left( E_{\theta_1}[\phi^*(X)] - k E_{\theta_0}[\phi^*(X)] \right)
- \left( E_{\theta_1}[\phi(X)] - k E_{\theta_0}[\phi(X)] \right) \\
= E_{\theta_0}\left[ (\phi^*(X) - \phi(X)) \left( \frac{pdf(X;\theta_1)}{pdf(X;\theta_0)} - k \right) \right]
+ E_{\theta_1}\left[ (\phi^*(X) - \phi(X)) I(pdf(X;\theta_0)=0) \right]
$$

최강력 검정 $\phi^*(X)$는 가능도비 $\frac{pdf(X;\theta_1)}{pdf(X;\theta_0)}$가 $k$보다 크면 1, 작으면 0, 같으면 $\gamma$로 정의된다. 따라서 $(\phi^*(X) - \phi(X)) \left( \frac{pdf(X;\theta_1)}{pdf(X;\theta_0)} - k \right)$는 항상 0 이상이므로,

$$\Delta \ge 0$$

즉,

$$
E_{\theta_1}[\phi^*(X)] - k E_{\theta_0}[\phi^*(X)] \ge E_{\theta_1}[\phi(X)] - k E_{\theta_0}[\phi(X)]
$$

이제, $E_{\theta_0}[\phi(X)] \le \alpha$를 만족하는 임의의 $\phi(X)$에 대해, $k$와 $\gamma$를 적절히 선택하여 $E_{\theta_0}[\phi^*(X)] = \alpha$가 되도록 하면,

$$
E_{\theta_1}[\phi^*(X)] - E_{\theta_1}[\phi(X)] \ge k \left( E_{\theta_0}[\phi^*(X)] - E_{\theta_0}[\phi(X)] \right) = k (\alpha - E_{\theta_0}[\phi(X)]) \ge 0
$$

따라서 $\phi^*(X)$는 유의수준 $\alpha$에서 대립가설 하에서 검정력이 최대가 되는, 즉 최강력 검정임을 알 수 있다.

#### 다른 증명 (최적화 관점에서의 네이만–피어슨 정리)
단순가설 $H_0:\theta=\theta_0$ vs $H_1:\theta=\theta_1$에서, 검정 함수 $\phi(x)\in[0,1]$에 대해  
최적화 문제:

$$
\max_\phi\; E_{\theta_1}[\phi(X)] \quad \text{subject to} \quad E_{\theta_0}[\phi(X)] \le \alpha
$$

기댓값을 적분으로 쓰면,

$$
E_{\theta_1}[\phi(X)] = \int \phi(x) f_1(x)\,dx,\quad
E_{\theta_0}[\phi(X)] = \int \phi(x) f_0(x)\,dx
$$

여기서 $f_i(x) = f(x;\theta_i)$.  
라그랑주 승수 $k\ge 0$를 도입하여 목적함수로 표현:

$$
L(\phi) = \int \phi(x) [f_1(x) - k f_0(x)]\,dx + k\alpha
$$

$\phi(x)$는 각 $x$에서 독립적으로 값을 가질 수 있으므로, 각 $x$에 대해 integrand $[f_1(x) - k f_0(x)]$를 최대화하는 $\phi(x)$를 선택하면 된다.

- $f_1(x) - k f_0(x) > 0$이면 $\phi(x)=1$이 최적
- $f_1(x) - k f_0(x) < 0$이면 $\phi(x)=0$이 최적
- $f_1(x) - k f_0(x) = 0$이면 $\phi(x)\in[0,1]$ 임의 (랜덤화 가능)

즉,

$$
\phi^*(x) =
\begin{cases}
1, & \frac{f_1(x)}{f_0(x)} > k \\
\gamma, & \frac{f_1(x)}{f_0(x)} = k \\
0, & \frac{f_1(x)}{f_0(x)} < k
\end{cases}
$$

$(0\le\gamma\le 1)$  
여기서 $k,\gamma$는 $E_{\theta_0}[\phi^*(X)] = \alpha$를 만족하도록 선택한다.

결론: 가능도비 검정이 네이만–피어슨 문제의 유일한 해임이 점별 최적화로부터 필연적으로 도출된다.

#### 예 9.1.2 (포아송 분포)
- 표본: $X_1,\dots,X_{100} \sim \mathrm{Poisson}(\theta)$
- 가설: $H_0:\theta=0.1 ,\quad H_1:\theta=0.05$
- 유의수준: $\alpha=0.05$

가능도비:

$$
\frac{pdf(x;\theta_1)}{pdf(x;\theta_0)}
= e^{-100(\theta_1-\theta_0)}
\left(\frac{\theta_1}{\theta_0}\right)^{\sum x_i}
= e^5 (1/2)^{\sum x_i}
$$

**계산 과정:**

귀무가설 $H_0:\theta=0.1$ 하에서 $\sum_{i=1}^{100} X_i \sim \mathrm{Poisson}(10)$이다.

유의수준 조건:

$$E_{\theta_0}[\phi^*(X)] = P_{\theta_0}\left(\sum X_i \le c-1\right) + \gamma \, P_{\theta_0}\left(\sum X_i = c\right) = 0.05$$

포아송 누적분포함수 표에서 $\mathrm{Poisson}(10)$의 누적확률을 확인하면:
- $P(\sum X_i \le 4) = 0.0293$
- $P(\sum X_i \le 5) = 0.0671$

따라서 $c=5$일 때, $P(\sum X_i \le 4) = 0.0293$이므로

$$0.0293 + \gamma \, P(\sum X_i = 5) = 0.05$$

$P(\sum X_i = 5) = P(\sum X_i \le 5) - P(\sum X_i \le 4) = 0.0671 - 0.0293 = 0.0378$

따라서

$$\gamma = \frac{0.05 - 0.0293}{0.0378} = \frac{0.0207}{0.0378} = \frac{21}{38}$$

계산 결과, $c=5, \gamma=\frac{21}{38}$


## 전역최강력 검정 *(UMP, Uniformly Most Powerful Tests)*
모집단 분포가 확률밀도함수 $f(x;\theta)$, $\theta \in \Omega$ 중의 하나인 경우에 랜덤표본 $X=(X_1,\dots,X_n)$을 이용하여 일반적인 가설

$$H_0:\theta\in\Omega_0 ,\quad H_1:\theta\in\Omega_1
\quad (\Omega_0\cap\Omega_1=\varnothing,\ \Omega_0\cup\Omega_1=\Omega)$$

을 유의수준 $\alpha$에서 검정할 때, 대립가설의 각 모수 값에서의 검정력

$$\gamma_\phi(\theta_1)=E_{\theta_1}\phi(X),\quad \theta_1\in\Omega_1$$

을 크게 하는 검정이 좋은 것이다.

### 도입: 전역최강력 검정
일반적인 가설 $H_0:\theta\in\Omega_0 ,\quad H_1:\theta\in\Omega_1$ 을 검정할 때, 다음을 만족시키는 검정 $\phi^{UMP}_\alpha$를 유의수준 $\alpha$의 전역최강력 검정이라 한다.

(i) (유의수준): 귀무가설 하에서 최악의 제1종 오류가 $\alpha$ 이하
$$\max_{\theta\in\Omega_0} E_\theta \phi^{UMP}_\alpha(X)\le \alpha$$

(ii) (대립가설 전역에서 최대의 검정력)

$$E_{\theta_1}\phi^{UMP}_\alpha(X)\ge E_{\theta_1}\phi(X),
\quad \forall\theta_1\in\Omega_1,\;
\forall\phi:\max_{\theta\in\Omega_0}E_\theta\phi(X)\le\alpha$$

> 참고: MP와의 차이  
> **MP 검정(Most Powerful Test)** 는 단순가설($H_0:\theta=\theta_0$ vs $H_1:\theta=\theta_1$)에서 유의수준 $\alpha$ 하에 대립가설 한 점($\theta_1$)에서 검정력이 최대가 되는 검정이다.  
> **UMP 검정(Uniformly Most Powerful Test)**  는 복수의 대립가설($\theta\in\Omega_1$)에 대해 유의수준 $\alpha$ 하에서 대립가설 전체에 대해 항상(모든 $\theta_1\in\Omega_1$에서) 검정력이 최대가 되는 검정이다.  
> 즉, MP는 한 점에서, UMP는 대립가설 전체에서 "최강력" 조건을 만족한다는 차이가 있다.
> - MP: 단순가설(점 대 점), 특정 $\theta_1$에서만 최강력
> - UMP: 복합가설(구간 등), 대립가설 전체에서 항상 최강력
> - UMP 검정은 모든 상황에서 존재하지 않음(특히 양쪽 검정 등)

#### 예 9.2.1
정규분포 $N(\mu,1)$에서의 랜덤표본을 이용하여 유의수준 $\alpha$에서 검정할 때

$$ H_0:\mu=\mu_0 ,\quad H_1:\mu>\mu_0$$

전역최강력 검정이 아래와 같음을 보여라
$$
\phi^*(x)=
\begin{cases}
1, & \bar{x}-\mu_0\ge z_\alpha/\sqrt{n} \\
0, & \bar{x}-\mu_0< z_\alpha/\sqrt{n}
\end{cases}
$$

**풀이**  
정규분포 $N(\mu,1)$에서 표본 $X_1,\dots,X_n$을 이용하여 $H_0:\mu=\mu_0 ,\quad H_1:\mu>\mu_0$ 을 검정할 때, 네이만–피어슨 정리에 따라 최강력 검정은 가능도비  

$$\frac{pdf(x;\mu_1)}{pdf(x;\mu_0)}
= \exp\left[n(\mu_1-\mu_0) 
\left(\bar{x}-\frac{\mu_1+\mu_0}{2}\right) \right]$$

가 임계값 $k$보다 큰 경우 귀무가설을 기각한다.  
$\mu_1 > \mu_0$일 때, 가능도비는 $\bar{x}$의 증가함수이므로, 기각역은  

$$\bar{x} \ge c$$

유의수준 $\alpha$를 만족시키기 위해 $P_{\mu_0}(\bar{X} \ge c) = \alpha$ 이다. $\bar{X} \sim N(\mu_0, 1/n)$이므로  

$$P_{\mu_0}\left( \frac{\bar{X} - \mu_0}{1/\sqrt{n}} \ge \frac{c - \mu_0}{1/\sqrt{n}} \right) = \alpha \\
\frac{c - \mu_0}{1/\sqrt{n}} = z_\alpha \implies c = \mu_0 + \frac{z_\alpha}{\sqrt{n}}$$

따라서 검정함수는  

$$\phi^*(x) = \begin{cases}
1, & \bar{x} \ge \mu_0 + \frac{z_\alpha}{\sqrt{n}} \\
0, & \text{otherwise}
\end{cases}$$

이 검정은 $\mu_1$의 값에 관계없이 항상 동일하게 적용되므로, 모든 $\mu_1 > \mu_0$에 대해 대립가설에서 검정력이 최대가 되는 전역최강력 검정(UMP test)이다.

#### 예 9.2.2
정규분포 $N(\mu,1)$에서 $H_0(-):\mu\le\mu_0,\ H_1:\mu>\mu_0$ 을 검정할 때, 예 9.2.1의 검정이 유의수준 $\alpha$의 전역최강력 검정임을 보여라.

**풀이**  
예 9.2.1의 검정함수 $\phi^*$와 검정력 함수는

$$\phi^*(x) = \begin{cases}
1, & \bar{x} - \mu_0 \ge z_\alpha/\sqrt{n} \\
0, & \bar{x} - \mu_0 < z_\alpha/\sqrt{n}
\end{cases} \\
\gamma_{\phi^*}(\mu) = E_\mu[\phi^*(X)] = P_\mu\left(\bar{X} - \mu_0 \ge \frac{z_\alpha}{\sqrt{n}}\right)$$

이다. $\bar{X} \sim N(\mu, 1/n)$이므로,

$$
P_\mu\left(\bar{X} - \mu_0 \ge \frac{z_\alpha}{\sqrt{n}}\right)
= P\left(\frac{\bar{X} - \mu}{1/\sqrt{n}} \ge \frac{\frac{z_\alpha}{\sqrt{n}} + \mu_0 - \mu}{1/\sqrt{n}}\right)
= P\left(Z \ge z_\alpha + \sqrt{n}(\mu_0 - \mu)\right) \\
Z \sim N(0,1)
$$

이 함수는 $\mu$의 증가함수이므로, $\mu$가 커질수록 검정력이 커진다. 따라서

$$
\max_{\mu \le \mu_0} E_\mu[\phi^*(X)] = E_{\mu_0}[\phi^*(X)] = P_{\mu_0}\left(\bar{X} - \mu_0 \ge \frac{z_\alpha}{\sqrt{n}}\right) = P(Z \ge z_\alpha) = \alpha
$$

이제, 임의의 검정 $\phi$에 대해

$$
\{\phi : \max_{\mu \le \mu_0} E_\mu[\phi(X)] \le \alpha, \; 0 \leq \phi(X) \le 1\} \subseteq \{\phi : E_{\mu_0}[\phi(X)] \le \alpha, \; 0 \leq \phi(X) \le 1\}
$$

이므로, $\phi^*$는 유의수준 $\alpha$를 만족한다.

또한, 네이만–피어슨 정리에 의해 $\phi^*$는 $H_0: \mu = \mu_0$ 대 $H_1: \mu = \mu_1\ (\mu_1 > \mu_0)$에서 최강력 검정이므로, 모든 $\mu_1 > \mu_0$에 대해 검정력이 최대가 된다.

따라서 $\phi^*$는 $H_0: \mu \le \mu_0$ 대 $H_1: \mu > \mu_0$에 대한 유의수준 $\alpha$의 전역최강력 검정(UMP test)임을 알 수 있다.

### 정리 9.2.1 단일모수 지수족과 전역최강력 한쪽 검정
모집단 분포의 확률밀도함수가

$$
f(x;\theta) = \exp\{g(\theta)T(x)-B(\theta)+S(x)\},
\quad x\in\mathcal{X},\ \theta\in\Omega\subset\mathbb{R}
$$

와 같이 나타내어지는 단일모수 지수족이고, $g(\theta)$가 $\theta$의 증가함수일 때, 가설

$$H_0:\theta\le\theta_0 ,\quad H_1:\theta>\theta_0$$

을 유의수준 $\alpha$에서 검정한다고 하자.  
이때 다음 조건을 만족시키는 검정 $\phi^*$는 유의수준 $\alpha$의 전역최강력 검정이다.

**(a) (가능도비 검정 꼴)**  

$$
\phi^*(x)=
\begin{cases}
1, & T(x_1)+\cdots+T(x_n)>c \\
\gamma, & T(x_1)+\cdots+T(x_n)=c \\
0, & T(x_1)+\cdots+T(x_n)<c
\end{cases}
$$

**(b) (검정의 크기)**  
$E_{\theta_0}\phi^*(X)=\alpha$

#### 증명
네이만–피어슨 정리에 따라, $H_0:\theta = \theta_0$ 대 $H_1:\theta = \theta_1 > \theta_0$에서 최강력 검정은 가능도비

$$
\frac{f(x;\theta_1)}{f(x;\theta_0)} = \exp\{[g(\theta_1) - g(\theta_0)]T(x) - [B(\theta_1) - B(\theta_0)]\}$$

가 임계값 $k$보다 큰 경우 귀무가설을 기각한다.  
$g(\theta_1) > g(\theta_0)$이므로, 가능도비는 $T(x)$의 증가함수이다. 따라서 검정함수는

$$
\phi^*(x) =
\begin{cases}
1, & T(x_1) + \cdots + T(x_n) > c \\
\gamma, & T(x_1) + \cdots + T(x_n) = c \\
0, & T(x_1) + \cdots + T(x_n) < c
\end{cases}
$$

이 검정은 모든 $\theta_1 > \theta_0$에 대해 동일하게 적용되므로, 대립가설 전체에서 검정력이 최대가 되는 전역최강력 검정(UMP test)이다.

임계값 $c, \gamma$는 $E_{\theta_0}[\phi^*(X)] = \alpha$ 를 만족하도록 선택한다.

따라서 단일모수 지수족에서 $g(\theta)$가 증가함수일 때, 위와 같은 꼴의 검정이 유의수준 $\alpha$의 전역최강력 검정임이 증명된다.

#### 증명 다른방법
$\phi^*(x)$의 검정력 함수 $E_\theta[\phi^*(X)]$가 $\theta$의 증가함을 보이자.

모수공간 임의의 $\theta', \theta''$ ($\theta' < \theta''$)에 대해 $\alpha' = E_{\theta'}\phi^*(X)$라 하고, 단순가설 $H_0(\theta'):\theta = \theta'$ vs $H_1(\theta''):\theta = \theta''$을 유의수준 $\alpha'$에서 검정한다고 하자. 이때 항상 $\alpha'$의 확률로 기각하는 랜덤화검정 $\phi_{\alpha'}(x)$는 유의수준 $\alpha'$의 검정이므로, 이 경우의 최강력검정 $\phi^{MP}_{\alpha'}$보다 검정력이 작거나 같다. 즉,

$$E_{\theta''}\phi_{\alpha'}(X) \leq E_{\theta''}\phi^{MP}_{\alpha'}(X)$$

그런데 정리 9.1.2(네이만–피어슨 정리)로부터, 이 경우의 최강력검정 $\phi^{MP}_{\alpha'}$는 (a) 가능도비 검정 꼴과 (b) $E_{\theta'}[\phi^{MP}_{\alpha'}(X)] = \alpha'$에 의해 정해진다. $\phi^*$는 바로 이 조건을 만족하므로 $\phi^{MP}_{\alpha'} = \phi^*$이다.

따라서

$$E_{\theta''}\phi^*(X) = E_{\theta''}\phi^{MP}_{\alpha'}(X) \geq E_{\theta''}\phi_{\alpha'}(X) = \alpha' = E_{\theta'}\phi^*(X)$$

즉, $E_\theta[\phi^*(X)]$는 $\theta$의 증가함수임이 증명된다.

이를 증명했으므로, 
검정 $\phi^*$는 귀무가설이 $H_0: \theta \leq \theta_0$인 경우에도 유의수준 $\alpha$의 검정이다. 즉,

$$
\max_{\theta \leq \theta_0} E_\theta[\phi^*(X)] = E_{\theta_0}[\phi^*(X)] = \alpha
$$

이다. 또한

$$
\left\{ \phi : \max_{\theta \leq \theta_0} E_\theta[\phi(X)] \leq \alpha \right\} \subseteq \left\{ \phi : E_{\theta_0}[\phi(X)] \leq \alpha \right\}
$$

이므로, $\phi^*$는 $H_0: \theta \leq \theta_0$ 대 $H_1: \theta > \theta_0$에 대한 유의수준 $\alpha$의 전역최강력 검정(UMP test)임을 알 수 있다.

#### 예 9.2.3
포아송분포 $\mathrm{Poisson}(\theta)$, $0<\theta<+\infty$에서 $n=100$개의 랜덤표본을 이용하여

$$H_0:\theta\ge0.1 ,\quad H_1:\theta<0.1$$

을 검정할 때 유의수준 $\alpha=0.05$의 전역최강력 검정은

$$
\phi^*(x)=
\begin{cases}
1, & x_1+\cdots+x_n\le c-1 \\
\gamma, & x_1+\cdots+x_n=c \\
0, & x_1+\cdots+x_n\ge c+1
\end{cases}
$$

이고
$E_{\theta_0}\phi^*(X)=0.05,\quad (\theta_0=0.1)$
를 만족시키는 $c=5,\ \gamma=21/38$이다.

#### 예 9.2.4
지수분포 $\mathrm{Exp}(\theta)$, $0<\theta<+\infty$에서 랜덤표본 $X_1,\dots,X_n$을 이용하여

$$H_0:\theta\le\theta_0 ,\quad H_1:\theta>\theta_0$$

을 검정할 때 유의수준 $\alpha$의 전역최강력 검정은

$$
\phi^*(x)=
\begin{cases}
1, & x_1+\cdots+x_n\ge c \\
0, & x_1+\cdots+x_n<c
\end{cases},
\quad
E_{\theta_0}\phi^*(X)=\alpha
$$

$\theta=\theta_0$일 때

$$
\sum_{i=1}^n X_i/\theta_0\sim \mathrm{Gamma}(n,1),\quad
2\sum_{i=1}^n X_i/\theta_0\sim\chi^2(2n)$$

이므로 $c=\theta_0\chi^2_\alpha(2n)/2$ 이다.  
즉 전역최강력검정의 기각역은

$$\frac{\bar{X}}{\theta_0} \geq \frac{1}{2n} \chi^2_\alpha(2n)$$

#### 예 9.2.5  
정규분포 $N(\mu,1)$에서 랜덤표본을 이용하여 $H_0:\mu=\mu_0 ,\quad H_1:\mu\ne\mu_0$ 을 유의수준 $\alpha$에서 검정할 때, 전역최강력 검정이 존재하지 않음을 보여라.

**풀이**  
귀류법(proof by contradiction)으로 양쪽 검정에 대한 전역최강력 검정이 존재하지 않음을 보인다.

**가정:** 양쪽 검정 $H_0:\mu=\mu_0$ vs $H_1:\mu\ne\mu_0$에 대한 유의수준 $\alpha$의 전역최강력 검정 $\phi^*$가 존재한다고 하자.

**Step 1: $\mu_1 < \mu_0$인 경우**  
$\phi^*$가 UMP라면, 특히 단순가설 $H_0:\mu=\mu_0$ vs $H_1:\mu=\mu_1$ ($\mu_1 < \mu_0$)에 대해서도 유의수준 $\alpha$의 최강력(MP) 검정이어야 한다.

네이만–피어슨 정리에 의해 이 경우의 MP 검정은 다음의 형태이다. (왼쪽 한쪽 검정)

$$\phi^*(x) = 1\Big(\bar{x} \le c_1\Big)$$

**Step 2: $\mu_2 > \mu_0$인 경우**  
마찬가지로 $\phi^*$가 UMP라면, 단순가설 $H_0:\mu=\mu_0$ vs $H_1:\mu=\mu_2$ ($\mu_2 > \mu_0$)에 대해서도 유의수준 $\alpha$의 MP 검정이어야 한다.

네이만–피어슨 정리에 의해 이 경우의 MP 검정은 다음의 형태이다. (오른쪽 한쪽 검정)

$$\phi^*(x) = 1\Big(\bar{x} \ge c_2\Big)$$

**Step 3: 모순**  
같은 검정 함수 $\phi^*$가 동시에 다음 두 조건을 만족해야 한다:
- (i) $\bar{x} \le c_1$일 때만 기각 (Step 1)
- (ii) $\bar{x} \ge c_2$일 때만 기각 (Step 2)

$c_1 < \mu_0 < c_2$이므로, 하나의 함수가 "$\bar{x} \le c_1$ 또는 $\bar{x} \ge c_2$일 때만 기각"하면서 동시에 "두 구간 중 정확히 하나"에서만 기각하는 것은 불가능하다.

따라서 두 조건을 동시에 만족하는 $\phi^*$는 존재하지 않으며, 이는 UMP 검정의 정의에 모순이다.  
**결론:** 양쪽 검정 $H_0:\mu=\mu_0$ vs $H_1:\mu\ne\mu_0$에 대한 전역최강력 검정은 존재하지 않는다.

> **참고:** 이 현상은 정규분포뿐 아니라 일반적인 위치모수 모형에서도 성립한다. 양쪽 검정의 경우 "모든 대립가설에 대해 균일하게 최강력"인 검정이 존재하지 않으므로, 대신 **불편성(unbiasedness)** 같은 추가 조건을 부과하여 **전역최강력불편검정(UMPU test)** 을 찾는 방법이 사용된다. (예 9.2.6 참조)

TODO: 이 부분 교재에서도 엄밀한 증명 생략되있어서 일단 스킵함. 셤준비에 중요하면 깊이 파헤쳐야함
#### 예 9.2.6 정규분포의 평균에 대한 양쪽 검정의 성질
정규분포 $N(\theta,1)$에서 랜덤표본을 이용하여 $H_0:\theta=\theta_0 ,\quad H_1:\theta\ne\theta_0$ 을 유의수준 $\alpha$에서 검정할 때, 최대가능도비 검정

$$
\phi^*(x)=
\begin{cases}
1, & \sqrt{n}|\bar{x}-\theta_0| \ge z_{\alpha/2} \\
0, & \sqrt{n}|\bar{x}-\theta_0| < z_{\alpha/2}
\end{cases}
$$

에 대해 다음을 보여라.  
(a) 
$E_{\theta_0}[\phi^*(X)] = \alpha, \quad
\left.\frac{d}{d\theta}E_\theta[\phi^*(X)]\right|_{\theta=\theta_0} = 0$

(b) 
$\forall\phi:\ E_{\theta_0}[\phi(X)] = \alpha,\ 
\left.\frac{d}{d\theta}E_\theta[\phi(X)]\right|_{\theta=\theta_0} = 0, \quad E_{\theta_1}[\phi^*(X)] \ge E_{\theta_1}[\phi(X)],\ \forall\theta_1: \theta_1 \ne\theta_0$

**[증명]**  
(a) 검정 $\phi^*(X)$의 검정력 함수는 표준정규분포의 누적분포함수 $\Phi(z)$를 이용하여 나타낸다.

검정 $\phi^*(X)$의 기각역은 $\sqrt{n}|\bar{X}-\theta_0| \ge z_{\alpha/2}$이므로,  

$$E_\theta[\phi^*(X)] = P_\theta\left(\sqrt{n}|\bar{X}-\theta_0| \ge z_{\alpha/2}\right) = P_\theta\left(\bar{X} \ge \theta_0 + \frac{z_{\alpha/2}}{\sqrt{n}}\right) + P_\theta\left(\bar{X} \le \theta_0 - \frac{z_{\alpha/2}}{\sqrt{n}}\right) \\
= P\left(Z \ge z_{\alpha/2} - \sqrt{n}(\theta-\theta_0)\right) + P\left(Z \le -z_{\alpha/2} - \sqrt{n}(\theta-\theta_0)\right)$$

여기서 $Z \sim N(0,1)$, $\delta = \sqrt{n}(\theta-\theta_0)$로 두면, $E_\theta[\phi^*(X)] = 1-\Phi(z_{\alpha/2}-\delta) + \Phi(-z_{\alpha/2}-\delta)$  

따라서 $E_{\theta_0}[\phi^*(X)] = 1-\Phi(z_{\alpha/2}) + \Phi(-z_{\alpha/2}) = \alpha$ 이고, $\frac{d}{d\theta}E_\theta[\phi^*(X)]|_{\theta=\theta_0} = \sqrt{n}\frac{d}{d\delta}E_\theta[\phi^*(X)]|_{\delta=0} = \phi(z_{\alpha/2}) - \phi(-z_{\alpha/2}) = 0$ 이다.

(b) $E_\theta[\phi(X)]$의 $\theta$에 대한 도함수는  

> **(미적분 교환 정당화: $\frac{d}{d\theta}\int = \int \frac{\partial}{\partial\theta}$)**  
> 아래와 같은 표준 충분조건이 성립하면, 기대값 미분에서 미분과 적분의 교환이 가능하다.
>
> 1. **밀도의 $\theta$-미분 가능성** : $p_\theta(x)$가 $\theta$에 대해 미분 가능(거의 모든 $x$).
>
> 2. **지배함수 존재 (Dominated Convergence 형태)** : 어떤 적분가능 함수 $g(x)$가 존재하여, $\theta_0$의 근방에서
>    
>    $$\left|\phi(x)\frac{\partial}{\partial\theta}p_\theta(x)\right|\le g(x), \quad \int g(x)\,dx<\infty$$
>    
> 3. **검정함수의 유계성** : 보통 $0\le \phi(x)\le 1$ 이므로 $\phi$는 자동으로 bounded.
>
> 위 세 조건을 만족하므로, 따라서
> 
> $$\frac{d}{d\theta}E_\theta[\phi(X)]
> =\frac{d}{d\theta}\int \phi(x)p_\theta(x)\,dx
> =\int \phi(x)\frac{\partial}{\partial\theta}p_\theta(x)\,dx$$
> 
> 가 정당화된다.  
> (실무적으로는 3번 덕분에 2번 확인이 크게 단순화된다.)

$$\frac{d}{d\theta}E_\theta[\phi(X)] = \int \phi(x) \frac{\partial}{\partial\theta} pdf(x;\theta)\,dx
= \int \phi(x) \frac{p'_\theta(x)}{p_\theta(x)} p_\theta(x)\,dx
= E_\theta\left[\phi(X)\frac{p'_\theta(X)}{p_\theta(X)}\right] \\
\therefore \left.\frac{d}{d\theta}E_\theta[\phi(X)]\right|_{\theta=\theta_0}
= E_{\theta_0}\left[\phi(X)\frac{p'_{\theta_0}(X)}{p_{\theta_0}(X)}\right]
$$

이제, 정리9.1.2처럼 $E_{\theta_0}[\phi(X)] = \alpha$이고 $\left.\frac{d}{d\theta}E_\theta[\phi(X)]\right|_{\theta=\theta_0} = 0$을 만족하는 임의의 검정 $\phi$에 대해, $\phi^*$가 모든 $\theta \ne \theta_0$에서 검정력이 최대임을 보인다.

$\theta_1 \neq \theta_0$에 대해 미정승수 $k_1, k_2$를 도입하여 다음 식을 최대로하는 검정 $\phi^{**}$를 알아보자:  
이때, $p_\theta(x)$는 $X$의 확률밀도함수, $p'_{\theta_0}(x) = \left.\frac{\partial}{\partial\theta}p_\theta(x)\right|_{\theta=\theta_0}$  

$$
E_{\theta_1}[\phi(X)] - k_1 E_{\theta_0}[\phi(X)] - k_2 E_{\theta_0}\left[\phi(X)\frac{p'_{\theta_0}(X)}{p_{\theta_0}(X)}\right] \\ = E_{\theta_0}\left[\phi(X)\left\{\frac{p_{\theta_1}(X)}{p_{\theta_0}(X)} - k_1 - k_2 \frac{p'_{\theta_0}(X)}{p_{\theta_0}(X)}\right\}\right]
$$

이 기대값을 $\phi(X)$에 대해 최대화하려면, 각 $x$에 대해
- 괄호 $\{\}$ 안이 양수이면 $\phi(X)=1$,
- 음수이면 $\phi(X)=0$
로 하는 것이 최적이다.

따라서 최적의 검정은

$$
\phi^{**}(x) =
\begin{cases}
1, & \frac{p_{\theta_1}(x)}{p_{\theta_0}(x)} - k_1 - k_2 \frac{p'_{\theta_0}(x)}{p_{\theta_0}(x)} > 0 \\
0, & \text{otherwise}
\end{cases}
$$

따라서

$$
E_{\theta_1}[\phi^{**}(X)] - k_1 E_{\theta_0}[\phi^{**}(X)] - k_2 E_{\theta_0}\left[\phi^{**}(X)\frac{p'_{\theta_0}(X)}{p_{\theta_0}(X)}\right] \\
\geq E_{\theta_1}[\phi(X)] - k_1 E_{\theta_0}[\phi(X)] - k_2 E_{\theta_0}\left[\phi(X)\frac{p'_{\theta_0}(X)}{p_{\theta_0}(X)}\right]
$$

이므로

$$
E_{\theta_1}[\phi^{**}(X)] - E_{\theta_1}[\phi(X)]
\geq
k_1 \left( E_{\theta_0}[\phi^{**}(X)] - E_{\theta_0}[\phi(X)] \right) \\
+ k_2 \left( E_{\theta_0}\left[\phi^{**}(X)\frac{p'_{\theta_0}(X)}{p_{\theta_0}(X)}\right] - E_{\theta_0}\left[\phi(X)\frac{p'_{\theta_0}(X)}{p_{\theta_0}(X)}\right] \right)
$$

이제 $\phi^{**}$가 조건 (a)를 만족하도록 $k_1, k_2$를 적절히 정할 수 있다면, (자세한 증명 생략)

$$E_{\theta_1}[\phi^{**}(X)] \geq E_{\theta_1}[\phi(X)]$$

가 모든 $\theta_1: \theta_1 \neq \theta_0$ 및 조건 (a)를 만족하는 임의의 $\phi$에 대해 성립한다.  

마지막으로, $\phi^{**}$가 조건 (a)를 만족하도록 $k_1, k_2$를 적절히 정할 수 있다면 $\phi^{**}$가 바로 $\phi^*$로 주어지는 것을 밝힐 수 있다: 

**최적검정의 기각조건**: 라그랑주 미정승수 방법에서 최적검정 $\phi^{**}$의 기각역은

$$\frac{p_{\theta_1}(x)}{p_{\theta_0}(x)} - k_1 - k_2 \frac{p'_{\theta_0}(x)}{p_{\theta_0}(x)} > 0$$

을 정리하면

$$\exp\left\{n(\theta_1-\theta_0)\bar{x} - \frac{n}{2}(\theta_1^2-\theta_0^2)\right\} - k_1 - k_2 n(\bar{x}-\theta_0) > 0$$

$a := n(\theta_1-\theta_0)$, $b := -\frac{n}{2}(\theta_1^2-\theta_0^2)$로 두면

$$\exp\{a\bar{x}+b\} > k_1 + k_2 n(\bar{x}-\theta_0)$$

**$c_1, c_2$ 결정: 대칭성 활용**: 조건 (a)에서 $E_{\theta_0}[\phi^{**}(X)] = \alpha$를 만족하고, 조건 (b)에서 도함수 조건 $\left.\frac{d}{d\theta}E_\theta[\phi^{**}(X)]\right|_{\theta=\theta_0} = 0$ 이 성립해야 한다. 이는 검정력 함수가 $\theta=\theta_0$에서 **극값**(최솟값)을 가진다는 뜻. 정규분포의 대칭성으로부터, 양측검정의 기각역은 $\theta_0$에 대해 대칭이어야 하므로

$$\phi^{**}(x) = \begin{cases}
1, & \bar{x} \le c_1 \text{ 또는 } \bar{x} \ge c_2 \\
0, & c_1 < \bar{x} < c_2
\end{cases}$$

형태가 되며, 대칭성에 의해 $c_1 = \theta_0 - d$, $c_2 = \theta_0 + d$ (어떤 $d>0$)로 놓을 수 있다.

**임계값 확정: 유의수준 조건**: 유의수준 조건 $E_{\theta_0}[\phi^{**}(X)] = \alpha$에서

$$P_{\theta_0}(\bar{X} \le \theta_0-d) + P_{\theta_0}(\bar{X} \ge \theta_0+d) = \alpha$$

대칭성으로

$$2P_{\theta_0}\left(\bar{X} \ge \theta_0+d\right) = \alpha, \quad P_{\theta_0}\left(\frac{\bar{X}-\theta_0}{1/\sqrt{n}} \ge \sqrt{n}d\right) = \frac{\alpha}{2}$$

표준정규분포에서

$$\sqrt{n}d = z_{\alpha/2} \quad \Rightarrow \quad d = \frac{z_{\alpha/2}}{\sqrt{n}}$$

따라서

$$\boxed{c_1 = \theta_0 - \frac{z_{\alpha/2}}{\sqrt{n}}, \quad c_2 = \theta_0 + \frac{z_{\alpha/2}}{\sqrt{n}}}$$

**결론: 최적 양측검정**: 조건 (a), (b)를 모두 만족하는 불편(unbiased) 검정은

$$\phi^*(x) = \begin{cases}
1, & \sqrt{n}|\bar{x}-\theta_0| \ge z_{\alpha/2} \\
0, & \sqrt{n}|\bar{x}-\theta_0| < z_{\alpha/2}
\end{cases}$$

이며, 이는 정규분포의 평균에 대한 양측검정에서 **전역최강력불편검정(UMPU test)**

> TODO: 왜 여기선 대립가설에서 검정력이 \alpha 이상일까? 

> 참고: 전역최강력 불편검정(Uniformly Most Powerful Unbiased Test, UMPU)  
> 일반적으로, 다음 조건을 만족하는 검정 $\phi(X)$를 **유의수준 $\alpha$의 불편검정(unbiased test)** 이라 한다.
>
> $$\max_{\theta \in \Omega_0} E_\theta[\phi(X)] \leq \alpha, \quad \min_{\theta \in \Omega_1} E_\theta[\phi(X)] \geq \alpha$$
> 
> 즉, 귀무가설 하에서의 제1종 오류 확률이 $\alpha$ 이하이고, 대립가설 하에서는 검정력이 $\alpha$ 이상이 되도록 하는 검정이다.
> 
> 이러한 조건을 만족하면서, 대립가설 전체에 대해 검정력이 가장 큰 검정을 **유의수준 $\alpha$의 전역최강력 불편검정(UMPU test)** 이라 한다.
> 
> 예 9.2.6에서 조건 (a)는 이러한 불편성의 조건을 대신한 것이며, (b)로부터 해당 양쪽 검정이 전역최강력불편검정임을 알 수 있다. 이러한 UMPU 검정은 단일모수 지수족뿐만 아니라 다중모수 지수족의 경우에도 유사한 방법으로 찾을 수 있음이 알려져 있다.


## 비모수적 검정과 점근적 비교 *(Nonparametric Tests and Asymptotic Comparisons)*
모집단 분포에 특정한 형태를 가정하지 않는 경우의 검정에 대해 살펴보자. 

#### 예 9.3.1 위치모수 모형에서 부호검정
예 8.4.3에서와 같이, 모집단 분포가 연속형이고 확률밀도함수가 $f(x-\theta)$, $-\infty<\theta<+\infty$의 꼴로서 $\theta$에 관해 대칭($f(-x)=f(x)$)이고, $f$에 대응하는 누적분포함수 $F$가 순증가함수인 모형을 생각한다. 랜덤표본 $X_1,\dots,X_n$을 이용하여

$$
H_0(\theta_0):\theta=\theta_0 ,\quad H_1:\theta>\theta_0
$$

을 유의수준 $\alpha$에서 검정할 때, 통계량 $S_n = \sum_{i=1}^n I(X_i > \theta_0)$을 이용해보자. 즉, 개별 데이터를 가지고 $\theta_0$보다 큰지 작은지를 판단하여, $\theta_0$보다 큰 데이터의 개수를 세는 검정이다.  
$S_n$의 분포는

$$S_n \sim B(n, p(\theta)),\quad p(\theta) = P_\theta(X_1 > \theta_0) = 1 - F(\theta_0 - \theta)$$

이고, 위의 가설이 $p(\theta)$에 관한 가설

$$H_0(1/2):p(\theta)=1/2 ,\quad H_1:p(\theta)>1/2$$

에 대응하므로 다음과 같은 검정을 유의수준 $\alpha$의 검정으로 사용할 수 있다:

$$
\phi_s(X_1,\dots,X_n) = 
\begin{cases}
1, & S_n \ge c+1 \\
\gamma, & S_n = c \quad (0 \le \gamma \le 1) \\
0, & S_n \le c-1
\end{cases} \\
E_{\theta_0}[\phi_s(X)] = P_{\theta_0}(S_n \ge c+1) + \gamma P_{\theta_0}(S_n = c) = \alpha
$$

즉,

$$\sum_{k=c+1}^n \binom{n}{k}(1/2)^n + \gamma \binom{n}{c}(1/2)^n = \alpha$$

$\phi_s(x_1,\dots,x_n)$은 각 성분 $x_i$의 증가함수이므로

$$\max_{\theta \le \theta_0} E_\theta[\phi_s(X)] = E_{\theta_0}[\phi_s(X)] = \alpha$$

따라서 $\phi_s$는 $H_0:\theta \le \theta_0 ,\quad H_1:\theta > \theta_0$ 에 대한 유의수준 $\alpha$의 검정이다.  

> 이와 같이 $S_n$을 사용하여 연속형 분포의 중앙값에 대한 검정을 하는 방법을 **부호검정(sign test)** 이라고 한다.
> 
> 분포의 대칭성은 중앙값을 위치모수로 해석하기 위해 흔히 가정되는데, 필수는 아님.
> 부호검정은 모집단 분포에 특정한 함수 형태를 가정하지 않고 사용할 수 있는 반면(범용성, robustness가 있다), 특정 모집단에 적용하면 효율성이 떨어질 수 있다. 이런 효율성 판단에는 특정 대립가설에서의 검정력을 일정 수준으로 유지하기 위한 표본크기를 비교기준으로 한다.

#### 예 9.3.2 위치모수 모형에서 부호검정의 검정력 근사
예 9.3.1의 가설에 대해 부호검정 통계량 $S_n$의 극한분포에 이항분포의 정규근사를 적용하면

$$\frac{S_n - n p(\theta)}{\sqrt{n \sigma^2(\theta)}} \xrightarrow{d} N(0,1),\quad \sigma^2(\theta) = p(\theta)(1-p(\theta))$$

이 성립하므로, 표본크기가 클 때 유의수준 $\alpha$의 기각역은

$$
S_n \ge c_n,\quad
c_n \simeq n p(\theta_0) + \sqrt{n} \sigma(\theta_0) z_\alpha,\quad p(\theta_0) = 1/2,\ \sigma(\theta_0) = 1/2
$$

대립가설 $\theta = \theta_1 (\theta_1 > \theta_0)$에서 

$$
c_n \simeq n p(\theta_0) + \sqrt{n} \sigma(\theta_0) z_\alpha
\Rightarrow \frac{c_n - n p(\theta_1)}{\sqrt{n \sigma^2(\theta_1)}} \simeq \frac{n (p(\theta_0) - p(\theta_1)) + \sqrt{n} \sigma(\theta_0) z_\alpha}{\sqrt{n \sigma^2(\theta_1)}} \\
$$

이므로 검정력 근사는 
$$
\gamma_n(\theta_1) = P_{\theta_1}(S_n \ge c_n)
= P_{\theta_1}\left( \frac{S_n - n p(\theta_1)}{\sqrt{n \sigma^2(\theta_1)}} \ge \frac{c_n - n p(\theta_1)}{\sqrt{n \sigma^2(\theta_1)}} \right) \\
\simeq P\left( Z \ge \frac{c_n - n p(\theta_1)}{\sqrt{n \sigma^2(\theta_1)}} \right),\quad Z \sim N(0,1) \\
= 1 - \Phi\left(-\sqrt{n} \frac{p(\theta_1) - p(\theta_0)}{\sigma(\theta_1)} + \frac{\sigma(\theta_0)}{\sigma(\theta_1)} z_\alpha \right)
$$

그러므로 고정된 대립가설 $\theta = \theta_1 \ (\theta_1 > \theta_0)$에 대해

$$\lim_{n\to\infty} \gamma_n(\theta_1) = 1$$

이 성립하고, 귀무가설에 가까이 접근하는 대립가설 $\theta_{1n} \simeq \theta_0 + K/\sqrt{n} \quad (K>0)$에 대해 아래 근사식이 성립한다.

> **왜 대립가설을 $\theta_{1n} ≃ \theta_0 + K/\sqrt{n}$ 형태로 두는가?**
> 
> 이 형태의 대립가설은 **귀무가설 H0: $\theta=\theta_0$에 "가까워지는" 대립가설**, 즉 *local alternatives* (특히 **Pitman local alternatives**)를 다루기 위해 등장한다.
> 
> 1) "가까운 대립가설"이 갑자기 나오는 이유  
> 큰 표본(n→∞)에서는 대부분의 일관적인 검정이, **고정된 대립가설$(\theta_1 \neq \theta_0)$** 에 대해 검정력이 1로 수렴한다.
> - 그러면 서로 다른 검정들을 비교할 때, "어차피 다 1로 가는데?"라는 문제가 생겨 **검정력 비교가 무의미**해진다.
> - 따라서 **H0에 점점 가까워지는** 대립가설을 설정해서, 큰 표본에서도 **검정력이 0과 1 사이의 비자명한 값**으로 남도록 만들어 검정들을 비교한다.
> 
> 2) 왜 하필 $1/\sqrt n$ 스케일인 이유: 추정량/검정통계량은 보통
>     - (추정량 − $\theta_0$) 같은 차이가 **표준오차가 $O(1/\sqrt{n})$** 으로 줄어들고,
>     - 중심극한정리/점근정규성에서 $(·)×\sqrt{n}$ 스케일이 자연스럽게 등장한다.  
>
> 3) 이걸 왜 고려하나? (비교의 목적)
> "Comparison of Tests(검정 비교)" 문맥에서는 보통
>     - 같은 유의수준 α에서 어떤 검정이 **H0 근처의 미세한 차이**를 더 잘 잡는지,
>     - 즉 **점근상대효율(ARE)** 같은 개념으로 검정의 성능을 비교하려고 한다.
> 
>     - local alternatives를 쓰면 **"각 검정의 점근적 분포(shift된 정규 등) → 점근적 검정력 함수 →  효율/우수성 비교
"** 라는 흐름으로 자연스럽게 이어진다.
> 
> 요약: $\theta_{1n} = \theta_0 + K/\sqrt{n}$은 "큰 표본에서도 검정력 비교가 의미 있게 남도록" H0 근처에서의 성능을 분석하기 위한 표준 설정이다.

$$
\gamma_n(\theta_{1n}) \simeq 1 - \Phi\left(
-\sqrt{n} (\theta_{1n} - \theta_0) \dot{p}(\theta_0)/\sigma(\theta_0) + z_\alpha \right),\\
\dot{p}(\theta) = \frac{d}{d\theta}p(\theta) = \frac{d}{d\theta}(1-F(\theta_0-\theta)) = -F'(\theta_0-\theta)\cdot(-1)
= f(\theta_0 - \theta)
$$

따라서 $\gamma_n(\theta_{1n}) \simeq \gamma$가 되기 위한 표본크기는 근사적으로 아래와 같이 구할 수 있다
> 이때 $\gamma$는 상수. 통계적 검정에서 표본 크기 $n$에 따라 검정의 힘(power) 또는 임계값이 특정 값 $\gamma$에 근접하도록 하는 조건이다. 즉, 주어진 표본 크기에서 검정의 성능이 목표하는 수준 $\gamma$에 도달하도록 표본 크기를 근사적으로 결정하는 과정에 대한 논의다. 이 식은 검정의 효율성이나 신뢰도를 평가할 때 사용된다.

$$
\sqrt{n} (\theta_{1n} - \theta_0) \dot{p}(\theta_0)/\sigma(\theta_0) - z_\alpha \simeq z_{1-\gamma} \\
\therefore n \simeq \left( \frac{z_\alpha + z_{1-\gamma}}{2f(0)(\theta_{1n} - \theta_0)} \right)^2
$$

> **정리: $z_{1-\gamma}$의 의미 (표준정규 분위수)**  
> 표준정규분포 $Z\sim N(0,1)$의 누적분포함수 $\Phi$에 대해  
> $\Phi\!\left(z_{1-\gamma}\right)=1-\gamma \quad\Big(\Leftrightarrow\ P(Z\le z_{1-\gamma})=1-\gamma\Big)$  
> 따라서 오른쪽 꼬리확률은 $P(Z>z_{1-\gamma})=\gamma$이 된다.  
> 예: $\gamma=0.05$이면 $z_{0.95}\approx 1.645$.

### 정리 9.3.1 검정력의 근사와 표본크기 *(Power Approximation and Sample Size)*
실수 모수 $\theta$에 관한 가설 $H_0(\theta_0):\theta=\theta_0 ,\quad H_1:\theta>\theta_0$  
을 유의수준 $\alpha$에서 검정할 때, 크기 $n$인 랜덤표본에 기초한 검정통계량 $T_n$을 이용한 크기 $\alpha$의 기각역이

$$\sqrt{n} \frac{T_n - \mu(\theta_0)}{\sigma(\theta_0)} \ge t_n$$

이고, $T_n$에 대해

$$\sqrt{n} \frac{T_n - \mu(\theta)}{\sigma(\theta)} \xrightarrow{d} N(0,1)$$

과 같은 점근정규성이 성립한다고 하자
  - $\mu(\theta), \sigma(\theta)$는 각각 미분가능하고 연속인 함수라 가정
    - 엄밀하게는 $\theta_{1n} \simeq \theta_0 + K/\sqrt{n}$ ($K>0$)에서의 점근정규성을 필요로 하며
    - 이는 $\theta_0$근방의 열린구간에서의 $\theta$들에 대한 균등점근정규성(uniform asymptotic normality)가 성립되면 충분하다.

(a) **검정력 근사**  
귀무가설에 가까운 대립가설 $\theta_{1n} \simeq \theta_0 + K/\sqrt{n}$ ($K>0$)에서 아래와 같은 검정력의 근사식이 성립함

$$
\gamma_n(\theta_{1n}) \simeq 1 - \Phi\left(
-\sqrt{n} (\theta_{1n} - \theta_0) \dot{\mu}(\theta_0)/\sigma(\theta_0) + z_\alpha
\right),\quad
\dot{\mu}(\theta) = \frac{d}{d\theta}\mu(\theta)
$$

(b) **표본크기 근사**  
(a)에서 대립가설하의 모수 $\theta_{1n}$에서의 검정력이 $\gamma_n(\theta_{1n}) \simeq \gamma$가 되기 위한 표본크기 $N(T_n;\gamma,\theta_{1n})$근사:

$$
N(T_n;\gamma,\theta_{1n}) \simeq \left( \frac{\dot{\mu}(\theta_0)}{\sigma(\theta_0)} \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\theta_{1n} - \theta_0} \right)^2
$$

> $\frac{\dot\mu(\theta_0)}{\sigma(\theta_0)}$는 검정통계량 $T_n$의 **효율성(efficiency)** 을 나타내는 지표로 해석할 수 있다.  
> 또한, 신호 대 잡음비(signal-to-noise ratio)로도 볼 수 있다.
> - $\dot\mu(\theta_0)$는 $T_n$의 기대값이 $\theta$에 대해 얼마나 민감하게 변화하는지를 나타내며, 
> - $\sigma(\theta_0)$는 $T_n$의 표준편차로서, 검정통계량의 변동성을 나타낸다. 
> - 따라서 $\frac{\dot\mu(\theta_0)}{\sigma(\theta_0)}$가 클수록, 즉 기대값이 모수 변화에 민감하고 변동성이 낮을수록, 검정의 효율성이 높아진다고 볼 수 있다.
 
**[증명]**  
가설검정의 기각역이 $\sqrt{n}\frac{T_n-\mu(\theta_0)}{\sigma(\theta_0)}\ge t_n$
이고 검정의 크기(size)가 $\alpha$이므로

$$P_{\theta_0}\!\left(\sqrt{n}\frac{T_n-\mu(\theta_0)}{\sigma(\theta_0)}\ge t_n\right)=\alpha$$

가 되도록 $t_n$을 정한다.

**1) $t_n\simeq z_\alpha$ (임계값의 근사)**  
정리의 가정(점근정규성)으로부터

$$\sqrt{n}\frac{T_n-\mu(\theta_0)}{\sigma(\theta_0)}\ \xrightarrow{d}\ N(0,1)$$

이므로 큰 $n$에서

$$
P_{\theta_0}\!\left(\sqrt{n}\frac{T_n-\mu(\theta_0)}{\sigma(\theta_0)}\ge t_n\right)
\approx P(Z\ge t_n)=1-\Phi(t_n)
$$

왼쪽이 $\alpha$가 되게 하려면 $1-\Phi(t_n)\approx \alpha$, 즉 $t_n \approx \Phi^{-1}(1-\alpha)=z_\alpha$ 이 된다. 따라서 $t_n\simeq z_\alpha$.

**2) 고정된 대립가설 $\theta=\theta_1$에서의 검정력 근사**  
검정력은

$$
\gamma_n(\theta_1)=P_{\theta_1}\!\left(\sqrt{n}\frac{T_n-\mu(\theta_0)}{\sigma(\theta_0)}\ge t_n\right)
$$

이때

$$
\sqrt{n}\frac{T_n-\mu(\theta_0)}{\sigma(\theta_0)}\ge t_n
\iff \sqrt{n}\frac{T_n-\mu(\theta_1)}{\sigma(\theta_1)} \ge \frac{\sigma(\theta_0)}{\sigma(\theta_1)}t_n -\sqrt{n}\frac{\mu(\theta_1)-\mu(\theta_0)}{\sigma(\theta_)}
$$

이고, 점근정규성으로 $\sqrt{n}\frac{T_n-\mu(\theta_1)}{\sigma(\theta_1)}\approx Z\sim N(0,1)$ 이므로

$$
\gamma_n(\theta_1)
\approx
P\!\left(
Z \ge \frac{\sigma(\theta_0)}{\sigma(\theta_1)}t_n -\sqrt{n}\frac{\mu(\theta_1)-\mu(\theta_0)}{\sigma(\theta_1)} \right)
= 1-\Phi\!\left( \frac{\sigma(\theta_0)}{\sigma(\theta_1)}t_n
-\sqrt{n}\frac{\mu(\theta_1)-\mu(\theta_0)}{\sigma(\theta_1)}
\right)
$$

여기서 $t_n\simeq z_\alpha$를 대입하면

$$
\gamma_n(\theta_1) \simeq 1 - \Phi\left(
-\sqrt{n} \frac{\mu(\theta_1) - \mu(\theta_0)}{\sigma(\theta_1)} + \frac{\sigma(\theta_0)}{\sigma(\theta_1)} z_\alpha
\right)
$$

**3) 로컬 대립가설 $\theta_{1n}\simeq \theta_0+\dfrac{K}{\sqrt{n}}$에서의 근사**  
$\mu,\sigma$가 $\theta_0$에서 미분가능이므로 테일러 전개로

$$
\mu(\theta_{1n})-\mu(\theta_0)
= \dot\mu(\theta_0)(\theta_{1n}-\theta_0)+o(\theta_{1n}-\theta_0),
\quad \sigma(\theta_{1n})=\sigma(\theta_0)+o(1)
$$

또한 $\theta_{1n}-\theta_0=O(n^{-1/2})$이므로

$$
\sqrt{n}\big(\mu(\theta_{1n})-\mu(\theta_0)\big)
= \sqrt{n}(\theta_{1n}-\theta_0)\dot\mu(\theta_0)+o(1),
\quad \frac{\sigma(\theta_0)}{\sigma(\theta_{1n})}=1+o(1)
$$

이를 (2)의 검정력 근사식에 대입하면

$$
\gamma_n(\theta_{1n})
\simeq
1-\Phi\left(
-\sqrt{n}(\theta_{1n}-\theta_0)\frac{\dot\mu(\theta_0)}{\sigma(\theta_0)}
+z_\alpha
\right)
$$

**4) 목표 검정력 $\gamma$를 위한 표본크기 근사**  

$$
\gamma_n(\theta_{1n})\simeq \gamma
\iff 1-\Phi(A)\simeq \gamma
\iff \Phi(A)\simeq 1-\gamma
\iff A\simeq z_{1-\gamma}, \\
A=-\sqrt{n}(\theta_{1n}-\theta_0)\frac{\dot\mu(\theta_0)}{\sigma(\theta_0)} +z_\alpha$$

따라서

$$-\sqrt{n}(\theta_{1n}-\theta_0)\frac{\dot\mu(\theta_0)}{\sigma(\theta_0)}+z_\alpha \simeq -z_{1-\gamma}$$

이고, 이를 $n$에 대해 풀면

$$
\sqrt{n}\,(\theta_{1n}-\theta_0)\frac{\dot\mu(\theta_0)}{\sigma(\theta_0)}
\simeq z_\alpha + z_{1-\gamma}
$$

결국

$$
N(T_n;\gamma,\theta_{1n})
\simeq
\left(\frac{\dot\mu(\theta_0)}{\sigma(\theta_0)}\right)^{-2}
\left(\frac{z_\alpha+z_{1-\gamma}}{\theta_{1n}-\theta_0}\right)^2
$$

>$N(T_n;\gamma,\theta_{1n})$는 "검정통계량 $T_n$을 이용한 검정에서, 대립가설 $\theta_{1n}$에서의 검정력이 $\gamma$가 되도록 하는 표본크기"를 나타내는 표기다.  

### 점근상대효율성 (asymptotic relative efficiency, ARE)
실수 모수 $\theta$에 관한 가설 $H_0(\theta_0):\theta=\theta_0 ,\quad H_1:\theta>\theta_0$을 유의수준 $\alpha$에서 검정할 때, 크기 $n$인 랜덤표본에 기초한 검정통계량 $T_{in}$ ($i=1,2$)들이 정리 9.3.1의 조건을 만족한다고 하자. 대립가설 $\theta_{1n} \simeq \theta_0 + K/\sqrt{n}$에서 검정력이 $\gamma$가 되기 위한 표본크기를 $N(T_{in};\gamma,\theta_{1n})$라 할 때, 이들 표본크기의 역수의 극한값

$$
\lim_{n\to\infty} \frac{N^{-1}(T_{1n};\gamma,\theta_{1n})}{N^{-1}(T_{2n};\gamma,\theta_{1n})}
$$

을 $T_{1n}$에 의한 검정 방법 1의 $T_{2n}$에 의한 검정 방법 2에 대한 **점근상대효율성**(asymptotic relative efficiency, ARE)이라 한다.  
기호로는 $\mathrm{ARE}(T_{1n}, T_{2n})$로 나타내며, 정리 9.3.1로부터

$$
\mathrm{ARE}(T_{1n}, T_{2n}) 
= \lim_{n\to\infty} \frac{N^{-1}(T_{1n};\gamma,\theta_{1n})}{N^{-1}(T_{2n};\gamma,\theta_{1n})}
= \frac{(\dot{\mu}_1(\theta_0)/\sigma_1(\theta_0))^2}{(\dot{\mu}_2(\theta_0)/\sigma_2(\theta_0))^2}
$$

> **관례적 약기(abuse of notation)**  
> 여기서 $\mathrm{ARE}(T_{1n},T_{2n})$의 $T_{in}$은 "통계량"을 뜻하는 기호이지만, 실제로는 각 $n$에 대해 $T_{in}$으로 **정의되는 검정 절차(임계값 선택까지 포함한 크기 $\alpha$의 검정) 전체의 열** $\{\phi_{i,n}\}_{n\ge1}$을 대표해서 적는 관례적 표기이다.  
> 즉, 엄밀히는
> 
> $$\mathrm{ARE}\big(\{\phi_{1,n}\},\{\phi_{2,n}\}\big)$$
> 
> 처럼 "검정들의 열"에 대한 점근 비교이지만, 독자가 "ARE는 점근 개념이며 $n$에 따른 절차의 열을 비교한다"는 전제를 안다고 보고 교재에서는 중괄호(또는 $\{\cdot\}$ 표기)를 생략해 $\mathrm{ARE}(T_{1n},T_{2n})$로 쓴다.

> **점근상대효율성의 의의**  
> 점근상대효율성(ARE)은 두 검정 절차의 성능을 대립가설이 귀무가설에 가까워지는 상황에서 비교하는 지표로, 다음과 같은 의미를 갖는다:
> 1) **검정력 비교**: ARE는 대립가설이 귀무가설에 점점 가까워지는 상황에서 두 검정 절차의 검정력을 비교하는 지표로 사용된다. ARE가 1보다 크면 절차 1이 절차 2보다 더 효율적이라는 것을 의미한다.
> 2) **표본 크기 비교**: ARE는 두 검정 절차가 동일한 검정력을 달성하기 위해 필요한 표본 크기의 비율로 해석할 수 있다. 예를 들어, ARE가 0.5이면 절차 1이 절차 2보다 동일한 검정력을 달성하기 위해 필요한 표본 크기가 절차 2의 절반이라는 것을 의미한다.
> 3) **모수적 vs 비모수적 검정 비교**: ARE는 모수적 검정과 비모수적 검정 간의 효율성을 비교하는 데 자주 사용된다. 예를 들어, 부호검정과 t-검정의 ARE를 계산하여 두 검정의 상대적 효율성을 평가할 수 있다.

#### 예 9.3.3 부호검정의 $t$-검정에 대한 점근상대효율성
모집단 분포가 연속형이고 $\mu$에 대해 대칭이며 확률밀도함수가

$$\frac{1}{\sigma} f\left(\frac{x-\mu}{\sigma}\right),\quad -\infty<\mu<+\infty,\ \sigma>0$$

인 경우, 가설 $H_0(\mu_0):\mu=\mu_0 ,\quad H_1:\mu>\mu_0$ 을 유의수준 $\alpha\ (0<\alpha<1)$에서 검정할 때, 통계량 $S_n = \sum_{i=1}^n I(X_i > \mu_0)$ 을 이용하는 부호검정을 예 9.3.1과 같이 적용할 수 있다.  
이때 $\mu_{1n} \simeq \mu_0 + K/\sqrt{n}\ (K>0)$에서 부호검정의 검정력이 $\gamma$가 되기 위한 표본크기는

$$
N(S_n;\gamma,\mu_{1n}) \simeq \left( \frac{2f(0)}{\sigma} \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\mu_{1n} - \mu_0} \right)^2
$$

한편 모집단분포가 정규분포면 $\sqrt n(\bar X -\mu_0)/S \geq t_\alpha(n-1)$로 주어지는 t검정을 사용할 것이다. 이런 t검정의 통계량 $T_n = \sqrt n(\bar X -\mu_0)/S$에 대한 점근정규성이 성립한다.  

$$\sqrt{n}\left( \frac{\bar{X} - \mu_0}{S} - \frac{\mu_{1n} - \mu_0}{\sqrt{\mathrm{Var}(X_1)}} \right) \xrightarrow{d} N(0,1) \\
\sqrt{\mathrm{Var}(X_1)}=\sigma\left(\int_{-\infty}^{\infty} z^2 f(z)\,dz\right)^{1/2}$$

> 모평균이 $\mu_{1n} \simeq \mu_0 + K/\sqrt{n}$ 일 때의 극한분포를 뜻한다. 유도과정이 좀 김

> **로컬 대립가설 $\mu_{1n}\simeq \mu_0+K/\sqrt{n}$에서 $t$-검정 통계량의 점근정규성 유도**  
> $t$-검정 통계량을
> 
> $$T_n=\sqrt{n}\frac{\bar X-\mu_0}{S}$$
> 
> 로 두고, 로컬 대립가설을 $\mu=\mu_{1n}=\mu_0+K/\sqrt{n}$로 둔다.
> 
> **1) 표본평균의 CLT (로컬 대립가설 하에서도 동일)**  
> 
> $$\sqrt{n}\frac{\bar X-\mu_{1n}}{\tau}\ \xrightarrow{d}\ N(0,1)$$
> 
> (이유: $X_i-\mu_{1n}$은 평균 $0$, 분산 $\tau^2$이며 $n$에 따라 분포가 변하지 않으므로 CLT 적용)
> 
> **2) 표본표준편차의 일치성**  
> 
> $$S \xrightarrow{p} \tau
> \quad\Rightarrow\quad
> \frac{\tau}{S}\xrightarrow{p}1,\qquad \frac{1}{S}-\frac{1}{\tau}\xrightarrow{p}0$$
> 
> (이유: $S^2$는 $\mathrm{Var}(X_1)=\tau^2$의 일치추정량)
> 
> **3) 목표 식의 변형**  
> 
> $$A_n:=\sqrt{n}\left(\frac{\bar X-\mu_0}{S}-\frac{\mu_{1n}-\mu_0}{\tau}\right)$$
> 
> 라 하자. $\bar X-\mu_0=(\bar X-\mu_{1n})+(\mu_{1n}-\mu_0)$이므로
> 
> $$\frac{\bar X-\mu_0}{S}-\frac{\mu_{1n}-\mu_0}{\tau}
> = \frac{\bar X-\mu_{1n}}{S}
> +(\mu_{1n}-\mu_0)\left(\frac{1}{S}-\frac{1}{\tau}\right) \\
> \therefore A_n = \underbrace{\sqrt{n}\frac{\bar X-\mu_{1n}}{S}}_{(I)}
> + \underbrace{\sqrt{n}(\mu_{1n}-\mu_0)\left(\frac{1}{S}-\frac{1}{\tau}\right)}_{(II)}$$
> 
> **4) 각 항의 극한분포/확률수렴**  
> **(I)항:** 곱의 형태로 쓴다.
> 
> $$\sqrt{n}\frac{\bar X-\mu_{1n}}{S} = \left(\sqrt{n}\frac{\bar X-\mu_{1n}}{\tau}\right)\left(\frac{\tau}{S}\right)$$
>
> 여기서
> 
> $$\sqrt{n}\frac{\bar X-\mu_{1n}}{\tau}\xrightarrow{d}N(0,1),\qquad \frac{\tau}{S}\xrightarrow{p}1$$
>
> 슬럿츠키 정리에 의해 $(I)\ \xrightarrow{d}\ N(0,1)$
>
> **(II)항:** 로컬 대립가설에서 $\sqrt{n}(\mu_{1n}-\mu_0)\to K \quad(\text{상수})$  
> 또한 $\left(\frac{1}{S}-\frac{1}{\tau}\right)\xrightarrow{p}0$이므로
> 
> $$(II)\ \xrightarrow{p}\ 0$$
>
> 
> **5) 결론 (슬럿츠키 정리)**  
> 
> $$A_n=(I)+(II)\ \xrightarrow{d}\ N(0,1)$$
>
> 즉
>
> $$\boxed{ \sqrt{n}\left( \frac{\bar{X} - \mu_0}{S} - \frac{\mu_{1n} - \mu_0}{\sqrt{\mathrm{Var}(X_1)}} \right) \xrightarrow{d} N(0,1) }$$
>
> 이며 $X = \mu + \sigma Z$ ($Z$는 표준화된 확률변수)로 표현할 수 있으므로 $\mathrm{Var}(X_1) = \sigma^2 \int_{-\infty}^{\infty} z^2 f(z)\,dz$ 이다. 따라서
>
> $$\boxed{\sqrt{\mathrm{Var}(X_1)}=\sigma\left(\int_{-\infty}^{\infty} z^2 f(z)\,dz\right)^{1/2}}$$
> 
> (참고) $T_n$ 자체로 쓰면 $\sqrt{n}(\mu_{1n}-\mu_0)\to K$이므로
> 
> $$T_n =
> \sqrt{n}\frac{\bar X-\mu_{1n}}{S}
> +\sqrt{n}\frac{\mu_{1n}-\mu_0}{S}
> \Rightarrow N\!\left(\frac{K}{\tau},\,1\right)$$
>
> 의 shifted normal 형태가 된다.

따라서 $t$-검정의 표본크기 근사식은

$$
N(T_n;\gamma,\mu_{1n}) \simeq \left( \frac{1}{\sqrt{\mathrm{Var}(X_1)}} \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\mu_{1n} - \mu_0} \right)^2
$$

따라서 부호검정의 $t$-검정에 대한 점근상대효율성(ARE)은

$$
\mathrm{ARE}(S_n, T_n) = \frac{(2f(0)/\sigma)^2}{(1/\sqrt{\mathrm{Var}(X_1)})^2} = 4(f(0))^2 \int_{-\infty}^{+\infty} z^2 f(z)\,dz
$$

이며, 이는 표본중앙값의 표본평균에 대한 점근상대효율성과 같다.

**표 9.3.1 부호검정의 $t$-검정에 대한 점근상대효율성**
- 모집단 분포 $N(\mu,\sigma^2)$: $2/\pi \approx 0.636$
- 모집단 분포 $L(\mu,\sigma)$: $\pi^2/12 \approx 0.822$
- 모집단 분포 $DE(\mu,\sigma)$: $2$

부호검정처럼 모집단 분포의 형태를 가정하지 않고 사용할 수 있는 검정 방법을 **비모수적(nonparametric) 검정**이라 하며, 모집단 분포에 대한 가정이 어려운 경우에 유용하다. 대표적 비모수적 검정인 크기 순서를 이용하는 방법을 살펴보자.
#### 예 9.3.4 위치모수 모형에서 부호순위 검정통계량
모집단 분포가 연속형이고 확률밀도함수가 $f(x-\theta),\quad -\infty<\theta<+\infty$의 꼴로서 $\theta$에 대해 대칭($f(-x)=f(x)$)인 경우, 랜덤표본 $X_1,\dots,X_n$을 이용하여 $H_0(\theta_0):\theta=\theta_0 ,\quad H_1:\theta>\theta_0$을 검정한다. 이때 $|X_1-\theta_0|,\dots,|X_n-\theta_0|$을 크기순으로 나열할 때 $|X_i-\theta_0|$의 순위를 $R(|X_i-\theta_0|)$라 하며,

$$R(|X_i-\theta_0|) = 1 + \sum_{j=1}^n I(|X_j-\theta_0| < |X_i-\theta_0|)$$

이다. 다음 통계량을 **부호순위(signed rank) 검정통계량**이라 한다.

$$W_n = \sum_{i=1}^n \operatorname{sgn}(X_i-\theta_0) R(|X_i-\theta_0|)$$

여기서 $\operatorname{sgn}(x)$는 $x$의 부호 함수이다.

#### 예: 부호순위(signed rank) 통계량 계산 예시

$X=(2.1,\ -0.4,\ 1.3,\ -2.0,\ 0.7)$

| $i$ | $X_i$ | $\operatorname{sgn}(X_i)$ | $\lvert X_i\rvert$ | 순위 $R(\lvert X_i\rvert)$ | $\operatorname{sgn}(X_i)\,R(\lvert X_i\rvert)$ |
|---:|---:|:---:|---:|---:|---:|
| 1 | 2.1  | $+1$ | 2.1 | 5 | 5  |
| 2 | -0.4 | $-1$ | 0.4 | 1 | -1 |
| 3 | 1.3  | $+1$ | 1.3 | 4 | 4  |
| 4 | -2.0 | $-1$ | 2.0 | 3 | -3 |
| 5 | 0.7  | $+1$ | 0.7 | 2 | 2  |  

$W_5=5+(-1)+4+(-3)+2=7$

### 정리 9.3.2 부호순위 검정통계량의 귀무가설하의 분포
귀무가설 $H_0(\theta_0):\theta=\theta_0$ 하에서 부호순위 검정통계량의 분포는 다음과 같다.

(a)

$$
W_n \overset{d}{\equiv} \sum_{j=1}^n j S(j),\quad S(j):\text{iid},\ P(S(j)=-1)=P(S(j)=+1)=1/2
$$

(b)

$$
\frac{W_n - E_{\theta_0} W_n}{\sqrt{\mathrm{Var}_{\theta_0}(W_n)}} \xrightarrow{d} N(0,1) \\
E_{\theta_0}(W_n) = 0,\quad \mathrm{Var}_{\theta_0}(W_n) = \sum_{j=1}^n j^2 = \frac{n(n+1)(2n+1)}{6}
$$

#### 증명
증명 과정에서 부호 벡터와 순위 벡터를 각각 다음과 같이 나타내기로 한다.

$$
S=(S(1),\cdots,S(n))^t=(\mathrm{sgn}(X_1-\theta_0),\cdots,\mathrm{sgn}(X_n-\theta_0))^t, \\
R=(R(1),\cdots,R(n))^t=(R(|X_1-\theta_0|),\cdots,R(|X_n-\theta_0|))^t
$$

**(a)** 첫째로 귀무가설 $H_0(\theta_0):\theta=\theta_0$하에서 $X_1$의 분포가 $\theta_0$에 관하여 대칭이므로

$$
P_{\theta_0}(|X_1-\theta_0|\le x,\mathrm{sgn}(X_1-\theta_0)=+1) \\
= P_{\theta_0}(0<X_1-\theta_0\le x) \\
= \frac{1}{2}P_{\theta_0}(|X_1-\theta_0|\le x) \\
= P_{\theta_0}(|X_1-\theta_0|\le x)P_{\theta_0}(\mathrm{sgn}(X_1-\theta_0)=+1)
$$

$$
P_{\theta_0}(|X_1-\theta_0|\le x,\mathrm{sgn}(X_1-\theta_0)=-1) \\
= P_{\theta_0}(|X_1-\theta_0|\le x)P_{\theta_0}(\mathrm{sgn}(X_1-\theta_0)=-1)
$$

즉 귀무가설 $H_0(\theta_0):\theta=\theta_0$하에서 $|X_i-\theta_0|$와 $\mathrm{sgn}(X_i-\theta_0)$가 서로 독립이고, 부호 벡터 $S=(S(1),\cdots,S(n))^t$ 와 순위 벡터 $R=(R(1),\cdots,R(n))^t$ 는 서로 독립이다.

둘째로 $R(i)=j$일 때 $i=R^{-1}(j)$로 나타내는 역순위 벡터를 $
R^{-1}=(R^{-1}(1),\cdots,R^{-1}(n))$ 라고 하면 부호순위 검정통계량을 다음과 같이 나타낼 수 있다.

$$
W_n=\sum_{i=1}^n \mathrm{sgn}(X_i-\theta_0)R(|X_i-\theta_0|)
=\sum_{i=1}^n S(i)R(i)
=\sum_{j=1}^n S(R^{-1}(j))j
$$

한편 $X_1,\cdots,X_n$이 서로 독립이고 동일한 분포를 따르므로, ${1,2,\cdots,n}$의 임의의 치환 $\pi$에 대하여 다음이 성립함을 알 수 있다.

$$
(S(\pi^{-1}(j)))_{1\le j\le n} = (\mathrm{sgn}(X_{\pi^{-1}(j)}-\theta_0))_{1\le j\le n}
\overset{d}{\equiv}(\mathrm{sgn}(X_j-\theta_0))_{1\le j\le n} \\
=(S(j))_{1\le j\le n}
$$

또한 귀무가설 $H_0(\theta_0):\theta=\theta_0$하에서, $S=(S(1),\cdots,S(n))^t$ 와 $R=(R(1),\cdots,R(n))^t$ 의 독립성으로부터 다음이 성립함을 알 수 있다.

$$
P_{\theta_0}(S(R^{-1}(1))=s_1,\cdots,S(R^{-1}(n))=s_n) \\
=\sum_{\pi\in\Pi}P_{\theta_0}(S(R^{-1}(1))=s_1,\cdots,S(R^{-1}(n))=s_n,R=\pi) \\
=\sum_{\pi\in\Pi}P_{\theta_0}(S(\pi^{-1}(1))=s_1,\cdots,S(\pi^{-1}(n))=s_n,R=\pi) \\
=\sum_{\pi\in\Pi}P_{\theta_0}(S(\pi^{-1}(1))=s_1,\cdots,S(\pi^{-1}(n))=s_n)P_{\theta_0}(R=\pi) \\
=\sum_{\pi\in\Pi}P_{\theta_0}(S(1)=s_1,\cdots,S(n)=s_n)P_{\theta_0}(R=\pi) \\
= P_{\theta_0}(S(1)=s_1,\cdots,S(n)=s_n)
$$

따라서 귀무가설 $H_0(\theta_0):\theta=\theta_0$하에서 다음이 성립함을 알 수 있다.

$$
W_n=\sum_{j=1}^n S(R^{-1}(j))j \overset{d}{\equiv} \sum_{j=1}^n jS(j)
$$

또한 귀무가설 $H_0(\theta_0):\theta=\theta_0$하에서 $X_j$의 분포가 $\theta_0$에 관하여 대칭이므로

$$
P_{\theta_0}(S(j)=-1)=P_{\theta_0}(S(j)=+1)=1/2
$$

이고, (a)가 성립하는 것을 알 수 있다.

(b) $\sigma_n=\sqrt{\mathrm{Var}_{\theta_0}(W_n)}, \quad Z_n=W_n/\sigma_n$ 이라고 하면, (a)로부터

$$
E_{\theta_0}(W_n)=0,\qquad \sigma_n^2=\mathrm{Var}_{\theta_0}(W_n)=\sum_{j=1}^n j^2=\frac{n(n+1)(2n+1)}{6}
$$

또한 (a)로부터 귀무가설 $H_0(\theta_0):\theta=\theta_0$하에서 $Z_n$의 누율생성함수를 다음과 같이 근사할 수 있다.

$$
cgf_{Z_n}(t;\theta_0)=\sum_{j=1}^n \log{(\exp(-jt/\sigma_n)+\exp(jt/\sigma_n))/2} \\
=\sum_{j=1}^n \log\left\{1+\frac{1}{2}\frac{j^2}{\sigma_n^2}t^2+\frac{1}{4!}\frac{j^4}{\sigma_n^4}t^4+\cdots\right\} \\
=\sum_{j=1}^n \left\{\left(\frac{1}{2}\frac{j^2}{\sigma_n^2}t^2+\frac{1}{4!}\frac{j^4}{\sigma_n^4}t^4+\cdots\right)-\frac{1}{2}\left(\frac{1}{2}\frac{j^2}{\sigma_n^2}t^2+\cdots\right)^2+\cdots\right\} \\
\simeq \sum_{j=1}^n \frac{1}{2}\frac{j^2}{\sigma_n^2}t^2+\cdots \\
\simeq \frac{1}{2}t^2+\cdots
$$

따라서 귀무가설 $H_0(\theta_0):\theta=\theta_0$하에서 $W_n$의 점근정규성이 성립한다. 즉

$$
\frac{W_n-E_{\theta_0}W_n}{\sqrt{\mathrm{Var}_{\theta_0}(W_n)}}=Z_n \overset{d}{\longrightarrow} N(0,1)
$$

### 정리 9.3.3 부호순위 검정통계량의 표현
$R(|X_i-\theta_0|)$를 $|X_1-\theta_0|,\dots,|X_n-\theta_0|$의 순위라 하고, $\operatorname{sgn}(x)$를 부호 함수라 하고, 부호순위 검정통계량을 아래로 정의하자.

$$W_n = \sum_{i=1}^n \operatorname{sgn}(X_i-\theta_0) R(|X_i-\theta_0|),\quad W_n^+ = \sum_{i=1}^n \mathbf{1}(X_i-\theta_0 > 0) R(|X_i-\theta_0|)$$

그러면 아래 두 가지가 성립한다.  
(a) $W_n$과 $W_n^+$의 관계

$$W_n = 2W_n^+ - \frac{n(n+1)}{2}$$

(b) 귀무가설 $H_0(\theta_0):\theta=\theta_0$ 하에서의 분포 및 정규근사
    
$$
W_n^+ \overset{d}{\equiv} \sum_{j=1}^n j B_j,\qquad B_j \overset{iid}{\sim} \mathrm{Bernoulli}(1/2) \\
\frac{W_n^+ - n(n+1)/4}{\sqrt{n(n+1)(2n+1)/24}} \overset{d}{\to} N(0,1)\quad(n\to\infty)
$$

#### 증명
**(a)** 각 관측치에 대해

$$
\operatorname{sgn}(X_i-\theta_0)=
\begin{cases}
+1,& X_i-\theta_0>0\\
-1,& X_i-\theta_0<0
\end{cases}
= 2*\mathbf{1}(X_i-\theta_0>0)-1 \\
\begin{aligned}
\therefore W_n
&=\sum_{i=1}^n\big(2*\mathbf{1}(X_i-\theta_0>0)-1\big)\,R(|X_i-\theta_0|)\\
&=2W_n^+-\sum_{i=1}^n R(|X_i-\theta_0|).
\end{aligned}
$$

순위의 합은 항상 $1+2+\cdots+n=n(n+1)/2$ 이므로

$$W_n=2W_n^+-\frac{n(n+1)}{2}$$

**(b)** 

$$
W_n \ \overset{d}{\equiv}\ \sum_{j=1}^n j\,S(j),\quad S(j): \text{iid},\ P(S(j)=\pm1)=\tfrac12 \\
W_n^+=\frac{W_n+\frac{n(n+1)}{2}}{2}
\ \overset{d}{\equiv}\
\sum_{j=1}^n j\,\frac{S(j)+1}{2}
$$

여기서

$$B_j:=\frac{S(j)+1}{2}\in\{0,1\},\quad P(B_j=1)=P(S(j)=1)=\tfrac12$$

이므로 $B_j \overset{iid}{\sim}\mathrm{Bernoulli}(1/2)$. 따라서

$$W_n^+ \ \overset{d}{\equiv}\ \sum_{j=1}^n j\,B_j$$

또한 $E(B_j)=\tfrac12,\ \mathrm{Var}(B_j)=\tfrac14$ 이고 서로 독립이므로

$$
E_{\theta_0}(W_n^+)=\sum_{j=1}^n j\,E(B_j)=\frac12\sum_{j=1}^n j=\frac{n(n+1)}{4}, \\
\mathrm{Var}_{\theta_0}(W_n^+)=\sum_{j=1}^n j^2\,\mathrm{Var}(B_j)
=\frac14\sum_{j=1}^n j^2 =\frac{n(n+1)(2n+1)}{24}
$$

가중합 $\sum_{j=1}^n j(B_j-\tfrac12)$에 중심극한정리를 적용하면

$$
\frac{W_n^+ - n(n+1)/4}{\sqrt{n(n+1)(2n+1)/24}} \overset{d}{\to}\ N(0,1)\quad(n\to\infty)
$$

### 정리 9.3.4 한쪽 가설에 대한 부호순위 검정
정리9.3.2로부터 귀무가설 $H_0(\theta_0): \theta = \theta_0$하에서 부호순위 검정통계량 $W_n$의 분포는 모집단분포의 확률밀도함수 형태와 관계없다는 것을 알 수 있고, $W_n$의 큰 값은 대립가설에 대한 증거라 할 수 있다. 따라서  
- 모형(위치모수, 대칭): 모집단 밀도 $f(x-\theta)$, $-\infty<\theta<\infty$, $f(-x)=f(x)$
- 가설은 $H_0:\theta\le \theta_0\quad \text{vs}\quad H_1:\theta>\theta_0$
- 유의수준 $\alpha (0 \lt \alpha \lt 1)$에서 검정

부호순위 검정 $\phi_{SR}$에 대해 아래 성질이 성립한다.

(a) **(U-통계량 표현 / 단조성)** 연속형 분포에서

$$
W_n^+ = \sum\sum_{1\le i\le j\le n}\mathbf{1}\!\left(X_i+X_j>2\theta_0\right)
$$

(b) 검정력 함수 $\gamma_{\phi_{SR}}(\theta) = E_\theta[\phi_{SR}(X)]$는 $\theta$의 증가함수이고, 다음이 성립한다:

$$\max_{\theta\le \theta_0} E_\theta[\phi_{SR}(X)] = E_{\theta_0}[\phi_{SR}(X)] = \alpha$$

#### 증명
**(a)** $Y_i:=X_i-\theta_0$라 두고 $W_n^+$를 다음과 같이 변형한다:

$$
W_n^+ =\sum_{i=1}^n \mathbf{1}(Y_i>0)\,R(|Y_i|) \\
= \sum_{i=1}^n \mathbf{1}(Y_i>0)\left(1+\sum_{j=1}^n \mathbf{1}(|Y_j|<|Y_i|)\right) \\
= \sum_{i=1}^n \mathbf{1}(Y_i>0) + \sum_{i=1}^n \sum_{j=1}^n \mathbf{1}(Y_i>0, -Y_i < Y_j < Y_i) $$

정렬표본을 $Y_{(1)}\le \cdots \le Y_{(n)}$라 하면, $Y_{(i)}>0$일 때 $Y_{(j)}<Y_{(i)}$는 $j<i$와 동치다. 또한 $Y_{(i)}>0$이면 $-Y_{(i)}<0<Y_{(j)}$이므로 $-Y_{(i)} < Y_{(j)}$는 항상 성립한다. 따라서

$$= \sum_{i=1}^n \mathbf{1}(Y_{(i)}>0) + \sum_{i=1}^n \sum_{j<i} \mathbf{1}(Y_{(i)}>0, -Y_{(i)} < Y_{(j)}) $$

또한 $j<i$이고 $Y_{(i)}+Y_{(j)}>0$이면 자동으로 $Y_{(i)}>0$이어야 한다. 왜냐하면 $Y_{(j)}\le Y_{(i)}$이므로, 만약 $Y_{(i)}\le 0$라면 $Y_{(j)}\le 0$도 되어 합이 양수가 될 수 없기 때문이다.

$$
= \sum_{i=1}^n \mathbf{1}(Y_{(i)}>0) + \sum_{i=1}^n \sum_{j<i} \mathbf{1}(Y_{(i)}+Y_{(j)}>0) \\
= \sum_{i=1}^n \mathbf{1}(Y_i>0) + \sum_{i=1}^n \sum_{j<i} \mathbf{1}(Y_i+Y_j>0) $$

이제 대각선 항 $i=j$와 비대각선 항 $j<i$를 하나의 합으로 묶은 것이다. $i=j$이면 $Y_i+Y_j=2Y_i>0$이므로 $i=j$인 항은 $Y_i>0$인 항과 정확히 일치한다. 따라서

$$
= \sum\sum_{1\le i\le j\le n} \mathbf{1}(Y_i+Y_j>0)
$$

$$\therefore \boxed{
W_n^+ = \sum\sum_{1\le i\le j\le n}\mathbf{1}(X_i+X_j>2\theta_0)
}
$$

**(b)** 부호순위 검정의 정의상 $\phi_{SR}(X)$는 $W_n^+$에 대한 (비감소) 함수이다. 그리고 (a)에서

$$W_n^+=\sum_{1\le i\le j\le n}\mathbf{1}(X_i+X_j>2\theta_0)$$

로 나타났으므로, 각 항 $\mathbf{1}(X_i+X_j>2\theta_0)$는 $(X_1,\dots,X_n)$의 각 성분에 대한 증가함수이고, 합 $W_n^+$ 역시 각 성분에 대한 증가함수이다. 따라서 $\phi_{SR}(X)$도 각 성분에 대한 증가함수이다.

이제 위치모수 모형에서 $X_i=\theta+Z_i$ ($Z_i$ iid, 밀도 $f$)로 둘 수 있다. $\theta'<\theta''$에 대해 같은 $Z=(Z_1,\dots,Z_n)$로 결합(coupling)하면

$$X(\theta'')=\theta''+Z \ \ge\ \theta'+Z=X(\theta') \quad(\text{성분별})$$

그리고 $\phi_{SR}$가 증가함수이므로

$$\phi_{SR}(X(\theta''))\ge \phi_{SR}(X(\theta'))$$

양변에 기댓값을 취하면

$$E_{\theta''}[\phi_{SR}(X)]\ge E_{\theta'}[\phi_{SR}(X)]$$

즉 검정력 함수 $\gamma_{\phi_{SR}}(\theta)=E_\theta[\phi_{SR}(X)]$는 $\theta$의 증가함수이다. 그러므로

$$\max_{\theta\le \theta_0}E_\theta[\phi_{SR}(X)]=E_{\theta_0}[\phi_{SR}(X)]=\alpha$$

가 된다(임계값 $c,\gamma$를 $E_{\theta_0}[\phi_{SR}(X)]=\alpha$가 되도록 잡았으므로). 따라서 $\phi_{SR}$는 유의수준 $\alpha$의 검정이다.

### 정리 9.3.5 부호순위 검정의 점근정규성
정리9.3.4에서 알 수 있듯이, 연속형 대칭인 분포의 중앙값에 대한 한쪽가설이나 양쪽가설의 검정에 부호순위 검정을 사용할 수 있고, 이는 비모수적 검정이다. 이런 부호순위 검정의 효율성을 알아보려면 아래 정리와 같은 점근 정규성이 필요하다. 이 정리의 증명은 이 책의 수준을 넘으므로 생략한다.

(a) 점근정규성
    
$$\frac{W_n^+ - E_\theta(W_n^+)}{\sqrt{\mathrm{Var}_\theta(W_n^+)}} \overset{d}{\to} N(0,1)$$

(b) 평균/분산 근사

$$\mu(\theta) = \frac{1}{2} P_\theta(X_1 + X_2 > 2\theta_0)\\
\sigma^2(\theta) = \mathrm{Cov}_\theta\left( \mathbf{1}(X_1 + X_2 > 2\theta_0),\ \mathbf{1}(X_1 + X_3 > 2\theta_0) \right)$$

라고 하면,

$$E_\theta(W_n^+) \simeq n^2 \mu(\theta),\quad \mathrm{Var}_\theta(W_n^+) \simeq n^3 \sigma^2(\theta)$$

따라서 $W_n^+/n^2$에 대해 앞의 점근적 검정력/표본크기 근사 공식을 적용할 수 있으므로 부호순위검정의 검정력에 대한 근사식을 아래 정리와 같이 구할 수 있다.

### 정리 9.3.6 부호순위 검정의 검정력 근사와 표본크기
정리 9.3.4의 가정(대칭 위치모수 모형) 하에서, 대립가설하의 모수 $\theta_{1n} \simeq \theta_0 + \frac{K}{\sqrt{n}}\quad(K>0)$ 이 귀무가설에 근접할 때, 

(a) 검정력의 근사

$$
\gamma_{\phi_{SR}}(\theta_{1n}) \simeq 1 - \Phi\left( -\sqrt{n} (\theta_{1n} - \theta_0) \frac{\dot\mu(\theta_0)}{\sigma(\theta_0)} + z_\alpha \right) \\
\dot\mu(\theta_0) = \int_{-\infty}^{\infty} f^2(x)\,dx,\qquad \sigma^2(\theta_0) = \frac{1}{12}
$$

(b) 표본크기의 근사: $\gamma_n(\theta_{1n}) \simeq \gamma$ 이기 위해 필요한 표본크기 $N(W_n^+;\gamma,\theta_{1n})$에 대하여

$$
N(W_n^+;\gamma,\theta_{1n}) \simeq \left( \sqrt{12} \int_{-\infty}^{\infty} f^2(x)\,dx \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\theta_{1n} - \theta_0} \right)^2
$$

#### 증명
정리 9.3.5로부터 $\mu(\theta), \sigma^2(\theta_0)$를 구해보면

$$
\mu(\theta) = \frac{1}{2} \int_{-\infty}^{\infty} (1-F(2\theta_0 -2\theta -x))f(x)dx \\
\sigma^2(\theta_0) = E[(1-F(-Z))^2] - (E[1-F(-Z)])^2, \quad Z\sim f \\
= E(U^2) - (E[U])^2, \quad U\sim \mathrm{Uniform}(0,1) \\
$$

이므로 정리 9.3.1로부터 (a), (b)가 성립.

#### 예 9.3.5 부호순위 검정의 $t$-검정에 대한 점근상대효율성(ARE)
대칭 위치-척도 모형:

$$
\frac{1}{\sigma} f\left(\frac{x-\mu}{\sigma}\right),\quad -\infty<\mu<\infty,\ \sigma>0
$$

가설: $H_0(\mu_0):\mu=\mu_0\quad \text{vs}\quad H_1:\mu>\mu_0$ 을 통계량 $W_n := \sum_{i=1}^n \operatorname{sgn}(X_i-\theta_0)\,R(|X_i-\theta_0|)$을 이용하는 부호순위 검정을 (정리9.3.4)와 같이 적용할 수 있다.  
이 경우에도 로컬 대립 $\mu_{1n} \simeq \mu_0 + K/\sqrt{n}$에서 부호검정에 의함 검정력이 $\gamma$이기 위해 필요한 표본의 크기는 정리 9.3.6에서와 같이 주어진다:

$$
N(W_n;\gamma,\mu_{1n}) \simeq \left( \sqrt{12} \int f^2(z)\,dz /\sigma \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\mu_{1n} - \mu_0} \right)^2
$$

한편, (예 9.3.3)로부터 $t$-검정의 경우에 필요한 표본크기는

$$
N(T_n;\gamma,\mu_{1n}) \simeq \left( \frac{1}{\sqrt{\mathrm{Var}(X_1)}} \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\mu_{1n} - \mu_0} \right)^2\\
\sqrt{\mathrm{Var}(X_1)} = \sigma \left( \int z^2 f(z)\,dz \right)^{1/2}
$$

따라서 부호순위 검정의 t검정에 대한 점근상대효율성(ARE)은

$$
\mathrm{ARE}(\{W_n\}, \{T_n\}) = 12 \left( \int f^2(z)\,dz \right)^2 \left( \int z^2 f(z)\,dz \right)
$$

- 아래 표와 부호검정에 대한 표 9.3.1을 비교해보면 부호순위 검정의 상대효율성이 부호검정에 비해 매우 높은 것을 알 수 있다. 즉 부호순위 검정은 범용성과 더불어 효율성도 비교적 높아서 매우 유용한 검정 방법이다.
- **표 9.3.2: $\mathrm{ARE}(\{W_n\}, \{T_n\})$**
    - $N(\mu,\sigma^2)$: $3/\pi \approx 0.954$
    - $L(\mu,\sigma)$: $\pi^2/9 \approx 1.096$
    - $DE(\mu,\sigma)$: $1.5$


>## 보충: 전통 검정론의 한계와 현대 통계학의 확장
>
>**검정론만으로는 부족한 이유**  
>제7장과 제9장에서 다룬 검정 이론은 통계학의 핵심적 기초이지만, 현대 응용통계와 인과추론 분야에서는 검정 이론만으로는 충분하지 않다는 점을 이해하는 것이 중요하다.
>
>- "유의하다"와 "중요하다"의 구분
>  - $p$-값이 0.05 미만이어서 "통계적으로 유의"하다는 것과 그 결과가 "실질적으로 중요"하다는 것은 다르다. 특히 표본 크기 $n$이 크면, 아주 작고 무시할 수 있는 효과도 통계적으로 유의할 수 있다.
>
>- 기각 여부만으로는 효과를 이해하기 어렵다
>  - 검정은 "귀무가설을 기각하는가/하지 않는가"라는 이진 결정만 제공한다. 하지만 실무에서는 다음 질문들이 더 중요하다:
>    - 효과의 크기(effect size)는 얼마나 되는가?
>    - 신뢰구간은 어떻게 되는가?
>    - 실질적 유의성(practical significance)은 어느 정도인가?
>
>- 인과질문에는 단순 검정이 부족하다
>  - 상관관계 검정은 "두 변수가 서로 관련이 있는가"를 답하지만, 인과추론의 핵심 질문인 "변수 A를 개입(intervention)했을 때 변수 B에 미치는 인과효과는 얼마인가"는 답하지 못한다. 교란변수(confounder)의 통제, 역인과성(reverse causality), 선택편향(selection bias) 같은 문제를 다루려면 검정 이론 그 이상이 필요하다.
>
>- 모델 구조와 식별 가능성이 더 근본적이다
>  - 특히 인과추론에서는 다음이 전통적 검정 패러다임보다 우선한다:
>    - 데이터 생성 과정(data generating process)이 무엇인가?
>    - 관심 모수(parameter of interest)를 "식별할 수 있는가(identifiable)"?
>    - 관측 데이터에서 과연 무엇을 추정할 수 있는가?
>
>  - "0과의 통계적 차이"보다는 "무엇을 추정할 수 있고 없는가"가 더 중요한 질문이 된 것이다.
>
>**현대 통계학의 확장된 관심사**
>
>제7~9장: **전통 검정론 중심**
>- 모수를 얼마나 잘 추정할 것인가
>- 유의수준 $\alpha$에서 최강력 검정은 무엇인가
>- 충분통계량, UMP, UMPU 같은 최적성 이론
>
>**현대 응용통계학 및 인과추론**
>- 실제 데이터 생성 과정이 무엇인가
>- 모델이 데이터에 얼마나 잘 맞는가
>- 교란을 어떻게 통제할 것인가
>- 표본 외 예측(out-of-sample prediction)은 어떤가
>- 인과효과를 동정(identification)할 수 있는가
>
>이러한 확장은 검정론을 부정하거나 대체하는 것이 아니라, 그것을 포함하면서 더 넓은 맥락에서 데이터와 의사결정을 다루는 방향으로의 진화를 의미한다.

