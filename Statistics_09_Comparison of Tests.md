# 제9장 검정의 비교 *(Comparison of Tests)*

## 검정법 비교의 기준 *(Criteria for Comparing Tests)*
**[단순 가설(simple hypothesis)]**: 귀무가설과 대립가설이 각각 하나의 확률밀도함수로 주어지므로, 검정 방법의 비교 기준을 이해하기 쉽다.  
- 모집단 분포: 확률밀도함수 $f(x;\theta)$
- 모수 공간: $\theta \in \Omega = \{\theta_0, \theta_1\}$
- 랜덤표본: $X = (X_1, \dots, X_n)$

검정 문제: $H_0: \theta = \theta_0 \quad \text{vs} \quad H_1: \theta = \theta_1$

**[복합 가설(composite hypothesis)]**  
단순 가설과 달리, 귀무가설 또는 대립가설이 하나의 확률분포가 아니라 여러 개의 모수값(즉, 여러 확률분포)으로 정의되는 경우를 **복합 가설(composite hypothesis)**이라 한다.  
- 예시: $H_0: \theta \leq \theta_0$ 또는 $H_1: \theta > \theta_0$  
    이 경우 $H_0$는 $\theta$가 $\theta_0$ 이하인 모든 값을 포함하므로, 하나의 확률분포가 아니라 여러 분포의 집합이 된다.
- 일반적으로 실제 통계적 검정에서는 복합가설이 더 자주 등장한다.

> **정리:**  
> - **단순 가설(simple hypothesis):** 모수의 값이 하나로 특정됨 ($\theta = \theta_0$ 등)
> - **복합 가설(composite hypothesis):** 모수의 값이 여러 개 포함됨 ($\theta \in \Omega_0$ 등)

**[검정과 오류 확률]**  
**(1) 비랜덤화 검정**  
기각역 $C \subset \mathcal{X}^n$에 대해
$$
X \in C \Rightarrow H_0 \text{ 기각}
$$
오류 확률 (검정이 실패할 확률):
- 제1종 오류: $P_{\theta_0}(X \in C)$
- 제2종 오류: $P_{\theta_1}(X \notin C) = 1 - P_{\theta_1}(X \in C)$

**(2) 랜덤화 검정**  
검정 함수:
$$
\phi(X) = \phi(X_1,\dots,X_n), \quad 0 \le \phi(X) \le 1
$$
- $\phi(X)$: 귀무가설을 기각할 확률

오류 확률:
- 제1종 오류: $E_{\theta_0}[\phi(X)]$
- 제2종 오류: $E_{\theta_1}[1 - \phi(X)]$

**[검정 비교의 필요성]**  
두 오류 확률을 동시에 작게 만드는 검정이 이상적이지만, 한쪽 오류 확률을 줄이면 다른 쪽 오류 확률이 커지는 trade-off가 존재한다. 따라서 오류를 종합적으로 평가하는 기준이 필요하다.

### 최대오류확률과 베이지안 평균오류확률 *(Maximum Error Probability and Bayesian Average Error Probability)*
> **용어설명: 검정함수 $\phi(x)$**  
> - **검정함수 $\phi(x)$**: 관측값 $x$에 대해 귀무가설 $H_0$를 기각할 확률을 나타내는 함수. $\phi(x) \in [0,1]$의 값을 가지며, $\phi(x)=1$이면 $H_0$를 반드시 기각, $\phi(x)=0$이면 $H_0$를 절대 기각하지 않음, $0<\phi(x)<1$이면 확률적으로 기각(랜덤화 검정).
> - **비랜덤화 검정**: 기각역 $C$에 대해 $\phi(x) = \mathbb{I}(x \in C)$로 정의할 수 있으며, 이때 $E_\theta[\phi(X)] = P_\theta(X \in C)$로 기존의 오류 확률 표현과 동일하다.
> - **랜덤화 검정**: $\phi(x)$가 $[0,1]$의 임의의 값을 가질 수 있어, 관측값에 따라 확률적으로 기각 여부를 결정한다.
> 
> 검정함수 $\phi(x)$는 비랜덤화/랜덤화 검정 모두에 적용되는 일반적인 표현이다.

**(a) 최대오류확률 기준 *(Maximum Error Probability Criterion)*** 
$$
\max\left\{ E_{\theta_0}[\phi(X)],\; E_{\theta_1}[1-\phi(X)] \right\}
$$
→ 값이 작을수록 좋은 검정 (The smaller the value, the better the test)

**(b) 베이지안 평균오류확률 기준 *(Bayesian Average Error Probability Criterion)***
$$
\pi_0 E_{\theta_0}[\phi(X)] + \pi_1 E_{\theta_1}[1-\phi(X)]
$$
- 가중치: $\pi_0 + \pi_1 = 1,\quad \pi_0 > 0,\; \pi_1 > 0$
- $(\pi_0, \pi_1)$: 사전확률(prior probability)

### 정리 9.1.1 단순 가설의 베이즈 검정
단순 가설 $H_0:\theta=\theta_0 \quad\text{vs}\quad H_1:\theta=\theta_1$
에 대해, 다음 검정
$$
\phi^\pi(x) =
\begin{cases}
1, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} > \dfrac{\pi_0}{\pi_1} \\
0, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} < \dfrac{\pi_0}{\pi_1}
\end{cases}
$$
을 만족하는 검정은 베이지안 평균오류확률을 최소로 한다.

#### 증명
베이지안 평균오류확률은
$$
\pi_0 E_{\theta_0}[\phi(X)] + \pi_1 E_{\theta_1}[1-\phi(X)]
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
이므로, 이 경우 $\phi(x)=1$로 하고, 그렇지 않으면 $\phi(x)=0$으로 한다.  
즉,
$$
\phi^\pi(x) =
\begin{cases}
1, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} > \dfrac{\pi_0}{\pi_1} \\
0, & \text{otherwise}
\end{cases}
$$
가 베이지안 평균오류확률을 최소로 한다.

#### 예 9.1.1 (정규분포)
- 표본: $X_1,\dots,X_n \sim N(\mu,1)$
- 가설: $H_0:\mu=\mu_0 \quad\text{vs}\quad H_1:\mu=\mu_1$

가능도비:
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

베이지안 평균오류확률이나 최대오류확률 기준은  
귀무가설과 대립가설을 대칭적으로 취급한다.  
이는 제7장에서 다룬 전통적 검정과 다르다.

### 최강력 검정 *(Most Powerful Test)*
전통적 검정 이론에서는
- 제1종 오류 확률을 유의수준 $\alpha$ 이하로 제한하고
- 그 조건에서 검정력
    $$
    \gamma_\phi(\theta_1) = E_{\theta_1}[\phi(X)]
    $$
    을 최대화하는 검정을 찾는다.

**정의: 유의수준 $\alpha$의 최강력 검정**  
단순 가설 $H_0:\theta=\theta_0 \quad\text{vs}\quad H_1:\theta=\theta_1$에서 검정 $\phi^{MP}_\alpha(X)$가 다음을 만족하면 **최강력 검정**(Most Powerful Test, MP test)이라 한다.

- **최강력 검정(Most Powerful Test, MP test)**:  
    주어진 유의수준 $\alpha$에서, 대립가설 하에서 검정력(귀무가설을 기각할 확률)이 가능한 한 가장 큰 검정. 즉, 같은 유의수준을 만족하는 모든 검정 중에서 대립가설 하에서 귀무가설을 기각할 확률이 최대가 되는 검정. 이런 검정을 기호로 $\phi^{MP}_\alpha(X)$ 또는 $\phi^{MP}_\alpha$ 또는 $\phi^*_\alpha(X)$로 쓴다.
    
1. (유의수준)
        $$
        E_{\theta_0}[\phi^{MP}_\alpha(X)] \le \alpha \quad (0\lt \alpha \lt 1)
        $$
2. (최대 검정력)
        $$
        E_{\theta_1}[\phi^{MP}_\alpha(X)] \ge E_{\theta_1}[\phi(X)] \quad \forall \phi: E_{\theta_0}[\phi(X)] \le \alpha
        $$

### 정리 9.1.1 단순 가설의 최강력 검정 (Neyman–Pearson Lemma)
검정 (가능도비 검정 꼴)
$$
\phi^*(x)=
\begin{cases}
1, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} > k \\
\gamma, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} = k \\
0, & \dfrac{pdf(x;\theta_1)}{pdf(x;\theta_0)} < k
\end{cases} \\
(0 \leq \gamma \leq 1, k \geq 0)
$$
가
$
E_{\theta_0}[\phi^*(X)] = \alpha
$
를 만족하면, 이는 유의수준 $\alpha$의 **최강력 검정**이다.
> 일반적으로 a,b를 만족시키는 $k, \gamma$가 존재하는 것이 알려져 있다

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

라그랑주 승수 $k\ge 0$를 도입하여 목적함수:
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

#### 증명
먼저, 임의의 검정 함수 $\phi(X)$에 대해 다음 식을 살펴보자.
$$
E_{\theta_1}[\phi(X)] - k E_{\theta_0}[\phi(X)]
$$
여기서 $E_{\theta_1}[\phi(X)]$는 대립가설 $\theta_1$ 하에서의 검정력, $E_{\theta_0}[\phi(X)]$는 귀무가설 $\theta_0$ 하에서의 제1종 오류 확률, $k$는 상수이다.

확률밀도함수 $pdf(X;\theta)$를 이용해 위 식을 다음과 같이 변형할 수 있다.
$$
E_{\theta_1}[\phi(X)] = \int \phi(x) pdf(x;\theta_1) dx, \quad 
E_{\theta_0}[\phi(X)] = \int \phi(x) pdf(x;\theta_0) dx \\
E_{\theta_1}[\phi(X)] - k E_{\theta_0}[\phi(X)]
= \int \phi(x) [pdf(x;\theta_1) - k\, pdf(x;\theta_0)] dx
$$

이제 $pdf(x;\theta_0) > 0$인 부분과 $pdf(x;\theta_0) = 0$인 부분으로 나누어 생각하면,
$$
= \int_{pdf(x;\theta_0)>0} \phi(x) [pdf(x;\theta_1) - k\, pdf(x;\theta_0)] dx
+ \int_{pdf(x;\theta_0)=0} \phi(x) pdf(x;\theta_1) dx
$$
첫 번째 항은 $pdf(x;\theta_0)>0$인 부분에서 $pdf(x;\theta_1)/pdf(x;\theta_0) - k$로 쓸 수 있으므로,
$$
= \int_{pdf(x;\theta_0)>0} \phi(x) pdf(x;\theta_0) \left( \frac{pdf(x;\theta_1)}{pdf(x;\theta_0)} - k \right) dx
+ \int_{pdf(x;\theta_0)=0} \phi(x) pdf(x;\theta_1) dx
$$
이를 기대값 표기로 쓰면,
$$
E_{\theta_1}[\phi(X)] - k E_{\theta_0}[\phi(X)]
= E_{\theta_0}\left[ \phi(X) \left( \frac{pdf(X;\theta_1)}{pdf(X;\theta_0)} - k \right) \right]
+ E_{\theta_1}\left[ \phi(X) I(pdf(X;\theta_0)=0) \right]
$$
여기서 $I(\cdot)$는 지시함수이다.

이제 최강력 검정 $\phi^*(X)$와 임의의 검정 $\phi(X)$의 차이를 비교해보자.
$$
\Delta = \left( E_{\theta_1}[\phi^*(X)] - k E_{\theta_0}[\phi^*(X)] \right)
- \left( E_{\theta_1}[\phi(X)] - k E_{\theta_0}[\phi(X)] \right) \\
= E_{\theta_0}\left[ (\phi^*(X) - \phi(X)) \left( \frac{pdf(X;\theta_1)}{pdf(X;\theta_0)} - k \right) \right]
+ E_{\theta_1}\left[ (\phi^*(X) - \phi(X)) I(pdf(X;\theta_0)=0) \right]
$$
최강력 검정 $\phi^*(X)$는 가능도비 $\frac{pdf(X;\theta_1)}{pdf(X;\theta_0)}$가 $k$보다 크면 1, 작으면 0, 같으면 $\gamma$로 정의된다. 따라서 $(\phi^*(X) - \phi(X)) \left( \frac{pdf(X;\theta_1)}{pdf(X;\theta_0)} - k \right)$는 항상 0 이상이므로,
$$
\Delta \ge 0
$$
즉,
$$
E_{\theta_1}[\phi^*(X)] - k E_{\theta_0}[\phi^*(X)] \ge E_{\theta_1}[\phi(X)] - k E_{\theta_0}[\phi(X)]
$$

이제, $E_{\theta_0}[\phi(X)] \le \alpha$를 만족하는 임의의 $\phi(X)$에 대해, $k$와 $\gamma$를 적절히 선택하여 $E_{\theta_0}[\phi^*(X)] = \alpha$가 되도록 하면,
$$
E_{\theta_1}[\phi^*(X)] - E_{\theta_1}[\phi(X)] \ge k \left( E_{\theta_0}[\phi^*(X)] - E_{\theta_0}[\phi(X)] \right) = k (\alpha - E_{\theta_0}[\phi(X)]) \ge 0
$$

따라서 $\phi^*(X)$는 유의수준 $\alpha$에서 대립가설 하에서 검정력이 최대가 되는, 즉 최강력 검정임을 알 수 있다.

#### 예 9.1.2 (포아송 분포)
- 표본: $X_1,\dots,X_{100} \sim \mathrm{Poisson}(\theta)$
- 가설: $H_0:\theta=0.1 \quad\text{vs}\quad H_1:\theta=0.05$
- 유의수준: $\alpha=0.05$

가능도비:
$$
\frac{pdf(x;\theta_1)}{pdf(x;\theta_0)}
=
e^{-100(\theta_1-\theta_0)}
\left(\frac{\theta_1}{\theta_0}\right)^{\sum x_i}
=
e^5 (1/2)^{\sum x_i}
$$

따라서 최강력 검정은
$$
\phi^*(x)=
\begin{cases}
1, & \sum x_i \le c-1 \\
\gamma, & \sum x_i = c \\
0, & \sum x_i \ge c+1
\end{cases}
$$
이며,
$$
E_{\theta_0}[\phi^*(X)] = 0.05
$$
을 만족하도록 $(c, \gamma)$를 정한다.

→ 계산 결과: $c=5,\quad \gamma=\frac{21}{38}$


## 전역최강력 검정 *(Uniformly Most Powerful Tests)*
모집단 분포가 확률밀도함수 $f(x;\theta)$, $\theta \in \Omega$ 중의 하나인 경우에 랜덤표본 $X=(X_1,\dots,X_n)$을 이용하여 일반적인 가설
$$
H_0:\theta\in\Omega_0
\quad\text{vs}\quad
H_1:\theta\in\Omega_1
\quad
(\Omega_0\cap\Omega_1=\varnothing,\ \Omega_0\cup\Omega_1=\Omega)
$$
을 유의수준 $\alpha$에서 검정할 때, 대립가설의 각 모수 값에서의 검정력
$$
\gamma_\phi(\theta_1)=E_{\theta_1}\phi(X),\quad \theta_1\in\Omega_1
$$
을 크게 하는 검정이 좋은 것이다.

### 도입: 전역최강력 검정
일반적인 가설
$$
H_0:\theta\in\Omega_0
\quad\text{vs}\quad
H_1:\theta\in\Omega_1
$$
을 검정할 때, 다음을 만족시키는 검정 $\phi^{UMP}_\alpha$를 유의수준 $\alpha$의 전역최강력 검정이라 한다.

(i) (유의수준):  
귀무가설 하에서 최악의 제1종 오류가 $\alpha$ 이하.
$$
\max_{\theta\in\Omega_0} E_\theta \phi^{UMP}_\alpha(X)\le \alpha
$$

(ii) (대립가설 전역에서 최대의 검정력)
$$
E_{\theta_1}\phi^{UMP}_\alpha(X)\ge E_{\theta_1}\phi(X),
\quad
\forall\theta_1\in\Omega_1,\;
\forall\phi:\max_{\theta\in\Omega_0}E_\theta\phi(X)\le\alpha
$$

> 참고: MP와의 차이  
> **MP 검정(Most Powerful Test)** 는 단순가설($H_0:\theta=\theta_0$ vs $H_1:\theta=\theta_1$)에서 유의수준 $\alpha$ 하에 대립가설 한 점($\theta_1$)에서 검정력이 최대가 되는 검정이다.  
> **UMP 검정(Uniformly Most Powerful Test)**  는 복수의 대립가설($\theta\in\Omega_1$)에 대해 유의수준 $\alpha$ 하에서 대립가설 전체에 대해 항상(모든 $\theta_1\in\Omega_1$에서) 검정력이 최대가 되는 검정이다.  
> 즉, MP는 한 점에서, UMP는 대립가설 전체에서 "최강력" 조건을 만족한다는 차이가 있다.
> - MP: 단순가설(점 대 점), 특정 $\theta_1$에서만 최강력
> - UMP: 복합가설(구간 등), 대립가설 전체에서 항상 최강력
> - UMP 검정은 모든 상황에서 존재하지 않음(특히 양쪽 검정 등)

#### 예 9.2.1
정규분포 $N(\mu,1)$에서의 랜덤표본을 이용하여 유의수준 $\alpha$에서 검정할 때
$$
H_0:\mu=\mu_0
\quad\text{vs}\quad
H_1:\mu>\mu_0
$$
전역최강력 검정이 아래와 같음을 보여라.
$$
\phi^*(x)=
\begin{cases}
1, & \bar{x}-\mu_0\ge z_\alpha/\sqrt{n} \\
0, & \bar{x}-\mu_0< z_\alpha/\sqrt{n}
\end{cases}
$$

**풀이**  
정규분포 $N(\mu,1)$에서 표본 $X_1,\dots,X_n$을 이용하여  
$$
H_0:\mu=\mu_0 \quad\text{vs}\quad H_1:\mu>\mu_0
$$
을 검정할 때, 네이만–피어슨 정리에 따라 최강력 검정은 가능도비  
$$
\frac{pdf(x;\mu_1)}{pdf(x;\mu_0)}
=
\exp\left[
n(\mu_1-\mu_0)
\left(
\bar{x}-\frac{\mu_1+\mu_0}{2}
\right)
\right]
$$
가 임계값 $k$보다 큰 경우 귀무가설을 기각한다.

$\mu_1 > \mu_0$일 때, 가능도비는 $\bar{x}$의 증가함수이므로, 기각역은  
$$
\bar{x} \ge c
$$
유의수준 $\alpha$를 만족시키기 위해  
$$
P_{\mu_0}(\bar{X} \ge c) = \alpha
$$
이다. $\bar{X} \sim N(\mu_0, 1/n)$이므로  
$$
P_{\mu_0}\left( \frac{\bar{X} - \mu_0}{1/\sqrt{n}} \ge \frac{c - \mu_0}{1/\sqrt{n}} \right) = \alpha
$$
즉,  
$$
\frac{c - \mu_0}{1/\sqrt{n}} = z_\alpha \implies c = \mu_0 + \frac{z_\alpha}{\sqrt{n}}
$$

따라서 검정함수는  
$$
\phi^*(x) =
\begin{cases}
1, & \bar{x} \ge \mu_0 + \frac{z_\alpha}{\sqrt{n}} \\
0, & \text{otherwise}
\end{cases}
$$

이 검정은 $\mu_1$의 값에 관계없이 항상 동일하게 적용되므로, 모든 $\mu_1 > \mu_0$에 대해 대립가설에서 검정력이 최대가 되는 전역최강력 검정(UMP test)이다.

#### 예 9.2.2
정규분포 $N(\mu,1)$에서
$$
H_0(-):\mu\le\mu_0
\quad\text{vs}\quad
H_1:\mu>\mu_0
$$
을 검정할 때, 예 9.2.1의 검정이 유의수준 $\alpha$의 전역최강력 검정임을 보여라.

**풀이**  
예 9.2.1의 검정 $\phi^*$와 검정력 함수는
$$
\phi^*(x) =
\begin{cases}
1, & \bar{x} - \mu_0 \ge z_\alpha/\sqrt{n} \\
0, & \bar{x} - \mu_0 < z_\alpha/\sqrt{n}
\end{cases} \\
\gamma_{\phi^*}(\mu) = E_\mu[\phi^*(X)] = P_\mu\left(\bar{X} - \mu_0 \ge \frac{z_\alpha}{\sqrt{n}}\right)
$$
이다. $\bar{X} \sim N(\mu, 1/n)$이므로,
$$
P_\mu\left(\bar{X} - \mu_0 \ge \frac{z_\alpha}{\sqrt{n}}\right)
= P\left(\frac{\bar{X} - \mu}{1/\sqrt{n}} \ge \frac{\frac{z_\alpha}{\sqrt{n}} + \mu_0 - \mu}{1/\sqrt{n}}\right)
= P\left(Z \ge z_\alpha + \sqrt{n}(\mu_0 - \mu)\right)
$$
($Z \sim N(0,1)$)

이 함수는 $\mu$의 증가함수이므로, $\mu$가 커질수록 검정력이 커진다. 따라서
$$
\max_{\mu \le \mu_0} E_\mu[\phi^*(X)] = E_{\mu_0}[\phi^*(X)] = P_{\mu_0}\left(\bar{X} - \mu_0 \ge \frac{z_\alpha}{\sqrt{n}}\right) = P(Z \ge z_\alpha) = \alpha
$$
이제, 임의의 검정 $\phi$에 대해
$$
\{\phi : \max_{\mu \le \mu_0} E_\mu[\phi(X)] \le \alpha\} \subset \{\phi : E_{\mu_0}[\phi(X)] \le \alpha\}
$$
이므로, $\phi^*$는 유의수준 $\alpha$를 만족한다.

또한, 네이만–피어슨 정리에 의해 $\phi^*$는 $H_0: \mu = \mu_0$ 대 $H_1: \mu = \mu_1\ (\mu_1 > \mu_0)$에서 최강력 검정이므로, 모든 $\mu_1 > \mu_0$에 대해 검정력이 최대가 된다.

따라서 $\phi^*$는 $H_0: \mu \le \mu_0$ 대 $H_1: \mu > \mu_0$에 대한 유의수준 $\alpha$의 전역최강력 검정(UMP test)임을 알 수 있다.

### 정리 9.2.1 단일모수 지수족과 전역최강력 한쪽 검정
모집단 분포의 확률밀도함수가
$$
f(x;\theta)
=
\exp\{g(\theta)T(x)-B(\theta)+S(x)\},
\quad
x\in\mathcal{X},\ \theta\in\Omega\subset\mathbb{R}
$$
와 같이 나타내어지는 단일모수 지수족이고, $g(\theta)$가 $\theta$의 증가함수일 때, 가설
$$
H_0:\theta\le\theta_0
\quad\text{vs}\quad
H_1:\theta>\theta_0
$$
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
\frac{f(x;\theta_1)}{f(x;\theta_0)} = \exp\{[g(\theta_1) - g(\theta_0)]T(x) - [B(\theta_1) - B(\theta_0)]\}
$$
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
꼴이 된다.

이 검정은 모든 $\theta_1 > \theta_0$에 대해 동일하게 적용되므로, 대립가설 전체에서 검정력이 최대가 되는 전역최강력 검정(UMP test)이다.

임계값 $c, \gamma$는
$$
E_{\theta_0}[\phi^*(X)] = \alpha
$$
를 만족하도록 선택한다.

따라서 단일모수 지수족에서 $g(\theta)$가 증가함수일 때, 위와 같은 꼴의 검정이 유의수준 $\alpha$의 전역최강력 검정임이 증명된다.

#### 증명 다른방법
$\phi^*(x)$의 검정력 함수 $E_\theta[\phi^*(X)]$가 $\theta$의 증가함을 보이자.

모수공간 임의의 $\theta', \theta''$ ($\theta' < \theta''$)에 대해 $\alpha' = E_{\theta'}\phi^*(X)$라 하고, 단순가설 $H_0(\theta'):\theta = \theta'$ vs $H_1(\theta''):\theta = \theta''$을 유의수준 $\alpha'$에서 검정한다고 하자. 이때 항상 $\alpha'$의 확률로 기각하는 랜덤화검정 $\phi_{\alpha'}(x)$는 유의수준 $\alpha'$의 검정이므로, 이 경우의 최강력검정 $\phi^{MP}_{\alpha'}$보다 검정력이 작거나 같다. 즉,
$$
E_{\theta''}\phi_{\alpha'}(X) \leq E_{\theta''}\phi^{MP}_{\alpha'}(X)
$$

그런데 정리 9.1.2(네이만–피어슨 정리)로부터, 이 경우의 최강력검정 $\phi^{MP}_{\alpha'}$는 (a) 가능도비 검정 꼴과 (b) $E_{\theta'}[\phi^{MP}_{\alpha'}(X)] = \alpha'$에 의해 정해진다. $\phi^*$는 바로 이 조건을 만족하므로 $\phi^{MP}_{\alpha'} = \phi^*$이다.

따라서
$$
E_{\theta''}\phi^*(X) = E_{\theta''}\phi^{MP}_{\alpha'}(X) \geq E_{\theta''}\phi_{\alpha'}(X) = \alpha' = E_{\theta'}\phi^*(X)
$$
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
$$
H_0:\theta\ge0.1
\quad\text{vs}\quad
H_1:\theta<0.1
$$
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
$$
H_0:\theta\le\theta_0
\quad\text{vs}\quad
H_1:\theta>\theta_0
$$
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
2\sum_{i=1}^n X_i/\theta_0\sim\chi^2(2n)
$$
이므로 $c=\theta_0\chi^2_\alpha(2n)/2$ 이다.  
즉 전역최강력검정의 기각역은
$$
\frac{\bar{X}}{\theta_0} \geq \frac{1}{2n} \chi^2_\alpha(2n)
$$

#### 예 9.2.5  
정규분포 $N(\mu,1)$에서 랜덤표본을 이용하여  
$$
H_0:\mu=\mu_0 \quad\text{vs}\quad H_1:\mu\ne\mu_0
$$
을 유의수준 $\alpha$에서 검정할 때, 전역최강력 검정이 존재하지 않음을 보여라.

**풀이**  
전역최강력 검정(UMP test)이 존재한다면, 모든 $\mu_1 < \mu_0 < \mu_2$에 대해  
$$
H_0:\mu=\mu_0\ \text{vs}\ H_1:\mu=\mu_1, \qquad
H_0:\mu=\mu_0\ \text{vs}\ H_1:\mu=\mu_2
$$
각각에 대해 유의수준 $\alpha$의 최강력 검정(MP test)이 되어야 한다.

정규분포에서 네이만–피어슨 정리에 따라,  
- $H_0:\mu=\mu_0$ vs $H_1:\mu=\mu_1$ ($\mu_1<\mu_0$)의 최강력 검정은 $\bar{X} \le c_1$ 꼴,
- $H_0:\mu=\mu_0$ vs $H_1:\mu=\mu_2$ ($\mu_2>\mu_0$)의 최강력 검정은 $\bar{X} \ge c_2$ 꼴이다.

즉, 한쪽 검정에서는 $\bar{X}$가 작을 때(왼쪽) 또는 클 때(오른쪽)만 기각한다.  
양쪽 검정에서는 $\bar{X}$가 너무 작거나 너무 클 때 모두 기각해야 한다.  
따라서 두 경우를 모두 만족하려면  
$$
\phi^*(x) =
\begin{cases}
1, & \bar{x} \le c_1 \text{ 또는 } \bar{x} \ge c_2 \\
0, & c_1 < \bar{x} < c_2
\end{cases}
$$
꼴이어야 한다.

문제는, 각각의 대립가설($\mu_1 < \mu_0$ 또는 $\mu_2 > \mu_0$)에 따라 최강력 검정의 임계값($c_1$, $c_2$)이 달라진다는 점이다.  
즉, 모든 $\mu_1 < \mu_0 < \mu_2$에 대해 동시에 유의수준 $\alpha$를 만족시키는 $c_1$, $c_2$를 정할 수 없다.

결론적으로, 양쪽 검정에서는 모든 대립가설에 대해 항상 최강력인(UMP) 검정이 존재하지 않는다.  
따라서 전역최강력 검정은 존재하지 않는다.

#### 예 9.2.6 정규분포의 평균에 대한 양쪽 검정의 성질
정규분포 $N(\theta,1)$에서 랜덤표본을 이용하여
$$
H_0:\theta=\theta_0 \quad\text{vs}\quad H_1:\theta\ne\theta_0
$$
을 유의수준 $\alpha$에서 검정할 때, 최대가능도비 검정
$$
\phi^*(x)=
\begin{cases}
1, & \sqrt{n}|\bar{x}-\theta_0| \ge z_{\alpha/2} \\
0, & \sqrt{n}|\bar{x}-\theta_0| < z_{\alpha/2}
\end{cases}
$$
에 대해 다음을 보여라.

(a)
$E_{\theta_0}[\phi^*(X)] = \alpha, \qquad
\left.\frac{d}{d\theta}E_\theta[\phi^*(X)]\right|_{\theta=\theta_0} = 0$

(b)
$\forall\phi:\ E_{\theta_0}[\phi(X)] = \alpha,\ 
\left.\frac{d}{d\theta}E_\theta[\phi(X)]\right|_{\theta=\theta_0} = 0
\implies
E_\theta[\phi^*(X)] \ge E_\theta[\phi(X)],\ \forall\theta\ne\theta_0$

**[증명]**  
(a)
검정 $\phi^*(X)$의 기각역은 $\sqrt{n}|\bar{X}-\theta_0| \ge z_{\alpha/2}$이므로,  
$$
E_\theta[\phi^*(X)] = P_\theta\left(\sqrt{n}|\bar{X}-\theta_0| \ge z_{\alpha/2}\right) \\
= P_\theta\left(\bar{X} \ge \theta_0 + \frac{z_{\alpha/2}}{\sqrt{n}}\right) + P_\theta\left(\bar{X} \le \theta_0 - \frac{z_{\alpha/2}}{\sqrt{n}}\right) \\
= P\left(Z \ge z_{\alpha/2} - \sqrt{n}(\theta-\theta_0)\right) + P\left(Z \le -z_{\alpha/2} - \sqrt{n}(\theta-\theta_0)\right)
$$
여기서 $Z \sim N(0,1)$, $\delta = \sqrt{n}(\theta-\theta_0)$로 두면,
$$
E_\theta[\phi^*(X)] = 1-\Phi(z_{\alpha/2}-\delta) + \Phi(-z_{\alpha/2}-\delta)
$$
따라서 $E_{\theta_0}[\phi^*(X)] = 1-\Phi(z_{\alpha/2}) + \Phi(-z_{\alpha/2}) = \alpha$이고,  
또한 $\left.\frac{d}{d\theta}E_\theta[\phi^*(X)]\right|_{\theta=\theta_0} = 0$임을 알 수 있다.

(b)
$E_\theta[\phi(X)]$의 $\theta$에 대한 도함수는  
$$
\frac{d}{d\theta}E_\theta[\phi(X)] = \int \phi(x) \frac{\partial}{\partial\theta}pdf(x;\theta)\,dx
= \int \phi(x) \frac{p'_\theta(x)}{p_\theta(x)} p_\theta(x)\,dx
= E_\theta\left[\phi(X)\frac{p'_\theta(X)}{p_\theta(X)}\right]
$$
따라서
$$
\left.\frac{d}{d\theta}E_\theta[\phi(X)]\right|_{\theta=\theta_0}
= E_{\theta_0}\left[\phi(X)\frac{p'_{\theta_0}(X)}{p_{\theta_0}(X)}\right]
$$

이제, $E_{\theta_0}[\phi(X)] = \alpha$이고 $\left.\frac{d}{d\theta}E_\theta[\phi(X)]\right|_{\theta=\theta_0} = 0$을 만족하는 임의의 검정 $\phi$에 대해, $\phi^*$가 모든 $\theta \ne \theta_0$에서 검정력이 최대임을 보인다.
조건 (a), (b)를 만족하는 검정 $\phi^*$가 전역최강력임을 직접 증명해보자.

$\theta_1 \neq \theta_0$에 대해 미정승수 $k_1, k_2$를 도입하여 다음 식을 고려한다:
$$
E_{\theta_1}[\phi(X)] - k_1 E_{\theta_0}[\phi(X)] - k_2 E_{\theta_0}\left[\phi(X)\frac{p'_{\theta_0}(X)}{p_{\theta_0}(X)}\right]
$$
여기서 $p_\theta(x)$는 $X$의 확률밀도함수, $p'_{\theta_0}(x) = \left.\frac{\partial}{\partial\theta}p_\theta(x)\right|_{\theta=\theta_0}$.

이 식은
$$
E_{\theta_0}\left[\phi(X)\left\{\frac{p_{\theta_1}(X)}{p_{\theta_0}(X)} - k_1 - k_2 \frac{p'_{\theta_0}(X)}{p_{\theta_0}(X)}\right\}\right]
$$
로 쓸 수 있다.

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

이제 $\phi^{**}$가 조건 (a) $E_{\theta_0}[\phi(X)] = \alpha$, (b) $\left.\frac{d}{d\theta}E_\theta[\phi(X)]\right|_{\theta=\theta_0} = 0$을 만족하도록 $k_1, k_2$를 적절히 정할 수 있다면,
$$
E_{\theta_1}[\phi^{**}(X)] \geq E_{\theta_1}[\phi(X)]
$$
가 모든 $\theta_1 \neq \theta_0$ 및 조건 (a), (b)를 만족하는 임의의 $\phi$에 대해 성립한다.  
즉, $\phi^{**}$가 바로 $\phi^*$와 같은 꼴이므로, $\phi^*$가 전역최강력불편검정임이 증명된다.
$$
E_\theta[\phi^*(X)] \ge E_\theta[\phi(X)],\quad \forall \theta \ne \theta_0
$$

> 참고: 전역최강력 불편검정(Uniformly Most Powerful Unbiased Test, UMPU)  
> 일반적으로, 다음 조건을 만족하는 검정 $\phi(X)$를 **유의수준 $\alpha$의 불편검정(unbiased test)** 이라 한다.
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
모집단 분포가 연속형이고 확률밀도함수가 $f(x-\theta)$, $-\infty<\theta<+\infty$의 꼴로서 $\theta$에 관해 대칭($f(-x)=f(x)$)이고, $f$에 대응하는 누적분포함수 $F$가 순증가함수인 모형을 생각한다. 랜덤표본 $X_1,\dots,X_n$을 이용하여
$$
H_0(\theta_0):\theta=\theta_0 \quad\text{vs}\quad H_1:\theta>\theta_0
$$
을 유의수준 $\alpha$에서 검정할 때, 통계량 $S_n = \sum_{i=1}^n I(X_i > \theta_0)$을 이용해보자.

$S_n$의 분포는
$$
S_n \sim B(n, p(\theta)),\quad p(\theta) = P_\theta(X_1 > \theta_0) = 1 - F(\theta_0 - \theta)
$$
이고, 위의 가설이 $p(\theta)$에 관한 가설
$$
H_0(1/2):p(\theta)=1/2 \quad\text{vs}\quad H_1:p(\theta)>1/2
$$
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
$$
\sum_{k=c+1}^n \binom{n}{k}(1/2)^n + \gamma \binom{n}{c}(1/2)^n = \alpha
$$

$\phi_s(x_1,\dots,x_n)$은 각 성분 $x_i$의 증가함수이므로
$$
\max_{\theta \le \theta_0} E_\theta[\phi_s(X)] = E_{\theta_0}[\phi_s(X)] = \alpha
$$
따라서 $\phi_s$는 $H_0:\theta \le \theta_0 \quad\text{vs}\quad H_1:\theta > \theta_0$ 에 대한 유의수준 $\alpha$의 검정이다.  

> 이와 같이 $S_n$을 사용하여 연속형 분포의 중앙값에 대한 검정을 하는 방법을 **부호검정(sign test)** 이라고 한다.
> 
> 분포의 대칭성은 중앙값을 위치모수로 해석하기 위해 흔히 가정되는데, 필수는 아님.
> 부호검정은 모집단 분포에 특정한 함수 형태를 가정하지 않고 사용할 수 있는 반면(범용성, robustness가 있다), 특정 모집단에 적용하면 효율성이 떨어질 수 있다. 이런 효율성 판단에는 특정 대립가설에서의 검정력을 일정 수준으로 유지하기 위한 표본크기를 비교기준으로 한다.

#### 예 9.3.2 위치모수 모형에서 부호검정의 검정력 근사
예 9.3.1의 가설에 대해 부호검정 통계량 $S_n$의 극한분포에 이항분포의 정규근사를 적용하면
$$
\frac{S_n - n p(\theta)}{\sqrt{n \sigma^2(\theta)}} \xrightarrow{d} N(0,1),\quad \sigma^2(\theta) = p(\theta)(1-p(\theta))
$$
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
그러므로 고정된 대립가설 $\theta = \theta_1$에 대해
$$
\lim_{n\to\infty} \gamma_n(\theta_1) = 1
$$

> **왜 대립가설을 $θ_{1n} ≃ θ_0 + K/√n$ 형태로 두는가? (문맥/흐름 정리)**
> 
> 이 형태의 대립가설은 **귀무가설 H0: θ=θ0에 “가까워지는” 대립가설**, 즉 *local alternatives* (특히 **Pitman local alternatives**)를 다루기 위해 등장한다.
> 
> 1) “가까운 대립가설”이 갑자기 나오는 이유  
> 큰 표본(n→∞)에서는 대부분의 일관적인 검정이, **고정된 대립가설(θ1 ≠ θ0)** 에 대해 검정력이 1로 수렴한다.
> - 그러면 서로 다른 검정들을 비교할 때, “어차피 다 1로 가는데?”라는 문제가 생겨 **검정력 비교가 무의미**해진다.
> - 따라서 **H0에 점점 가까워지는** 대립가설을 설정해서, 큰 표본에서도 **검정력이 0과 1 사이의 비자명한 값**으로 남도록 만들어 검정들을 비교한다.
> 
> 2) 왜 하필 $1/\sqrt n$ 스케일인가?  
> 추정량/검정통계량은 보통
> - (추정량 − θ0) 같은 차이가 **표준오차가 O(1/√n)** 으로 줄어들고,
> - 중심극한정리/점근정규성에서 (·)×√n 스케일이 자연스럽게 등장한다.
> 
> 그래서 대립가설의 차이도
> - $θ_{1n} − θ0 = K/√n$ 로 두면,
> - $√n(θ_{1n} − θ0) = K$ 가 **상수로 남아** 검정통계량의 점근분포가
>     - H0에서는 평균 0,
>     - local alternative에서는 평균이 K만큼 이동한 형태(“shift”)가 되어,
>     **점근적 검정력(power) 근사식**을 깔끔하게 얻을 수 있다.
> 
> > 더 빠르게 가까워지면(예: 1/n) 차이를 검정이 거의 못 느껴 power≈α로 가고,  
> > 더 느리게 가까워지면(예: 상수 차이) power→1로 가서 비교가 다시 무의미해진다.  
> > 1/√n은 “비자명한(0<power<1)” 비교가 가능해지는 대표적인 임계 스케일이다.
> 
> ### 3) 이걸 왜 고려하나? (비교의 목적)
> “Comparison of Tests(검정 비교)” 문맥에서는 보통
> - 같은 유의수준 α에서 어떤 검정이 **H0 근처의 미세한 차이**를 더 잘 잡는지,
> - 즉 **점근상대효율(ARE)** 같은 개념으로 검정의 성능을 비교하려고 한다.
> 
> local alternatives를 쓰면 각 검정의
> - 점근적 분포(shift된 정규 등) → 점근적 검정력 함수 →  효율/우수성 비교  
> 라는 흐름으로 자연스럽게 이어진다.
> 
> 요약: $θ_{1n} = θ0 + K/√n$은 “큰 표본에서도 검정력 비교가 의미 있게 남도록” H0 근처에서의 성능을 분석하기 위한 표준 설정이다.

귀무가설에 가까이 접근하는 대립가설 $\theta_{1n} \simeq \theta_0 + K/\sqrt{n} \quad (K>0)$에 대해 아래 근사식이 성립한다.
$$
\gamma_n(\theta_{1n}) \simeq 1 - \Phi\left(
-\sqrt{n} (\theta_{1n} - \theta_0) \dot{p}(\theta_0)/\sigma(\theta_0) + z_\alpha
\right),\quad
\dot{p}(\theta) = \frac{d}{d\theta}p(\theta) = f(\theta_0 - \theta)
$$
따라서 $\gamma_n(\theta_{1n}) \simeq \gamma$가 되기 위한 표본크기는 근사적으로 아래와 같이 구할 수 있다
  - 이때 $\gamma$는 상수. 통계적 검정에서 표본 크기 $n$에 따라 검정의 힘(power) 또는 임계값이 특정 값 $\gamma$에 근접하도록 하는 조건이다. 즉, 주어진 표본 크기에서 검정의 성능이 목표하는 수준 $\gamma$에 도달하도록 표본 크기를 근사적으로 결정하는 과정에 대한 논의다. 이 식은 검정의 효율성이나 신뢰도를 평가할 때 사용된다.
$$
\sqrt{n} (\theta_{1n} - \theta_0) \dot{p}(\theta_0)/\sigma(\theta_0) - z_\alpha \simeq z_{1-\gamma} \\
\therefore n = \left( \frac{z_\alpha + z_{1-\gamma}}{2f(0)(\theta_{1n} - \theta_0)} \right)^2
$$
> **정리: $z_{1-\gamma}$의 의미 (표준정규 분위수)**  
> 표준정규분포 $Z\sim N(0,1)$의 누적분포함수 $\Phi$에 대해  
> $$\Phi\!\left(z_{1-\gamma}\right)=1-\gamma \quad\Big(\Leftrightarrow\ P(Z\le z_{1-\gamma})=1-\gamma\Big)$$
> 따라서 오른쪽 꼬리확률은  
> $$P(Z>z_{1-\gamma})=\gamma$$
> 예: $\gamma=0.05$이면 $z_{0.95}\approx 1.645$.

### 정리 9.3.1 검정력의 근사와 표본크기 *(Power Approximation and Sample Size)*
실수 모수 $\theta$에 관한 가설 $H_0(\theta_0):\theta=\theta_0 \quad\text{vs}\quad H_1:\theta>\theta_0$  
을 유의수준 $\alpha$에서 검정할 때, 크기 $n$인 랜덤표본에 기초한 검정통계량 $T_n$을 이용한 크기 $\alpha$의 기각역이
$$
\sqrt{n} \frac{T_n - \mu(\theta_0)}{\sigma(\theta_0)} \ge t_n
$$
이고, $T_n$에 대해
$$
\sqrt{n} \frac{T_n - \mu(\theta)}{\sigma(\theta)} \xrightarrow{d} N(0,1)
$$
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

**[증명]**  
가설검정의 기각역이 $\sqrt{n}\frac{T_n-\mu(\theta_0)}{\sigma(\theta_0)}\ge t_n$
이고 검정의 크기(size)가 $\alpha$이므로
$$
P_{\theta_0}\!\left(\sqrt{n}\frac{T_n-\mu(\theta_0)}{\sigma(\theta_0)}\ge t_n\right)=\alpha
$$
가 되도록 $t_n$을 정한다.

**1) $t_n\simeq z_\alpha$ (임계값의 근사)**  
정리의 가정(점근정규성)으로부터
$$
\sqrt{n}\frac{T_n-\mu(\theta_0)}{\sigma(\theta_0)}\ \xrightarrow{d}\ N(0,1)
$$
이므로 큰 $n$에서
$$
P_{\theta_0}\!\left(\sqrt{n}\frac{T_n-\mu(\theta_0)}{\sigma(\theta_0)}\ge t_n\right)
\approx P(Z\ge t_n)=1-\Phi(t_n).
$$
왼쪽이 $\alpha$가 되게 하려면 $1-\Phi(t_n)\approx \alpha$, 즉
$$
t_n \approx \Phi^{-1}(1-\alpha)=z_\alpha
$$
이 된다. 따라서 $t_n\simeq z_\alpha$.

**2) 고정된 대립가설 $\theta=\theta_1$에서의 검정력 근사**  
검정력은
$$
\gamma_n(\theta_1)=P_{\theta_1}\!\left(\sqrt{n}\frac{T_n-\mu(\theta_0)}{\sigma(\theta_0)}\ge t_n\right)
$$
이때
$$
\sqrt{n}\frac{T_n-\mu(\theta_0)}{\sigma(\theta_0)}\ge t_n
\iff
\sqrt{n}\frac{T_n-\mu(\theta_1)}{\sigma(\theta_1)}
\ge
\frac{\sigma(\theta_0)}{\sigma(\theta_1)}t_n
-\sqrt{n}\frac{\mu(\theta_1)-\mu(\theta_0)}{\sigma(\theta_1)}.
$$
이고, 점근정규성으로 $\sqrt{n}\frac{T_n-\mu(\theta_1)}{\sigma(\theta_1)}\approx Z\sim N(0,1)$ 이므로
$$
\gamma_n(\theta_1)
\approx
P\!\left(
Z \ge
\frac{\sigma(\theta_0)}{\sigma(\theta_1)}t_n
-\sqrt{n}\frac{\mu(\theta_1)-\mu(\theta_0)}{\sigma(\theta_1)}
\right)
=
1-\Phi\!\left(
\frac{\sigma(\theta_0)}{\sigma(\theta_1)}t_n
-\sqrt{n}\frac{\mu(\theta_1)-\mu(\theta_0)}{\sigma(\theta_1)}
\right).
$$
여기서 $t_n\simeq z_\alpha$를 대입하면
$$
\gamma_n(\theta_1) \simeq 1 - \Phi\left(
-\sqrt{n} \frac{\mu(\theta_1) - \mu(\theta_0)}{\sigma(\theta_1)} + \frac{\sigma(\theta_0)}{\sigma(\theta_1)} z_\alpha
\right).
$$

**3) 로컬 대립가설 $\theta_{1n}\simeq \theta_0+\dfrac{K}{\sqrt{n}}$에서의 근사**  
$\mu,\sigma$가 $\theta_0$에서 미분가능이므로 테일러 전개로
$$
\mu(\theta_{1n})-\mu(\theta_0)
=
\dot\mu(\theta_0)(\theta_{1n}-\theta_0)+o(\theta_{1n}-\theta_0),
\qquad
\sigma(\theta_{1n})=\sigma(\theta_0)+o(1).
$$
또한 $\theta_{1n}-\theta_0=O(n^{-1/2})$이므로
$$
\sqrt{n}\big(\mu(\theta_{1n})-\mu(\theta_0)\big)
=
\sqrt{n}(\theta_{1n}-\theta_0)\dot\mu(\theta_0)+o(1),
\qquad
\frac{\sigma(\theta_0)}{\sigma(\theta_{1n})}=1+o(1).
$$
이를 (2)의 검정력 근사식에 대입하면
$$
\gamma_n(\theta_{1n})
\simeq
1-\Phi\left(
-\sqrt{n}(\theta_{1n}-\theta_0)\frac{\dot\mu(\theta_0)}{\sigma(\theta_0)}
+z_\alpha
\right).
$$

**4) 목표 검정력 $\gamma$를 위한 표본크기 근사**  
$$
\gamma_n(\theta_{1n})\simeq \gamma
\iff
1-\Phi(A)\simeq \gamma
\iff
\Phi(A)\simeq 1-\gamma
\iff
A\simeq z_{1-\gamma},
$$
여기서
$$
A=
-\sqrt{n}(\theta_{1n}-\theta_0)\frac{\dot\mu(\theta_0)}{\sigma(\theta_0)}
+z_\alpha.
$$
따라서
$$
-\sqrt{n}(\theta_{1n}-\theta_0)\frac{\dot\mu(\theta_0)}{\sigma(\theta_0)}+z_\alpha
\simeq z_{1-\gamma}
$$
이고, 이를 $n$에 대해 풀면
$$
\sqrt{n}\,(\theta_{1n}-\theta_0)\frac{\dot\mu(\theta_0)}{\sigma(\theta_0)}
\simeq z_\alpha+z_{1-\gamma}.
$$
결국
$$
N(T_n;\gamma,\theta_{1n})
\simeq
\left(\frac{\dot\mu(\theta_0)}{\sigma(\theta_0)}\right)^{-2}
\left(\frac{z_\alpha+z_{1-\gamma}}{\theta_{1n}-\theta_0}\right)^2.
$$

### 점근상대효율성 (asymptotic relative efficiency, ARE)
실수 모수 $\theta$에 관한 가설 $H_0(\theta_0):\theta=\theta_0 \quad\text{vs}\quad H_1:\theta>\theta_0$을 유의수준 $\alpha$에서 검정할 때, 크기 $n$인 랜덤표본에 기초한 검정통계량 $T_{in}$ ($i=1,2$)들이 정리 9.3.1의 조건을 만족한다고 하자. 대립가설 $\theta_{1n} \simeq \theta_0 + K/\sqrt{n}$에서 검정력이 $\gamma$가 되기 위한 표본크기를 $N(T_{in};\gamma,\theta_{1n})$라 할 때, 이들 표본크기의 역수의 극한값
$$
\lim_{n\to\infty} \frac{N^{-1}(T_{1n};\gamma,\theta_{1n})}{N^{-1}(T_{2n};\gamma,\theta_{1n})}
$$
을 $T_{1n}$에 의한 검정 방법 1의 $T_{2n}$에 의한 검정 방법 2에 대한 **점근상대효율성**(asymptotic relative efficiency, ARE)이라 한다.  
기호로는 $\mathrm{ARE}(T_{1n}, T_{2n})$로 나타내며, 정리 9.3.1로부터
$$
\mathrm{ARE}(T_{1n}, T_{2n}) = \frac{(\dot{\mu}_1(\theta_0)/\sigma_1(\theta_0))^2}{(\dot{\mu}_2(\theta_0)/\sigma_2(\theta_0))^2}
$$
> **관례적 약기(abuse of notation)**  
> 여기서 $\mathrm{ARE}(T_{1n},T_{2n})$의 $T_{in}$은 “통계량”을 뜻하는 기호이지만, 실제로는 각 $n$에 대해 $T_{in}$으로 **정의되는 검정 절차(임계값 선택까지 포함한 크기 $\alpha$의 검정) 전체의 열** $\{\phi_{i,n}\}_{n\ge1}$을 대표해서 적는 관례적 표기이다.  
> 즉, 엄밀히는
> $$
> \mathrm{ARE}\big(\{\phi_{1,n}\},\{\phi_{2,n}\}\big)
> $$
> 처럼 “검정들의 열”에 대한 점근 비교이지만, 독자가 “ARE는 점근 개념이며 $n$에 따른 절차의 열을 비교한다”는 전제를 안다고 보고 교재에서는 중괄호(또는 $\{\cdot\}$ 표기)를 생략해 $\mathrm{ARE}(T_{1n},T_{2n})$로 쓴다.

#### 예 9.3.3 부호검정의 $t$-검정에 대한 점근상대효율성
모집단 분포가 연속형이고 $\mu$에 대해 대칭이며 확률밀도함수가
$$
\frac{1}{\sigma} f\left(\frac{x-\mu}{\sigma}\right),\quad -\infty<\mu<+\infty,\ \sigma>0
$$
인 경우, 가설 $H_0(\mu_0):\mu=\mu_0 \quad\text{vs}\quad H_1:\mu>\mu_0$ 을 유의수준 $\alpha\ (0<\alpha<1)$에서 검정할 때, 통계량
$$
S_n = \sum_{i=1}^n I(X_i > \mu_0)
$$
을 이용하는 부호검정을 예 9.3.1과 같이 적용할 수 있다. 이때 $\mu_{1n} \simeq \mu_0 + K/\sqrt{n}\ (K>0)$에서 부호검정의 검정력이 $\gamma$가 되기 위한 표본크기는
$$
N(S_n;\gamma,\mu_{1n}) \simeq \left( \frac{2f(0)}{\sigma} \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\mu_{1n} - \mu_0} \right)^2
$$

한편 모집단분포가 정규분포면 $\sqrt n(\bar X -\mu_0)/S \geq t_\alpha(n-1)$로 주어지는 t검정을 사용할 것이다. 이런 t검정의 통계량 $T_n = \sqrt n(\bar X -\mu_0)/S$에 대한 점근정규성 성립은 아래 증명과정을 따른다. (좀 김)
> **로컬 대립가설 $\mu_{1n}\simeq \mu_0+K/\sqrt{n}$에서 $t$-검정 통계량의 점근정규성 유도**  
> $t$-검정 통계량을
> $$
> T_n=\sqrt{n}\frac{\bar X-\mu_0}{S}
> $$
> 로 두고, 로컬 대립가설을 $\mu=\mu_{1n}=\mu_0+K/\sqrt{n}$로 둔다.
> 
> **1) 표본평균의 CLT (로컬 대립가설 하에서도 동일)**  
> $$
> \sqrt{n}\frac{\bar X-\mu_{1n}}{\tau}\ \xrightarrow{d}\ N(0,1).
> $$
> (이유: $X_i-\mu_{1n}$은 평균 $0$, 분산 $\tau^2$이며 $n$에 따라 분포가 변하지 않으므로 CLT 적용)
> 
> **2) 표본표준편차의 일치성**  
> $$
> S \xrightarrow{p} \tau
> \quad\Rightarrow\quad
> \frac{\tau}{S}\xrightarrow{p}1,\qquad \frac{1}{S}-\frac{1}{\tau}\xrightarrow{p}0.
> $$
> (이유: $S^2$는 $\mathrm{Var}(X_1)=\tau^2$의 일치추정량)
> 
> **3) 목표 식의 변형**  
> $$
> A_n:=\sqrt{n}\left(\frac{\bar X-\mu_0}{S}-\frac{\mu_{1n}-\mu_0}{\tau}\right)
> $$
> 라 하자. $\bar X-\mu_0=(\bar X-\mu_{1n})+(\mu_{1n}-\mu_0)$이므로
> $$
> \frac{\bar X-\mu_0}{S}-\frac{\mu_{1n}-\mu_0}{\tau}
> =
> \frac{\bar X-\mu_{1n}}{S}
> +(\mu_{1n}-\mu_0)\left(\frac{1}{S}-\frac{1}{\tau}\right) \\
> \therefore A_n = \underbrace{\sqrt{n}\frac{\bar X-\mu_{1n}}{S}}_{(I)}
> +
> \underbrace{\sqrt{n}(\mu_{1n}-\mu_0)\left(\frac{1}{S}-\frac{1}{\tau}\right)}_{(II)}.
> $$
> 
> **4) 각 항의 극한분포/확률수렴**  
> **(I)항:** 곱의 형태로 쓴다.
> $$
> \sqrt{n}\frac{\bar X-\mu_{1n}}{S}
> =
> \left(\sqrt{n}\frac{\bar X-\mu_{1n}}{\tau}\right)\left(\frac{\tau}{S}\right).
> $$
> 여기서
> $$
> \sqrt{n}\frac{\bar X-\mu_{1n}}{\tau}\xrightarrow{d}N(0,1),\qquad \frac{\tau}{S}\xrightarrow{p}1.
> $$
> 슬럿츠키 정리에 의해
> $$
> (I)\ \xrightarrow{d}\ N(0,1).
> $$
> 
> **(II)항:** 로컬 대립가설에서
> $$
> \sqrt{n}(\mu_{1n}-\mu_0)\to K \quad(\text{상수}),
> $$
> 또한 $\left(\frac{1}{S}-\frac{1}{\tau}\right)\xrightarrow{p}0$이므로
> $$
> (II)\ \xrightarrow{p}\ 0.
> $$
> 
> **5) 결론 (슬럿츠키 정리)**  
> $$
> A_n=(I)+(II)\ \xrightarrow{d}\ N(0,1).
> $$
> 즉,
> $$
> \boxed{
> \sqrt{n}\left( \frac{\bar{X} - \mu_0}{S} - \frac{\mu_{1n} - \mu_0}{\sqrt{\mathrm{Var}(X_1)}} \right) \xrightarrow{d} N(0,1)
> }
> $$
> 이며,
> $$
> \boxed{
> \sqrt{\mathrm{Var}(X_1)}=\sigma\left(\int_{-\infty}^{\infty} z^2 f(z)\,dz\right)^{1/2}.
> }
> $$
> 
> (참고) $T_n$ 자체로 쓰면 $\sqrt{n}(\mu_{1n}-\mu_0)\to K$이므로
> $$
> T_n
> =
> \sqrt{n}\frac{\bar X-\mu_{1n}}{S}
> +\sqrt{n}\frac{\mu_{1n}-\mu_0}{S}
> \Rightarrow N\!\left(\frac{K}{\tau},\,1\right)
> $$
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
모집단 분포가 연속형이고 확률밀도함수가 $f(x-\theta),\quad -\infty<\theta<+\infty$의 꼴로서 $\theta$에 대해 대칭($f(-x)=f(x)$)인 경우, 랜덤표본 $X_1,\dots,X_n$을 이용하여 $H_0(\theta_0):\theta=\theta_0 \quad\text{vs}\quad H_1:\theta>\theta_0$을 검정한다. 이때 $|X_1-\theta_0|,\dots,|X_n-\theta_0|$을 크기순으로 나열할 때 $|X_i-\theta_0|$의 순위를 $R(|X_i-\theta_0|)$라 하며,
$$
R(|X_i-\theta_0|) = 1 + \sum_{j=1}^n I(|X_j-\theta_0| < |X_i-\theta_0|)
$$
이다. 다음 통계량을 **부호순위(signed rank) 검정통계량**이라 한다.
$$
W_n = \sum_{i=1}^n \operatorname{sgn}(X_i-\theta_0) R(|X_i-\theta_0|)
$$
여기서 $\operatorname{sgn}(x)$는 $x$의 부호 함수이다.

#### 예: 부호순위(signed rank) 통계량 계산 예시
$$
X=(2.1,\ -0.4,\ 1.3,\ -2.0,\ 0.7)
$$
| $i$ | $X_i$ | $\operatorname{sgn}(X_i)$ | $\lvert X_i\rvert$ | 순위 $R(\lvert X_i\rvert)$ | $\operatorname{sgn}(X_i)\,R(\lvert X_i\rvert)$ |
|---:|---:|:---:|---:|---:|---:|
| 1 | 2.1  | $+1$ | 2.1 | 5 | 5  |
| 2 | -0.4 | $-1$ | 0.4 | 1 | -1 |
| 3 | 1.3  | $+1$ | 1.3 | 4 | 4  |
| 4 | -2.0 | $-1$ | 2.0 | 3 | -3 |
| 5 | 0.7  | $+1$ | 0.7 | 2 | 2  |  

$$
W_5=5+(-1)+4+(-3)+2=7.
$$

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
연속형 분포이므로 $|X_i-\theta_0|$에 동점이 생길 확률은 $0$이라 가정해도 무방하다.

**(a)** $|X_i-\theta_0|$를 오름차순으로 정렬했을 때 $j$번째로 작은 값을 갖는 표본의 인덱스를 $i(j)$라 두면 (즉, $R(|X_{i(j)}-\theta_0|)=j$),
$$
W_n=\sum_{j=1}^n \operatorname{sgn}(X_{i(j)}-\theta_0)\, j.
$$
귀무가설 $H_0(\theta_0)$ 하에서 $X_i-\theta_0$의 분포는 $0$을 중심으로 대칭이므로
$$
P(\operatorname{sgn}(X_i-\theta_0)=+1)=P(\operatorname{sgn}(X_i-\theta_0)=-1)=\tfrac12.
$$
연속형이고 $\theta_0$에 대해 대칭인 분포에서 $Y:=X_1-\theta_0$라 두면 $Y$의 밀도는 $f_Y(y)=f_Y(-y)$이고 $P(Y=0)=0$이다. 임의의 $x\ge 0$에 대해  
$$
P_{\theta_0}\big(|X_1-\theta_0|\le x,\ \operatorname{sgn}(X_1-\theta_0)=+1\big)
= P(0<Y\le x)
= \int_0^x f_Y(y)\,dy. \\
P_{\theta_0}(|X_1-\theta_0|\le x)=P(|Y|\le x)=\int_{-x}^x f_Y(y)\,dy
=2\int_0^x f_Y(y)\,dy,
$$
그리고 대칭성으로  
$$
P_{\theta_0}\big(\operatorname{sgn}(X_1-\theta_0)=+1\big)=P(Y>0)=\tfrac12. \\
\therefore P_{\theta_0}\big(|X_1-\theta_0|\le x,\ \operatorname{sgn}(X_1-\theta_0)=+1\big)
=\frac12\,P_{\theta_0}(|X_1-\theta_0|\le x) \\
= P_{\theta_0}(|X_1-\theta_0|\le x)\,P_{\theta_0}(\operatorname{sgn}(X_1-\theta_0)=+1),
$$
즉 $|X_1-\theta_0|$와 $\operatorname{sgn}(X_1-\theta_0)$는 독립이다. ($-1$에 대해서도 동일하게 성립)

또한 $X_1,\dots,X_n$이 iid이므로 $Y_i:=X_i-\theta_0$들도 서로 독립이고, $S_i:=\operatorname{sgn}(Y_i)$는 $Y_i$의 함수이므로 $S_1,\dots,S_n$도 서로 독립이다 (정리 2.4.2). 더 나아가 앞에서 보인 것처럼 각 $i$에 대해 $|Y_i|$와 $S_i$는 독립이므로, 정렬 인덱스 $i(j)$ (즉 $|Y_{i(j)}|$가 $j$번째로 작은 표본) 는 오직 $\{|Y_i|\}$에 의해서만 결정되고 $\{S_i\}$와 독립이다. 따라서 정렬 후 부호열
$$
S(j):=\operatorname{sgn}(Y_{i(j)})\quad (j=1,\dots,n)
$$
은 $\{S_i\}$의 임의 재배열과 동일한 분포를 가지며, 서로 iid이고 $P(S(j)=\pm1)=1/2$를 만족한다.

정렬은 오직 $|X_i-\theta_0|$들에 의해 결정되므로, 정렬된 순서에 붙는 부호열
$$
S(j):=\operatorname{sgn}(X_{i(j)}-\theta_0)\quad (j=1,\dots,n)
$$
은 서로 iid이고 $P(S(j)=\pm1)=1/2$를 만족한다. 따라서
$$
W_n \ \overset{d}{\equiv}\ \sum_{j=1}^n j\,S(j)
$$
가 성립한다.

**(b)** (a)의 표현으로부터
$$
E_{\theta_0}(W_n)=\sum_{j=1}^n j\,E(S(j))=0,
\qquad
\mathrm{Var}_{\theta_0}(W_n)=\sum_{j=1}^n j^2\,\mathrm{Var}(S(j))=\sum_{j=1}^n j^2
=\frac{n(n+1)(2n+1)}{6}.
$$
또한 $jS(j)$들은 독립이고
$$
\frac{\max_{1\le j\le n} j^2}{\sum_{j=1}^n j^2}=\frac{n^2}{n(n+1)(2n+1)/6}\to 0
$$
이므로 (예: Lindeberg–Feller 또는 Lyapunov CLT) 중심극한정리를 적용할 수 있다. 따라서
$$
\frac{W_n-E_{\theta_0}(W_n)}{\sqrt{\mathrm{Var}_{\theta_0}(W_n)}}\xrightarrow{d}N(0,1)
$$
이 성립한다.

TODO: 교재의 증명방법은 조금 다른데, 한번 읽어보기

### 정리 9.3.3 부호순위 검정통계량의 표현
$R(|X_i-\theta_0|)$를 $|X_1-\theta_0|,\dots,|X_n-\theta_0|$의 순위라 하고,  
$\operatorname{sgn}(x)$를 부호 함수라 하고,  
부호순위 검정통계량을 아래로 정의하자.
$$
W_n = \sum_{i=1}^n \operatorname{sgn}(X_i-\theta_0) R(|X_i-\theta_0|),\qquad
W_n^+ = \sum_{i=1}^n \mathbf{1}(X_i-\theta_0 > 0) R(|X_i-\theta_0|)
$$
그러면 아래 두 가지가 성립한다.  
(a) $W_n$과 $W_n^+$의 관계
    $$
    W_n = 2W_n^+ - \frac{n(n+1)}{2}
    $$

(b) 귀무가설 $H_0(\theta_0):\theta=\theta_0$ 하에서의 분포 및 정규근사
    $$
    W_n^+ \overset{d}{\equiv} \sum_{j=1}^n j B_j,\qquad B_j \overset{iid}{\sim} \mathrm{Bernoulli}(1/2)
    $$
    $$
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
=
2\mathbf{1}(X_i-\theta_0>0)-1 \\
\begin{aligned}
\therefore W_n
&=\sum_{i=1}^n\big(2\mathbf{1}(X_i-\theta_0>0)-1\big)\,R(|X_i-\theta_0|)\\
&=2W_n^+-\sum_{i=1}^n R(|X_i-\theta_0|).
\end{aligned}
$$
순위의 합은 항상 $1+2+\cdots+n=n(n+1)/2$ 이므로
$$
W_n=2W_n^+-\frac{n(n+1)}{2}.
$$

**(b)** 
$$
W_n \ \overset{d}{\equiv}\ \sum_{j=1}^n j\,S(j),\qquad S(j)\ \text{iid},\ P(S(j)=\pm1)=\tfrac12 \\
W_n^+=\frac{W_n+\frac{n(n+1)}{2}}{2}
\ \overset{d}{\equiv}\
\sum_{j=1}^n j\,\frac{S(j)+1}{2}.
$$
여기서
$$
B_j:=\frac{S(j)+1}{2}\in\{0,1\},\qquad
P(B_j=1)=P(S(j)=1)=\tfrac12
$$
이므로 $B_j \overset{iid}{\sim}\mathrm{Bernoulli}(1/2)$. 따라서
$$
W_n^+ \ \overset{d}{\equiv}\ \sum_{j=1}^n j\,B_j.
$$

또한 $E(B_j)=\tfrac12,\ \mathrm{Var}(B_j)=\tfrac14$ 이고 서로 독립이므로
$$
E_{\theta_0}(W_n^+)=\sum_{j=1}^n j\,E(B_j)=\frac12\sum_{j=1}^n j=\frac{n(n+1)}{4}, \\
\mathrm{Var}_{\theta_0}(W_n^+)=\sum_{j=1}^n j^2\,\mathrm{Var}(B_j)
=\frac14\sum_{j=1}^n j^2
=\frac{n(n+1)(2n+1)}{24}.
$$
가중합 $\sum_{j=1}^n j(B_j-\tfrac12)$에 중심극한정리를 적용하면
$$
\frac{W_n^+ - n(n+1)/4}{\sqrt{n(n+1)(2n+1)/24}}
\ \overset{d}{\to}\ N(0,1)\quad(n\to\infty)
$$

### 정리 9.3.4 한쪽 가설에 대한 부호순위 검정
정리9.3.2로부터 귀무가설 $H_0(\theta_0): \theta = \theta_0$하에서 부호순위 검정통계량 $W_n$의 분포는 모집단분포의 확률밀도함수 형태와 관계없다는 것을 알 수 있고, $W_n$의 큰 값은 대립가설에 대한 증거라 할 수 있다. 따라서  
- 모형(위치모수, 대칭): 모집단 밀도 $f(x-\theta)$, $-\infty<\theta<\infty$, $f(-x)=f(x)$
- 가설은 $H_0:\theta\le \theta_0\quad \text{vs}\quad H_1:\theta>\theta_0$
- 유의수준 $\alpha$의 부호순위 검정(임계값 $c$, 랜덤화 $\gamma\in[0,1]$)
    $$
    \phi_{SR}(X_1,\dots,X_n)=
    \begin{cases}
        1, & W_n \ge c+1 \\
        \gamma, & W_n = c \\
        0, & W_n \le c-1
    \end{cases}
    \qquad E_{\theta_0}[\phi_{SR}(X)] = \alpha
    $$
    또는 $W_n^+$로 동치 표현 (정리 9.3.3):
    $$
    \phi_{SR}(X_1,\dots,X_n)=
    \begin{cases}
        1, & W_n^+ \ge c^+ + 1 \\
        \gamma, & W_n^+ = c^+ \\
        0, & W_n^+ \le c^+ - 1
    \end{cases}
    \qquad E_{\theta_0}[\phi_{SR}(X)] = \alpha
    $$
>    - (참고: 정리 9.3.2, 정리 9.3.3으로부터 정규근사에 의한 큰 표본 기각역)
>        $$
>        \frac{W_n}{\sqrt{n(n+1)(2n+1)/6}} \ge z_\alpha
>        \quad\text{또는}\quad
>        \frac{W_n^+ - n(n+1)/4}{\sqrt{n(n+1)(2n+1)/24}} \ge z_\alpha
>        $$
- 위 상황에서, 아래 성질이 성립한다.
    - (U-통계량 표현 / 단조성) 연속형 분포에서
        $$
        W_n^+ \;=\; \sum_{1\le i\le j\le n}\mathbf{1}\!\left(X_i+X_j>2\theta_0\right)
        $$
        이므로 $W_n^+$ (따라서 $W_n$도)는 각 성분 $X_k$에 대한 **증가함수**이고, 기각역 $\{W_n^+\ge c^+\}$ (또는 $\{W_n\ge c\}$)는 상향집합(upward closed set)이다.
    - 검정력 함수 $\gamma_{\phi_{SR}}(\theta) = E_\theta[\phi_{SR}(X)]$는 $\theta$의 증가함수
    - 따라서
        $$
        \max_{\theta\le \theta_0} E_\theta[\phi_{SR}(X)] = E_{\theta_0}[\phi_{SR}(X)] = \alpha
        $$

#### 증명
**(a)** 연속형 분포를 가정하면 $|Y_i|$에 동점이 생길 확률은 $0$이므로(거의 surely) 순위가 잘 정의된다.  
$Y_i:=X_i-\theta_0$라 두고, $|Y_i|$의 오름차순 정렬 인덱스를 $i(1),\dots,i(n)$이라 하며
$$
Y_{(k)}:=Y_{i(k)}\quad(k=1,\dots,n)
$$
로 쓰면 $R(|Y_{(k)}|)=k$이다. 이제 $W_n^+$를 다음과 같이 변형한다:
$$
\begin{aligned}
W_n^+
&=\sum_{i=1}^n \mathbf{1}(Y_i>0)\,R(|Y_i|)\\
&=\sum_{k=1}^n \mathbf{1}(Y_{(k)}>0)\Big(1+\sum_{\ell<k}1\Big)\\
&=\sum_{k=1}^n \mathbf{1}(Y_{(k)}>0)
\;+\;
\sum_{k=1}^n\sum_{\ell<k}\mathbf{1}(Y_{(k)}>0)\\
&=\sum_{k=1}^n \mathbf{1}(Y_{(k)}>0)
\;+\;
\sum_{k=1}^n\sum_{\ell<k}\mathbf{1}\!\big(Y_{(k)}>0,\ |Y_{(\ell)}|<|Y_{(k)}|\big)
&&\text{(정렬로 }\ell<k\Rightarrow |Y_{(\ell)}|<|Y_{(k)}|\text{)}\\
&=\sum_{k=1}^n \mathbf{1}(Y_{(k)}>0)
\;+\;
\sum_{k=1}^n\sum_{\ell<k}\mathbf{1}\!\big(Y_{(k)}>0,\ -Y_{(k)}<Y_{(\ell)}<Y_{(k)}\big)
&&\text{( }Y_{(k)}>0\Rightarrow |Y_{(\ell)}|<Y_{(k)}\iff -Y_{(k)}<Y_{(\ell)}<Y_{(k)}\text{)}\\
&=\sum_{k=1}^n \mathbf{1}\!\big(Y_{(k)}+Y_{(k)}>0\big)
\;+\;
\sum_{k=1}^n\sum_{\ell<k}\mathbf{1}\!\big(Y_{(k)}+Y_{(\ell)}>0\big)
&&\text{(앞 항은 }Y_{(k)}>0\iff 2Y_{(k)}>0;\\
&&&\quad \ell<k\text{이면 }|Y_{(\ell)}|<Y_{(k)}\Rightarrow Y_{(\ell)}<Y_{(k)}\text{라 상한은 자동)}\\
&=\sum_{1\le \ell\le k\le n}\mathbf{1}\!\big(Y_{(k)}+Y_{(\ell)}>0\big)
&&\text{(대각 }\ell=k\text{ 항과 }\ell<k\text{ 항을 합쳐 하나의 이중합으로 결합)}\\
&=\sum_{1\le i\le j\le n}\mathbf{1}(Y_i+Y_j>0)
&&\text{( }(Y_{(1)},\dots,Y_{(n)})\text{는 }(Y_1,\dots,Y_n)\text{의 재배열이므로 지표합은 불변)}.
\end{aligned} \\
\therefore \boxed{
W_n^+ \;=\; \sum_{1\le i\le j\le n}\mathbf{1}(Y_i+Y_j>0)
\;=\;
\sum_{1\le i\le j\le n}\mathbf{1}(X_i+X_j>2\theta_0)
}.
$$

**(b)** 부호순위 검정의 정의상 $\phi_{SR}(X)$는 $W_n^+$에 대한 (비감소) 함수이다. 그리고 (a)에서
$$
W_n^+=\sum_{1\le i\le j\le n}\mathbf{1}(X_i+X_j>2\theta_0)
$$
로 나타났으므로, 각 항 $\mathbf{1}(X_i+X_j>2\theta_0)$는 $(X_1,\dots,X_n)$의 각 성분에 대한 증가함수이고, 합 $W_n^+$ 역시 각 성분에 대한 증가함수이다. 따라서 $\phi_{SR}(X)$도 각 성분에 대한 증가함수이다.

이제 위치모수 모형에서 $X_i=\theta+Z_i$ ($Z_i$ iid, 밀도 $f$)로 둘 수 있다. $\theta'<\theta''$에 대해 같은 $Z=(Z_1,\dots,Z_n)$로 결합(coupling)하면
$$
X(\theta'')=\theta''+Z \ \ge\ \theta'+Z=X(\theta') \quad(\text{성분별}),
$$
그리고 $\phi_{SR}$가 증가함수이므로
$$
\phi_{SR}(X(\theta''))\ge \phi_{SR}(X(\theta')).
$$
양변에 기댓값을 취하면
$$
E_{\theta''}[\phi_{SR}(X)]\ge E_{\theta'}[\phi_{SR}(X)],
$$
즉 검정력 함수 $\gamma_{\phi_{SR}}(\theta)=E_\theta[\phi_{SR}(X)]$는 $\theta$의 증가함수이다. 그러므로
$$
\max_{\theta\le \theta_0}E_\theta[\phi_{SR}(X)]=E_{\theta_0}[\phi_{SR}(X)]=\alpha
$$
가 된다(임계값 $c,\gamma$를 $E_{\theta_0}[\phi_{SR}(X)]=\alpha$가 되도록 잡았으므로). 따라서 $\phi_{SR}$는 유의수준 $\alpha$의 검정이다.

### 정리 9.3.5 부호순위 검정의 점근정규성
정리9.3.4에서 알 수 있듯이, 연속형 대칭인 분포의 중앙값에 대한 한쪽가설이나 양쪽가설의 검정에 부호순위 검정을 사용할 수 있고, 이는 비모수적 검정이다. 이런 부호순위 검정의 효율성을 알아보려면 아래 정리와 같은 점근 정규성이 필요하다. 이 정리의 증명은 이 책의 수준을 넘으므로 생략한다.
- (a) 점근정규성
    $$
    \frac{W_n^+ - E_\theta(W_n^+)}{\sqrt{\mathrm{Var}_\theta(W_n^+)}} \overset{d}{\to} N(0,1)
    $$
- (b) 평균/분산 근사
    $$
    \mu(\theta) = \frac{1}{2} P_\theta(X_1 + X_2 > 2\theta_0),\qquad
    \sigma^2(\theta) = \mathrm{Cov}_\theta\left( \mathbf{1}(X_1 + X_2 > 2\theta_0),\ \mathbf{1}(X_1 + X_3 > 2\theta_0) \right) \\
    \Rightarrow E_\theta(W_n^+) \simeq n^2 \mu(\theta),\qquad \mathrm{Var}_\theta(W_n^+) \simeq n^3 \sigma^2(\theta)
    $$
    따라서 $W_n^+/n^2$에 대해 앞의 점근적 검정력/표본크기 근사 공식을 적용할 수 있으므로 부호순위검정의 검정력에 대한 근사식을 아래 정리와 같이 구할 수 있다.

### 정리 9.3.6 부호순위 검정의 검정력 근사와 표본크기
- 정리 9.3.4의 가정(대칭 위치모수 모형) 하에서, 대립가설하의 모수
    $$
    \theta_{1n} \simeq \theta_0 + \frac{K}{\sqrt{n}}\quad(K>0)
    $$
    이 귀무가설에 근접할 때, 

- (a) 검정력이 근사한다
    $$
    \gamma_{\phi_{SR}}(\theta_{1n}) \simeq 1 - \Phi\left( -\sqrt{n} (\theta_{1n} - \theta_0) \frac{\dot\mu(\theta_0)}{\sigma(\theta_0)} + z_\alpha \right)
    $$
    여기서
    $$
    \dot\mu(\theta_0) = \int_{-\infty}^{\infty} f^2(x)\,dx,\qquad \sigma^2(\theta_0) = \frac{1}{12}
    $$

- (b) 목표 검정력 $\gamma$를 얻기 위한 표본크기 근사
    $N(W_n^+;\gamma,\theta_{1n})$를 “검정력 $\gamma$”에 필요한 표본크기라 하면
    $$
    N(W_n^+;\gamma,\theta_{1n}) \simeq \left( \sqrt{12} \int_{-\infty}^{\infty} f^2(x)\,dx \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\theta_{1n} - \theta_0} \right)^2
    $$

#### 증명
정리 9.3.5로부터
$$
\frac{W_n^+-E_\theta(W_n^+)}{\sqrt{\mathrm{Var}_\theta(W_n^+)}}\ \overset{d}{\to}\ N(0,1),
\qquad
E_\theta(W_n^+)\simeq n^2\mu(\theta),\quad \mathrm{Var}_\theta(W_n^+)\simeq n^3\sigma^2(\theta)
$$
이므로
$$
T_n:=\frac{W_n^+}{n^2}
$$
에 대해
$$
E_\theta(T_n)\simeq \mu(\theta),\qquad \mathrm{Var}_\theta(T_n)\simeq \frac{\sigma^2(\theta)}{n},
$$
따라서
$$
\sqrt{n}\,\frac{T_n-\mu(\theta)}{\sigma(\theta)}
=\frac{W_n^+-n^2\mu(\theta)}{n^{3/2}\sigma(\theta)}
\ \overset{d}{\to}\ N(0,1)
$$
가 성립한다. 즉 $T_n$은 정리 9.3.1의 점근정규성 가정을 만족한다(여기서 정리 9.3.1의 $\mu(\theta),\sigma(\theta)$에 각각 위 $\mu(\theta),\sigma(\theta)$를 대응시킨다).

이제 대칭 위치모수 모형에서 $X_i=\theta+Z_i$ ($Z_i$ iid, 밀도 $f$, $f(z)=f(-z)$)라 두면
$$
\mu(\theta)=\frac12P_\theta(X_1+X_2>2\theta_0)
=\frac12P(Z_1+Z_2>2(\theta_0-\theta)).
$$
$Y:=Z_1+Z_2$의 밀도를 $g$라 하면(컨볼루션)
$$
g(t)=(f*f)(t)=\int_{-\infty}^{\infty} f(u)f(t-u)\,du.
$$
또한 $a(\theta):=2(\theta_0-\theta)$라 두면
$$
\mu(\theta)=\frac12P(Y>a(\theta))=\frac12\{1-G(a(\theta))\},
$$
이므로(연쇄법칙)
$$
\dot\mu(\theta)=\frac12\cdot\{-g(a(\theta))\}\cdot a'(\theta)
=\frac12\cdot(-g(a(\theta)))\cdot(-2)=g(a(\theta)).
$$
따라서
$$
\dot\mu(\theta_0)=g(0)=(f*f)(0)=\int_{-\infty}^{\infty} f(u)f(-u)\,du=\int_{-\infty}^{\infty} f^2(u)\,du.
$$

한편 귀무가설($\theta=\theta_0$) 하에서 정리 9.3.3의 표현을 쓰면
$$
W_n^+ \overset{d}{\equiv}\sum_{j=1}^n jB_j,\qquad B_j\overset{iid}{\sim}\mathrm{Bernoulli}(1/2),
$$
따라서
$$
\mathrm{Var}_{\theta_0}(W_n^+)=\sum_{j=1}^n j^2\mathrm{Var}(B_j)=\frac14\sum_{j=1}^n j^2
=\frac{n(n+1)(2n+1)}{24}\sim \frac{n^3}{12}.
$$
정리 9.3.5의 근사 $\mathrm{Var}_\theta(W_n^+)\simeq n^3\sigma^2(\theta)$와 비교하면
$$
\sigma^2(\theta_0)=\frac{1}{12}.
$$

이제 정리 9.3.1을 $T_n=W_n^+/n^2$에 적용하면, 로컬 대립가설 $\theta_{1n}\simeq \theta_0+K/\sqrt{n}$에서

(a) (검정력 근사)
$$
\gamma_{\phi_{SR}}(\theta_{1n})
\simeq
1-\Phi\!\left(
-\sqrt{n}(\theta_{1n}-\theta_0)\frac{\dot\mu(\theta_0)}{\sigma(\theta_0)}+z_\alpha
\right)
$$
이고, 위에서 구한
$$
\dot\mu(\theta_0)=\int f^2(x)\,dx,\qquad \sigma^2(\theta_0)=\frac{1}{12}
$$
를 대입하면 정리의 (a)가 된다.

(b) (표본크기 근사)
정리 9.3.1의 표본크기 근사식을 그대로 적용하여
$$
N(W_n^+;\gamma,\theta_{1n})
\simeq
\left(\frac{\dot\mu(\theta_0)}{\sigma(\theta_0)}\right)^{-2}
\left(\frac{z_\alpha+z_{1-\gamma}}{\theta_{1n}-\theta_0}\right)^2
=
\left(\sqrt{12}\int f^2(x)\,dx\right)^{-2}
\left(\frac{z_\alpha+z_{1-\gamma}}{\theta_{1n}-\theta_0}\right)^2
$$
를 얻는다.

#### 예 9.3.5 부호순위 검정의 $t$-검정에 대한 점근상대효율성(ARE)
- 대칭 위치-척도 모형:
    $$
    \frac{1}{\sigma} f\left(\frac{x-\mu}{\sigma}\right),\quad -\infty<\mu<\infty,\ \sigma>0
    $$
    가설: $H_0(\mu_0):\mu=\mu_0\quad \text{vs}\quad H_1:\mu>\mu_0$
- 통계량 $W_n := \sum_{i=1}^n \operatorname{sgn}(X_i-\theta_0)\,R(|X_i-\theta_0|)$
- 부호순위 검정의 표본크기 근사(로컬 대립 $\mu_{1n} \simeq \mu_0 + K/\sqrt{n}$) (정리9.3.6)
    $$
    N(W_n;\gamma,\mu_{1n}) \simeq \left( \sqrt{12} \frac{\int f^2(z)\,dz}{\sigma} \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\mu_{1n} - \mu_0} \right)^2
    $$
- 정규모형에서의 $t$-검정 표본크기 근사 (정리9.3.3)
    $$
    N(T_n;\gamma,\mu_{1n}) \simeq \left( \frac{1}{\sqrt{\mathrm{Var}(X_1)}} \right)^{-2} \left( \frac{z_\alpha + z_{1-\gamma}}{\mu_{1n} - \mu_0} \right)^2
    $$
    $$
    \sqrt{\mathrm{Var}(X_1)} = \sigma \left( \int z^2 f(z)\,dz \right)^{1/2}
    $$

- 따라서 점근상대효율성(ARE)
    $$
    \mathrm{ARE}(W_n, T_n) = 12 \left( \int f^2(z)\,dz \right)^2 \left( \int z^2 f(z)\,dz \right)
    $$

- 아래 표와 부호검정에 대한 표 9.3.1을 비교해보면 부호순위 검정의 상대효율성이 부호검정에 비해 매우 높은 것을 알 수 있다. 즉 부호순위 검정은 범용성과 더불어 효율성도 비교적 높아서 매우 유용한 검정 방법이다.
- **표 9.3.2: $\mathrm{ARE}(W_n, T_n)$**
    - $N(\mu,\sigma^2)$: $3/\pi \approx 0.954$
    - $L(\mu,\sigma)$: $\pi^2/9 \approx 1.096$
    - $DE(\mu,\sigma)$: $1.5$




FIXME:
7장에서는 "통계적 가설검정은 모집단에 대한 확률모형을 설정하고, 관측된 자료가 해당모형와 얼마나 부합하는지 평가하여 귀무가설을 가각할지 채택할지 결정하는 절차로, 이 과정에서 기각역과 채택역은 서로 여집합 관계로, 기각역은 유의수준(제1종 오류 확률) 제약을 만족해야 하고, 그 범위 내에서 대립가설에 대한 검정력이 최대가 되도록 선택된다." 이 과정으로 검정을 수행했는데, 
이번 9장에서 세가지 방법들 중 무엇에 해당하나?