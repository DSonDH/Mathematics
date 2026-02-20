# 제7장 검정 *(Hypothesis Testing)*

## 검정의 형식 논리 *(Formal Structure of Hypothesis Tests)*
현상에 대한 확률모형을 설정하고, 관측된 자료가 해당 모형과 얼마나 부합하는지를 판단하는 과정이 통계적 검정이다.
이 과정에서는 먼저 경험적·이론적 지식을 바탕으로 하나의 모형을 가정하고, 실제 관측 결과가 이 모형 하에서 일어나기 어려운지 여부를 평가한다.

만약 설정한 모형 하에서 관측된 결과가 매우 드물게 발생하는 결과라면, 해당 모형을 부정하고 이를 대체할 수 있는 다른 모형을 고려하게 된다.
이와 같이 **모형을 채택할지 또는 기각할지를 판단하는 절차**를 통계적 가설검정(statistical hypothesis testing), 또는 간단히 검정(testing)이라 한다.

요약: 
  - 통계적 가설검정은 모집단에 대한 확률모형을 설정하고, 관측된 자료가 해당 모형과 얼마나 부합하는지 평가하여 귀무가설을 기각할지 채택할지 결정하는 절차. 
  - 이 과정에서 기각역과 채택역은 서로 여집합 관계로, 기각역은 유의수준(제1종 오류 확률) 제약을 만족해야 하고, 그 범위 내에서 대립가설에 대한 검정력이 최대가 되도록 선택된다. 
  - 즉, 기각역은 임의로 정할 수 없고, 오류 가능성을 고려해 통계적으로 최적화되어야 한다.
    - **검정력(power)**: 모수 $\theta$에 대해, 검정력은 $\theta$가 대립가설 영역($\Omega_1$)에 있을 때 귀무가설이 기각될 확률로 정의된다.
        $$
        \text{검정력}(\theta) = P_\theta(\text{귀무가설 기각}) = P_\theta(X \in C),\quad \theta \in \Omega_1
        $$
        여기서 $C$는 기각역이다. 검정력은 대립가설이 참일 때 올바르게 귀무가설을 기각할 확률을 의미한다.
    - 검정 설계에서 제1종 오류 확률은 반드시 $\alpha$ 이하로 고정되며, 이 제약 하에서 제2종 오류 $1-\text{검정력}$을 최소화하는 것이 목표다. 최적 검정에서는 제1종 오류 확률이 정확히 $\alpha$가 되도록 기각역이 선택된다.
- 관측값이 기각역에 들어왔을 때, 허용되는 결론 문장
    - **유의수준 $\alpha$에서 귀무가설을 기각할 충분한 통계적 근거가 있다.**
    - 관측값이 기각역에 들어오지 않았을 때, 허용되는 결론 문장은 다음과 같다.
        - **유의수준 $\alpha$에서 귀무가설을 기각할 충분한 근거가 없다.**
    - 여기에는 "맞다", "틀리다", "확률이 몇 %다"와 같은 표현이 절대 들어가지 않는다.

- 주의
    - 통계적 가설검정은 "가설이 맞을 확률"을 계산하거나 "맞다/틀리다"를 선언하는 절차가 아니다.
    - 유의수준이라는 오류 상한 하에서, 관측된 데이터가 귀무가설 하에서 얼마나 드문지를 평가하여 귀무가설을 기각할 통계적 근거가 있는지 여부만 판단한다.
        - "이 검정은 $\mu=\mu_1$일 때 검정력이 80%다"
        - ❌ 말할 수 없다: "이번 결과에서 검정력이 80%다", "그래서 가설이 맞을 확률이 80%다"

- 참고: **검정력과 p-value의 차이**
    - **p-value**:  
        $P(\text{data 이상으로 극단적} \mid H_0)$  
        → 실제 데이터를 관측한 후, 귀무가설 하에서 이보다 더 극단적인 결과가 나올 확률을 계산  
        → 검정 결과의 "강도"를 나타내며, 작을수록 귀무가설에 불리한 증거
    - **검정력**:  
        $P(\text{기각} \mid \theta \in \Omega_1)$  
        → 검정 설계 단계에서, 대립가설이 참일 때 귀무가설을 기각할 확률  
        → 검정의 "성능"을 의미하며, 값이 클수록 대립가설을 잘 검출
    - **차이점**:  
        - p-value는 관측 후 계산, 검정력은 관측 전(설계 단계)에서 평가  
        - p-value는 귀무가설 하에서의 데이터 해석, 검정력은 대립가설 하에서의 검정 성공 확률  
        - 서로 다른 시간축(사후 vs. 사전)과 해석적 의미를 가진다

### 가설의 설정
검정에서는 두 개의 가설을 설정한다.
* **귀무가설(null hypothesis)**: 반증의 대상이 되는 가설
* **대립가설(alternative hypothesis)**: 귀무가설이 기각될 경우 채택되는 가설

일반적으로 귀무가설은 "차이가 없다", "기존과 같다"와 같이 보수적인 진술로 설정된다.

#### 예 7.1.1
한 제약회사의 기존 진통제는 복용 후 진통 효과가 나타나기까지의 시간이 평균 30분, 표준편차 5분인 정규분포를 따른다고 알려져 있다.
연구진은 새로운 진통제를 개발하였으며, 이 약의 효과가 기존 제품보다 더 빨리 나타난다고 주장한다.

이를 검증하기 위해 회사는 환자 100명을 랜덤하게 추출하여 새로운 진통제를 복용하게 한 뒤, 진통 효과가 나타나는 시간의 표본평균 $\bar X$를 관측한다.

새로운 진통제의 평균 진통 효과 발현 시간을 $\mu$라 하면 가설은 다음과 같이 설정된다.
* (여기서는 새로운 진통제 효과가 나타나는 시간의 분포가 정규분포이고, 평균은 $\mu$, 표준편차는 5분이라 가정하고 계산한 값이다. 표준편차가 미지의 모수인 경우는 7장 2절에 소개될 것이다.)
* 귀무가설:
  $$
  H_0 : \mu = 30
  $$

* 대립가설:
  $$
  H_1 : \mu < 30
  $$

이 예제에서 관심사는 새로운 진통제가 기존 제품보다 **더 빠르다는 증거가 충분한지** 여부이다.

**검정통계량과 기각역**  
표본 크기가 100이고 모분산이 알려져 있으므로, 표본평균은
$$
\bar X \sim N\left(\mu, \frac{5^2}{100}\right)
$$
  - 귀무가설을 설정하면 표본을 선택하고 관찰하여, 그 가설을 기각할 것인가 채택할 것인가를 결정하기 위한 통계량을 구해야 한다
  - 여기서 표본평균과 같이 귀무가설의 기각역을 나타내는 데 사용되는 통계량을 검정통계량(test statistic)이라 한다
  - 검정통계량도 통계량이므로 확률변수이며, 어떤 확률분포를 갖는다.
    - 검정통계량의 분포를 알면, 그 영역을 기각역, 채택역으로 나눠 검정통계량 값이 기각역에 위치하면 귀무가설을 기각하고, 채택역에 속하면 채택한다. 기각역과 채택역을 나누는 경계치를 기각치(critical value)라 한다.

귀무가설 $H_0$ 하에서
$$
Z = \frac{\bar X - 30}{5/\sqrt{100}} \sim N(0,1)
$$
예를 들어 $\bar X \le 29$라는 관측 결과가 나왔다면,
$$
P(\bar X \le 29 \mid H_0)
= P\left( Z \le -2 \right)
= 0.023
$$
이는 귀무가설이 참일 때 약 100번 중 2~3번 정도밖에 발생하지 않는 결과이므로, $H_0$에 대한 강한 반증으로 해석할 수 있다.

**기각역과 채택역**  
* $\bar X \le 29$인 영역을 **기각역(rejection region)** 이라 한다.
* $\bar X > 29$인 영역을 **채택역(acceptance region)** 이라 한다.

    - 검정은 관측된 통계량이 기각역에 포함되는지 여부에 따라 귀무가설을 기각 또는 채택한다.
    - 가설은 모수공간에서, 기각역은 샘플공간에서! 다른것임.

**검정에서의 오류**  
검정 결과는 항상 오류 가능성을 내포한다.

| 실제 가설    | $H_0$ 채택 | $H_0$ 기각 |
| -------- | -------- | -------- |
| $H_0$가 참 | 올바른 결정   | 제1종 오류   |
| $H_1$이 참 | 제2종 오류   | 올바른 결정   |

* **제1종 오류(Type I error)**: $H_0$가 참인데 기각
* **제2종 오류(Type II error)**: $H_1$이 참인데 $H_0$를 채택

**유의수준 *(Significance Level)***  
귀무가설이 참일 때 이를 잘못 기각할 확률의 최대 허용 한계를 **유의수준** $\alpha$라 한다.

$$
P(\text{기각 } H_0 \mid H_0 \text{ 참}) \le \alpha
$$
- 뚜렷한 반증이 있을 때에 기각하고자 하는 가설이 귀무가설이므로, 이런 결정에 따르는 오류, 즉 제 1종오류 확률이 미리 지정한 작은 값 이하인 점정을 사용하도록 한다.
- 제 1종오류를 범할 확률의 최대 허용한계를 $\alpha$, 유의수준(significance level)이라 한다.
- 보통 $\alpha = 0.1, 0.05, 0.01$을 사용한다.

#### 예 7.1.2
예7.1.1에서 유의수준 $\alpha = 0.05$에서 위의 검정을 고려한다.  
귀무가설 $H_0: \mu = 30$, 대립가설 $H_1: \mu < 30$이고, $\bar X \sim N(\mu, 5^2/100)$이므로, 기각역을 $\bar X \leq 29$로 설정한 검정에서
$$
P(\bar X \leq 29 \mid H_0) = P\left( Z \leq \frac{29-30}{5/\sqrt{100}} \right) = P(Z \leq -2) \approx 0.023
$$
이 되어, 이는 $0.05$ 이하이므로 유의수준 $\alpha = 0.05$의 검정이 된다.
임계값 $z_{0.05} = 1.645$이므로,
$$
\bar X \le 30 - 1.645 \times \frac{5}{\sqrt{100}}
= 29.1775
$$
가 기각역이 된다 (유의수준 0.05의 검정이다). 이는 $P(\text{기각 } H_0 \mid H_0) = 0.05$를 만족한다.

#### 예 7.1.3
위의 예에서, $H_0$가정 하에 임계값이 $29$인 경우와 $29.1775$인 경우의 기각확률을 비교하면,  
제1종 오류를 제한하는 유의수준에 의해 기각역의 범위가 우선적으로 정해진 상태에서, 그 범위 내에서 기각역을 더 넓게 설정할수록 제2종 오류가 감소함을 알 수 있다. 즉, 유의수준(제1종 오류 확률)을 유지하면서 기각역을 최대화하는 것이 제2종 오류(검정력 부족)를 줄이는 방향임을 확인할 수 있다.

### 검정력 함수 *(Power Function)*
모평균이 $\mu$일 때 귀무가설이 기각될 확률을 검정력 함수 $\gamma(\mu)$라 하고, 표준정규분포의 누적함수 $\Phi$로 나타낼 수 있다. 

$$
\gamma(\mu)
= P_\mu(\bar X \le c)
= \Phi\left(\frac{c-\mu}{5/\sqrt{100}}\right)
$$

검정력 함수는 $\mu$의 감소함수이며, $\mu < 30$일수록 커진다.

#### 예 7.1.4
$X_i \sim \text{Poisson}(\theta)$, $n=100$일 때

* $H_0:\theta=0.1$
* $H_1:\theta<0.1$

$X_1+\cdots+X_{100} \sim \text{Poisson}(10)$ 이며,
정확한 유의수준 $0.05$를 만족하는 결정적 기각역이 존재하지 않으므로 랜덤화 검정을 사용한다.

### 일반적인 검정의 수학적 구조
모집단 분포가 $f(x;\theta)$로 주어지고, $\theta \in \Omega$라 하자.

* 귀무가설: $H_0 : \theta \in \Omega_0$
* 대립가설: $H_1 : \theta \in \Omega_1$
* ($\Omega_0 \cap \Omega_1 = \varnothing$)

랜덤표본 $X_1, \dots, X_n$의 관측결과가 대립가설($H_1$)의 증거로서 (귀무가설($H_0$)에 대한 반증으로서) 확률의 정도가 뚜렷한가를 판단하는 것이 검정의 과정이다.  

기각역 $C_\alpha$는 다음을 만족하도록 설정된다.
$$
\sup_{\theta \in \Omega_0} P_\theta\left((X_1,\dots,X_n)\in C_\alpha\right) \le \alpha
$$
- 잘못해서 귀무가설을 기각하는 확률의 최대 허용한계가 $\alpha$
- 즉 제 2종 오류 확률을 줄이려면 기각역을 크게 해야하므로
### 랜덤화 검정 *(Randomized Test)*
어떤 경우에는 위 조건을 정확히 만족하는 기각역이 존재하지 않을 수 있다.
이때는 기각 여부를 확률적으로 결정하는 랜덤화 검정을 사용한다.  
>(책 표현: 기각역의 확률이 예를들어 0.05 미만에서 초과로 변하는 경계에서 확률적으로 기각하는 방법)
$\sum x_1, \dots, x_n \leq 4$이면 $H_0$를 기각하고, $\sum x_1, \dots, >x_n = 5$이면 $\gamma$의 확률로 기각한다. 즉, 검정함수는 다음과 같이 정의된다.
>$$
>\phi(x_1, \dots, x_n) =
>\begin{cases}
>1 & \text{if } \sum x_i \leq 4 \\
>\gamma & \text{if } \sum x_i = 5 \\
>0 & \text{if } \sum x_i > 5
>\end{cases}
>$$
>여기서 $\gamma$는 다음 식을 만족하도록 정해진다.
>$$
P\left(\sum x_i \leq 4\right) + \gamma\, P\left(\sum x_i = 5\right) = >\alpha
>$$
>따라서 $\gamma = \dfrac{\alpha - P(\sum x_i \leq 4)}{P(\sum x_i = 5)}$로 계산한다.

검정함수 $\phi(x_1,\dots,x_n)$를 다음과 같이 정의한다.

* $\phi = 1$: 귀무가설 기각
* $0 < \phi < 1$: 확률적으로 기각
* $\phi = 0$: 귀무가설 채택

검정력 함수(test function)는
$$
\gamma_\phi(\theta) = E_\theta[\phi(X_1,\dots,X_n)], \quad \theta \in \Omega
$$
검정의 크기(size)는
$$
\sup_{\theta \in \Omega_0} \gamma_\phi(\theta) = \max_{\theta \in \Omega_0} E_\theta(\phi(X_1, \dots, X_n))
$$
이며, 유의수준 $\alpha$ 이하인 검정을 유의수준 $\alpha$의 검정이라 한다.

이런 유의수준을 만족시키는 검정 중에서 제 2종 오류를 작게 하는 것이 좋으므로, 가능한 기각확률을 크게 주기 위하여 그 크기가 지정된 유의수준과 같게되는, 즉 
$$
\max_{\theta \in \Omega_0} E_\theta(\phi(X_1, \dots, X_n)) = \alpha
$$
인 검정을 사용하는 것이다!

#### 예 7.1.5
정규분포 평균에 대한 $H_0:\mu \ge 30 \quad \text{vs} \quad H_1:\mu < 30$
의 경우에도 유의수준 $\alpha$에서의 기각역은
$$
\bar X \le 30 - z_\alpha \frac{5}{\sqrt{100}}
$$
유의수준 $\alpha$에서의 기각역 $C$가 $\bar X \le 30 - z_\alpha \frac{5}{\sqrt{100}}$일 때, 모평균이 $\mu$일 경우의 검정력 함수는  
$$
\gamma(\mu) = P_\mu(\bar X \le 30 - z_\alpha \frac{5}{\sqrt{100}})
= \Phi\left( \frac{30 - z_\alpha \frac{5}{\sqrt{100}} - \mu}{5/\sqrt{100}} \right)
$$
여기서 $\Phi$는 표준정규분포의 누적분포함수이다.  
$\mu < 30$일수록 검정력 $\gamma(\mu)$가 커진다.

#### 예 7.1.6
예 7.1.4의 포아송 분포 평균 $\theta$에 대한 귀무가설, 대립가설은 $H_0:\theta = 0.1 \quad \text{vs} \quad H_1:\theta < 0.1$이고, 크기 100 랜덤표본으로 5%유의수준 검정을 하면, 랜덤화 검정의 검정력 함수는
$$
\gamma(\theta)
= P_\theta(X_1+\cdots+X_{100}\le 4)

* \frac{21}{38}P_\theta(X_1+\cdots+X_{100}=5)
  $$
  로 주어지며, $\theta$에 따라 단조 감소한다.


## 최대가능도 검정법 *(Maximum Likelihood Ratio Test)*
### 7.2 최대가능도 검정법의 기본 아이디어
모집단의 확률밀도함수 $f(x;\theta)$, $\theta\in\Omega$에서 랜덤표본 $X_1,\dots,X_n$을 관측했다고 하자.  
이때 모수 $\theta$에 대한 가설은 다음과 같이 **모수공간의 분할**로 표현된다.

$$
H_0:\theta\in\Omega_0
\quad\text{vs}\quad
H_1:\theta\in\Omega_1,
\qquad
\Omega_0\cap\Omega_1=\varnothing,\quad
\Omega_0\cup\Omega_1=\Omega
$$

관측값 $x=(x_1,\dots,x_n)^t$에 대해, 각 가설 하에서 이 데이터가 가장 잘 설명되는 **최대가능도**를 비교하여 가설을 판단한다.  
랜덤표본 $X_1,\dots,X_n$의 **가능도함수**는
$$
L(\theta;x)=\prod_{i=1}^n f(x_i;\theta),\qquad \theta\in\Omega
$$

- **전체 모수공간**에서의 최대가능도: $\displaystyle \max_{\theta\in\Omega} L(\theta;x)$  
- **귀무가설 하** 최대가능도: $\displaystyle \max_{\theta\in\Omega_0} L(\theta;x)$  
- **대립가설 하** 최대가능도: $\displaystyle \max_{\theta\in\Omega_1} L(\theta;x)$

전체 공간에서의 최대가능도는 두 값 중 큰 값이 된다.
$$
\max_{\theta\in\Omega}L(\theta;x)
=
\max\left\{
\max_{\theta\in\Omega_0}L(\theta;x),\
\max_{\theta\in\Omega_1}L(\theta;x)
\right\}
$$

**최대가능도비 검정통계량**  
관측값 $x$에 대해 **최대가능도비**를 다음과 같이 정의한다.
$$
\Lambda(x)
=
\frac{\max_{\theta\in\Omega_0}L(\theta;x)}
    {\max_{\theta\in\Omega}L(\theta;x)}
$$

- $\Lambda(x)$가 **작을수록** 귀무가설 하 최대가능도가 상대적으로 작으므로, 귀무가설에 불리한 증거가 된다.
- 따라서 $\Lambda(x)$가 충분히 작으면 귀무가설을 기각한다.

**로그가능도 표현**  
로그가능도함수를 $\ell(\theta;x)=\log L(\theta;x)$라 하면,
$$
\hat\theta = \arg\max_{\theta\in\Omega}\ell(\theta;x),\qquad
\hat\theta_0 = \arg\max_{\theta\in\Omega_0}\ell(\theta;x)
$$
$$
-2\log\Lambda(x)
=
2\bigl(\ell(\hat\theta;x)-\ell(\hat\theta_0;x)\bigr)
$$

이때, 전체 모수공간 $\Omega$에서의 최대가능도는
$$
\max_{\theta\in\Omega} L(\theta;x)
=
\max\left\{
\max_{\theta\in\Omega_0} L(\theta;x),\ 
\max_{\theta\in\Omega_1} L(\theta;x)
\right\}
$$
와 같이, 귀무가설과 대립가설 각각에서의 최대가능도 중 더 큰 값이 된다.

**기각역의 일반형**  
유의수준 $\alpha$에서의 최대가능도비 검정의 기각역은
$$
C_\alpha
=
\left\{
x:\ 2\bigl(\ell(\hat\theta;x)-\ell(\hat\theta_0;x)\bigr)\ge c
\right\}
$$
여기서 상수 $c$는 다음 **유의수준 조건**을 만족하도록 정한다.
$$
\max_{\theta\in\Omega_0}
P_\theta\bigl((X_1,\dots,X_n)^t\in C_\alpha\bigr)
=
\alpha
$$
  - 필요하면 이 조건을 만족시키기 위하여 예 7.1.4처럼 랜덤화 검정을 고려한다.
  - 각 가설 하에서의 MLE(최대우도추정량) 계산은 **모수공간의 제약**에 따라 달라진다.  
    - **전체 모수공간**에서는 $\theta$가 가질 수 있는 모든 값에 대해 가능도함수를 최대화하여 MLE를 구한다. 즉, 제한 없이 최적의 값을 찾는다.
    - **귀무가설 하**에서는 $\theta$가 반드시 귀무가설이 허용하는 값(예: $\theta = \theta_0$ 또는 $\mu \le \mu_0$ 등)만 가질 수 있으므로, 이 **제약된 모수공간** 내에서만 가능도함수를 최대화한다.
        - 전체 모수공간에서의 MLE는 **제약이 없는 최적화**의 결과이고,  
        - 귀무가설 하의 MLE는 **가설이 허용하는 범위 내에서의 최적화** 결과이다.
        - 이렇게 두 가지 MLE를 비교함으로써, 실제 데이터가 귀무가설 하에서 얼마나 잘 설명되는지(가능도의 손실이 얼마나 큰지)를 판단할 수 있다.

### 정리: 최대가능도비 검정 (Maximum Likelihood Ratio Test)
가능도비 검정(likelihood ratio test)나 우도비 검정이라고도 함.  

유의수준 $\alpha$에서 $H_0:\theta\in\Omega_0
\quad\text{vs}\quad
H_1:\theta\in\Omega_1$을 검정할 때,

- **검정통계량**: $2\bigl(\ell(\hat\theta)-\ell(\hat\theta_0)\bigr)$
- **기각역**: $C_\alpha = \left\{x:\ 2(\ell(\hat\theta)-\ell(\hat\theta_0))\ge c\right\}$
  - 참고: Wilks 정리에 의해, 앞의 2는 ‘$\chi^2$ 근사를 가장 표준적인 형태로 만들기 위한 정규화 상수
- **상수 $c$** 는 $\max_{\theta\in\Omega_0}P_\theta((X1, \dots, X_n)^T \in C_\alpha)=\alpha$가 되도록 선택

#### 예 7.2.1 정규분포 평균에 대한 양측 검정
$$
X_1,\dots,X_n \sim N(\mu,\sigma^2),\quad n\ge2 \\
H_0:\mu=\mu_0 \quad\text{vs}\quad H_1:\mu\neq\mu_0
$$
유의수준 $\alpha$의 최대가능도비 검정을 구하라.  
**풀이**  
로그가능도함수는
$$
\ell(\mu,\sigma^2) = -\frac{1}{2\sigma^2}\sum_{i=1}^n(x_i-\mu)^2 - \frac{n}{2}\log\sigma^2 - \frac{n}{2}\log(2\pi)
$$
- 전체 모수공간에서의 MLE:
    $$
    \hat\mu = \bar x, \quad \hat\sigma^2 = \frac{1}{n}\sum_{i=1}^n(x_i-\bar x)^2
    $$
- 귀무가설 하 MLE:
    $$
    \hat\mu_0 = \mu_0, \quad \hat\sigma_0^2 = \frac{1}{n}\sum_{i=1}^n(x_i-\mu_0)^2
    $$
따라서 최대가능도비 검정통계량은
$$
2(\ell(\hat\theta)-\ell(\hat\theta_0)) = n\log\frac{\hat\sigma_0^2}{\hat\sigma^2} = n\log\left(1+\frac{(\bar x-\mu_0)^2}{\hat\sigma^2}\right)
$$
- 평균은 각 가설 하에서 이미 최적으로 선택되어 소거됨

**$t$-통계량과의 연결**  
$\hat\sigma^2$는 표본분산 $S^2$로 대체할 수 있으므로,
$$
\frac{|\bar X-\mu_0|}{S/\sqrt n}, \ 
S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar X)^2
$$
형태의 $t$-통계량과 본질적으로 같다.

> 참고: 크기 n이 충분히 크고 모분산을 알면 모집단 분포와 상관없이 정규분포를 사용하여 검정할 수 있다.
> 모집단이 정규분포를 따르지만 모분산을 모를 때, $\mu$에 관해 검정하는 경우 표본크기가 작으면 $t$-분포를 사용하고, 크면 $Z$-분포를 사용한다

**기각역의 해석**  
따라서, 유의수준 $\alpha$에서의 기각역은
$$
\left|\frac{\bar X-\mu_0}{S/\sqrt n}\right| \ge t_{\alpha/2}(n-1)
$$
가 된다. 즉, **양측(two-sided) $t$-검정**과 완전히 동일하다.

> 최대가능도비 검정은 귀무가설 하에서 평균을 $\mu_0$로 고정했을 때, 실제 데이터의 평균과의 차이로 인해 발생하는 가능도(우도)의 손실이 얼마나 큰지를 측정한다. 이 손실이 충분히 크면(즉, $t$-통계량이 임계값을 넘으면) 귀무가설을 기각한다.  
> 즉, $t$-검정은 최대가능도비 검정의 특수한 경우로 해석할 수 있다.

#### 예 7.2.2 정규분포 평균에 대한 한쪽 검정
$$
H_0:\mu\le\mu_0 \quad\text{vs}\quad H_1:\mu>\mu_0
$$
유의수준 $\alpha \ (0 < \alpha < 1)$의 최대가능도비 검정을 구하라.

1. **로그가능도함수**  
    $X_1,\dots,X_n \sim N(\mu, \sigma^2)$에서 로그가능도함수는
    $$
    \ell(\mu, \sigma^2) = -\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2 - \frac{n}{2}\log\sigma^2 + \text{상수}
    $$

2. **MLE 계산**  
    - 전체 모수공간($\mu \in \mathbb{R}$)에서의 MLE:
      $$
      \hat\mu = \bar x, \quad \hat\sigma^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \bar x)^2
      $$
    - 귀무가설 하($\mu \le \mu_0$)의 MLE:
      - 만약 $\bar x \le \mu_0$이면 $\hat\mu_0 = \bar x$
      - 만약 $\bar x > \mu_0$이면 $\hat\mu_0 = \mu_0$
      $$
      \hat\mu_0 = \min(\bar x, \mu_0), \quad \hat\sigma_0^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \hat\mu_0)^2
      $$

3. **가능도비 검정통계량**  
    $$
    2(\ell(\hat\theta) - \ell(\hat\theta_0)) = n\log\frac{\hat\sigma_0^2}{\hat\sigma^2}
    $$
    - $\bar x \le \mu_0$이면 $\hat\mu_0 = \bar x$이므로 $\hat\sigma_0^2 = \hat\sigma^2$이고, 검정통계량은 0이 되어 기각하지 않음.
    - $\bar x > \mu_0$이면 $\hat\mu_0 = \mu_0$
    - $\hat\sigma_0^2 = \frac{1}{n}\left(\sum_{i=1}^n (x_i - \bar x)^2 + n(\bar x - \mu_0)^2\right)$로 분해되므로, 
      $$
      n\log\left(1 + \frac{(\bar x - \mu_0)^2}{\hat\sigma^2}\right)
      $$
    4. **기각역 도출**  
        - $S^2$는 표본분산, $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar X)^2$
        - $\bar x \le \mu_0$이면 기각하지 않음.
        - $\bar x > \mu_0$이면, 검정통계량
          $$
          t = \frac{\bar X - \mu_0}{S/\sqrt{n}}
          $$
          을 사용한다.
        - 자세한 유도과정은 넣지 않았음. 교과서 307~308p 참고
        - 귀무가설 $H_0$ 하에서 이 $t$-통계량은 자유도 $n-1$의 $t$-분포를 따른다.  
          따라서, 유의수준 $\alpha$를 만족시키기 위해
          $$
          P_{\mu_0}\left( T \ge t_\alpha(n-1) \right) = \alpha
          $$
          가 되도록 임계값 $t_\alpha(n-1)$를 선택하여 기각역을 설정한다.

        - 즉, **유의수준 $\alpha$에서의 기각역**은
          $$
          \frac{\bar X - \mu_0}{S/\sqrt{n}} \ge t_\alpha(n-1)
          $$
          이다.

        - 이는 한쪽(one-sided) $t$-검정과 완전히 동일하며, 실제로 최대가능도비 검정이 한쪽 $t$-검정과 일치함을 의미한다.

> 최대가능도비 검정은 귀무가설 하에서 평균을 $\mu_0$로 고정했을 때 실제 데이터의 평균과의 차이로 인한 가능도 손실이 충분히 크면 귀무가설을 기각한다. 이때의 검정통계량은 한쪽 $t$-검정과 일치한다.

**양측검정 vs. 한측검정 비교**

| 구분         | 양측검정 (Two-sided)                | 한측검정 (One-sided)                |
|--------------|-------------------------------------|-------------------------------------|
| 귀무가설     | $H_0: \mu = \mu_0$                  | $H_0: \mu \le \mu_0$ 또는 $H_0: \mu \ge \mu_0$ |
| 대립가설     | $H_1: \mu \neq \mu_0$               | $H_1: \mu > \mu_0$ 또는 $H_1: \mu < \mu_0$    |
| 기각역       | $\|\bar X - \mu_0\| \ge$ 임계값       | $\bar X - \mu_0 \ge$ 임계값 또는 $\bar X - \mu_0 \le$ 임계값 |
| 검정통계량   | $\|t\| \ge t_{\alpha/2}$              | $t \ge t_\alpha$ 또는 $t \le -t_\alpha$       |
| 적용 상황    | 평균이 기준값과 다를지(크거나 작을지) 모두 검증 | 평균이 기준값보다 클지/작을지 한 방향만 검증 |

#### 예 7.2.3 지수분포 평균에 대한 검정
$$
X_1,\dots,X_n \sim \mathrm{Exp}(\theta) \\
H_0:\theta=\theta_0 \quad\text{vs}\quad H_1:\theta\neq\theta_0
$$

1. **가능도함수 및 로그가능도함수**  
$$
L(\theta; x) = \prod_{i=1}^n \frac{1}{\theta} e^{-x_i/\theta}
= \theta^{-n} \exp\left(-\frac{\sum x_i}{\theta}\right) \\
\ell(\theta; x) = -n\log\theta - \frac{\sum x_i}{\theta}
$$

2. **MLE 계산**  
- 전체 모수공간($\theta > 0$)에서의 MLE:
    $$
    \frac{\partial \ell}{\partial \theta} = -\frac{n}{\theta} + \frac{\sum x_i}{\theta^2} = 0 \\
    \Rightarrow \hat\theta = \bar X = \frac{1}{n}\sum x_i
    $$
- 귀무가설 하($\theta = \theta_0$)의 MLE: $\hat\theta_0 = \theta_0$

3. **최대가능도비 검정통계량**  
$$
2\left(\ell(\hat\theta) - \ell(\hat\theta_0)\right)
= 2\left[
    -n\log\bar X - n
    + n\log\theta_0 + \frac{n\bar X}{\theta_0}
\right] \\
= 2n\left(
    \frac{\bar X}{\theta_0} - 1 - \log\frac{\bar X}{\theta_0}
\right)
$$

4. **기각역**  
- 검정통계량 $2n\left(\frac{\bar X}{\theta_0} - 1 - \log\frac{\bar X}{\theta_0}\right)$이 충분히 크면 $H_0$를 기각한다.
- 이는 양측 검정(two-sided test)이므로, 임계값 $c$를 정해
    $$
    2n\left(\frac{\bar X}{\theta_0} - 1 - \log\frac{\bar X}{\theta_0}\right) \ge c
    $$
    이면 $H_0$를 기각한다.

> 참고: 대수의 법칙에 의해 $n$이 충분히 크면 이 검정통계량은 자유도 1의 $\chi^2$ 분포로 근사된다. 따라서 유의수준 $\alpha$에서 임계값 $c = \chi^2_{1,\alpha}$를 사용한다.

#### 예 7.2.4 두 정규분포 평균 비교를 위한 검정
두 집단에서 각각 랜덤표본 $X_{11},\dots,X_{1n_1} \sim N(\mu_1, \sigma^2)$, $X_{21},\dots,X_{2n_2} \sim N(\mu_2, \sigma^2)$를 관측했다고 하자. 두 집단의 분산은 동일하다고 가정한다.
- **귀무가설**: $H_0: \mu_1 = \mu_2$
- **대립가설**: $H_1: \mu_1 \neq \mu_2$

**1. 가능도함수 및 로그가능도함수**  
$$
L(\mu_1, \mu_2, \sigma^2) = \prod_{i=1}^{n_1} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x_{1i}-\mu_1)^2}{2\sigma^2}}
\prod_{j=1}^{n_2} \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x_{2j}-\mu_2)^2}{2\sigma^2}}
$$
$$
\ell(\mu_1, \mu_2, \sigma^2) = -\frac{n_1+n_2}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\left(\sum_{i=1}^{n_1}(x_{1i}-\mu_1)^2 + \sum_{j=1}^{n_2}(x_{2j}-\mu_2)^2\right)
$$

**2. MLE 계산**  
- 전체 모수공간($\mu_1, \mu_2$ 자유):  
    $$
    \hat\mu_1 = \bar X_1 = \frac{1}{n_1}\sum_{i=1}^{n_1} x_{1i}, \quad \hat\mu_2 = \bar X_2 = \frac{1}{n_2}\sum_{j=1}^{n_2} x_{2j}
    $$
    $$
    \hat\sigma^2 = \frac{1}{n_1+n_2}\left(\sum_{i=1}^{n_1}(x_{1i}-\bar X_1)^2 + \sum_{j=1}^{n_2}(x_{2j}-\bar X_2)^2\right)
    $$
- 귀무가설 하($\mu_1 = \mu_2 = \mu$):  
    $$
    \hat\mu_0 = \frac{n_1\bar X_1 + n_2\bar X_2}{n_1+n_2}
    $$
    $$
    \hat\sigma_0^2 = \frac{1}{n_1+n_2}\left(\sum_{i=1}^{n_1}(x_{1i}-\hat\mu_0)^2 + \sum_{j=1}^{n_2}(x_{2j}-\hat\mu_0)^2\right)
    $$

**3. 최대가능도비 검정통계량**  
$$
2(\ell(\hat\theta) - \ell(\hat\theta_0)) = (n_1+n_2)\log\frac{\hat\sigma_0^2}{\hat\sigma^2}
$$
이때 
$$
\sum_{i=1}^{n_1}(x_{1i}-\hat\mu_0)^2 = \sum_{i=1}^{n_1}(x_{1i}-\bar X_1)^2 + n_1(\bar X_1 - \hat\mu_0)^2 \\
\sum_{j=1}^{n_2}(x_{2j}-\hat\mu_0)^2 = \sum_{j=1}^{n_2}(x_{2j}-\bar X_2)^2 + n_2(\bar X_2 - \hat\mu_0)^2
$$

따라서,
$$
\hat\sigma_0^2 = \frac{1}{n_1+n_2} \left[ \sum_{i=1}^{n_1}(x_{1i}-\bar X_1)^2 + \sum_{j=1}^{n_2}(x_{2j}-\bar X_2)^2 + n_1(\bar X_1 - \hat\mu_0)^2 + n_2(\bar X_2 - \hat\mu_0)^2 \right]
$$

또한,
$$
n_1(\bar X_1 - \hat\mu_0)^2 + n_2(\bar X_2 - \hat\mu_0)^2 = \frac{n_1 n_2}{n_1+n_2} (\bar X_1 - \bar X_2)^2
$$

따라서,
$$
\hat\sigma_0^2 = \hat\sigma^2 + \frac{n_1 n_2}{n_1+n_2} (\bar X_1 - \bar X_2)^2
$$

최종적으로 최대가능도비 검정통계량은
$$
2(\ell(\hat\theta) - \ell(\hat\theta_0)) = (n_1+n_2)\log\left(1 + \frac{n_1 n_2}{(n_1+n_2)^2} \frac{(\bar X_1 - \bar X_2)^2}{\hat\sigma^2}\right)
$$

**4. 기각역**  
최대가능도비 검정의 기각역은
$$
\frac{n_1 n_2}{(n_1+n_2)^2} \frac{(\bar X_1 - \bar X_2)^2}{\hat\sigma^2} \ge d
$$
와 같이 표현된다. 여기서 $c$ 또는 $d$는 유의수준 $\alpha$에서 정해지는 임계값이다.

**5. $t$-검정과의 연결**  
이 통계량은 다음과 같이 표준화된 두 표본 평균의 차이로 표현된다.
$$
T = \frac{\bar X_1 - \bar X_2}{S_p\sqrt{1/n_1 + 1/n_2}}
$$
여기서 $S_p^2$는 두 집단의 결합표본분산(pooled variance)이다.
$$
S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}
$$
$$
S_1^2 = \frac{1}{n_1-1}\sum_{i=1}^{n_1}(x_{1i}-\bar X_1)^2, \quad S_2^2 = \frac{1}{n_2-1}\sum_{j=1}^{n_2}(x_{2j}-\bar X_2)^2
$$

이때, 자유도 $n_1+n_2-2$인 $t$-분포를 사용하여 유의수준 $\alpha$에서의 기각역은
$$
\left|T\right| \ge t_{\alpha/2}(n_1+n_2-2)
$$
즉, **두 표본 $t$-검정**과 완전히 동일하다.

#### 예 7.2.5 두 정규분포의 분산 비교를 위한 검정
두 집단에서 각각 $X_{11},\dots,X_{1n_1} \sim N(\mu_1, \sigma_1^2)$, $X_{21},\dots,X_{2n_2} \sim N(\mu_2, \sigma_2^2)$를 관측했다고 하자. $H_0: \sigma_1^2 = \sigma_2^2,\ H_1: \sigma_1^2 \neq \sigma_2^2$

**1. 가능도함수 및 로그가능도함수**  
$$
L(\sigma_1^2, \sigma_2^2) = \prod_{i=1}^{n_1} \frac{1}{\sqrt{2\pi\sigma_1^2}} e^{-\frac{(x_{1i}-\mu_1)^2}{2\sigma_1^2}}
\prod_{j=1}^{n_2} \frac{1}{\sqrt{2\pi\sigma_2^2}} e^{-\frac{(x_{2j}-\mu_2)^2}{2\sigma_2^2}} \\
\ell(\mu_1, \mu_2, \sigma_1^2, \sigma_2^2) = -\frac{n_1}{2}\log(2\pi\sigma_1^2) - \frac{1}{2\sigma_1^2} \sum_{i=1}^{n_1} (x_{1i} - \mu_1)^2
- \frac{n_2}{2}\log(2\pi\sigma_2^2) - \frac{1}{2\sigma_2^2} \sum_{j=1}^{n_2} (x_{2j} - \mu_2)^2
$$

**2. MLE 계산**  
- 전체 모수공간($\mu_1, \mu_2, \sigma_1^2, \sigma_2^2$ 자유):  
    $$
    \hat\mu_1 = \bar X_{1} = \frac{1}{n_1}\sum_{i=1}^{n_1} x_{1i},\quad
    \hat\mu_2 = \bar X_{2} = \frac{1}{n_2}\sum_{j=1}^{n_2} x_{2j}
    $$
    $$
    \hat\sigma_1^2 = S_1^2 = \frac{1}{n_1}\sum_{i=1}^{n_1} (x_{1i} - \bar X_{1})^2,\quad
    \hat\sigma_2^2 = S_2^2 = \frac{1}{n_2}\sum_{j=1}^{n_2} (x_{2j} - \bar X_{2})^2
    $$
- 귀무가설 하($\sigma_1^2 = \sigma_2^2 = \sigma^2$):  
    $$
    \hat\mu_1 = \bar X_{1},\quad \hat\mu_2 = \bar X_{2}
    $$
    $$
    \hat\sigma^2 = S_p^2 = \frac{n_1 S_1^2 + n_2 S_2^2}{n_1 + n_2}
    $$

**3. 최대가능도비 검정통계량**  
$$
2(\ell(\hat\theta) - \ell(\hat\theta_0)) = n_1 \log\frac{S_p^2}{S_1^2} + n_2 \log\frac{S_p^2}{S_2^2}
$$

**4. 기각역 및 $F$-검정과의 연결**  
최대가능도비 검정통계량  
$$
2(\ell(\hat\theta) - \ell(\hat\theta_0)) = n_1 \log\frac{S_p^2}{S_1^2} + n_2 \log\frac{S_p^2}{S_2^2}
$$
은 두 표본 분산의 비로 표현할 수 있다. 여기서 $S_1^2$와 $S_2^2$는 각각 두 집단의 표본분산, $S_p^2$는 결합표본분산이다.

표본분포 이론에 따르면, 귀무가설 $H_0: \sigma_1^2 = \sigma_2^2$가 참일 때  
$$
\frac{S_1^2}{S_2^2}
$$
는 자유도 $(n_1-1, n_2-1)$의 $F$-분포를 따른다. 즉,
$$
\frac{S_1^2/\sigma^2}{S_2^2/\sigma^2} = \frac{S_1^2}{S_2^2} \sim F(n_1-1, n_2-1)
$$
이므로, 유의수준 $\alpha$에서의 최대가능도비 검정의 기각역은 다음과 같이 설정된다.
$$
\frac{S_1^2}{S_2^2} \ge F_{1-\alpha/2}(n_1-1, n_2-1)\quad \text{또는}\quad \frac{S_1^2}{S_2^2} \le F_{\alpha/2}(n_1-1, n_2-1)
$$
즉, 표본분산의 비가 $F$-분포의 양쪽 임계값을 벗어날 때 귀무가설을 기각한다.

**5. $F$-검정과의 관계**  
이 결과는 바로 두 표본 $F$-검정과 완전히 동일하다. 즉, 두 집단의 분산이 같은지 검정할 때, 표본분산의 비를 $F$-분포 임계값과 비교하는 $F$-검정이 최대가능도비 검정의 특수한 경우임을 알 수 있다.

#### 예 7.2.6 포아송 분포 평균에 대한 검정과 랜덤화 검정
포아송 분포 $X_1, \dots, X_n \sim \mathrm{Poisson}(\theta)$에서, $H_0: \theta = 0.1, H_1: \theta < 0.1$이고 $n=100$일 때 $\alpha = 0.05$의 최대가능도비 검정을 구해라.
1. **가능도함수 및 로그가능도함수**
    $$
    L(\theta; x) = \prod_{i=1}^{100} \frac{\theta^{x_i} e^{-\theta}}{x_i!} \\
    \ell(\theta; x) = \sum_{i=1}^{100} x_i \log\theta - 100\theta + \text{상수}
    $$

2. **MLE 계산**
    - 전체 모수공간($\theta > 0$)에서의 MLE:
      $$
      \frac{\partial \ell}{\partial \theta} = \frac{\sum x_i}{\theta} - 100 = 0 \implies \hat\theta = \bar X
      $$
    - 귀무가설 하($\theta = 0.1$)의 MLE: $\hat\theta_0 = 0.1$

3. **최대가능도비 검정통계량**
    $$
    2(\ell(\hat\theta) - \ell(\hat\theta_0)) = 2\left[ \sum x_i \log\frac{\bar X}{0.1} - 100(\bar X - 0.1) \right]
    $$

4. **기각역**
    - $H_0$ 하에서 $\sum x_i \sim \mathrm{Poisson}(10)$
    - $\bar X < 0.1$ (즉, $\sum x_i < 10$)일 때 $H_0$에 불리한 증거가 됨
    - $\sum x_i \le k$인 $k$를 찾아 $P(\sum x_i \le k \mid H_0) \le 0.05$가 되도록 설정
    - 정확히 $\alpha=0.05$를 맞추기 위해 $k$에서 랜덤화 검정 적용:
      $$
      \phi(x) =
      \begin{cases}
         1 & \text{if } \sum x_i < k \\
         \gamma & \text{if } \sum x_i = k \\
         0 & \text{if } \sum x_i > k
      \end{cases}
      $$
      여기서 $\gamma$는
      $$
      P(\sum x_i < k) + \gamma P(\sum x_i = k) = 0.05
      $$
      $$
      \gamma = \frac{0.05 - P(\sum x_i < k)}{P(\sum x_i = k)}
      $$


## 최대가능도 검정법의 근사 *(Asymptotic Likelihood Ratio Tests)*
표본 크기가 충분히 클 때, 최대가능도 추정량의 극한분포를 이용해 최대가능도비 검정통계량의 분포를 근사할 수 있다. 크기 $n$인 랜덤표본 $X_1,\dots,X_n$에 대해 로그가능도함수와 평균 로그가능도함수는 다음과 같이 정의된다.
$$
l(\theta) = \sum_{i=1}^n \log f(X_i;\theta), \qquad
\bar l(\theta) = \frac{1}{n} \sum_{i=1}^n \log f(X_i;\theta)
$$
모수공간 $\Omega$가 $k$차원 열린집합($\Omega \subset \mathbb{R}^k$)이라고 하자.

**최대가능도 추정량의 점근 전개**  
적절한 정규성 조건 하에서, 최대가능도 추정량 $\hat\theta_n$은 다음의 점근 전개를 만족한다.
$$
\sqrt{n}(\hat\theta_n - \theta)
= [I(\theta)]^{-1} \sqrt{n}\, \bar l'(\theta) + o_p(1)
$$

- $\hat\theta_n$: 표본 크기 $n$에서 관측된 데이터로부터 계산한 **최대가능도추정량(MLE, Maximum Likelihood Estimator)**으로, 실제로는 $\theta$의 "추정값". 
  - $\theta$는 미지의 진짜 값이고, $\hat\theta_n$은 표본 데이터에 기반해 계산된 값이다.
- $I(\theta)$: 정보량 행렬
    $$
    I(\theta) = \mathrm{Var}_\theta\left( \frac{\partial}{\partial\theta} \log f(X_1;\theta) \right)
    = E_\theta\left[ -\frac{\partial^2}{\partial\theta\partial\theta^t} \log f(X_1;\theta) \right]
    $$
- $\bar l'(\theta)$: 평균 점수함수
    $$
    \bar l'(\theta) = \left(
        \frac{\partial}{\partial\theta_1} \bar l(\theta), \dots,
        \frac{\partial}{\partial\theta_k} \bar l(\theta)
    \right)^t
    $$

**단순 귀무가설에서의 최대가능도비 통계량**  
귀무가설이 $H_0: \theta = \theta_0$로 주어질 때, 최대가능도비 검정통계량은
$$
2\{l(\hat\theta_n) - l(\theta_0)\}
$$

**테일러 전개와 근사**  
로그가능도함수를 $\theta_0$에서 $\hat\theta_n$까지 테일러 전개하면,
$$
l(\theta_0) \approx l(\hat\theta_n)
+ l'(\hat\theta_n)^t (\theta_0 - \hat\theta_n)
+ \frac{1}{2} (\theta_0 - \hat\theta_n)^t l''(\theta_n^*) (\theta_0 - \hat\theta_n)
$$
($\theta_n^*$는 $\theta_0$와 $\hat\theta_n$ 사이의 값)

- $\hat\theta_n$은 가능도방정식의 해이므로 $l'(\hat\theta_n) = 0$
- $-\frac{1}{n} l''(\theta_n^*) \simeq -\bar{\ddot l}(\theta_0) \simeq E_{\theta_0}\left[ -\frac{\partial^2}{\partial\theta_0^2}\log f(X_1;\theta_0) \right]\to I(\theta_0)$ (대수의 법칙)

따라서, 
$$
l(\theta_0) \approx l(\hat\theta_n) - \frac{n}{2} (\theta_0 - \hat\theta_n)^t I(\theta_0) (\theta_0 - \hat\theta_n)
$$
즉,
$$
2\{l(\hat\theta_n) - l(\theta_0)\} \approx n (\hat\theta_n - \theta_0)^t I(\theta_0) (\hat\theta_n - \theta_0)
$$

**극한분포**  
$H_0$가 사실일 때,
$$
\sqrt{n}(\hat\theta_n - \theta_0) \xrightarrow{d} N_k(0, [I(\theta_0)]^{-1})
$$
따라서, 정리5.3.2, 정리4.4.5로부터 $H_0$가 사실일 때 
$$
\sqrt n (\hat\theta_n - \theta_0)^t I(\theta_0) \sqrt n  (\hat\theta_n - \theta_0) \xrightarrow{d} \chi^2(k)
$$
즉, 최대가능도비 검정통계량은 표본 크기가 충분히 크면 자유도 $k$인 카이제곱 분포로 근사된다.

### 정리 7.3.1 (단순 귀무가설의 최대가능도비 검정)
확률밀도함수 $f(x;\theta)$, $\theta\in\Omega\subset\mathbb{R}^k$에서 정리 6.4.4의 조건 $(R0)\sim(R7)$이 만족되면,
$$
H_0:\theta=\theta_0 \quad \text{vs} \quad H_1:\theta\neq\theta_0
$$
에 대한 최대가능도비 검정통계량은
$$
2\{l(\hat\theta_n) - l(\theta_0)\} \xrightarrow{d} \chi^2(k)
$$
따라서 표본 크기가 충분히 클 때, 유의수준 $\alpha$에서의 기각역은
$$
2\{l(\hat\theta_n^\Omega) - l(\theta_0)\} \ge \chi^2_\alpha(k)
$$

또는, 기각역을 다음과 같이 근사적으로 표현할 수 있다 (**Wald 검정통계량**).
$$
2\{l(\hat\theta_n) - l(\theta_0)\} = n(\hat\theta_n^\Omega - \theta_0)^T I(\theta_0)(\hat\theta_n^\Omega - \theta_0) + r_n
$$
여기서 $r_n \xrightarrow{P_{\theta_0}} 0$이므로, 표본 크기가 충분히 크면 근사적으로
$$
n(\hat\theta_n^\Omega - \theta_0)^T I(\theta_0)(\hat\theta_n^\Omega - \theta_0) \ge \chi^2_\alpha(k)
$$
인 경우 $H_0$를 기각한다.

또 다른 표현(**Rao(Score) 검정통계량**)으로, 점수함수(평균 로그가능도 1차 미분)를 이용해
$$
2\{l(\hat\theta_n) - l(\theta_0)\} = n\, \bar{\dot l}(\theta_0)^T [I(\theta_0)]^{-1} \bar{\dot l}(\theta_0) + r_n
$$
($r_n \xrightarrow{P_{\theta_0}} 0$)이므로, 표본 크기가 충분히 크면 근사적으로 유의수준 $\alpha$에서의 기각역은
$$
n\, \bar{\dot l}(\theta_0)^T [I(\theta_0)]^{-1} \bar{\dot l}(\theta_0) \ge \chi^2_\alpha(k)
$$

증명은 정리 6.4.4처럼 잉여항의 엄밀한 처리를 통해 할 수 있다. 생략.  

**최대가능도비(MLE), Wald, Rao(Score) 검정 비교**
| 구분         | 최대가능도비 검정 (Likelihood Ratio, MLE) | Wald 검정                      | Rao(Score) 검정                |
|--------------|-------------------------------------------|-------------------------------|-------------------------------|
| **아이디어** | 전체 모수공간과 귀무가설 하 최대가능도의 차이 측정 (귀무가설을 강제로 맞추면 데이터 설명력이 얼마나 떨어지는가) | 추정량과 귀무가설 값의 거리 측정 (추정치가 귀무가설 값에서 통계적으로 의미 있게 멀리 떨어져 있는가) | 점수함수(미분)와 정보량 이용 (귀무가설이 맞다면 그 지점에서 가능도는 평평해야 하는데, 실제로는 얼마나 강하게 증가하려 하는지) |
| **검정통계량** | $2\{l(\hat\theta_n) - l(\theta_0)\}$      | $n(\hat\theta_n - \theta_0)^T I(\theta_0)(\hat\theta_n - \theta_0)$ | $n\, \bar l'(\theta_0)^T [I(\theta_0)]^{-1} \bar l'(\theta_0)$ |
| **계산 위치** | 전체 모수공간, 귀무가설 하 MLE 필요        | 전체 모수공간 MLE 필요         | 귀무가설 하에서만 계산         |
| **분포 근사** | $\chi^2$ 분포 (자유도: 모수 차이)         | $\chi^2$ 분포 (자유도: 모수 차이) | $\chi^2$ 분포 (자유도: 모수 차이) |
| **장점**     | 넓은 적용, 표본 크기 커질수록 강건         | 계산 간단, 직관적 해석         | 귀무가설 하 정보만 필요        |
| **단점**     | MLE 계산 복잡, 수치해석 필요              | 정보량 행렬 추정 필요          | 점수함수, 정보량 계산 필요     |
| **적용 예**  | $t$-검정, $F$-검정, 카이제곱 검정 등      | 회귀계수 유의성, 분산분석 등   | 로지스틱 회귀, 일반화 선형모형 등 |

> **요약:**  
> 세 검정은 표본 크기가 충분히 크면 동일한 결론(동일한 $\chi^2$ 근사)으로 수렴하며, 실제 분석에서는 데이터와 모형 특성에 따라 적절한 방법을 선택한다.

#### 예 7.3.1 지수분포 평균에 대한 검정의 근사
지수분포 $X_1,\dots,X_n \sim \mathrm{Exp}(\theta)$에서
$$
H_0:\theta=\theta_0 \quad\text{vs}\quad H_1:\theta\neq\theta_0
$$
를 검정한다.

- **로그가능도함수**  
    $$
    l(\theta) = \sum_{i=1}^n \log f(X_i;\theta) = -n\log\theta - \frac{1}{\theta}\sum_{i=1}^n X_i
    $$
- **MLE**  
    $$
    \frac{\partial l(\theta)}{\partial\theta} = -\frac{n}{\theta} + \frac{\sum X_i}{\theta^2} \implies \hat\theta = \bar X
    $$
- **정보량**  
    $$
    I(\theta) = E_\theta\left[-\frac{1}{n}\frac{\partial^2 l(\theta)}{\partial\theta^2}\right] = \frac{1}{\theta^2}
    $$
- **최대가능도비 검정통계량**  
    $$
    2\{l(\hat\theta)-l(\theta_0)\} = 2n\left(\frac{\bar X}{\theta_0} - 1 - \log\frac{\bar X}{\theta_0}\right)
    $$
- **근사 분포**  
    $$
    2\{l(\hat\theta)-l(\theta_0)\} \xrightarrow{d} \chi^2(1)
    $$
- **Wald 검정통계량**  
    $$
    n\frac{(\bar X-\theta_0)^2}{\theta_0^2}
    $$
- **기각역**  
    $$
    2n\left(\frac{\bar X}{\theta_0}-1-\log\frac{\bar X}{\theta_0}\right) \ge \chi^2_\alpha(1)
    $$
    또는
    $$
    n\frac{(\bar X-\theta_0)^2}{\theta_0^2} \ge \chi^2_\alpha(1)
    $$

#### 예 7.3.2 다항분포 모형에 대한 검정의 근사
$Z_1,\dots,Z_n \sim \mathrm{Multi}(1,(p_1,\dots,p_k)^t)$에서
$$
H_0:p=p_0 \quad\text{vs}\quad H_1:p\neq p_0
$$
를 검정한다.
- **충분통계량**  
    - **충분통계량**(sufficient statistic)이란, 표본 $(Z_1, \dots, Z_n)$에서 어떤 모수 $p$에 대한 정보를 모두 담고 있어, 충분통계량의 값만 알면 표본 전체를 알 때와 동일하게 $p$에 대한 추론이 가능한 통계량.  
    - 다항분포의 경우, 각 범주별 합계 $X_i = \sum_{j=1}^n Z_{ji}$ $(i=1,\dots,k)$가 $p$에 대한 충분통계량이 된다.
        $$
        X_i = \sum_{j=1}^n Z_{ji},\quad (X_1,\dots,X_k)\sim\mathrm{Multi}(n,(p_1,\dots,p_k)^t)
        $$
- **모수 벡터의 차원**  
    - $p = (p_1, \dots, p_k)^t$이지만, $\sum_{i=1}^k p_i = 1$이므로 자유도는 $r = k-1$이다.
    - 즉, $\theta = (p_1, \dots, p_{k-1})^t \in \mathbb{R}^{k-1}$로 두고, $p_k = 1 - \sum_{i=1}^{k-1} p_i = 1 - \theta\cdot$로 표현할 수 있다.
        $$
        \theta = (p_1, \dots, p_{k-1})^t, \quad r = k-1, \quad \theta\cdot = \sum_{i=1}^{r} \theta_i
        $$
- **로그가능도함수**  
        $$
        l(p) = \sum_{i=1}^k X_i\log p_i + \text{상수} \\
        l(\theta) = \sum_{i=1}^r x_i\log \theta_i + x_k\log(1-\theta\cdot) \\
        l'(\theta) = (x_i/\theta_i - x_k/(1-\theta\cdot))_{1\leq i \leq r} \\
        l''(\theta) = 
        \frac{\partial^2 l(\theta)}{\partial \theta_i \partial \theta_j}
        = \begin{cases}
            -\frac{x_i}{\theta_i^2} - \frac{x_k}{(1-\theta\cdot)^2} & (i = j) \\
            -\frac{x_k}{(1-\theta\cdot)^2} & (i \neq j)
        \end{cases}
        $$
- **MLE**  
    $$  
    \hat p_i = \frac{X_i}{n}  
    $$  
    - $l'(\theta) =0$인 지점에서는,
    - $p_i = \dfrac{X_i}{X_k}(1 - \theta_\cdot)$ $(i=1,\dots,k-1)$
    - $\theta_\cdot = \sum_{i=1}^{k-1} p_i = \sum_{i=1}^{k-1} \dfrac{X_i}{X_k}(1 - \theta_\cdot) = \dfrac{n - X_k}{X_k}(1 - \theta_\cdot)$
    - 정리하면 $\theta_\cdot X_k = (n - X_k)(1 - \theta_\cdot)$
    - $\theta_\cdot X_k + \theta_\cdot (n - X_k) = n - X_k$
    - $\theta_\cdot n = n - X_k \implies \theta_\cdot = \dfrac{n - X_k}{n}$
    - $1 - \theta_\cdot = \dfrac{X_k}{n}$
    - $\therefore p_i = \dfrac{X_i}{X_k}(1 - \theta_\cdot) = \dfrac{X_i}{X_k} \cdot \dfrac{X_k}{n} = \dfrac{X_i}{n}$ $(i=1,\dots,k-1)$

- **정보량 행렬**  
    정보량 행렬(Fisher information matrix)은 $E[X_i] = n p_i$, $E[X_k] = n(1-\theta\cdot)$이므로  
    $$  
    I(p) = -E\left[\frac{1}{n} l''(\theta)\right] = \mathrm{diag}\left(\frac{1}{\theta_i}\right) + \frac{1}{1-\theta\cdot}\mathbf{1}\mathbf{1}^t\\
    = \mathrm{diag}\left(\frac{1}{p_1},\dots,\frac{1}{p_{k-1}}\right) + \frac{1}{1-\theta\cdot}\mathbf{1}\mathbf{1}^t  
    $$  
    - **최대가능도비 검정통계량**  
        $$
        2\{l(\hat p)-l(p_0)\} = 2\sum_{i=1}^k X_i\log\frac{X_i}{np_{0i}}
        $$
    - **근사 분포**  
        $$
        2\{l(\hat p)-l(p_0)\} \xrightarrow{d} \chi^2(k-1)
        $$
        - **Wald 검정통계량**
            $$
            W = n(\hat p - p_0)^T \left[I(\theta_0)\right] (\hat p - p_0) \\
            = \sum_{i=1}^k \frac{(X_i - n p_{0i})^2}{n p_{0i}}
            $$
        - **Rao(Score) 검정통계량**
            $$
            S = \left(\frac{\partial l}{\partial p}\Big|_{p_0}\right)^T \left[I(\theta_0)\right]^{-1} \left(\frac{\partial l}{\partial p}\Big|_{p_0}\right)
            $$
            다항분포에서 점수함수와 정보량 행렬을 대입하면 역시
            $$
            S = \sum_{i=1}^k \frac{(X_i - n p_{0i})^2}{n p_{0i}}
            $$
        - 즉, **Wald, Rao, 최대가능도비 검정통계량 모두** 표본 크기가 충분히 크면 아래의 Pearson 카이제곱 통계량과 근사적으로 일치한다.
            $$
            \sum_{i=1}^k \frac{(X_i - n p_{0i})^2}{n p_{0i}}
            $$

    - **기각역**  
        $$
        \sum_{i=1}^k\frac{(X_i-np_{0i})^2}{np_{0i}} \ge \chi^2_\alpha(k-1)
        $$

    > **정리:**  
    > 다항분포에서 귀무가설 $H_0: p = p_0$을 검정할 때, 최대가능도비 검정, Wald 검정, Rao(Score) 검정 모두 표본 크기가 충분히 크면 Pearson 카이제곱 검정통계량과 동일한 결론에 도달한다.  
    > 즉,  
    > $$
    > \sum_{i=1}^k \frac{(X_i - n p_{0i})^2}{n p_{0i}} \ge \chi^2_\alpha(k-1)
    > $$
    > 이면 $H_0$를 기각한다.

### 일반적인 귀무가설의 최대가능도비 검정
예7.2.1에서 처럼, 모수는 여러개 이지만 귀무가설은 모수 중 일부만 관여하는 경우가 있다. 이러첨 다차원 모수 중 일부에 관한 귀무가설을 검증하는 경우에도 최대가능도비 검정통계량의 근사가 다음처럼 가능하다.  

정리 6.4.4의 조건 $R_0$ ~ $R_7$이 만족되고,
모수 $\theta = (\xi^t, \eta^t)^t \in \Omega \subset \mathbb{R}^k$에서, 일부 성분 $\xi \in \mathbb{R}^{k_1}$에 대해
$$
H_0: \xi = \xi_0
$$
와 같은 형태의 귀무가설을 고려한다. 이때 귀무가설 하의 모수공간은
$$
\Omega_0 = \{ (\xi_0^t, \eta^t)^t \in \Omega : \eta \in \mathbb{R}^{k_0} \}, \qquad k_0 = k - k_1
$$
로서 $k_0$차원의 열린집합.

> **요약:**  
> 귀무가설은 전체 $k$차원 모수공간에서 $k_0$차원을 고정(제약)하는 역할을 한다.  
> 즉, 귀무가설 하에서는 $k_0$방향으로는 움직일 수 없고, 남은 $k - k_0$개의 자유방향만 변화가 가능하다.  
> 검정통계량은 이 $k - k_0$개의 자유방향에서의 제곱합(거리)을 측정하며,  
> 표본이 충분히 크면 그 분포가 $\chi^2(k - k_0)$로 근사된다.
> 제곱합이므로 $\chi^2$분포를 따른다.
### 정리 7.3.2 (일반 귀무가설의 최대가능도비 검정)
전체 모수공간과 귀무가설 하 모수공간에서의 최대가능도 추정량을 각각 $\hat\theta_n^\Omega$, $\hat\theta_n^{\Omega_0}$라 하면,
**(1) $H_0:\xi=\xi_0$이 사실일 때**  
$$
2\{l(\hat\theta_n^\Omega) - l(\hat\theta_n^{\Omega_0})\} \xrightarrow{d} \chi^2(k - k_0)
$$
가 성립한다.

따라서 표본 크기가 충분히 클 때, 유의수준 $\alpha$에서의 기각역은
$$
2\{l(\hat\theta_n^\Omega) - l(\hat\theta_n^{\Omega_0})\} \ge \chi^2_\alpha(k - k_0)
$$

**(2) $H_0:\xi=\xi_0$이 사실이고 $\theta_0 = (\xi_0^t, \eta_0^t)^t \in \Omega_0$일때**  
$$
2\{l(\hat\theta_n^\Omega) - l(\hat\theta_n^{\Omega_0})\} = n
(\hat\theta_n^{\Omega_0} - \theta_0)^T I(\theta_0) (\hat\theta_n^{\Omega_0} - \theta_0) + r_n
$$
여기서 $r_n \xrightarrow{P_{\theta_0}} 0$이므로, 표본 크기가 충분히 크면 근사적으로
$$
n(\hat\theta_n^{\Omega_0} - \theta_0)^T I(\theta_0) (\hat\theta_n^{\Omega_0} - \theta_0) \ge \chi^2_\alpha(k - k_0)
$$
인 경우 $H_0$를 기각한다.

**(3) $H_0:\xi=\xi_0$이 사실이고 $\theta_0 = (\xi_0^t, \eta_0^t)^t \in \Omega_0$일때**  
평균 로그가능도함수 $\bar l(\theta)$의 점수벡터를
$$
\bar l'(\theta) = \begin{pmatrix} \bar l'_\xi(\theta) \\ \bar l'_\eta(\theta) \end{pmatrix}
$$
로, 정보행렬 $I(\theta)$를
$$
I(\theta) = \begin{pmatrix}
I_{\xi\xi}(\theta) & I_{\xi\eta}(\theta) \\
I_{\eta\xi}(\theta) & I_{\eta\eta}(\theta)
\end{pmatrix}
$$
와 같이 블록 행렬로 분해한다.

Rao(Score) 검정통계량은 $H_0$가 사실이고 $\theta_0 = (\xi_0^t, \eta_0^t)^t \in \Omega_0$일 때,
$$
S = n\, \bar l'_\xi(\theta_0)^T \left[ I_{\xi\xi}(\theta_0) - I_{\xi\eta}(\theta_0) I_{\eta\eta}(\theta_0)^{-1} I_{\eta\xi}(\theta_0) \right]^{-1} \bar l'_\xi(\theta_0)
$$
정규성 조건이 만족되면, $H_0$ 하에서
$$
S \xrightarrow{d} \chi^2(k_1)
$$
이므로, 유의수준 $\alpha$에서의 기각역은
$$
S \ge \chi^2_\alpha(k_1)
$$
즉, Rao(Score) 검정은 일부 모수에 대한 귀무가설을 점수함수와 정보행렬의 블록 구조를 이용해 검정하며, 표본이 충분히 크면 자유도 $k_1$인 카이제곱 분포로 근사된다.

(증명스케치는 교과서 참고)

**정리 7.3.2 응용**  
> 한편, 귀무가설이 미분가능한 함수 $g_1(\theta), \dots, g_{k_1}(\theta)$로
> $$
> H_0: g_1(\theta) = 0,\, \dots,\, g_{k_1}(\theta) = 0 \qquad (k_1 = k - k_0)
> $$
> 와 같이 주어지는 경우, 모수변환을 통해
> $$
> \xi_1 = g_1(\theta),\, \dots,\, \xi_{k_1} = g_{k_1}(\theta),\quad
> \eta_1 = g_{k_1+1}(\theta),\, \dots,\, \eta_{k_0} = g_k(\theta)
> $$
> ($k_1 + k_0 = k$)로 나타낼 수 있다. 이때 정리 7.3.2의 귀무가설 $H_0: \xi = 0$ 형태로 변환하여 동일하게 최대가능도비 검정의 근사 결과를 적용할 수 있다.  
> 즉, 이러한 형태의 귀무가설에 대해서도 정리 7.3.2의 점근적(근사) 검정이 가능하다.

#### 예 7.3.3 분할표에서의 독립성 검정
> **참고: 이원분류**  
> - **이원분류(two-way classification)**: 두 범주형 변수(행 $r$개, 열 $c$개)에 따라 관측값을 분류
> - **분할표(contingency table)**: 이원분류 결과를 $r \times c$ 표로 정리, 각 셀 $(i,j)$에 관측도수 $X_{ij}$ 기록
> - **$X_{ij}$의 의미**: 행 $i$ 범주와 열 $j$ 범주에 모두 속하는 관측값의 개수
> 예를 들어, 두 가지 약물(A, B)의 효과(효과 있음/없음)를 조사한 결과가 다음과 같이 분할표로 정리되었다고 하자.
> 
> |           | 효과 있음 | 효과 없음 | 합계 |
> |-----------|-----------|-----------|------|
> | 약물 A    |    30     |    20     |  50  |
> | 약물 B    |    10     |    40     |  50  |
> | **합계**  |    40     |    60     | 100  |
> 
> - $X_{11}=30$, $X_{12}=20$, $X_{21}=10$, $X_{22}=40$, $n=100$
> - 기대도수 $E_{ij} = \dfrac{(\text{행합}) \times (\text{열합})}{n}$
> 
> 계산:
> - $E_{11} = \dfrac{50 \times 40}{100} = 20$
> - $E_{12} = \dfrac{50 \times 60}{100} = 30$
> - $E_{21} = \dfrac{50 \times 40}{100} = 20$
> - $E_{22} = \dfrac{50 \times 60}{100} = 30$
> 
> 피어슨 카이제곱 통계량:
> $$
> \chi^2 = \frac{(30-20)^2}{20} + \frac{(20-30)^2}{30} + \frac{(10-20)^2}{20} + \frac{(40-30)^2}{30}
> = \frac{100}{20} + \frac{100}{30} + \frac{100}{20} + \frac{100}{30}
> = 5 + 3.33 + 5 + 3.33 = 16.66
> $$
> 
> 자유도 $(2-1)(2-1)=1$, 유의수준 $\alpha=0.05$에서 임계값 $\chi^2_{0.05}(1) \approx 3.84$.
> 
> 따라서 $16.66 > 3.84$이므로, 두 약물의 효과와 약물 종류는 독립이 아니라고 결론내린다.

$$
X = (X_{ij}) \sim \mathrm{Multi}(n, (p_{ij})), \qquad \sum_{i=1}^r \sum_{j=1}^c p_{ij} = 1,\quad p_{ij} > 0
$$

**귀무가설**: 두 분류가 서로 독립이다.
$$
H_0: p_{ij} = p_{i\cdot} p_{\cdot j}
$$
- $p_{i\cdot} = \sum_{j=1}^c p_{ij}$ (행합 확률)
- $p_{\cdot j} = \sum_{i=1}^r p_{ij}$ (열합 확률)

**대립가설**: $H_1:$ $H_0$가 아니다 (즉, 독립이 아님).

**풀이**
1. **로그가능도함수**
    $$
    l(p) = \sum_{i=1}^r \sum_{j=1}^c X_{ij} \log p_{ij}
    $$
2. **최대가능도추정값(MLE)**
    - **전체 모수공간**: $\hat p_{ij} = \dfrac{X_{ij}}{n}$
      - 왜 이렇게 되는지는 예6.3.6나 예7.3.2 참고
    - **귀무가설 $H_0$ 하**: $p_{ij} = p_{i\cdot} p_{\cdot j}$ 형태이므로,
      $$
      \hat p_{i\cdot} = \frac{X_{i\cdot}}{n},\qquad \hat p_{\cdot j} = \frac{X_{\cdot j}}{n}
      $$
      $$
      \hat p_{ij}^{(0)} = \hat p_{i\cdot} \hat p_{\cdot j} = \frac{X_{i\cdot} X_{\cdot j}}{n^2}
      $$

3. **최대가능도비 검정통계량**
    $$
    2\{l(\hat p) - l(\hat p^{(0)})\}
    = 2 \sum_{i=1}^r \sum_{j=1}^c X_{ij} \log \left( \frac{\hat p_{ij}}{\hat p_{ij}^{(0)}} \right )
    = 2 \sum_{i=1}^r \sum_{j=1}^c X_{ij} \log \left( \frac{X_{ij} \cdot n}{X_{i\cdot} X_{\cdot j}} \right )
    $$

    표본이 충분히 크면, 피어슨 카이제곱 통계량으로 근사된다:
    $$
    \chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(X_{ij} - E_{ij})^2}{E_{ij}}
    $$
    여기서 $E_{ij} = n \hat p_{ij}^{(0)} = \dfrac{X_{i\cdot} X_{\cdot j}}{n}$ (기대도수).

4. **자유도**
    - 전체 모수공간 자유도: $rc - 1$
    - 귀무가설 하 모수공간 자유도: $(r-1) + (c-1)$
    - 차이: $(r-1)(c-1)$

5. **기각역 (유의수준 $\alpha$)**
    $$
    \chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(X_{ij} - E_{ij})^2}{E_{ij}} \ge \chi^2_\alpha((r-1)(c-1))
    $$
    이면 $H_0$를 기각한다.

**정리:**  
분할표의 독립성 검정에서, 표본이 충분히 크면 피어슨 카이제곱 검정통계량을 사용하여 유의수준 $\alpha$에서 자유도 $(r-1)(c-1)$의 카이제곱 분포 임계값과 비교한다.  
즉,
$$
\sum_{i=1}^r \sum_{j=1}^c \frac{(X_{ij} - E_{ij})^2}{E_{ij}} \ge \chi^2_\alpha((r-1)(c-1))
$$
이면 두 분류는 독립이 아니라고 결론내린다.

> 흔히 Wald, Rao 검정통계량을 이용한 기각역을
> $$
> \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - \hat E_{ij}^{(0)})^2}{\hat E_{ij}^{(0)}} \ge \chi^2_\alpha((r-1)(c-1))
> $$
> 와 같이 나타내며, 여기서 $O_{ij} = X_{ij}$는 관측도수, $\hat E_{ij}^{(0)}$는 귀무가설 하에서의 기대도수를 의미한다. 이러한 검정을 분할표에서의 **카이제곱 검정**(chi-squared test)이라고 한다.

### 일반적인 경우의 최대가능도를 이용한 추정, 검정
서로 독립인 관측값 $X_1, \dots, X_n$에 대해, 적절한 정규성 조건이 만족되면 정리 7.3.1, 7.3.2와 유사하게 최대가능도비 검정통계량의 극한분포를 얻을 수 있다.

이 경우에도 가능도 및 로그가능도 함수의 정의는 동일하며, 정보량 행렬은 다음과 같이 정의된다.
$$
I_n(\theta) = E_\theta\left[ -\frac{\partial^2}{\partial\theta^2} \sum_{i=1}^n \log f(X_i;\theta) \right], \qquad \bar{I}_n(\theta) = \frac{I_n(\theta)}{n}
$$

귀무가설 $H_0: \theta = \theta_0$의 경우, 표본 크기가 충분히 크면 Wald 또는 Rao(Score) 검정통계량을 다음과 같이 쓸 수 있다.
- **Wald 검정통계량**:
    $$
    W_n(\theta_0) = n\, (\hat\theta_n - \theta_0)^T\, \bar{I}_n(\theta_0)\, (\hat\theta_n - \theta_0)
    $$
- **Rao(Score) 검정통계량**:
    $$
    R_n(\theta_0) = n\, \bar{l}'(\theta_0)^T\, [\bar{I}_n(\theta_0)]^{-1}\, \bar{l}'(\theta_0)
    $$
    여기서 $\hat\theta_n$은 전체 모수공간에서의 최대가능도추정량, $\bar{l}'(\theta_0)$는 평균 점수함수이다.

일반적인 귀무가설(예: 일부 모수만 고정된 경우)에서는, 미지의 $\theta_0$ 대신 귀무가설 하에서의 최대가능도추정값 $\hat\theta_n^{(0)}$를 사용하여 검정통계량을 구성한다. 이때도 정리 7.3.2와 같이 근사적으로 $\chi^2$ 분포를 이용한 검정이 가능하다.

#### 예 7.3.4 분할표에서의 동일성 검정
$r$개의 집단에서 각각 독립적인 다항분포 랜덤벡터 $X_i = (X_{i1}, \dots, X_{ic})^t \sim \mathrm{Multi}(n_i, p_i)$ $(i=1,\dots,r)$를 관측한다고 하자. 각 집단의 확률벡터는 $p_i = (p_{i1}, \dots, p_{ic})^t$이며, $\sum_{j=1}^c p_{ij} = 1$, $p_{ij} > 0$이다. 전체 표본 크기는 $n = n_1 + \cdots + n_r$이다.

(TODO: 교재 풀이 나중에 참고)

**귀무가설**: $H_0: p_1 = \cdots = p_r$ (모든 집단의 분포가 동일)  
**대립가설**: $H_1:$ $H_0$가 아님

**1. 로그가능도함수**  
관측값 $x_{ij}$에 대해, 로그가능도함수(상수 제외)는  
$$
l(p) = \sum_{i=1}^r \sum_{j=1}^c x_{ij} \log p_{ij}
$$

**2. 최대가능도추정값**  

- **(1) 전체 모수공간**: 각 집단별로 $\sum_{j=1}^c p_{ij} = 1$ 제약 하에서  
    $$
    \hat p_{ij} = \frac{x_{ij}}{n_i}
    $$
- **(2) 귀무가설 $H_0$ 하**: 모든 집단이 동일한 확률벡터 $p$를 가짐  
    $$
    \hat p_j^{(0)} = \frac{x_{\cdot j}}{n}, \qquad x_{\cdot j} = \sum_{i=1}^r x_{ij}
    $$

**3. 최대가능도비 검정통계량**  
$$
2\{l(\hat p) - l(\hat p^{(0)})\}
= 2 \sum_{i=1}^r \sum_{j=1}^c x_{ij} \log \left( \frac{\hat p_{ij}}{\hat p_j^{(0)}} \right )
= 2 \sum_{i=1}^r \sum_{j=1}^c x_{ij} \log \left( \frac{x_{ij}}{n_i \hat p_j^{(0)}} \right )
$$

**4. 카이제곱 근사 (피어슨 형태)**  
표본이 충분히 크면 $\log(1+u) \approx u - \frac{1}{2}u^2$ 근사로 인해  
$$
2\{l(\hat p) - l(\hat p^{(0)})\} \approx \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij}^{(0)})^2}{E_{ij}^{(0)}}
$$
여기서  
- $O_{ij} = x_{ij}$ (관측도수)
- $E_{ij}^{(0)} = n_i \hat p_j^{(0)} = n_i \frac{x_{\cdot j}}{n}$ (기대도수)

**5. 자유도와 기각역**  
- 전체 모수공간 자유도: $k = r(c-1)$
- 귀무가설 하 자유도: $k_0 = c-1$
- 차이: $(r-1)(c-1)$

따라서,  
- 검정통계량 $2\{l(\hat p) - l(\hat p^{(0)})\}$는 $H_0$ 하에서 자유도 $(r-1)(c-1)$인 $\chi^2$ 분포로 근사
- 유의수준 $\alpha$에서의 기각역:
    $$
    2\{l(\hat p) - l(\hat p^{(0)})\} \ge \chi^2_\alpha((r-1)(c-1))
    $$
    또는
    $$
    \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij}^{(0)})^2}{E_{ij}^{(0)}} \ge \chi^2_\alpha((r-1)(c-1))
    $$
