# 1. 리만적분 *(Riemann Integral)*
## (1) 리만적분의 정의 *(Definition of the Riemann Integral)*
### 분할과 세분 *(Partition, Refinement)*
* $[a,b]$가 **유계**인 **폐구간**이고
  $$
  \varphi=\{x_0,x_1,\dots,x_n\},\quad a=x_0<x_1<\cdots<x_n=b
  $$
  이면 $\varphi$를 $[a,b]$의 **분할** *(partition)* 이라 한다.

* 두 분할 $\varphi,\varphi^*$에 대해
  $\varphi\subset\varphi^*$이면 $\varphi^*$를 $\varphi$의 **세분** *(refinement)* 이라 한다.
  - 세분: 더 쪼개기

### 상합과 하합 *(Upper Sum, Lower Sum)*
$f:[a,b]\to\mathbb R$가 유계일 때, 분할 $\varphi=\{x_i\}$에 대해
$$
M_i=\sup\{f(x) | x_{i-1}\le x\le x_i\},\quad
m_i=\inf\{f(x) | x_{i-1}\le x\le x_i\}
$$
로 두면,

* **상합** *(Upper sum)*:
  $$
  U(\varphi,f)=\sum_{i=1}^n M_i\Delta x_i
  $$

* **하합** *(Lower sum)*:
  $$
  L(\varphi,f)=\sum_{i=1}^n m_i\Delta x_i
  $$

### 상적분과 하적분 *(Upper Integral, Lower Integral)*
$f$가 $[a,b]$에서 유계이면
$$
\overline{\int_a^b} f(x)dx = \overline{\int_a^b} f
=\inf U(\varphi,f),\quad
\underline{\int_a^b} f(x)dx = \underline{\int_a^b} f
=\sup L(\varphi,f)
$$
를 각각 **상적분** *(upper integral)*, **하적분** *(lower integral)* 이라 한다.
- 잘개쪼갤수록 상합은 작이지고, 하합은 커질거다.

### Thm. : Upper–Lower Sum Inequalities
다음 명제들이 성립한다.
- $\varphi^*$가 $[a, b]$의 분할 $\varphi$의 세분이면 
$$
L(\varphi, f) \le L(\varphi^*, f) \le U(\varphi^*, f) \le U(\varphi, f)
$$

- $[a, b]$의 임의의 두 분할 $\varphi_1, \varphi_2$에 대하여 $L(\varphi_1, f) \le U(\varphi_2, f)$이다.

- f가 $[a, b]$에서 유계이면, 
$$
\underline{\int_a^b} f \le \overline{\int_a^b} f
$$

#### 증명
**1. 세분에 대한 부등식**  
$\varphi^*$가 $\varphi$의 세분이므로, 먼저 $\varphi^*$가 $\varphi$에 점 하나만 추가된 경우를 고려하자.

$\varphi = \{x_0, x_1, \dots, x_n\}$이고 $\varphi^* = \varphi \cup \{x^*\}$이며, $x_i < x^* < x_{i+1}$이라 하자.

$[x_i, x_{i+1}]$에서:
- $M_i = \sup\{f(x) | x_i \le x \le x_{i+1}\}$
- $M_i' = \sup\{f(x) | x_i \le x \le x^*\}$,  
 $M_i'' = \sup\{f(x) | x^* \le x \le x_{i+1}\}$

그러면 $M_i' \le M_i$, $M_i'' \le M_i$이므로
$$
M_i'(x^* - x_i) + M_i''(x_{i+1} - x^*) \\ 
\le M_i(x_{i+1} - x^*) + M_i(x^* - x_{i})= M_i(x_{i+1} - x_i)
$$
따라서 $U(\varphi^*, f) \le U(\varphi, f)$.

유사하게 $m_i \le m_i'$, $m_i \le m_i''$이므로
$$
m_i(x_{i+1} - x_i) \le m_i'(x^* - x_i) + m_i''(x_{i+1} - x^*)
$$
따라서 $L(\varphi, f) \le L(\varphi^*, f)$.

귀납적으로 $L(\varphi, f) \le L(\varphi^*, f) \le U(\varphi^*, f) \le U(\varphi, f)$.

**2. 임의의 두 분할에 대한 부등식**  
$\varphi_1$, $\varphi_2$를 $[a,b]$의 임의의 두 분할이라 하고, $\varphi = \varphi_1 \cup \varphi_2$라 하자.

그러면 $\varphi$는 $\varphi_1$과 $\varphi_2$ 모두의 세분이므로:
$$
L(\varphi_1, f) \le L(\varphi, f) \le U(\varphi, f) \le U(\varphi_2, f)
$$

**3. 하적분과 상적분의 부등식**  
앞의 결과에서 임의의 분할 $\varphi_1, \varphi_2$에 대해 $L(\varphi_1, f) \le U(\varphi_2, f)$이므로:

$\varphi_2$를 고정하면, 모든 $\varphi_1$에 대해 $L(\varphi_1, f) \le U(\varphi_2, f)$

따라서 $\sup_{\varphi_1} L(\varphi_1, f) \le U(\varphi_2, f)$

이제 $\varphi_2$를 변화시키면:
$$
\underline{\int_a^b} f = \sup_{\varphi_1} L(\varphi_1, f) \le \inf_{\varphi_2} U(\varphi_2, f) = \overline{\int_a^b} f
$$

### 리만적분가능성 *(Riemann Integrability)*
$f$가 $[a,b]$에서 유계일 때
$$
\overline{\int_a^b} f=\underline{\int_a^b} f
$$
이면 $f$는 $[a,b]$에서 **리만적분가능** *(Riemann integrable)* 하다고 하고,
$$
\int_a^b f(x)dx = \int_a^b f
$$
로 그 값을 쓴다.  
또한, $[a, b]$에서 유계인 리만적분가능한 함수 f들의 집합은 $\mathcal{R}[a, b]$로 나타낸다 ($f \in \mathcal{R}[a, b]$)

## (2) 주요 정리 *(Main Theorems on the Riemann Integral)*
### Thm.1 판별법 *(Riemann Integrability Criterion)*
$f$가 $[a,b]$에서 유계이면
$$
f\in\mathcal R[a,b]
\iff
\forall\varepsilon>0,\ \exists\varphi\text{ s.t. }
U(\varphi,f)-L(\varphi,f)<\varepsilon
$$

#### 증명
**($\Rightarrow$)** $f \in \mathcal{R}[a,b]$라 하자. 그러면 정의에 의해
$$
\overline{\int_a^b} f = \underline{\int_a^b} f
$$

임의의 $\varepsilon > 0$에 대하여, 상적분과 하적분의 정의로부터:
- $\inf U(\varphi, f) = \overline{\int_a^b} f$이므로, 어떤 분할 $\varphi_1$이 존재하여
    $$
    U(\varphi_1, f) < \overline{\int_a^b} f + \frac{\varepsilon}{2}
    $$

- $\sup L(\varphi, f) = \underline{\int_a^b} f$이므로, 어떤 분할 $\varphi_2$가 존재하여
    $$
    L(\varphi_2, f) > \underline{\int_a^b} f - \frac{\varepsilon}{2}
    $$

$\varphi = \varphi_1 \cup \varphi_2$라 하면, $\varphi$는 $\varphi_1$과 $\varphi_2$ 모두의 세분이므로:
$$
U(\varphi, f) \le U(\varphi_1, f) < \overline{\int_a^b} f + \frac{\varepsilon}{2}
$$
$$
L(\varphi, f) \ge L(\varphi_2, f) > \underline{\int_a^b} f - \frac{\varepsilon}{2}
$$

따라서
$$
U(\varphi, f) - L(\varphi, f) < \left(\overline{\int_a^b} f + \frac{\varepsilon}{2}\right) - \left(\underline{\int_a^b} f - \frac{\varepsilon}{2}\right) = \varepsilon
$$

**($\Leftarrow$)** 임의의 $\varepsilon > 0$에 대하여 $U(\varphi, f) - L(\varphi, f) < \varepsilon$인 분할 $\varphi$가 존재한다고 하자.

상적분과 하적분의 정의에 의해:
$$
\underline{\int_a^b} f = \sup L(\varphi, f) \le L(\varphi, f)
$$
$$
\overline{\int_a^b} f = \inf U(\varphi, f) \le U(\varphi, f)
$$

따라서
$$
0 \le \overline{\int_a^b} f - \underline{\int_a^b} f \le U(\varphi, f) - L(\varphi, f) < \varepsilon
$$

$\varepsilon$는 임의의 양수이므로
$$
\overline{\int_a^b} f = \underline{\int_a^b} f
$$

즉, $f \in \mathcal{R}[a,b]$이다.

### Thm.2 연속함수의 적분가능성 *(Continuity implies Riemann Integrability)*
$f$가 $[a,b]$에서 연속이면
$$
f\in\mathcal R[a,b].
$$

* 불연속인데 리만적분 가능한 경우도 있긴 함! 

#### 증명 
$f$가 $[a,b]$에서 연속이므로, 유계 폐구간에서 연속인 함수는 **균등연속** *(uniformly continuous)* 이다. (by 하이네–칸토어 정리(Heine–Cantor theorem))

즉, 임의의 $\varepsilon > 0$에 대하여 $\delta > 0$이 존재하여
$$
|x - y| < \delta \implies |f(x) - f(y)| < \frac{\varepsilon}{b-a}
$$

이제 $[a,b]$의 분할 $\varphi = \{x_0, x_1, \dots, x_n\}$을 $\|\varphi\| = \max_i \Delta x_i < \delta$가 되도록 선택하자.
- $\max_i \Delta x_i$는 분할에서 가장 긴 소구간의 길이
- $\|\varphi\|$: 분할의 크기 또는 메시(mesh) 라고 부른다.

각 소구간 $[x_{i-1}, x_i]$에서 $f$는 연속이므로 **최대·최소 정리**에 의해 최댓값과 최솟값을 갖는다. 즉,
$$
M_i = \max_{x \in [x_{i-1}, x_i]} f(x), \quad m_i = \min_{x \in [x_{i-1}, x_i]} f(x)
$$

$\Delta x_i < \delta$이므로 균등연속성에 의해
$$
M_i - m_i < \frac{\varepsilon}{b-a}
$$

따라서
$$
U(\varphi, f) - L(\varphi, f) = \sum_{i=1}^n (M_i - m_i)\Delta x_i < \frac{\varepsilon}{b-a} \sum_{i=1}^n \Delta x_i = \frac{\varepsilon}{b-a} \cdot (b-a) = \varepsilon
$$

**Thm.1 판별법**에 의해 $f \in \mathcal{R}[a,b]$이다.

### Thm.3 적분의 평균값 정리 *(Mean Value Theorem for Integrals)*
$f$가 $[a,b]$에서 연속이면
$$
\exists c\in(a,b)\text{ s.t. }
\int_a^b f(x),dx=f(c)(b-a).
$$

#### 증명
$f$가 $[a,b]$에서 연속이므로, **Thm.2**에 의해 $f \in \mathcal{R}[a,b]$이다.

연속함수의 **최대·최소 정리**에 의해
$$
m = \min_{x \in [a,b]} f(x), \quad M = \max_{x \in [a,b]} f(x)
$$
가 존재한다.

따라서 모든 $x \in [a,b]$에 대해
$$
m \le f(x) \le M
$$

적분의 단조성에 의해
$$
m(b-a) \le \int_a^b f(x)\,dx \le M(b-a)
$$

양변을 $(b-a)$로 나누면
$$
m \le \frac{1}{b-a}\int_a^b f(x)\,dx \le M
$$

$f$는 $[a,b]$에서 연속이므로 **중간값 정리**에 의해, 어떤 $c \in [a,b]$가 존재하여
$$
f(c) = \frac{1}{b-a}\int_a^b f(x)\,dx
$$

따라서
$$
\int_a^b f(x)\,dx = f(c)(b-a)
$$

$f$가 상수함수가 아니면 $m < M$이므로 $c \in (a,b)$이다.

## (3) 리만적분의 연산 *(Properties of the Riemann Integral)*
$f,g\in\mathcal R[a,b]$이면 다음이 성립한다.

### 연산1. 선형성 *(Linearity)*
  $$
  \int_a^b (f\pm g)=\int_a^b f\pm\int_a^b g
  $$

  - 증명  

  **($\alpha f + \beta g$의 적분가능성)**  
  임의의 $\varepsilon > 0$에 대하여, $f, g \in \mathcal{R}[a,b]$이므로 **Thm.1 판별법**에 의해 분할 $\varphi_1, \varphi_2$가 존재하여
  $$
  U(\varphi_1, f) - L(\varphi_1, f) < \frac{\varepsilon}{2|\alpha|}, \quad U(\varphi_2, g) - L(\varphi_2, g) < \frac{\varepsilon}{2|\beta|}
  $$
  ($\alpha, \beta \neq 0$인 경우. 0이면 자명)

  $\varphi = \varphi_1 \cup \varphi_2$라 하면, 각 소구간에서:
  - $\alpha > 0$이면 $M_i(\alpha f) = \alpha M_i(f)$, $m_i(\alpha f) = \alpha m_i(f)$
  - $\alpha < 0$이면 $M_i(\alpha f) = \alpha m_i(f)$, $m_i(\alpha f) = \alpha M_i(f)$

  따라서
  $$
  U(\varphi, \alpha f + \beta g) - L(\varphi, \alpha f + \beta g) \\ \le |\alpha|(U(\varphi, f) - L(\varphi, f)) + |\beta|(U(\varphi, g) - L(\varphi, g)) \\
  < \varepsilon
  $$

  고로 $\alpha f + \beta g \in \mathcal{R}[a,b]$.

  **($\int(f+g) = \int f + \int g$)**  
  분할 $\varphi$에 대해 각 소구간에서
  $$
  m_i(f) + m_i(g) \le f(x) + g(x) \le M_i(f) + M_i(g)
  $$

  따라서
  $$
  m_i(f) + m_i(g) \le m_i(f+g) \le M_i(f+g) \le M_i(f) + M_i(g)
  $$

  이를 합하면
  $$
  L(\varphi, f) + L(\varphi, g) \le L(\varphi, f+g) \le U(\varphi, f+g) \le U(\varphi, f) + U(\varphi, g)
  $$

  극한을 취하면
  $$
  \int_a^b f + \int_a^b g = \int_a^b (f+g)
  $$

### 연산2. 구간의 가법성 *(Additivity over intervals)*
  $$
  f \in \mathcal{R}[a,b] \Leftrightarrow
  \forall c\in(a,b),\int_a^b f=\int_a^c f+\int_c^b f
  $$

  - 증명  

  **Step1: ($f \in \mathcal{R}[a,b] \Rightarrow f \in \mathcal{R}[a,c], f \in \mathcal{R}[c,b]$)**  
  $f \in \mathcal{R}[a,b]$이므로, 임의의 $\varepsilon > 0$에 대하여 **Thm.1 판별법**에 의해 분할 $\varphi$가 존재하여
  $$
  U(\varphi, f) - L(\varphi, f) < \varepsilon
  $$

  $c \in (a,b)$에 대해 $\varphi^* = \varphi \cup \{c\}$라 하면, $\varphi^*$는 $\varphi$의 세분이므로
  $$
  U(\varphi^*, f) - L(\varphi^*, f) \le U(\varphi, f) - L(\varphi, f) < \varepsilon
  $$

  $\varphi^* = \{x_0, x_1, \dots, x_k = c, x_{k+1}, \dots, x_n\}$이라 하자.

  $\varphi_1 = \{x_0, x_1, \dots, x_k\}$를 $[a,c]$의 분할, $\varphi_2 = \{x_k, x_{k+1}, \dots, x_n\}$를 $[c,b]$의 분할이라 하면
  $$
  U(\varphi^*, f) = U(\varphi_1, f) + U(\varphi_2, f)
  $$
  $$
  L(\varphi^*, f) = L(\varphi_1, f) + L(\varphi_2, f)
  $$

  따라서
  $$
  (U(\varphi_1, f) - L(\varphi_1, f)) + (U(\varphi_2, f) - L(\varphi_2, f)) < \varepsilon
  $$

  각 항이 모두 비음수이므로
  $$
  U(\varphi_1, f) - L(\varphi_1, f) < \varepsilon, \quad U(\varphi_2, f) - L(\varphi_2, f) < \varepsilon
  $$

  **Thm.1**에 의해 $f \in \mathcal{R}[a,c]$, $f \in \mathcal{R}[c,b]$이다.

  **Step2: ($\int_a^b f = \int_a^c f + \int_c^b f$)**  
  $f \in \mathcal{R}[a,c]$, $f \in \mathcal{R}[c,b]$이므로, 임의의 $\varepsilon > 0$에 대하여 분할 $\varphi_1$ (of $[a,c]$), $\varphi_2$ (of $[c,b]$)가 존재하여
  $$
  U(\varphi_1, f) - L(\varphi_1, f) < \frac{\varepsilon}{2}, \quad U(\varphi_2, f) - L(\varphi_2, f) < \frac{\varepsilon}{2}
  $$

  $\varphi = \varphi_1 \cup \varphi_2$는 $[a,b]$의 분할이고
  $$
  U(\varphi, f) - L(\varphi, f) = (U(\varphi_1, f) + U(\varphi_2, f)) - (L(\varphi_1, f) + L(\varphi_2, f)) < \varepsilon
  $$

  따라서 $f \in \mathcal{R}[a,b]$.

  Step3: 또한
  $$
  L(\varphi_1, f) + L(\varphi_2, f) = L(\varphi, f) \le \int_a^b f \le U(\varphi, f) = U(\varphi_1, f) + U(\varphi_2, f)
  $$

  $\varepsilon \to 0$으로 보내면
  $$
  \int_a^b f = \int_a^c f + \int_c^b f
  $$

# 2. 미적분학의 기본정리 *(Fundamental Theorem of Calculus)*
## (1) 제1 기본정리 *(First Fundamental Theorem of Calculus)*
**Def.**  
$f\in\mathcal R[a,b]$일 때 $x\in [a,b]$에 대해  
$$
F(x)=\int_a^x f(t),dt
$$
이 함수 $F$를 $[a,b]$에서 $f$의 부정적분(indefinite integral)이라 한다.

**Thm.**  
$f\in\mathcal R[a,b]$이면,
1. $F$는 $[a,b]$에서 균등연속이다.
2. $f$가$[a,b]$에서 연속이면 $F$는 $(a,b)$에서 미분가능하고, $$\forall x\in [a,b], \ F'(x)=f(x).$$
  - 미분과 적분은 서로 역연산 관계

### 증명
**1. $F$는 $[a,b]$에서 균등연속이다.**

$f \in \mathcal{R}[a,b]$이므로 $f$는 유계이다. 즉, $|f(x)| \le M$ for some $M > 0$ and all $x \in [a,b]$.

임의의 $x, y \in [a,b]$에 대하여 ($x < y$라 가정)
$$
|F(y) - F(x)| = \left|\int_a^y f(t)\,dt - \int_a^x f(t)\,dt\right| = \left|\int_x^y f(t)\,dt\right|
$$

적분의 단조성에 의해
$$
\left|\int_x^y f(t)\,dt\right| \le \int_x^y |f(t)|\,dt \le M(y-x)
$$

따라서 임의의 $\varepsilon > 0$에 대하여 $\delta = \frac{\varepsilon}{M}$로 선택하면
$$
|y - x| < \delta \implies |F(y) - F(x)| \le M|y-x| < M \cdot \frac{\varepsilon}{M} = \varepsilon
$$

이는 $x, y$의 위치에 무관하므로 $F$는 $[a,b]$에서 균등연속이다.

**2. $f$가 $[a,b]$에서 연속이면 $F'(x) = f(x)$**

$x \in [a,b]$를 고정하자. $f$가 $x$에서 연속이므로, 임의의 $\varepsilon > 0$에 대하여 $\delta > 0$이 존재하여
$$
|t - x| < \delta \implies |f(t) - f(x)| < \varepsilon
$$

$0 < |h| < \delta$이고 $x + h \in [a,b]$인 $h$에 대하여
$$
\frac{F(x+h) - F(x)}{h} = \frac{1}{h}\int_x^{x+h} f(t)\,dt
$$

($h < 0$인 경우 $\int_x^{x+h} = -\int_{x+h}^x$로 처리)

$h > 0$인 경우를 고려하면
$$
\left|\frac{F(x+h) - F(x)}{h} - f(x)\right| = \left|\frac{1}{h}\int_x^{x+h} f(t)\,dt - f(x)\right|
$$
$$
= \left|\frac{1}{h}\int_x^{x+h} (f(t) - f(x))\,dt\right| \le \frac{1}{h}\int_x^{x+h} |f(t) - f(x)|\,dt
$$

$|t - x| \le h < \delta$이므로 $|f(t) - f(x)| < \varepsilon$. 따라서
$$
\left|\frac{F(x+h) - F(x)}{h} - f(x)\right| < \frac{1}{h} \cdot \varepsilon h = \varepsilon
$$

$h < 0$인 경우도 유사하게 증명된다.

따라서 $\lim_{h \to 0} \frac{F(x+h) - F(x)}{h} = f(x)$, 즉 $F'(x) = f(x)$.

## (2) 제2 기본정리 *(Second Fundamental Theorem of Calculus)*
**Def.**  
역도함수 (Antiderivative)**  
함수 $f$가 구간 $D$에서 정의될 때, $F'(x) = f(x)$ for all $x \in D$를 만족하는 함수 $F$를 $f$의 **역도함수** 또는 **원시함수** *(antiderivative)* 라 한다.

**Thm.**  
$f\in\mathcal R[a,b]$이고 $F: [a, b] \to \mathbb{R}$가 $[a, b]$에서 연속이고 $(a, b)$에서 미분가능하다고 하자.  
$F$가 $f$의 **원시함수** *(antiderivative)* 이면
$$
\int_a^b f(x)dx=F(b)-F(a)
$$

### 증명
**$f$의 원시함수가 존재한다고 가정**  
$F$를 $f$의 원시함수, 즉 $F' = f$라 하자.

**Step 1: 분할을 이용한 근사**  
$[a,b]$의 분할 $\varphi = \{x_0, x_1, \dots, x_n\}$ (단, $a = x_0 < x_1 < \cdots < x_n = b$)을 택하자.

평균값 정리에 의해 각 소구간 $[x_{i-1}, x_i]$에서 어떤 $c_i \in (x_{i-1}, x_i)$가 존재하여
$$
F(x_i) - F(x_{i-1}) = F'(c_i)(x_i - x_{i-1}) = f(c_i)\Delta x_i
$$

**Step 2: 텔레스코핑 합**  
(텔레스코핑 합 (telescoping sum): 합을 전개하면 중간항들이 서로 소거되어 처음 항과 마지막 항만 남는 합)

양변을 $i = 1$부터 $n$까지 더하면
$$
\sum_{i=1}^n (F(x_i) - F(x_{i-1})) = \sum_{i=1}^n f(c_i)\Delta x_i
$$

좌변은 텔레스코핑 합이므로
$$
F(x_n) - F(x_0) = F(b) - F(a)
$$

**Step 3: 리만합으로의 수렴**  
각 소구간에서 $m_i \le f(c_i) \le M_i$이므로
$$
L(\varphi, f) \le \sum_{i=1}^n f(c_i)\Delta x_i \le U(\varphi, f)
$$

$f \in \mathcal{R}[a,b]$이므로 $\|\varphi\| \to 0$일 때
$$
L(\varphi, f) \to \int_a^b f(x)\,dx, \quad U(\varphi, f) \to \int_a^b f(x)\,dx
$$

샌드위치 정리에 의해
$$
\sum_{i=1}^n f(c_i)\Delta x_i \to \int_a^b f(x)\,dx
$$

**Step 4: 결론**  
Step 2와 Step 3을 결합하면
$$
\int_a^b f(x)\,dx = F(b) - F(a)
$$

## (3) 따름정리 *(Corollaries)*
### 치환적분 *(Change of Variables / Substitution Rule)*
$g$는 $[a,b]$에서 연속이고,
$(a,b)$에서 미분가능하며,
$g'\in\mathcal R[a,b]$,
$f$는 $g([a,b])$에서 연속이면
$$
\int_a^b f(g(t))g'(t),dt=\int_{g(a)}^{g(b)} f(x),dx.
$$

- 증명  

$F(x) = \int_{g(a)}^{g(x)} f(t)\,dt$라 하자. 

제1 기본정리에 의해 $F$는 미분가능하고 $F'(x) = f(g(x))  
(*Lemma $f, g \in \mathcal{R}[a,b] \Rightarrow fg\in \mathcal{R}[a,b]$ 5장 참고자료 pdf)

연쇄법칙(chain rule)을 적용하면, $G(t) = F(g(t))$에 대해
$$
G'(t) = F'(g(t)) \cdot g'(t) = f(g(t)) \cdot g'(t)
$$

따라서 $G$는 $f(g(t))g'(t)$의 원시함수이다.

제2 기본정리에 의해
$$
\int_a^b f(g(t))g'(t)\,dt = G(b) - G(a) = F(g(b)) - F(g(a))
$$
$$
= \int_{g(a)}^{g(b)} f(x)\,dx - \int_{g(a)}^{g(a)} f(x)\,dx = \int_{g(a)}^{g(b)} f(x)\,dx
$$

**다변수 치환적분 *(Change of Variables in Multiple Integrals)***  
$Y$를 $\mathbb{R}^n$의 열린집합이고, $w: Y \to X$가 일대일 대응이며 미분가능하고 1차 편도함수가 연속이며, 야코비안 행렬식
$$
J_w(y) = \det\left(\frac{\partial w(y)}{\partial y}\right) \neq 0, \quad \forall y \in Y
$$
일 때, $f$가 $w(Y)$에서 적분가능하면
$$
\int_{w(Y)} f(x)\,dx = \int_Y f(w(y))\left|\det\left(\frac{\partial w(y)}{\partial y}\right)\right|\,dy
$$

**역함수를 이용한 계산:**  
치환 $y = u(x)$의 역함수 $w(y) = u^{-1}(y)$로 나타낼 때, **역함수 정리**로부터
$$
\frac{\partial w(y)}{\partial y} = \left(\frac{\partial u(x)}{\partial x}\right)^{-1}, \quad x = w(y)
$$

따라서 야코비안 행렬식은
$$
\det\left(\frac{\partial w(y)}{\partial y}\right) = \frac{1}{\det\left(\frac{\partial u(x)}{\partial x}\right)}
$$

이를 이용하면 치환 공식을
$$
\int_{\text{w.r.t. } y} f(u(x))\,dx = \int_{\text{w.r.t. } x} f(u(x))\left|\det\left(\frac{\partial u(x)}{\partial x}\right)\right|^{-1}\,dy
$$
로 표현할 수 있다.

**예: 극좌표 변환을 이용한 가우스 적분**  
$I = \int_{-\infty}^{\infty} e^{-x^2/2}\,dx$를 구하자.

$$I^2 = \left(\int_{-\infty}^{\infty} e^{-x^2/2}\,dx\right)^2 = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} e^{-(x_1^2+x_2^2)/2}\,dx_1\,dx_2$$

let $x_1 = r\cos\theta, \quad x_2 = r\sin\theta$이면 
$$J = \begin{vmatrix} \frac{\partial x_1}{\partial r} & \frac{\partial x_1}{\partial \theta} \\ \frac{\partial x_2}{\partial r} & \frac{\partial x_2}{\partial \theta} \end{vmatrix} = \begin{vmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{vmatrix} = r$$

따라서 $dx_1\,dx_2 = r\,dr\,d\theta$ 이므로
$$I^2 = \int_0^{2\pi}\int_0^{\infty} e^{-r^2/2} \cdot r\,dr\,d\theta = \int_0^{2\pi}d\theta \int_0^{\infty} re^{-r^2/2}\,dr$$
내부 적분: $u = r^2/2$로 치환하면 $du = r\,dr$

$$\int_0^{\infty} re^{-r^2/2}\,dr = \int_0^{\infty} e^{-u}\,du = [-e^{-u}]_0^{\infty} = 1 \\
\therefore I^2 = \int_0^{2\pi}1\,d\theta = 2\pi$$
따라서
$$I = \int_{-\infty}^{\infty} e^{-x^2/2}\,dx = \sqrt{2\pi}$$



### 부분적분 *(Integration by Parts)*
$f,g$가 $[a,b]$에서 연속, $(a, b)$에서 미분가능하며  
$f', g'\in \mathcal{R}[a,b]$이면
$$
\int_a^b f'g
=f(b)g(b)-f(a)g(a)-\int_a^b fg'
$$

- 증명

Lemma: $f'g, fg'\in \mathcal{R}[a,b]$ 증명은 생략.  

곱의 미분법칙에 의해 $(fg)' = f'g + fg'$

양변을 $[a,b]$에서 적분하면
$$
\int_a^b (fg)'\,dt = \int_a^b f'g\,dt + \int_a^b fg'\,dt
$$

좌변에 제2 기본정리를 적용하면
$$
\int_a^b (fg)'\,dt = [fg]_a^b = f(b)g(b) - f(a)g(a)
$$

따라서
$$
f(b)g(b) - f(a)g(a) = \int_a^b f'g\,dt + \int_a^b fg'\,dt
$$

정리하면
$$
\int_a^b f'g\,dt = f(b)g(b) - f(a)g(a) - \int_a^b fg'\,dt
$$


### 예제 3. $\displaystyle\int x^k e^{-x}\,dx$ (부정적분, $k \in \mathbb{N}$)
이 적분은 **부분적분**을 반복 적용하여 구할 수 있다.  
$k=1$인 경우를 먼저 살펴보자.

**$k=1$일 때:**  
$f(x) = x$, $g'(x) = e^{-x}$로 두면  
$f'(x) = 1$, $g(x) = -e^{-x}$

부분적분 공식에 의해
$$
\int x e^{-x}\,dx = -xe^{-x} - \int (-e^{-x})\,dx = -xe^{-x} - e^{-x} + C = -(x+1)e^{-x} + C
$$

**일반적인 $k \in \mathbb{N}$인 경우:**  
$f(x) = x^k$, $g'(x) = e^{-x}$로 두면  
$f'(x) = kx^{k-1}$, $g(x) = -e^{-x}$

부분적분 공식에 의해
$$
\int x^k e^{-x}\,dx = -x^k e^{-x} + k\int x^{k-1} e^{-x}\,dx
$$

이 과정을 반복하면 최종적으로
$$
\int x^k e^{-x}\,dx = -e^{-x}\sum_{j=0}^{k} \frac{k!}{j!}x^j + C
$$

**정적분 예시:** $\displaystyle\int_0^\infty x^k e^{-x}\,dx$  
위의 부정적분 결과를 이용하여
$$
\int_0^\infty x^k e^{-x}\,dx = \lim_{b\to\infty}\left[-e^{-x}\sum_{j=0}^{k} \frac{k!}{j!}x^j\right]_0^b
$$

$x \to \infty$일 때 지수함수가 다항식보다 빠르게 증가하므로 $\lim_{x\to\infty} x^j e^{-x} = 0$ (모든 $j$에 대해)

따라서
$$
\int_0^\infty x^k e^{-x}dx = 0 - \left(-e^0 \cdot \frac{k!}{0!} \cdot 0^0\right) = k!
$$

이는 **감마함수** $\Gamma(n+1) = n!$ (단, $n \in \mathbb{N}$)의 특수한 경우이다.

### 예제 4. $\displaystyle\int_0^\infty x^a e^{-bx}\,dx$ (일반화, $a > -1$, $b > 0$)

**치환적분을 이용한 일반화:**  
$u = bx$로 치환하면 $du = b\,dx$, 즉 $dx = \frac{1}{b}du$

$x = 0$일 때 $u = 0$, $x \to \infty$일 때 $u \to \infty$

따라서
$$
\int_0^\infty x^a e^{-bx}\,dx = \int_0^\infty \left(\frac{u}{b}\right)^a e^{-u} \frac{1}{b}\,du = \frac{1}{b^{a+1}}\int_0^\infty u^a e^{-u}\,du
$$

**$a = k$ (자연수)인 경우:**  
예제 3의 결과를 이용하면
$$
\int_0^\infty u^k e^{-u}\,du = k!
$$

따라서
$$
\int_0^\infty x^k e^{-bx}\,dx = \frac{k!}{b^{k+1}}
$$

**일반적인 $a > -1$인 경우 (감마함수):**  
감마함수는 $\Gamma(a+1) = \int_0^\infty u^a e^{-u}\,du$로 정의되며,

- $a$가 자연수일 때: $\Gamma(a+1) = a!$
- 일반적으로: $\Gamma(a+1) = a\Gamma(a)$ (재귀 관계식)
- 특히: $\Gamma\left(\frac{1}{2}\right) = \sqrt{\pi}$

따라서 일반적인 결과는
$$
\int_0^\infty x^a e^{-bx}\,dx = \frac{\Gamma(a+1)}{b^{a+1}}, \quad (a > -1, b > 0)
$$

# 3. 리만적분의 확장 *(Extensions of the Riemann Integral)*
## (1) 특이적분 *(Improper Integral)*
적분 구간에 개구간이 포함되면 어떻게 정의하는지 살펴보자.  

### Def. 1. $(a,b]$ 또는 $[a,b)$의 경우

① $f:(a,b]\to\mathbb R$가 임의의 $c\in(a,b)$에 대하여
$f\in\mathcal R[c,b]$이면, $(a,b]$에서 $f$의 특이적분은
$$
\int_a^b f := \lim_{c\to a^+}\int_c^b f
$$
로 정의한다.

* $f:[a,b)\to\mathbb R$인 경우
  $$
  \int_a^b f := \lim_{c\to b^-}\int_a^c f
  $$

② ①에서 우변의 극한이 존재하면 각 구간에 대해 $f$는 **특이적분가능**하다고 한다.

③ $f:[a,b]\setminus{c}\to\mathbb R$가 $[a,c)$와 $(c,b]$에서 각각 특이적분가능하면
$f$는 $[a,b]$에서 **특이적분가능**하다고 하고
$$
\int_a^b f
:=\lim_{x\to c^-}\int_a^x f
+\lim_{y\to c^+}\int_y^b f
$$
로 정의한다.

- 즉 한 포인트를 걸러주는게 얼마든지 가능하다

#### 예시
1. $$\int_1^\infty x^{-n} dx < +\infty \Leftrightarrow n > 1$$
2. $$\int_0^1 x^{\alpha-1} dx < +\infty \Leftrightarrow \alpha > 0$$
3. $$\int_{-\infty}^{\infty} \frac{x}{1+x^2} dx$$
   - **양의 부분**: $f^+(x) = \max\left(\frac{x}{1+x^2}, 0\right)$
     $$\int_0^{\infty} \frac{x}{1+x^2} dx = \infty$$
   - **음의 부분**: $f^-(x) = \max\left(-\frac{x}{1+x^2}, 0\right)$
     $$\int_{-\infty}^{0} \frac{x}{1+x^2} dx = \infty$$
   - **결론**: 둘 다 무한대이므로 $\int f^+ d\mu - \int f^- d\mu = \infty - \infty$ (부정형)
   - 따라서 **적분 불가능** (발산)
   - 주의: 리만 적분에서 Cauchy principal value로는 0이지만, 르벡 적분으로는 정의 불가

### Def. 2. $[a,\infty)$ 또는 $(-\infty,b]$의 경우

① $f:[a,\infty)\to\mathbb R$가 임의의 실수 $c(>a)$에 대하여
$f\in\mathcal R[a,c]$이면 $[a,\infty)$에서 f의 특이적분은
$$
\int_a^\infty f := \lim_{c\to\infty}\int_a^c f
$$
로 정의한다.

* $f:(-\infty,b]\to\mathbb R$인 경우
  $$
  \int_{-\infty}^b f := \lim_{c\to-\infty}\int_c^b f
  $$

② ①에서 우변의 극한이 존재하면 각 구간에 대해 $f$는 **특이적분가능**하다고 한다.

③ $f$가 적당한 $p\in\mathbb R$에 대하여
$(-\infty,p]$와 $[p,\infty)$에서 특이적분가능하면
$f$는 $\mathbb R$에서 특이적분가능하다고 하고
$$
\int_{-\infty}^{\infty} f
:=\int_{-\infty}^p f+\int_p^{\infty} f
$$
로 정의한다.

## 특이적분 예시
### 예제 1. $\displaystyle\int_0^1 \frac{1}{\sqrt{x}}\,dx$
$f(x) = \frac{1}{\sqrt{x}}$는 $x=0$에서 정의되지 않으므로 $(0,1]$에서의 특이적분이다.

임의의 $c \in (0,1)$에 대하여 $f \in \mathcal{R}[c,1]$이므로
$$
\int_0^1 \frac{1}{\sqrt{x}}\,dx = \lim_{c\to 0^+}\int_c^1 \frac{1}{\sqrt{x}}\,dx
$$

$F(x) = 2\sqrt{x}$는 $f(x) = \frac{1}{\sqrt{x}}$의 원시함수이므로
$$
\int_c^1 \frac{1}{\sqrt{x}}\,dx = [2\sqrt{x}]_c^1 = 2\sqrt{1} - 2\sqrt{c} = 2 - 2\sqrt{c}
$$

따라서
$$
\int_0^1 \frac{1}{\sqrt{x}}\,dx = \lim_{c\to 0^+}(2 - 2\sqrt{c}) = 2
$$

### 예제 2. $\displaystyle\int_1^\infty \frac{1}{x^2}\,dx$

$f(x) = \frac{1}{x^2}$에 대하여 임의의 실수 $c > 1$에 대해 $f \in \mathcal{R}[1,c]$이므로
$$
\int_1^\infty \frac{1}{x^2}\,dx = \lim_{c\to\infty}\int_1^c \frac{1}{x^2}\,dx
$$

$F(x) = -\frac{1}{x}$는 $f(x) = \frac{1}{x^2}$의 원시함수이므로
$$
\int_1^c \frac{1}{x^2}\,dx = \left[-\frac{1}{x}\right]_1^c = -\frac{1}{c} - \left(-\frac{1}{1}\right) = 1 - \frac{1}{c}
$$

따라서
$$
\int_1^\infty \frac{1}{x^2}\,dx = \lim_{c\to\infty}\left(1 - \frac{1}{c}\right) = 1
$$


## (2) 스틸체스적분 *(Riemann–Stieltjes Integral)*
리만적분은 적분 구간이 균일하게 변한다는 가정하에 가능했음. 이를 일반화 한게 스틸체스 적분.  
불연속적인 적분도 가능하게!!  
이산/연속에서 확률분포 정의해서 적분하기  

### Def. 1. [스틸체스 상합과 하합]
$[a,b]$에서 유계인 함수 $f$와 증가함수 $\alpha$에 대하여,
$[a,b]$의 분할
$$
\varphi=\{x_0,x_1,\dots,x_n\},\quad a=x_0<\cdots<x_n=b
$$
및
$$
\Delta\alpha_i=\alpha(x_i)-\alpha(x_{i-1})
$$
에 대하여 다음을 정의한다.

① **스틸체스 상합** *(Riemann–Stieltjes upper sum)*:
$$
U(\varphi,f,\alpha)=\sum_{i=1}^n M_i\Delta\alpha_i
$$

② **스틸체스 하합** *(Riemann–Stieltjes lower sum)*:
$$
L(\varphi,f,\alpha)=\sum_{i=1}^n m_i\Delta\alpha_i
$$

여기서
$$
M_i=\sup\{f(x)\mid x_{i-1}\le x\le x_i\}, \\
m_i=\inf\{f(x)\mid x_{i-1}\le x\le x_i\}
$$
이다.
이를 각각 $\alpha$에 관한 $f$의 **스틸체스 상합**, **스틸체스 하합**이라 한다.
($i=1,\dots,n$)

### Def. 2. [스틸체스 상적분과 하적분]
$[a,b]$에서 유계인 함수 $f$와 증가함수 $\alpha$에 대하여,
$[a,b]$의 분할 $\varphi=\{x_0,x_1,\dots,x_n\}$에 대해
$$
\Delta\alpha_i=\alpha(x_i)-\alpha(x_{i-1})
$$
로 두고,
$$
U(\varphi,f,\alpha)=\sum_{i=1}^n M_i\Delta\alpha_i,\quad
L(\varphi,f,\alpha)=\sum_{i=1}^n m_i\Delta\alpha_i
$$
라 하자.
(여기서 $M_i=\sup{f(x):x_{i-1}\le x\le x_i}$,
$m_i=\inf{f(x):x_{i-1}\le x\le x_i}$이다.)


이때
$$
\int_a^b f\ d\alpha
:=\inf\{U(\varphi,f,\alpha)\}
$$
를 $\alpha$에 관한 $f$의 **스틸체스 상적분**이라 하고,
$$
\int_a^b f\ d\alpha
:=\sup\{L(\varphi,f,\alpha)\}
$$
를 $\alpha$에 관한 $f$의 **스틸체스 하적분**이라 한다.

### Def. 3. [스틸체스적분가능성]

$f$가 $[a,b]$에서 유계이고 $\alpha$가 $[a,b]$에서 증가함수일 때
$$
\overline{\int_a^b} f\ d\alpha
=\underline{\int_a^b} f\ d\alpha
$$
이면 $f$는 $[a,b]$에서 $\alpha$에 관하여 **스틸체스적분가능**하다고 하며,
$$
\int_a^b f,d\alpha = 
\overline{\int_a^b} f\ d\alpha
\underline{\int_a^b} f\ d\alpha
$$
로 나타낸다.
이를 $\alpha$에 관한 $f$의 **스틸체스적분**이라 한다.
(이때 $f\in\mathcal R_\alpha[a,b]$라 쓴다.)

### Thm.

$f\in\mathcal R[a,b]$이고,
$\alpha$가 $[a,b]$에서 증가하고 $(a,b)$에서 미분가능한 함수이며
$\alpha'\in\mathcal R_\alpha[a,b]$이면,
$$
f\in\mathcal R_\alpha[a,b]
$$
이고 다음이 성립한다.
$$
\int_a^b f\ d\alpha =
\int_a^b f(x)\alpha'(x)\ dx
$$


# 참고: 이중합, 이중적분
## 토넬리 정리 (Tonelli's Theorem)
$f(x_1, x_2)$가 $[a,b] \times [c,d]$에서 비음수인 유계함수일 때:  
$$\int_{a}^{b} \int_{c}^{d} f(x_1, x_2) \, dx_2 \, dx_1 = \int_{c}^{d} \int_{a}^{b} f(x_1, x_2) \, dx_1 \, dx_2 = \int_{[a,b] \times [c,d]} f(x_1, x_2) \, d(x_1, x_2)$$

- **의미**: 비음수 함수의 이중적분은 반복적분으로 계산 가능
- **순서 교환**: 적분 순서를 자유롭게 바꿀 수 있음
- **조건**: $f(x_1, x_2) \geq 0$이면 충분 (적분 순서 교환 가능)

**이중합 버전 (Double Sum)**  
가산개의 비음수 항 $a_{i,j} \geq 0$ (단, $i, j \in \mathbb{N}$)에 대해:

$$\sum_{i=1}^{\infty} \sum_{j=1}^{\infty} a_{i,j} = \sum_{j=1}^{\infty} \sum_{i=1}^{\infty} a_{i,j} = \sum_{(i,j) \in \mathbb{N}^2} a_{i,j}$$

- **합의 순서 무관**: 비음수 이중합은 합의 순서와 무관하게 같음
- **부분합의 수렴**: 어떤 합의 순서로 진행하든 같은 값으로 수렴
- **반례**: 항이 음수를 포함하면 순서에 따라 값이 달라질 수 있음

**일반적인 함수의 이중합과 이중적분**  
$f$가 $[a,b] \times [c,d]$에서 정의된 함수일 때, $f$의 양수부분과 음수부분을 다음과 같이 정의한다:
$$
f^+(x_1, x_2) = \max(f(x_1, x_2), 0), \quad f^-(x_1, x_2) = \max(-f(x_1, x_2), 0)
$$

그러면 $f = f^+ - f^-$이고, $f^+ \geq 0$, $f^- \geq 0$이다.

**절대수렴 조건 하에서 이중적분:**

$f(x_1, x_2)$가 $[a,b] \times [c,d]$에서 절대수렴, 즉
$$
\int_{a}^{b} \int_{c}^{d} |f(x_1, x_2)| \, dx_2 \, dx_1 < \infty
$$
일 때, 다음이 성립한다:
$$
\int_{[a,b] \times [c,d]} f \, d(x_1, x_2) = \int_{[a,b] \times [c,d]} f^+ \, d(x_1, x_2) - \int_{[a,b] \times [c,d]} f^- \, d(x_1, x_2)
$$

그리고 반복적분도 같은 값으로 수렴한다:
$$
\int_{a}^{b} \int_{c}^{d} f(x_1, x_2) \, dx_2 \, dx_1 = \int_{c}^{d} \int_{a}^{b} f(x_1, x_2) \, dx_1 \, dx_2
$$

**이중합 버전 (일반적인 경우):**

$\{a_{i,j}\}_{i,j \in \mathbb{N}}$이 절대수렴, 즉
$$
\sum_{i=1}^{\infty} \sum_{j=1}^{\infty} |a_{i,j}| < \infty
$$
일 때:
$$
\sum_{i=1}^{\infty} \sum_{j=1}^{\infty} a_{i,j} = \sum_{j=1}^{\infty} \sum_{i=1}^{\infty} a_{i,j} = \sum_{(i,j) \in \mathbb{N}^2} a_{i,j}
$$

이는 다음과 같이 양수부분과 음수부분으로 분해하여 증명할 수 있다:
$$
\sum_{i,j} a_{i,j} = \sum_{i,j} a_{i,j}^+ - \sum_{i,j} a_{i,j}^-
$$

여기서 $a_{i,j}^+ = \max(a_{i,j}, 0)$, $a_{i,j}^- = \max(-a_{i,j}, 0)$이고, 비음수 항들의 합은 순서와 무관하다.

## 후비니 정리 *(Fubini's Theorem)*
$f$가 $[a,b] \times [c,d]$에서 적분가능하면
$$
\int_{[a,b] \times [c,d]} f \, d(x,y) = \int_a^b \int_c^d f(x,y) \, dy \, dx = \int_c^d \int_a^b f(x,y) \, dx \, dy
$$

**특수한 경우:**
- $f(x,y) = g(x)h(y)$ (곱셈 가능)이면
$$
\int_a^b \int_c^d f(x,y) \, dy \, dx = \left(\int_a^b g(x) \, dx\right)\left(\int_c^d h(y) \, dy\right)
$$

**적용 조건:**
- 절대수렴할 때만 적분순서 교환 가능
- 음수항이 있으면 반드시 절대수렴 확인 필요



# [연습문제]

1. 함수
   $$
   f(x)=
   \begin{cases}
   0, & 0\le x\le \frac12,\\
   x, & \frac12<x\le 1
   \end{cases}
   $$
   가 $f\in\mathcal R[0,1]$임을 보이시오.

2. 다음 명제들의 반례를 제시하시오.
   
   (1) $f\in\mathcal R[a,b]$이면 $[a,b]$에서 $f$의 불연속점의 개수는 유한개이다.

   (2) $f,g\in\mathcal R[a,b]$이면
   $$
   \int_a^b (f\times g)
   =
   \left(\int_a^b f\right)\times
   \left(\int_a^b g\right)
   $$
   이다.

3. 다음 적분을 구하시오.  
   (1) $\displaystyle\int_0^\pi \sin x\ dx$  
   (2) $\displaystyle\int_0^\pi x\sin x\ dx$  
   (3) $\displaystyle\int_0^{\sqrt\pi} x\sin(x^2)\ dx$  

4. 다음 특이적분이 가능한지 판별하고, 가능하다면 그 값을 구하시오.  
   (1) $\displaystyle\int_0^1 \frac1{\sqrt[3]{x}}\ dx$  
   (2) $\displaystyle\int_2^\infty \frac1{x\ln x}\ dx$  

5. 다음 스틸체스적분을 구하시오.  
  (1) $\displaystyle\int_0^3 x^2\ de^x$  
  (2) $\displaystyle\int_2^3 (x-1)\ d(x^2+2)$  

#### 풀이

**(1)** $\displaystyle\int_0^3 x^2\ de^x$

$f(x) = x^2$, $\alpha(x) = e^x$로 두면 $\alpha'(x) = e^x$이다.

정리에 의해
$$
\int_0^3 x^2\ de^x = \int_0^3 x^2 \cdot e^x\ dx
$$

부분적분을 사용한다. $u = x^2$, $dv = e^x dx$로 두면  
$du = 2x\,dx$, $v = e^x$

$$
\int_0^3 x^2 e^x\ dx = [x^2 e^x]_0^3 - \int_0^3 2xe^x\ dx
$$

다시 부분적분: $u = 2x$, $dv = e^x dx$이면  
$du = 2\,dx$, $v = e^x$

$$
\int_0^3 2xe^x\ dx = [2xe^x]_0^3 - \int_0^3 2e^x\ dx = [2xe^x]_0^3 - [2e^x]_0^3
$$

따라서
$$
\int_0^3 x^2 e^x\ dx = [x^2 e^x]_0^3 - ([2xe^x]_0^3 - [2e^x]_0^3)
$$
$$
= [x^2 e^x - 2xe^x + 2e^x]_0^3 = [e^x(x^2 - 2x + 2)]_0^3
$$
$$
= e^3(9 - 6 + 2) - e^0(0 - 0 + 2) = 5e^3 - 2
$$

**(2)** $\displaystyle\int_2^3 (x-1)\ d(x^2+2)$

$f(x) = x-1$, $\alpha(x) = x^2+2$로 두면 $\alpha'(x) = 2x$이다.

정리에 의해
$$
\int_2^3 (x-1)\ d(x^2+2) = \int_2^3 (x-1) \cdot 2x\ dx
$$
$$
= \int_2^3 2x(x-1)\ dx = \int_2^3 (2x^2 - 2x)\ dx
$$
$$
= \left[\frac{2x^3}{3} - x^2\right]_2^3
$$
$$
= \left(\frac{2 \cdot 27}{3} - 9\right) - \left(\frac{2 \cdot 8}{3} - 4\right)
$$
$$
= (18 - 9) - \left(\frac{16}{3} - 4\right) = 9 - \frac{4}{3} = \frac{23}{3}
$$
