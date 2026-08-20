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
  - 공통 세분 (common refinement): $\varphi_1\cup\varphi_2$는 $\varphi_1,\varphi_2$의 공통 세분
    - $L(\varphi_1,f)\le L(\varphi_1\cup\varphi_2,f)\le U(\varphi_1\cup\varphi_2,f)\le U(\varphi_2,f)$

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
### Thm.1 리만 적분 판별법 *(Riemann Integrability Criterion)*
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
U(\varphi, f) \le U(\varphi_1, f) < \overline{\int_a^b} f + \frac{\varepsilon}{2}\\
L(\varphi, f) \ge L(\varphi_2, f) > \underline{\int_a^b} f - \frac{\varepsilon}{2}
$$

따라서

$$
U(\varphi, f) - L(\varphi, f) < \left(\overline{\int_a^b} f + \frac{\varepsilon}{2}\right) - \left(\underline{\int_a^b} f - \frac{\varepsilon}{2}\right) = \varepsilon
$$

**($\Leftarrow$)** 임의의 $\varepsilon > 0$에 대하여 $U(\varphi, f) - L(\varphi, f) < \varepsilon$인 분할 $\varphi$가 존재한다고 하자.

상적분과 하적분의 정의에 의해:

$$
\overline{\int_a^b} f = \inf U(\varphi, f) \le U(\varphi, f) \\
\underline{\int_a^b} f = \sup L(\varphi, f) \ge L(\varphi, f)
$$

따라서

$$
\overline{\int_a^b} f - \underline{\int_a^b} f \le U(\varphi, f) - L(\varphi, f) < \varepsilon
$$

$\varepsilon$는 임의의 양수이므로

$$
\overline{\int_a^b} f = \underline{\int_a^b} f
$$

즉, $f \in \mathcal{R}[a,b]$이다.

### Thm.2 연속함수의 적분가능성 *(Continuity implies Riemann Integrability)*
$f$가 $[a,b]$에서 연속이면

$$
f\in\mathcal R[a,b]
$$

* 역은 성립 안함: 불연속인데 리만적분 가능한 경우도 있긴 함! 

>**증명**  
>
>$f$가 $[a,b]$에서 연속이므로, 유계 폐구간에서 연속인 함수는 **균등연속** *(uniformly continuous, 고른연속)* 이다. (by 하이네-칸토어 정리(Heine–Cantor theorem))
>
>즉, 임의의 $\varepsilon > 0$에 대하여 $\delta > 0$이 존재하여
>
>$$
>|x - y| < \delta \implies |f(x) - f(y)| < \frac{\varepsilon}{b-a}
>$$
>
>이제 $[a,b]$의 분할 $\varphi = \{x_0, x_1, \dots, x_n\}$을 $\|\varphi\| = \max_i \Delta x_i < \delta$가 되도록 선택하자.
>- $\max_i \Delta x_i$는 분할에서 가장 긴 소구간의 길이
>- $\|\varphi\|$: 분할의 크기 또는 메시(mesh) 라고 부른다.
>
>각 소구간 $[x_{i-1}, x_i]$에서 $f$는 연속이므로 **최대·최소 정리**에 의해 최댓값과 최솟값을 갖는다. 즉,
>
>$$
>M_i = \max_{x \in [x_{i-1}, x_i]} f(x), \quad m_i = \min_{x \in [x_{i-1}, x_i]} f(x)
>$$
>
>$\Delta x_i < \delta$이므로 균등연속성에 의해 $M_i - m_i < \frac{\varepsilon}{b-a}$ 
>
>따라서
>
>$$
>U(\varphi, f) - L(\varphi, f) = \sum_{i=1}^n (M_i - m_i)\Delta x_i < \frac{\varepsilon}{b-a} \sum_{i=1}^n \Delta x_i = \frac{\varepsilon}{b-a} \cdot (b-a) = \varepsilon
>$$
>
>**Thm.1 판별법**에 의해 $f \in \mathcal{R}[a,b]$이다.

### Thm.3 적분의 평균값 정리 *(Mean Value Theorem for Integrals)*
$f$가 $[a,b]$에서 연속이면

$$
\exists c\in(a,b)\text{ s.t. }
\int_a^b f(x),dx=f(c)(b-a)
$$

>**증명**
>
>$f$가 $[a,b]$에서 연속이므로, **Thm.2**에 의해 $f \in \mathcal{R}[a,b]$이다.
>
>연속함수의 **최대·최소 정리**에 의해
>
>$$
>m = \min_{x \in [a,b]} f(x), \quad M = \max_{x \in [a,b]} f(x)
>$$
>
>가 존재한다.
>
>따라서 모든 $x \in [a,b]$에 대해
>
>$$
>m \le f(x) \le M
>$$
>
>적분의 단조성에 의해
>
>$$
>m(b-a) \le \int_a^b f(x)\,dx \le M(b-a)
>$$
>
>양변을 $(b-a)$로 나누면
>
>$$
>m \le \frac{1}{b-a}\int_a^b f(x)\,dx \le M
>$$
>
>$f$는 $[a,b]$에서 연속이므로 **중간값 정리**에 의해, 어떤 $c \in [a,b]$가 존재하여
>
>$$
>f(c) = \frac{1}{b-a}\int_a^b f(x)\,dx
>$$
>
>따라서
>
>$$
>\int_a^b f(x)\,dx = f(c)(b-a)
>$$
>
>$f$가 상수함수가 아니면 $m < M$이므로 $c \in (a,b)$이다.

### 참고: 단조증가(감소) 함수와 적분가능성

함수 $f:[a,b]\to\mathbb{R}$가 증가함수, 즉 $x < y\implies f(x)\le f(y)$라고 하자. 그러면 $f$는 $[a,b]$에서 리만 적분 가능하다.

>**증명**
>
>모든 $x\in[a,b]$에 대해 $f(a)\le f(x)\le f(b)$ 이므로 $f$는 유계이다.
>
>구간 $[a,b]$를 같은 길이의 $n$개의 부분구간으로 나누는 분할 $P_n=\{x_0,x_1,\dots,x_n\}$ 을 생각하자. 여기서
>
>$$
>x_k=a+\frac{k}{n}(b-a), \qquad \Delta x_k=x_k-x_{k-1}=\frac{b-a}{n}
>$$
>
>이다. $f$가 증가함수이므로 각 부분구간 $[x_{k-1},x_k]$에서 최솟값과 최댓값은
>
>$$
>m_k=f(x_{k-1}), \qquad M_k=f(x_k)
>$$
>
>이다. 따라서 하합과 상합은
>
>$$
>L(f,P_n)=\sum_{k=1}^n f(x_{k-1})\frac{b-a}{n}, \qquad U(f,P_n)=\sum_{k=1}^n f(x_k)\frac{b-a}{n}
>$$
>
>이다. 두 합의 차이는
>
>$$
>U(f,P_n)-L(f,P_n)=\frac{b-a}{n}\sum_{k=1}^n (f(x_k)-f(x_{k-1}))
>$$
>
>이고, 우변의 중간 항들이 모두 소거되어
>
>$$
>U(f,P_n)-L(f,P_n)=\frac{b-a}{n}(f(x_n)-f(x_0))=\frac{b-a}{n}(f(b)-f(a)).
>$$
>
>따라서
>
>$$
>\lim_{n\to\infty}[U(f,P_n)-L(f,P_n)] = \lim_{n\to\infty} \frac{b-a}{n}(f(b)-f(a)) = 0.
>$$
>
>적분 가능성에 대한 수열 판정법에 의해 $f$는 $[a,b]$에서 리만 적분 가능하다.
>
>$\varepsilon$-형식으로 쓰면, $f(b)>f(a)$일 때
>
>$$
>n>\frac{(b-a)(f(b)-f(a))}{\varepsilon}
>$$
>
>로 $n$을 선택하면
>
>$$
>U(f,P_n)-L(f,P_n)<\varepsilon.
>$$
>
>$f(b)=f(a)$이면 증가성에 의해 $f$는 상수함수이므로 바로 적분 가능하다. 연속성을 가정하지 않아도, 증가함수가 불연속점을 가질 수 있어도 위의 상합과 하합의 차이가 0으로 수렴하기 때문에 적분 가능하다.

## (3) 리만적분의 연산 *(Properties of the Riemann Integral)*
$f,g\in\mathcal R[a,b]$이면 다음이 성립한다.

### 연산1. 선형성 *(Linearity)*

$$
\int_a^b (f\pm g)=\int_a^b f\pm\int_a^b g
$$



>**증명**  
>
>**($\alpha f + \beta g$의 적분가능성)**  
>임의의 $\varepsilon > 0$에 대하여, $f, g \in \mathcal{R}[a,b]$이므로 **Thm.1 판별법**에 의해 분할 $\varphi_1, \varphi_2$가 존재하여
>
>$$
>U(\varphi_1, f) - L(\varphi_1, f) < \frac{\varepsilon}{2|\alpha|}, \quad U(\varphi_2, g) - L(\varphi_2, g) < \frac{\varepsilon}{2|\beta|}
>$$
>
>($\alpha, \beta \neq 0$인 경우. 0이면 자명)
>
>$\varphi = \varphi_1 \cup \varphi_2$라 하면, 각 소구간에서:
>- $\alpha > 0$이면 $M_i(\alpha f) = \alpha M_i(f)$, $m_i(\alpha f) = \alpha m_i(f)$
>- $\alpha < 0$이면 $M_i(\alpha f) = \alpha m_i(f)$, $m_i(\alpha f) = \alpha M_i(f)$
>
>따라서
>
>$$
>U(\varphi, \alpha f + \beta g) - L(\varphi, \alpha f + \beta g) \\ \le |\alpha|(U(\varphi, f) - L(\varphi, f)) + |\beta|(U(\varphi, g) - L(\varphi, g)) \\
>< \varepsilon
>$$
>
>고로 $\alpha f + \beta g \in \mathcal{R}[a,b]$.
>
>**($\int(f+g) = \int f + \int g$)**  
>분할 $\varphi$에 대해 각 소구간에서
>
>$$
>m_i(f) + m_i(g) \le f(x) + g(x) \le M_i(f) + M_i(g)
>$$
>
>따라서
>
>$$
>m_i(f) + m_i(g) \le m_i(f+g) \le M_i(f+g) \le M_i(f) + M_i(g)
>$$
>
>이를 합하면
>
>$$
>L(\varphi, f) + L(\varphi, g) \le L(\varphi, f+g) \le U(\varphi, f+g) \le U(\varphi, f) + U(\varphi, g)
>$$
>
>극한을 취하면
>
>$$
>\int_a^b f + \int_a^b g = \int_a^b (f+g)
>$$

### 연산2. 상수배 *(Scalar multiples)*

$$
f\in\mathcal R[a,b],\quad k\in\mathbb R \implies kf\in\mathcal R[a,b],\quad \int_a^b kf = k\int_a^b f.
$$

>**증명**  
>선형성에서 $g=0$으로 취하면 바로 얻어진다. 또한 $k\neq 0$일 때 각 소구간에서
>
>$$
>M_i(kf)=kM_i(f),\quad m_i(kf)=km_i(f)
>$$
>
>또는 $k<0$일 때는 $M_i(kf)=km_i(f)$, $m_i(kf)=kM_i(f)$가 되므로 상합과 하합의 차이는 $|k|$배로 늘어난다. 따라서 적분 가능성이 유지되고, 적분값도 $k$배가 된다.
>
### 연산3. 상한·하한에 의한 부등식 *(Bounds from minimum and maximum)*

$$
m\le f(x)\le M\quad (\forall x\in[a,b]) \implies m(b-a)\le \int_a^b f\le M(b-a).
$$

>**증명**  
>분할 $
ho$에 대해 각 소구간 $[x_{i-1},x_i]$에서
>
>$$
>m\le m_i(f)\le M_i(f)\le M
>$$
>
>이므로
>
>$$
>m\Delta x_i\le L_i(f,\rho)\le U_i(f,\rho)\le M\Delta x_i.
>$$
>
>모든 소구간을 합하면
>
>$$
>m(b-a)\le L(f,\rho)\le U(f,\rho)\le M(b-a).
>$$
>
>따라서 적분값도 이 사이에 들어간다.
>
### 연산4. 단조성 *(Monotonicity)*

$$
f(x)\le g(x)\quad (\forall x\in[a,b]) \implies \int_a^b f\le \int_a^b g.
$$

>**증명**  
>$h=g-f\ge0$이라 하면 $h\in\mathcal R[a,b]$이고, 연산3에 의해
>
>$$
>0\le \int_a^b h = \int_a^b(g-f) = \int_a^b g - \int_a^b f.
>$$
>
>따라서
>
>$$
>\int_a^b f \le \int_a^b g.
>$$
>
### 연산5. 절대값의 적분 가능성과 절대값 부등식 *(Absolute value and inequality)*

$$
f\in\mathcal R[a,b] \implies |f|\in\mathcal R[a,b],\qquad \left|\int_a^b f\right|\le \int_a^b |f|.
$$

>**증명**  
>분할 $
ho$에 대해 각 소구간에서
>
>$$
>M_i(|f|)-m_i(|f|)\le M_i(f)-m_i(f)
>$$
>
>이 성립하므로, $f$의 상합과 하합의 차이가 작아질 때 $|f|$의 상합과 하합의 차이도 작아진다. 따라서 $|f|\in\mathcal R[a,b]$이다.
>
>또한
>
>$$
>-|f|\le f\le |f|
>$$
>
>이므로 연산4를 적용하면
>
>$$
>-\int_a^b |f|\le \int_a^b f\le \int_a^b |f|.
>$$
>
>따라서
>
>$$
>\left|\int_a^b f\right|\le \int_a^b |f|.
>$$
>

### 연산6. 함수 곱의 적분도 적분가능 *()*


$$
f^2 \in \mathcal R
$$

$$
fg \in \mathcal R
$$

>**증명**
>
>$f$가 $[a,b]$에서 리만 적분 가능하므로 유계이다 (리만 적분 전제조건임). 따라서 어떤 $M>0$가 존재하여 $|f(x)|\le M \quad $x\in[a,b]$$ 이다.  
>임의의 분할 $P=\{x_0,x_1,\dots,x_n\}$ 을 잡고 부분구간을 $I_k=[x_{k-1},x_k]$ 이라 하자. 각 부분구간에서 함수의 진동 폭을
>
>$$
>\omega_k(f)=\sup_{x,y\in I_k}|f(x)-f(y)|
>$$
>
>라고 하면
>
>$$
>\omega_k(f)=\sup_{I_k}f-\inf_{I_k}f
>$$
>
>마찬가지로
>
>$$
>\omega_k(f^2)=\sup_{x,y\in I_k}|f(x)^2-f(y)^2|
>$$
>
>이다. 위의 유계성에 의해 모든 $x,y\in I_k$에 대해 다음과 같이 전개할 수 있다.
>
>$$
>|f(x)^2-f(y)^2|=|f(x)-f(y)|\,|f(x)+f(y)| \\
>\le |f(x)-f(y)|\bigl(|f(x)|+|f(y)|\bigr) \\
>\le |f(x)-f(y)|\cdot( M+M )
>=2M|f(x)-f(y)|.
>$$
>
>이므로 $\omega_k(f^2)\le 2M\,\omega_k(f)$  
>따라서 상합과 하합의 차이에 대해
>
>$$
>U(f^2,P)-L(f^2,P)=\sum_{k=1}^n\omega_k(f^2)(x_k-x_{k-1}) \\
>\le 2M\sum_{k=1}^n\omega_k(f)(x_k-x_{k-1})
>=2M\bigl(U(f,P)-L(f,P)\bigr).
>$$
>
>이제 $\varepsilon>0$를 임의로 잡는다. $f$가 적분 가능하므로 어떤 분할 $P$가 존재하여 $U(f,P)-L(f,P)<\frac{\varepsilon}{2M}$ 이 되게 할 수 있다. 그러면
>
>$$
>U(f^2,P)-L(f^2,P)\le 2M\bigl(U(f,P)-L(f,P)\bigr)
><2M\cdot\frac{\varepsilon}{2M}=\varepsilon.
>$$
>
>그러므로 리만 적분 가능성의 판정법에 의해 $f^2$은 $[a,b]$에서 적분 가능하다.
>
> ---
>f와 g가 적분 가능하므로 합에 관한 정리에 의해 $f+g$ 도 적분 가능하다. $f,g,f+g$ 각각 위 공식에 의해 $f^2,g^2,(f+g)^2$ 이 모두 적분 가능하다.  
>한편 $(f+g)^2=f^2+2fg+g^2$ 이므로
>
>$$
>fg=	\frac12\bigl((f+g)^2-f^2-g^2\bigr).
>$$
>
>적분 가능한 함수들의 합, 차 및 상수배는 적분 가능하므로 우변이 적분 가능하다. 따라서
>
>$fg$도 $[a,b]$에서 리만 적분 가능하다.

### 정의. 적분의 방향 반전과 0 길이 구간

$f\in\mathcal R[a,b]$일 때, 다음이 성립한다.

$$
\int_a^b f(x)\,dx = -\int_b^a f(x)\,dx,
\qquad
\int_c^c f(x)\,dx = 0 \quad (c\in[a,b]).
$$

- 첫째 식은 적분의 방향을 반대로 바꾸면 부호가 반대로 바뀐다는 뜻
- 둘째 식은 구간 $[c,c]$의 길이가 0이므로, 임의의 분할에 대한 하합과 상합이 모두 0이 되어 적분값도 0으로 정한 것
- 적분 계산을 간단히 만들기 위한 관습이다.

### 정리. 구간의 가법성 *(Additivity over intervals)*

$$
f \in \mathcal{R}[a,b] \Leftrightarrow
\forall c\in(a,b),\int_a^b f=\int_a^c f+\int_c^b f
$$

- 무한히 많은 구간으로 분해하려면 별도의 극한 논증이 필요

**증명**

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
U(\varphi^*, f) = U(\varphi_1, f) + U(\varphi_2, f) \\
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

### 유한합과 적분의 교환
리만 적분이 가능한 함수들의 유한합은 적분 가능하고, 적분과 유한합의 순서를 바꿀 수 있다.

함수 $f_1, f_2, \dots, f_n$이 모두 $[a,b]$에서 리만 적분 가능하고 $c_1, c_2, \dots, c_n \in \mathbb{R}$이면

$$
\sum_{k=1}^n c_k f_k \in \mathcal{R}[a,b]
$$

이며

$$
\int_a^b \left(\sum_{k=1}^n c_k f_k(x)\right) dx
= \sum_{k=1}^n c_k \int_a^b f_k(x)\,dx.
$$

특히 두 함수에 대해서는

$$
\int_a^b (f+g)\,dx = \int_a^b f\,dx+\int_a^b g\,dx,
\qquad
\int_a^b cf\,dx = c\int_a^b f\,dx.
$$

여기서 중요한 점은 **유한합**이라는 것이다. 유한합은 항의 개수가 고정되어 있으므로 각 항의 적분가능성과 선형성을 유한 번 적용하면 된다. 따라서 합과 적분의 교환을 위해 급수처럼 별도의 수렴 조건을 확인할 필요가 없다.

예를 들어 구간을 $[a,b]$의 분할점 $a=x_0<x_1<\dots<x_m=b$로 나누면

$$
\int_a^b \sum_{k=1}^n f_k(x)\,dx
= \sum_{j=1}^m \int_{x_{j-1}}^{x_j}\sum_{k=1}^n f_k(x)\,dx
= \sum_{j=1}^m\sum_{k=1}^n\int_{x_{j-1}}^{x_j}f_k(x)\,dx.
$$

이중합도 유한합이므로 합의 순서를 바꿀 수 있어

$$
=\sum_{k=1}^n\sum_{j=1}^m\int_{x_{j-1}}^{x_j}f_k(x)\,dx
=\sum_{k=1}^n\int_a^b f_k(x)\,dx.
$$

따라서 적분의 선형성은 다음과 같이 정리된다.

$$
\boxed{\displaystyle
\int_a^b\left(\sum_{k=1}^n c_k f_k\right)
=\sum_{k=1}^n c_k\int_a^b f_k}
$$


## 불연속점이 있는 함수의 적분
불연속점에서 발생하는 함수의 좋지 못한 성질을 분할의 매우 작은 부분구간에 격리 시켜 적분을 계산할 수 있다.

- 유계함수의 함수값을 유한개의 점에서만 변경하면 불연속점이 새로 생길 수 있지만, 리만 적분 가능성과 적분값은 변하지 않는다.

### Thm. 불연속점이 유한개인 함수의 적분가능성 *(Integrability of Functions with Finitely Many Discontinuities)
$f: [a,b] \to \mathbb{R}$가 유계이고, 모든 $c \in (a,b)$에 대해 $f$가 $[c, b]$에서 적분가능하면 $f$는 $[a, b]$에서 적분가능하다.

마찬가지로 모든 $c \in (a,b)$에 대해 $f$가 $[a, c]$에서 적분가능하면 $f$는 $[a, b]$에서 적분가능하다.

- 끝점에서 불연속점을 가지는 유계함수는 닫힌 구간에서 여전히 적분가능하다.
- $[a,b]$와 $[b,d]$에서 적분가능성은 $[a,d]$에서의 적분가능성과 동치다
  - 귀납적으로 유한개 불연속점을 가지는 함수도 모두 적분 가능하다!
- 토메함수는 무한개 불연속점을 가지지만, $[0,1]$ 전체에서 불연속점은 아니고, 리만적분 가능하다!

>**증명**
>
>$\epsilon >0$이라 하자. $U(f,P), L(f,P) < \epsilon$이 되게하는 분할 $P$를 찾으면 증명이 완료된다.
>
>임의의 분할에 대해 다음이 항상 성립한다:
>
>$$U(f,P) - L(f,P) = \sum_{k=1}^n (M_k - m_k) \Delta x_k \\= (M_1 -m_1)(x_1-a) + \sum_{k=2}^n (M_k - m_k) \Delta x_k$$
>
>여기서 $(M_1 - m_1)(x_1 - a) < \epsilon/2$이 되도록 $x_1$을 선택할 수 있다. 이는 $f$가 유계이므로 어떤 $M > 0$이 있어 모든 $x \in [a,b]$에 대해 $|f(x)| \leq M$ 이다. 상한값이 $M$ 보다 같거나 작고, 하한값이 $-M$ 보다 크거나 같으므로 $(M_1 - m_1) \leq 2M$ 이다. 
>
>여기서 $x_1$을 $(x_1 - a) < \epsilon/(4M)$이 되도록 선택한다.
>
>가정에 의해  $f$는 $[x_1, b]$에서 적분가능하므로, 적분 판정법에 의해 $[x_1, b]$의 분할 $P_1$이 존재하여 $U(f, P_1) - L(f, P_1) \leq \epsilon/2$ 다.
>
>마지막으로 $P = {a} \cup P_1$을 $[a,b]$의 분할로 선택하면,
>
>$$ U(f,P) - L(f,P) = (M_1 - m_1)(x_1-a) + \sum_{k=2}^n (M_k - m_k) \Delta x_k \\
>= 2M(x_1-a) + U(f,P_1)-L(f,P_1) < \epsilon/2 + \epsilon/2 = \epsilon$$
>

### 정리. 유한개 불연속점을 가지는 균등수렴 함수열의 적분가능성

$f_n$이 $f$로 균등수렴하고, 각 $f_n$은 유한개의 불연속점을 가지면 $f$는 적분가능하다.

>**증명**
>
>임의의 $\epsilon>0$를 잡자. 균등수렴이므로 어떤 $N$에 대하여
>$$\|f-f_N\|_\infty=\sup_{x\in[0,1]}|f(x)-f_N(x)|<\epsilon/4.$$
>
>$f_N$은 적분가능하므로 어떤 분할 $P$가 존재하여
>
>$$U(f_N,P)-L(f_N,P)<\epsilon/2$$
>
>각 부분구간 $I_k$에 대하여
>
>$$\sup_{I_k}f\le\sup_{I_k}f_N+\epsilon/4,\qquad \inf_{I_k}f\ge\inf_{I_k}f_N-\epsilon/4,$$
>
>따라서
>
>$$\sup_{I_k}f-\inf_{I_k}f\le(\sup_{I_k}f_N-\inf_{I_k}f_N)+\epsilon/2.$$
>
>그러므로 전체 길이가 1임을 이용하면
>
>$$U(f,P)-L(f,P)\le U(f_N,P)-L(f_N,P)+(\epsilon/2)\sum_k\Delta x_k<\epsilon/2+\epsilon/2 = \epsilon.$$
>
>따라서 적분판별법에 의해 $f$는 적분가능하다.


### 정리. 적분가능성과 절댓값

$f\in\mathcal R[a,b]$이면 $|f|\in\mathcal R[a,b]$이고

$$
\left|\int_a^b f(x)\,dx\right|\le \int_a^b |f(x)|\,dx.
$$

> **증명**
>
> 먼저 $|f|$가 적분가능함을 보이자.
>
> 임의의 분할 $P=\{x_0,\dots,x_n\}$을 잡고, 각 소구간 $I_i=[x_{i-1},x_i]$에 대해
>
> $$
> M_i=\sup_{x\in I_i}f(x),\qquad m_i=\inf_{x\in I_i}f(x) \\ 
M_i^*=\sup_{x\in I_i}|f(x)|,\qquad m_i^*=\inf_{x\in I_i}|f(x)|
> $$
>
> 를 정의하자.
>
> 임의의 $x,y\in I_i$에 대하여 역삼각부등식에 의해
>
> $$
> \bigl||f(x)|-|f(y)|\bigr|\le |f(x)-f(y)|
> $$
>
> 그런데 $x,y\in I_i$이므로 $f(x),f(y)\in[m_i,M_i]$이고, 따라서 $|f(x)-f(y)|\le M_i-m_i$ 이므로
>
> $$
> \bigl||f(x)|-|f(y)|\bigr|\le M_i-m_i
> $$
>
> 양변의 supremum을 취하면
>
> $$
> M_i^*-m_i^*\le M_i-m_i.
> $$
>
> 이제 각 소구간에 대해 이를 곱하고 합하면
>
> $$
> U(P,|f|)-L(P,|f|)\le U(P,f)-L(P,f).
> $$
>
> $f\in\mathcal R[a,b]$이므로, 적분판별법에 의해 임의의 $\varepsilon>0$에 대하여 어떤 분할 $P$가 존재하여
>
> $$
> U(P,f)-L(P,f)<\varepsilon.
> $$
>
> 따라서
>
> $$
> U(P,|f|)-L(P,|f|)<\varepsilon.
> $$
>
> 다시 적분판별법에 의해 $|f|\in\mathcal R[a,b]$이다.
>
> 다음으로 절대값 부등식을 보이자.
>
> $$
> -|f(x)|\le f(x)\le |f(x)|
> $$
>
> 이므로, 적분의 단조성을 적용하면
>
> $$
> -\int_a^b |f(x)|\,dx\le \int_a^b f(x)\,dx\le \int_a^b |f(x)|\,dx.
> $$
>
> 따라서
>
> $$
> \left|\int_a^b f(x)\,dx\right|\le \int_a^b |f(x)|\,dx.
> $$
>
> 특히, $g=f_n-f$라 두면 $g\in\mathcal R[a,b]$이고, 위 결과를 $g$에 적용하면
>
> $$
> \left|\int_a^b (f_n-f)\right|
> \le \int_a^b |f_n-f|.
> $$
>
> 이 부등식은 균등수렴 함수열의 적분값 수렴 증명에서 바로 쓰인다.

### 정리. 균등수렴 함수열과 적분가능성

$[a,b]$에서 $f_n$이 $f$로 균등수렴하고, 각 $f_n$이 적분가능하다고 가정하자. 이때 $f$는 적분가능하며 다음이 성립한다:

$$ \lim_{n\to \infty}\int_a^b f_n = \int_a^b f$$

일반적으로

$$ \lim_{n\to\infty}\int_a^b f_n(x)\,dx \neq \int_a^b \lim_{n\to\infty} f_n(x)\ dx$$

이지만, 위 정리에서 보인 바와 같이 $f_n$이 $[a,b]$에서 $f$로 균등수렴하면 극한과 적분을 교환할 수 있다.

>**증명**
>
> $f$의 적분가능성은 위 정리에서 보였다. 이제 적분값의 수렴을 보이자. 한 $N$을 택하면, $n\ge N$에 대해
>
> $$
> \|f_n-f\|_\infty < \frac{\varepsilon}{b-a}.
> $$
>
> 그리고
>
> $$
> \left|\int_a^b f_n-\int_a^b f\right|
> =\left|\int_a^b (f_n-f)\right|
> \le \int_a^b |f_n-f|
> \le (b-a)\|f_n-f\|_\infty
> <\varepsilon.
> $$
>
> 따라서
>
> $$
> \lim_{n\to\infty}\int_a^b f_n=\int_a^b f.
> $$
>


# 2. 미적분학의 기본정리 *(Fundamental Theorem of Calculus)*

미분, 적분은 독립적으로 정의되었고, 각각 수학적으로 엄밀한 용어로 기술되었다. 미분은 접선기울기를 찾는 문제였고, 함수 평균변화율의 극한으로 표현된다. 적분은 함수 그래프 아래 넓이를 계산하려는 시도에서 유한합의 상한, 하한을 이용하여 정의했다. 미적분학의 기본정리는 이 둘 사이에 역연산 관계가 있음을 설명한다!

## (1) 제1 기본정리 *(First Fundamental Theorem of Calculus)*
**Def.**  
$f\in\mathcal R[a,b]$일 때 $x\in [a,b]$에 대해  

$$
F(x)=\int_a^x f(t)\,dt
$$

이 함수 $F$를 $[a,b]$에서 $f$의 부정적분(indefinite integral)이라 한다.

**Thm.**  
$f\in\mathcal R[a,b]$이면,
1. $F$는 $[a,b]$에서 균등연속이다.
2. $f$가 $c \in [a,b]$에서 연속이면 $F$는 $c$에서 미분가능하고, $F'(c)=f(c)$
  - 미분과 적분은 서로 역연산 관계

>### 증명
>**1. $F$는 $[a,b]$에서 균등연속이다.**
>
>$f \in \mathcal{R}[a,b]$이므로 $f$는 유계이다. 즉, $|f(x)| \le M$ for some $M > 0$ and all $x \in [a,b]$.  
>임의의 $x, y \in [a,b]$에 대하여 ($x < y$라 가정)
>
>$$
>|F(y) - F(x)| = \left|\int_a^y f(t)\,dt - \int_a^x f(t)\,dt\right| = \left|\int_x^y f(t)\,dt\right|
>$$
>
>적분의 단조성에 의해
>
>$$
>\left|\int_x^y f(t)\,dt\right| \le \int_x^y |f(t)|\,dt \le M(y-x)
>$$
>
>따라서 임의의 $\varepsilon > 0$에 대하여 $\delta = \frac{\varepsilon}{M}$로 선택하면
>
>$$
>|y - x| < \delta \implies |F(y) - F(x)| \le M|y-x| < M \cdot \frac{\varepsilon}{M} = \varepsilon
>$$
>
>이는 $x, y$의 위치에 무관하므로 $F$는 $[a,b]$에서 균등연속이다. (참고: 립쉬츠 함수면 고른연속, $F$가 립쉬츠 함수임)
>
>**2. $f$가 $[a,b]$에서 연속이면 $F'(x) = f(x)$**
>
>$x \in [a,b]$를 고정하자. $f$가 $x$에서 연속이므로, 임의의 $\varepsilon > 0$에 대하여 $\delta > 0$이 존재하여
>
>$$
>|t - x| < \delta \implies |f(t) - f(x)| < \varepsilon
>$$
>
>$0 < |h| < \delta$이고 $x + h \in [a,b]$인 $h$에 대하여
>
>$$
>\frac{F(x+h) - F(x)}{h} = \frac{1}{h}\int_x^{x+h} f(t)\,dt
>$$
>
>($h < 0$인 경우 $\int_x^{x+h} = -\int_{x+h}^x$로 처리)
>
>$h > 0$인 경우를 고려하면
>
>$$
>\left|\frac{F(x+h) - F(x)}{h} - f(x)\right| = \left|\frac{1}{h}\int_x^{x+h} f(t)\,dt - f(x)\right|\\
>= \left|\frac{1}{h}\int_x^{x+h} (f(t) - f(x))\,dt\right| \le \frac{1}{h}\int_x^{x+h} |f(t) - f(x)|\,dt
>$$
>
>$|t - x| \le h < \delta$이므로 $|f(t) - f(x)| < \varepsilon$. 따라서
>
>$$
>\left|\frac{F(x+h) - F(x)}{h} - f(x)\right| < \frac{1}{h} \cdot \varepsilon h = \varepsilon
>$$
>
>$h < 0$인 경우도 유사하게 증명된다.
>
>$$\therefore \lim_{h \to 0} \frac{F(x+h) - F(x)}{h} = f(x) = F'(x)$$


## (2) 제2 기본정리 *(Second Fundamental Theorem of Calculus)*
**Def. 역도함수 (Antiderivative)**  

함수 $f$가 구간 $D$에서 정의될 때, $F'(x) = f(x)$ for all $x \in D$를 만족하는 함수 $F$를 $f$의 **역도함수** 또는 **원시함수** *(antiderivative)* 라 한다.

**Thm.**  
$f\in\mathcal R[a,b]$이고 $F: [a, b] \to \mathbb{R}$가 $[a, b]$에서 연속이고 $(a, b)$에서 미분가능하다고 하자.  
$F$가 $f$의 **원시함수** *(antiderivative)* 이면

$$
\int_a^b f(x)dx=F(b)-F(a)
$$

>### 증명
>**$f$의 원시함수가 존재한다고 가정**  
>$F$를 $f$의 원시함수, 즉 $F' = f$라 하자.
>
>**Step 1: 분할을 이용한 근사**  
>$[a,b]$의 분할 $\varphi = \{x_0, x_1, \dots, x_n\}$ (단, $a = x_0 < x_1 < \cdots < x_n = b$)을 택하자.  
>평균값 정리에 의해 각 소구간 $[x_{i-1}, x_i]$에서 어떤 $c_i \in (x_{i-1}, x_i)$가 존재하여
>
>$$
>F(x_i) - F(x_{i-1}) = F'(c_i)(x_i - x_{i-1}) = f(c_i)\Delta x_i
>$$
>
>**Step 2: 텔레스코핑 합 (망원급수)**  
>(텔레스코핑 합 (telescoping sum): 합을 전개하면 중간항들이 서로 소거되어 처음 항과 마지막 항만 남는 합)  
>양변을 $i = 1$부터 $n$까지 더하면
>
>$$
>\sum_{i=1}^n (F(x_i) - F(x_{i-1})) = \sum_{i=1}^n f(c_i)\Delta x_i
>$$
>
>좌변은 텔레스코핑 합이므로
>
>$$
>F(x_n) - F(x_0) = F(b) - F(a)
>$$
>
>**Step 3: 리만합으로의 수렴**  
>각 소구간에서 $m_i \le f(c_i) \le M_i$이므로
>
>$$
>L(\varphi, f) \le \sum_{i=1}^n f(c_i)\Delta x_i \le U(\varphi, f)
>$$
>
>$f \in \mathcal{R}[a,b]$이므로 $\|\varphi\| \to 0$일 때
>
>$$
>L(\varphi, f) \to \int_a^b f(x)\,dx, \quad U(\varphi, f) \to \int_a^b f(x)\,dx
>$$
>
>샌드위치 정리에 의해
>
>$$
>\sum_{i=1}^n f(c_i)\Delta x_i \to \int_a^b f(x)\,dx
>$$
>
>**Step 4: 결론** : Step 2와 Step 3을 결합하면
>
>$$
>\int_a^b f(x)\,dx = F(b) - F(a)
>$$

## (3) 따름정리 *(Corollaries)*
### 치환적분 *(Change of Variables / Substitution Rule)*
$g$는 $[a,b]$에서 연속이고, $(a,b)$에서 미분가능하며, $g'\in\mathcal R[a,b]$, $f$는 $g([a,b])$에서 연속이면

$$
\int_a^b f(g(t))g'(t),dt=\int_{g(a)}^{g(b)} f(x),dx
$$

>**증명**
>
>$$F(x) = \int_{g(a)}^{g(x)} f(t)\,dt$$
>
>라 하자. 제1 기본정리에 의해 $F$는 미분가능하고 $F'(x) = f(g(x))$  
>(*Lemma $f, g \in \mathcal{R}[a,b] \Rightarrow fg\in \mathcal{R}[a,b]$ 5장 참고자료 pdf)
>
>연쇄법칙(chain rule)을 적용하면, $G(t) = F(g(t))$에 대해
>
>$$
>G'(t) = F'(g(t)) \cdot g'(t) = f(g(t)) \cdot g'(t)
>$$
>
>따라서 $G$는 $f(g(t))g'(t)$의 원시함수이다.
>
>제2 기본정리에 의해
>
>$$
>\int_a^b f(g(t))g'(t)\,dt = G(b) - G(a) = F(g(b)) - F(g(a))\\
>= \int_{g(a)}^{g(b)} f(x)\,dx - \int_{g(a)}^{g(a)} f(x)\,dx = \int_{g(a)}^{g(b)} f(x)\,dx
>$$

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
$I = \int_{-\infty}^{\infty} e^{-x^2/2}\,dx$를 구하자
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
$f,g$가 $[a,b]$에서 연속, $(a, b)$에서 미분가능하며 $f', g'\in \mathcal{R}[a,b]$이면

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


### 예제 5. $\displaystyle\int \ln x\,dx$ (부정적분)

부분적분을 이용하여 계산한다. $u = \ln x$, $dv = dx$로 두면 $du = \frac{1}{x}dx$, $v = x$

부분적분 공식에 의해

$$\int \ln x\,dx = x\ln x - \int x \cdot \frac{1}{x}\,dx = x\ln x - \int 1\,dx$$

따라서

$$\int \ln x\,dx = x\ln x - x + C = x(\ln x - 1) + C$$

**정적분 예시:** $\displaystyle\int_1^e \ln x\,dx$

위의 부정적분 결과를 이용하면 $\int_1^e \ln x\,dx = [x\ln x - x]_1^e = (e\ln e - e) - (1\ln 1 - 1) \\ = (e \cdot 1 - e) - (0 - 1) = 0 + 1 = 1$

### 예제 ?. 자연로그와 오일러 상수

$x>0$에 대하여

$$L(x)=\int_1^x\frac{1}{t}\,dt$$

로 정의한다.

**(a)** $L(1)$을 구하고, $L$의 미분 가능성 및 $L'(x)$를 구하시오.

**(b)** $L(xy)=L(x)+L(y)$ ($y>0$)임을 보이시오.  
힌트: $y$를 상수 취급하고 $g(x) = L(xy)$를 미분한다.

**(c)** $L(x/y)=L(x)-L(y)$ ($x,y>0$)임을 보이시오.

**(d)** 수열 $\gamma_n=\left(1+\frac{1}{2}+\cdots+\frac{1}{n}\right)-L(n)$이 수렴함을 보이시오.
- 상수 $\gamma = \lim \gamma_n$을 오일러 상수(Euler's constant)라 한다.

**(e)** $L(2)=1-\frac12+\frac13-\frac14+\dots$ 를 수열 $\gamma_{2n}-\gamma_n$을 이용하여 유도하라.

>**풀이**
>
>(a) $L(1)$, 미분 가능성 및 $L'(x)$
>
>정적분의 성질에 의해
>
>$$\boxed{L(1)=\int_1^1\frac{1}{t}\,dt=0}$$
>
>이다.
>
>함수 $t\mapsto 1/t$는 $(0,\infty)$에서 연속이다. 따라서 미적분학의 기본정리에 의해 $L$은 $(0,\infty)$에서 미분 가능하며
>
>$$\boxed{L'(x)=\frac{1}{x}}$$
>
>이다.
>
>---
>
>(b) $L(xy)=L(x)+L(y)$
>
>$y>0$를 고정하고
>
>$$g(x)=L(xy)$$
>
>라고 하자. 연쇄법칙에 의해
>
>$$g'(x)=L'(xy)\cdot y=\frac{1}{xy}\cdot y=\frac{1}{x}$$
>
>한편
>
>$$L'(x)=\frac{1}{x}$$
>
>이므로
>
>$$g'(x)=L'(x)$$
>
>따라서 $g(x)-L(x)$의 도함수는 $0$이므로 어떤 상수 $C$에 대하여
>
>$$g(x)-L(x)=C$$
>
>이다. $x=1$을 대입하면
>
>$$C=g(1)-L(1)=L(y)-0=L(y)$$
>
>그러므로
>
>$$\boxed{L(xy)=L(x)+L(y)}$$
>
>이다.
>
>---
>
>(c) $L(x/y)=L(x)-L(y)$
>
>$x,y>0$일 때
>
>$$x=\frac{x}{y}\cdot y$$
>
>이다. (b)를 적용하면
>
>$$L(x)=L\left(\frac{x}{y}\right)+L(y)$$
>
>따라서
>
>$$\boxed{L\left(\frac{x}{y}\right)=L(x)-L(y)}$$
>
>이다.
>
>특히 $x=1$이면
>
>$$L\left(\frac{1}{y}\right)=-L(y)$$
>
>이다.
>
>---
>
>(d) $(\gamma_n)$의 수렴성
>
>수열을
>
>$$\gamma_n=\left(1+\frac{1}{2}+\cdots+\frac{1}{n}\right)-L(n)$$
>
>으로 정의한다.
>
>**1. $(\gamma_n)$이 감소함을 보인다**
>
>$$\begin{aligned}
>\gamma_{n+1}-\gamma_n
>&=\frac{1}{n+1}-\bigl(L(n+1)-L(n)\bigr)\\
>&=\frac{1}{n+1}-L\left(\frac{n+1}{n}\right)\\
>&=\frac{1}{n+1}-\int_n^{n+1}\frac{1}{t}\,dt
>\end{aligned}$$
>
>$n<t<n+1$이면
>
>$$\frac{1}{t}>\frac{1}{n+1}$$
>
>이므로
>
>$$\int_n^{n+1}\frac{1}{t}\,dt>\int_n^{n+1}\frac{1}{n+1}\,dt=\frac{1}{n+1}$$
>
>따라서
>
>$$\gamma_{n+1}-\gamma_n<0$$
>
>즉,
>
>$$\boxed{(\gamma_n)\text{은 감소수열이다}}$$
>
>**2. $(\gamma_n)$이 아래로 유계임을 보인다**
>
>각 $k=1,\ldots,n$에 대하여 $k\le t\le k+1$이면
>
>$$\frac{1}{t}\le\frac{1}{k}$$
>
>이므로
>
>$$\int_k^{k+1}\frac{1}{t}\,dt\le\frac{1}{k}$$
>
>이를 모두 더하면
>
>$$\int_1^{n+1}\frac{1}{t}\,dt\le\sum_{k=1}^n\frac{1}{k}$$
>
>즉,
>
>$$L(n+1)\le 1+\frac{1}{2}+\cdots+\frac{1}{n}$$
>
>따라서
>
>$$\begin{aligned}
>\gamma_n
>&=\sum_{k=1}^n\frac{1}{k}-L(n)\\
>&\ge L(n+1)-L(n)\\
>&=L\left(\frac{n+1}{n}\right)>0
>\end{aligned}$$
>
>그러므로 $(\gamma_n)$은 $0$을 하한으로 갖는다.
>
>감소하면서 아래로 유계인 수열은 수렴하므로
>
>$$\boxed{(\gamma_n)\text{은 수렴한다}}$$
>
>그 극한
>
>$$\boxed{\gamma=\lim_{n\to\infty}\left(\sum_{k=1}^n\frac{1}{k}-L(n)\right)}$$
>
>을 **오일러 상수**(Euler-Mascheroni constant)라고 한다.
>
>---
>
>(e) 교대급수로 $L(2)$ 나타내기
>
>$\gamma_n\to\gamma$이므로 부분수열도 같은 극한으로 수렴한다.
>
>$$\gamma_{2n}\to\gamma,\qquad \gamma_n\to\gamma$$
>
>따라서
>
>$$\gamma_{2n}-\gamma_n\to 0 \tag{1}$$
>
>한편
>
>$$\begin{aligned}
>\gamma_{2n}-\gamma_n
>&=\left(\sum_{k=1}^{2n}\frac{1}{k}-L(2n)\right)-\left(\sum_{k=1}^n\frac{1}{k}-L(n)\right)\\
>&=\sum_{k=n+1}^{2n}\frac{1}{k}-\bigl(L(2n)-L(n)\bigr)
>\end{aligned}$$
>
>(c)에 의해
>
>$$L(2n)-L(n)=L\left(\frac{2n}{n}\right)=L(2)$$
>
>이므로
>
>$$\gamma_{2n}-\gamma_n=\sum_{k=n+1}^{2n}\frac{1}{k}-L(2) \tag{2}$$
>
>이제 첫째 항을 살펴보면
>
>$$\begin{aligned}
>\sum_{k=n+1}^{2n}\frac{1}{k}
>&=\sum_{k=1}^{2n}\frac{1}{k}-\sum_{k=1}^n\frac{1}{k}\\
>&=\sum_{k=1}^{2n}\frac{1}{k}-2\sum_{k=1}^n\frac{1}{2k}
>\end{aligned}$$
>
>따라서 짝수 번째 항들을 두 번 빼는 형태가 되어
>
>$$\begin{aligned}
>\sum_{k=n+1}^{2n}\frac{1}{k}
>&=1-\frac{1}{2}+\frac{1}{3}-\frac{1}{4}+\cdots+\frac{1}{2n-1}-\frac{1}{2n}
>\end{aligned}$$
>
>그러므로 식 (2)는
>
>$$\gamma_{2n}-\gamma_n=\left(1-\frac{1}{2}+\frac{1}{3}-\frac{1}{4}+\cdots-\frac{1}{2n}\right)-L(2)$$
>
>가 된다. 식 (1)에 의해 $n\to\infty$로 보내면
>
>$$0=\lim_{n\to\infty}\left(1-\frac{1}{2}+\frac{1}{3}-\cdots-\frac{1}{2n}\right)-L(2)$$
>
>따라서
>
>$$\boxed{L(2)=1-\frac{1}{2}+\frac{1}{3}-\frac{1}{4}+\frac{1}{5}-\frac{1}{6}+\cdots}$$
>
>이다. 이는 자연로그를 사용하는 표기에서는
>
>$$\boxed{\ln 2=\sum_{k=1}^{\infty}\frac{(-1)^{k+1}}{k}}$$
>
>라는 결과이다.

# 3. 리만적분의 확장 *(Extensions of the Riemann Integral)*
## (1) 특이적분 *(Improper Integral, 이상적분)*
적분 구간에 개구간이 포함되면 어떻게 정의하는지 살펴보자.  
정상(proper) 적분의 극한값으로 정의한다:

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

③ $f:[a,b]\setminus{c}\to\mathbb R$가 $[a,c)$와 $(c,b]$에서 각각 특이적분가능하면 $f$는 $[a,b]$에서 **특이적분가능**하다고 하고

$$
\int_a^b f
:=\lim_{x\to c^-}\int_a^x f
+\lim_{y\to c^+}\int_y^b f
$$

로 정의한다.

- 즉 한 포인트를 걸러주는게 얼마든지 가능하다

#### 예시
1. 

$$\int_1^\infty x^{-n} dx < +\infty \Leftrightarrow n > 1$$


2. 

$$\int_0^1 x^{\alpha-1} dx < +\infty \Leftrightarrow \alpha > 0$$


3. 

$$\int_{-\infty}^{\infty} \frac{x}{1+x^2} dx$$

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

② ①에서 우변의 극한이 존재하면 각 구간에 대해 $f$는 **특이적분가능**하다고 하고, 이상적분 $\int^{\infty}_b f$는 수렴한다(converge)고 한다.

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

### 이상적분 수렴 판정법
**코시 판정법(Cauchy criterion)**  
이상적분 $\int_a^\infty f$가 수렴하는 필요충분 조건은 임의의 $\epsilon >0$에 대해 다음을 만족하는 $M > a$가 존재하는 것이다.

$$d > c \geq M \Rightarrow \left|\int_c^d f \right| < \epsilon$$

**비교판정법(comparison test)**  
$0 \leq f \leq g$이고 $\int_a^\infty g$가 수렴하면 $\int_a^\infty f$ 도 수렴한다

>**증명**
>
>[코시 판정법]
>
>필요성 $(\Rightarrow)$
>
>$\int_a^\infty f=L$ 이라고 하자. 즉, $\lim_{b\to\infty}F(b)=L,\ F(b)=\int_a^b f$ 이다. 임의의 $\epsilon>0$을 택한다. 극한의 정의에 의해 어떤 $M>a$가 존재하여 $b\ge M$이면
>
>$$
>|F(b)-L|<\frac{\epsilon}{2}
>$$
>
>이다. 이제 $d>c\ge M$이면
>
>$$
>\int_c^d f=F(d)-F(c)
>$$
>
>이므로 삼각부등식에 의해
>
>$$
>\left|\int_c^d f\right|=|F(d)-F(c)|
>\le |F(d)-L|+|F(c)-L|
><\frac{\epsilon}{2}+\frac{\epsilon}{2}=\epsilon.
>$$
>
>따라서 코시 조건이 성립한다.
>
>충분성 $(\Leftarrow)$
>
>이제 임의의 $\epsilon>0$에 대해 어떤 $M>a$가 존재하여
>
>$$
>d>c\ge M\ \Rightarrow\ \left|\int_c^d f\right|<\epsilon
>$$
>
>이라고 가정한다. 힌트에 따라 수열
>
>$$
>a_n=\int_a^{a+n}f
>$$
>
>을 생각하자. 충분히 큰 $m>n$에 대하여 $a+n\ge M$이면
>
>$$
>|a_m-a_n|=
>\left|\int_a^{a+m}f-\int_a^{a+n}f\right|
>=\left|\int_{a+n}^{a+m}f\right|<\epsilon.
>$$
>
>따라서 $(a_n)$은 코시수열이다. $\mathbb{R}$의 완비성에 의해 어떤 $L\in\mathbb{R}$이 존재하여 $a_n\to L$ 이다.
>
>이제 $F(b)\to L$임을 보이자. 임의의 $\epsilon>0$이 주어졌다고 하자. 충분히 큰 $N$을 택하여
>
>$$
>a+N\ge M,\qquad |a_N-L|<\frac{\epsilon}{2}
>$$
>
>가 되게 한다. 코시 조건을 $\epsilon/2$에 적용하면, $b>a+N$일 때
>
>$$
>\begin{aligned}
>|F(b)-L|
>&\le |F(b)-a_N|+|a_N-L|\\
>&=\left|\int_{a+N}^b f\right|+|a_N-L|\\
>&<\frac{\epsilon}{2}+\frac{\epsilon}{2}=\epsilon.
>\end{aligned}
>$$
>
>따라서 $\lim_{b\to\infty}F(b)=L$ 이므로 $\int_a^\infty f$ 가 수렴한다.
>
>---
>[비교판정법]
>
>코시 판정법에 따라 임의의 $\epsilon>0$에 대해 어떤 $M>a$가 존재하여 $d>c\ge M$이면 $\int_c^d g(t)\,dt<\epsilon$ 이다. $g\ge0$이므로 이는
>
>$$
>0\le\int_c^d g(t)\,dt<\epsilon
>$$
>
>이라는 뜻이다. 또한 $0\le f\le g$이므로 적분의 단조성에 의해
>
>$$
>0\le\int_c^d f(t)\,dt\le\int_c^d g(t)\,dt<\epsilon.
>$$
>
>따라서 $\displaystyle\int_c^d f(t)\,dt<\epsilon$이다. 코시 판정법에 의해
>
>$$
>0\le f\le g,\quad \int_a^\infty g(t)\,dt\text{ 수렴}
>\Longrightarrow \int_a^\infty f(t)\,dt\text{ 수렴}
>$$
>
>---
>[절대수렴 판정법]
>
>$\displaystyle\int_a^\infty |f(t)|\,dt$가 수렴한다고 하자. 코시 판정법에 의해 임의의 $\epsilon>0$에 대해 어떤 $M>a$가 존재하여 $d>c\ge M$이면
>
>$$
>\int_c^d |f(t)|\,dt<\epsilon
>$$
>
>이다. 적분의 삼각부등식에 의해
>
>$$
>\left|\int_c^d f(t)\,dt\right| \le\int_c^d |f(t)|\,dt<\epsilon.
>$$
>
>따라서 $\displaystyle\int_a^\infty f(t)\,dt$도 코시 조건을 만족하므로 수렴한다.


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

이고 다음이 성립한다
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

## 적분 기호 속 미분
$f(x,t)$가 모든 $a \leq x \leq b, c\leq t \leq d$에 대해 정의되는 이변수 함수라 하자. $f$의 정의역은 $\mathbb R^2$의 직사각형 $D$다.

$f$가 $D$의 점$(x_0, t_0)$에서 연속이라는 것의 판단은 거리(distance)를 유클리드 거리로 바꿔 확인한다.

### 정의. 이변수 함수의 연속
모든 $\epsilon >0$에 대해 다음을 만족하는 $\delta >0$이 존재하면, 함수 $f: D \to \mathbb R$는 점 $(x_0, t_0)$에서 연속이라 한다.

$$\|(x,t)-(x_0,t) \| < \delta \Rightarrow |f(x,t)-f(x_0,t_0)| < \epsilon $$

### 예제.
함수 $f:D\to\mathbb R,\quad D=[a,b]\times[c,d]$ 가 직사각형 $D$에서 연속이라고 하자. 모든 $x\in[a,b]$에 대해

$$
F(x)=\int_c^d f(x,t)\,dt
$$

가 잘 정의되는 이유는? 

임의의 $x\in[a,b]$를 하나 고정한다. 그러면 $g_x(t)=f(x,t),\ t\in[c,d]$ 라는 일변수 함수를 얻는다. $f$가 $D$에서 연속이므로, $x$를 고정했을 때 $g_x(t)=f(x,t)$는 닫힌 유계구간 $[c,d]$에서 $t$에 대한 연속함수다. 

닫힌 유계구간에서 연속인 함수는 리만 적분 가능하므로 $\int_c^d f(x,t)\,dt$ 가 존재한다. 이 결론은 모든 $x\in[a,b]$에 대해 성립한다.

따라서

$$
F(x)=\int_c^d f(x,t),dt
$$

는 모든 $x\in[a,b]$에서 잘 정의된다.

### 정리 8.4.5

$f(x,t)$ 가 콤팩트한 직사각형 $D=[a,b]\times[c,d]$ 에서 연속이면 $F(x)=\int_c^d f(x,t),dt$ 는 $[a,b]$에서 고른 연속이다.

**증명**

1. $f$의 고른 연속성

집합 $D$는 $\mathbb R^2$의 닫힌 유계집합이므로 콤팩트하다. 콤팩트 집합에서 연속인 함수는 고른 연속이므로, $f$는 $D$에서 고른 연속이다.

임의의 $\varepsilon>0$이 주어졌다고 하자. $d>c$라고 하면 $f$의 고른 연속성에 의해 어떤 $\delta>0$가 존재하여 $|(x,t)-(y,s)|<\delta$ 이면 $|f(x,t)-f(y,s)| < \frac{\varepsilon}{d-c}$ 이다.

특히 $s=t$로 놓으면 $|(x,t)-(y,t)| =\sqrt{(x-y)^2+(t-t)^2} =|x-y|$ 따라서 $|x-y|<\delta$이면 모든 $t\in[c,d]$에 대해

$$
|f(x,t)-f(y,t)| < \frac{\varepsilon}{d-c}
$$

이다. 여기서 중요한 점은 하나의 $\delta$가 모든 $x,y,t$에 공통으로 적용된다는 것이다.

2. $F(x)-F(y)$ 평가

$$
\begin{aligned}
|F(x)-F(y)|
&=
\left|
\int_c^d f(x,t),dt - \int_c^d f(y,t),dt
\right|
&=
\left|
\int_c^d
\bigl(f(x,t)-f(y,t)\bigr),dt
\right|\\
&\le
\int_c^d
|f(x,t)-f(y,t)|,dt.
\end{aligned}
$$

$|x-y|<\delta$이면

$$
\begin{aligned}
|F(x)-F(y)|
&<
\int_c^d\frac{\varepsilon}{d-c},dt\
&=
\frac{\varepsilon}{d-c}(d-c)\
&=\varepsilon.
\end{aligned}
$$

따라서

$$
|x-y|<\delta
\quad\Longrightarrow\quad
|F(x)-F(y)|<\varepsilon
$$

이며, $\delta$는 $x,y$의 위치와 무관하다. 그러므로 $f$는 $[a,b]$에서 고른 연속이다.

$$
f\in C(D)
\quad\Longrightarrow\quad
F(x)=\int_c^d f(x,t),dt
\text{는 }[a,b]\text{에서 고른 연속이다}
$$

### 정리 8.4.6. 적분기호 속의 미분

함수 $f(x,t)$ 가 $x$에 대해 미분가능하고, 편도함수 $f_x(x,t)=\frac{\partial f}{\partial x}(x,t)$ 가 $D=[a,b]\times[c,d]$ 에서 연속이라고 가정한다. 다음 함수를 정의한다.

$$
F(x)=\int_c^d f(x,t),dt.
$$

이 함수는 미분가능하며, 그 도함수는 

$$
F'(x)=\int_c^d f_x(x,t),dt
$$

이다.

- 이 정리를 활용하여 적분 기호 속을 미분할 수 있게 된다.

**증명**

1. 차분몫 정리

$x\in(a,b)$를 고정하고 $z\neq x$라고 하자. 그러면

$$
\begin{aligned}
\frac{F(z)-F(x)}{z-x}
&=
\frac1{z-x}
\left(
\int_c^d f(z,t),dt-\int_c^d f(x,t),dt
\right)\
&=
\int_c^d
\frac{f(z,t)-f(x,t)}{z-x},dt.
\end{aligned}
$$

따라서

$$
\begin{aligned}
&\frac{F(z)-F(x)}{z-x}
-\int_c^d f_x(x,t),dt\
&\qquad=
\int_c^d
\left[
\frac{f(z,t)-f(x,t)}{z-x}
-f_x(x,t)
\right]dt.
\end{aligned}
$$

그러므로 적분의 삼각부등식에 의해

$$
\begin{aligned}
\left|
\frac{F(z)-F(x)}{z-x}
-\int_c^d f_x(x,t),dt
\right|
\le
\int_c^d
\left|
\frac{f(z,t)-f(x,t)}{z-x}
-f_x(x,t)
\right|dt.
\end{aligned}
$$

이 우변을 0에 가깝게 만들면 된다.

2. 평균값정리 적용

각각의 고정된 $t\in[c,d]$에 대해 일변수 함수 $u\longmapsto f(u,t)$ 를 생각한다. 평균값정리에 의해 $x$와 $z$ 사이에 어떤 점 $\xi_t$가 존재하여

$$
\frac{f(z,t)-f(x,t)}{z-x}
=f_x(\xi_t,t)
$$

이다. $\xi_t$는 $t$에 따라 달라질 수 있다.

따라서

$$
\left|
\frac{f(z,t)-f(x,t)}{z-x} -f_x(x,t)
\right| = |f_x(\xi_t,t)-f_x(x,t)|.
$$

3. $f_x$의 고른 연속성 이용

$f_x$는 콤팩트 집합 $D$에서 연속이므로 $D$에서 고른 연속이다. 임의의 $\varepsilon>0$이 주어졌다고 하자. $f_x$의 고른 연속성에 의해 어떤 $\delta>0$가 존재하여 $|(u,t)-(x,t)|<\delta$ 이면 $|f_x(u,t)-f_x(x,t)| < \frac{\varepsilon}{d-c}$ 이다.

이제 $0<|z-x|<\delta$ 라고 하자. $\xi_t$는 $x$와 $z$ 사이에 있으므로 $|\xi_t-x|\le |z-x|<\delta.$ 따라서 $|(\xi_t,t)-(x,t)| =|\xi_t-x| <\delta$ 이고, 모든 $t\in[c,d]$에 대해

$$
|f_x(\xi_t,t)-f_x(x,t)| < \frac{\varepsilon}{d-c}
$$

이다.

4. 차분몫의 극한

앞의 부등식을 사용하면

$$
\begin{aligned}
&\left|
\frac{F(z)-F(x)}{z-x}
-\int_c^d f_x(x,t),dt
\right|\
&\le
\int_c^d
|f_x(\xi_t,t)-f_x(x,t)|,dt\
&<
\int_c^d\frac{\varepsilon}{d-c},dt\
&=\varepsilon.
\end{aligned}
$$

따라서 $z\to x$ 일 때

$$
\frac{F(z)-F(x)}{z-x}
\longrightarrow
\int_c^d f_x(x,t),dt.
$$

그러므로 $f$는 $x$에서 미분 가능하고

$$
F'(x)=\int_c^d f_x(x,t),dt.
$$

$x\in(a,b)$가 임의의 점이었으므로

$$
F'(x)=\frac{d}{dx}\int_c^d f(x,t),dt
=\int_c^d\frac{\partial f}{\partial x}(x,t),dt
$$
이다. 끝점 $a,b$에서는 각각 오른쪽 미분과 왼쪽 미분으로 같은 결론을 얻는다.

핵심은 $f_x$의 연속성 자체보다, 콤팩트한 직사각형 $D$ 위에서 얻어지는 $f_x$의 고른 연속성이다. 이것이 모든 $t\in[c,d]$에 대해 차분몫의 오차를 하나의 $\delta$로 동시에 제어하게 한다.

---

어떤 집합 $A \subseteq \mathbb R$에서 $x$를 고정하자. $x$에 대해 다음 식의 극한이 존재하면 함수 $F(x)$를 정의할 수 있다:

$$F(x) = \int_c^\infty f(x,t)\ dt = \lim_{d\to \infty}\int_c^d f(x,t)\ dt$$

위 등식은 점별로(pointwise) 서술되어있다. $x \in A$와 $\epsilon >0$이 주어지면 $d \ge M$일때 다음을 만족하는 $M$을 착을 수 있다.

$$ \left|F(x)-\int_c^d f(x,t)\ dt \right| < \epsilon$$

이를 무한으로 확장시켜 이상적분에 대해 정의하면 다음과 같다:
### 정의. 이상적분의 고른 수렴
$D = \{(x,t): x\in A,  c \leq t\}$에서 정의된 $f(x,t)$가 주어질 때 모든 $x\in A$에 대해 $F(x)=\int_c^\infty f(x,t)\ dt$가 존재한다 가정하자. 모든 $\epsilon>0$에 대해 $M>c$이 존재하여 모든 $d\ge M$와 $x \in A$에 대해 다음이 성립할 때, 이상적분이 $A$에서 $F(x)$로 고르게 수렴한다 (converges uniformly)고 한다.

$$ \left|F(x)-\int_c^d f(x,t)\ dt \right| < \epsilon$$

핵심은 $M$을 $x$와 무관하게 선택할 수 있어야 한다는 것이다.

#### 예제. 문제 15
다음 이상적분을 생각한다.

$$
F(x)=\int_0^\infty e^{-xt},dt=\frac1x,\qquad x>0.
$$

1. 이상적분이 집합 $[1/2,\infty)$에서 $1/x$로 고르게 수렴함을 보여라.
2. 집합 $(0,\infty)$에서도 고르게 수렴하는지 판정하라.

1. $[1/2,\infty)$에서의 고른 수렴 증명

$F(x)=\int_0^\infty e^{-xt},dt=\frac1x$ 이다. 유한한 $d>0$에 대해서는

$$
\begin{aligned}
\int_0^d e^{-xt},dt
&=\left[-\frac1x e^{-xt}\right]_0^d\
&=\frac{1-e^{-xd}}x.
\end{aligned}
$$

따라서 절단 적분과 이상적분 사이의 오차는

$$
\begin{aligned}
\left| \frac1x-\int_0^d e^{-xt},dt \right|
&= 
\left| \frac1x-\frac{1-e^{-xd}}x \right|\
&=\frac{e^{-xd}}x.
\end{aligned}
$$

이제 $x\ge1/2$이면 $\frac1x\le2$ 이고 $e^{-xd}\le e^{-d/2}$ 이므로 $\frac{e^{-xd}}x\le2e^{-d/2}$.  
임의의 $\varepsilon>0$이 주어졌다고 하자. $M$을 충분히 크게 택하여 $2e^{-M/2}<\varepsilon$ 이 되게 한다. 예를 들어 $0<\varepsilon<2$이면 $M>2\log\frac2\varepsilon$ 로 택할 수 있다.

그러면 모든 $d\ge M$와 $x\ge1/2$에 대해

$$
\begin{aligned}
\left|
\frac1x-\int_0^d e^{-xt},dt
\right|
&\le2e^{-d/2}\
&\le2e^{-M/2}\
&<\varepsilon.
\end{aligned}
$$

여기서 $M$은 $x$와 무관하다. 따라서

$$
\boxed{
\int_0^\infty e^{-xt},dt
\text{는 }[1/2,\infty)\text{에서 }\frac1x\text{로 고르게 수렴한다.}
}
$$

2. $(0,\infty)$에서의 고른 수렴 여부 판단.

오차는 $R_d(x)=\frac{e^{-xd}}x$ 이다.

고정된 $d>0$에 대하여 $x\to0^+$이면 $e^{-xd}\to1,\quad \frac1x\to\infty$ 이므로 $R_d(x)=\frac{e^{-xd}}x\to\infty.$

따라서 모든 유한한 $d>0$에 대해

$$
\sup_{x>0} \left| \frac1x-\int_0^d e^{-xt},dt \right| =\infty.
$$

특히 이 상한은 $d\to\infty$일 때 $0$으로 수렴하지 않는다. 따라서

$$
\boxed{
\int_0^\infty e^{-xt},dt
\text{는 }(0,\infty)\text{에서 고르게 수렴하지 않는다.}
}
$$

각각의 고정된 $x>0$에서는 수렴하지만, $x$가 $0$에 가까워질수록 $e^{-xt}$의 감소가 느려진다.

### 정리. 이상적분에 대한 바이어슈트라스 $M$-판정법
모든 $x\in A$와 $t\ge a$에 대해 $|f(x,t)|\le g(t)$ 이고 $\int_a^\infty g(t),dt$ 가 수렴하면

$$
\int_a^\infty f(x,t),dt
$$

가 $A$에서 고르게 수렴함을 보여라.

>**증명**  
>
>먼저 각각의 고정된 $x\in A$에 대해 $|f(x,t)|\le g(t)$ 이다. $\int_a^\infty g(t),dt$가 수렴하므로 비교판정법에 의해 $\int_a^\infty |f(x,t)|,dt$ 가 수렴한다. 따라서 각각의 $x\in A$에 대해 $\int_a^\infty f(x,t),dt$ 가 절대수렴한다. 다음과 같이 정의한다. $F(x)=\int_a^\infty f(x,t),dt.$
>
>임의의 $d>a$에 대해 $F(x)-\int_a^d f(x,t),dt = \int_d^\infty f(x,t),dt$ 이므로
>
>$$
>\left| F(x)-\int_a^d f(x,t),dt \right| = \left| \int_d^\infty f(x,t),dt \right|\ \le \int_d^\infty |f(x,t)|,dt\ \le \int_d^\infty g(t),dt.
>$$
>
>$\int_a^\infty g(t),dt$가 수렴하므로 임의의 $\varepsilon>0$에 대해 어떤 $M>a$가 존재하여 $d\ge M \Rightarrow \int_d^\infty g(t),dt<\varepsilon$ 이다. 따라서 모든 $d\ge M$와 모든 $x\in A$에 대해
>
>$$
>\left|
>F(x)-\int_a^d f(x,t),dt
>\right|
>\le
>\int_d^\infty g(t),dt
><\varepsilon.
>$$
>
>여기서 $M$은 $x$와 무관하다. 그러므로
>
>$$
>\boxed{
>\int_a^\infty f(x,t),dt
>\text{는 }A\text{에서 고르게 수렴한다.}
>}
>$$

### 정리 8.4.8
함수 $f(x,t)$가 $D=\{(x,t):a\le x\le b,\ c\le t\}$ 에서 연속이라고 하자.   이상적분

$$
F(x)=\int_c^\infty f(x,t),dt
$$

가 $[a,b]$에서 고르게 수렴하면 $F$는 $[a,b]$에서 고른 연속이다.

>**증명**  
>
>자연수 $n$에 대해 절단 적분함수 $F_n(x)=\int_c^{c+n}f(x,t),dt$ 를 정의한다.
>
>1. 각각의 $F_n$은 연속이다
>
>고정된 $n$에 대해 $f$는 콤팩트 직사각형 $[a,b]\times[c,c+n]$ 에서 연속이다. 정리 8.4.5에 의해 $F_n(x)=\int_c^{c+n}f(x,t),dt$ 는 $[a,b]$에서 연속이다. 실제로 정리 8.4.5는 $F_n$이 고른 연속임도 보장한다.
>
>2. $F_n$은 $F$로 고르게 수렴한다
>
>가정에 의해 이상적분 $F(x)=\int_c^\infty f(x,t),dt$ 가 $[a,b]$에서 고르게 수렴한다. 따라서 모든 $\varepsilon>0$에 대해 어떤 $M>c$가 존재하여, 모든 $d\ge M$와 모든 $x\in[a,b]$에 대해
>
>$$
>\left| F(x)-\int_c^d f(x,t),dt \right|<\varepsilon
>$$
>
>이다. 충분히 큰 $n$에 대해 $c+n\ge M$이므로 $|F(x)-F_n(x)|<\varepsilon$ 가 모든 $x\in[a,b]$에 대해 성립한다. 따라서 $F_n\to F$ 는 $[a,b]$에서 고른 수렴이다.
>
>3. 고른 극한의 연속성
>
>각 $F_n$은 연속이고 $F_n\to F$가 고르게 수렴하므로, 연속함수의 고른 극한에 관한 정리에 의해 $F$는 $[a,b]$에서 연속이다.
>
>또한 $[a,b]$는 콤팩트 집합이다. 콤팩트 집합에서 연속인 함수는 고른 연속이므로 $F$는 $[a,b]$에서 고른 연속이다.
>
>따라서
>
>$$
>\boxed{
>f\text{가 }D\text{에서 연속이고 }
>\int_c^\infty f(x,t),dt
>\text{가 고르게 수렴하면, }F\text{는 }[a,b]\text{에서 고른 연속이다.}
>}
>$$
>
>단순한 점별수렴만으로는 연속성이 극한함수에 전달되지 않는다. 고른 수렴이 연속성을 보존하는 핵심 조건이다.

### 정리 8.4.9: 이상적분의 적분기호 속 미분

함수 $f(x,t)$가 $D=\{(x,t):a\le x\le b,\ c\le t\}$ 에서 연속이고, 각 $x\in[a,b]$에 대해 $F(x)=\int_c^\infty f(x,t),dt$ 가 존재한다고 하자. 편도함수 $f_x(x,t)=\frac{\partial f}{\partial x}(x,t)$ 가 존재하고 $D$에서 연속이라고 하자. 또한 이상적분 $\int_c^\infty f_x(x,t),dt$ 가 $[a,b]$에서 고르게 수렴한다고 하자.

그러면 $F$는 미분 가능하고

$$
\boxed{
F'(x)=\int_c^\infty f_x(x,t),dt
}
$$

가 성립한다.

**증명**

다음 함수를 정의한다.

$$
G(x)=\int_c^\infty f_x(x,t),dt.
$$

가정에 의해 이 이상적분은 $[a,b]$에서 고르게 수렴한다.

1. 절단 적분함수의 미분

유한한 $d>c$에 대해

$$
F_d(x)=\int_c^d f(x,t),dt
$$

라고 정의한다. $f$와 $f_x$가 콤팩트 직사각형 $[a,b]\times[c,d]$ 에서 연속이므로 정리 8.4.6에 의해 적분기호 안에서 미분할 수 있다.

$$
F_d'(x)=\int_c^d f_x(x,t),dt.
$$

다음과 같이 놓는다.

$$
G_d(x)=\int_c^d f_x(x,t),dt.
$$

그러면 $F_d'(x)=G_d(x)$이다.

2. $G$의 연속성

각각의 $G_d$는 정리 8.4.5에 의해 $x$에 대한 연속함수다. 또한 가정에 의해 $G_d(x)=\int_c^d f_x(x,t),dt$ 는 $G(x)=\int_c^\infty f_x(x,t),dt$ 로 $[a,b]$에서 고르게 수렴한다. 따라서 연속함수의 고른 극한에 관한 정리에 의해 $G$는 $[a,b]$에서 연속이다. 이는 정리 8.4.8을 $f_x$에 적용한 결과이기도 하다.

3. $F(y)-F(x)$ 계산

임의의 $x,y\in[a,b]$를 고정한다. 각각의 유한한 $d>c$에 대해 미적분학의 기본정리를 적용하면

$$
F_d(y)-F_d(x) = \int_x^y F_d'(s),ds = \int_x^yG_d(s),ds.
$$

이를 원래의 적분으로 쓰면

$$
\int_c^d f(y,t),dt - \int_c^d f(x,t),dt = \int_x^y
\left( \int_c^d f_x(s,t),dt \right)ds.
$$

이제 $d\to\infty$로 보낸다. 각 $x,y$에서 이상적분이 존재하므로 $F_d(y)\to F(y), \quad F_d(x)\to F(x).$ 따라서 왼쪽은 $F(y)-F(x)$ 로 수렴한다.

한편 $G_d\to G$가 $[a,b]$에서 고르게 수렴하므로 극한과 유한구간 적분의 순서를 교환할 수 있다.

$$
\lim_{d\to\infty}\int_x^yG_d(s),ds = \int_x^yG(s),ds.
$$

따라서

$$
F(y)-F(x)=\int_x^yG(s),ds.
$$

즉,

$$
F(y)=F(x)+\int_x^yG(s),ds.
$$

4. 미분

함수 $G$는 연속이므로 미적분학의 기본정리에 의해

$$
\frac{d}{dy}\int_x^yG(s),ds=G(y).
$$

따라서

$$
F'(y)=G(y) = \int_c^\infty f_x(y,t),dt.
$$

변수 이름을 다시 $x$로 바꾸면

$$
\boxed{
\frac{d}{dx}\int_c^\infty f(x,t),dt = \int_c^\infty
\frac{\partial f}{\partial x}(x,t),dt
}
$$

를 얻는다.

이 결론은 $x\in(a,b)$에서 보통의 양쪽 미분으로 성립한다. 끝점 $a,b$에서는 각각 오른쪽 미분과 왼쪽 미분으로 해석한다.



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
   = \left(\int_a^b f\right)\times \left(\int_a^b g\right)
   $$

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
\int_0^3 x^2 e^x\ dx = [x^2 e^x]_0^3 - ([2xe^x]_0^3 - [2e^x]_0^3)\\
= [x^2 e^x - 2xe^x + 2e^x]_0^3 = [e^x(x^2 - 2x + 2)]_0^3\\
= e^3(9 - 6 + 2) - e^0(0 - 0 + 2) = 5e^3 - 2
$$

**(2)** $\displaystyle\int_2^3 (x-1)\ d(x^2+2)$

$f(x) = x-1$, $\alpha(x) = x^2+2$로 두면 $\alpha'(x) = 2x$이다.

정리에 의해

$$
\int_2^3 (x-1)\ d(x^2+2) = \int_2^3 (x-1) \cdot 2x\ dx\\
= \int_2^3 2x(x-1)\ dx = \int_2^3 (2x^2 - 2x)\ dx\\
= \left[\frac{2x^3}{3} - x^2\right]_2^3\\
= \left(\frac{2 \cdot 27}{3} - 9\right) - \left(\frac{2 \cdot 8}{3} - 4\right)\\
= (18 - 9) - \left(\frac{16}{3} - 4\right) = 9 - \frac{4}{3} = \frac{23}{3}
$$
