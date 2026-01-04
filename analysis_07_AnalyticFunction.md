해석함수(Analytic Function)  

# 1. 테일러급수 전개 *(Taylor Series Expansion)*
## Def. [해석함수] *(Analytic Function)*
$\delta>0$가 존재하여 구간 $(c-\delta,c+\delta)$에서
$$
f(x)=\sum_{n=0}^{\infty} a_n (x-c)^n
$$
으로 표현될 수 있으면, 함수 $f$를 **$x=c$에서 해석적** *(analytic at $c$)* 이라 한다.

또한 함수 $f$가 어떤 열린구간 $I$의 모든 점에서 해석적이면,
$f$를 **$I$에서의 해석함수** *(analytic function on $I$)* 라 한다.

## Thm. [테일러급수 전개] *(Taylor Series Expansion Theorem)*
함수 $f$가 열린구간 $I$에서 해석함수이면,
$f$는 무한번 미분가능하고, 임의의 $c\in I$에 대하여
$$
f(x)=\sum_{n=0}^{\infty}\frac{f^{(n)}(c)}{n!}(x-c)^n
\quad (|x-c|<\delta)
$$
를 만족하는 $\delta>0$가 존재한다.

* 우변의 멱급수를 **해석함수 $f$의 테일러급수**라 한다.
* 특히 $c=0$인 경우 이를 **맥클로린급수** *(Maclaurin series)* 라 한다.

### 증명
함수 $f$가 $x=c$에서 해석적이면, 정의에 의해
$$
f(x)=\sum_{n=0}^{\infty} a_n (x-c)^n
$$
로 표현된다.

멱급수는 수렴반경 내에서 항별미분이 가능하므로,
$$
f'(x)=\sum_{n=1}^{\infty} na_n (x-c)^{n-1}
$$
$$
f''(x)=\sum_{n=2}^{\infty} n(n-1)a_n (x-c)^{n-2}
$$
$$
\vdots
$$
$$
f^{(k)}(x)=\sum_{n=k}^{\infty} n(n-1)\cdots(n-k+1)a_n (x-c)^{n-k}
$$

$x=c$를 대입하면,
$$
f(c)=a_0,\quad f'(c)=a_1,\quad f''(c)=2a_2,\quad \ldots,\quad f^{(n)}(c)=n!a_n
$$

따라서
$$
a_n=\frac{f^{(n)}(c)}{n!}
$$

이를 원래 급수에 대입하면
$$
f(x)=\sum_{n=0}^{\infty}\frac{f^{(n)}(c)}{n!}(x-c)^n
$$
를 얻는다. $\square$

## Thm. [테일러 정리] *(Taylor's Theorem)*
함수 $f$가 열린구간 $(a-h, a+h)$ $(h>0)$에서 $n$번 미분가능하고,
$n$차 도함수 $f^{(n)}$이 연속함수이면,
그 구간 내 모든 $x$에 대해 다음 등식이 성립한다.

$$
f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots + \frac{f^{(n-1)}(a)}{(n-1)!}(x-a)^{n-1} + R_n(x, a)
$$

여기서 $R_n(x, a)$는 **나머지항** *(remainder term)* 또는 **오차항**이며, 다음과 같이 표현된다.

**적분 형태:**
$$
R_n(x, a) = \frac{1}{(n-1)!}\int_0^1 f^{(n)}(a+u(x-a))(1-u)^{n-1}du\cdot(x-a)^n
$$

**라그랑주 형태** *(Lagrange form)*:
$$
R_n(x, a) = \frac{f^{(n)}(\xi)}{n!}(x-a)^n
$$
여기서 $\xi$는 $a$와 $x$ 사이의 어떤 값, 즉 $|\xi - a| < |x-a|$를 만족한다.

### 참고
* $n\to\infty$일 때 $R_n(x,a)\to 0$이면 $f$는 테일러급수로 완전히 표현된다.
* 라그랑주 나머지항은 오차 추정에 유용하게 사용된다.

# 2. 여러 가지 해석함수의 예 *(Examples of Analytic Functions)*
다음 함수들은 해당 구간에서 멱급수로 전개되며 해석함수이다.
* **기하급수**
  $$
  \frac{1}{x}
  =\sum_{n=0}^{\infty}(1-x)^n
  \quad (0<x<2)
  $$
  $$
  \frac{1}{1-x}
  =\sum_{n=0}^{\infty}x^n
  \quad (-1<x<1)
  $$
  $$
  \frac{1}{1+x}
  =\sum_{n=0}^{\infty}(-1)^n x^n
  \quad (-1<x<1)
  $$

* **제곱근 함수**
  $$
  \sqrt{x}
  =1-\frac{1-x}{2}-\frac{(1-x)^2}{8}-\frac{(1-x)^3}{16}-\cdots
  \quad (0<x<2)
  $$
  $$
  \sqrt{1+x}
  =1+\frac{x}{2}-\frac{x^2}{8}+\frac{x^3}{16}-\frac{5x^4}{128}+\cdots
  \quad (-1<x\le 1)
  $$

* **지수함수**
  $$
  e^x=\sum_{n=0}^{\infty}\frac{x^n}{n!}
  \quad (-\infty<x<\infty)
  $$
  $$
  e^{-x}=\sum_{n=0}^{\infty}\frac{(-1)^n x^n}{n!}
  \quad (-\infty<x<\infty)
  $$

* **로그함수**
  $$
  \ln x
  =\sum_{n=1}^{\infty}\frac{(-1)^{n+1}}{n}(x-1)^n
  \quad (0<x\le 2)
  $$
  $$
  \ln(1+x)
  =\sum_{n=1}^{\infty}\frac{(-1)^{n+1}}{n}x^n
  \quad (-1<x\le 1)
  $$
  $$
  \ln(1-x)
  =-\sum_{n=1}^{\infty}\frac{x^n}{n}
  \quad (-1\le x<1)
  $$

* **삼각함수**
  $$
  \sin x=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}x^{2n+1},
  \quad
  \cos x=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n)!}x^{2n}
  \quad (-\infty<x<\infty)
  $$
  $$
  \tan x=x+\frac{x^3}{3}+\frac{2x^5}{15}+\frac{17x^7}{315}+\cdots
  \quad \left(-\frac{\pi}{2}<x<\frac{\pi}{2}\right)
  $$

* **쌍곡함수**
  $$
  \sinh x=\sum_{n=0}^{\infty}\frac{x^{2n+1}}{(2n+1)!}=x+\frac{x^3}{6}+\frac{x^5}{120}+\cdots
  \quad (-\infty<x<\infty)
  $$
  $$
  \cosh x=\sum_{n=0}^{\infty}\frac{x^{2n}}{(2n)!}=1+\frac{x^2}{2}+\frac{x^4}{24}+\cdots
  \quad (-\infty<x<\infty)
  $$

* **이항급수** *(Binomial Series)*
  $$
  (1+x)^{\alpha}=\sum_{n=0}^{\infty}\binom{\alpha}{n}x^n
  =1+\alpha x+\frac{\alpha(\alpha-1)}{2!}x^2+\frac{\alpha(\alpha-1)(\alpha-2)}{3!}x^3+\cdots
  \quad (-1<x<1)
  $$
  여기서 $\alpha$는 임의의 실수이고, $\binom{\alpha}{n}=\frac{\alpha(\alpha-1)\cdots(\alpha-n+1)}{n!}$

* **역삼각함수**
  $$
  \arctan x=\sum_{n=0}^{\infty}\frac{(-1)^n}{2n+1}x^{2n+1}
  =x-\frac{x^3}{3}+\frac{x^5}{5}-\frac{x^7}{7}+\cdots
  \quad (-1\le x\le 1)
  $$
  $$
  \arcsin x=\sum_{n=0}^{\infty}\frac{(2n)!}{4^n(n!)^2(2n+1)}x^{2n+1}
  =x+\frac{x^3}{6}+\frac{3x^5}{40}+\cdots
  \quad (-1\le x\le 1)
  $$

## 2.1. 기하급수의 증명
$f(x)=\frac{1}{x}$를 $x=1$에서 전개하면,

$$
f^{(n)}(x)=\frac{(-1)^n n!}{x^{n+1}}
$$

$x=1$을 대입하면 $f^{(n)}(1)=(-1)^n n!$

따라서
$$
\frac{1}{x}=\sum_{n=0}^{\infty}\frac{(-1)^n n!}{n!}(x-1)^n=\sum_{n=0}^{\infty}(-1)^n(x-1)^n=\sum_{n=0}^{\infty}(1-x)^n
$$

수렴반경은 $R=1$이므로 $0<x<2$에서 성립한다.

## 2.2. 제곱근 함수의 증명
$f(x)=\sqrt{x}=x^{1/2}$를 $x=1$에서 전개하면,

$$
f(x)=x^{1/2},\quad f'(x)=\frac{1}{2}x^{-1/2},\quad f''(x)=-\frac{1}{4}x^{-3/2}
$$
$$
f^{(n)}(x)=\frac{1}{2}\cdot\frac{-1}{2}\cdot\frac{-3}{2}\cdots\frac{3-2n}{2}x^{(1-2n)/2}
$$

$x=1$을 대입하면
$$
f^{(n)}(1)=\frac{1\cdot(-1)\cdot(-3)\cdots(3-2n)}{2^n}
$$

이항급수 전개를 이용하여
$$
\sqrt{x}=\sum_{n=0}^{\infty}\binom{1/2}{n}(x-1)^n
$$

수렴반경 $R=1$이므로 $0<x<2$에서 성립한다.

## 2.3. 지수함수의 증명
$f(x)=e^x$에 대하여 $f^{(n)}(x)=e^x$

$x=0$을 대입하면 $f^{(n)}(0)=1$

따라서
$$
e^x=\sum_{n=0}^{\infty}\frac{1}{n!}x^n=\sum_{n=0}^{\infty}\frac{x^n}{n!}
$$

비율판정법에 의해 수렴반경 $R=\infty$이므로 모든 실수에서 성립한다.

## 2.4. 로그함수의 증명
$f(x)=\ln x$를 $x=1$에서 전개하면,

$$
f(x)=\ln x,\quad f'(x)=\frac{1}{x},\quad f''(x)=-\frac{1}{x^2}
$$
$$
f^{(n)}(x)=\frac{(-1)^{n-1}(n-1)!}{x^n}
$$

$x=1$을 대입하면 $f^{(n)}(1)=(-1)^{n-1}(n-1)!$

따라서 $(n\geq 1)$
$$
\ln x=\sum_{n=1}^{\infty}\frac{(-1)^{n-1}(n-1)!}{n!}(x-1)^n=\sum_{n=1}^{\infty}\frac{(-1)^{n-1}}{n}(x-1)^n=\sum_{n=1}^{\infty}\frac{(-1)^{n+1}}{n}(x-1)^n
$$

수렴반경 $R=1$이며, $x=2$에서 교대급수로 수렴하므로 $0<x\le 2$에서 성립한다.

## 2.5. 삼각함수의 증명
**사인함수:** $f(x)=\sin x$에 대하여

$$
f^{(n)}(x)=\begin{cases}
\sin x & n\equiv 0\pmod{4}\\
\cos x & n\equiv 1\pmod{4}\\
-\sin x & n\equiv 2\pmod{4}\\
-\cos x & n\equiv 3\pmod{4}
\end{cases}
$$

$x=0$을 대입하면 $f^{(2n)}(0)=0$, $f^{(2n+1)}(0)=(-1)^n$

따라서
$$
\sin x=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}x^{2n+1}
$$

**코사인함수:** $f(x)=\cos x$에 대하여 유사하게

$x=0$을 대입하면 $f^{(2n)}(0)=(-1)^n$, $f^{(2n+1)}(0)=0$

따라서
$$
\cos x=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n)!}x^{2n}
$$

비율판정법에 의해 두 급수 모두 수렴반경 $R=\infty$이다.

## 2.6. 해석함수를 이용한 근삿값 계산
해석함수의 테일러급수를 이용하면 함수값의 근삿값을 계산할 수 있다.

### 예제 1: $e$의 근삿값
$e=e^1$의 값을 구하기 위해 지수함수의 맥클로린급수를 이용한다.
$$
e=\sum_{n=0}^{\infty}\frac{1}{n!}=1+1+\frac{1}{2}+\frac{1}{6}+\frac{1}{24}+\frac{1}{120}+\cdots
$$

처음 6항까지 더하면
$$
e\approx 1+1+0.5+0.167+0.042+0.008=2.717
$$

실제값 $e\approx 2.71828$과 비교하면 오차는 약 $0.001$이다.

### 예제 2: $\sqrt{1.2}$의 근삿값
$\sqrt{1.2}$를 구하기 위해 제곱근 함수를 $x=1$에서 전개한 급수를 이용한다.
$$
\sqrt{x}=1+\frac{1}{2}(x-1)-\frac{1}{8}(x-1)^2-\frac{1}{16}(x-1)^3-\cdots
$$

$x=1.2$를 대입하면 $(x-1)=0.2$이므로
$$
\sqrt{1.2}\approx 1+\frac{1}{2}(0.2)-\frac{1}{8}(0.04)-\frac{1}{16}(0.008)
=1+0.1-0.005-0.0005=1.0945
$$

실제값 $\sqrt{1.2}\approx 1.09545$과 거의 일치한다.

### 예제 3: $\sin(0.1)$의 근삿값
$\sin(0.1)$을 구하기 위해 사인함수의 맥클로린급수를 이용한다.
$$
\sin x=x-\frac{x^3}{6}+\frac{x^5}{120}-\cdots
$$

$x=0.1$을 대입하면
$$
\sin(0.1)\approx 0.1-\frac{(0.1)^3}{6}+\frac{(0.1)^5}{120}
=0.1-\frac{0.001}{6}+\frac{0.00001}{120}
\approx 0.1-0.000167=0.099833
$$

실제값 $\sin(0.1)\approx 0.0998334$과 거의 일치한다.

# 3. 해석함수와 연산 *(Operations on Analytic Functions)*
## Thm. 1. [해석함수의 사칙연산] *(Algebra of Analytic Functions)*
함수 $f,g$가 각각 열린구간 $I,J$에서 해석적이면 다음이 성립한다.

1. $cf,\ f\pm g,\ fg$는 $I\cap J$에서 해석적이다.
2. $g(x_0)\neq 0$ ($x_0 \in I \cap J$)이면 $\dfrac{f}{g}$는 $x=x_0$의 근방에서 해석적이다.

### 증명
**(1) 덧셈, 뺄셈, 곱셈의 경우**

$f$가 $I$에서 해석적이고 $g$가 $J$에서 해석적이면, $I\cap J$의 임의의 점 $c$에 대하여
$$
f(x)=\sum_{n=0}^{\infty} a_n (x-c)^n, \quad g(x)=\sum_{n=0}^{\infty} b_n (x-c)^n
$$
로 표현된다.

멱급수의 선형결합과 곱셈은 수렴반경 내에서 항별로 계산 가능하므로,
$$
(f+g)(x)=\sum_{n=0}^{\infty} (a_n+b_n)(x-c)^n
$$
$$
(f-g)(x)=\sum_{n=0}^{\infty} (a_n-b_n)(x-c)^n
$$
$$
(fg)(x)=\sum_{n=0}^{\infty}\left(\sum_{k=0}^{n}a_k b_{n-k}\right)(x-c)^n
$$

따라서 $cf,\ f\pm g,\ fg$는 $I\cap J$에서 해석적이다.

**(2) 나눗셈의 경우**

$g(x_0)\neq 0$이면 연속성에 의해 $x_0$의 어떤 근방에서 $g(x)\neq 0$이다.

$g(x)=\sum_{n=0}^{\infty} b_n (x-x_0)^n$에서 $b_0=g(x_0)\neq 0$이므로,
$$
\frac{1}{g(x)}=\frac{1}{b_0}\cdot\frac{1}{1+\sum_{n=1}^{\infty}\frac{b_n}{b_0}(x-x_0)^n}
$$

이는 기하급수를 이용하여 멱급수로 전개 가능하며,
$$
\frac{f}{g}=f\cdot\frac{1}{g}
$$
는 멱급수의 곱으로 $x_0$의 근방에서 해석적이다. $\square$

## Thm. 2. [해석함수의 합성] *(Composition of Analytic Functions)*
$f$가 열린구간 $I$에서 해석적이고,
$g$가 열린구간 $J$에서 해석적이며 $f(I)\subset J$이면,
합성함수 $g\circ f$는 $I$에서 해석적이다.

### 증명
$f$가 $x=c\in I$에서 해석적이면
$$
f(x)=\sum_{n=0}^{\infty} a_n (x-c)^n
$$
로 표현되고, $f(c)=a_0$이다.

$g$가 $y=a_0\in J$에서 해석적이면
$$
g(y)=\sum_{m=0}^{\infty} b_m (y-a_0)^m
$$
로 표현된다.

$y=f(x)$를 대입하면
$$
g(f(x))=\sum_{m=0}^{\infty} b_m (f(x)-a_0)^m
=\sum_{m=0}^{\infty} b_m \left(\sum_{n=1}^{\infty} a_n (x-c)^n\right)^m
$$

$f(x)-a_0=\sum_{n=1}^{\infty} a_n (x-c)^n$는 $x=c$에서 멱급수이고,
이를 $m$제곱하여 전개하면 $(x-c)$의 멱급수가 된다.

따라서 $g\circ f$는 $x=c$의 근방에서
$$
(g\circ f)(x)=\sum_{k=0}^{\infty} c_k (x-c)^k
$$
의 형태로 표현되므로 $x=c$에서 해석적이다.

$I$의 모든 점에서 같은 논리가 성립하므로 $g\circ f$는 $I$에서 해석적이다. $\square$

### 
예제: $e^{x^2}$의 전개

지수함수 $e^x=\sum_{n=0}^{\infty}\frac{x^n}{n!}$와 $f(x)=x^2$의 합성을 이용하면

$$
e^{x^2}=(e\circ f)(x)=\sum_{n=0}^{\infty}\frac{(x^2)^n}{n!}=\sum_{n=0}^{\infty}\frac{x^{2n}}{n!}
$$

이는 모든 실수에서 수렴하므로 $e^{x^2}$는 $\mathbb{R}$에서 해석적이다.

**예제: $\sin(x^2)$의 전개**

사인함수 $\sin x=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}x^{2n+1}$와 $f(x)=x^2$의 합성을 이용하면

$$
\sin(x^2)=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}(x^2)^{2n+1}=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}x^{4n+2}
$$

이는 모든 실수에서 수렴하므로 $\sin(x^2)$는 $\mathbb{R}$에서 해석적이다.
