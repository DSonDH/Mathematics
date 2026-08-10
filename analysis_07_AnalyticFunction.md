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
함수 $f$가 $x=c$에서 해석적이면, 정의에 의해$f(x)=\sum_{n=0}^{\infty} a_n (x-c)^n$ 로 표현된다. 멱급수는 수렴반경 내에서 항별미분이 가능하므로,

$$
f'(x)=\sum_{n=1}^{\infty} na_n (x-c)^{n-1}\\
f''(x)=\sum_{n=2}^{\infty} n(n-1)a_n (x-c)^{n-2}\\
\vdots\\
f^{(k)}(x)=\sum_{n=k}^{\infty} n(n-1)\cdots(n-k+1)a_n (x-c)^{n-k}
$$

$x=c$를 대입하면,

$$
f(c)=a_0,\quad f'(c)=a_1,\quad f''(c)=2a_2,\quad \ldots,\quad f^{(n)}(c)=n!a_n
$$

따라서

$$A_n=\frac{f^{(n)}(c)}{n!}$$

이를 원래 급수에 대입하면

$$
f(x)=\sum_{n=0}^{\infty}\frac{f^{(n)}(c)}{n!}(x-c)^n
$$

$\square$

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

**(참고)**

* $n\to\infty$일 때 $R_n(x,a)\to 0$이면 $f$는 테일러급수로 완전히 표현된다.
* 라그랑주 나머지항은 오차 추정에 유용하게 사용된다.
  - 분모에 $n!$이 있으므로, $f^{(n)}$이 충분히 작으면 $R_n(x,a)$는 급격히 작아진다.
  - 분자에 $(x-a)^n$이 있으므로, $x$가 $a$에 가까울수록 $R_n(x,a)$는 작아진다.
  - $f^{(n)}$이 연속이고 유계이면, 적절한 상계를 이용하여 처리할 수 있다.
* 라그랑주 나머지항과 코시 나머지항은 같은 테일러 오차를 서로 다른 형태로 표현하지만, 오차를 상계할 때 어느 형태가 더 유리한지는 함수에 따라 달라진다.
  - 라그랑주 나머지항

    $$
    R_N(x)=\frac{f^{(N+1)}(\xi)}{(N+1)!}(x-a)^{N+1}
    $$

    에서는 $f^{(N+1)}(\xi)$ 자체를 상계해야 한다. 고계도함수가 $N$에 따라 빠르게 커지면 이 상계가 지나치게 커져, 실제로 테일러 급수가 수렴하는 영역에서도 $R_N(x)\to0$을 증명하지 못할 수 있다. 즉, 이는 실제 발산이 아니라 나머지항 추정법의 한계일 수 있다.
  - 반면 코시 나머지항

    $$
    R_N(x)=\frac{f^{(N+1)}(c)}{N!}(x-c)^N(x-a)
    $$

    은 고계도함수의 증가와 함께 $(x-c)^N$이라는 추가적인 구조가 남는다. 따라서 $f^{(N+1)}(c)$에서 생기는 큰 인자를 $(x-c)^N$과 결합하여 더 작은 기하급수적 인자로 만들 수 있는 경우가 있다.
* 결국 테일러 급수의 수렴 증명에서는 나머지항 공식 자체보다, 그 공식을 통해 $R_N(x)\to0$를 얼마나 날카롭게 추정할 수 있는지가 중요하다.

> **라그랑주 나머지항 정리 증명**  
> $f$가 $a$와 $x$를 포함하는 구간에서 $N+1$번 미분 가능하다고 하자. $f$의 $a$에서의 $N$차 테일러 다항식을
> 
> $$
> S_N(t)=\sum_{n=0}^{N}\frac{f^{(n)}(a)}{n!}(t-a)^n
> $$
> 
> 이라 하고 나머지를 $R_N(t)=f(t)-S_N(t)$ 로 정의한다. 테일러 다항식의 정의에 의해 $S_N^{(k)}(a)=f^{(k)}(a),\ k=0,1,\ldots,N$ 이므로
> 
> $$
> R_N(a)=R_N'(a)=\cdots=R_N^{(N)}(a)=0.
> $$
> 
> 또한 $S_N$는 N차 이하의 다항식이므로 $S_N^{(N+1)}(t)=0$
> 
> 이제 $R_N(t)$와 $g(t)=(t-a)^{N+1}$ 에 코시 평균값정리를 적용한다. 어떤 $x_1\in(a,x)$가 존재하여
> 
> $$
> \frac{R_N(x)-R_N(a)}{(x-a)^{N+1}-(a-a)^{N+1}} = \frac{R_N'(x_1)}{(N+1)(x_1-a)^N}.
> $$
> 
> $R_N(a)=0$이므로
> 
> $$
> \frac{R_N(x)}{(x-a)^{N+1}} = \frac{R_N'(x_1)}{(N+1)(x_1-a)^N} \tag{1}
> $$
> 
> 이제 구간 $[a,x_1]$에서 $R_N'$와 $(t-a)^N$에 다시 코시 평균값정리를 적용한다. $R_N'(a)=0$이므로 어떤 $x_2\in(a,x_1)$가 존재하여
> 
> $$
> \frac{R_N'(x_1)}{(x_1-a)^N} = \frac{R_N''(x_2)}{N(x_2-a)^{N-1}}
> $$
> 
> 이를 (1)에 대입하면
> 
> $$
> \frac{R_N(x)}{(x-a)^{N+1}} = \frac{R_N''(x_2)}{(N+1)N(x_2-a)^{N-1}}.
> $$
> 
> 이 과정을 계속 반복한다. 매 단계마다 $R_N^{(k)}(a)=0$이므로 코시 평균값정리를 적용할 수 있다. 결국 어떤 $\xi\in(a,x)$가 존재하여
> 
> $$
> \frac{R_N(x)}{(x-a)^{N+1}} = \frac{R_N^{(N+1)}(\xi)}{(N+1)!}.
> $$
> 
> 그런데
> 
> $$
> R_N^{(N+1)}(\xi) =f^{(N+1)}(\xi)-S_N^{(N+1)}(\xi) =f^{(N+1)}(\xi)
> $$
> 
> 이므로
> 
> $$
> R_N(x) = \frac{f^{(N+1)}(\xi)}{(N+1)!}(x-a)^{N+1}.
> $$


# 2. 여러 가지 해석함수의 예 *(Examples of Analytic Functions)*
다음 함수들은 해당 구간에서 멱급수로 전개되며 해석함수이다.
* **기하급수**

$$
\frac{1}{x}
=\sum_{n=0}^{\infty}(1-x)^n
\quad (0<x<2)\\
\frac{1}{1-x}
=\sum_{n=0}^{\infty}x^n
\quad (-1<x<1)\\
\frac{1}{1+x}
=\sum_{n=0}^{\infty}(-1)^n x^n
\quad (-1<x<1)
$$

* **제곱근 함수**

$$
\sqrt{x}
=1+\frac{1}{2}(x-1)-\frac{1}{2^3}(x-1)^2+\frac{1}{2^4}(x-1)^3-\cdots
\quad (0<x<2)\\
\sqrt{1+x}
=1+\frac{1}{2}x-\frac{1}{2^3}x^2+\frac{1}{2^4}x^3-\frac{5}{2^7}x^4+\cdots
\quad (-1<x\le 1)
$$

* **지수함수**

$$
e^x=\sum_{n=0}^{\infty}\frac{x^n}{n!}
\quad (-\infty<x<\infty) \\
e^{-x}=\sum_{n=0}^{\infty}\frac{(-1)^n x^n}{n!}
\quad (-\infty<x<\infty) \\
a^x=\sum_{n=0}^{\infty}\frac{(\ln a)^n}{n!}x^n
$$

* **로그함수**

$$
\ln(1-x)
=-\sum_{n=1}^{\infty}\frac{x^n}{n}
\quad (-1\le x<1)\\
\ln(1+x)
=\sum_{n=1}^{\infty}\frac{(-1)^{n+1}}{n}x^n
\quad (-1<x\le 1)\\
\ln x
=\sum_{n=1}^{\infty}\frac{(-1)^{n+1}}{n}(x-1)^n
\quad (0<x\le 2)
$$

* **삼각함수**

$$
\sin x=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}x^{2n+1},
\quad
\cos x=\sum_{n=0}^{\infty}\frac{(-1)^n}{(2n)!}x^{2n}
\quad (-\infty<x<\infty)\\
\tan x=x+\frac{x^3}{3}+\frac{2x^5}{15}+\frac{17x^7}{315}+\cdots
\quad \left(-\frac{\pi}{2}<x<\frac{\pi}{2}\right)
$$

* **쌍곡함수**

$$
\sinh x=\sum_{n=0}^{\infty}\frac{x^{2n+1}}{(2n+1)!}=x+\frac{x^3}{6}+\frac{x^5}{120}+\cdots
\quad (-\infty<x<\infty)\\
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
\quad (-1\le x\le 1)\\
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
f(x)=x^{1/2},\quad f'(x)=\frac{1}{2}x^{-1/2},\quad f''(x)=-\frac{1}{4}x^{-3/2}\\
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
f(x)=\ln x,\quad f'(x)=\frac{1}{x},\quad f''(x)=-\frac{1}{x^2}\\
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
(f+g)(x)=\sum_{n=0}^{\infty} (a_n+b_n)(x-c)^n\\
(f-g)(x)=\sum_{n=0}^{\infty} (a_n-b_n)(x-c)^n\\
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

# 근사이론 (Approximation Theory)

## Thm. [바이어슈트라스 근사정리] *(Weierstrass Approximation Theorem)*

$f: [a,b] \to \mathbb{R}$가 연속함수이면, 임의의 $\epsilon>0$에 대하여 다항식 $P(x)$가 존재하여 다음 부등식이 $\forall x \in [a,b]$에 대해 성립한다.

$$
|f(x)-P(x)|<\epsilon
$$

- 닫힌 구간에서 연속함수는 항상 다항함수에 의해 고르게 근사될 수 있다
- 테일러급수는 $f$가 무한번 미분가능해야하고, 그렇다 하더라도 테일러급수가 항상 수렴하는 것은 아니므로, 바이어슈트라스 근사정리는 테일러급수보다 더 일반적인 근사정리다.

**증명**

해석학 첫걸음 책 참고. 아래 보간법 내용 + 특수한 함수꼴 활용이 필요한 긴 내용이 있다.

- 바이어슈트라스 정리에서 compactness가 본질적인 역할을 한다. 즉, 열린구간에서 더 이상 성립하지 않고, 무한으로 뻗는 구간에서도 성립하지 않는다.

## Def. [보간법] *(Interpolation)*
함수 $\phi$에 대해 각 부분구간 $[x_i, x_{i+1}]$에서 ($i = 1, ..., n)$. 일차함수가 되는 $[a,b]$의 분할이 $a= x_0 < x_1 < ... < x_n = b$로 주어졌다고 하자. 

연속함수 $\phi: [a,b] \to \mathbb R$를 다각형 함수(Polygonal) 라고 한다.

보간법(Interpolation)은 주어진 데이터 점들을 통과하는 다항함수 또는 다각형 함수를 찾는 방법이다.

### 정리. [다각형 함수 존재성]
$f: [a,b] \to \mathbb{R}$가 연속함수이면, 임의의 $\epsilon>0$에 대하여 다각형 함수 $\phi(x)$가 존재하여 다음 부등식이 $\forall x \in [a,b]$에 대해 성립한다.

$$
|f(x)-\phi(x)|<\epsilon
$$

**증명**  

1. 연속함수 $f$는 균등연속이다.

$[a,b]$는 닫히고 유계인 구간이고 $f$가 연속이므로 $f$는 $[a,b]$에서 균등연속이다.
따라서 임의의 $\varepsilon>0$에 대하여 어떤 $\delta>0$가 존재하여

$$
|x-y|<\delta \quad\Longrightarrow\quad |f(x)-f(y)|<\varepsilon
\tag{1}
$$

가 모든 $x,y\in[a,b]$에 대해 성립한다.

2. 구간을 충분히 잘게 나눈다.

$[a,b]$를

$$
a=x_0<x_1<\cdots<x_n=b

$$
로 나누되, 모든 $i$에 대하여 $x_i-x_{i-1}<\delta$ 가 되도록 한다.
각 점 $(x_0,f(x_0)),(x_1,f(x_1)),\ldots,(x_n,f(x_n))$을 직선으로 연결하여 다각형 함수 $\phi$를 정의한다.  
구체적으로 $x\in[x_{i-1},x_i]$에서

$$
\phi(x)
=\frac{x_i-x}{x_i-x_{i-1}}f(x_{i-1})
+\frac{x-x_{i-1}}{x_i-x_{i-1}}f(x_i)
\tag{2}
$$

로 정의한다.

3. 오차를 계산한다.

$x\in[x_{i-1},x_i]$이고

$$
\lambda=\frac{x-x_{i-1}}{x_i-x_{i-1}},
$$

이면 $0\le\lambda\le1$이고

$$
\phi(x)=(1-\lambda)f(x_{i-1})+\lambda f(x_i).
$$

따라서

$$
\begin{aligned}
|f(x)-\phi(x)|
&= \left|
f(x)-(1-\lambda)f(x_{i-1})-\lambda f(x_i)
\right|\\
&= \left|
(1-\lambda)(f(x)-f(x_{i-1}))
+\lambda (f(x)-f(x_i))
\right|\\
&\le (1-\lambda)|f(x)-f(x_{i-1})|+\lambda|f(x)-f(x_i)|.
\end{aligned}
$$

$x\in[x_{i-1},x_i]$이고 $x_i-x_{i-1}<\delta$이므로 $|x-x_{i-1}|<\delta$ 및 $|x-x_i|<\delta$이다. 균등연속성 (1)에 의해

$$
|f(x)-f(x_{i-1})|<\varepsilon,\qquad
|f(x)-f(x_i)|<\varepsilon.
$$

따라서

$$
|f(x)-\phi(x)|<(1-\lambda)\varepsilon+\lambda\varepsilon = \varepsilon.
$$

이는 모든 $x\in[a,b]$에 대해 성립하므로

$$
\sup_{x\in[a,b]}|f(x)-\phi(x)|<\varepsilon.
$$

### $f(x) = \sqrt{1-x}$ 의 테일러 급수
$f(x) = \sqrt{1-x}$ 의 테일러 급수의 계수 $a_n$은 $a_0 = 1$ 이고, $n \ge 1$에 대해

$$
a_n = (-1)^n \frac{1 \cdot 3 \cdot 5 \cdots (2n-3)}{2^n n!}
$$

따라서 

$$ 
\sqrt{1-x} = \sum_{n=0}^{\infty} a_n x^n = 1 - \frac{1}{2}x + \frac{1}{8}x^2 - \frac{1}{16}x^3 + \cdots
$$

이는 $|x| \leq 1$에서 수렴한다.

>**증명**  
>
>Taylor 정리를 Cauchy 나머지항 형태로 사용하면 중심이 0인 경우에
>
>$$
>E_N(x)=f(x)-\sum_{n=0}^N a_n x^n=\frac{f^{(N+1)}(c)}{N!}(x-c)^N x
>$$
>
>을 만족하는 $c$가 $0$와 $x$ 사이에 존재한다. 이는 일반적인 Cauchy 나머지항
>
>$$
>R_N(x)=\frac{f^{(N+1)}(c)}{N!}(x-c)^N(x-a)
>$$
>
>에서 $a=0$일 때의 특별한 경우이다.
>
>이제 $f(x)=\sqrt{1-x}$에 대해 적용한다. 
>
>$$
>f^{(N+1)}(c)=-\frac{1\cdot3\cdot5\cdots(2N-1)}{2^{N+1}}(1-c)^{-(N+1/2)}
>$$
>
>임을 구했으므로, Cauchy 나머지항으로부터
>
>$$
>|E_N(x)|
>= A_N \frac{|x-c|^N|x|}{(1-c)^{N+1/2}},
>\qquad
>A_N=\frac{1\cdot3\cdots(2N-1)}{2^{N+1}N!}
>$$
>
>이다.
>
>계수 $A_N$는
>
>$$
>A_N=\frac12\frac{1}{2}\frac{3}{4}\frac{5}{6}\cdots\frac{2N-1}{2N}\le\frac12
>\tag{1}
>$$
>
>이므로 계수 자체는 $N$에 대해 전혀 문제가 되지 않는다.
>
>이제 $x$의 부호에 따라 나누어 생각한다.
>
>**경우 1. $0<x<1$**
>
>이때 $0<c<x$이고,
>
>$$
>\frac{x-c}{1-c}\le x
>$$
>
>이 성립한다. 실제로
>
>$$
>x-c\le x(1-c)\iff -c\le -xc\iff c\ge xc
>$$
>
>이고 $0<x<1$, $c>0$에서 참이다.
>
>따라서
>
>$$
>|E_N(x)|
>\le A_N\left(\frac{x-c}{1-c}\right)^N\frac{x}{\sqrt{1-c}}
>\le \frac12 x^N \frac{x}{\sqrt{1-c}}.
>$$
>
>또 $c<x$이므로 $1-c>1-x$이고,
>
>$$
>\frac1{\sqrt{1-c}}<\frac1{\sqrt{1-x}}.
>$$
>
>결국
>
>$$
>|E_N(x)|\le \frac{x^{N+1}}{2\sqrt{1-x}}.
>$$
>
>$0<x<1$이 고정되어 있으면 $x^{N+1}\to0$이므로 $E_N(x)\to0$.
>
>**경우 2. $-1<x<0$**
>
>이때 $x<c<0$이고,
>
>$$
>|x-c|<|x|,
>\qquad 1-c>1
>$$
>
>이므로
>
>$$
>\frac{|x-c|}{1-c}<|x|.
>$$
>
>따라서
>
>$$
>|E_N(x)|\le A_N |x|^N \frac{|x|}{\sqrt{1-c}}\le \frac12 |x|^{N+1}.
>$$
>
>$|x|<1$이므로 역시 $E_N(x)\to0$. $x=0$에서는 자명하게 $E_N(0)=0$이다.
>
>따라서 모든 $-1<x<1$에 대해 $E_N(x)\to0$이며,
>
>$$
>\sqrt{1-x}=\sum_{n=0}^{\infty}a_n x^n,
>\qquad -1<x<1.
>$$
>
>즉,
>
>$$
>\sqrt{1-x}=1-\frac12x-\frac18x^2-\frac1{16}x^3-\frac5{128}x^4-\cdots,
>\qquad |x|<1.
>$$
>
>이 증명에서 핵심은 라그랑주 나머지항 대신 Cauchy 나머지항을 사용하면 오차항이 $\frac{|x-c|}{1-c}$ 형태로 바뀌어 $|x|<1$ 전체에서 제어할 수 있다는 점이다.
>
>(a) 한편, $c_n=\frac{1\cdot3\cdot5\cdots(2n-1)}{2\cdot4\cdot6\cdots2n}$에 대하여 $c_n<\frac{2}{\sqrt{2n+1}}$임을 귀납법으로 증명할 수 있다.
>
>$|a_n|=\frac{2\cdot4\cdots(2n-2)}{1\cdot3\cdots(2n-3)}\frac{2^n}{1} =2^n c_{n-1}$ 
>
>----
>테일러 급수가 끝점 $x=\pm1$에서도 성립함을 보이자.
>
>우선, $c_n=\frac{1\cdot3\cdot5\cdots(2n-1)}{2\cdot4\cdot6\cdots2n}$ 에 대하여 $c_n<\frac{2}{\sqrt{2n+1}}$ 임을 보인다.
>
>수학적 귀납법을 사용한다.
>
>$n=1$일 때 $c_1=\frac12<\frac2{\sqrt3}$ 이므로 성립한다.
>
>이제 $c_n < \frac2{\sqrt{2n+1}}$ 이라고 가정한다.
>
>정의에서 $c_{n+1} =c_n\frac{2n+1}{2n+2}$  
>따라서 $c_{n+1} < \frac2{\sqrt{2n+1}} \frac{2n+1}{2n+2}$
>
>양변이 양수이므로 제곱하면 $(2n+1)(2n+3)< (2n+2)^2$
>
>그런데 $(2n+1)(2n+3) = (2n+2)^2-1 <(2n+2)^2$
>
>따라서
>
>$$c_{n+1}<\frac2{\sqrt{2n+3}}$$
>
>이고 귀납법에 의해
>
>$$
>\boxed{c_n<\frac2{\sqrt{2n+1}}}
>$$
>
>
>$\sum a_n$의 절대수렴
>
>$n\ge1$일 때 $a_n = -\frac{1\cdot3\cdots(2n-3)}{2\cdot4\cdots2n}$ 이었다.
>
>따라서
>
>$$
>|a_n|
>=
>\frac{1\cdot3\cdots(2n-3)}
>{2\cdot4\cdots(2n-2)}
>\frac1{2n}.
>$$
>
>첫 번째 부분은 바로 $c_{n-1}$이므로
>
>$$
>\boxed{|a_n|=\frac{c_{n-1}}{2n}}.
>$$
>
>좀전의 결과를 적용하면 $n\ge2$에서 $c_{n-1}<\frac2{\sqrt{2n-1}}$
>
>따라서
>
>$$
>|a_n| < \frac{1}{n\sqrt{2n-1}}
>$$
>
>그리고 $2n-1\ge n$이므로
>
>$$
>|a_n| < \frac{1}{n\sqrt{2n-1}} < \frac{1}{n^{3/2}}.
>$$
>
>그런데 $p$-급수
>
>$$
>\sum_{n=1}^{\infty}\frac1{n^{3/2}}
>$$
>
>는 $3/2>1$이므로 수렴한다. 즉, 비교판정법에 의해 $\sum_{n=0}^{\infty}|a_n|<\infty$ 이고, 특히
>
>$$
>\boxed{\sum_{n=0}^{\infty}a_n\text{은 절대수렴한다}.}
>$$
>
>---
>
>모든 $x\in[-1,1]$에서 등식 (1)이 성립함을 증명해보자.
>
>$$
>\sqrt{1-x} = \sum_{n=0}^{\infty}a_nx^n \tag{1}
>$$
>
>이 $-1<x<1$ 에서 성립함을 증명했다.
>
>문제는 끝점 $x=\pm1$이다.  
>아까 $\sum_{n=0}^{\infty}|a_n|<\infty$ 를 보였고, 그리고 $|x|\le1$이면 $|a_nx^n|\le |a_n|$ 이므로 
>
>따라서 Weierstrass M-test에 의해 $\sum_{n=0}^{\infty}a_nx^n$ 은 $[-1,1]$에서 균등수렴한다.
>
>각 $a_nx^n$은 연속함수이므로 균등수렴한 극한 $S(x)=\sum_{n=0}^{\infty}a_nx^n$ 도 $[-1,1]$에서 연속이다.
>
>그런데 $(-1,1)$에서는 $S(x)=\sqrt{1-x}$ 양쪽 함수 모두 연속이므로 끝점에서도 극한을 취할 수 있다.
>
>$x\to1^{-}$이면 $S(1) =\lim_{x\to1^-}S(x) =\lim_{x\to1^-}\sqrt{1-x} =0$ 따라서 $\sum_{n=0}^{\infty}a_n=0$
>
>마찬가지로 $x\to-1^+$이면 $S(-1) = \lim_{x\to-1^+}\sqrt{1-x} =\sqrt2$ 따라서 $\sum_{n=0}^{\infty}a_n(-1)^n=\sqrt2$
>
>결국
>
>$$
>\boxed{
>\sqrt{1-x}
>= \sum_{n=0}^{\infty}a_nx^n
>\qquad\text{for every }x\in[-1,1]
>}
>$$
>
>$$
>\begin{aligned}
>\text{문제 4}&:\quad a_n\text{의 정확한 형태를 구함}\\
>\text{문제 5}&:\quad |x|<1\text{에서 테일러 급수임을 증명}\\
>\text{문제 6(a)(b)}&:\quad \sum |a_n|<\infty\text{를 증명}\\
>\text{문제 6(c)}&:\quad \text{균등수렴을 이용해 }x=\pm1\text{까지 확장}
>\end{aligned}
>$$
>
>특히 조심할 점은 단순히 "$x=1$을 급수에 대입하면 된다"고 해서는 안 된다는 것이다. 문제 5에서 증명한 것은 $|x|<1$뿐이기 때문이다. 문제 6(b)의 절대수렴 → 균등수렴 → 연속성이라는 과정이 끝점을 정당화한다.
>
>
>이제 확인할 것은, 앞의 문제에서 구한 $\sqrt{1-t}$ 의 다항식 >근사를 이용해서, 미분 불가능한 함수 $|x|$ 까지 다항식으로 균등근사할 수 있음을 보이는 중요한 단계다.
>
>**(a)**
>
>보이고 싶은 것은 임의의 $\varepsilon>0$에 대하여 다항함수 $q(x)$가 존재하여
>
>$$
>\big||x|-q(x)\big|<\varepsilon \quad \text{for all }x\in[-1,1]
>$$
>
>이 성립한다는 것이다.
>
>문제에서 힌트를 주었다:  $|a|=\sqrt{a^2}$ 이를 조금 다르게 쓰면
>
>$$
>|x| =\sqrt{x^2} =\sqrt{1-(1-x^2)} \tag{1}
>$$
>
>이전 문제에서 $\sqrt{1-t} = \sum_{n=0}^{\infty}a_nt^n \quad>(-1\leq t\leq1)$ 이고 $\sum_{n=0}^{\infty}|a_n|<\infty$ 임을 증명했다.
>
>이제 $t=1-x^2$ 를 대입한다. $x\in[-1,1]$ 이면 $0\leq1-x^2\leq1$ 이므로 위 급수에 대입할 수 있다.
>
>따라서
>
>$$
>\begin{aligned} 
>|x|
>&=\sqrt{x^2}\
>&=\sqrt{1-(1-x^2)}\
>&=\sum_{n=0}^{\infty}a_n(1-x^2)^n.
>\end{aligned}
>\tag{2}
>$$
>
>
>부분합을 $q_N(x)$라고 하자: $q_N(x)=\sum_{n=0}^{N}a_n(1-x^2)^n$. 각 $(1-x^2)^n$은 다항식이므로 $q_N(x)$도 다항식이다.
>
>이제 오차를 계산하면
>
>$$
>\big||x|-q_N(x)\big| =
>\left| \sum_{n=N+1}^{\infty}  a_n(1-x^2)^n \right| \leq
>\sum_{n=N+1}^{\infty} |a_n||1-x^2|^n
>$$
>
>그런데 $x\in[-1,1]$에서 $|1-x^2|\leq1$ 이므로
>
>$$
>\big||x|-q_N(x)\big| \leq
>\sum_{n=N+1}^{\infty}|a_n|.
>\tag{3}
>$$
>
>이전에 $\sum_{n=0}^{\infty}|a_n|$ 이 수렴함을 이미 보였으므로 그 꼬리합은 0으로 수렴한다. 즉 충분히 큰 $N$을 선택하면 $\sum_{n=N+1}^{\infty}|a_n|<\varepsilon$ 로 만들 수 있다.
>
>따라서 모든 $x\in[-1,1]$에 대하여 동시에 $\big||x|-q_N(x)\big|><\varepsilon$ 이다. 즉 $|x|$는 $[-1,1]$에서 다항식으로 균등근사할 수 있다.
>
>
>**(b) 임의의 ([a,b])로 일반화**
>
>이제 임의의 유한한 닫힌 구간 $[a,b]$ 를 생각한다.
>
>$R=\max\{|a|,|b|\}$ 이라고 하자. 그러면 $[a,b]\subseteq[-R,R]$.  ($R>0$)이라고 하자. (a)에 의해 임의의 $\delta>0$에 대하여 다항식 $q$가 존재하여 $|t|-q(t)\big|<\delta \quad(t\in[-1,1])$ 이다. 여기서 $\delta=\frac{\varepsilon}{R}$ 로 선택한다.
>
>이제 $\boxed{p(x)=R\,q\left(\frac{x}{R}\right)}$ 라고 정의한다. $q$가 다항식이므로 $p$도 $x$에 관한 다항식이다.
>
>$x\in[a,b]\subset[-R,R]$이면 $\frac{x}{R}\in[-1,1]$. 따라서
>
>$$
>\begin{aligned}
>\big||x|-p(x)\big| &= \left| R\left|\frac{x}{R}\right| - Rq\left(\frac{x}{R}\right) \right|\\
>&= R\left| \left|\frac{x}{R}\right| - q\left(\frac{x}{R}\right) \right|\\
>&< R\frac{\varepsilon}{R}\\
>&=\varepsilon.
>\end{aligned}
>$$
>
>그러므로
>
>$$
>\boxed{
>\forall\varepsilon>0,\quad
>\exists\text{ polynomial }p:\quad
>\big||x|-p(x)\big|<\varepsilon\quad\forall x\in[a,b].
>}
>$$
>
>즉, $|x|$ 는 임의의 닫힌 유계구간 $[a,b]$ 에서 다항식으로 >균등근사 가능하다.
>
>이 결과가 중요한 이유는 다음 문제 8에서 드러난다. 다각형 함수의 각 '꺾이는 모서리'를 $|x-a|$ 형태로 표현할 수 있고, 방금 $|x-a|$를 다항식으로 근사할 방법을 얻었기 때문이다. 따라서
>$\text{연속함수}\rightarrow\text{다각형 함수}\rightarrow |x-a|>\text{들의 조합}\rightarrow\text{다항함수}$ 라는 연결고리가 완성되면서 바이어슈트라스 근사 정리의 증명으로 이어진다.
>
>---
>
>문제 8은 지금까지의 모든 결과를 결합하여 바이어슈트라스 근사 정리를 실제로 완성하는 문제다. 특히 (c)의 표현 $\phi(x)=\phi(-1)+\sum b_i h_{a_i}(x)$ 가 핵심이다.
>
>**8-(a)**
>
>$$
>h_a(x)=\frac12\bigl(|x-a|+(x-a)\bigr)
>$$
>
>이다. $x\le a$이면 $|x-a|=a-x=-(x-a)$이므로 $h_a(x)=0$ 반대로 $x\ge a$이면 $|x-a|=x-a$ 이므로 $h_a(x)=x-a$
>
>따라서
>
>$$
>h_a(x)= \begin{cases}
>0,&x\le a,\\
>x-a,&x\ge a \end{cases}
>$$
>
>즉 (h_a)의 그래프는 (a)까지는 수평이고, (a) 이후부터 기울기 (1)인 직선이다. 이는 ReLU 함수와 정확히 같은 형태다.
>
>**8-(b)**
>
>$h_a$를 $[-1,1]$ 에서 다항식으로 균등근사할 수 있음을 보인다.
>
>이전 문제에 의해 절댓값 함수 $|t|$는 임의의 닫힌 구간에서 다항식으로 균등근사할 수 있다. $x\in[-1,1]$이면 $x-a\in[-1-a,1-a]$
>
>따라서 임의의 $\varepsilon>0$에 대하여 어떤 다항식 $q(t)$가 존재하여 $||t|-q(t)|<2\varepsilon$ 가 모든 $t\in[-1-a,1-a]$에서 성립한다.
>
>이제 
>
>$$
>p(x)=\frac12{q(x-a)+(x-a)}
>$$
>
>라고 하자. $q(x-a)$ 역시 $x$에 대한 다항식이므로 $p(x)$도 다항식이다.
>
>그리고 $|h_a(x)-p(x)| = \frac12 \big||x-a|-q(x-a)\big| < \frac12(2\varepsilon) = \varepsilon$ 
>
>따라서 $h_a$ 는 $[-1,1]$ 에서 다항식으로 균등근사 가능하다.
>
>**8-(c)**
>
>다각형 함수 $\phi$가 $-1=a_0 < a_1 < \cdots < a_n=1$ 의 각 부분구간에서 선형이라고 하자. 각 구간 $[a_i,a_{i+1}]$에서 $\phi$의 기울기를
>
>$$
>m_i= \frac{\phi(a_{i+1})-\phi(a_i)}
>{a_{i+1}-a_i},\qquad i=0,\ldots,n-1
>$$
>
>이라고 하자.
>
>다음과 같이 놓는다. $b_0=m_0$. 그리고 $b_i=m_i-m_{i-1},\quad i=1,\ldots,n-1$
>
>그러면
>
>$$
>\phi(x) =\phi(-1) +b_0h_{a_0}(x) +b_1h_{a_1}(x) +\cdots +b_{n-1}h_{a_{n-1}}(x) \tag{1}
>$$
>
>임을 보일 수 있다. 왜 그런지 직관적으로 보는 것이 중요하다.
>
>$h_{a_i}$의 기울기는
>
>$$
>\begin{cases}
>0, & x<a_i,\\
>1, & x>a_i
>\end{cases}
>$$
>
>이다.
>
>따라서 $b_i h_{a_i}$를 추가하는 것은 $x=a_i$를 통과하는 순간 전체 함수의 기울기를 $b_i$만큼 변화시키는 역할을 한다.
>
>첫 구간에서는 기울기가 $b_0=m_0$, 두 번째 구간에서는 $b_0+b_1 = m_0+(m_1-m_0) = m_1$, 세 번째에서는 $b_0+b_1+b_2 = m_2$, 일반적으로 $[a_k,a_{k+1}]$에서는 $\sum_{i=0}^k b_i = m_k$
>
>따라서 (1)의 함수와 $\phi$는 모든 구간에서 기울기가 같고, $x=-1$에서 둘 다 $\phi(-1)$이므로 완전히 같은 함수다.
>
>따라서 원하는 $b_i$들이 존재한다.
>
>**8-(d) 바이어슈트라스 근사 정리의 완성**
>
>이제 전체 증명이 연결된다. $f:[-1,1]\to\mathbb R$ 가 연속이라고 하고 임의의 $\varepsilon>0$ 을 잡는다.
>
>1단계: (f)를 다각형 함수로 근사한다. 정리 6.7.3에 의해 다각형 함수 $\phi$ 가 존재하여 $|f(x)-\phi(x)|<\frac{\varepsilon}{2}$ 가 모든 $x\in[-1,1]$에서 성립한다.
>
>2단계: (\phi)를 (h_a)들의 합으로 표현한다. (c)에 의해 $\phi(x) = \phi(-1)+ \sum_{i=0}^{n-1}b_i h_{a_i}(x)$. 편의상 $B=\sum_{i=0}^{n-1}|b_i|$ 라고 하자.
>
>3단계: 각각의 $h_{a_i}$를 다항식으로 근사한다. (b)에 의해 각 $h_{a_i}$를 다항식 $q_i$로 원하는 만큼 정확하게 근사할 수 있다. 특히 $|h_{a_i}(x)-q_i(x)| < \frac{\varepsilon}{2(1+B)}$ 가 되도록 $q_i$를 선택한다.
>
>그리고
>
>$$
>\boxed{
>p(x)=\phi(-1)+
>\sum_{i=0}^{n-1}b_iq_i(x)
>}
>$$
>
>라고 정의한다. 유한개의 다항식의 선형결합이므로 (p)도 다항식이다. 그러면
>
>$$
>|\phi(x)-p(x)| \leq \sum_{i=0}^{n-1} |b_i||h_{a_i}(x)-q_i(x)|
>< \frac{\varepsilon}{2(1+B)} \sum_{i=0}^{n-1}|b_i| 
>= \frac{\varepsilon B}{2(1+B)} < \frac{\varepsilon}{2}.
>$$
>
>따라서 삼각부등식에 의해
>
>$$
>|f(x)-p(x)| \leq
>|f(x)-\phi(x)| + |\phi(x)-p(x)| <
>\frac{\varepsilon}{2} +\frac{\varepsilon}{2}  = \varepsilon
>$$
>
>즉,  $|f(x)-p(x)|<\varepsilon  \quad \forall x\in[-1,1]$
>
>따라서 바이어슈트라스 근사 정리가 $[-1,1]$에서 증명되었다.
>
>
>마지막: 임의의 ([a,b])로 확장
>
>임의의 $f:[a,b]\to\mathbb R$가 연속이라고 하자. $[-1,1]$과 $[a,b]$ 사이의 선형변환 $x=\frac{b-a}{2}t+\frac{a+b}{2}$ 을 사용한다.
>
>즉
>
>$$
>F(t)=f\left(\frac{b-a}{2}t+\frac{a+b}{2}\right),\qquad -1\le t\le1.
>$$
>
>$F$는 $[-1,1]$에서 연속이므로 방금 증명한 결과에 의해 다항식 $P(t)$가 존재하여 $|F(t)-P(t)|<\varepsilon$
>
>역변환은 $t=\frac{2x-a-b}{b-a}$ 이므로
>
>$$
>p(x)=P\left(\frac{2x-a-b}{b-a}\right)
>$$
>
>라고 하면 $p$는 $x$의 다항식이며
>
>$$
>|f(x)-p(x)|<\varepsilon\qquad\forall x\in[a,b].
>$$
>
>따라서 최종적으로
>
>$$
>\boxed{
>f\in C([a,b])\Longrightarrow\forall\varepsilon>0,\exists p\text{ polynomial}:
>\sup_{x\in[a,b]}|f(x)-p(x)|<\varepsilon
>}
>$$
>
>가 성립한다.
>
>즉 이것으로 교재에서 문제 2부터 8까지 준비해 온 **바이어슈트라스 근사 정리의 증명이 완전히 끝난다.**
>