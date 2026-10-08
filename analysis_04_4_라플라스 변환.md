# 라플라스 변환 (Laplace Transform)

**라플라스 변환은 함수 $h(x)$에 지수 가중치 $e^{-sx}$를 곱하여 적분한 것을, 새로운 변수 $s$의 함수로 나타내는 변환이다.** 완비성 증명에서는 특히 다음 유일성 성질을 사용한다.

$$
\boxed{
\int_0^\infty h(x)e^{-sx}\,dx=0
\quad\text{모든 충분히 큰 }s
\quad\Longrightarrow\quad
h(x)=0\ \text{거의 모든 }x.
}
$$

**1. 정의와 존재 조건**

함수 $h:[0,\infty)\to\mathbb R$의 단측 라플라스 변환은

$$
\boxed{
H(s)=\mathcal L\{h\}(s)
:=\int_0^\infty e^{-sx}h(x)\,dx
}
$$

이다. 아래에서는 $s$를 실수로 다룬다.

- $x$는 적분변수이다.
- $s$는 변환 결과 $H$의 입력변수이다.
- $\mathcal L$은 함수 $h$를 함수 $H$로 보내는 연산자이다.

통계학에서 사용하는 $h$는 밀도함수일 필요가 없으며, 음수도 가질 수 있다.

가장 직접적인 존재 조건은 절대수렴이다.

$$
\int_0^\infty |h(x)|e^{-sx}\,dx<\infty.
$$

이를 보장하는 대표적인 충분조건은 다음과 같다.

> $h$가 각 유한 구간에서 적분 가능하고, 충분히 큰 $x$에서 $|h(x)|\le Ce^{ax}$를 만족하면, $s>a$에서 라플라스 변환이 존재한다.

**증명.** 위 부등식이 $x\ge x_0$에서 성립한다고 하면

$$
\begin{aligned}
\int_0^\infty |h(x)|e^{-sx}\,dx
&=\int_0^{x_0}|h(x)|e^{-sx}\,dx
+\int_{x_0}^\infty |h(x)|e^{-sx}\,dx\\
&\le \int_0^{x_0}|h(x)|e^{-sx}\,dx
+C\int_{x_0}^\infty e^{-(s-a)x}\,dx.
\end{aligned}
$$

첫 번째 적분은 유한하고, 두 번째 적분은 $s>a$에서 유한하다. 이러한 성장 조건을 **지수차수 조건**이라고 한다. [math.colorado.edu](https://math.colorado.edu/~yuhu6917/teaching/Math3430_20Spring/Laplace_I.pdf?utm_source=chatgpt.com)

**2. 기본 변환식을 직접 계산한다.**

| 함수 $h(x)$ | 라플라스 변환 $H(s)$ | 수렴 조건 |
|---|---|---|
| $1$ | $1/s$ | $s>0$ |
| $e^{ax}$ | $1/(s-a)$ | $s>a$ |
| $x^m$, $m=0,1,\ldots$ | $m!/s^{m+1}$ | $s>0$ |
| $x^{\alpha-1}$, $\alpha>0$ | $\Gamma(\alpha)/s^\alpha$ | $s>0$ |

첫 번째와 두 번째는 지수함수의 적분으로 바로 얻는다.

$$
\mathcal L\{1\}(s)
=\int_0^\infty e^{-sx}\,dx
=\frac1s,
$$

$$
\mathcal L\{e^{ax}\}(s)
=\int_0^\infty e^{-(s-a)x}\,dx
=\frac1{s-a}.
$$

일반적인 거듭제곱은 $u=sx$로 치환하면

$$
\begin{aligned}
\mathcal L\{x^{\alpha-1}\}(s)
&=\int_0^\infty x^{\alpha-1}e^{-sx}\,dx\\
&=\frac1{s^\alpha}
\int_0^\infty u^{\alpha-1}e^{-u}\,du\\
&=\frac{\Gamma(\alpha)}{s^\alpha}.
\end{aligned}
$$

$\alpha=m+1$이면 $\Gamma(m+1)=m!$이므로 정수 거듭제곱의 공식도 얻는다.

아래에서는

$$
H(s)=\mathcal L\{h\}(s),\qquad
G(s)=\mathcal L\{g\}(s)
$$

로 표기한다. 각 등식은 관련 적분이 절대수렴하는 $s$에서 사용한다.

**3. 선형성**

$$
\boxed{
\mathcal L\{c_1h+c_2g\}(s)
=c_1H(s)+c_2G(s).
}
$$

**증명.** 적분의 선형성으로

$$
\begin{aligned}
\int_0^\infty e^{-sx}[c_1h(x)+c_2g(x)]\,dx
&=c_1\int_0^\infty e^{-sx}h(x)\,dx\\
&\quad+c_2\int_0^\infty e^{-sx}g(x)\,dx.
\end{aligned}
$$

단, $c_1,c_2$는 적분변수 $x$에 의존하지 않아야 한다.

**4. 지수함수를 곱하면 변환변수가 이동한다.**

$$
\boxed{
\mathcal L\{e^{ax}h(x)\}(s)=H(s-a).
}
$$

**증명.**

$$
\begin{aligned}
\mathcal L\{e^{ax}h(x)\}(s)
&=\int_0^\infty e^{-sx}e^{ax}h(x)\,dx\\
&=\int_0^\infty e^{-(s-a)x}h(x)\,dx\\
&=H(s-a).
\end{aligned}
$$

예를 들어

$$
\mathcal L\{xe^{ax}\}(s)=\frac1{(s-a)^2}.
$$

**5. 입력변수의 척도변환**

$c>0$이면

$$
\boxed{
\mathcal L\{h(cx)\}(s)=\frac1cH\left(\frac sc\right).
}
$$

**증명.** $u=cx$, $dx=du/c$로 치환하면

$$
\begin{aligned}
\int_0^\infty e^{-sx}h(cx)\,dx
&=\frac1c\int_0^\infty e^{-(s/c)u}h(u)\,du\\
&=\frac1cH\left(\frac sc\right).
\end{aligned}
$$

여기서는 실제 적분변수를 바꾸므로 $1/c$가 생긴다. 반면 앞서 사용한 $\lambda=1/\theta$는 모수의 재표현이어서 이러한 적분변수 보정이 없었다.

**6. 입력을 지연시키면 지수 인자가 곱해진다.**

$a\ge0$이고

$$
h_a(x)=I_{\{x\ge a\}}h(x-a)
$$

라면

$$
\boxed{
\mathcal L\{h_a\}(s)=e^{-as}H(s).
}
$$

**증명.** $u=x-a$로 치환하면

$$
\begin{aligned}
\mathcal L\{h_a\}(s)
&=\int_a^\infty e^{-sx}h(x-a)\,dx\\
&=\int_0^\infty e^{-s(u+a)}h(u)\,du\\
&=e^{-as}H(s).
\end{aligned}
$$

지수 인자 곱셈, 척도변환, 지연에 관한 식은 표준 라플라스 변환 성질이다. [math.utah.edu](https://www.math.utah.edu/~gustafso/s2019/2280/lectureslides/laplaceTheory2008.pdf?utm_source=chatgpt.com)

**7. $x$를 곱하면 변환함수를 미분한다.**

$$
\boxed{
\mathcal L\{xh(x)\}(s)=-H'(s).
}
$$

더 일반적으로

$$
\boxed{
\mathcal L\{x^mh(x)\}(s)=(-1)^mH^{(m)}(s).
}
$$

**증명.** 적분 안에서 미분할 수 있는 조건 아래에서

$$
\begin{aligned}
H'(s)
&=\frac d{ds}\int_0^\infty e^{-sx}h(x)\,dx\\
&=\int_0^\infty (-x)e^{-sx}h(x)\,dx\\
&=-\mathcal L\{xh(x)\}(s).
\end{aligned}
$$

반복해서 미분하면 일반식이 나온다.

**미분과 적분의 교환도 정당화할 수 있다.** 어떤 $s_0$에서

$$
\int_0^\infty |h(x)|e^{-s_0x}\,dx<\infty
$$

라면, $s>s_0$에서는

$$
x^m e^{-sx}|h(x)|
=
\left[x^m e^{-(s-s_0)x}\right]
e^{-s_0x}|h(x)|
$$

이다. 대괄호 안의 함수는 유계이므로, $s>s_0$의 작은 근방에서 적분 가능한 지배함수를 잡을 수 있다.

**8. 원래 함수를 미분하면 변환함수에 $s$가 곱해진다.**

$h$가 국소 절대연속이고, 필요한 변환이 존재하며, $e^{-sx}h(x)\to0$이라면

$$
\boxed{
\mathcal L\{h'\}(s)=sH(s)-h(0+).
}
$$

**증명.** 부분적분으로

$$
\begin{aligned}
\int_0^\infty e^{-sx}h'(x)\,dx
&=\left[e^{-sx}h(x)\right]_0^\infty
+s\int_0^\infty e^{-sx}h(x)\,dx\\
&=-h(0+)+sH(s).
\end{aligned}
$$

이를 반복하면

$$
\boxed{
\mathcal L\{h^{(m)}\}(s)
=s^mH(s)
-\sum_{j=0}^{m-1}s^{m-1-j}h^{(j)}(0+).
}
$$

각 단계에 필요한 미분 가능성과 수렴 조건을 가정한 것이다. 특히 **불연속 점프가 있는 함수에는 위 공식을 그대로 적용하면 안 된다.** 점프에 대한 보정항이 필요하다. [dlmf.nist.gov](https://dlmf.nist.gov/1.14?utm_source=chatgpt.com)

**9. 원래 함수를 적분하면 변환함수를 $s$로 나눈다.**

$s>0$에서

$$
\boxed{
\mathcal L\left\{\int_0^x h(u)\,du\right\}(s)
=\frac{H(s)}s.
}
$$

**증명.** 적분 순서를 바꾸면

$$
\begin{aligned}
\int_0^\infty e^{-sx}\left[\int_0^x h(u)\,du\right]dx
&=\int_0^\infty h(u)\left[\int_u^\infty e^{-sx}\,dx\right]du\\
&=\int_0^\infty h(u)\frac{e^{-su}}s\,du\\
&=\frac{H(s)}s.
\end{aligned}
$$

적분 순서 교환은

$$
\int_0^\infty\int_0^x e^{-sx}|h(u)|\,du\,dx
=\frac1s\int_0^\infty |h(u)|e^{-su}\,du<\infty
$$

로 정당화된다.

**10. 합성곱의 변환은 변환함수들의 곱이다.**

합성곱을

$$
(h*g)(x)=\int_0^x h(u)g(x-u)\,du
$$

로 정의하면

$$
\boxed{
\mathcal L\{h*g\}(s)=H(s)G(s).
}
$$

**증명.** $v=x-u$로 바꾸면 적분 영역 $0\le u\le x$가 $u,v\ge0$으로 바뀐다.

$$
\begin{aligned}
\mathcal L\{h*g\}(s)
&=\int_0^\infty\int_0^x
e^{-sx}h(u)g(x-u)\,du\,dx\\
&=\int_0^\infty\int_0^\infty
e^{-s(u+v)}h(u)g(v)\,dv\,du\\
&=\left[\int_0^\infty e^{-su}h(u)\,du\right]
\left[\int_0^\infty e^{-sv}g(v)\,dv\right]\\
&=H(s)G(s).
\end{aligned}
$$

두 변환이 절대수렴하면 이중적분의 절대수렴도 성립한다. 이 성질은 독립 확률변수의 합의 분포를 계산할 때 사용된다. [dlmf.nist.gov](https://dlmf.nist.gov/1.14?utm_source=chatgpt.com)

**11. 유일성 정리와 증명**

통계학에 필요한 형태로 서술하면 다음과 같다.

> $h$가 가측함수이고, 어떤 실수 $a$에 대해 모든 $s>a$에서
>
> $$
> \int_0^\infty |h(x)|e^{-sx}\,dx<\infty
> $$
>
> 라고 하자. 이때
>
> $$
> \mathcal L\{h\}(s)=0,\qquad\forall s>a
> $$
>
> 이면 $h(x)=0$이 거의 모든 $x\ge0$에서 성립한다.

증명에는 **연속함수를 다항식으로 균등근사할 수 있다는 Weierstrass 근사정리**를 사용한다.

**① 반무한구간을 유한구간으로 바꾼다.**

$b>a$를 하나 고정한다. 가정에 의해 모든 정수 $k\ge0$에 대해

$$
0=\mathcal L\{h\}(b+k) =\int_0^\infty h(x)e^{-bx}e^{-kx}\,dx.
$$

$u=e^{-x}$, $x=-\log u$, $dx=-du/u$로 치환하면

$$
0=\int_0^1h(-\log u)u^{b-1}u^k\,du.
$$

따라서

$$
q(u):=h(-\log u)u^{b-1}
$$

로 두면

$$
\int_0^1q(u)u^k\,du=0,\qquad k=0,1,2,\ldots.
$$

또한

$$
\int_0^1|q(u)|\,du
=\int_0^\infty |h(x)|e^{-bx}\,dx<\infty.
$$

즉, $q$는 적분 가능한 함수이다.

**② 모든 다항식에 대한 적분이 0이다.**

선형성으로 임의의 다항식 $p$에 대해

$$
\int_0^1q(u)p(u)\,du=0.
$$

**③ 모든 연속함수에 대한 적분도 0이다.**

임의의 연속함수 $\varphi$를 다항식 $p_m$으로 균등근사하면 ($\| ... \|_\infty$ 는 상한 노름)

$$
\begin{aligned}
\left|\int_0^1q(u)[\varphi(u)-p_m(u)]\,du\right|
&\le
\|\varphi-p_m\|_\infty
\int_0^1|q(u)|\,du\\
&\longrightarrow0.
\end{aligned}
$$

그러므로

$$
\int_0^1q(u)\varphi(u)\,du=0
$$

이다.

**④ $q=0$이 거의 모든 점에서 성립한다.**

구간 $[0,v]$의 지시함수를 값이 $0$과 $1$ 사이인 연속함수들로 근사하면, 지배수렴정리에 의해

$$
\int_0^v q(u)\,du=0,\qquad 0\le v\le1.
$$

왼쪽은 절대연속함수이며 그 도함수는 거의 모든 $v$에서 $q(v)$이다. 따라서

$$
q(v)=0\quad\text{거의 모든 }v\in(0,1).
$$

이제 $u^{b-1}>0$이므로 변수변환을 되돌리면

$$
\boxed{h(x)=0\quad\text{거의 모든 }x\ge0.}
$$

유일성은 라플라스 변환 이론의 핵심 정리이며, 연속함수에 대한 정리와 그 형식적 증명도 공개되어 있다. 위에서는 완비성에 맞추어 적분 가능한 가측함수의 형태로 증명하였다. [Archive of Formal Proofs](https://www.isa-afp.org/entries/Laplace_Transform.html?utm_source=chatgpt.com)

**“거의 모든 점”이라는 결론이 필요한 이유도 분명하다.** 한 점의 함수값만 바꾸어도 적분은 변하지 않으므로, 변환만으로 그 한 점의 값을 구분할 수 없다.
