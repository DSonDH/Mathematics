# 월리스 곱

$$
\sum_{n=1}^{\infty}\frac{1}{n^2}= \frac{\pi^2}{6}
$$

위 등식을 오일러가 유도함을 6.1절에서 살펴봤다. 이를 위해 $\sin (x)$를 다음 두 가지 방식으로 나타낸다.

첫 번째는 테일러 급수이다.

$$
\sin (x)= x-\frac{x^3}{3!}+\frac{x^5}{5!}-\frac{x^7}{7!}+\cdots \tag 1
$$

두 번째는 $\sin (x)$의 영점 $0,\pm\pi,\pm2\pi,\ldots$을 이용한 무한곱 표현이다.

$$
\sin (x) =
x\left(1-\frac{x}{\pi}\right)
\left(1+\frac{x}{\pi}\right)
\left(1-\frac{x}{2\pi}\right)
\left(1+\frac{x}{2\pi}\right)\cdots\ =
x\prod_{n=1}^{\infty}
\left(1-\frac{x^2}{n^2\pi^2}\right) \tag 2
$$

이 절에서는 이 무한곱 표현의 특수한 경우를 이용하여 월리스 곱을 유도한다.
테일러 급수 전개와 고르게 수렴하는 급수의 성질을 이용한 최부림 교수의 증명을 피터 두렌(Peter Duren)이 일부 간소화한 증명을 간략히 소개한다.

---
등식 (2)에 소개한 $\sin(x)$에 대한 무한곱 표현의 특수한 경우를 증명해보자.

문제 1: $x=\pi/2$일 때 $\sin (x)$의 무한곱 표현이 다음 등식(3)과 같음을 보여라.

$$
\frac{\pi}{2}= \lim_{n\to\infty}
\left(\frac{2\cdot2}{1\cdot3}\right)
\left(\frac{4\cdot4}{3\cdot5}\right)
\cdots
\left(\frac{2n\cdot2n}{(2n-1)(2n+1)}\right) \tag 3
$$

- 이 등식은 존 월리스(John Wallis)가 처음 발견했다. 오일러 합 공식을 증명할 때 요기하게 쓰일 것이고, 팩토리얼 함수 정의에서 다시 쓸 것이다.

**증명**  

$\sin (x)$의 무한곱 표현은

$$
\sin (x)= x\prod_{k=1}^{\infty}
\left(1-\frac{x^2}{k^2\pi^2}\right)
$$

이다. 여기에 $x=\pi/2$를 대입하면

$$
1 = \sin\frac{\pi}{2}  \frac{\pi}{2} \prod_{k=1}^{\infty} \left(1-\frac{1}{4k^2}\right) = \frac{\pi}{2} \prod_{k=1}^{\infty} \frac{(2k-1)(2k+1)}{(2k)^2}
$$

이다. 따라서 양변을 정리하면

$$
\frac{\pi}{2}= \prod_{k=1}^{\infty}
\frac{(2k)^2}{(2k-1)(2k+1)} = \lim_{n\to\infty} \prod_{k=1}^{n} \frac{(2k)^2}{(2k-1)(2k+1)} \\
= \lim_{n\to\infty}
\left(\frac{2\cdot2}{1\cdot3}\right)
\left(\frac{4\cdot4}{3\cdot5}\right)
\cdots
\left(\frac{2n\cdot2n}{(2n-1)(2n+1)}\right)
$$

---

문제 2: $h(x)$와 $k(x)$가 $[a,b]$에서 연속인 도함수를 가질 때 부분적분 공식을 유도하여라.

$$
\int_a^b h(t)k'(t) dt= h(b)k(b)-h(a)k(a) \int_a^b h'(t)k(t) dt
$$

**증명**  

곱의 미분법에 의하여 $\frac{d}{dt}\bigl(h(t)k(t)\bigr)= h'(t)k(t)+h(t)k'(t)$ 이다. 양변을 $a$에서 $b$까지 적분하면

$$
\int_a^b\frac{d}{dt}\bigl(h(t)k(t)\bigr) dt= \int_a^b h'(t)k(t) dt
+ \int_a^b h(t)k'(t) dt
$$

미적분학의 기본정리에 따라 좌변은 $\int_a^b\frac{d}{dt}\bigl(h(t)k(t)\bigr) dt= h(b)k(b)-h(a)k(a)$ 이다. 따라서

$$
h(b)k(b)-h(a)k(a)= \int_a^b h'(t)k(t) dt + \int_a^b h(t)k'(t) dt
$$

이다. 마지막으로 항을 이항하면 증명 끝.

---

문제 3: 항등식 $\sin^n (x)= \sin^{n-1}(x)\sin (x)$과 부분적분법 공식으로 아래 $b_n$의 정의에 따른 점화식 $b_n = \frac{n-1}n b_{n-2}$ (단 $n \geq 2$)를 유도하라.

$$
b_n=\int_0^{\pi/2}\sin^n (x)\ dx, \quad n=0,1,2,\ldots \\
b_0=\int_0^{\pi/2}1\ dx=\frac{\pi}{2}, \quad b_1=\int_0^{\pi/2}\sin (x)\ dx=1
$$

**증명**  

$\sin^n (x)$를 다음과 같이 나눈다: $\sin^n (x)= \sin^{n-1}(x)\sin (x)$. 따라서 $b_n= \int_0^{\pi/2}\sin^{n-1}x\sin (x)\ dx$

부분적분을 위해 $h(x)=\sin^{n-1}x,\ k'(x)=\sin (x)$ 로 둔다. 그러면 $h'(x)= (n-1)\sin^{n-2}x\cos x$ 이고 $k(x)=-\cos x$ 이다.

부분적분 공식에 따라

$$
b_n =\left[-\sin^{n-1}x\cos x\right]_0^{\pi/2}+(n-1) \int_0^{\pi/2}\sin^{n-2}x\cos^2x\ dx.
$$

$n\geq2$일 때 경계항은 $0$이다. 또한 $\cos^2x=1-\sin^2x$ 
이므로

$$
b_n = (n-1) \int_0^{\pi/2} \sin^{n-2}x(1-\sin^2x)\ dx\ =
(n-1)\left(
\int_0^{\pi/2}\sin^{n-2}x\ dx - \int_0^{\pi/2}\sin^n (x)\ dx
\right)\\ =
(n-1)(b_{n-2}-b_n) =(n-1)b_{n-2}-(n-1)b_n
$$

이고 $nb_n=(n-1)b_{n-2}$ 이므로 증명 완료.

짝수 첨자의 일반항: 점화식을 반복하면

$$
b_{2n} =
\frac{2n-1}{2n}b_{2n-2}\ = \frac{2n-1}{2n} \frac{2n-3}{2n-2}\cdots \frac{1}{2}b_0.
$$

$b_0=\pi/2$이므로

$$
\boxed{
b_{2n}= \frac{1\cdot3\cdot5\cdots(2n-1)} {2\cdot4\cdot6\cdots(2n)} \frac{\pi}{2} }
$$

홀수 첨자의 일반항도 마찬가지로

$$
b_{2n+1} =
\frac{2n}{2n+1}b_{2n-1}\ = \frac{2n}{2n+1} \frac{2n-2}{2n-1}\cdots \frac{2}{3}b_1
$$

$b_1=1$이므로

$$
\boxed{ b_{2n+1}= \frac{2\cdot4\cdot6\cdots(2n)}{3\cdot5\cdot7\cdots(2n+1)} }
$$

---

문제 4: 다음을 증명하고, 이를 이용하여 월리스 곱 공식을 완성하여라.

$$
\lim_{n\to\infty}\frac{b_{2n}}{b_{2n+1}}=1.
$$

1. 적분의 단조성

$0\leq x\leq\pi/2$에서 $0\leq\sin (x)\leq1$ 이므로 $\sin^{n+1}x\leq\sin^n (x)$ 이다. 이를 적분하면 $b_{n+1}\leq b_n$ 이므로 $(b_n)$은 감소수열이다. 특히 $b_{2n+1}\leq b_{2n}\leq b_{2n-1}$ 이다. 모든 항이 양수이므로 $b_{2n+1}$로 나누면

$$
1
\leq
\frac{b_{2n}}{b_{2n+1}}
\leq
\frac{b_{2n-1}}{b_{2n+1}}.
$$

점화식을 변환하면

$$
\frac{b_{2n-1}}{b_{2n+1}}= \frac{2n+1}{2n}.
$$

따라서

$$
1 \leq \frac{b_{2n}}{b_{2n+1}} \leq \frac{2n+1}{2n}.
$$

그런데 $\lim_{n\to\infty}\frac{2n+1}{2n}=1$ 이므로 조임정리에 따라 $\lim_{n\to\infty}\frac{b_{2n}}{b_{2n+1}}=1$ 이다.

2. 월리스 곱의 증명

문제 3의 일반항을 대입하면

$$
\frac{b_{2n}}{b_{2n+1}} =
\frac{
\frac{1\cdot3\cdots(2n-1)}{2\cdot4\cdots(2n)} \frac{\pi}{2}
}{\frac{2\cdot4\cdots(2n)}{3\cdot5\cdots(2n+1)}}\ =
\frac{\pi}{2}
\frac{(1\cdot3\cdots(2n-1))(3\cdot5\cdots(2n+1))
}{(2\cdot4\cdots(2n))^2}.
$$

다음 부분곱을 정의하자.

$$
W_n= \prod_{k=1}^{n} \frac{(2k)^2}{(2k-1)(2k+1)}.
$$

그러면 위 식은

$$
\frac{b_{2n}}{b_{2n+1}}= \frac{\pi/2}{W_n}
$$

으로 쓸 수 있다. 따라서

$$
W_n= \frac{\pi/2}{b_{2n}/b_{2n+1}}.
$$

$n\to\infty$일 때 $b_{2n}/b_{2n+1}\to1$이므로 $\lim_{n\to\infty}W_n= \frac{\pi}{2}.$

결국

$$
\boxed{
\frac{\pi}{2}= \lim_{n\to\infty}
\prod_{k=1}^{n}
\frac{(2k)^2}{(2k-1)(2k+1)}
= \lim_{n\to\infty}
\left(\frac{2\cdot2}{1\cdot3}\right)
\left(\frac{4\cdot4}{3\cdot5}\right)
\cdots
\left(\frac{2n\cdot2n}{(2n-1)(2n+1)}\right)
}
$$

---

문제 5: 월리스 곱을 이용하여 다음 표현을 유도하여라.

$$
\sqrt{\pi}= \lim_{n\to\infty}
\frac{2^{2n}(n!)^2}{(2n)!\sqrt n}.
$$

1. 짝수와 홀수의 곱

짝수를 연달아 곱하면 $2\cdot4\cdot6\cdots(2n)= 2^n n!$ 이다. 한편 $(2n)!= (1\cdot3\cdots(2n-1)) (2\cdot4\cdots2n)$ 이므로 $1\cdot3\cdots(2n-1)= \frac{(2n)!}{2^n n!}$ 이다. 이제 다음과 같이 놓자. $E_n=2\cdot4\cdots(2n), \quad O_n=1\cdot3\cdots(2n-1)$. 그러면 $E_n=2^n n!, \quad O_n=\frac{(2n)!}{2^n n!}$ 따라서

$$
\frac{E_n}{O_n}= \frac{2^n n!}{(2n)!/(2^n n!)} = \frac{2^{2n}(n!)^2}{(2n)!}.
$$

2. 월리스 곱과 연결

월리스 부분곱은 $W_n= \prod_{k=1}^{n} \frac{(2k)^2}{(2k-1)(2k+1)}$ 이다.

분자의 곱은 $E_n^2$이고, 분모의 두 곱은 각각 $1\cdot3\cdots(2n-1)=O_n$ 과 $3\cdot5\cdots(2n+1)=(2n+1)O_n$ 이다. 따라서

$$
W_n= \frac{E_n^2}{(2n+1)O_n^2} = \frac{1}{2n+1} \left(\frac{E_n}{O_n}\right)^2.
$$

즉,

$$
\left(\frac{E_n}{O_n}\right)^2= (2n+1)W_n.
$$

양변을 $n$으로 나누면

$$
\left(\frac{E_n}{O_n\sqrt n}\right)^2= \frac{2n+1}{n}W_n.
$$

$n\to\infty$일 때

$$
\frac{2n+1}{n}\longrightarrow2, \quad W_n\longrightarrow\frac{\pi}{2}
$$

이다. 따라서

$$
\lim_{n\to\infty} \left(\frac{E_n}{O_n\sqrt n}\right)^2= \pi.
$$

모든 항이 양수이므로 양의 제곱근을 취하면

$$
\lim_{n\to\infty} \frac{E_n}{O_n\sqrt n}= \sqrt{\pi}.
$$

마지막으로

$$
\frac{E_n}{O_n}= \frac{2^{2n}(n!)^2}{(2n)!}
$$

이므로

$$
\boxed{
\sqrt{\pi}= \lim_{n\to\infty}
\frac{2^{2n}(n!)^2}{(2n)!\sqrt n}
}
$$

# 테일러 급수
증명의 다음 단계는 $\arcsin(x)$에 대한 테일러 급수를 만드는 것이다. 


앞의 문제에서 월리스 곱을 이용하여 $\sqrt{\pi} = \lim_{n\to\infty} \frac{2^{2n}(n!)^2}{(2n)!\sqrt n}$ 을 얻었다.

이제 함수

$$
f(x)=\frac{1}{\sqrt{1-x}}=(1-x)^{-1/2}
$$

의 테일러 급수를 구하고, 이 급수가 실제로 함수 $f(x)$에 수렴하는 범위를 조사한다.


문제 6: 다음 테일러 전개를 생각하자.

$$
\frac{1}{\sqrt{1-x}} = \sum_{n=0}^{\infty}c_nx^n.
$$

$c_0=1$이고, $n\geq1$일 때

$$
c_n = \frac{(2n)!}{2^{2n}(n!)^2} = \frac{1\cdot3\cdot5\cdots(2n-1)} 
{2\cdot4\cdot6\cdots2n}
$$

임을 보여라.

함수를 $f(x)=(1-x)^{-1/2}$  라고 하자. 도함수는

$$
f'(x) = \frac{1}{2}(1-x)^{-3/2} \\
f''(x) = \frac{1\cdot3}{2^2}(1-x)^{-5/2} \\
f^{(3)}(x) = \frac{1\cdot3\cdot5}{2^3}(1-x)^{-7/2} \\
f^{(n)}(x) = \frac{1\cdot3\cdot5\cdots(2n-1)}{2^n} (1-x)^{-(2n+1)/2}
$$

이다. 특히 $x=0$을 대입하면 $f^{(n)}(0) = \frac{1\cdot3\cdot5\cdots(2n-1)} {2^n}$

테일러 계수는 $c_n=\frac{f^{(n)}(0)}{n!}$ 이므로 $c_n = \frac{1\cdot3\cdot5\cdots(2n-1)}{2^nn!}$

그런데 $2^nn!=2\cdot4\cdot6\cdots(2n)$ 이므로

$$
\boxed{
c_n = \frac{1\cdot3\cdot5\cdots(2n-1)}
{2\cdot4\cdot6\cdots(2n)}
}
$$


문제 7: 다음 두 사실을 보여라: $\lim_{n\to\infty}c_n=0$ 그러나 $\sum_{n=0}^{\infty}c_n$ 은 발산한다.

1. 월리스 곱의 결과 이용

문제 5의 결과는 $\sqrt{\pi} = \lim_{n\to\infty} \frac{2^{2n}(n!)^2}{(2n)!\sqrt n}$ 이다.

문제 6에서 $c_n=\frac{(2n)!}{2^{2n}(n!)^2}$ 이므로

$$
\frac{2^{2n}(n!)^2}{(2n)!\sqrt n} = \frac{1}{c_n\sqrt n}.
$$

따라서 문제 5의 결과는 $\sqrt{\pi} = \lim_{n\to\infty}\frac{1}{c_n\sqrt n}$ 으로 쓸 수 있다. 양변의 역수를 취하면

$$
\lim_{n\to\infty}c_n\sqrt n = \frac{1}{\sqrt{\pi}}
$$

이다. 즉, $n$이 클 때 $c_n\sim\frac{1}{\sqrt{\pi n}}$ 이다.

2. $c_n\to0$의 증명

위 극한에서 $c_n$ 분자는 $1/\sqrt{\pi}$로 수렴하고 분모는 무한대로 발산하므로

$$
\boxed{\lim_{n\to\infty}c_n=0}
$$

3. $\sum c_n$의 발산

다음 급수는 $p=1/2$인 $p$-급수이다. $\sum_{n=1}^{\infty}\frac{1}{\sqrt n}$. 이때 $p\leq1$이므로 이 급수는 발산한다.

$c_n$과 $1/\sqrt n$에 극한비교판정법을 적용하면

$$
\lim_{n\to\infty}
\frac{c_n}{1/\sqrt n} = \lim_{n\to\infty}c_n\sqrt n = \frac{1}{\sqrt{\pi}}.
$$

극한값이 양의 유한한 수이므로 두 급수는 같은 수렴·발산 성질을 갖는다. 따라서

$$
\boxed{\sum_{n=1}^{\infty}c_n\text{은 발산한다}
}
$$

이고, $c_0=1$을 추가해도 발산성은 변하지 않으므로

$$
\boxed{\sum_{n=0}^{\infty}c_n\text{은 발산한다}
}
$$

---



논의: 테일러 급수의 수렴 문제

$$
\frac{1}{\sqrt{1-x}} = \sum_{n=0}^{\infty}c_nx^n \tag{4}
$$

이라는 형식적인 테일러 급수를 얻었다. 여기서 $c_n = \frac{(2n)!}{2^{2n}(n!)^2}$

이다. 또한, $c_n\to0$ 이지만 $\sum_{n=0}^{\infty}c_n$ 은 발산함을 확인했다.

$x=1$에서 급수가 발산하는 이유: 식 (4)에 $x=1$을 형식적으로 대입하면 우변은
$\sum_{n=0}^{\infty}c_n$ 이 된다. 문제 7에 의해 이 급수는 발산한다. 이는 원래 함수의 성질과도 일치한다. 함수 $f(x)=\frac{1}{\sqrt{1-x}}$ 
는 $x=1$에서 분모가 $0$이 되므로 정의되지 않는다. 따라서 $x=1$에서 테일러 급수가 발산하는 것은 자연스러운 결과이다.

이제 목표는 모든 $x\in(-1,1)$ 에 대하여 식 (4)가 실제로 성립함을 증명하는 것이다.

여기서 한 가지 주의해야 한다. 테일러 계수 $c_n$을 계산했다는 사실만으로
$\sum_{n=0}^{\infty}c_nx^n$ 이 함수 $1/\sqrt{1-x}$에 수렴한다고 주장할 수는 없다. 테일러 급수의 계수는 함수의 도함수로부터 계산되지만, 그 급수의 부분합이 원래 함수에 수렴하는지는 별도로 증명해야 한다.

원래 함수와 부분합 사이의 오차를

$$
E_N(x) = \frac{1}{\sqrt{1-x}} - \sum_{n=0}^{N}c_nx^n
$$

으로 정의한다. 식 (4)가 특정한 $x$에서 성립한다는 것은 정확히

$$
\lim_{N\to\infty}E_N(x)=0
$$

이라는 뜻이다. 일반적으로 라그랑주 나머지항 정리를 사용하지만, 이는 문제가 있다. 어떤 문제가 있는지 문제8로 살펴보자.

문제 8: 라그랑주 나머지항 정리를 이용하여 $\frac{1}{\sqrt{1-x}} = \sum_{n=0}^{\infty}c_nx^n$ 이 모든 $|x|<1/2$에서 성립함을 보여라.  
또한 $x\in(1/2,1)$에서 같은 방법을 사용하려 할 때 어떤 문제가 발생하는지 설명하여라.

1. 부분합과 오차 함수

함수를 $f(x)=\frac{1}{\sqrt{1-x}}$ 라고 하고, $N$차 테일러 다항식을 $S_N(x)=\sum_{n=0}^{N}c_nx^n$ 이라고 하자. 오차 함수는 $E_N(x) = f(x)-S_N(x)$ 이다. 급수가 $f(x)$에 수렴함을 보이려면 $\lim_{N\to\infty}E_N(x)=0$ 임을 증명하면 된다.

2. 라그랑주 나머지항

라그랑주 나머지항 정리에 따르면 $0$과 $x$ 사이의 어떤 수 $\xi$가 존재하여

$$
E_N(x) = \frac{f^{(N+1)}(\xi)}{(N+1)!}x^{N+1}
$$

이 성립한다.

문제 6에서 구한 도함수 공식에 따라

$$
f^{(N+1)}(\xi) = \frac{1\cdot3\cdots(2N+1)} {2^{N+1}} (1-\xi)^{-(2N+3)/2}.
$$

또한

$$
c_{N+1} = \frac{1\cdot3\cdots(2N+1)}
{2^{N+1}(N+1)!}
$$

이므로

$$
\frac{f^{(N+1)}(\xi)}{(N+1)!} = c_{N+1}(1-\xi)^{-(N+3/2)}.
$$

따라서

$$
\boxed{
E_N(x) = c_{N+1}
\frac{x^{N+1}}{(1-\xi)^{N+3/2}}
}
$$

이다.

3. 나머지항의 절댓값 추정

$\xi$는 $0$과 $x$ 사이에 있으므로 $|\xi|\leq|x|$.  따라서 $|1-\xi| \geq1-|\xi| \geq1-|x|$

또한 $0<c_{N+1}\leq1$  이므로

$$|E_N(x)| = c_{N+1} \frac{|x|^{N+1}}{|1-\xi|^{N+3/2}}\ \leq \frac{|x|^{N+1}}{(1-|x|)^{N+3/2}}$$

오른쪽을 정리하면

$$
|E_N(x)| \leq \frac{1}{(1-|x|)^{1/2}} \left(\frac{|x|}{1-|x|}\right)^{N+1}
$$

4. $|x|<1/2$에서의 수렴

$|x|<1/2$이면 $|x|<1-|x|$ 이므로 $0 \leq  \frac{|x|}{1-|x|} <1$

따라서 기하수열의 성질에 의해 $ \lim_{N\to\infty} \left(\frac{|x|}{1-|x|}\right)^{N+1}=0$. 그러므로 $\lim_{N\to\infty}|E_N(x)|=0$ 이고$\lim_{N\to\infty}E_N(x)=0$ 이다.

결국 모든 $|x|<1/2$에 대하여

$$
\boxed{
\frac{1}{\sqrt{1-x}} = \sum_{n=0}^{\infty}c_nx^n
}
$$

이 성립한다. 계수를 대입하면

$$
\boxed{
\frac{1}{\sqrt{1-x}} = \sum_{n=0}^{\infty}
\frac{(2n)!}{2^{2n}(n!)^2}x^n,
\qquad |x|<\frac12
}
$$

이다.

5. $x\in(1/2,1)$에서 발생하는 문제

$x\in(1/2,1)$이면 $\frac{x}{1-x}>1$

따라서 앞에서 구한 상계

$$
|E_N(x)|
\leq
\frac{1}{(1-x)^{1/2}}
\left(\frac{x}{1-x}\right)^{N+1}
$$

의 오른쪽은 $N\to\infty$일 때 $0$으로 수렴하지 않는다. 오히려

$$
\left(\frac{x}{1-x}\right)^{N+1}
\to\infty
$$

이다.

따라서 라그랑주 나머지항에 대한 이 추정만으로는

$$
E_N(x)\to0
$$

임을 증명할 수 없다.

중요한 점은 이것이 급수가 실제로 발산한다는 뜻은 아니라는 것이다. 실제로 테일러 급수는 모든 $|x|<1$ 에서 수렴한다. 다만 현재 사용한 라그랑주 나머지항의 상계가 너무 거칠기 때문에 $1/2<x<1$ 에서는 수렴성을 증명하지 못하는 것이다.

즉, 문제는 다음과 같이 정리된다.

$$
\boxed{
x\in\left(\frac12,1\right)
\text{이면 }
\frac{x}{1-x}>1
\text{이므로 라그랑주 나머지항의 상계가 }0\text{으로 가지 않는다}
}
$$

# 나머지항의 적분 표현

### 정리. 적분 나머지항 정리 (integral remainder theorem)

함수 $f$가 $(-R,R)$에서 $N+1$번 미분 가능하고 $f^{(N+1)}$이 연속이라고 가정한다. $n=0,1,\ldots,N$에 대하여 $a_n=\frac{f^{(n)}(0)}{n!}$ 으로 정의하고, 부분합을 $S_N(x) = a_0+a_1x+a_2x^2+\cdots+a_Nx^N$ 이라고 하자.

테일러 나머지항(오차함수)을 $E_N(x)=f(x)-S_N(x)$ 이라고 하면, 모든 $x\in(-R,R)$에 대하여

$$
\boxed{
E_N(x) = \frac{1}{N!}
\int_0^x f^{(N+1)}(t)(x-t)^N\ dt
}
$$

가 성립한다.

따라서 함수는 다음과 같이 표현된다.

$$
\boxed{
f(x) = \sum_{n=0}^{N}\frac{f^{(n)}(0)}{n!}x^n
+
\frac{1}{N!}
\int_0^x f^{(N+1)}(t)(x-t)^N\ dt
}
$$


문제 9 (a): 
다음을 보여라. $f(x)=f(0)+\int_0^x f'(t)\ dt$

미적분학의 기본정리에 의해 $\int_0^x f'(t)\ dt = f(x)-f(0)$ 이다. 따라서 $f(x)=f(0)+\int_0^x f'(t)\ dt$ 이다.  
이는 $N=0$인 적분 나머지항 공식이다. 실제로 $S_0(x)=f(0)$ 이고 $E_0(x) = f(x)-f(0) = \int_0^x f'(t)\ dt$ 이다.

문제 9-(b): 부분적분을 사용하여 다음을 보여라. $ f(x) = f(0)+f'(0)x + \int_0^x f''(t)(x-t)\ dt$

$\int_0^x f''(t)(x-t)\ dt$을 생각하자. 부분적분에서 $h(t)=x-t, k'(t)=f''(t)$ 로 놓으면 $h'(t)=-1, k(t)=f'(t)$ 이다. 따라서

$$
\int_0^x f''(t)(x-t)\ dt =
\left[(x-t)f'(t)\right]_0^x + \int_0^x f'(t)\ dt\ = -xf'(0)+f(x)-f(0)
$$

이를 정리하면 증명완료. 이는 $N=1$인 적분 나머지항 공식이다.

문제 9-(c): 앞의 과정을 계속하여 적분 나머지항 정리를 증명하여라.

문제 9-(a), (b)의 과정을 반복하면 다음 형태를 예상할 수 있다.

$$
f(x) = \sum_{n=0}^{N} \frac{f^{(n)}(0)}{n!}x^n + \frac{1}{N!}
\int_0^x f^{(N+1)}(t)(x-t)^N\ dt
$$

이를 수학적 귀납법으로 증명한다.

1단계: 초깃값  
$N=0$일 때 문제 9-(a)에 의해 $f(x) = f(0)+\int_0^x f'(t)\ dt$ 이므로 공식이 성립한다.

2단계: 귀납가정  
어떤 $N-1$에 대하여

$$
f(x) = \sum_{n=0}^{N-1} \frac{f^{(n)}(0)}{n!}x^n +
\frac{1}{(N-1)!} \int_0^x f^{(N)}(t)(x-t)^{N-1}\ dt
$$

가 성립한다고 가정한다.

나머지항을 $R_{N-1}(x) = \frac{1}{(N-1)!} \int_0^x f^{(N)}(t)(x-t)^{N-1}\ dt$ 라고 하자.

다음 적분에 부분적분을 적용한다.

$$
\int_0^x f^{(N+1)}(t)(x-t)^N\ dt
$$

$u=(x-t)^N,\ dv=f^{(N+1)}(t)\ dt$ 로 놓으면 $du=-N(x-t)^{N-1}\ dt, \ v=f^{(N)}(t)$ 이다. 따라서

$$
\int_0^x f^{(N+1)}(t)(x-t)^N\ dt = \left[f^{(N)}(t)(x-t)^N\right]_0^x +
N\int_0^x f^{(N)}(t)(x-t)^{N-1}\ dt\ \\
= -f^{(N)}(0)x^N\ + N\int_0^x f^{(N)}(t)(x-t)^{N-1}\ dt
$$

양변을 $N!$로 나누면

$$
\frac{1}{N!}
\int_0^x f^{(N+1)}(t)(x-t)^N\ dt =
-\frac{f^{(N)}(0)}{N!}x^N\ + \frac{1}{(N-1)!} \int_0^x f^{(N)}(t)(x-t)^{N-1}\ dt
$$

따라서

$$
R_{N-1}(x) = \frac{f^{(N)}(0)}{N!}x^N
+
\frac{1}{N!}
\int_0^x f^{(N+1)}(t)(x-t)^N\ dt
$$

이를 귀납가정에 대입하면

$$
f(x) = \sum_{n=0}^{N}
\frac{f^{(n)}(0)}{n!}x^n
+
\frac{1}{N!}
\int_0^x f^{(N+1)}(t)(x-t)^N\ dt
$$

따라서 모든 $N\geq0$에 대하여

$$
\boxed{
E_N(x) = \frac{1}{N!}
\int_0^x f^{(N+1)}(t)(x-t)^N\ dt
}
$$

가 성립한다.

---

이렇게 증명된 정리를 $f(x)=\frac{1}{\sqrt{1-x}}$ 에 적용할 수 있다.

문제 10-(a)

구간 $(-1,1)$에서 $\frac{1}{\sqrt{1-x}}$ 와 $S_2(x)$의 그래프는 대략 어떻게 생겼는가?  $x=\frac12,\ x=\frac34,\ x=\frac89$ 에서 $E_2(x)$를 계산하여라.

계수는 $c_0=1, c_1=\frac12, c_2=\frac38$ 이므로 $S_2(x)=1+\frac12x+\frac38x^2$ 이다.

오차는 $E_2(x) = \frac{1}{\sqrt{1-x}} = \left(1+\frac12x+\frac38x^2\right)$이다.

1. $x=1/2$인 경우 $\frac{1}{\sqrt{1-1/2}}=\sqrt2$ 이고 $S_2\left(\frac12\right) = 1+\frac14+\frac{3}{32} = \frac{43}{32}$. 따라서 $E_2\left(\frac12\right) = \sqrt2-\frac{43}{32} \approx0.07046$

2. $x=3/4$인 경우

$\frac{1}{\sqrt{1-3/4}}=2$ 이고

$S_2\left(\frac34\right) =
1+\frac38+\frac38\cdot\frac9{16}\ =
1+\frac38+\frac{27}{128}\ =
\frac{203}{128}$ 따라서 $E_2\left(\frac34\right) = 2-\frac{203}{128} = \frac{53}{128}$

3. $x=8/9$인 경우 $\frac{1}{\sqrt{1-8/9}}=3$  이고 $S_2\left(\frac89\right) =
1+\frac49+\frac38\cdot\frac{64}{81}\ =
1+\frac49+\frac8{27}\ =
\frac{47}{27}$. 따라서 $E_2\left(\frac89\right) = 3-\frac{47}{27} = \frac{34}{27}$ 이다.

$x$가 $1$에 가까워질수록 함수 $1/\sqrt{1-x}$가 급격하게 증가하므로, 고정된 이차다항식 $S_2(x)$과의 오차도 커진다.

문제 10-(b): $-1<x<1$일 때 다음이 성립함을 보여라.

$$
E_2(x) = \frac{15}{16}
\int_0^x
\left(\frac{x-t}{1-t}\right)^2
\frac{1}{(1-t)^{3/2}}\ dt
$$

적분 나머지항 정리에서 $N=2$로 놓으면 $E_2(x) = \frac{1}{2 !} \int_0^x f^{(3)}(t)(x-t)^2\ dt$

$f(x)=(1-x)^{-1/2}$의 세 번째 도함수는 $f^{(3)}(t) = \frac{15}{8}(1-t)^{-7/2}$ 이다. 따라서

$$
E_2(x) = \frac{15}{16}
\int_0^x
\frac{(x-t)^2}{(1-t)^{7/2}}\ dt
$$

그런데

$$
\frac{(x-t)^2}{(1-t)^{7/2}} = \left(\frac{x-t}{1-t}\right)^2
\frac{1}{(1-t)^{3/2}}
$$

이므로

$$
\boxed{
E_2(x) = \frac{15}{16}
\int_0^x
\left(\frac{x-t}{1-t}\right)^2
\frac{1}{(1-t)^{3/2}}\ dt
}
$$

문제 10-(c): 다음을 설명하여라.

$$
\left| \frac{x-t}{1-t} \right| \leq |x|.
$$

이를 이용하여 $|E_2(x)|$의 상계를 구하여라.

적분 구간에서 $t$는 $0$과 $x$ 사이에 있다. $0\leq x<1$인 경우: $0\leq t\leq x$이므로 $0\leq\frac{x-t}{1-t}$. 또한 $\frac{x-t}{1-t}\leq x$ 는 $-t\leq x(1-t)$ 와 동치이고, 이는 $t(1-x)\geq0$ 이므로 성립한다.

$-1<x<0$ 인 경우: $x\leq t\leq0$이다. 이때도 직접 정리하면 $\left|\frac{x-t}{1-t}\right| \leq|x|$ 임을 얻는다.

따라서 모든 $-1<x<1$에서 $\left| \frac{x-t}{1-t} \right| \leq|x|$

문제 10-(d): 임의의 $x\in(-1,1)$에 대하여 $\lim_{N\to\infty}E_N(x)=0$ 임을 보여라.

일반적인 도함수는 $f^{(N+1)}(t) = \frac{1\cdot3\cdots(2N+1)} {2^{N+1}} (1-t)^{-(N+3/2)} $ 이다. 한편 $c_{N+1} = \frac{1\cdot3\cdots(2N+1)} {2^{N+1}(N+1)!}$ 이므로

$$
\frac{f^{(N+1)}(t)}{N!} = (N+1)c_{N+1}(1-t)^{-(N+3/2)}.
$$

따라서 적분 나머지항은

$$
E_N(x) = (N+1)c_{N+1}
\int_0^x
\frac{(x-t)^N}{(1-t)^{N+3/2}}\ dt
$$

이를 다음과 같이 변형한다.

$$
\boxed{
E_N(x) = (N+1)c_{N+1}
\int_0^x
\left(\frac{x-t}{1-t}\right)^N
\frac{1}{(1-t)^{3/2}}\ dt
}
$$

앞에서 증명한 부등식

$$
\left|
\frac{x-t}{1-t}
\right|
\leq|x|
$$

을 사용하면

$$
|E_N(x)| \leq
(N+1)c_{N+1}|x|^N
\left|
\int_0^x
\frac{1}{(1-t)^{3/2}}\ dt
\right|\ =
2(N+1)c_{N+1}|x|^N
\left|
\frac{1}{\sqrt{1-x}}-1
\right|
$$

$c_{N+1}$은 양수이고 $0<c_{N+1}\leq1$ 이므로 $|E_N(x)| \leq 2(N+1)|x|^N \left| \frac{1}{\sqrt{1-x}}-1 \right|$

고정된 $x\in(-1,1)$에 대하여 $|x|<1$이므로 $\lim_{N\to\infty}(N+1)|x|^N=0$ 이다. 따라서 조임정리에 의해 $\lim_{N\to\infty}E_N(x)=0$ 이다.

결국 모든 $|x|<1$에 대하여

$$
\boxed{
\frac{1}{\sqrt{1-x}} = \sum_{n=0}^{\infty}c_nx^n
}
$$

이 성립한다. 즉,

$$
\boxed{
\frac{1}{\sqrt{1-x}} = \sum_{n=0}^{\infty}
\frac{(2n)!}{2^{2n}(n!)^2}x^n,
\qquad |x|<1
}
$$

이다.

라그랑주 나머지항으로는 $|x|<1/2$까지만 직접 증명할 수 있었지만, 적분 나머지항을 사용하면 정확한 수렴구간 $|x|<1$ 전체를 다룰 수 있다.

---

3. $\boldsymbol{\arcsin x}$의 테일러 급수

앞에서 $\frac{1}{\sqrt{1-x}} = \sum_{n=0}^{\infty}c_nx^n,\ |x|<1$ 을 증명했다.

여기에서 $x$ 대신 $x^2$을 대입하면

$$
\boxed{
\frac{1}{\sqrt{1-x^2}} = \sum_{n=0}^{\infty}c_nx^{2n},
\qquad |x|<1
}
$$

을 얻는다.

한편 $(\arcsin x)' = \frac{1}{\sqrt{1-x^2}}$ 이다. 따라서 이 급수를 항별로 적분하면 $\arcsin x$의 테일러 급수를 얻을 수 있다. 다만 무한급수를 항별로 미분하거나 적분할 때는 그 연산이 정당한지를 확인해야 한다.

문제 11: 다음을 증명하여라.

$$
\arcsin x = \sum_{n=0}^{\infty}
\frac{c_n}{2n+1}x^{2n+1},
\qquad |x|<1.
\tag{5}
$$

$|x|<1$에서 $\frac{1}{\sqrt{1-x^2}} = \sum_{n=0}^{\infty}c_nx^{2n}$ 이다.

$|x|<1$을 고정하면 $0$과 $x$를 포함하는 더 작은 닫힌구간에서 이 멱급수는 고르게 수렴한다. 따라서 항별적분이 가능하다.

양변을 $0$에서 $x$까지 적분하면

$$
\int_0^x\frac{1}{\sqrt{1-t^2}}\ dt = \sum_{n=0}^{\infty}
c_n\int_0^x t^{2n}\ dt
$$

좌변은

$$
\int_0^x\frac{1}{\sqrt{1-t^2}}\ dt = \arcsin x-\arcsin0 = \arcsin x
$$

우변은

$$
\sum_{n=0}^{\infty}
\frac{c_n}{2n+1}x^{2n+1}
$$
따라서

$$
\boxed{
\arcsin x = \sum_{n=0}^{\infty}
\frac{c_n}{2n+1}x^{2n+1},
\qquad |x|<1
}
$$

$c_n$의 공식을 대입하면

$$
\boxed{
\arcsin x = \sum_{n=0}^{\infty}
\frac{(2n)!}{2^{2n}(n!)^2(2n+1)}
x^{2n+1},
\qquad |x|<1
}
$$

이다.

처음 몇 항을 쓰면

$$
\boxed{
\arcsin x = x+\frac{x^3}{6}
+\frac{3x^5}{40}
+\frac{5x^7}{112}
+\cdots
}
$$

이다.

---

문제 12: 식 (5)가 닫힌구간 $[-1,1]$에서 $\arcsin x$로 고르게 수렴함을 설명하여라.

식 (5)의 일반항을 $g_n(x) = \frac{c_n}{2n+1}x^{2n+1}$ 이라고 하자. $x\in[-1,1]$이면 $|x^{2n+1}|\leq1$ 이므로 $|g_n(x)| \leq \frac{c_n}{2n+1}$

문제 5와 문제 7의 결과에 의해 $c_n\sqrt n\longrightarrow\frac{1}{\sqrt{\pi}}$. 따라서 $c_n\sim\frac{1}{\sqrt{\pi n}}$ 이고 $\frac{c_n}{2n+1} \sim \frac{1}{2\sqrt{\pi}}\frac{1}{n^{3/2}}$

급수 $\sum_{n=1}^{\infty}\frac{1}{n^{3/2}}$ 은 수렴하므로 극한비교판정법에 의해 $\sum_{n=0}^{\infty}\frac{c_n}{2n+1}$ 도 수렴한다.

그러므로 모든 $x\in[-1,1]$에서

$$
\left|
\frac{c_n}{2n+1}x^{2n+1}
\right|
\leq
\frac{c_n}{2n+1}
$$

이고, 우변의 급수가 수렴한다. 바이어슈트라스 $M$-판정법에 의해

$$
\sum_{n=0}^{\infty} \frac{c_n}{2n+1}x^{2n+1}
$$

은 $[-1,1]$에서 고르게 수렴한다.

$(-1,1)$에서는 이미 그 합이 $\arcsin x$임을 알고 있다. 또한 $\arcsin x$는 $[-1,1]$에서 연속이고, 고르게 수렴하는 연속함수열의 극한도 연속이다. 따라서 양 끝점에서도 같은 등식이 성립한다.

결국

$$
\boxed{
\arcsin x = \sum_{n=0}^{\infty}
\frac{c_n}{2n+1}x^{2n+1} \tag 5
}
$$

은 닫힌구간 $[-1,1]$ 전체에서 고르게 성립한다. 특히 $x=1$에서는

$$
\frac{\pi}{2} = \sum_{n=0}^{\infty}\frac{c_n}{2n+1}
$$

이고, $x=-1$에서는

$$
-\frac{\pi}{2} = -\sum_{n=0}^{\infty}\frac{c_n}{2n+1}
$$

이다.

---

# 오일러 합 $\boldsymbol{\sum 1/n^2}$ 구하기

$-\pi/2\leq\theta\leq\pi/2$라고 하자. 이 구간에서는

$$
\arcsin(\sin\theta)=\theta
$$

이다. $\arcsin x$의 급수 (식 5) 에 $x=\sin\theta$를 대입하면

$$
\boxed{
\theta = \sum_{n=0}^{\infty}
\frac{c_n}{2n+1}
\sin^{2n+1}\theta
}
$$

를 얻는다.

$\arcsin x$의 급수가 $[-1,1]$에서 고르게 수렴하고 $\sin\theta\in[-1,1]$이므로, 위 급수도 $[-\pi/2,\pi/2]$에서 고르게 수렴한다. 따라서 이 급수를 항별로 적분할 수 있다.

---

문제 13-(a)

다음을 보여라.

$$
\int_0^{\pi/2}\theta \ d\theta = \sum_{n=0}^{\infty}
\frac{c_n}{2n+1}b_{2n+1}.
$$

앞에서 얻은 급수

$$
\theta = \sum_{n=0}^{\infty}
\frac{c_n}{2n+1}\sin^{2n+1}\theta
$$

를 $0$에서 $\pi/2$까지 적분한다.

급수가 고르게 수렴하므로 항별적분이 가능하다. 따라서

$$
\int_0^{\pi/2}\theta \ d\theta =
\int_0^{\pi/2}
\sum_{n=0}^{\infty}
\frac{c_n}{2n+1}
\sin^{2n+1}\theta \ d\theta\ =
\sum_{n=0}^{\infty}
\frac{c_n}{2n+1}
\int_0^{\pi/2}\sin^{2n+1}\theta \ d\theta
$$

앞에서

$$
b_m = \int_0^{\pi/2}\sin^m\theta \ d\theta
$$

로 정의했으므로

$$
\int_0^{\pi/2}\sin^{2n+1}\theta \ d\theta = b_{2n+1}.
$$

따라서

$$
\boxed{
\int_0^{\pi/2}\theta \ d\theta = \sum_{n=0}^{\infty}
\frac{c_n}{2n+1}b_{2n+1}
}
$$

이다.

좌변은

$$
\int_0^{\pi/2}\theta \ d\theta = \left[\frac{\theta^2}{2}\right]_0^{\pi/2} = \frac{\pi^2}{8}
$$

이다.

문제 13-(b): 다음을 유도하여라.

$$
\frac{\pi^2}{8} = \sum_{n=0}^{\infty}\frac{1}{(2n+1)^2}.
$$

그리고 이를 이용하여

$$
\sum_{n=1}^{\infty}\frac{1}{n^2} = \frac{\pi^2}{6}
$$

을 증명하여라.

1단계: $c_nb_{2n+1}$ 계산

앞에서 구한 공식에 의해

$$
c_n = \frac{1\cdot3\cdots(2n-1)}
{2\cdot4\cdots(2n)}
$$

이고

$$
b_{2n+1} = \frac{2\cdot4\cdots(2n)}
{3\cdot5\cdots(2n+1)}
$$

이다.

두 식을 곱하면

$$
c_nb_{2n+1} =
\frac{1\cdot3\cdots(2n-1)}
{2\cdot4\cdots(2n)}
\frac{2\cdot4\cdots(2n)}
{3\cdot5\cdots(2n+1)}\ =
\frac{1}{2n+1}
$$

따라서

$$
\boxed{
c_nb_{2n+1}=\frac{1}{2n+1}
}
$$

이다.

문제 13-(a)의 식에 대입하면

$$
\frac{\pi^2}{8} =
\sum_{n=0}^{\infty}
\frac{c_n}{2n+1}b_{2n+1}\ =
\sum_{n=0}^{\infty}
\frac{1}{(2n+1)^2}
$$

그러므로

$$
\boxed{
\frac{\pi^2}{8} = \sum_{n=0}^{\infty}
\frac{1}{(2n+1)^2}
}
$$

이다. 즉,

$$
1+\frac1{3^2}+\frac1{5^2}+\frac1{7^2}+\cdots = \frac{\pi^2}{8}
$$

이다.

2단계: 전체 합 계산

전체 합을

$$
S=\sum_{n=1}^{\infty}\frac{1}{n^2}
$$

이라고 하자.

전체 합은 홀수항의 합과 짝수항의 합으로 분리된다.

$$
S = \sum_{n=0}^{\infty}\frac{1}{(2n+1)^2} + \sum_{n=1}^{\infty}\frac{1}{(2n)^2}.
$$

홀수항의 합은

$$
\sum_{n=0}^{\infty}\frac{1}{(2n+1)^2} = \frac{\pi^2}{8}
$$

이다.

짝수항의 합은

$$
\sum_{n=1}^{\infty}\frac{1}{(2n)^2} =
\frac14\sum_{n=1}^{\infty}\frac{1}{n^2}\ =
\frac14S
$$

따라서

$$
S=\frac{\pi^2}{8}+\frac14S.
$$

이를 정리하면

$$
\frac34S=\frac{\pi^2}{8}
$$

이므로 $S = \frac43\cdot\frac{\pi^2}{8} =\frac{\pi^2}{6}$ 결국 오일러 합은

$$
\boxed{
\sum_{n=1}^{\infty}\frac{1}{n^2} = \frac{\pi^2}{6}
}
$$


# 리만–제타 함수

오일러는 $\sum_{n=1}^{\infty}\frac{1}{n^2}$ 뿐 아니라 보다 일반적인 급수

$$
\sum_{n=1}^{\infty}\frac{1}{n^s}
$$

를 연구했다. 이를 변수 $s$에 관한 함수로 보면 리만–제타 함수가 된다.

$$
\boxed{
\zeta(s) = \sum_{n=1}^{\infty}\frac{1}{n^s},
\qquad s>1
}
$$

이다. 특히

$$
\zeta(2) = \sum_{n=1}^{\infty}\frac{1}{n^2} = \frac{\pi^2}{6}
$$

이다.

오일러는 $s=2$뿐 아니라 양의 짝수 $s=4,6,8,\ldots$에 대해서도 값을 계산했다. 예를 들면

$$
\zeta(4) = \sum_{n=1}^{\infty}\frac{1}{n^4} = \frac{\pi^4}{90}
$$

이다.

일반적으로 양의 짝수 $2m$에 대해서는

$$
\zeta(2m) = (-1)^{m+1}
\frac{B_{2m}(2\pi)^{2m}}{2(2m)!}
$$

이 성립한다. 여기서 $B_{2m}$은 베르누이 수이다.

리만-제타함수의 성질 중 으뜸은 다음 오일러 식에서 나타나는 소수와의 연계성이다:

$$\sum_{n=1}^{\infty}\frac{1}{n^s} = \left(\frac1{1-2^{-s}}\right)
\left(\frac1{1-3^{-s}}\right) \left(\frac1{1-5^{-s}}\right) \left(\frac1{1-7^{-s}}\right) \dots \tag 6$$

우변의 곱은 모든 소수에 대한 곱이다. 이를 다루는 기반 수학은 매우 복잡하지만, 등식 이해는 꽤 쉽게 할 수 있다. 각 소수 $p$에 대해 

$$ \frac1{1-p^{-s}} = 1+ \frac1{p^s}+\frac1{p^{2s}}+\frac1{p^{3s}}+\frac1{p^{4s}}+\dots$$

가 성립한다. 이런 식으로 식 (6)에서 우변의 곱셈을 곱하고, 모든 $n \in \mathbb N$에 대해 $n$을 소인수분해하는 방법은 유일하므로 위 식을 자연스럽게 유도할 수 있다.
