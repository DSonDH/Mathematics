# 푸리에 급수 (Fourier Series)

푸리에 급수는 주기 함수를 사인 함수와 코사인 함수의 무한한 합으로 나타내는 방법이다. 복잡한 주기 신호도 기본 주파수와 그 정수배 주파수인 고조파들의 조합으로 분석할 수 있다.

주기가 $2\pi$인 함수 $f(x)$는 적절한 조건에서 다음과 같이 표현된다.

$$
f(x) \sim \frac{a_0}{2} + \sum_{n=1}^{\infty}(a_n\cos nx+b_n\sin nx)
$$

$$
a_0=\frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\ dx,
\quad
a_n=\frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\cos nx\ dx,
\quad
b_n=\frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\sin nx\ dx
$$

- $f(x)$가 짝함수이면 $b_n=0$이므로 코사인 항만 남는다.
- $f(x)$가 홀함수이면 $a_0=a_n=0$이므로 사인 항만 남는다.

이 성질을 이용하면 적분 구간을 절반으로 줄여 계산할 수 있다.

### 계수 증명

함수 $f(x)$가 푸리에 급수로 다음과 같이 표현된다고 가정한다: $f(x) = a_0+\sum_{n=1}^{\infty} \left(a_n\cos(nx)+b_n\sin(nx)\right)$

모든 $m\geq1$에 대하여 다음 계수 공식을 유도하는 것이 목표다. 삼각함수들의 직교성을 이용한다.

$$
a_m=\frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\cos(mx)\ dx, \quad
b_m=\frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\sin(mx)\ dx
$$

1. **$a_m$의 공식 유도**

고정된 자연수 $m\geq1$을 잡는다. 푸리에 급수의 양변에 $\cos(mx)$를 곱한다.

$$
f(x)\cos(mx) =a_0\cos(mx)\ +\sum_{n=1}^{\infty}
a_n\cos(nx)\cos(mx)\ +\sum_{n=1}^{\infty}
b_n\sin(nx)\cos(mx)
$$

이제 양변을 $-\pi$에서 $\pi$까지 적분한다.

$$
\int_{-\pi}^{\pi}f(x)\cos(mx)\ dx =
a_0\int_{-\pi}^{\pi}\cos(mx)\ dx\ +
\sum_{n=1}^{\infty}a_n \int_{-\pi}^{\pi} \cos(nx)\cos(mx)\ dx \\ 
+ \sum_{n=1}^{\infty}b_n \int_{-\pi}^{\pi}
\sin(nx)\cos(mx)\ dx
$$

각 항을 살펴보면, $\int_{-\pi}^{\pi}\cos(mx)\ dx=0$ 
이므로 $a_0\int_{-\pi}^{\pi}\cos(mx)\ dx=0$

모든 $m,n\in\mathbb N$에 대하여 $\int_{-\pi}^{\pi}\sin(nx)\cos(mx)\ dx=0$ 이다. 따라서 $\sum_{n=1}^{\infty}b_n \int_{-\pi}^{\pi}\sin(nx)\cos(mx)\ dx=0$ 이다.

한편 

$$
\int_{-\pi}^{\pi}\cos(nx)\cos(mx)\ dx = 
\begin{cases}
0,&n\neq m,\\
\pi,&n=m
\end{cases}
$$

이므로 무한합에서 $n=m$인 항만 남는다: $\sum_{n=1}^{\infty}a_n \int_{-\pi}^{\pi}\cos(nx)\cos(mx)\ dx =a_m\int_{-\pi}^{\pi}\cos^2(mx)\ dx\ =a_m\pi$  
결국 $\int_{-\pi}^{\pi}f(x)\cos(mx)\ dx = \pi a_m$ 이다.  

양변을 $\pi$로 나누면 $a_m$ 증명 끝.

2. **$b_m$의 공식 유도**

이번에는 푸리에 급수의 양변에 $\sin(mx)$를 곱한다.

$$
f(x)\sin(mx) =a_0\sin(mx)\ +\sum_{n=1}^{\infty} a_n\cos(nx)\sin(mx)\ +\sum_{n=1}^{\infty} b_n\sin(nx)\sin(mx)
$$

양변을 $-\pi$에서 $\pi$까지 적분한다.

$$
\int_{-\pi}^{\pi}f(x)\sin(mx)\ dx =
a_0\int_{-\pi}^{\pi}\sin(mx)\ dx\ + \sum_{n=1}^{\infty}a_n \int_{-\pi}^{\pi} \cos(nx)\sin(mx)\ dx \\ + \sum_{n=1}^{\infty}b_n \int_{-\pi}^{\pi} \sin(nx)\sin(mx)\ dx
$$

각 항을 살펴본다.

$\sin(mx)$는 홀함수이므로 $\int_{-\pi}^{\pi}\sin(mx)\ dx=0$ 이다. 따라서  $a_0\int_{-\pi}^{\pi}\sin(mx)\ dx=0$ 이다.

모든 $m,n\in\mathbb N$에 대하여 $\int_{-\pi}^{\pi}\cos(nx)\sin(mx)\ dx=0$ 이다. 따라서 코사인에 관한 모든 항이 사라진다. $\sum_{n=1}^{\infty}a_n \int_{-\pi}^{\pi}\cos(nx)\sin(mx)\ dx=0$

$$
\int_{-\pi}^{\pi}\sin(nx)\sin(mx)\ dx = 
\begin{cases}
0,&n\neq m,\\
\pi,&n=m
\end{cases}
$$

이다. 따라서 무한합에서 $n=m$인 항만 남는다: $\sum_{n=1}^{\infty}b_n \int_{-\pi}^{\pi}\sin(nx)\sin(mx)\ dx = b_m\int_{-\pi}^{\pi}\sin^2(mx)\ dx\ =b_m\pi$

그러므로$\int_{-\pi}^{\pi}f(x)\sin(mx)\ dx = \pi b_m $ 이다.  

양변을 $\pi$로 나누면 $b_m$ 를 얻는다.

3. **상수항 $a_0$의 공식**

푸리에 급수의 양변을 $-\pi$에서 $\pi$까지 적분한다.

$$
\int_{-\pi}^{\pi}f(x)\ dx =
\int_{-\pi}^{\pi}
\left[
a_0+
\sum_{n=1}^{\infty}
\left(
a_n\cos(nx)+b_n\sin(nx)
\right)
\right]dx
$$

삼각함수 항들의 적분은 모두 $0$이므로

$$
\int_{-\pi}^{\pi}f(x)\ dx =  \int_{-\pi}^{\pi}a_0\ dx =2\pi a_0
$$

>**직교 관련 해석**  
>
>푸리에 급수에 $\cos(mx)$를 곱해 적분하면, 직교성 때문에 $\cos(mx)$와 주파수가 다른 모든 항이 사라진다.
>
>$$
>\int_{-\pi}^{\pi}\cos(nx)\cos(mx)\ dx=0
>\qquad(n\neq m)
>$$
>
>그러나 자기 자신과의 내적은 $0$이 아니다. $\int_{-\pi}^{\pi}\cos^2(mx)\ dx=\pi$
>
>따라서 $a_m$이 곱해진 항만 살아남아 $\pi a_m$이 된다.
>
>마찬가지로 $\sin(mx)$를 곱해 적분하면 $b_m$이 곱해진 항만 남는다.
>
>즉, 유한차원 벡터에서 벡터를 직교기저 방향으로 정사영하여 좌표를 구하는 것과 동일하다.
>
>벡터 $\mathbf v$를 직교벡터 $\mathbf e_m$ 방향으로 정사영할 때 그 계수는
>
>$$
>\frac{\mathbf v\cdot\mathbf e_m}
>{\mathbf e_m\cdot\mathbf e_m}
>$$
>
>이다. 함수의 경우에는 내적이 적분이므로
>
>$$
>a_m = 
>\frac{
>\left\langle f,\cos(mx)\right\rangle
>}{
>\left\langle\cos(mx),\cos(mx)\right\rangle
>}
>$$
>
>가 된다. 분자와 분모를 계산하면
>
>$$
>a_m = 
>\frac{
>\displaystyle\int_{-\pi}^{\pi}f(x)\cos(mx)\ dx
>}{
>\displaystyle\int_{-\pi}^{\pi}\cos^2(mx)\ dx
>}
>=
>\frac{1}{\pi}
>\int_{-\pi}^{\pi}f(x)\cos(mx)\ dx \\
>b_m = 
>\frac{
>\left\langle f,\sin(mx)\right\rangle
>}{
>\left\langle\sin(mx),\sin(mx)\right\rangle
>}
>=
>\frac{1}{\pi}
>\int_{-\pi}^{\pi}f(x)\sin(mx)\ dx
>$$
>
>이다.
>
>따라서 문제 3의 계산은 푸리에 계수가 함수 $f$를 각각의 삼각함수 방향으로 정사영한 값이라는 사실을 보여준다.

## 수렴
부분합을

$$S_N(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty}(a_n\cos nx+b_n\sin nx)$$

라 하자. $f(x)$를 삼각함수로 표현한다는 것은, 다음 등식을 만족하는 계수 $(a_n)_{n=0}^\infty$ 와 $(b_n)_{n=0}^\infty$ 을 찾는것을 말한다.

$$f(x) = \lim_{N\to \infty} S_N(x)$$

위 수렴은 어떤 종류의 수렴을 말하는건가?
- 점별수렴? 고른수렴?
- $L^2$수렴: $\int_{-\pi}^\pi |S_N(x)-f(x)|^2 \ dx \to 0$
- 체사로 평균 수렴 (Cesaro mean convergence): 부분합의 평균이 $f(x)$로 고르게 수렴함을 보임

### 예제: 문제 4

(a) $0$을 포함하는 구간에서 고른 수렴이 아닌 이유

함수는 다음과 같다.

$$
f(x)=
\begin{cases}
1,&0<x<\pi,\\
0,&x=0\text{ 또는 }x=\pi,\\
-1,&-\pi<x<0
\end{cases}
$$

푸리에 부분합은

$$
S_N(x) =
\frac{4}{\pi}
\sum_{k=0}^{N}
\frac{\sin((2k+1)x)}{2k+1}
$$

이다.

각 $S_N$은 유한 개의 사인함수 합이므로 $\mathbb R$에서 연속이다.

고른 수렴의 중요한 정리: 연속함수열 $(S_N)$이 어떤 구간 $I$에서 함수 $f$로 고르게 수렴하면, 극한함수 $f$도 $I$에서 연속이어야 한다. 그런데 $f$는 $x=0$에서 불연속이다. 실제로 $f(0)=0$ 이지만 $\lim_{x\to0^-}f(x)=-1, \quad \lim_{x\to0^+}f(x)=1$ 이다.

따라서 $0$을 포함하는 비자명한 구간에서 $f$는 연속이 아니므로, 연속함수인 부분합 $S_N$이 $f$로 고르게 수렴할 수 없다.

**직접적인 모순**

모든 $N$에 대하여 $S_N(0)=0$ 이다. 또한 $S_N$은 연속이므로, 각 $N$에 대해 $0$에 충분히 가까운 양수 $x_N$을 잡으면 $|S_N(x_N)|<\frac12$ 가 되게 할 수 있다. 그러나 $x_N>0$이므로 $f(x_N)=1$ 이다.  
따라서 $|S_N(x_N)-f(x_N)| > \frac12$ 이다. 즉, $\sup_{x\in I}|S_N(x)-f(x)| \geq\frac12$ 가 되어 오차의 최댓값이 $0$으로 갈 수 없다. 따라서 고른 수렴이 아니다.

(불연속점 주변에서 나타나는 이러한 진동과 초과 현상을 깁스 현상이라고 한다.)

---

(b) $g(x)=|x|$의 푸리에 급수

$g(x)=|x|$를 $[-\pi,\pi]$에서 정의하고 주기 $2\pi$가 되도록 확장한다.

푸리에 급수를 다음과 같이 놓는다. $g(x) = a_0+ \sum_{n=1}^{\infty} (a_n\cos(nx)+b_n\sin(nx))$  
$g$는 짝함수다. $g(-x)=|-x|=|x|=g(x)$ 따라서 $g(x)\sin(nx)$는 짝함수와 홀함수의 곱이므로 홀함수다. 그러므로 $b_n = \frac1\pi \int_{-\pi}^{\pi}|x|\sin(nx)\ dx =0$ 이다. 즉, $g$의 푸리에 급수에는 사인항이 나타나지 않는다.

상수항 $a_0$

$$
a_0= \frac{1}{2\pi}
\int_{-\pi}^{\pi}|x|\ dx = \frac{1}{2\pi}
\cdot2\int_0^\pi x\ dx = \frac1\pi
\left[\frac{x^2}{2}\right]_0^\pi = \frac1\pi\cdot\frac{\pi^2}{2} =\frac{\pi}{2}
$$

코사인 계수 $a_n$

$$
a_n =
\frac1\pi
\int_{-\pi}^{\pi}|x|\cos(nx)\ dx
$$

$|x|\cos(nx)$는 짝함수이므로 $a_n =\frac{2}{\pi}\int_0^\pi x\cos(nx)\ dx$ 이다.

부분적분을 사용한다. $u=x, \quad dv=\cos(nx)\ dx$ 로 놓으면 $du=dx, \quad v=\frac{\sin(nx)}{n}$ 이다. 따라서

$$
\int_0^\pi x\cos(nx)\ dx= \left[\frac{x\sin(nx)}{n}\right]_0^\pi
-\frac1n\int_0^\pi\sin(nx)\ dx = 0+\frac1{n^2}
\left[\cos(nx)\right]_0^\pi \\
= \frac{\cos(n\pi)-1}{n^2} = \frac{(-1)^n-1}{n^2}
$$

그러므로 $a_n = \frac{2}{\pi} \frac{(-1)^n-1}{n^2}$ 이다. $n$이 짝수이면 $(-1)^n=1$이므로 $a_n=0$, $n$이 홀수이면 $(-1)^n=-1$이므로 $a_n = -\frac{4}{\pi n^2}$

따라서

$$
a_n=
\begin{cases}
-\dfrac{4}{\pi n^2},&n\text{이 홀수},\\
0,&n\text{이 짝수}
\end{cases}
$$


---

최종 푸리에 급수: 홀수 $n$을 $n=2k+1$로 표시하면

$$
\boxed{
|x| =
\frac{\pi}{2}
-\frac4\pi
\sum_{k=0}^{\infty}
\frac{\cos((2k+1)x)}{(2k+1)^2}
=
\frac{\pi}{2}
-\frac4\pi
\left(
\cos x
+\frac{\cos3x}{3^2}
+\frac{\cos5x}{5^2}
+\frac{\cos7x}{7^2}
+\cdots
\right)
}
$$

항이 증가할수록 부분합이 $|x|$의 V자 형태에 빠르게 가까워지는 것을 확인할 수 있다. 다만 $x=0$과 주기적 연결점 $x=\pm\pi$에서는 함수가 미분 가능하지 않으므로 근처에서 수렴이 상대적으로 느리다.

---

**계수만 보고 고른 수렴을 판단하는 방법**

코사인 계수의 절댓값은 $|a_n| \leq\frac{4}{\pi n^2}$ 을 만족한다. 또한 모든 $x$에 대하여 $|\cos(nx)|\leq1$ 이므로 $|a_n\cos(nx)| \leq \frac{4}{\pi n^2}$ 이다.

그런데 $\sum_{n=1}^{\infty}\frac1{n^2}<\infty$ 이므로 바이어슈트라스 $M$-판정법에 의하여 $\sum_{n=1}^{\infty}a_n\cos(nx)$ 는 $\mathbb R$에서 절대적으로, 그리고 고르게 수렴한다.

따라서 $|x|$의 푸리에 급수는 고르게 수렴한다 고 할 수 있다. 정확하게는 계수의 절댓값 합이 유한하면 충분하다.

$$
\boxed{
\sum_{n=1}^{\infty}
\left(|a_n|+|b_n|\right)<\infty
\Longrightarrow \text{푸리에 급수는 고르게 수렴한다}
}
$$

여기서는 $a_n=O(n^{-2})$이므로 이 조건을 만족한다.

반면 예제 8.5.1의 불연속함수에서는 계수가 $O(n^{-1})$이다.

$$
\sum_{k=0}^{\infty}\frac1{2k+1}
$$

은 발산하므로 절대수렴과 $M$-판정법을 이용할 수 없다. 실제로 그 급수는 불연속점이 포함된 구간에서 고르게 수렴하지 않는다.

---

(c) 푸리에 급수의 항별 미분

예제 8.5.1의 푸리에 급수는 $f(x) = \frac4\pi \sum_{k=0}^{\infty} \frac{\sin((2k+1)x)}{2k+1}$ 이다. 이를 형식적으로 항별 미분하면

$$
f'(x)
\stackrel{?}{=} \frac4\pi \sum_{k=0}^{\infty} \cos((2k+1)x)
$$

을 얻는다.

그러나 이 급수는 일반적으로 수렴하지 않는다. 특히 $x=0$에서는 $\cos((2k+1)0)=1$ 이므로

$$
\frac4\pi
\sum_{k=0}^{\infty}\cos((2k+1)0) =
\frac4\pi
\sum_{k=0}^{\infty}1
$$

이 되어 발산한다. 부분합도 다음과 같이 계산된다.

$$
\sum_{k=0}^{N}\cos((2k+1)x) =
\frac{\sin(2(N+1)x)}{2\sin x}
$$

이다. 일반적인 $x$에서는 분자가 계속 진동하므로 극한이 존재하지 않는다.

따라서 예제 8.5.1의 푸리에 급수를 항별 미분하는 것은 정당하지 않다.

$$
\boxed{
\frac{d}{dx}
\sum_{k=0}^{\infty}
\frac{\sin((2k+1)x)}{2k+1}
\neq
\sum_{k=0}^{\infty}
\cos((2k+1)x)
\text{라고 무조건 쓸 수 없다}
}
$$

---

**$g(x)=|x|$의 급수를 항별 미분**

$|x|$의 푸리에 급수는

$$
|x| =
\frac{\pi}{2}
-\frac4\pi
\sum_{k=0}^{\infty}
\frac{\cos((2k+1)x)}{(2k+1)^2}
$$

이다. 이를 형식적으로 항별 미분하면

$$
g'(x)
\stackrel{?}{=}
-\frac4\pi
\sum_{k=0}^{\infty}
\frac{-(2k+1)\sin((2k+1)x)}{(2k+1)^2} = \frac4\pi
\sum_{k=0}^{\infty}
\frac{\sin((2k+1)x)}{2k+1}
$$

을 얻는다. 오른쪽 급수는 바로 예제 8.5.1의 푸리에 급수다. 따라서

$$
\frac4\pi
\sum_{k=0}^{\infty}
\frac{\sin((2k+1)x)}{2k+1} =
\begin{cases}
-1,&-\pi<x<0,\\
0,&x=0,\\
1,&0<x<\pi
\end{cases}
$$

로 수렴한다.

이는 $|x|$의 실제 도함수와 일치한다.

$$
g'(x) =
\begin{cases}
-1,&-\pi<x<0,\\
1,&0<x<\pi
\end{cases}
$$

다만 $x=0$에서는 $|x|$가 미분 가능하지 않다. 주기적으로 확장했을 때의 연결점 $x=\pm\pi$에서도 미분 가능하지 않다.

따라서 다음과 같이 이해해야 한다.

$$
\boxed{
\frac{d}{dx}|x| =
\frac4\pi
\sum_{k=0}^{\infty}
\frac{\sin((2k+1)x)}{2k+1}
\qquad
(x\neq k\pi)
}
$$

여기서 우변은 불연속점 $x=k\pi$에서 좌우 극한의 평균인 $0$으로 수렴하지만, 이것이 그 점에서 $|x|$가 미분 가능하다는 뜻은 아니다.

---

**계수로 보는 항별 미분의 위험성**

푸리에 급수를 미분하면 각 항의 계수에 $n$이 곱해진다.

$$
\frac{d}{dx}\bigl(a_n\cos(nx)\bigr) =
-na_n\sin(nx)
$$

$$
\frac{d}{dx}\bigl(b_n\sin(nx)\bigr) =
nb_n\cos(nx)
$$

따라서 계수가 $a_n=O(n^{-p})$라면, 미분한 급수의 계수는 대략

$$
na_n=O(n^{-(p-1)})
$$

이 된다.

즉, 미분할 때마다 계수의 감소 속도가 한 단계 느려진다.

* 원래 계수가 $O(n^{-2})$이면 미분 후에는 $O(n^{-1})$이 된다.
* 원래 계수가 $O(n^{-1})$이면 미분 후에는 $O(1)$이 되어 일반적으로 항 자체가 $0$으로 가지 않는다.

예제 8.5.1에서는 계수가 $O(n^{-1})$이므로 미분 후 계수가 $O(1)$이 되어 급수가 발산한다.

$|x|$의 경우에는 계수가 $O(n^{-2})$이므로 미분 후 $O(n^{-1})$이 된다. 미분한 급수는 점별로 수렴하지만, 불연속점을 포함하는 구간에서는 고르게 수렴하지 않는다.

---

정리 '**미분가능성과 함수급수 *(Differentiation of Series of Functions)*'을 적용할 수 있는가?**

항별 미분 정리의 전형적인 가정은 다음과 같다.

* 각 항이 미분 가능하다.
* 미분한 급수가 해당 구간에서 고르게 수렴한다.
* 원래 급수가 적어도 한 점에서 수렴한다.

그러면 원래 급수의 합은 미분 가능하고, 합의 도함수는 미분한 급수의 합과 같다.

예 8.5.1에서는 미분한 급수 $\frac4\pi\sum_{k=0}^{\infty}\cos((2k+1)x)$ 가 일반적으로 발산하므로 정리를 적용할 수 없다.

$g(x)=|x|$: 미분한 급수는 $\frac4\pi \sum_{k=0}^{\infty} \frac{\sin((2k+1)x)}{2k+1}$ 이다. 이 급수는 $x=0$에서 불연속인 함수로 수렴하므로 $0$을 포함하는 구간에서는 고르게 수렴하지 않는다. 따라서 $[-\pi,\pi]$ 전체에는 정리 6.4.3을 적용할 수 없다.

결론적으로 $[-\pi,\pi]$ 전체에서는 두 예제 모두 정리 6.4.3을 적용할 수 없다

다만 $|x|$의 경우 $0$과 주기적 연결점을 피한 닫힌구간에서는 미분한 급수가 고르게 수렴한다. 예를 들어 $[\delta,\pi-\delta], \quad [-\pi+\delta,-\delta] \quad(\delta>0)$ 에서는 정리를 국소적으로 적용할 수 있다. 이 구간들에서는 각각 $g'(x)=1, \quad g'(x)=-1$ 을 얻는다.

### 정리. 리만-르벡 보조정리(Riemann-Lebesgue lemma)
$h(x)$가 $(-\pi, \pi]$에서 연속이며, $2\pi$주기로도 실수 전체에서 연속이라 가정하자. $n \to \infty$일때 다음이 성립한다:

$$
\int_{-\pi}^\pi h(x)\sin{(nx)}\ dx \to 0, \quad
\int_{-\pi}^\pi h(x)\cos{(nx)}\ dx \to 0
$$

- $n$이 커지면 진동주기가 매우 짧아지고, $h(x)$가 연속함수면 $\sin(nx)$가 짧은 주기를 한 번씩 도는 종안 $h$의 함숫값은 크게 변하지 않는다. 즉, 양의 진동과 음의 진동이 상쇄를 자주 일으켜 0으로 수렴한다.


**증명**


1. $h$가 $\mathbb R$에서 고른연속임을 보이자.

$h$가 주기 $2\pi$라는 것은 모든 $x\in\mathbb R$에 대하여 $h(x+2\pi)=h(x)$ 라는 뜻이다. 원래 $h$를 구간 $(-\pi,\pi]$에서 생각한 다음, 주기적으로 확장한다. 이 확장이 $\mathbb R$에서 연속이려면 양 끝이 자연스럽게 연결되어야 한다.

$$
\lim_{x\to-\pi^+}h(x)=h(\pi)
$$

이 조건을 포함하여 주기적으로 확장된 $h$가 $\mathbb R$에서 연속이라고 가정한다.

닫힌 유계구간에서의 고른 연속성: $h$는 $\mathbb R$에서 연속이므로 닫힌 유계구간 $[-2\pi,2\pi]$ 에서 연속이다. 하이네–칸토어 정리에 의하여 닫힌 유계구간에서 연속인 함수는 고른 연속이다. 따라서 임의의 $\varepsilon>0$에 대하여 어떤 $\delta_0>0$가 존재하여 $u,v\in[-2\pi,2\pi], \quad |u-v|<\delta_0$ 이면 $|h(u)-h(v)|<\varepsilon$ 이다.

필요하다면 $\delta_0$를 더 작게 잡을 수 있으므로 $0<\delta\leq\min{\delta_0,\pi}$ 로 놓는다.

임의의 $x,y\in\mathbb R$에 적용: 이제 임의의 $x,y\in\mathbb R$이 $|x-y|<\delta$ 를 만족한다고 하자. 주기성을 이용하면 어떤 정수 $k\in\mathbb Z$를 선택하여 $x-2k\pi\in[-\pi,\pi]$ 가 되게 할 수 있다.

다음과 같이 놓는다. $u=x-2k\pi, \quad v=y-2k\pi$ 그러면 $u\in[-\pi,\pi]$ 이고

$$
|u-v|= |(x-2k\pi)-(y-2k\pi)| = |x-y| < \delta \leq\pi
$$

이다. 따라서 $v\in[-2\pi,2\pi]$ 이다. 결국 $u,v\in[-2\pi,2\pi]$이고 $|u-v|<\delta_0$이므로

$$
|h(u)-h(v)|<\varepsilon
$$

이다.

그런데 $h$는 주기 $2\pi$이므로 $h(u)=h(x-2k\pi)=h(x)$ 이고 $h(v)=h(y-2k\pi)=h(y)$ 이다. 따라서

$$
|h(x)-h(y)|= |h(u)-h(v)| < \varepsilon
$$

이므로 $h$ 는 $\mathbb R$ 에서 고른 연속이다

---

2. $|\int_a^b h(x)\sin(nx)\ dx| < \frac{\varepsilon}{n}$ 증명

$h$는 $\mathbb R$에서 고른 연속이고 주기 $2\pi$인 함수다. 임의의 $\varepsilon>0$을 잡는다. 이전에 확인한 고른 연속성에 의하여 어떤 $\delta>0$가 존재하여 $|x-y|<\delta$ 이면 $|h(x)-h(y)|<\frac{\varepsilon}{2}$ 이다.

$n$을 충분히 크게 잡아 $\frac{\pi}{n}<\delta$ 가 되게 한다.

이제 길이가 $\frac{2\pi}{n}$ 인 구간 $[a,b]$를 생각한다. 즉, $b-a=\frac{2\pi}{n}$ 이다. 그러면 $\sin(nx)$의 주기는 $2\pi/n$이므로 $\sin(nx)$는 $[a,b]$에서 정확히 한 번 진동한다.

- 적분을 두 구간으로 나누기

구간 $[a,b]$의 중점을 $c=a+\frac{\pi}{n}$ 이라고 하자. 그러면 $b=a+\frac{2\pi}{n}$ 이므로

$$
\int_a^b h(x)\sin(nx)\ dx= \int_a^{a+\pi/n}h(x)\sin(nx)\ dx
+
\int_{a+\pi/n}^{a+2\pi/n}h(x)\sin(nx)\ dx
$$

두 번째 적분에서 $x=y+\frac{\pi}{n}$

으로 치환한다. 그러면 $dx=dy$ 이고, 적분구간은 $x=a+\frac{\pi}{n} \Longrightarrow y=a, \quad x=a+\frac{2\pi}{n} \Longrightarrow y=a+\frac{\pi}{n}$ 으로 변한다.

따라서

$$
\int_{a+\pi/n}^{a+2\pi/n}h(x)\sin(nx)\ dx =
\int_a^{a+\pi/n} h\left(y+\frac{\pi}{n}\right) \sin\left(n\left(y+\frac{\pi}{n}\right)\right)\ dy
$$

이다.

그런데 $\sin\left(n\left(y+\frac{\pi}{n}\right)\right)= \sin(ny+\pi) = -\sin(ny)$ 이다. 따라서

$$
\int_{a+\pi/n}^{a+2\pi/n}h(x)\sin(nx)\ dx\
=
-\int_a^{a+\pi/n}
h\left(y+\frac{\pi}{n}\right)\sin(ny)\ dy
$$

이다.

적분변수 $y$를 다시 $x$로 바꾸어 쓰면

$$
\int_{a+\pi/n}^{a+2\pi/n}h(x)\sin(nx)\ dx= -\int_a^{a+\pi/n}
h\left(x+\frac{\pi}{n}\right)\sin(nx)\ dx
$$

이다.

- 양의 진동과 음의 진동을 상쇄하기

두 적분을 합하면

$$
\int_a^b h(x)\sin(nx)\ dx =
\int_a^{a+\pi/n}h(x)\sin(nx)\ dx -
\int_a^{a+\pi/n} h\left(x+\frac{\pi}{n}\right)\sin(nx)\ dx\
\\ = \int_a^{a+\pi/n} \left[ h(x)-h\left(x+\frac{\pi}{n}\right) \right] \sin(nx)\ dx
$$

이다. 이 식이 증명의 핵심이다. $\sin(nx)$의 양의 반주기와 음의 반주기를 서로 짝지으면 $h$의 값의 차이만 남는다.

- 고른 연속성 적용

$\pi/n<\delta$이므로 $|x-\left(x+\frac{\pi}{n}\right)|= \frac{\pi}{n} < \delta$ 이다. 따라서 고른 연속성에 의하여

$$
\left| h(x)-h\left(x+\frac{\pi}{n}\right) \right| < \frac{\varepsilon}{2}
$$

이다.

그러므로

$$
\left| \int_a^b h(x)\sin(nx)\ dx \right| =
\left| \int_a^{a+\pi/n} \left[h(x)-h\left(x+\frac{\pi}{n}\right) \right] \sin(nx)\ dx \right|\\
\leq \int_a^{a+\pi/n} \left| h(x)-h\left(x+\frac{\pi}{n}\right) \right| |\sin(nx)|\ dx\
< \frac{\varepsilon}{2} \int_a^{a+\pi/n}|\sin(nx)|\ dx
$$

- 사인 절댓값의 적분

길이가 $\pi/n$인 구간에서 $\sin(nx)$는 반주기를 지난다. 따라서 $\int_a^{a+\pi/n}|\sin(nx)|\ dx= \frac{2}{n}$ 이다. 따라서


$$
\left| \int_a^b h(x)\sin(nx)\ dx \right|
< \frac{\varepsilon}{2}\cdot\frac2n\ =\frac{\varepsilon}{n}
$$


- 리만–르베그 보조정리 증명 완성

이제 구간 $[-\pi,\pi]$를 길이가 $2\pi/n$인 $n$개의 구간으로 나눈다.

$$
x_j=-\pi+\frac{2\pi j}{n}, \quad j=0,1,\ldots,n
$$

라고 놓으면 $[-\pi,\pi]= \bigcup_{j=1}^{n}[x_{j-1},x_j]$ 이고 각 구간의 길이는 $x_j-x_{j-1}=\frac{2\pi}{n}$ 이다.

따라서 이전 결과에 의해 각 $j$에 대하여

$$
\left| \int_{x_{j-1}}^{x_j} h(x)\sin(nx)\ dx \right|
< \frac{\varepsilon}{n}
$$

이다.

전체 구간에서의 적분은

$$
\int_{-\pi}^{\pi}h(x)\sin(nx)\ dx= \sum_{j=1}^{n}
\int_{x_{j-1}}^{x_j}
h(x)\sin(nx)\ dx
$$

이므로 삼각부등식을 적용하면

$$
\left| \int_{-\pi}^{\pi}h(x)\sin(nx)\ dx \right|
\leq \sum_{j=1}^{n}
\left| \int_{x_{j-1}}^{x_j} h(x)\sin(nx)\ dx \right|\
< \sum_{j=1}^{n}\frac{\varepsilon}{n}\ = n\cdot\frac{\varepsilon}{n}\ =\varepsilon
$$

이다. 따라서 충분히 큰 모든 $n$에 대하여 $| \int_{-\pi}^{\pi}h(x)\sin(nx)\ dx | < \varepsilon$이다. 이는

$$
\boxed{
\int_{-\pi}^{\pi}h(x)\sin(nx)\ dx
\longrightarrow0
}
$$

임을 의미한다.

- 코사인 적분

코사인에 대해서도 같은 방법을 사용할 수 있다. 핵심 관계는 $\cos(n(x+\pi/n))= \cos(nx+\pi)= -\cos(nx)$ 이다. 따라서 길이가 $2\pi/n$인 구간 $[a,b]$에 대하여

$$
\int_a^b h(x)\cos(nx)\ dx= \int_a^{a+\pi/n}
\left[
h(x)-h\left(x+\frac{\pi}{n}\right)
\right]
\cos(nx)\ dx
$$

를 얻는다.

사인에서와 동일한 추정을 적용하면

$$
\left| \int_a^b h(x)\cos(nx)\ dx \right| < \frac{\varepsilon}{n}
$$

이고, $[-\pi,\pi]$를 $n$개의 구간으로 나누면

$$
\boxed{
\int_{-\pi}^{\pi}h(x)\cos(nx)\ dx
\longrightarrow0
}
$$

를 얻는다.

결론적으로 리만–르베그 보조정리가 성립한다.

$$
\boxed{
\int_{-\pi}^{\pi}h(x)\sin(nx)\ dx\to0,
\qquad
\int_{-\pi}^{\pi}h(x)\cos(nx)\ dx\to0
}
$$

이는 진동수가 커질수록 $\sin(nx)$와 $\cos(nx)$의 양의 부분과 음의 부분이 $h(x)$를 곱한 뒤에도 거의 상쇄된다는 뜻이다.

### 정리.

함수 $f$가 $2\pi$-주기이고 $x$에서 미분 가능하다고 하자. 푸리에 급수의 $N$번째 부분합을

$$
S_N(x)=\frac{a_0}{2}+\sum_{n=1}^{N}\left(a_n\cos(nx)+b_n\sin(nx)\right)
$$

로 정의하면, $f'(x)$가 존재하는 모든 $x \in (-\pi, \pi]$에서 다음과 같이 점별수렴한다:

$$
\boxed{
\lim_{N\to\infty}S_N(x)=f(x)
}
$$

- 증명을 위해서는 디리클레 핵(Dirichlet kernel) 항등식이 필요하다.
  - $\int_{-\pi}^{\pi}D_N(\theta) \ d\theta = \pi$가 성립합도 필요함
- 리만-르벡보조정리도 필요함
- 제거가능한 특이점을 고려한 적분도 필요

> 추가: 페예르 정리, 페예르 커널




## 5. 복소 푸리에 급수

오일러 공식 $e^{inx}=\cos nx+i\sin nx$를 사용하면 다음과 같이 쓸 수 있다.

$$
f(x) \sim \sum_{n=-\infty}^{\infty}c_ne^{inx},
\qquad
c_n=\frac{1}{2\pi}\int_{-\pi}^{\pi}f(x)e^{-inx}\ dx
$$

복소형은 미분 방정식, 신호 처리, 양자역학 등에서 계산을 간결하게 해 준다.

## 6. 활용

푸리에 급수는 열전도 방정식과 파동 방정식의 해를 구하고, 소리와 전기 신호를 주파수별로 분석하며, 이미지 압축과 필터 설계에 활용된다. 비주기 함수는 푸리에 변환을 통해 연속적인 주파수 성분으로 분석할 수 있다.

