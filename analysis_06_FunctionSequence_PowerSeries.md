함수열·멱급수
# 1. 정의

## Def. 1. [함수열과 함수급수] *(Sequence of Functions and Series of Functions)*

$\varphi:D\subset\mathbb R\to\mathbb R$이고 모든 $n\in\mathbb N$에 대하여
$$
f_n:D\to\mathbb R
$$
일 때 ${f_n}$을 $D$에서의 **함수열** *(sequence of functions)* 이라 한다.

또한 ${f_n}$이 함수열일 때
$$
\sum_{n=1}^\infty f_n
$$
을 **함수급수** *(series of functions)* 라 한다.

## Def. 2. [멱급수] *(Power Series)*

실수 $c$와 수열 ${a_n}$에 대하여 함수열 ${f_n}$이
$$
f_n(x)=a_n(x-c)^n
$$
과 같이 표현될 때, 함수급수
$$
\sum_{n=0}^\infty f_n = \sum_{n=0}^\infty a_n(x-c)^n
$$
을 **멱급수** *(power series)* 라 한다.

## Def. 3. [해석함수] *(Analytic Function)*

어떤 $\delta>0$에 대하여 $(c-\delta,c+\delta)$에서 함수 $f$가 멱급수
$$
f(x)=\sum_{n=0}^\infty a_n(x-c)^n
$$
로 표현될 수 있으면, $f$는 $x=c$에서 **해석적** *(analytic)* 이라 한다.

또한 함수 $f$가 열린구간 $I$의 모든 점에서 해석적이면
$f$를 $I$에서의 **해석함수**라 한다.

# 2. 점별수렴과 균등수렴 *(Pointwise and Uniform Convergence)*

## (1) 함수열의 수렴

### Def. [점별수렴과 균등수렴] *(Pointwise Convergence and Uniform Convergence)*

${f_n}$과 $f$가 $D$에서 정의된 함수라 하자.

① 임의의 $x\in D$와 $\varepsilon>0$에 대하여
$$
\exists N\in\mathbb N\ \text{s.t.}\ n\ge N \Rightarrow |f_n(x)-f(x)|<\varepsilon
$$
이면 ${f_n}$은 $D$에서 $f$로 **점별수렴** *(pointwise convergence)* 한다.

② 임의의 $\varepsilon>0$에 대하여
$$
\exists N\in\mathbb N\ \text{s.t.}\ \forall x\in D,\ n\ge N \Rightarrow |f_n(x)-f(x)|<\varepsilon
$$
이면 ${f_n}$은 $D$에서 $f$로 **균등수렴** *(uniform convergence)* 한다.

### Thm.

${f_n}$이 $D$에서 균등수렴하면 점별수렴한다.

## (2) 함수급수의 수렴

### Thm. 1. [코시판정법] *(Cauchy Criterion for Series of Functions)*

$f_n:D\to\mathbb R$라 할 때, 다음을 만족하면
$$
\sum_{n=1}^\infty f_n
$$
은 $D$에서 균등수렴한다.

$$
\forall\varepsilon>0,\ \exists N\in\mathbb N\ \text{s.t.}\
\forall m>n\ge N,\ \forall x\in D,\
\left|\sum_{k=n+1}^m f_k(x)\right|<\varepsilon
$$

### Thm. 2. [바이어슈트라스 판정법] *(Weierstrass M-test)*

각 $n\in\mathbb N$에 대해 $f_n:D\to\mathbb R$라 하자.
적당한 양의 상수 $M_n>0$가 존재하여
$$
|f_n(x)|\le M_n\quad(\forall x\in D)
$$
이고
$$
\sum_{n=1}^\infty M_n<\infty
$$
이면
$$
\sum_{n=1}^\infty f_n
$$
은 $D$에서 균등수렴한다.

# 3. 멱급수 *(Power Series)*

## (1) 멱급수의 수렴

### Thm. 1. [근판정법] *(Root Test)*

모든 $n\in\mathbb N$에 대해 $a_n\ge0$이고
$$
\lim_{n\to\infty}\sqrt[n]{a_n}=M
$$
일 때,

① $M<1$이면 $\sum_{n=1}^\infty a_n$은 수렴한다.
② $M>1$이면 $\sum_{n=1}^\infty a_n$은 발산한다.

### Cor. [멱급수의 수렴반경] *(Radius of Convergence)*

멱급수
$$
\sum_{n=0}^\infty a_n(x-c)^n
$$
에 대하여
$$
\alpha=\limsup_{n\to\infty}\sqrt[n]{|a_n|}
$$
이면 수렴반경 $R=\frac1\alpha$이다.

* $|x-c|<R$에서 절대수렴
* $|x-c|>R$에서 발산

$\alpha=0$이면 $R=\infty$,
$\alpha=\infty$이면 $R=0$으로 간주한다.

### Def. [수렴반지름과 수렴구간] *(Radius and Interval of Convergence)*

$R$을 멱급수의 **수렴반지름**이라 하고,
멱급수가 수렴하는 점들의 전체 집합을 **수렴구간**이라 한다.

### Thm. 1. [수렴반지름과 균등수렴] *(Uniform Convergence inside Radius)*

멱급수의 수렴반지름이 $R$이고 $0<r<R$이면
$$
\sum_{n=0}^\infty a_n(x-c)^n
$$
은 $[c-r,c+r]$에서 균등수렴한다.

## (2) 멱급수의 연속

### Thm. 1. [함수열의 연속] *(Continuity under Uniform Convergence)*

$I$에서 연속인 함수열 ${f_n}$이 $I$에서 $f$로 균등수렴하면
$f$는 $I$에서 연속이다.

### Cor. [함수급수의 연속] *(Continuity of Series of Functions)*

$I$에서 연속인 함수열 ${f_n}$에 대해
$$
\sum_{n=1}^\infty f_n
$$
이 $I$에서 균등수렴하면 그 합함수도 $I$에서 연속이다.

### Lemma. [아벨의 공식] *(Abel's Summation Formula)*

수열 ${a_k},{b_k}$와 자연수 $n,m$ $(n>m)$에 대하여
$$
\sum_{k=m+1}^n a_k b_k 
= a_n\sum_{k=m+1}^n b_k
- \sum_{j=m+1}^{n-1}(a_{j+1}-a_j)\sum_{k=m+1}^j b_k
$$

### Thm. 2. [아벨정리] *(Abel’s Theorem)*

멱급수
$$
\sum_{n=0}^\infty a_n(x-c)^n
$$
이 $x=c+R$에서 수렴하면 $(c-R,c+R)$의 임의의 부분집합에서 균등수렴한다.

### Thm. 3. [멱급수의 연속] *(Continuity of Power Series)*

멱급수
$$
f(x)=\sum_{n=0}^\infty a_n(x-c)^n
$$
는 수렴구간에서 연속이다.

## (3) 멱급수의 미분 *(Differentiation of Power Series)*

### Thm. 1. [함수열의 미분] *(Differentiation under Uniform Convergence)*

함수열 ${f_n}$이 다음을 만족하면

* 어떤 $x_0\in I$에서 ${f_n(x_0)}$가 수렴
* ${f_n}$이 $I$에서 미분가능
* ${f_n'}$이 $I$에서 균등수렴

$f_n\to f$라 할 때 $f$는 $I$에서 미분가능하며
$$
f'(x)=\lim_{n\to\infty}f_n'(x)
$$

### Cor. [함수급수의 미분] *(Term-by-term Differentiation)*

다음이 성립하면
$$
\sum_{n=1}^\infty f_n
$$
은 $I$에서 미분가능하다.

① $\sum f_n(x_0)$가 수렴
② $\sum f_n'$이 $I$에서 균등수렴

이때
$$
\left(\sum_{n=1}^\infty f_n\right)'
=\sum_{n=1}^\infty f_n'
$$

### Thm. 3. [멱급수의 미분] *(Differentiation of Power Series)*

멱급수
$$
f(x)=\sum_{n=0}^\infty a_n(x-c)^n
$$
의 수렴반지름이 $R$이면 $(c-R,c+R)$에서 미분가능하며
$$
f'(x)=\sum_{n=1}^\infty n a_n(x-c)^{n-1}
$$

## (4) 멱급수의 적분 *(Integration of Power Series)*

### Thm. 1. [균등수렴과 적분] *(Integration under Uniform Convergence)*

${f_n}$이 $[a,b]$에서 $f$로 균등수렴하고
$f_n\in\mathcal R[a,b]$이면 $f\in\mathcal R[a,b]$이며
$$
\lim_{n\to\infty}\int_a^b f_n=\int_a^b f
$$

### Thm. 2. [항별적분] *(Term-by-term Integration)*

$f_n\in\mathcal R[a,b]$인 함수열 ${f_n}$에 대해
$$
\sum_{n=1}^\infty f_n
$$
이 $[a,b]$에서 균등수렴하면
$$
\int_a^b\sum_{n=1}^\infty f_n
=\sum_{n=1}^\infty\int_a^b f_n
$$

### Thm. 3. [멱급수의 적분] *(Integration of Power Series)*

멱급수
$$
f(x)=\sum_{n=0}^\infty a_n(x-c)^n
$$
가 $[a,b]$에서 수렴하면
$$
\int_a^b f(x)\,dx
=\sum_{n=0}^\infty a_n\int_a^b (x-c)^n\,dx
$$

### Thm. 4. [멱급수의 특이적분] *(Improper Integration of Power Series)*

$f(x)=\sum_{n=0}^\infty a_n(x-c)^n$이 $[a,b)$에서 수렴하고
$$
\sum_{n=0}^\infty \frac{a_n}{n+1}(b-c)^{n+1}
$$
이 수렴하면
$$
\int_a^b f(x)\,dx
= 
\sum_{n=0}^\infty a_n\int_a^b (x-c)^n\,dx
$$

# [연습문제]

*(Exercises)*

1. 다음 함수열이 $[0,1]$에서
   ① 점별수렴, ② 균등수렴하는지 판별하시오.
   (1) $f_n(x)=x^n$
   (2) $f_n(x)=\dfrac{x^n}{n}$

2. $\displaystyle\sum_{n=0}^\infty\frac12(x-1)^n$의 수렴반지름과 수렴구간을 구하시오.

3. $f(x)=\displaystyle\sum_{n=1}^\infty\frac{\cos nx}{n^2}$가 $(-\infty,\infty)$에서 연속임을 보이시오.

4. $f_n(x)=|x|^{1+1/n}$과 극한함수 $f$가 $(-1,1)$에서 미분가능하지 않은 이유를 설명하시오.

5. 다음을 구하시오.
   (1) $f(x)=\displaystyle\sum_{n=0}^\infty\frac{x^{2n}}{2^n(2n+1)}$의 $\int_0^1 f(x)\,dx$
   (2) $f(x)=\displaystyle\sum_{n=1}^\infty\frac{(-1)^n(x-1)^n}{n}$의 $\int_1^2 f(x)\,dx$
