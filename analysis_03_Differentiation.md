미분 *(Differentiation)*

# 1. 미분계수 *(Differential Coefficient)*
## (1) 미분계수의 정의 *(Definition of Differential Coefficient)*
### Def 1. 평균변화율 *(Average Rate of Change)*
함수 $f:[a,b]\to\mathbb{R}$에 대하여
$$
\frac{\Delta y}{\Delta x}
=\frac{f(b)-f(a)}{b-a}
=\frac{f(a+\Delta x)-f(a)}{\Delta x}
$$
를 점 $a$에서의 $y=f(x)$의 평균변화율이라 한다.

### Def 2. 미분계수와 미분가능 *(Differential Coefficient and Differentiability)*

함수 $f:(a,b)\to\mathbb{R}$와 $c\in(a,b)$에 대하여
$$
f'(c)
=\lim_{\Delta x\to 0}\frac{\Delta y}{\Delta x}
=\lim_{x\to c}\frac{f(x)-f(c)}{x-c}
=\lim_{\Delta x\to 0}\frac{f(c+\Delta x)-f(c)}{\Delta x}
$$
가 존재하면 이를 $x=c$에서의 **미분계수(Differential Coefficient)** 라 한다.
이때 $f$는 $x=c$에서 **미분가능(differentiable)** 하다고 한다.
- 평균변화율의 극한
- 미분가능 <=> 미분계수 존재 (우미분계수 == 좌미분계수)

### Def 3. 우미분계수와 좌미분계수 *(Right-hand and Left-hand Differential Coefficient)*
함수 $f:[a,b)\to \mathbb{R}$에 대해  
1. 우미분계수
   $$
   f'_+(a)=\lim_{\Delta x\to 0^+}\frac{f(a+\Delta x)-f(a)}{\Delta x}
   $$
함수 $f:(a,b]\to \mathbb{R}$에 대해  
2. 좌미분계수
   $$
   f'_-(a)=\lim_{\Delta x\to 0^-}\frac{f(a+\Delta x)-f(a)}{\Delta x}
   $$

두 값이 존재하고 같으면 $f$는 $x=a$에서 미분가능하다.

### Def 4. 미분가능함수 *(Differentiable Function)*
1. $f:(a,b)\to\mathbb{R}$가 $(a,b)$의 모든 점에서 미분가능하면
   $f$를 **미분가능함수(differentiable function)** 라 한다.

2. $f:[a,b]\to\mathbb{R}$가 다음을 만족하면 $[a,b]$에서 미분가능함수라 한다.
  * $f$는 $(a,b)$에서 미분가능
  * $f$는 $x=a$에서 우미분가능
  * $f$는 $x=b$에서 좌미분가능

## (2) 미분계수의 연산 *(Rules of Differentiation)*

$f,g:D\to\mathbb{R}$가 $a\in D$에서 미분가능하면 다음이 성립한다.
1. $(f+g)'(a)=f'(a)+g'(a)$
2. $(f-g)'(a)=f'(a)-g'(a)$
3. $(fg)'(a)=f'(a)g(a)+f(a)g'(a)$
4. $\left(\frac{f}{g}\right)'(a)=\frac{f'(a)g(a)-f(a)g'(a)}{g(a)^2}$  (단, $g(a)\ne 0$)

## (3) 주요 정리 *(Main Theorems)*
### Thm 1. 미분가능성은 연속성을 함의 *(Differentiability implies Continuity)*
$f$가 $x=a$에서 미분가능하면 $x=a$에서 연속이다.  
$x=a$에서 불연속이면 $x=a$에서 미분불가능하다!  
- 연속이라고 미분가능은 아님

### Thm 2. 극값과 미분계수 *(interior extremum theorem, Fermat’s Theorem)*
$f:D\to\mathbb{R}$가 $D$의 내부점 $x=a$에서 극값을 갖고, $x=a$에서 미분가능하면 $f'(a)=0$ 이다.

### Thm 3. 연쇄법칙 *(Chain Rule)*
$f:D\to \mathbb{R}$가 $x=a$에서 미분가능하고 $g$가 $x=f(a)$에서 미분가능하면
$$
(g\circ f)'(a)=g'(f(a))f'(a)
$$

증명:  
#### Lemma 
$f:D\to\mathbb R$가 $x=a$에서 미분가능이면, 어떤 함수 $\phi:D\to\mathbb R$가 존재하여
* $\phi(a)=f'(a)$
* $\phi$는 $x=a$에서 연속
* 모든 $x\in D$에 대해
  $$
  f(x)=f(a)+\phi(x)(x-a)
  $$
  를 만족함.

(증명 스케치: $x\neq a$에서 $\phi(x)=\dfrac{f(x)-f(a)}{x-a}$, $\phi(a)=f'(a)$로 두면 됨. $f'(a)$의 정의가 곧 $\phi(x)\to \phi(a)$이므로 연속성도 따라옴.) 

가정: $f$가 $a$에서 미분가능, $g$가 $y=f(a)$에서 미분가능이라 하자. (정의역 문제를 피하려면 $g$의 정의역이 $f(D)$의 근방을 포함한다고 보면 충분함.)

1. $f$에 Lemma 적용
   $\exists,\phi$ with $\phi(a)=f'(a)$, $\phi$ 연속 at $a$, 그리고
   $$
   f(x)=f(a)+\phi(x)(x-a)
   $$
   따라서
   $$
   f(x)-f(a)=\phi(x)(x-a).
   $$

2. $g$에 Lemma 적용(점은 $y=f(a)$에서)
   $\exists,\psi$ with $\psi(f(a))=g'(f(a))$, $\psi$ 연속 at $f(a)$, 그리고 (모든 $y$가 $g$의 정의역에 있을 때)
   $$
   g(y)=g(f(a))+\psi(y)\bigl(y-f(a)\bigr).
   $$

3. $y=f(x)$를 대입하여 합성
   $f(x)$가 $g$의 정의역에 들어가는 $x$들에 대해
   $$
   g(f(x))=g(f(a))+\psi(f(x))\bigl(f(x)-f(a)\bigr).
   $$
   여기에 1)에서 $f(x)-f(a)=\phi(x)(x-a)$를 대입하면
   $$
   g(f(x))=g(f(a))+\psi(f(x)),\phi(x),(x-a).
   $$

4. 미분계수 계산
   $x\neq a$에서
   $$
   \frac{g(f(x))-g(f(a))}{x-a}=\psi(f(x)),\phi(x).
   $$
   이제 $x\to a$를 보내면,

* $f$가 $a$에서 연속이므로 $f(x)\to f(a)$
* $\psi$가 $f(a)$에서 연속이므로 $\psi(f(x))\to \psi(f(a))=g'(f(a))$
* $\phi$가 $a$에서 연속이므로 $\phi(x)\to \phi(a)=f'(a)$

따라서 곱의 극한으로
$$
\lim_{x\to a}\frac{g(f(x))-g(f(a))}{x-a}
=\lim_{x\to a}\psi(f(x))\phi(x)
=\psi(f(a))\phi(a)
=g'(f(a))f'(a).
$$
즉 $(g\circ f)$는 $a$에서 미분가능하고
$$
(g\circ f)'(a)=g'(f(a)),f'(a)
$$
가 성립함.

# 2. 도함수 *(Derivative)*
## (1) 도함수의 정의 *(Definition of Derivative)*
$f:D\to\mathbb{R}$가 각 $x\in D$에서 미분가능할 때
$$ f'(x)=\frac{df}{dx}
=\lim_{y\to x}\frac{f(y)-f(x)}{y-x}$$
를 $f$의 **도함수(derivative function)** 라 한다.
미분계수의 정의를 일반적인 x로 확장한 것.  

## (2) 여러 함수의 도함수 *(Derivatives of Elementary Functions)*
* $(c)'=0$  $(c\in\mathbb{R})$
* $(x^n)'=nx^{n-1}$
* $(e^x)'=e^x$
* $(a^x)'=a^x\ln a$
* $(\ln x)'=\frac{1}{x}$
* $(\sin x)'=\cos x$
* $(\cos x)'=-\sin x$
* $(\tan x)'=\sec^2 x$
* $(\cot x)'=-\csc^2 x$
* $(\sec x)'=\sec x\tan x$
* $(\csc x)'=-\csc x\cot x$

# 3. 평균값 정리 *(Mean Value Theorem)*
## (0) 롤의 정리
$f:[a,b]\to\mathbb{R}$가 $[a,b]$에서 연속이고 $(a,b)$에서 미분가능하면
$$
f(a) = f(b) \Rightarrow \exists c\in(a,b)\ \text{s.t.}\ f'(c)=0
$$

- 증명: $f$는 $[a,b]$에서 연속이므로 최대·최소 정리에 의해 최댓값과 최솟값을 가진다.  
**Case 1**: $f$가 상수함수인 경우  
모든 $x\in(a,b)$에서 $f'(x)=0$이므로 자명.  
**Case 2**: $f$가 상수함수가 아닌 경우  
$f$가 상수함수가 아니므로 최댓값 또는 최솟값 중 적어도 하나는 $f(a)=f(b)$와 다른 값이다.  
  
  $M=\max_{x\in[a,b]}f(x)$, $m=\min_{x\in[a,b]}f(x)$라 하자.
  - $M > f(a)=f(b)$인 경우: $\exists c\in(a,b)$ such that $f(c)=M$  
   $c$는 내부점에서의 최댓값(극댓값)이므로 Thm 2에 의해 $f'(c)=0$
   - $m < f(a)=f(b)$인 경우: $\exists c\in(a,b)$ such that $f(c)=m$  
   $c$는 내부점에서의 최솟값(극솟값)이므로 Thm 2에 의해 $f'(c)=0$

따라서 $f'(c)=0$을 만족하는 $c\in(a,b)$가 존재한다. ∎

- 이렇게도 보일 수 있다.  
$p\in[a,b]$ such that $f(p)=m$  
$q\in[a,b]$ such that $f(q)=M$

- Case 1: $q\in(a,b)$
$q$는 내부점에서의 최댓값 → Thm 2에 의해 $f'(q)=0$

- Case 2: $q=a$ 또는 $q=b$
  - 이때 $M=f(q)=f(a)=f(b)$
  - 그런데 $f$가 상수함수가 아니므로 $m<M$
  - 따라서 최소값을 주는 점 $p$는 $a,b$가 될 수 없고 $p\in(a,b)$
  - → Thm 2에 의해 $f'(p)=0$

## (1) 평균값 정리 *(Mean Value Theorem, MVT)*
$f:[a,b]\to\mathbb{R}$가 $[a,b]$에서 연속이고 $(a,b)$에서 미분가능하면
$$
\exists c\in(a,b)\ \text{s.t.}\
f'(c)=\frac{f(b)-f(a)}{b-a}
$$

- 증명  

보조함수 $g(x)$를 다음과 같이 정의하자:
$$
g(x) = f(x) - \frac{f(b)-f(a)}{b-a}(x-a)
$$

$g(x)$는 $f(x)$에서 두 점 $(a,f(a))$, $(b,f(b))$를 잇는 직선을 뺀 함수이다.

**$g$가 롤의 정리 조건을 만족함을 보이자:**
1. $g$는 $[a,b]$에서 연속: $f$가 연속이고 일차함수도 연속이므로 $g$도 연속
2. $g$는 $(a,b)$에서 미분가능: $f$가 미분가능하고 일차함수도 미분가능하므로 $g$도 미분가능
3. $g(a)=g(b)$인지 확인:
   $$
   g(a) = f(a) - \frac{f(b)-f(a)}{b-a}(a-a) = f(a)
   $$
   
   $$
   g(b) = f(b) - \frac{f(b)-f(a)}{b-a}(b-a) = f(b) - (f(b)-f(a)) = f(a)
   $$
   따라서 $g(a)=g(b)$

**롤의 정리 적용:**  
$g$가 롤의 정리의 조건을 모두 만족하므로
$$
\exists c\in(a,b)\ \text{s.t.}\ g'(c)=0
$$

**$g'(c)=0$로부터 결론 도출:**
$$
g'(x) = f'(x) - \frac{f(b)-f(a)}{b-a}
$$
이므로
$$
g'(c) = f'(c) - \frac{f(b)-f(a)}{b-a} = 0
$$

따라서
$$
f'(c) = \frac{f(b)-f(a)}{b-a}
$$
∎

## (2) 코시 평균값 정리 *(Cauchy Mean Value Theorem)*
$f,g:[a,b]\to\mathbb{R}$가 $[a,b]$에서 연속이고 $(a,b)$에서 미분가능하면
$$
\exists c\in(a,b)\ \text{s.t.}\
(g(b)-g(a))f'(c)=(f(b)-f(a))g'(c)
$$

- $h(x) = (f(b)-f(a))g(x) - (g(b)-g(a))f(x)$로 세팅하고 롤의정리 사용하면 증명 가능
- 평균값의 정리와 시사하는 바가 조금은 다름
  - **평균값 정리 (MVT)**
    - 기하학적 의미: 곡선 위에 두 점을 잇는 **할선의 기울기**와 같은 기울기를 갖는 **접선**이 곡선 내부 어딘가에 존재
    - 대수적 의미: $f'(c) = \frac{f(b)-f(a)}{b-a}$ 형태로 **평균 변화율 = 순간 변화율**
    - 특수한 경우: $g(x)=x$로 두면 코시 평균값 정리에서 유도됨
  - **코시 평균값 정리 (Cauchy MVT)**
    - 기하학적 의미: 매개변수 곡선 $(g(t), f(t))$에서 두 점을 잇는 할선의 기울기 $\frac{f(b)-f(a)}{g(b)-g(a)}$와 같은 기울기를 갖는 접선 $\frac{f'(c)}{g'(c)}$이 존재
    - 대수적 의미: $\frac{f'(c)}{g'(c)} = \frac{f(b)-f(a)}{g(b)-g(a)}$ 형태로 **두 함수의 변화율 비율**
    - 일반화: 평균값 정리를 두 함수의 비율 관계로 확장한 것
    - 응용: **로피탈 정리의 증명**에 핵심적으로 사용됨

## (3) 로피탈의 정리 *(L’Hôpital’s Rule)*
$f,g$가 $a$를 제외한 어떤 근방에서 미분가능하고, $a$에서(또는 $a$로 갈 때) 다음이 성립한다고 하자.

* (H1) $\exists\delta>0$ s.t. $f,g$는 $I:=(a-\delta,a)\cup(a,a+\delta)$에서 미분가능이다.
* (H2) $g'(x)\neq 0$ for all $x\in I$.
* (H3) (형태) $\displaystyle \lim_{x\to a} f(x)=\lim_{x\to a} g(x)=0$  
또는  $\displaystyle |f(x)|\to\infty,\ |g(x)|\to\infty$ ($x\to a$).
* (H4) $\displaystyle \lim_{x\to a}\frac{f'(x)}{g'(x)}=L$가 존재한다. ($L\in\mathbb R\cup{\pm\infty}$)

그러면
$$
\lim_{x\to a}\frac{f(x)}{g(x)}=L
$$
이다. (좌극한/우극한도 동일하게 성립한다.)

- 로피탈의 정리 주의사항
  - 0/0 혹은 infi/infi 꼴인지 체크 필수!
  - 분모 g'(x)가 0이 되면 안됨
  - lim f'/g' -> lim f/g이지 역은 아니다

### 증명
#### 공통 준비: $g(x)\neq 0$를 근방에서 보장
(H2)로부터 $g'$는 $I$에서 0이 아니므로, $g$는 각 구간 $(a-\delta,a)$, $(a,a+\delta)$에서 단조이다(미분가능 + 도함수 부호 불변).  
따라서 각 구간에서 $x\neq a$이면 $g(x)\neq g(a)$가 성립한다. 특히 $0/0$ 경우에 $g(a)=0$로 정의하면, 충분히 $a$에 가까운 $x\neq a$에 대해 $g(x)\neq 0$이 되어 $\frac{f(x)}{g(x)}$가 정의된다.

(단조성은 평균값정리로도 보일 수 있다: $x_1<x_2$이면 어떤 $\xi$가 존재하여 $g(x_2)-g(x_1)=g'(\xi)(x_2-x_1)\neq 0$이므로 $g(x_2)\neq g(x_1)$, 따라서 단조.)

#### Case 1: $0/0$ 꼴
가정: $\displaystyle \lim_{x\to a}f(x)=\lim_{x\to a}g(x)=0$.

$ f(a):=0,\ g(a):=0$으로 재정의하면 $f,g$는 $a$에서 연속이 된다. 위의 공통 준비에 의해 $x\neq a$가 $a$에 충분히 가까우면 $g(x)\neq 0$이다.

이제 $x\neq a$를 $a$에 충분히 가깝게 잡고, 구간 $[a,x]$ (또는 $[x,a]$)에서 코시 평균값정리를 적용한다. 그러면 어떤 $c$가 $a$와 $x$ 사이에 존재하여
$$
\frac{f(x)-f(a)}{g(x)-g(a)}=\frac{f'(c)}{g'(c)}
$$
가 성립한다. $f(a)=g(a)=0$이므로
$$
\frac{f(x)}{g(x)}=\frac{f'(c)}{g'(c)}.
$$

이제 $x\to a$일 때, $c$는 $a$와 $x$ 사이에 있으므로 $c\to a$이다. 엄밀히는 임의의 수열 $x_n\to a$($x_n\neq a$)를 택하면, 이에 대응하는 $c_n$은 $c_n\to a$이며
$$
\frac{f(x_n)}{g(x_n)}=\frac{f'(c_n)}{g'(c_n)}.
$$
(H4)로부터 $c_n\to a$이면
$$
\frac{f'(c_n)}{g'(c_n)}\to L
$$
이므로
$$
\frac{f(x_n)}{g(x_n)}\to L.
$$
임의의 수열에 대해 성립하므로
$$
\lim_{x\to a}\frac{f(x)}{g(x)}=L.
$$

#### Case 2: $\infty/\infty$ 꼴
가정: $\displaystyle |f(x)|\to\infty,\ |g(x)|\to\infty$ $(x\to a)$.

(H4)의 정의에 의해 임의의 $\epsilon>0$에 대해 어떤 $\delta_1>0$가 존재하여
$$
0<|t-a|<\delta_1 \ \Rightarrow\ \left|\frac{f'(t)}{g'(t)}-L\right|<\epsilon
$$
가 성립한다.

이제 한쪽 구간(예: $(a,a+\delta)$)에서 증명하고 다른 쪽도 동일하게 하면 된다. $x>a$이고 $0 < x-a < \min(\delta,\delta_1)$인 $x$를 택한다. 또한 같은 구간에서 고정된 $y>a$를 $0 < y-a < \min(\delta,\delta_1)$로 택한다 (즉, $y$도 $a$에 충분히 가까운 하나의 점으로 고정한다).

코시 평균값정리를 $[y,x]$에 적용하면 어떤 $c$가 $y$와 $x$ 사이에 존재하여
$$
\frac{f(x)-f(y)}{g(x)-g(y)}=\frac{f'(c)}{g'(c)}.
$$
정리하면
$$
f(x)=f(y)+\frac{f'(c)}{g'(c)}(g(x)-g(y)).
$$
양변에서 $L g(x)$를 빼고 $g(x)$로 나누면
$$
\frac{f(x)}{g(x)}-L
=\frac{f(y)}{g(x)}+\left(\frac{f'(c)}{g'(c)}-L\right)\frac{g(x)-g(y)}{g(x)}-L\frac{g(y)}{g(x)}.
$$
절댓값을 취하고 삼각부등식을 쓰면
$$
\left|\frac{f(x)}{g(x)}-L\right|
\le
\left|\frac{f(y)}{g(x)}\right|
+
\left|\frac{f'(c)}{g'(c)}-L\right|\left|\frac{g(x)-g(y)}{g(x)}\right|
+
|L|\left|\frac{g(y)}{g(x)}\right|.
$$

이제 $x\to a$로 보낸다. $|g(x)|\to\infty$이므로 (고정된 $y$에 대해)
$$
\left|\frac{f(y)}{g(x)}\right|\to 0,\qquad \left|\frac{g(y)}{g(x)}\right|\to 0.
$$
또한 $c$는 $x$와 $y$ 사이에 있으므로 $x\to a$이면 $c\to a$가 된다(여기서 $y$는 $a$에 매우 가깝게 고정했으므로 $x,y\to a$를 함께 만족시키는 셈이다). 따라서 $x$를 충분히 $a$에 가깝게 잡으면 $c$도 $a$에 가까워져서
$$
\left|\frac{f'(c)}{g'(c)}-L\right|<\epsilon
$$
가 성립한다.

마지막으로
$$
\left|\frac{g(x)-g(y)}{g(x)}\right|
=\left|1-\frac{g(y)}{g(x)}\right|
\le 1+\left|\frac{g(y)}{g(x)}\right|
\to 1
$$
이므로, $x$를 충분히 $a$에 가깝게 하면 어떤 상수(예: $2$)로 유계시킬 수 있다. 즉 충분히 $a$에 가까운 $x$에 대해
$$
\left|\frac{g(x)-g(y)}{g(x)}\right|\le 2
$$
가 되게 할 수 있다.

따라서 $x$를 $a$에 충분히 가깝게 잡으면
$$
\left|\frac{f(x)}{g(x)}-L\right|
\le 0 + \epsilon\cdot 2 + 0
=2\epsilon
$$
가 된다. $\epsilon>0$가 임의이므로
$$
\lim_{x\to a}\frac{f(x)}{g(x)}=L
$$
을 얻는다.

좌측에서도 동일하게 증명된다. ∎

# 연습문제 *(Exercises)*
1. 다음 함수가 $x=0$에서 미분가능한지 설명하시오.

   * $f(x)=\begin{cases}x\sin\frac{1}{x}, & x\ne 0 \\ 0, & x=0\end{cases}$
   * $g(x)=\begin{cases}x^2\sin\frac{1}{x}, & x\ne 0 \\ 0, & x=0\end{cases}$

2. $(a,b)$에서 미분가능한 함수의 도함수 $f'$에 대해
   제거가능 불연속점을 가질수 있나? 비약 불연속점도 가질 수 있나?

3. 다음 함수의 도함수를 구하시오.
   * $f(x)=\ln(1+\sin x)$
   * $g(x)=e^{\cos\sqrt{x}}$
   * $h(x)=x^{\tan x}$

4. $f'(0)=\dfrac{f(b)-f(a)}{b-a}$를 만족하는 $a,b$가 항상 존재하지 않을수 있는지 설명하시오.

5. 다음 극한을 계산하시오.

   * $\lim_{x\to\pi}(x-\pi)\cot x$
   * $\lim_{x\to\infty}\sin\frac{1}{x}\ln x$

6. 다음 극한 계산이 틀린 이유를 설명하시오.
   $$
   \lim_{x\to 0}\frac{x^2}{x^2+\sin x}
   =\lim_{x\to 0}\frac{2x}{2x+\cos x}
   =\lim_{x\to 0}\frac{2}{2-\sin x}
   =1
   $$