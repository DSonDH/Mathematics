# Chapter 3 거리공간 (Metric Space)

## 3.1 거리공간 (Metric Space)

### 정의 3.1: 거리함수 (metric / distance function)
집합 $X$에서 거리함수(또는 거리, metric) $d: X \times X \to \mathbb{R}$는 다음 네 조건을 만족하는 함수이다:

1. **비음성(non-negativity)**: $d(x,y) \geq 0$ for all $x,y \in X$
2. **동일성(identity of indiscernibles)**: $d(x,y) = 0 \Leftrightarrow x = y$
3. **대칭성(symmetry)**: $d(x,y) = d(y,x)$ for all $x,y \in X$
4. **삼각부등식(triangle inequality)**: $d(x,z) \leq d(x,y) + d(y,z)$ for all $x,y,z \in X$

순서쌍 $(X, d)$를 거리공간(metric space)이라 한다.
  - 혹은 거리 $d$가 주어진 집합 $X$

### 정의 3.2: 열린공(open ball)
거리공간 $(X, d)$에서 점 $x \in X$와 $r > 0$에 대해, 열린공(open ball)을 다음과 같이 정의한다:

$$B_d(x, r) = \{y \in X : d(x,y) < r\}$$

### 정의 3.3: 거리위상 (metric topology)
거리공간 $(X, d)$에서 거리위상 $\mathscr{T}_d$는 열린공들을 기저로 하는 위상이다:

$$\mathscr{T}_d = \{U \subset X : \forall x \in U, \exists r > 0 \text{ s.t. } B_d(x,r) \subset U\}$$

### 정의. 수열의 수렴 (converge)

거리공간 $(X,d)$와 수열 $(x_n)\subseteq X$를 생각하자. 임의의 $\varepsilon>0$에 대하여 어떤 $N\in\mathbb N$이 존재하여

$$n\ge N\Rightarrow d(x_n,x)<\varepsilon$$

가 성립할 때, 수열 $(x_n)$이 $x\in X$로 수렴한다고 한다. 이를 

$$x_n\to x$$

또는

$$\lim_{n\to\infty}x_n=x$$

라고 표기한다.  
직관적으로는 $n$이 충분히 커지면 $x_n$과 $x$ 사이의 거리를 원하는 만큼 작게 만들 수 있다는 뜻이다.

### 정의. 코시 수열

거리공간 $(X,d)$의 수열 $(x_n)\subseteq X$를 생각하자. 임의의 $\varepsilon>0$에 대하여 어떤 $N\in\mathbb N$이 존재하여

$$m,n\ge N\Rightarrow\quad d(x_m,x_n)<\varepsilon $$

가 성립할 때, $(x_n)$을 코시 수열이라고 한다.

직관적으로는 수열의 항들이 뒤로 갈수록 서로 원하는 만큼 가까워진다는 뜻이다.

수렴과 코시 조건의 차이는 다음과 같다.

* 수렴 조건은 주어진 극한점 $x$와 $x_n$ 사이의 거리 $d(x_n,x)$를 살펴본다.
* 코시 조건은 극한점을 미리 사용하지 않고 뒤쪽 두 항 $x_m,x_n$ 사이의 거리 $d(x_m,x_n)$를 살펴본다.
- $\mathbb R$에서의 코시판정법은 필요충분조건이었지만, 거리공간에서 역은 항상 성립하지는 않는다. 
  - 즉, $\mathbb R$에서 코시판정법은 완비성공리와 동치였지만, 거리공간에서는 다른 것.
  - 완비성공리가 거리공간에도 성립하려면 상계같은 개념을 논할 수 있게 그 공간에 '순서'가 있어야 한다. 이 방법은 없다.
  - 거리공간에서는 대신에 코시수열의 수렴을 완비성의 정의로 한다.

#### 예제
수렴하는 수열은 항상 코시 수열임을 보여라.

**증명**

거리공간 $(X,d)$에서 수열 $(x_n)$이 $x\in X$로 수렴한다고 하자. 즉, $x_n\to x$ 라고 가정한다. $(x_n)$이 코시 수열임을 보이려면 다음을 증명해야 한다.

$$\forall\varepsilon>0,\quad
\exists N\in\mathbb N
\quad\text{such that}\quad
m,n\ge N\Rightarrow d(x_m,x_n)<\varepsilon$$

임의의 $\varepsilon>0$을 택하자. $x_n\to x$이므로, 수렴의 정의를 $\varepsilon/2>0$에 적용하면 어떤 $N\in\mathbb N$이 존재하여 $k\ge N\Rightarrow d(x_k,x)<\frac{\varepsilon}{2}$ 가 성립한다. 이제 $m,n\ge N$이라고 하자. 그러면 $d(x_m,x)<\frac{\varepsilon}{2}, d(x_n,x)<\frac{\varepsilon}{2}$ 이다.

거리 함수의 대칭성에 의해 $d(x,x_n)=d(x_n,x)<\frac{\varepsilon}{2}$ 이다. 

따라서 삼각부등식을 적용하면 $d(x_m,x_n) \le d(x_m,x)+d(x,x_n) <\frac{\varepsilon}{2}+\frac{\varepsilon}{2} =\varepsilon$  
즉, 임의의 $\varepsilon>0$에 대하여 적당한 $N\in\mathbb N$이 존재하여 
$m,n\ge N \Rightarrow d(x_m,x_n)<\varepsilon$ 가 성립한다. 따라서 코시 수열의 정의에 의해 $(x_n)$은 코시 수열이다.

### 정의. 완비거리공간 (complete metric space)

집합 $X$에 속하는 임의의 코시 수열이 항상 $X$의 어떤 원소로 수렴할 때 거리공간 $(X,d)$를 완비거리공간 (complete metric space)라 한다.

예를 들어 유리수 공간 $\mathbb Q$에 일반 거리

$$d(x,y)=|x-y|$$

를 부여하면 $\sqrt2$에 가까워지는 유리수 수열은 코시 수열이지만 그 극한 $\sqrt2$가 $\mathbb Q$에 속하지 않는다. 따라서 $\mathbb Q$는 완비거리공간이 아니다.

#### 예제

(a) $\mathbb R^2$ 에 이산 거리를 부여하자.

$$rho(x,y)= \begin{cases} 0,&x=y,\\ 1,&x\neq y \end{cases}$$

이 공간에서 코시 수열은 어떻게 생겼는가? $\mathbb R^2$은 이 거리에 대해 완비공간인가?

$(x_n)$이 이산 거리 $\rho$에 대한 코시 수열이라고 하자. 코시 수열의 정의를 $\varepsilon=\frac12$ 에 적용하면 어떤 $N\in\mathbb N$이 존재하여 $m,n\ge N \Rightarrow \rho(x_m,x_n)<\frac12$ 이다.  
그런데 이산 거리 $\rho(x_m,x_n)$는 0 또는 1 중 하나다. 따라서 $\rho(x_m,x_n)<\frac12$ 가 성립하려면 반드시 $\rho(x_m,x_n)=0$ 이어야 한다. 그러므로 $x_m=x_n$ 이다.  
특히 $m=N$으로 놓으면 모든 $n\ge N$ 에 대하여 $x_n=x_N$ 이므로 수열은 $N$번째 항부터 일정하다: $x_N=x_{N+1}=x_{N+2}=\cdots$ 가 되어야 한다.

**완비성**: 수열이 $N$번째 항부터 $x_N$으로 일정하므로 $x_n\to x_N$ 이다. 실제로 임의의 $\varepsilon>0$에 대하여 $n\ge N$이면 $\rho(x_n,x_N)=0<\varepsilon$ 이다. 따라서 모든 코시 수열이 $\mathbb R^2$의 원소로 수렴한다.

$$\boxed{(\mathbb R^2,\rho)\text{는 완비거리공간이다.}}$$

결론은 $\mathbb R^2$ 에만 국한되지 않는다. 임의의 집합 $X$에 이산 거리를 부여하면 $(X,\rho)$는 완비거리공간이다.

---

문제 (b): $C[0,1]$ 가 거리 $d(f,g)=\sup_{x\in[0,1]}\{|f(x)-g(x)|\}$에 대해 완비거리공간임을 보여라.

**1단계**: 코시 수열을 잡는다. $(f_n)$이 $C[0,1]$의 코시 수열이라고 하자. 그러면 임의의 $\varepsilon>0$에 대하여 어떤 $N\in\mathbb N$이 존재하여 $m,n\ge N \Rightarrow |f_m-f_n|_\infty<\varepsilon$ 이다. 즉, $m,n\ge N \Rightarrow \sup_{x\in[0,1]}\{|f_m(x)-f_n(x)|\}<\varepsilon$ 따라서 모든 $x\in[0,1]$ 에 대하여 $|f_m(x)-f_n(x)|<\varepsilon$ 이다.

**2단계**: 각 $x$에서 극한함수를 정의한다. 하나의 $x\in[0,1]$를 고정하자. 위 부등식에 의해 실수 수열 $(f_n(x))$ 은 $\mathbb R$에서 코시 수열이다.

$\mathbb R$은 완비이므로 $(f_n(x))$은 어떤 실수로 수렴한다. 이를 $f(x)$라고 정의하자. $f(x):=\lim_{n\to\infty}f_n(x).$ 이와 같이 각 $x\in[0,1]$에 대해 정의하면 함수 $f:[0,1]\to\mathbb R$ 를 얻는다.

**3단계**: $f_n\to f$가 균등수렴함을 보인다. 임의의 $\varepsilon>0$을 택하자. $(f_n)$이 코시 수열이므로 어떤 $N$이 존재하여 $m,n\ge N \Rightarrow |f_m(x)-f_n(x)|<\varepsilon$ 가 모든 $x\in[0,1]$에 대하여 성립한다. 여기서 $n\ge N$을 고정하고 $m\to\infty$ 로 보내면 $f_m(x)\to f(x)$ 이므로 $|f(x)-f_n(x)|\le\varepsilon$ 을 얻는다. 더 엄밀하게 $<\varepsilon$을 얻고 싶다면 처음부터 코시 조건에 $\varepsilon/2$를 적용하면 된다.

따라서 $n\ge N$이면 $\sup_{x\in[0,1]}|f_n(x)-f(x)|\le\varepsilon.$ 즉, $|f_n-f|_\infty\to0.$ 

따라서 $f_n$은 $f$로 균등수렴한다.

**4단계**: 극한함수 $f$의 연속성을 보인다. 각 $f_n$은 연속함수이고 $f_n\to f$는 균등수렴한다. 연속함수열의 균등극한은 연속함수이므로 $f$는 연속이다.

>이를 직접 증명하면 다음과 같다: $x_0\in[0,1]$와 $\varepsilon>0$을 고정하자. 균등수렴에 의해 어떤 $N$이 존재하여 $\|f_N-f\|_\infty<\frac{\varepsilon}{3}$ 이다.  
>$f_N$이 $x_0$에서 연속이므로 어떤 $\delta>0$이 존재하여 $|x-x_0|<\delta \Rightarrow |f_N(x)-f_N(x_0)|<\frac{\varepsilon}{3}$ 이다.
>
>따라서 $|x-x_0|<\delta$ 이면
>
>$$|f(x)-f(x_0)| \le |f(x)-f_N(x)| +|f_N(x)-f_N(x_0)| +|f_N(x_0)-f(x_0)| \\
><\frac{\varepsilon}{3} +\frac{\varepsilon}{3} +\frac{\varepsilon}{3} =\varepsilon$$

그러므로 $f$는 $x_0$에서 연속이고, $x_0$가 임의였으므로 $f\in C[0,1]$ 이다. 결국 $C[0,1]$의 모든 코시 수열이 $C[0,1]$의 원소로 수렴한다.

완비 노름공간을 바나흐 공간이라고 하므로 $C[0,1]$은 최소상계노름에 대한 바나흐 공간이다.

> 정의: 최소상계노름(sup norm, 최소상계노음, 최소상계노엄)
> 해석학에서 뭔가 의미있는 작업을 하려면 완비성이 전제되야 하므로, 거리 $d(f,g)=\sup_{x\in[0,1]}|f(x)-g(x)|$는 $C[0,1]$을 다룰 때 가장 자연스러운 거리다. 다음 표기법이 표준이다:
>
>$$ \|f-g\|_\infty =d(f,g) = \sup_{x\in[0,1]}\{|f(x)-g(x)|\}$$
>
>$g$가 0일때, 이 거리는 흔히 최소상계노름 (sup norm) 이라 한다. (발음: 슈프 놈)
>
>$$ \|f\|_\infty =d(f,0) = \sup_{x\in[0,1]}\{|f(x)|\}$$
>
> 별도 업급이 없으면 $C[0,1]$공간에 주어진 거리는 보통 모두 최소상계노름이다.
---

문제 (c) Let $C^1[0,1]={f:[0,1]\to\mathbb R: f\text{는 미분 가능하고 }f'\text{는 연속}}$ 라고 하자.  
$d(f,g)=\|f-g\|_\infty$ 를 부여했을 때 완비거리공간인가? 

반례가 있다. $C^1[0,1]$에 속하는 코시 수열이 $C^1[0,1]$에 속하지 않는 함수로 수렴하는 예를 찾으면 된다.

다음 함수열을 생각하자.

$$f_n(x)=\sqrt{\left(x-\frac12\right)^2+\frac1n}.$$

각 $f_n$은 미분 가능하고 

$$f_n'(x)=\frac{x-\frac12} {\sqrt{(x-\frac12)^2+\frac1n}}$$

도 연속이다. 따라서 $f_n\in C^1[0,1]$.  
한편 $f(x)=\left|x-\frac12\right|$ 라고 하면 다음 부등식이 성립한다. $0\le \sqrt{\left(x-\frac12\right)^2+\frac1n} -\left| -\frac12\right| \le \frac1{\sqrt n}.$
따라서 $|f_n-f|_\infty\le\frac1{\sqrt n}\to0.$ 즉, $f_n\to f$ 가 최소상계 거리에서 성립한다. 따라서 $(f_n)$은 수렴하는 수열이므로 코시 수열이다.

그러나 $f(x)=\left|x-\frac12\right|$ 는 $x=\frac12$ 에서 미분 가능하지 않다. 실제로 좌미분계수와 우미분계수가 각각 $-1,\ 1$ 로 서로 다르다. 그러므로 $f\notin C^1[0,1].$ 즉, $C^1[0,1]$의 코시 수열 $(f_n)$이 $C^1[0,1]$ 밖의 함수로 수렴한다. 따라서 $\left(C^1[0,1],|\cdot|_\infty\right)$ 는 완비거리공간이 아니다.

주의할 점은 $C^1[0,1]$ 자체가 항상 불완비인 것이 아니라 여기에서 사용한 거리가 함수값만 측정하는 최소상계 거리이기 때문에 불완비라는 점이다. 예를 들어 $|f|_{C^1}=|f\|_\infty+|f'\|_\infty$ 라는 노름을 사용하면 $C^1[0,1]$은 완비공간이 된다.


### 연속함수의 정의

두 거리공간 $(X,d_1)$, $(Y,d_2)$와 함수 $f:X\to Y$ 를 생각하자. 함수 $f$가 $x\in X$에서 연속이라는 것은 임의의 $\varepsilon>0$에 대하여 어떤 $\delta>0$가 존재하여 $d_1(x,y)<\delta \Rightarrow
d_2(f(x),f(y))<\varepsilon$ 가 성립한다는 뜻이다.

정의역과 공역이 함수공간일수도 있다. 

#### 예제 
$C[0,1]$ 에서 $\mathbb R$로 가는 연속함수인지 판단하라.

(a) 고정된 함수 $k\in C[0,1]$에 대하여 $G(f)=\int_0^1 f(x)k(x) dx$ 

$f,h\in C[0,1]$ 에 대하여

$$|G(f)-G(h)|
=\left| \int_0^1 f(x)k(x) dx -\int_0^1h(x)k(x) dx \right|\\
=\left| \int_0^1(f(x)-h(x))k(x) dx \right| \le\int_0^1|f(x)-h(x)||k(x)| dx$$

모든 $x\in[0,1]$에 대하여 $|f(x)-h(x)|\le\|f-h\|_\infty$ 이므로

$$|G(f)-G(h)| \le \|f-h\|_\infty\int_0^1|k(x)| dx $$

다음과 같이 놓자. $M=\int_0^1|k(x)| dx$. 그러면 $|G(f)-G(h)|\le M\|f-h\|_\infty$ 

임의의 $\varepsilon>0$에 대하여 $\delta=\frac{\varepsilon}{M+1}$ 로 선택하자. 만약 $\|f-h\|_\infty<\delta$ 이면

$$|G(f)-G(h)| \le M\|f-h\|_\infty\ <M\delta\ =\frac{M}{M+1}\varepsilon <\varepsilon$$

따라서 $G$는 모든 $f\in C[0,1]$에서 연속이다.

(b) $G(f)=f\left(\frac12\right)$ 

$f,h\in C[0,1]$ 에 대하여

$$|G(f)-G(h)| =\left|f\left(\frac12\right)-h\left(\frac12\right)\right| \le\sup_{x\in[0,1]}|f(x)-h(x)| =\|f-h\|_\infty$$

임의의 $\varepsilon>0$에 대해 $\delta=\varepsilon$ 로 선택한다. 그러면 $\|f-h\|_\infty<\delta$ 일 때

$$|G(f)-G(h)| \le\|f-h\|_\infty <\delta =\varepsilon.$$

따라서 $G$는 연속이다.

(c) $G(f)=f\left(\frac12\right)$ 이지만 $C[0,1]$에는 거리 $d_1(f,h)=\int_0^1|f(x)-h(x)| dx$ 가 주어진다.

이 경우는 연속함수가 아니다. 적분 거리는 함수값의 전체적인 차이는 측정하지만 한 점이나 매우 좁은 구간에서 발생하는 큰 차이를 제대로 통제하지 못한다.

반례: 다음과 같은 삼각형 모양의 연속함수를 정의하자. $f_n(x)=\max \{1-n\left|x-\frac12\right|,0\}.$  
이 함수는 중심 $x=\frac12$에서 높이가 1이고, 폭이 $2/n$인 삼각형 모양이다. 특히 $f_n\left(\frac12\right)=1.$ 영함수를 $0(x)=0$이라고 하면 $G(f_n)=1,\ G(0)=0$ 이므로 $|G(f_n)-G(0)|=1$ 이다.

한편 적분 거리에서는 $d_1(f_n,0)=\int_0^1|f_n(x)| dx$

$f_n$의 그래프는 밑변의 길이가 $2/n$, 높이가 1인 삼각형이므로

$$d_1(f_n,0) =\frac12\cdot\frac{2}{n}\cdot1\ =\frac1n\to0$$

따라서 $f_n\to0$ 가 적분 거리에서는 성립한다. 그러나 $G(f_n)=1\not\to0=G(0)$ 이므로 $G$는 0에서 연속이 아니다.

이를 $\varepsilon-\delta$ 정의로도 확인할 수 있다. $\varepsilon=\frac12$로 놓자. 임의의 $\delta>0$에 대하여 $1/n<\delta$가 되도록 $N$을 충분히 크게 선택하면 $d_1(f_n,0)=\frac1n<\delta$ 이지만 $|G(f_n)-G(0)|=1>\frac12$ 따라서 연속성의 조건을 만족하는 $\delta$가 존재하지 않는다.


## 3.2 거리공간의 기본 성질 (Basic Properties)

### 정의. $\epsilon$-근방($\epsilon$-neighborhood)
양수 $\epsilon>0$과 거리공간 $(X,d)$의 원소 $x$에 대하여 집합

$$V_\epsilon(x)=\{y \in X: d(x,y) < \epsilon\}$$

를 $x$의 **$\epsilon$-근방** 또는 **열린 공** 이라고도 한다. 중심이 $x$이고 반지름이 $\varepsilon$인 점들의 집합이다.  
이에 대응하여

$$C_\varepsilon(x) = \{y\in X:d(x,y)\le\varepsilon\}$$

를 **닫힌 공** 이라고 한다.

- 이 $\epsilon$-근방으로 열린 집합, 극한점, 닫힌 집합을 정의할 수 있다:
  - 모든 $x \in O$에 대해 $V_{\epsilon}(x) \subseteq O$인 근방을 찾을 수 있을 때, 집합 $O \subseteq X$를 열린 집합(open set)이라 한다.
  - 모든 $V_{\epsilon}(x)$에 대해 $V_{\epsilon}(x)$와 $A$가 $x$ 이외의 다른 교집합을 가질 때, 점 $x$를 $A$의 극한점(limit point)라 한다.
  - 집합 $C$가 자기 자신의 극한점을 모두 포함하면 닫힌 집합(closed set) 이라고 한다.
  - 위상수학에선 열린집합이면서 닫힌집합인 집합이 있다. 이를 클로펜 집합 (clopen set, 열린 닫힌 집합) 이라 한다.


#### 예제
$(X, d)$를 거리공간이라 하자.

(a) $\varepsilon$-근방 $V_\varepsilon(x)$가 열린집합임을 확인하라. 또한 $C_\varepsilon(x)=\{ y\in X:d(x,y)\le\varepsilon \}$ 는 닫힌집합인가?

>임의의 $y\in V_\varepsilon(x)$ 를 선택하자. 그러면 정의에 의해 $d(x,y)<\varepsilon$ 이다. 따라서  $\delta=\varepsilon-d(x,y)$ 라고 놓으면 $\delta>0$ 이다.  
>이제 $z\in V_\delta(y)$라고 하자. 그러면 $d(y,z)<\delta$ 이다. 삼각부등식에 의해 $$d(x,z) \le d(x,y)+d(y,z)\ <d(x,y)+\delta  =d(x,y)+\varepsilon-d(x,y) =\varepsilon$$
>
>따라서 $z\in V_\varepsilon(x)$ 이다. 그러므로 $V_\delta(y)\subseteq V_\varepsilon(x).$ 즉, $V_\varepsilon(x)$의 모든 점$y$가 $V_\varepsilon(x)$ 안에 포함되는 작은 근방을 갖는다. 따라서
>
>$$\boxed{V_\varepsilon(x)\text{는 열린집합이다.}}$$
>
>$C_\varepsilon(x)^c = \{y\in X:d(x,y)>\varepsilon\}$ 이다. 임의의 $y\in C_\varepsilon(x)^c$ 를 선택하자. 그러면 $d(x,y)>\varepsilon$ 이므로 $\delta=d(x,y)-\varepsilon>0$ 라고 정의할 수 있다.  
>이제 $z\in V_\delta(y)$라고 하자. 즉, $d(y,z)<\delta$ 라고 하자. 삼각부등식에서 $d(x,y)\le d(x,z)+d(z,y)$ 이므로 $d(x,z)\ge d(x,y)-d(z,y)$ 이다. 따라서
>
>$$d(x,z) \ge d(x,y)-d(z,y) >d(x,y)-\delta = d(x,y)-\bigl(d(x,y)-\varepsilon\bigr) =\varepsilon$$
>
>즉 $z\in C_\varepsilon(x)^c$ 이다. 그러므로 $V_\delta(y)\subseteq C_\varepsilon(x)^c.$ 따라서 $C_\varepsilon(x)^c$는 열린집합이고, $C_\varepsilon(x)$ 는 닫힌집합이다.
>
>(주의할 점은 $C_\varepsilon(x)$ 가 닫힌집합이라는 사실과 $C_\varepsilon(x)=\overline{V_\varepsilon(x)}$ 라는 주장은 서로 다르다는 것이다. 일반적인 거리공간에서는 후자가 항상 성립하지 않는다.()

(b) $E\subseteq X$ 가 열린집합일 필요충분조건은 여집합 $E^c$ 가 닫힌집합임을 보여라.

>($\Rightarrow$) $E$가 열려 있으면 $E^c$ 는 닫혀 있다
>
>$E$가 열린집합이라고 가정하자. $x$가 $E^c$ 의 극한점이라고 하자. $E^c$ 가 닫혔음을 보이려면 $x\in E^c$ 임을 보여야 한다.
>
>반대로 $x \notin E^c$라고 가정하면 $x\in E$ 이다. $E$가 열린집합이므로 어떤 $\varepsilon>0$ 가 존재하여 $V_\varepsilon(x)\subseteq E$ 이다. 그러면 $V_\varepsilon(x)\cap E^c=\varnothing$ 이다. 하지만 $x$가 $E^c$ 의 극한점이므로 모든 $x$의 근방은 $E^c$ 의 점을 포함해야 한다. 이는 모순이다.  
>따라서 $x\in E^c$ 이다. 즉, $E^c$ 는 자신의 모든 극한점을 포함하므로 닫힌집합이다.
>
>($\Leftarrow$) $E^c$ 가 닫혀 있으면 $E$는 열려 있다
>
>$E^c$ 가 닫힌집합이라고 가정하자. 임의의 $x\in E$ 를 선택한다. 그러면 $x\notin E^c$ 이다. $E^c$ 는 닫혀 있으므로 $E^c$ 의 모든 극한점을 포함한다. 따라서 $x\notin E^c$이면 $x$는 $E^c$ 의 극한점도 아니다. 그러므로 어떤 $\varepsilon>0$ 가 존재하여 $\bigl(V_\varepsilon(x)\setminus{x}\bigr)\cap E^c=\varnothing$ 이다.
>
>또한 $x\notin E^c$이므로 실제로 $V_\varepsilon(x)\cap E^c=\varnothing$ 이다. 따라서 $V_\varepsilon(x)\subseteq E$. 이는 모든 $x\in E$ 에 대하여 성립하므로 $E$는 열린집합이다.
>
>결론적으로
>
>$$\boxed{E\text{가 열린집합}\iff E^c\text{가 닫힌집합이다.}}$$
>
>동일하게 드모르간 법칙을 사용하면
>
>$$\boxed{E\text{가 닫힌집합}\iff E^c\text{가 열린집합이다}}$$

도 성립한다.

#### 예제

(a) 집합 $Y=\{f\in C[0,1]:\|f\|_\infty\le1\}$ 이 $C[0,1]$에서 닫힌집합임을 보여라.

집합$y$는 영함수를 중심으로 하고 반지름이 1인 닫힌 공이다. 위 예제 (a)에서 닫힌 공은 항상 닫힌집합임을 보였으므로 바로 결론을 내릴 수 있다.

>직접 증명하면 다음과 같다: **여집합이 열려 있음을 이용한 증명**
>
>임의의 $f\in Y^c$ 를 선택하자. 그러면 $\|f\|_\infty>1$ 이다. 다음과 같이 놓는다. $\varepsilon=\frac{\|f\|_\infty-1}{2}>0.$  
>이제 $\|f-g\|_\infty<\varepsilon$ 인 $g\in C[0,1]$를 생각하자. 
노름의 역삼각부등식에 의해 $|g|_\infty \ge \|f\|_\infty-\|f-g\|_\infty$
>이다. 따라서
>
>$$|g|_\infty >\|f\|_\infty-\varepsilon =\|f\|_\infty-\frac{\|f\|_\infty-1}{2} =\frac{\|f\|_\infty+1}{2} >1$$
>
>그러므로 $g\in Y^c$이다. 따라서 $V_\varepsilon(f)\subseteq Y^c.$ 즉, $Y^c$는 열린집합이므로 $Y$ 는 닫힌집합이다

(b) 집합 $T={f\in C[0,1]:f(0)=0}$ 이 열린집합인지, 닫힌집합인지, 혹은 둘 다 아닌지 판정하라.

>여집합은 $T^c={f\in C[0,1]:f(0)\neq0}$ 이다. 임의의 $f\in T^c$를 선택하자. 그러면 $f(0)\neq0$이므로 $\varepsilon=\frac{|f(0)|}{2}>0$ 라고 정의할 수 있다.  
>이제 $\|f-g\|_\infty<\varepsilon$ 라고 하자. 그러면 특히 $x=0$ 에서도 $|f(0)-g(0)| \le\|f-g\|_\infty <\varepsilon$ 이다. 역삼각부등식에 의해 $|g(0)| \ge |f(0)|-|f(0)-g(0)| >|f(0)|-\frac{|f(0)|}{2} =\frac{|f(0)|}{2}>0$ 따라서 $g(0)\neq0$, 즉 $g\in T^c$ 이다.  
>
>그러므로 $V_\varepsilon(f)\subseteq T^c.$ 따라서 $T^c$가 열린집합이므로 $T$ 는 닫힌집합이다.
>
>위상수학에선 클로펜 집합이 있으므로, 열린집합 여부는 따로 확인해야 한다. 
>
>이제 임의의 $f\in T$와 임의의 $\varepsilon>0$ 을 선택하면 문제의 정의에 따라 $f(0)=0$ 이다. 이제 $g(x)=f(x)+\frac{\varepsilon}{2}.$ 라 하면
>
>$$\|f-g\|_\infty =\sup_{x\in[0,1]} \left|f(x)-f(x)-\frac{\varepsilon}{2}\right| =\frac{\varepsilon}{2}<\varepsilon$$
>
>따라서 $g\in V_\varepsilon(f)$ 이다. 그러나 $g(0)=f(0)+\frac{\varepsilon}{2} =\frac{\varepsilon}{2}\neq0$ 이므로 $g\notin T.$
>
>즉, $f$를 중심으로 하는 모든 $\varepsilon$-근방은 $T$ 밖의 함수를 포함한다. 따라서 $T$ 내부에 완전히 포함되는 근방이 존재하지 않는다.
>
>따라서 $T$ 는 열린집합이 아니다


### 정의. 콤팩트 집합 (Compact set)

거리공간 $(X,d)$의 부분집합 $K$가 다음 조건을 만족하면 콤팩트하다고 한다:

> $K$의 모든 수열이 수렴하는 부분수열을 가지고, 그 부분수열의 극한값이 $K$에 속한다.

- 즉, 임의의 수열 $(x_n)\subseteq K$ 에 대하여 어떤 부분수열 $(x_{n_k})$와 어떤 $x\in K$가 존재하여 $x_{n_k}\to x$ 가 성립하면 $K$는 콤팩트다.
- 이를 수열적 콤팩트성이라고 한다. 거리공간에서는 수열적 콤팩트성과 열린 덮개를 이용한 콤팩트성 정의가 동치다.
- 실수공간과 일반 거리공간의 차이: $\mathbb R^n$ 에서는 하이네–보렐 정리에 의해 '$K\subseteq\mathbb R^n$ 가 콤팩트 $\iff K$ 가 닫히고 유계다' 가 성립한다.
  - 하지만 일반적인 거리공간에서는 '콤팩트 $\Rightarrow$  닫힌 유계' 만 항상 성립하고, '닫힌 유계 $\Rightarrow$ 콤팩트' 는 일반적으로 성립하지 않는다.


#### 예제
(a) 거리공간 $(X,d)$의 유계 (bounded) 인 부분집합을 정의하라.

>부분집합 $E\subseteq X$에 대하여 어떤 $x_0\in X$와 어떤 어떤 $M>0$가 존재하여 $\forall x\in E,\ d(x_0,x)\le M$ 가 성립하면 $E$를 유계집합이라고 한다. 
>- 즉, $E\subseteq C_M(x_0)$ 가 되는 중심 $x_0$와 반지름 $M$이 존재하면 $E$는 유계다.
>
>- 동치인 정의: $\exist M>0$, s.t. $\forall x,y\in E$에 대하여 $d(x,y)\le M$ 이면 $E$는 유계다.
>
>- 집합의 지름을 $\operatorname{diam}(E)=\sup{d(x,y):x,y\in E}$ 
>라고 하면 $E$ 가 유계 $\iff$ $\operatorname{diam}(E)<\infty$ 

(b) $K$가 거리공간 $(X,d)$의 콤팩트 부분집합이면 $K$는 닫힌 유계집합임을 보여라.

>두 부분으로 나누어 증명한다.
>
>1. 콤팩트집합은 유계다
>
>$K$가 콤팩트이지만 유계가 아니라고 가정하자. $K\neq\varnothing$인 경우 하나의 점 $x_0\in K$를 고정한다. $K$가 유계가 아니므로 모든 자연수 $n$에 대하여 $d(x_0,x_n)>n$ 을 만족하는 점 $x_n\in K$를 선택할 수 있다. 이렇게 얻은 수열 $(x_n)\subseteq K$ 를 생각하자.  
>$K$가 콤팩트이므로 $(x_n)$은 $K$의 어떤 점 $x\in K$로 수렴하는 부분수열 $x_{n_k}\to x$ 를 가져야 한다. 그런데 수렴하는 수열은 유계다 (실수에서와 유사하게 증명가능). 실제로 $x_{n_k}\to x$이므로 충분히 큰 $k$에 대하여 $d(x_{n_k},x)<1$ 이다. 따라서 삼각부등식에 의해 $d(x_0,x_{n_k}) \le d(x_0,x)+d(x,x_{n_k}) < d(x_0,x)+1$. 그러나 수열을 선택한 방법에 따르면 $d(x_0,x_{n_k})>n_k\to\infty$ 
>이다. 이는 모순이다. 따라서 $K$ 는 유계다.
>
>2. 콤팩트집합은 닫혀 있다
>
>$x$가 $K$의 극한점이라고 하자. $K$가 닫혀 있음을 보이려면 $x\in K$ 
>임을 보이면 된다.  
>$x$가 $K$의 극한점이므로 모든 $n\in\mathbb N$에 대하여 $V_{1/n}(x)$ 
>안에는 $x$와 다른 $K$의 점이 존재한다. 그러한 점 하나를 $x_n$이라고 선택하면 $x_n\in K,\ x_n\neq x,\ d(x_n,x)<\frac1n $ 이다. 그러므로 $x_n\to x$ 이다.
>
>한편 $K$가 콤팩트이므로 $(x_n)$은 $K$의 어떤 점 $y\in K$로 수렴하는 부분수열을 갖는다. $x_{n_k}\to y,\ y\in K$ 이다. 하지만 원래 $x_n\to x$이므로 모든 부분수열도 $x$로 수렴한다. $x_{n_k}\to x$ 이다.
>
>거리공간에서 수열의 극한은 유일하므로 $x=y$ 이다. $y\in K$이므로 $x\in K$ 이다. 
>
>따라서 $K$는 자신의 모든 극한점을 포함한다. $K$ 는 닫힌집합이다
>
>두 결과를 합치면
>
>$$
>\boxed{\text{거리공간에서 콤팩트집합은 항상 닫히고 유계다.}}
>$$
>
(c) $Y=\{f\in C[0,1]:\|f\|_\infty\le1\}$ 은 닫힌 유계집합이지만 콤팩트집합은 아님을 보여라.
>
>1단계: $Y$는 닫힌집합이다: 이전 문제(a)에서 증명했다. $Y=\{f\in C[0,1]:\|f\|_\infty\le1\}$ 은 영함수를 중심으로 하는 반지름 $1$의 닫힌 공이므로 닫힌집합이다.
>
>2단계: $Y$는 유계집합이다
>
>모든 $f\in Y$에 대하여 $d(f,0)=\|f\|_\infty\le1$ 이다. 따라서 $Y\subseteq C_1(0)$ 이므로 $Y$는 유계다.  
>또는 $f,g\in Y$에 대하여 $d(f,g) =\|f-g\|_\infty\ \le \|f\|_\infty+\|g\|_\infty\ \le 2$ 이므로 $\operatorname{diam}(Y)\le2$ 이다.
>
>3단계: $Y$는 콤팩트집합이 아니다
>
>다음 함수열을 생각하자: $f_n(x)=x^n,\ x\in[0,1]$. 각 $f_n$은 연속이고 $\|f_n\|_\infty = \max_{x\in[0,1]}x^n$ 이다. 따라서 $f_n\in Y$ 이다.
>
>이제 $(f_n)$이 최소상계 노름에서 수렴하는 부분수열을 갖지 않는다는 것을 보이면 된다.
>
>임의의 부분수열 $(f_{n_k})$을 생각하자. $n_k\to\infty$이므로 $0\le x<1$에 대하여 $x^{n_k}\to0$ 이다. 반면 $x=1$에서는 $1^{n_k}=1$ 이다.
>
>따라서 모든 부분수열의 점별 극한은
>
>$$
>f(x)= 
>\begin{cases} 0,&0\le x<1,\\
>1,&x=1 \end{cases}
>$$
>
>이 함수는 $x=1$에서 불연속이다.
>
>만약 어떤 부분수열 $(f_{n_k})$이 최소상계 노름에서 어떤 $g\in C[0,1]$로 수렴한다면 $\|f_{n_k}-g\|_\infty\to0$ 이므로 $f_{n_k}\to g$는 균등수렴이다. 균등수렴은 점별수렴을 함의하므로 $g$는 위 함수 $f$와 같아야 한다.
>
>그러나 연속함수열의 균등극한은 연속이어야 하는데 $f$는 불연속이다. 이는 모순이다.  
>따라서 $(f_n)$은 $Y$ 안에서 수렴하는 부분수열을 갖지 않는다. 그러므로 $Y$는 콤팩트하지 않다.
>
>$$
>\boxed{Y\text{는 닫히고 유계이지만 콤팩트하지 않다.}}
>$$

> 참고: 콤팩트성의 정의와 연속함수의 고름 극한이 연속함수임을 생각하면, 아르젤라-아스콜리 정리 (Arzela-Ascoli theorem)는 임의의 유계이면서 닫혀있고 동등역속인 함수의 집합은 $C[0,1]$의 콤팩트한 부분집합임을 의미한다.

### 정의. 폐포(closure)
거리공간 $(X,d)$와 부분집합 $E\subseteq X$를 생각하자.

**폐포**  
집합 $E$와 $E$의 모든 극한점의 합집합을 $E$의 폐포라고 하며

$$
\overline E
$$

로 표기한다. 즉, $\overline E=E\cup\{x\in X:x\text{는 }E\text{의 극한점}\}$.

#### 폐포의 근방을 이용한 표현

다음 조건들은 서로 동치다.

$x\in\overline E \iff \text{모든 }\varepsilon>0\text{에 대하여 }
V_\varepsilon(x)\cap E\neq\varnothing$

여기에서는 $x$가 $E$의 극한점일 필요까지는 없다. $x\in E$이면 모든 근방이 적어도 $x$ 자체를 포함하므로 자동으로 $x\in\overline E$다.

반면 $x$가 $E$의 극한점이라는 것은 더 강하게 $\bigl(V_\varepsilon(x)\setminus{x}\bigr)\cap E \neq\varnothing$ 가 모든 $\varepsilon>0$에 대해 성립한다는 뜻이다.

#### 폐포의 수열을 이용한 표현

거리공간에서는 다음도 동치다.

$x\in\overline E \iff E\text{의 어떤 수열 }(x_n)\text{이 존재하여 }x_n\to x$

실제로 $x\in\overline E$이면 모든 $n\in\mathbb N$에 대하여 $V_{1/n}(x)\cap E\neq\varnothing$ 이므로 $x_n\in V_{1/n}(x)\cap E$ 인 점을 선택할 수 있다. 그러면 $d(x_n,x)<\frac1n$ 이므로 $x_n\to x$다.

### 정의. 내부(interior)

집합 $E$의 내부를 다음과 같이 정의한다.

$$
E^\circ= \{x\in E: \text{어떤 }\varepsilon>0\text{에 대하여 }
V_\varepsilon(x)\subseteq E \}
$$

즉, $E^\circ$는 자신의 주변에 $E$ 안에 완전히 포함되는 근방을 갖는 점들의 집합이다.  
항상 $E^\circ\subseteq E$이다.

- 폐포와 내부는 쌍대 개념이다. 두 개념에 대한 결과는 쌍을 이루며 우아하고 유용한 대칭성을 내포한다. 
- 폐포 $\overline E$는 $E$에 경계에 있는 극한점들을 추가하여 닫힌집합으로 만든 것이다.
- 내부 $E^\circ$는 $E$에서 경계에 있는 점들을 제외하고, 충분히 작은 근방까지 모두 $E$ 안에 들어가는 점들만 남긴 것이다.
  * $\overline E$는 $E$를 포함하는 가장 작은 닫힌집합이다.
  * $E^\circ$는 $E$에 포함되는 가장 큰 열린집합이다.

**예시**  
실수의 일반 거리에서 $E=(0,1]$ 이라고 하자. 그러면 $\overline E=[0,1]$ 이고 $E^\circ=(0,1)$ 이다.

$0$은 $E$에 속하지 않지만 $E$의 극한점이므로 폐포에는 포함된다. 반면 $1$은 $E$에 속하지만 $1$을 중심으로 하는 어떤 열린 근방도 $E$ 안에 완전히 포함되지 않으므로 내부에는 포함되지 않는다.


#### 예제
$E$가 닫힌집합일 필요충분조건은 $\overline E=E$임을 보여라.

>($\Rightarrow$)  
>$E$가 닫힌집합이라고 가정하자. 닫힌집합은 자신의 모든 극한점을 포함한다. 따라서 $E$에 모든 극한점을 추가하더라도 새로운 점이 추가되지 않는다. 즉, $\overline E\subseteq E$ 이다.
>
>한편 폐포의 정의에 의해 항상 $E\subseteq\overline E$ 이다. 따라서 $\overline E=E$ 이다.
>
>($\Leftarrow$)  
>이제 $\overline E=E$ 라고 가정하자. $E$의 모든 극한점은 폐포 $\overline E$에 속한다. 그런데 $\overline E=E$이므로 $E$의 모든 극한점이 $E$에 속한다. 따라서 $E$는 자신의 모든 극한점을 포함하므로 닫힌집합이다.

$E$가 열린집합일 필요충분조건은 $E^\circ=E$임을 보여라.

>($\Rightarrow$)  
>$E$가 열린집합이라고 가정하자. 그러면 모든 $x\in E$에 대하여 어떤 $\varepsilon>0$가 존재하여 $V_\varepsilon(x)\subseteq E$ 이다. 따라서 모든 $x\in E$는 $E$의 내부점이므로 $E\subseteq E^\circ$ 이다.
>
>한편 내부의 정의에 의해 항상 $E^\circ\subseteq E$ 이므로 $E^\circ=E$ 이다.
>
>($\Leftarrow$)  
>이제 $E^\circ=E$ 라고 가정하자. 임의의 $x\in E$를 선택하면 $x\in E^\circ$다. 내부의 정의에 의해 어떤 $\varepsilon>0$가 존재하여 $V_\varepsilon(x)\subseteq E$ 이다. 이는 모든 $x\in E$에 대해 성립하므로 $E$는 열린집합이다.

#### 예제
다음을 보여라: $(\overline E)^c=(E^c)^\circ$  
이는 폐포 바깥의 점들이 여집합의 내부점이라는 뜻이다.
>($\Leftarrow$)  
$x\in(\overline E)^c$ 라고 하자. 그러면 $x\notin\overline E$ >이다.  
폐포의 근방 표현에 따르면 $x\notin\overline E$이라는 것은 어떤 >$\varepsilon>0$가 존재하여 $V_\varepsilon(x)\cap E=\varnothing$ 이라는 뜻이다. 따라서 $V_\varepsilon(x)\subseteq E^c$ 이다. 그러므로 $x$는 $E^c$의 내부점이다. 즉, $x\in(E^c)^\circ$ 이다. 따라서 $(\overline E)^c\subseteq(E^c)^\circ$ 이다.
>
>($\Rightarrow$)  
>이번에는 $x\in(E^c)^\circ$ 라고 하자.  
>내부의 정의에 의해 어떤 $\varepsilon>0$가 존재하여 $V_\varepsilon(x)\subseteq E^c$ 이다. 따라서 $V_\varepsilon(x)\cap E=\varnothing$ 이다. 그러므로 $x$는 $\overline E$에 속하지 않는다.  
>$x\notin\overline E$ 이므로 $x\in(\overline E)^c$ 이다. 따라서 $(E^c)^\circ\subseteq(\overline E)^c$ 이다.

또한 $(E^\circ)^c=\overline{E^c}$ 임을 보여라.

>앞에서 증명한 식 $(\overline E)^c=(E^c)^\circ$ 에서 $E$ 대신 $E^c$를 대입한다.  
>그러면 $\left(\overline{E^c}\right)^c=\left((E^c)^c\right)^\circ$ 이다. $(E^c)^c=E$이므로 $\left(\overline{E^c}\right)^c=E^\circ$ 이다.  
>양변에 다시 여집합을 취하면 $\overline{E^c}=(E^\circ)^c$ 이다. 

이 결과는 폐포와 내부의 대칭성으로 정리된다.

$$
\boxed{
  (\overline E)^c=(E^c)^\circ, \quad (E^\circ)^c=\overline{E^c}
}
$$

즉, 폐포에 여집합을 취하면 여집합의 내부가 되고, 내부에 여집합을 취하면 여집합의 폐포가 된다.

이 결과로부터 다음도 확인할 수 있다.

* $\overline E$는 닫힌집합이다.
* $E^\circ$는 열린집합이다.

실제로  $(\overline E)^c=(E^c)^\circ$ 이고 내부는 열린집합이므로 $(\overline E)^c$가 열려 있다. 따라서 $\overline E$는 닫혀 있다.  
마찬가지로 $(E^\circ)^c=\overline{E^c}$ 이고 폐포는 닫힌집합이므로 $(E^\circ)^c$가 닫혀 있다. 따라서 $E^\circ$는 열려 있다.

#### 예제

임의의 거리공간 $(X,d)$에 대하여 다음을 보여라: $\overline{V_\varepsilon(x)} \subseteq \{y\in X:d(x,y)\le\varepsilon \}$

>임의의 $y\in\overline{V_\varepsilon(x)}$ 를 선택하자. $d(x,y)>\varepsilon$ 라고 가정하자. 그러면 $\delta=d(x,y)-\varepsilon>0$ 이라고 놓을 수 있다.
>
>이제 임의의 $z\in V_\delta(y)$를 생각하자. 그러면 $d(y,z)<\delta$ 이다. 삼각부등식 $d(x,y)\le d(x,z)+d(z,y)$  에 의해 $d(x,z)\ge d(x,y)-d(z,y)$ 이다. 따라서
>
>$$
>d(x,z) \ge d(x,y)-d(z,y) >d(x,y)-\delta = d(x,y)-\bigl(d(x,y)-\varepsilon\bigr) = \varepsilon.
>$$
>
>그러므로 $z\notin V_\varepsilon(x)$ 이다.
>
>이는 모든 $z\in V_\delta(y)$에 대하여 성립하므로 $V_\delta(y)\cap V_\varepsilon(x)=\varnothing$ 이다.
>
>하지만 $y\in\overline{V_\varepsilon(x)}$이면 $y$의 모든 근방은 $V_\varepsilon(x)$와 교집합을 가져야 한다. 이는 모순이다.
>
>따라서 $d(x,y)\le\varepsilon$ 이다. 그러므로
>
>$$
>\boxed{
>\overline{V_\varepsilon(x)}
>\subseteq
>{y\in X:d(x,y)\le\varepsilon}
>}
>$$

>다른 설명: 
>닫힌 공 $C_\varepsilon(x)=\{y\in X:d(x,y)\le\varepsilon\}$ 은 닫힌집합임을 증명했다. 또한 $V_\varepsilon(x)\subseteq C_\varepsilon(x)$ 이다. 
>폐포는 주어진 집합을 포함하는 가장 작은 닫힌집합이므로 
$\overline{V_\varepsilon(x)} \subseteq C_\varepsilon(x)$ 이다.

#### 예제
다음을 만족하는 거리공간의 예를 구체적으로 구하라.
$$
\overline{V_\varepsilon(x)} \neq \{y\in X:d(x,y)\le\varepsilon\}
$$

일반적인 유클리드 거리에서는 열린 공의 폐포와 닫힌 공이 같다. 하지만 일반적인 거리공간에서는 반드시 같지는 않다.

>원소가 두 개 이상인 집합 $X$에 이산 거리
>
>$$
>d(x,y)=\begin{cases}
>0,&x=y,\\
>1,&x\neq y
>\end{cases}
>$$
>
>를 부여하자. 하나의 점 $x\in X$를 고정하고 $\varepsilon=1$ 로 선택한다. 열린 공은 $V_1(x)={y\in X:d(x,y)<1}$ 이다. 이산 거리에서 $d(x,y)<1$이려면 반드시 $d(x,y)=0$이어야 하므로 $y=x$다. 따라서 $V_1(x)={x}$ 이다. 이산 거리공간에서는 모든 한 점 집합이 닫힌집합이므로 $\overline{V_1(x)}=\overline{\{x\}} = \{x\}$ 이다.
>
>반면 닫힌 공은 $C_1(x) =\{y\in X:d(x,y)\le1\} =X$ 이다. 이산 거리의 값은 항상 $0$ 또는 $1$이므로 모든 $y\in X$가 포함된다.
>
>따라서 $X$에 $x$ 이외의 점이 존재하면 $\{x\}\neq X$ 이므로
>
>$$
>\boxed{
>\overline{V_1(x)}=\{x\} \neq X = \{y\in X:d(x,y)\le1\}
>}
>$$

### 정의. 조밀 집합 (dense set)

거리공간 $(X,d)$에서 부분집합 $A\subseteq X$가 $\overline A=X$ 를 만족하면 $A$가 $X$에서 조밀(dense) 하다고 한다.

즉, $A$의 폐포가 전체 공간을 덮는다는 뜻이다.

- 다음 조건들은 서로 동치다
  - $\overline A=X \iff
\text{모든 }x\in X\text{와 모든 }\varepsilon>0\text{에 대하여 }
V_\varepsilon(x)\cap A\neq\varnothing$
  - $A\text{는 }X\text{에서 조밀하다} \iff X\text{의 모든 공집합이 아닌 열린집합이 }A\text{와 교차한다}$
- 즉, $A$가 조밀하다는 것은 $X$의 어느 부분을 확대해도 $A$의 점을 찾을 수 있다는 뜻이다.

예: 유리수와 무리수  
일반 거리가 주어진 $\mathbb R$에서 $\overline{\mathbb Q}=\mathbb R$ 이므로 $\mathbb Q$는 $\mathbb R$에서 조밀하다. 마찬가지로 무리수 집합 $\mathbb R\setminus\mathbb Q$도 $\mathbb R$에서 조밀하다.

### 정의. 조밀한 곳이 없는 집합 (nowhere dense set)

부분집합 $E\subseteq X$가 $(\overline E)^\circ=\varnothing$ 을 만족하면 $E$가 $X$에서 조밀한 곳이 없다(nowhere dense)고 한다.

즉, $E$의 폐포가 어떤 공집합이 아닌 열린집합도 포함하지 않는다는 뜻이다.

- 빈 내부와 nowhere dense의 차이
  - $E^\circ=\varnothing$ 와 $(\overline E)^\circ=\varnothing $는 다르다.
  - 두 번째 조건이 더 강하다.

예를 들어 $\mathbb Q\subseteq\mathbb R$에 대하여 $\mathbb Q^\circ=\varnothing$ 이지만 $\overline{\mathbb Q}=\mathbb R$ 이므로 $(\overline{\mathbb Q})^\circ= \mathbb R^\circ = \mathbb R \neq\varnothing$ 이다.

따라서 $\mathbb Q$는 내부가 비어 있지만 nowhere dense는 아니다. 오히려 $\mathbb R$에서 조밀하다.

반면 한 점 집합 ${x}\subseteq\mathbb R$은 $\overline{\{x\}}=\{x\}$ 이고 ${x}^\circ=\varnothing$ 이므로 nowhere dense다.


#### 예제
거리공간 $(X,d)$의 부분집합 $E$에 대하여, $E$가 $X$에서 조밀한 곳이 없을 필요충분조건은 $(\overline E)^c$가 $X$에서 조밀한 것임을 보여라.

- 왼쪽은 $E$가 nowhere dense라는 뜻이고, 오른쪽은 $(\overline E)^c$가 $X$에서 조밀하다는 뜻이다.

>**증명1**  
>$A=(\overline E)^c$ 라 하자. 문제 11(b)에 의해 임의의 집합 $A\subseteq X$에 대하여 $(\overline A)^c=(A^c)^\circ$ 가 성립한다. 여기에 $A=(\overline E)^c$를 대입하면 $\left(\overline{(\overline E)^c}\right)^c=\left(\left((\overline E)^c\right)^c\right)^\circ$ 이다. 여집합을 두 번 취하면 원래 집합이므로 $\left((\overline E)^c\right)^c=\overline E$ 이다. 따라서 $\left(\overline{(\overline E)^c}\right)^c=(\overline E)^\circ$ 이다.
>
>이제 $E$가 nowhere dense라고 하자. 그러면 $(\overline E)^\circ=\varnothing$ 이므로 $\left(\overline{(\overline E)^c}\right)^c=\varnothing$ 이다. 어떤 집합의 여집합이 공집합이면 그 집합은 전체 공간이므로 $\overline{(\overline E)^c}=X$ 이다. 따라서 $(\overline E)^c$는 $X$에서 조밀하다.
>
>역으로 $(\overline E)^c$가 $X$에서 조밀하다고 하자. 그러면 $\overline{(\overline E)^c}=X$ 이다. 따라서$\left(\overline{(\overline E)^c}\right)^c = X^c =\varnothing $ 이다. 앞에서 얻은 등식에 의해 $(\overline E)^\circ=\varnothing$ 이다. 따라서 $E$는 $X$에서 조밀한 곳이 없다.

>**증명2: 열린집합의 관점**  
>
>$E$가 nowhere dense라는 것은  $(\overline E)^\circ=\varnothing$ 이라는 뜻이다. 이는 $\overline E$ 안에 공집합이 아닌 열린집합이 하나도 포함되지 않는다는 뜻이다. 따라서 임의의 공집합이 아닌 열린집합 $O\subseteq X$에 대하여 $O\not\subseteq\overline E$ 이다. 그러므로 $O\cap(\overline E)^c\neq\varnothing$ 이다.
>
>즉, 모든 공집합이 아닌 열린집합이 $(\overline E)^c$와 교차한다. 따라서 $(\overline E)^c$ 는 $X$에서 조밀하다.
>
>역으로 $(\overline E)^c$가 조밀하면 모든 공집합이 아닌 열린집합은 $(\overline E)^c$와 교차한다. 따라서 어떤 공집합이 아닌 열린집합도 $\overline E$ 안에 완전히 포함될 수 없다.
>
>그러므로 $(\overline E)^\circ=\varnothing$ 이고 $E$는 nowhere dense다.




### 정리 8.2.10

완비거리공간 $(X,d)$과 $X$의 조밀한 열린 부분집합들로 이루어진 셀 수 있는 모임 $\{O_n:n\in\mathbb N\}$ 을 생각하자. 이때

$$
\boxed{\bigcap_{n=1}^{\infty}O_n\neq\varnothing}
$$

이다. 즉, 완비거리공간에서 셀 수 있는 개수의 조밀한 열린집합을 교차시키면 공집합이 되지 않는다. 이 정리는 베르 범주 정리의 핵심 형태다.


>**증명** 
>
>$\mathbb R$에서 이 정리를 증명할때는 축소구간정리를 사용했다. 거리공간에서도 비슷한 성질이 있지만, 지금은 거리공간의 완비성을 정의한 방식인 코시수열의 수렴성으로 증명한다. 
>
>먼저 $x_1\in O_1$을 선택한다. $O_1$이 열린집합이므로 어떤 $\varepsilon_1>0$가 존재하여 $V_{\varepsilon_1}(x_1)\subseteq O_1$ 이다. $V_{\varepsilon_1}(x_1)$은 공집합이 아닌 열린집합이다. 
>
>한편 $O_2$는 $X$에서 조밀하다. 폐포의 근방을 이용한 표현과,  조밀성에 의해 모든 공집합이 아닌 열린집합은 $O_2$와 교차하므로 $V_{\varepsilon_1}(x_1)\cap O_2\neq\varnothing$ 이다. 따라서 $x_2\in V_{\varepsilon_1}(x_1)\cap O_2$ 인 점을 선택할 수 있다.
>
>한편, $x_2\in O_2$이고 $O_2$가 열린집합이므로 어떤 $r_2>0$가 존재하여 $V_{r_2}(x_2)\subseteq O_2$ 이다. 또한 $x_2\in V_{\varepsilon_1}(x_1)$ 이므로 $d(x_1,x_2)<\varepsilon_1$ 이다. 따라서 $s_2=\varepsilon_1-d(x_1,x_2)>0$ 라고 정의할 수 있다.
>
>이제 $0<\varepsilon_2 < \min\{\frac{\varepsilon_1}{2},r_2,s_2\}$ 가 되도록 $\varepsilon_2$를 선택한다. 그러면 $\varepsilon_2<r_2$이므로 $V_{\varepsilon_2}(x_2)\subseteq V_{r_2}(x_2)\subseteq O_2$ 이다.
>
>다음으로 $z\in\overline{V_{\varepsilon_2}(x_2)}$ 라고 하자. 예제 12(a)에 의해 $d(x_2,z)\le\varepsilon_2$ 이다. 따라서 삼각부등식에 의해 
>$d(x_1,z) \le d(x_1,x_2)+d(x_2,z) \le d(x_1,x_2)+\varepsilon_2 <d(x_1,x_2)+s_2 =\varepsilon_1$. 그러므로 $z\in V_{\varepsilon_1}(x_1)$ 이다.  
>따라서 $\overline{V_{\varepsilon_2}(x_2)} \subseteq V_{\varepsilon_1}(x_1)$ 이다.
>
>이 사실과 $(X,d)$의 완비성을 이용하여 모든 $n\in\mathbb N$에 대하여 $x\in O_n$ 인 점 $x$를 찾아보자: 점과 근방의 귀납적 구성
>
>위 방법을 반복하면 각 $n\in\mathbb N$에 대하여 점 $x_n\in X$와 양수 $\varepsilon_n>0$를 다음 조건을 만족하도록 선택할 수 있다: $V_{\varepsilon_n}(x_n)\subseteq O_n$
>이고 
>
>$$\varepsilon_{n+1}<\frac{\varepsilon_n}{2}$$
>
>이며 $\overline{V_{\varepsilon_{n+1}}(x_{n+1})} \subseteq V_{\varepsilon_n}(x_n)$ 이다.
>
>따라서 근방들은 다음과 같이 포개진다.
>$\overline{V_{\varepsilon_{n+1}}(x_{n+1})} \subseteq V_{\varepsilon_n}(x_n) \subseteq O_n$
>
>반지름에 대해서는 $\varepsilon_n < \frac{\varepsilon_1}{2^{n-1}}$ 이므로 $\varepsilon_n\to0$ 이다.
>
>$m>n$이라고 하자. 근방들의 포함관계에 의해 $x_m\in V_{\varepsilon_n}(x_n)$ 이다. 따라서 $d(x_m,x_n)<\varepsilon_n < \frac{\varepsilon_1}{2^{n-1}}$
>
>임의의 $\varepsilon>0$을 선택하면 충분히 큰 $N$에 대하여 $\frac{\varepsilon_1}{2^{N-1}}<\varepsilon$ 이다.  
>따라서 $m>n\ge N$이면
>
>$$d(x_m,x_n) < \frac{\varepsilon_1}{2^{n-1}} \le \frac{\varepsilon_1}{2^{N-1}} <\varepsilon.$$
>
>그러므로 $(x_n)$은 코시 수열이다.
>
>완비성 적용: $(X,d)$가 완비거리공간이므로 코시 수열 $(x_n)$은 $X$의 어떤 점 $x$로 수렴한다. $x_n\to x$
>
>이제 $x$가 모든 $O_n$에 속한다는 것을 보이면 된다.
>
>임의의 $n\in\mathbb N$을 고정하자. 근방들의 포함관계에 의해 모든 $m\ge n+1$에 대하여 $x_m\in V_{\varepsilon_{n+1}}(x_{n+1})$
>이다. 따라서 수열의 꼬리 부분이 $\overline{V_{\varepsilon_{n+1}}(x_{n+1})}$ 안에 들어 있다.
>
>폐포는 닫힌집합이고 $x_m\to x$이므로 $x\in \overline{V_{\varepsilon_{n+1}}(x_{n+1})}$. 그런데 구성에 의해
>$\overline{V_{\varepsilon_{n+1}}(x_{n+1})} \subseteq
>V_{\varepsilon_n}(x_n) \subseteq O_n$ 이다. 따라서 $x\in O_n$ 이다.
>
>$n$이 임의였으므로 $x\in\bigcap_{n=1}^{\infty}O_n.$ 따라서 $\boxed{ \bigcap_{n=1}^{\infty}O_n\neq\varnothing}$ 이다.

### 정리 8.2.11: 베르 범주 정리

완비거리공간은 조밀한 곳이 없는 집합(nowhere dense set) 들의 셀 수 있는 합집합(the union of a countable collection)으로 나타낼 수 없다.

즉, $(X,d)$가 완비거리공간이고 각 $E_n\subseteq X$가 nowhere dense라면
$\boxed{
X\neq\bigcup_{n=1}^{\infty}E_n
}$
이다.

**제1범주와 제2범주**

거리공간 $X$의 부분집합 $A$가 조밀한 곳이 없는 집합들의 셀 수 있는 합집합으로 표현되면 $A$를 **제1범주 집합 또는 빈약(meager) 집합** 이라고 한다. 즉, $A=\bigcup_{n=1}^{\infty}E_n$ 이고 각 $E_n$이 nowhere dense이면 $A$는 제1범주다.  
일반적으로는 $A\subseteq\bigcup_{n=1}^{\infty}E_n$ 처럼 셀 수 있는 nowhere dense 집합들의 합집합에 포함되는 경우도 제1범주라고 정의한다.

제1범주가 아닌 집합을 제2범주 집합이라고 한다. 베르 범주 정리는 완비거리공간 $X$가 자기 자신 안에서 제2범주라는 뜻이다.

- nowhere dense 집합은 폐포를 취해도 어떤 열린 근방 하나를 완전히 차지하지 못하는 집합이다. 따라서 공간 안에서 위상적으로 매우 얇은 집합으로 볼 수 있다.
- 베르 범주 정리는 완비거리공간 전체를 이처럼 얇은 집합들을 셀 수 있게 모아서 만들 수 없다는 정리다.
- 측도론에서 측도가 $0$인 집합이 "크기가 작은 집합"을 표현한다면, 범주론에서는 제1범주 집합이 "위상적으로 작은 집합"을 표현한다.
  - 다만 측도가 $0$이라는 것과 제1범주라는 것은 서로 다른 개념이다.

>**증명**
>
>반대로 완비거리공간 $X$가 조밀한 곳이 없는 집합들의 셀 수 있는 >합집합으로 표현된다고 가정하자.  
>$X=\bigcup_{n=1}^{\infty}E_n$ 이고 각 $E_n$은 nowhere dense라고 하자. 각 $n$에 대하여$O_n=(\overline{E_n})^c$ 라고 정의한다. 
>
>$\overline{E_n}$은 닫힌집합이므로 $O_n$은 열린집합이다. 또한 >$E_n$이 nowhere dense이므로 이전 예제에 의해 $(\overline{E_n})^c$ 는 $X$에서 조밀하다. 따라서 각 $O_n$은 조밀한 열린집합이다.
>
>이전 정리를 적용하면 $\bigcap_{n=1}^{\infty}O_n\neq\varnothing$
>이다. 따라서 어떤 $x\in X$가 존재하여 모든 $n\in\mathbb N$에 대해
>$x\in O_n=(\overline{E_n})^c$
이다. 그러므로 모든 $n$에 대하여 $x\notin\overline{E_n}$ 이다. >항상 $E_n\subseteq\overline{E_n}$이므로 $x\notin E_n$ 이다.
>
>따라서 $x\notin\bigcup_{n=1}^{\infty}E_n$. 하지만 가정에 따르면
$X=\bigcup_{n=1}^{\infty}E_n$ 이므로 모든 $x\in X$는 오른쪽 >합집합에 속해야 한다. 이는 모순이다.
>
>따라서 $X\neq\bigcup_{n=1}^{\infty}E_n$ 이다. 즉, 완비거리공간은 nowhere dense 집합들의 셀 수 있는 합집합으로 표현할 수 없다.

### 정리 8.2.12

다음 집합을 생각하자: $D= \{f\in C[0,1]:  \text{어떤 }x\in[0,1]\text{에서 }f'(x)\text{가 존재한다}\}.$

이 $D$는 $C[0,1]$에서 제1범주 집합이다. 즉, 적어도 한 점에서라도 미분 가능한 연속함수들의 집합은 최소상계 노름이 주어진 $C[0,1]$에서 위상적으로 매우 작은 집합이다.

>**정리 8.2.12와 베르 범주 정리의 의미**
>
>문제 5(b)에서 최소상계 노름이 주어진 공간
>$\left(C[0,1],|\cdot|_\infty\right)$
>이 완비거리공간임을 증명했다. 따라서 베르 범주 정리에 의해 $C[0,1]$ 전체는 제1범주 집합일 수 없다.
>
>그런데 적어도 한 점에서 미분 가능한 연속함수들의 집합 $D$는 제1범주다. 따라서 $D\neq C[0,1].$ 즉, $C[0,1]\setminus D\neq\varnothing$ 이다. $C[0,1]\setminus D$에 속하는 함수는 어떤 점에서도 미분 가능하지 않은 연속함수다.  
>따라서 $\boxed{ \text{모든 점에서 미분 불가능한 연속함수가 존재한다.}}$ 는 결론을 얻는다.
>
>정리 8.2.12는 단순히 그러한 함수가 하나 존재한다는 것보다 더 강한 의미를 갖는다. 적어도 한 점에서 미분 가능한 연속함수들의 집합이 제1범주이므로, 위상적 관점에서는 미분 가능한 연속함수보다 어디에서도 미분 가능하지 않은 연속함수가 훨씬 일반적이라는 뜻이다.

>**증명** 
>
>자연수 $m,n$에 대하여 다음 집합을 정의한다.
>
>$$A_{m,n}=\{ f\in C[0,1]: 
>\text{어떤 }x\in[0,1]\text{가 존재하여,}\\
>0<|x-t|<1/m\text{인 모든 }t\in[0,1]\text{에 대해}\
>\left|\frac{f(x)-f(t)}{x-t}\right| \le n
>\}$$
>
>이 정의는 어떤 점 $x$ 주변에서 모든 할선의 기울기 크기가 $n$ 이하로 제한된다는 뜻이다. 단, 이것만으로 $f$가 $x$에서 미분 가능하다는 뜻은 아니다. 차분몫이 유계라는 것과 차분몫이 하나의 값으로 수렴한다는 것은 서로 다른 조건이다.
>
>한편 (문제 16), $f\in C[0,1]$가 어떤 $x\in[0,1]$에서 미분 가능하면 어떤 $m,n\in\mathbb N$에 대하여 $f\in A_{m,n}$ 임을 보이자. 
>
>$f$가 미분 가능하므로 $\lim_{t\to x} \frac{f(x)-f(t)}{x-t}=f'(x)$ 이다. 미분계수 $f'(x)$는 유한한 실수다. 극한의 정의를 오차 $1$에 적용하면 어떤 $\delta>0$가 존재하여 $0<|x-t|<\delta$ 이면 $\left| \frac{f(x)-f(t)}{x-t} - f'(x) \right|<1$ 이다. 따라서 삼각부등식에 의해 $\left|\frac{f(x)-f(t)}{x-t}\right| \le \left| \frac{f(x)-f(t)}{x-t} -f'(x) \right| + |f'(x)| < 1+|f'(x)|$
>
>이제 $\frac1m<\delta$ 가 되도록 $m\in\mathbb N$을 선택하고 $n\ge |f'(x)|+1$ 이 되도록 $n\in\mathbb N$을 선택한다. 그러면 $0<|x-t|<\frac1m$ 인 모든 $t\in[0,1]$에 대하여 $\left| \frac{f(x)-f(t)}{x-t} \right| \le n$ 이다. 따라서 $f\in A_{m,n}.$ 
>
>이제 $f\in D$이면 $f$는 어떤 $x\in[0,1]$에서 미분 가능하므로, 문제 16에 의해 어떤 $m,n\in\mathbb N$에 대하여 $f\in A_{m,n}$이다. 따라서 $f\in \bigcup_{m,n\in\mathbb N}A_{m,n}$ 즉, $D\subseteq \bigcup_{m,n\in\mathbb N}A_{m,n}$  
>따라서 $D$는 셀 수 있는 개수의 집합 $A_{m,n}$들의 합집합 속에 포함된다. $\mathbb N\times\mathbb N$은 셀 수 있는 집합이므로 오른쪽은 셀 수 있는 개수의 집합들의 합집합이다.
>
>**$A_{m,n}$이 $C[0,1]$에서 조밀한 곳이 없음을 증명하기**
>
>우선 $A_{m,n}$이 닫혀 있음을 증명해야하고, $f\in A_{m,n}$임을 보이면 된다.  
>$m, n$을 고정하고 $(f_k)$를 $A_{m,n}$의 수열이라 하고 $C[0,1]$에서 $f_k \to f$라 가정하자.
>
>문제 17(a): 수열 $(x_k)$가 반드시 수렴한다고 말할 수는 없지만 수렴하는 부분수열 $(x_{k_\ell})$이 존재함을 설명하라. 그 극한을
>$x=\lim_{\ell\to\infty}x_{k_\ell}
>$ 이라고 하자.  
>모든 $k$에 대하여 $x_k\in[0,1]$ 이다. 구간 $[0,1]$은 $\mathbb R$에서 닫히고 유계이므로 콤팩트하다. 따라서 $[0,1]$의 모든 수열은 $[0,1]$ 안의 점으로 수렴하는 부분수열을 갖는다. 그러므로 어떤 부분수열 $(x_{k_\ell})$과 어떤 $x\in[0,1]$가 존재하여
>$x_{k_\ell}\to x$ 이다.
>
>문제 17(b): $f_{k_\ell}(x_{k_\ell})\to f(x)$를 증명하라.  
>다음과 같이 차이를 나눈다. $\left| f_{k_\ell}(x_{k_\ell})-f(x) \right| \le \left| f_{k_\ell}(x_{k_\ell})-f(x_{k_\ell}) \right| + \left| f(x_{k_\ell})-f(x) \right|.$  
> - 첫 번째 항은 $\left| f_{k_\ell}(x_{k_\ell})-f(x_{k_\ell}) \right| \le \|f_{k_\ell}-f\|_\infty$ 이다. $f_{k_\ell}\to f$가 최소상계 노름에서 성립하므로 $\|f_{k_\ell}-f\|_\infty\to0$ 이다.  
> - 두 번째 항은 $x_{k_\ell}\to x$이고 $f$가 연속이므로 $|f(x_{k_\ell})-f(x)|\to 0$ 이다. 
> 
> 따라서 $f_{k_\ell}(x_{k_\ell})\to f(x)$ 이다.
>
>문제 17(c): $A_{m,n}$이 닫힌집합임을 증명하라.  
>$f\in A_{m,n}$임을 보이면 된다. 임의의 $t\in[0,1]$가 $0<|x-t|<\frac1m$ 을 만족한다고 하자. $x_{k_\ell}\to x$이므로 충분히 큰 $\ell$에 대하여 $0<|x_{k_\ell}-t|<\frac1m$ 이다. 여기서 $x\neq t$이므로 충분히 큰 $\ell$에 대하여 $x_{k_\ell}\neq t$도 성립한다. $x_{k_\ell}$은 $f_{k_\ell}\in A_{m,n}$의 조건을 만족시키는 점이므로 $\left| \frac{f_{k_\ell}(x_{k_\ell})-f_{k_\ell}(t)}{x_{k_\ell}-t}  \right| \le n$ 이다. 이제 $\ell\to\infty$로 보낸다. 문제 17(b)에 의해 $f_{k_\ell}(x_{k_\ell})\to f(x)$ 이고, 균등수렴은 점별수렴을 함의하므로 $f_{k_\ell}(t)\to f(t)$ 이다. 또한 $x_{k_\ell}-t\to x-t\neq0$ 이다. 따라서 $\left| \frac{f(x)-f(t)}{x-t} \right| \le n$ 이다.  
>이는 $0<|x-t|<\frac1m$ 인 모든 $t\in[0,1]$에 대해 성립한다. 따라서 $x$가 $A_{m,n}$의 정의에 필요한 점 역할을 하므로
>$f\in A_{m,n}$. 즉, $A_{m,n}$ 안에서 수렴하는 모든 수열의 극한이 다시 $A_{m,n}$에 속한다. 거리공간에서 이는 닫힌집합과 동치이므로
>$A_{m,n}$ 은 닫힌집합이다. 따라서 $\overline{A_{m,n}}=A_{m,n}$ 이다.
>
>$A_{m,n}$이 nowhere dense임을 보이는 전략
> 
>$A_{m,n}$이 nowhere dense임을 보이려면 $(A_{m,n})^\circ=\varnothing$ 임을 보이면 충분하다. 이를 위해 임의의 $f\in A_{m,n}$와 임의의 $\varepsilon>0$에 대하여
>$V_\varepsilon(f)=\{g\in C[0,1]:\|f-g\|_\infty<\varepsilon\}$
>가 $A_{m,n}$에 완전히 포함되지 않음을 보인다. 즉, $\|f-g\|_\infty<\varepsilon$ 이지만 $g\notin A_{m,n}$ 인 연속함수 $g$를 구성한다.
>
>문제 18(a): 다음을 만족하는 다각형 함수 $p\in C[0,1]$가 존재함을 보여라: $\|f-p\|_\infty<\frac{\varepsilon}{2}$
>
>다각형 함수란 그래프가 유한 개의 선분으로 이루어진 연속함수다. $f\in C[0,1]$이고 $[0,1]$은 콤팩트하므로 $f$는 균등연속이다. 따라서 어떤 $\delta>0$가 존재하여 $|x-y|<\delta$ 이면 $|f(x)-f(y)|<\frac{\varepsilon}{2}$ 이다.
>
>이제 분할 $0=t_0<t_1<\cdots<t_r=1$ 을 각 부분구간의 길이가 $\delta$보다 작도록 선택한다. $t_{j+1}-t_j<\delta.$ 각 분할점에서
>$p(t_j)=f(t_j)$ 로 놓고, 각 구간 $[t_j,t_{j+1}]$에서는 $p$를 두 점
>$(t_j,f(t_j)), \ (t_{j+1},f(t_{j+1}))$ 을 연결하는 일차함수로 정의한다. $x\in[t_j,t_{j+1}]$이면 어떤 $\lambda\in[0,1]$에 대하여
>$p(x)=\lambda f(t_j) + (1-\lambda)f(t_{j+1})$ 로 나타낼 수 있다.
>
>따라서 $|p(x)-f(x)| = \left| \lambda\bigl(f(t_j)-f(x)\bigr) +
>(1-\lambda)\bigl(f(t_{j+1})-f(x)\bigr) \right| 
>\le \lambda|f(t_j)-f(x)| + (1-\lambda)|f(t_{j+1})-f(x)|$
>
>$x,t_j,t_{j+1}$은 같은 작은 구간에 있으므로 $|f(t_j)-f(x)|<\frac{\varepsilon}{2}$ 이고 $|f(t_{j+1})-f(x)|<\frac{\varepsilon}{2}$
>이다. 따라서 $|p(x)-f(x)|<\frac{\varepsilon}{2}$  
>이는 모든 $x\in[0,1]$에 대하여 성립하므로 $\|f-p\|_\infty<\frac{\varepsilon}{2}$ 이다.
>
>
>문제 18(b): $h\in C[0,1]$가 $\|h\|_\infty=1$ 을 만족하는 임의의 함수라고 하자. 
>다음과 같이 정의한 함수 $g$가 $V_\varepsilon(f)$에 속함을 보여라.
>$g(x)=p(x)+\frac{\varepsilon}{2}h(x)$
>
>삼각부등식에 의해
>$\|g-f\|_\infty = \left\| p+\frac{\varepsilon}{2}h-f \right\|_\infty\ \le \|p-f\|_\infty + \frac{\varepsilon}{2} \|h\|_\infty$
>
>문제 18(a)와 $\|h\|_\infty=1$을 이용하면
>$|g-f|_\infty < \frac{\varepsilon}{2} + \frac{\varepsilon}{2} =\varepsilon$ 따라서 $g\in V_\varepsilon(f)$ 이다.
>
>문제 18(c) (b)의 함수 $g$가 $A_{m,n}$에 속하지 않도록 하는 다각형 함수 $h\in C[0,1]$를 찾아라. 단, $\|h\|_\infty=1$ 이어야 한다. 이를 바탕으로 정리 8.2.12의 증명을 마무리하라.
>
>1. $p$의 기울기 상계
>
>$p$는 유한개의 선분으로 이루어진 다각형 함수다. 따라서 각 선분의 기울기 절댓값 중 최댓값이 존재한다. 이를 $L$이라고 하자. 그러면 $p$가 하나의 선분으로 표현되는 구간에서 $\left|\frac{p(x)-p(t)}{x-t}\right|\le L$ 이다.
>
>2. 빠르게 진동하는 다각형 함수 $h$의 구성
>
>$p$의 모든 꼭짓점을 포함하는 충분히 세밀한 분할을 선택한다. $0=s_0<s_1<\cdots<s_q=1.$ 각 부분구간의 길이가 다음을 만족하도록 분할한다: 
>
>$$\frac{2}{s_{j+1}-s_j} > \frac{2(n+L)}{\varepsilon}$$
>
>동치로 $s_{j+1}-s_j < \frac{\varepsilon}{n+L}$ 가 되도록 선택하면 된다.
>
>이제 각 분할점에서 $h$의 값을 번갈아 다음과 같이 정의한다. $h(s_j)=(-1)^j.$ 각 구간 $[s_j,s_{j+1}]$에서는 $h$를 선형으로 정의한다. 그러면 $h$는 연속인 다각형 함수이고 모든 함수값이 $-1$과 $1$ 사이에 있으므로 $\|h\|_\infty=1$ 이다.
>
>또한 $[s_j,s_{j+1}]$에서 $h$의 기울기 절댓값은  $\left| \frac{h(s_{j+1})-h(s_j)} {s_{j+1}-s_j} \right|=\frac{2}{s_{j+1}-s_j}$ 이다. 
>분할의 선택에 의해 $|h$의 기울기$|> \frac{2(n+L)}{\varepsilon}$ 이다.
>
>3. $g$의 기울기
>
>다음과 같이 정의한다. $g=p+\frac{\varepsilon}{2}h.$
>각 구간 $[s_j,s_{j+1}]$에서 $p$와 $h$는 모두 일차함수다. 따라서 $g$도 일차함수다. $g$의 기울기를 $a_g$, $p$의 기울기를 $a_p$, $h$의 기울기를 $a_h$라고 하면 $a_g=a_p+\frac{\varepsilon}{2}a_h.$
>역삼각부등식에 의해 $|a_g| = \left| a_p+\frac{\varepsilon}{2}a_h
>\right|\ \ge \frac{\varepsilon}{2}|a_h|-|a_p|\ > \frac{\varepsilon}{2} \cdot \frac{2(n+L)}{\varepsilon} -L\ =n.$ 따라서 $g$의 모든 선분의 기울기 절댓값은 $n$보다 크다.
>
>4. $g\notin A_{m,n}$임을 증명
>
>임의의 $x\in[0,1]$를 선택하자. $x$가 분할구간 내부에 있으면 $x$와 같은 선분 안에서 $x$와 충분히 가까운 점 $t\neq x$를 선택할 수 있다.
>$x$가 분할점이라면 그 점에 연결된 왼쪽 또는 오른쪽 선분 안에서 충분히 가까운 $t\neq x$를 선택할 수 있다.
>
>어느 경우든 $0<|x-t|<\frac1m$ 가 되도록 $t$를 충분히 가깝게 선택할 수 있다. 또한 $x$와 $t$는 $g$의 같은 선분 위에 있으므로 차분몫은 해당 선분의 기울기와 같다. $\left| \frac{g(x)-g(t)}{x-t} \right|=|\text{해당 선분의 기울기}| > n.$
>따라서 모든 $x\in[0,1]$에 대하여 $A_{m,n}$의 기울기 조건을 위반하는 $t$가 존재한다. 즉, 어떤 $x$도 $0<|x-t|<\frac1m \Rightarrow \left| \frac{g(x)-g(t)}{x-t} \right|\le n$ 을 만족하지 않는다.
>
>따라서 $g\notin A_{m,n}$ 이다. 한편 문제 18(b)에 의해 $g\in V_\varepsilon(f)$ 이다. 그러므로 임의의 $f\in A_{m,n}$와 임의의 $\varepsilon>0$에 대하여 $V_\varepsilon(f)\not\subseteq A_{m,n}$
>이다. 따라서 $A_{m,n}$은 내부점을 갖지 않는다. $(A_{m,n})^\circ=\varnothing.$
>
>문제 17(c)에서 $A_{m,n}$이 닫힌집합임을 증명했으므로 $\overline{A_{m,n}}=A_{m,n}$ 이다.  
>따라서 $\left(\overline{A_{m,n}}\right)^\circ= (A_{m,n})^\circ \varnothing.$ 그러므로 $A_{m,n}\text{은 nowhere dense다.}$
>
>**증명의 마무리**
>
>문제 16에서 $D\subseteq\bigcup_{m,n\in\mathbb N}A_{m,n}$ 임을 보였다.  
>또한 문제 17과 문제 18에서 각 $A_{m,n}$이 닫힌 nowhere dense 집합임을 보였다. $\mathbb N\times\mathbb N$은 셀 수 있으므로
>$\bigcup_{m,n\in\mathbb N}A_{m,n}$ 은 셀 수 있는 개수의 nowhere dense 집합들의 합집합이다.
>
>더 엄밀하게 $D = \bigcup_{m,n\in\mathbb N} \left(D\cap A_{m,n}\right)$ 이고 $D\cap A_{m,n}\subseteq A_{m,n}$.
>
>nowhere dense 집합의 부분집합도 nowhere dense이므로 각 $D\cap A_{m,n}$은 nowhere dense다.
>
>따라서 $D$는 nowhere dense 집합들의 셀 수 있는 합집합이다.
>
>$\boxed{D\text{는 }C[0,1]\text{에서 제1범주 집합이다.}}$
>
>이로써 정리 8.2.12의 증명이 완성된다.


### 정리 3.5: 가산국소기저와 폐포의 거리표현
거리공간에서 다음이 성립한다:

1. 각 점 $x \in X$에 대해 $\{B_d(x, 1/n) : n \in \mathbb{N}\}$은 $x$의 가산국소기저이다.

2. 부분집합 $A \subset X$의 폐포는 다음과 같이 표현된다:

$$\overline{A} = \{x \in X : d(x, A) = 0\}$$

여기서 $d(x, A) = \inf\{d(x,a) : a \in A\}$는 점 $x$에서 집합 $A$까지의 거리이다.

3. 유한부분집합은 항상 닫힌집합이다.

### 정리 3.6: 서로소 폐집합의 분리
거리공간에서 $A, B$가 공집합이 아닌 서로소 닫힌집합이면, 다음을 만족하는 서로소 열린집합 $U, V$가 존재한다:

$$A \subset U, \quad B \subset V, \quad U \cap V = \emptyset$$

즉, 거리공간은 정규공간이다.

## 3.3 분리 성질과 거리화가능성

### 정의 3.4: 거리화가능 (metrizable)
위상공간 $(X, \mathscr{T})$가 거리화가능(metrizable)이라는 것은, 어떤 거리 $d$가 존재하여 $\mathscr{T} = \mathscr{T}_d$인 것을 의미한다.

### 정리 3.7: 거리공간은 $T_4$-공간
모든 거리공간은 $T_4$-공간(즉, 정규 $T_1$-공간)이다.

