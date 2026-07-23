# 1. 함수의 극한 *(Limits of Functions)*

## (1) 무한소와 극한 *(Infinitesimals and Limits)*
* 무한소는 *무한히 작은 수*를 일컫는 직관적인 개념으로, 고전적으로 미적분을 설명하기 위해 쓰였다.
* 실수체계에는 무한소가 존재하지 않으며, $\varepsilon$–$\delta$ 논법으로 정의한 **극한(limit)** 으로 미적분을 전개한다.
* 초실수체계에서는 무한소로 미적분 설명이 가능하다. (비표준 해석학)
    - 극한을 정의하고나서는 무한소는 있을 수 없는 수.  
다만, 비표준 해석학에서 무한소를 정의 해서 특수한 재밌는 연구를 수행함
    - 비표준 해석학에선 무한소가 있어서 극한 개념을 안쓴다
    - 초실수체는 실수랑 다르다
    
## (2) 극한의 정의 *(Definition of Limits)*
해석적 관점만 다룸. 기하적 특성은 위상수학에서 다룸.  
- 기하적 특성은 극한점(limit point)의 정의를 활용함
  - 함수 $f: A \to \mathbb R$에서 $A$의 극한점 $c$에 대해 임의의 $\epsilon$-근방 $V_{\epsilon}(c)$와 $A$의 교집합은 $c$가 아닌 다른 원소를 반드시 가져야한다. 
  - $A$가 닫힌 집합이 아니라면 $A$의 극한점은 집합 $A$의 원소가 아닐 수도 있다.

### Def 1. *(수렴과 극한, Limit of a function at a point)*
$f : D \to \mathbb{R},\ a \in D,\ L \in \mathbb{R}$ 라 하자

$$ \forall \epsilon>0,\ \exists \delta>0\ \text{s.t. $x\in D$}, 0 < |x-a| < \delta \Rightarrow |f(x)-L|<\epsilon $$

이 성립하면 $f$는 $x=a$에서 극한값 $L$로 수렴한다고 하고, 
$\lim_{x\to a} f(x)=L$ 로 표기한다.

- 함숫값 f(x) = a는 고려안함. 따라서 죄부등식에 등호가 없음
- 수렴안하는 경우엔 발산한다고 함

> 위상으로 표현한 함수의 극한
>
> 점 $c$가 $f: A \to \mathbb R$의 정의역의 극한점이라 하자. $L$의 모든 $\epsilon$-근방 $V_{\epsilon}(L)$에 대해 ($x \in A$이면서) $c$와 다른 $x \in V_{\delta}(c)$이면 항상 $f(x) \in V_{\epsilon}(L)$이 되게 하는 $c$의 $\delta$-근방 $V_{\delta}(c)$가 존재할 때, $\lim_{x\to c} f(x)=L$라 한다.

### 함수의 극한에 대한 수열 판정법
함수 $f: A \to \mathbb R$와 $A$의 극한점 $c$가 주어졌을 때 다음 두 명제는 동치다

1. $\lim_{x\to c} f(x)=L$
2. $x_n \neq c$ 이면서 $(x_n)\to c$인 수열 $(x_n) \subseteq A$에 대해 $f(x_n)\to L$이다.


### 따름정리 [함수의 극한에 대한 발산 판정법]

$A$에서 정의된 함수 $f$와 $A$의 극한점 $c$를 생각하자. $A$에 포함되는 두 수열 $(x_n)$과 $(y_n)$이 있어 $x_n \neq c, y_n \neq c$이고 다음을 만족한다고 하자.

$$\lim x_n = \lim y_n = c \text{이지만} \lim f(x_n) \neq \lim f(y_n)$$

그러면 극한값 $\lim_{x\to c} f(x)$ 는 존재하지 않는다.


#### 예제1
함수

$$
f(x)=
\begin{cases}
x, & x\neq 0 \\
1, & x=0
\end{cases}
$$

에 대하여 $\lim_{x\to 0} f(x)=0$ 임을 보인다.

**증명**  
: 임의의 $\varepsilon>0$ 를 잡는다.   
$\delta=\varepsilon$ 로 둔다. (*이 delta를 뭘로두느냐가 관건이다. 결론 식을 역으로 고려하여 쉬운 delta를 잡도록 하자. 즉 delta를 마지막에 확정지을 수 있다.*)  

그러면
$ 0<|x-0|<\delta $ 이면 $ 0<|x|<\varepsilon.$  
$x\neq 0$ 이므로 $|f(x)|=|x|.$  
따라서 $|f(x)-0|=|x|<\varepsilon.$  
즉, $0<|x-0|<\delta \Rightarrow |f(x)-0|<\varepsilon.$
그러므로

$$
\lim_{x\to 0} f(x)=0
$$

#### 예제 2 (Dirichlet function)
함수

$$
f(x)=
\begin{cases}
1, & x\in \mathbb{Q} \\
0, & x\in \mathbb{I}
\end{cases}
$$

에 대하여 $\lim_{x\to 0} f(x)$는 존재하지 않음을 보인다.

**증명**  
: 귀류법으로, 모든 epsilon에 대해 증명하는게 아닌 어떤 epsilon에 대해 모순임을 보이는 방식.  

극한이 존재한다고 가정하여 $\lim_{x\to 0} f(x)=L$라 하자.
그러면 $\forall \varepsilon>0,\ \exists \delta>0$ s.t. $0<|x|<\delta \Rightarrow |f(x)-L|<\varepsilon$ 이어야 한다.

임의의 $\delta>0$에 대해 유리수, 무리수의 조밀성으로 $0<|q|<\delta$인 $q\in\mathbb{Q}$와  $0<|s|<\delta$인 $s\in\mathbb{I}$를 잡을 수 있고, 이때 $f(q)=1,\ f(s)=0$이다.

* $L>0$인 경우: $\varepsilon=\frac{L}{2}$를 택하면 $|f(s)-L|=|0-L|=L\ge \varepsilon$ 이므로 조건 $|f(x)-L|<\varepsilon$를 만족하지 못한다.

* $L\le 0$인 경우: $\varepsilon=\frac{1-L}{2}$를 택하면 $|f(q)-L|=|1-L|=1-L\ge \varepsilon$ 이므로 조건 $|f(x)-L|<\varepsilon$를 만족하지 못한다.

따라서 어떤 $L$에 대해서도 $\lim_{x\to 0} f(x)=L$가 될 수 없으므로 $\lim_{x\to 0} f(x)$는 존재하지 않는다. $\blacksquare$

### Def 2. *(좌극한/우극한, Left/Right Limits)*
$f:D\to\mathbb{R},\ a\in D',\ L\in\mathbb{R}$ 라 하자.

(좌극한)
$\forall\varepsilon>0,\text{s.t. } \exists\delta>0,$
$0< a-x <\delta \Rightarrow |f(x)-L|<\varepsilon \\ \Leftrightarrow\lim_{x\to a^-} f(x)=L=f(a^+)$

(우극한)
$\forall\varepsilon>0,\text{s.t. } \exists\delta>0,$
$0< x-a <\delta \Rightarrow |f(x)-L|<\varepsilon \\ \Leftrightarrow\lim_{x\to a^+} f(x)=L=f(a^-)$

### Def 3. *(무한대로의 극한, Limits at infinity)*
극한값이 fix되는 정의. 0보다 크다는 표현이 없으므로, 언저리값이 아님. 한쪽 방향으로 그것도 N 이후 쭈욱... 살펴보는 정의.  

$a, L \in \mathbb{R}$ 이라 하자.

1. (x는 무한대로, 함수는 수렴)

$\forall \varepsilon > 0,\ \exists N\in\mathbb{R}\ \text{s.t.}\ x\geq N \Rightarrow |f(x)-L| < \varepsilon$ 일때 $\displaystyle \lim_{x\to\infty} f(x)=L$
  - ex: $\displaystyle \lim_{x\to\infty} 1/x=0$
  - 증명: 
  임의의 $\varepsilon>0$를 잡는다. 
  $N=\frac{2}{\varepsilon}$ 로 둔다. (즉, $\frac{2}{\varepsilon}\in\mathbb{R}$)  
  $x\ge N$ 이면 $\displaystyle x \ge \frac{2}{\varepsilon}$ 이고, 양변에 역수를 취하면 $\displaystyle \frac{1}{x} \le \frac{\varepsilon}{2}$  
  따라서 $\displaystyle \left|\frac{1}{x}-0\right| = \left|\frac{1}{x}\right| \le \frac{\varepsilon}{2} < \varepsilon$  
  즉, $x\ge N \Rightarrow \left|\frac{1}{x}-0\right|<\varepsilon$  
  그러므로 $\displaystyle \lim_{x\to\infty}\frac{1}{x}=0$ 이다.

2. (x는 수렴, 함수는 발산)

$\forall M > 0,\ \exists \delta > 0$ s.t. $0 < |x -a| < \delta \Rightarrow |f(x)|>M$
일때  
$\displaystyle \lim_{x\to a} f(x)=\infty$
  - ex: $\displaystyle \lim_{x\to 0}\frac{1}{x^2}=\infty$
  - 증명: 임의의 $M>0$를 잡는다. $\delta=\sqrt{\frac{1}{M}}>0$ 로 둔다.  
  그러면 $0<|x-0|<\delta$ 이면 $0<|x|<\delta$ 이고, 양변 제곱하여 $0<x^2<\delta^2$.  
  따라서 역수를 취하면 $\displaystyle \frac{1}{x^2}>\frac{1}{\delta^2}$.  
  그런데 $\delta^2=\frac{1}{M}$ 이므로 $\displaystyle \frac{1}{\delta^2}=M$.  
  따라서 $\displaystyle \left|\frac{1}{x^2}\right|=\frac{1}{x^2}>M$.  
  즉, $0<|x|<\delta \Rightarrow \left|\frac{1}{x^2}\right|>M$.  
  그러므로 $\displaystyle \lim_{x\to 0}\frac{1}{x^2}=\infty$ 이다.

3.(x도, 함수도 발산)

$\forall M > 0,\ \exists N \in \mathbb{R}$ s.t. $x \geq N \Rightarrow f(x)>M$ 일때  
$\displaystyle \lim_{x\to\infty} f(x)=\infty$

## (3) 극한의 연산 *(Limit Laws)*
$A, B \in \mathbb{R}$ 이고 $f,g : D\to \mathbb{R},\ a\in D$라 하자. $\lim_{x \to a} f(x)=A,\ \lim_{x \to a} g(x)=B$ 이면:

1. $\displaystyle \lim_{x \to a} (f(x)+g(x)) = A+B$
2. $\displaystyle \lim_{x \to a} (f(x)-g(x)) = A-B$
3. $\displaystyle \lim_{x \to a} (f(x)g(x)) = AB$
4. $\displaystyle \lim_{x \to a} \frac{f(x)}{g(x)} = \frac{A}{B}$ (단, $B\ne 0$)

### 증명
1. 임의의 $\varepsilon>0$를 잡는다.  
$\lim_{x\to a} f(x)=A$ 이므로 $\exists\delta_1>0$ s.t. $|x-a|<\delta_1 \Rightarrow |f(x)-A|<\frac{\varepsilon}{2}.$  
$\lim_{x\to a} g(x)=B$ 이므로 $\exists\delta_2>0$ s.t. $|x-a|<\delta_2 \Rightarrow |g(x)-B|<\frac{\varepsilon}{2}.$ 
$\delta=\min\{\delta_1,\delta_2\}$로 두면,

$$
|x-a|<\delta \Rightarrow |(f(x)+g(x))-(A+B)|
\\ \le |f(x)-A|+|g(x)-B|<\varepsilon
$$

3. 임의의 $\varepsilon>0$를 잡는다.  삼각부등식에 의해, 

$$ |f(x)g(x)-AB|=|f(x)g(x)-Ag(x)+Ag(x)-AB| \\ \leq |fg-Ag| + |Ag-AB|= |g||f-A| + |A||g-B| $$

$\lim_{x\to a} g(x)=B$ 이므로 $\exists\delta_0>0$ s.t. $|x-a|<\delta_0 \Rightarrow |g(x)-B|<1$  
따라서 $|g(x)|\le |B|+1.$  

이제 임의의 $\varepsilon>0$에 대해  

$$|f(x)-A|<\frac{\varepsilon}{2(|B|+1)},\quad
|g(x)-B|<\frac{\varepsilon}{2(|A|+1)}$$

가 되도록 $\delta_1,\delta_2$를 잡는다.  
$\delta=\min\{\delta_0,\delta_1,\delta_2\}$ 이면  

$$|f(x)g(x)-AB| < |g||f-A| + |A||g-B| < \varepsilon/2 + \varepsilon/2$$

$$\therefore |f(x)g(x)-AB|<\varepsilon$$

4.
  - 임의의 $\varepsilon>0$를 잡는다.  

$$
\left|\frac{f(x)}{g(x)}-\frac{A}{B}\right|
=\left|\frac{Bf(x)-Ag(x)}{Bg(x)}\right|
\le \frac{|B||f(x)-A|+|A||g(x)-B|}{|B||g(x)|}
$$

$\lim_{x\to a} g(x)=B\ne0$ 이므로 $\exists\delta_0>0$ s.t
$$
|x-a|<\delta_0 \Rightarrow |g(x)|>\frac{|B|}{2}
$$

따라서

$$
\left|\frac{f(x)}{g(x)}-\frac{A}{B}\right|
\le \frac{2}{|B|^2}\Big(|B||f(x)-A|+|A||g(x)-B|\Big)
$$

이제 $|B||f(x)-A|<\frac{|B|^2\varepsilon}{4}$ 가 되도록 $\delta_1>0$ 을 택하고,  
$|A||g(x)-B|<\frac{|B|^2\varepsilon}{4}$ 가 되도록 $\delta_2>0$ 을 택한다.  

$\delta=\min\{\delta_0,\delta_1,\delta_2\}$ 로 두면 $|x-a|<\delta$ 일 때

$$
\left|\frac{f(x)}{g(x)}-\frac{A}{B}\right|
\le \frac{2}{|B|^2}\left(\frac{|B|^2\varepsilon}{4}+\frac{|B|^2\varepsilon}{4}\right)=\varepsilon
$$

즉,

$$
\lim_{x\to a}\frac{f(x)}{g(x)}=\frac{A}{B}
$$

## (4) 주요 정리
### Thm 1. (극한의 유일성, *Uniqueness of Limit*)
$f:D\to\mathbb{R},\ a\in D$일 때, $\lim_{x\to a} f(x)$가 수렴하면 그 극한값은 유일하다.

**증명**  
$\lim_{x\to a} f(x)=L$이고 동시에 $\lim_{x\to a} f(x)=M$이라고 가정한다.

임의의 $\varepsilon>0$를 잡는다.
극한의 정의에 의해 $\exists\delta_1>0$ s.t.
$|x-a|<\delta_1 \Rightarrow |f(x)-L|<\varepsilon/2$.

또한 $\exists\delta_2>0$ s.t.
$|x-a|<\delta_2 \Rightarrow |f(x)-M|<\varepsilon/2$.

$\delta=\min\{\delta_1,\delta_2\}$로 두면 $|x-a|<\delta$에서

$$|L-M| = |L-f(x)+f(x)-M| \le |f(x)-L|+|f(x)-M| < \varepsilon$$

임의의 $\varepsilon>0$에 대해 $|L-M|<\varepsilon$이므로 $L=M$이다.

### Thm 2. (샌드위치 정리, *Squeeze Theorem*)
$\forall x\in D,\ f(x)\le g(x)\le h(x)$이고 $L\in\mathbb{R}$에 대하여
$\lim_{x\to a} f(x)=\lim_{x\to a} h(x)=L$이면 $\lim_{x\to a} g(x)=L$이다.

>**증명**  
>임의의 $\varepsilon>0$를 잡는다.
>
>$\lim_{x\to a} f(x)=L$이므로 $\exists\delta_1>0$ s.t.
>$|x-a|<\delta_1 \Rightarrow |f(x)-L|<\varepsilon$, 즉
>$L-\varepsilon<f(x)<L+\varepsilon$.
>
>$\lim_{x\to a} h(x)=L$이므로 $\exists\delta_2>0$ s.t.
>$|x-a|<\delta_2 \Rightarrow |h(x)-L|<\varepsilon$, 즉
>$L-\varepsilon<h(x)<L+\varepsilon$.
>
>$\delta=\min\{\delta_1,\delta_2\}$로 두면 $|x-a|<\delta$에서
>
>$$
>L-\varepsilon<f(x)\le g(x)\le h(x)<L+\varepsilon
>$$
>
>따라서 $|g(x)-L|<\varepsilon$가 되어 $\lim_{x\to a} g(x)=L$이다.

**샌드위치 정리 예시**  
$\displaystyle \lim_{x\to 0}\frac{\sin x}{x}=1$ 증명  

$x\neq 0$라 하자. 삼각함수의 기본 부등식에 의해 $\sin x < x < \tan x \quad (0<x<\tfrac{\pi}{2})$
가 성립한다. 양변을 $x$로 나누면 $\frac{\sin x}{x} < 1 < \frac{\tan x}{x}.$  
여기서 $\tan x=\frac{\sin x}{\cos x}$ 이므로
$1 < \frac{1}{\cos x}$

따라서

$$
\cos x < \frac{\sin x}{x} < 1 \quad (0<x<\tfrac{\pi}{2})
$$

$x<0$인 경우에도 같은 부등식이 성립하므로, 충분히 작은 $x\neq 0$에 대해
$\cos x \le \frac{\sin x}{x} \le 1$이다.

이제
$\lim_{x\to 0}\cos x = 1, \quad \lim_{x\to 0} 1 = 1$
이므로 샌드위치 정리에 의해
$\lim_{x\to 0}\frac{\sin x}{x}=1$


# 2. 함수의 연속 *(Continuity of Functions)*
실변수 함수에 대한 엄밀한 이론을 확립하는 여정에 매우 중요한 개념이다. '구멍', '끊어지지 않는' 같은 직관적인 표현에서 벗어나 엄밀한 연속을 정의해보자.

## (1) 연속의 정의 *(Definition of Continuity)*
### Def 1. *(점 연속, Continuity at a point)*

$f:D\to\mathbb{R},\ a\in D$라 하자.

$$\forall \varepsilon>0,\ \exists\delta>0 \text{ s.t. $(x\in D$)},\ |x-a|<\delta \Rightarrow |f(x)-f(a)|<\varepsilon$$

이 성립하면 $f$는 $x=a$에서 연속이라 한다.  
- 극한의 정의와 비슷함. 차이점?
  - 극한값 L 대신 f(a). $|f(x)-f(a)|$ 조건이 극한 정의와 달리, 함숫값을 가리킴
  - $a\in D$: $f(a)$가 정의됨
  - $0 < |x-a|$ 조건 없음
- 즉, $f$는 $a$에서 연속 $\iff \lim_{x\to a} f(x)=f(a)$
* 위 정의를 만족하는 모든 대상이 연속이라는 개념을 만족한다.
  - 이산적인 연속도 정의 가능. 아래 예시 참고

#### 예시: 이산집합은 연속
 D = {0, 1, 2, 3}일때 f(x) = -x + 3이면 f는 x=2에서 연속 증명  

>**증명**  
>$f(2)=-2+3=1$ 이다. 임의의 $\varepsilon>0$를 잡고, 
>$\delta=\frac{1}{2}$로 둔다.  
>그러면 $x\in D$에 대해
>$|x-2|<\delta$ 를 만족하는 경우는 오직 $x=2$ 뿐이다.  
>따라서 $|x-2|<\delta$ 이면 반드시 $x=2$이고,
>$|f(x)-f(2)| = |1-1| = 0 < \varepsilon$  
>즉,
>
>$$
>\forall \varepsilon>0,\ \exists\delta>0\ \text{s.t.}\
>|x-2|<\delta \Rightarrow |f(x)-f(2)|<\varepsilon
>$$
>
>그러므로 $f$는 $x=2$에서 연속이다.

### Def 2. *(우연속, 좌연속 — Right/Left continuity)*
$f: D\to \mathbb{R}, a\in D$라 하자.  

(우연속)  
$$\forall\epsilon>0, \exists\delta>0 \text{ s.t. $(x\in D$)},\   0\leq x-a<\delta \Rightarrow |f(x)-f(a)|<\varepsilon$$

이 성립하면 $f$는 $x=a$에서 우연속이라 한다.

(좌연속)  
$$\forall\epsilon>0, \exists\delta>0 \text{ s.t. $(x\in D$)},\   
0\leq a-x<\delta \Rightarrow |f(x)-f(a)|<\varepsilon$$

이 성립하면 $f$는 $x=a$에서 좌연속이라 한다.

### Def 3. *(연속함수, Continuous function)*
$f: D\to \mathbb{R}, X \subseteq D$라 하자.  

- $f$가 $X$의 모든 점에서 연속이면 $f$는 $X$에서 연속함수라 한다.

- $f$가 $D$의 모든 점에서 연속이면 $f$는 연속함수라 한다.

- 증명방법
  - x는 a (a는 정의역 원소) 에서 연속임을 보인다
  - a가 원하는 집합의 임의의 점이라고 서술

- 예시: $f(x)=x^2$이 $x>0$에서 연속 증명
>
>임의의 $a\in X$를 잡는다. 즉, $a>0$이다.
>임의의 $\varepsilon>0$를 잡는다.
>Then $\forall x\in(0, \infty)$, with $|x-a|<\delta$
>
>$$
>|f(x)-f(a)| =|x^2-a^2| =|x-a||x+a|
>$$
>
>(만약 delta = 1이었으면, 즉 $|x-a|<1$이라고 추가로 가정하면)  
>$|x-a|<1$ 이면 $a-1 < x < a+1$ 이므로 $|x+a| < 2a+1$
>
>따라서 
>
>$$ |f(x)-f(a)| < (2a+1)|x-a|$$
>
>이제 $\delta = \min\{1, \frac{\varepsilon}{2a+1}\}$
>로 두면, $|x-a|<\delta\leq \frac{\varepsilon}{2a+1}$
>
>$$\therefore |f(x)-f(a)|<\varepsilon$$
>
>즉,
>
>$$
>\forall \varepsilon>0,\ \exists\delta>0\ \text{s.t.}\ |x-a|<\delta \Rightarrow |f(x)-f(a)|<\varepsilon
>$$
>
>따라서 $f$는 임의의 $a>0$에서 연속이다.  
>$a$가 $X$의 임의의 원소였으므로, $f$는 $x>0$에서 연속이다.


- (참고) $f$가 정의역 $D$의 모든 점에서 연속이면 $f$는 **집합 $A$에서 연속** 이라 한다.

### (추가) 연속을 정의하는 동치 표현들
함수 $f:A \to \mathbb{R}$와 점 $a \in A$에 대해 다음 조건 1~3중 하나를 만족하는 것과 함수 $f$가 $c$에서 연속인 것은 동치다.

1. 임의의 $\epsilon>0$에 대해 ($x \in A$이고) $|x - c| < \delta$면 $|f(x)-f(c)| < \epsilon$이 되게 하는 $\delta > 0$이 존재한다.

2. 임의의 $V_{\epsilon}(f(c))$에 대해 ($x \in A$이고) $x \in V_{\delta}(c)$면 $f(x) \in V_{\epsilon}(f(c))$가 되게 하는 $V_{\delta}(c)$가 존재한다.

3. $x \in A$이면서 $(x_n) \to c$인 임의의 $(x_n)$에 대해 $f(x_n) \ \to f(c)$ 이다.

$c$가 $A$의 극한점이면 조건 1,2,3은 4와 동치다:

4. $\lim_{x\to c}f(x) = f(c)$

- 1,2,4 명제는 거의 같은 표현이고, 명제3은 질적으로 다르다. 3번은 함수가 어떤 점에서 연속이 아님을 보이는 데 유용하다.

### (추가) [따름정리: 불연속성 판정법]
함수 $f: A\to \mathbb{R}$와 $A$의 극한점 $c \in A$에 대하여 $(x_n) \to c$ 이지만 $f(x_n)$은 $f(c)$로 수렴하지 않는 수열 $(x_n) \subseteq A$이 존재하면 $f$는 $c$에서 연속이 아니다.


### Def 4. *(불연속의 종류, Types of discontinuities)*
연속이 아니면 불연속.  
즉 함숫값 정의가 안되면 무조건 불연속.  
아래는 함숫값 정의되지만 불연속인 경우들.  

$f: D\to \mathbb{R}, a \in D$라 하자. (즉 함숫값이 존재할 때)  
(제 1종 불연속점)  
  - 제거가능(removable discontinuity) 불연속점 
  
$$x=a \quad \text{s.t. }\lim_{x\to a^+}f(x)=\lim_{x\to a^-}f(x)\neq f(a)$$

이 값만 대체하면 연속이 되므로

  - 비약 불연속점 (jump discontinuity) 

$$x=a \quad \text{s.t. }\lim_{x\to a^+}f(x)\neq\lim_{x\to a^-}f(x)$$

좌극한, 우극한 이지러진 경우

(제 2종 불연속점)  
  - $\displaystyle \lim_{x\to a^+}f(x)$와$\displaystyle \lim_{x\to a^-}f(x)$중에 적어도 하나가 존재하지 않는다.
    - 진동하는 경우 보통은 극한이 없다

#### 예제: 최대정수함수의 연속성

최대정수함수 $h(x)=\lfloor x\rfloor$는 각 $x\in\mathbb{R}$에 대하여 $n\le x$를 만족하는 최대의 정수 $n\in\mathbb{Z}$를 함수값으로 갖는다.  
이 함수는 계단 모양의 그래프를 가지며, 정수점에서 불연속일 것으로 예상된다. 이를 수열판정법과 $\varepsilon$-$\delta$ 정의를 이용하여 엄밀하게 증명하라.

>**1. 정수에서의 불연속성**
>
>임의의 정수 $m\in\mathbb{Z}$를 잡는다.  
>수열 $x_n=m-\frac{1}{n}$을 정의하면 $x_n\to m$이다.  
>한편 $m-1<x_n<m$이므로 최대정수함수의 정의에 의해 $h(x_n)=m-1$이다.  
>따라서 $h(x_n)\to m-1$.
>
>그런데 $h(m)=m$이므로 $h(x_n)\to m-1\neq m=h(m)$
>
>따름정리 4.3.3(수열판정법)에 의해 $h$는 모든 $m\in\mathbb{Z}$에서 불연속이다. ✓
>
>**2. 정수가 아닌 점에서의 연속성**
>
>이제 $c\notin\mathbb{Z}$에서 연속임을 보인다.
>
>연속의 정의에 따라 임의의 $\varepsilon>0$에 대하여 $x\in V_\delta(c)$이면 $h(x)\in V_\varepsilon(h(c))$가 되도록 하는 $\delta$-근방 $V_\delta(c)$를 찾아야 한다.
>
>**$\delta$의 선택:**
>
>$c\notin\mathbb{Z}$이므로 어떤 정수 $n\in\mathbb{Z}$에 대하여 $n<c<n+1$이 성립한다.  
>이때 $\delta=\min\{\,c-n,\,(n+1)-c\,\}$로 잡는다. $\delta$를 이렇게 선택하면 $x\in V_\delta(c)$인 모든 $x$는 여전히 $n<x<n+1$을 만족한다.  
>따라서 최대정수함수의 정의에 의해 $h(x)=n=h(c)$.  
>즉, $x\in V_\delta(c)$이면 $h(x)=h(c)$가 성립한다.  
>
>따라서 $h(x)\in V_\varepsilon(h(c))$도 자동으로 성립한다.  
>결국 $h$는 $c\notin\mathbb{Z}$에서 연속이다. ✓
>
>**3. 설명**
>
>이 증명의 특징은 $\delta$의 선택이 $\varepsilon$에 전혀 의존하지 않는다는 점이다.  
>일반적인 연속성 증명에서는 $\varepsilon$을 더 작게 선택하면 그에 맞추어 $\delta$도 더 작게 선택해야 하는 경우가 많다.
>
>그러나 최대정수함수는 정수가 아닌 점 근처에서는 함수값이 일정하므로 $h(x)=h(c)$가 되어 $|h(x)-h(c)|=0$이 항상 성립한다.
>
>따라서 어떤 $\varepsilon>0$에 대해서도 동일한 $\delta$를 사용할 수 있다.
>

## (2) 균등연속 *(Uniform Continuity, 고른연속)*
- 연속의 정의상, 연속점 c에 따라 $\delta$가 달라지는 경우가 있다.
- c에 무관하게 $\delta$를 정할 수 있는 더 강한 연속 조건을 균등연속이라 한다.

### 정의
$f: D\to \mathbb{R}$이라 하자.  

$$\forall \varepsilon>0,\ \exists \delta>0,\ (x,y\in D), |x-y|<\delta \Rightarrow |f(x)-f(y)|<\varepsilon$$

이 성립하면 $f$는 $D$에서 균등연속이다.

- f가 균등연속이면 연속.

- 예: $f(x) = x^2$이 $[-2, 2)$에서 균등연속 
>증명:  
>정의역을 $D=[-2,2)$라 하자.  
>임의의 $\varepsilon>0$를 잡는다.  
>임의의 $x,y\in D$에 대해
>$|f(x)-f(y)| =|x^2-y^2| =|x-y||x+y|$
>
>$x,y\in[-2,2)$ 이므로
>$|x|\le 2, |y|\le 2$
>이고, 따라서 $|x+y|\le |x|+|y|\le 4$
>
>그러므로 $|f(x)-f(y)|\le 4|x-y|$
>
>이제 $\delta=\frac{\varepsilon}{4}$
>로 두면, $|x-y|<\delta$일 때
>$|f(x)-f(y)|<4\cdot\frac{\varepsilon}{4}=\varepsilon$
>
>즉,
>$\forall \varepsilon>0,\ \exists\delta>0\ \text{s.t.}\
>|x-y|<\delta \Rightarrow |f(x)-f(y)|<\varepsilon
>$
>가 모든 $x,y\in[-2,2)$에 대해 성립한다.
>
>따라서 $f(x)=x^2$는 $[-2,2)$에서 균등연속이다.

### (추가) 고른연속에 대한 수열 판정법
함수 $f: A \to \mathbb R$가 고른연속이 아니기 위한 필요충분조건은 다음을 만족하는 어떤 $\epsilon_0 >0$와 $A$의 두 수열 $(x_n)$과 $(y_n)$이 존재하는 것이다

$$
|x_n - y_n| \to 0 \text{이지만, } |f(x_n) - f(y_n)| > \epsilon_0
$$

- 연속성은 한 점에서 정의되지만, 고른연속성은 항상 특정 정의역에서 정의된다

> 증명  
> 균등연속의 정의를 부정하면 다음과 같다: 함수 f가 A에서 고른연속이 아니라는 것과 어떤 $\epsilon_0 >0$이 존재하여 모든 $\delta > 0$에 대해 $|x_1-y_1| < \delta$지만 $|f(x)-f(y)|>\epsilon_0$인 두 점 x와 y를 찾을 수 있다는 것은 동치다.  
>
> 따라서 $\delta_1=1$로 두면 $|x_1-y_1|<1$ 이지만 $|f(x_1)-f(y_1)| \geq \epsilon_0$가 되는 두 점 $x_1$과 $y_1$이 존재한다.  
>
> 비슷하게 $n \in \mathbb N$일때 $\delta_n = 1/n$로 두면 $|x_n - y_n| < 1/n$이지만 $|f(x_n)-f(y_n)| \geq \epsilon_0$가 되는 두 점 $x_n$, $y_n$이 존재한다.  
>이렇게 만들어진 수열 $(x_n)$, $(y_n)$이 정리에서 말하는 수열이다.
>
> 역으로 $\epsilon_0$와 두 수열 $(x_n)$, $(y_n)$이 제시된 바와 같이 존재하면 $\epsilon_0$에 대한 적절한 $\delta > 0$이 존재하지 않음을 바로 보일 수 있다.

### 정리: 콤팩트 집합에서 고른연속
콤팩트 집합 $K$에서 연속인 함수는 $K$에서 고른 연속이다.

> 증명
> 귀류법 사용: $f$가 $K$에서 고른연속이 아니라면, 어떤 $\epsilon_0 >0$에 대해 다음을 만족하는 두 수열 $(x_n), (y_n)$이 $K$에 존재한다.
>
> $\text{lim} |x_n-y_n| = 0$ 이지만 $|f(x_n)-f(y_n)| \geq \epsilon_0$
>
> $K$는 콤팩트집합이므로 수열 xn은 수렴하는 부분수열 xnk를 가지며 이때 x = lim xnk도 K에 있다. K는 콤팩트집합이므로 (yn)의 수렴하는 부분수열을 만들 수 있지만, (y_n)의 항 중에서 수렴하는 부분수열 (x_nk)의 항에 대응하는 항으로 이루어진 부분수열 (y_nk)를 생각해보자.  
> 극한과 사칙연산 성질에 의해 $lim (y_{nk}) = lim ((y_{nk}-x_{nk})+x_{nk}) = 0 + x$
>
> 따라서 $(x_{nk})$와 $(y_{nk})$ 모두 $x \in K$로 수렴한다. f는 x에서 연속이라 가정했으므로  $lim f(x_{nk}) = f(x), lim f(y_{nk}) = f(x)$이고 따라서 $lim (f(x_{nk})-f(y_{nk})) = 0$ 이다.  
> 그런데 (x_n), (y_n)은 모든 $n \in \mathbb N$에 대해 다음 부등식을 만족하도록 택했으므로 가정에 모순이다: $|f(x_n)-f(y_n)| \geq \epsilon_0$
>
> 따라서 f는 K에서 고른연속이다.

## (3) 연속함수의 연산 *(Operations of continuous functions)*
$a\in D, f,g: D \to \mathbb{R}$가 $x=a$ 에서 연속일때, 다음이 성립한다:
1. 임의의 $k \in \mathbb R$에 대해 $kf$는 $x=a$ 에서 연속
2. $f-g$는 $x=a$ 에서  연속
3. $fg$는 $x=a$ 에서  연속
4. $g(a)\ne 0$ 이면 $\frac{f}{g}$는 $x=a$ 에서  연속

## (4) 주요 정리
### Thm 1. 최대 최소 정리 *(Extreme Value Theorem)*
$f$가 $[a,b]$에서 연속이면 최대·최소 존재 $\exists a_0, b_0 \in [a, b]$ s.t. $\forall x\in [a, b], f(a_0)\leq f(x) \leq f(b_0)$

### Thm 2. 사잇값(중간값) 정리 *(Intermediate Value Theorem)*
$f$가 $[a,b]$에서 연속이고 $f(a)<f(b)$ 이면
$\exists c\in(a,b)$ s.t. $f(c)=p$, $f(a)<p<f(b)$.  
$f(b)<f(a)$이면 $f(b)<p<f(a)$.

- 이 정리는 근의 존재성 증명에 사용된다.
- (TMI) 오일러, 가우스를 포함한 18세기 수학자들은 이 정리를 증명없이 자유롭게 사용했다. 1817년 볼차노가 최초로 현대적 연속성 정의를 포함하여 논문을 쓰기 전까지 아무도 이 정리를 해석학적으로 증명하지 않았다. 볼차노와 동시대 수학자들이 수학의 기초를 확고히 하는것은, 단순히 누락된 증명을 채우는 단순한 문제가 아니었다. 관련 개념에 대한 철저히 합의된 이해가 있어야 수학의 최전방에서 격렬한 전투를 벌일 수 있다. 즉, 본질에 대한 이해가 '증명가능할 만큼' 충분히 성숙해져야 하는 것이다.
- 중간값정리는 본질적으로 1차원에서만 성립하지만, 아래 연결집합 관련 내용은 더 높은 차원에서도 성립한다. 즉 고차원 버전에서 연속성과 연결성 정의를 적용할 수 있게 해준다.

#### 정리: 연결집합의 보존

함수 $f:G \to \mathbb R$이 연속이라 하자. 집합 $E \subseteq G$가 연결집합이면 $f(E)$도 연결집합이다.

-중간값 정리를 연속함수가 연결집합(connected set)을 연결집합으로 옮긴다는 사실의 따름정리로 볼 수 있다.

> 증명  
> $f(E)$가 연결집합이 아니라고 가정하자. 그러면 $f(E)$의 분리 $A,B$가 존재하여
> $f(E)=A\cup B, \quad A\cap B=\varnothing, \quad A,B\neq\varnothing$ 이고 $\overline A^{\,f(E)}\cap B=\varnothing, \quad A\cap\overline B^{\,f(E)}=\varnothing$ 이다.  
> ($\bar A$는 A의 폐포(closure))
>
> 다음과 같이 둔다: $C=\{x\in E:f(x)\in A\}, \quad D=\{x\in E:f(x)\in B\}$.
> 그러면 $C,D\neq\varnothing$이고
> $C\cap D=\varnothing, \quad E=C\cup D$이다.
>
> $E$가 연결집합이므로 $C,D$가 $E$의 분리가 될 수 없다. 따라서 둘 중 한쪽의 점들로 이루어진 수열이 다른 쪽의 점으로 수렴한다. 일반성을 잃지 않고 $x_n\in C, \quad x_n\to x\in D$ 라고 하자.
>
> $f$가 연속이므로 $f(x_n)\to f(x)$.
> 그런데 $f(x_n)\in A$이고 $f(x)\in B$이므로
> $f(x)\in\overline A^{\,f(E)}\cap B$이다. 이는 $A,B$가 분리라는 사실에 모순이다.  
>따라서 $f(E)$는 연결집합이다.
>
> ### 더 간단한 표준 증명
> 수열을 사용하지 않으면 훨씬 간결하다.  
> $f(E)$가 연결되지 않았다고 가정하고, $A,B$를 $f(E)$의 분리라고 하자. 제한함수 $f|_E:E\to f(E)$ 도 연속이다. 따라서 상대적으로 열린 집합 $A,B\subseteq f(E)$의 원상 $C=(f|_E)^{-1}(A), \quad D=(f|_E)^{-1}(B)$ 는 $E$에서 상대적으로 열린 집합이다. 또한 $C,D\neq\varnothing, \quad C\cap D=\varnothing, \quad E=C\cup D$.  
> 그러므로 $C,D$는 $E$의 분리가 된다. 이는 $E$가 연결집합이라는 가정에 모순이다. 따라서 $f(E)$는 연결집합이다.
>

#### 연결집합의 보존으로 중간값 정리 증명
$f(a)<p<f(b)$인 임의의 $p$를 잡자.

닫힌구간 $[a,b]$는 연결집합이고 $f$는 $[a,b]$에서 연속이므로, 연결집합의 보존 정리에 따라 $f([a,b])$ 도 연결집합이다.

또한 $f(a),f(b)\in f([a,b])$ 이다. $\mathbb R$의 연결집합은 구간이므로, $f(a)$와 $f(b)$ 사이의 모든 값을 포함한다. 따라서 $f(a)<p<f(b)$ 인 $p$도 $f([a,b])$에 포함된다. 즉, $p\in f([a,b])$.

함수의 상의 정의에 따라 어떤 $c\in[a,b]$가 존재하여 $f(c)=p$ 이다. 그런데 $f(a)<p<f(b)$ 이므로 $f(c)=f(a)$ 또는 $f(c)=f(b)$일 수 없다. 따라서 $c\neq a,b$이고, $c\in(a,b)$ 이다. 그러므로 $\exists c\in(a,b)\text{ s.t. }f(c)=p$.

$f(b)<f(a)$인 경우에도 $f(b)<p<f(a)$인 $p$를 잡아 같은 논리를 적용하면 된다. □

#### 중간값 정리 다른 버전
사잇값 정리의 영점 형태부터 증명한다.  

**$f:[a,b]\to\mathbb R$가 연속이고 $f(a)<0<f(b)$ 이면 어떤 $c\in(a,b)$가 존재하여 $f(c)=0$이다.**

일반적인 사잇값 정리는 $g(x)=f(x)-p$에 이 결과를 적용하면 얻어진다.

#### (a) 완비성 공리를 이용한 증명

다음 집합을 정의한다: $K=\{x\in[a,b]:f(x)\le0\}$.

$f(a)<0$이므로 $a\in K$이고, 따라서 $K\neq\varnothing$이다.  
또한 $K\subseteq[a,b]$이므로 $b$는 $K$의 상한이다. 따라서 완비성 공리에 의해 $K$의 최소상한이 존재한다.

$c=\sup K$라고 두자. 이제 $f(c)=0$임을 보이면 된다.

**경우 1. $f(c)>0$**

$\varepsilon=\frac{f(c)}2>0$.
$f$가 $c$에서 연속이므로 어떤 $\delta>0$가 존재하여

$|x-c|<\delta$이면 $|f(x)-f(c)|<\frac{f(c)}2$.

따라서 $f(x)>f(c)-\frac{f(c)}2=\frac{f(c)}2>0$.

한편 $c=\sup K$이므로 $K$에는 $c$의 왼쪽에서 임의로 가까운 점이 존재한다. 따라서 어떤 $x\in K$가 존재하여 $c-\delta<x\le c$.

그런데 $|x-c|<\delta$이므로 $f(x)>0$이고, 이는 $x\in K$, 즉 $f(x)\le0$라는 사실에 모순이다.  

따라서 $f(c)>0$는 불가능하다.

**경우 2. $f(c)<0$**

$\varepsilon=-\frac{f(c)}2>0$.
연속성에 의해 어떤 $\delta>0$가 존재하여

$|x-c|<\delta$이면 $|f(x)-f(c)|< -\frac{f(c)}2$.

따라서 $f(x)<f(c)+\left(-\frac{f(c)}2\right)=\frac{f(c)}2<0$.

또한 $f(b)>0$이므로 $c\neq b$, 즉 $c<b$이다. 

그러므로 $c<x<\min\{c+\delta,b\}$ 를 만족하는 $x$를 선택할 수 있다.
그러면 $|x-c|<\delta$이므로 $f(x)<0$이고,  
따라서 $x\in K$이다. 하지만 $x>c$이므로 이는 $c$가 $K$의 상한이라는 사실에 모순이다. 

따라서 $f(c)<0$도 불가능하다.

두 경우가 모두 불가능하므로 남는 가능성은 $f(c)=0$.

또한 $f(a)<0<f(b)$이므로 $c\neq a,b$이다. 

따라서 $c\in(a,b),\ f(c)=0$.

#### (b) 축소구간 성질을 이용한 증명

처음 구간을 $I_0=[a_0,b_0]=[a,b]$ 로 둔다. 그러면 $f(a_0)<0<f(b_0)$.

$I_0$의 중점을 $z_0=\frac{a_0+b_0}2$ 라고 하자.

- $f(z_0)=0$이면 증명이 끝난다.
- $f(z_0)>0$이면 $I_1=[a_0,z_0]$로 둔다.
- $f(z_0)<0$이면 $I_1=[z_0,b_0]$로 둔다.

그러면 어느 경우든 $I_1=[a_1,b_1]$은 $f(a_1)<0<f(b_1)$ 을 만족하고, 길이는 $b_1-a_1=\frac{b-a}2$ 이다.

이 과정을 귀납적으로 반복한다. $I_n=[a_n,b_n]$이 정해졌다고 하자. 중점을 $z_n=\frac{a_n+b_n}2$ 라고 둔다.

$f(z_n)=0$이면 증명이 끝난다. 그렇지 않으면

$$I_{n+1}=\begin{cases}
[a_n,z_n],& f(z_n)>0,\\[2mm]
[z_n,b_n],& f(z_n)<0
\end{cases}$$

로 정의한다.

그러면 모든 $n$에 대하여 $I_{n+1}\subseteq I_n$, $f(a_n)<0<f(b_n)$ 이고, $b_n-a_n=\frac{b-a}{2^n}$. 

따라서 $b_n-a_n\to0$.

축소구간 성질에 의해 모든 $I_n$에 공통으로 포함되는 점 $c$가 존재한다.

$c\in\bigcap_{n=0}^{\infty}I_n$.
또한 구간의 길이가 0으로 수렴하므로 이러한 $c$는 유일하다.

모든 $n$에 대하여 $a_n\le c\le b_n$ 이므로

$0\le c-a_n\le b_n-a_n=\frac{b-a}{2^n}$,

$0\le b_n-c\le b_n-a_n=\frac{b-a}{2^n}$.

샌드위치 정리에 의해 $a_n\to c$, $b_n\to c$.

$f$가 $c$에서 연속이므로 $f(a_n)\to f(c)$, $f(b_n)\to f(c)$.

그런데 모든 $n$에 대하여 $f(a_n)<0<f(b_n)$ 이므로 극한을 취하면 
$f(c)\le0$, $f(c)\ge0$.

따라서 $f(c)=0$.  
또한 $f(a)<0<f(b)$이므로 $c\neq a,b$이다.  
따라서 $c\in(a,b),\ f(c)=0$.

#### 일반적인 사잇값 정리의 도출

이제 $f(a)<p<f(b)$ 라고 하자.  
다음 함수를 정의한다. $g(x)=f(x)-p$.

$g$는 $[a,b]$에서 연속이고, $g(a)=f(a)-p<0$, $g(b)=f(b)-p>0$.

방금 증명한 영점 정리에 의해 어떤 $c\in(a,b)$가 존재하여 $g(c)=0$.

즉, $f(c)-p=0$,

따라서 $f(c)=p$.

이것이 일반적인 사잇값 정리다.

#### 정의: 중간값(사잇값) 성질
구간 $[a,b]$에 속한 모든 $x< y$와 $f(x)$와 $f(y)$ 사이의 모든 $L$에 대해 $f(c)=L$인 점 $c \in (x,y)$를 항상 찾을 수 있을 때, 함수 $f$는 구간 $[a,b]$에서 사잇값 성질을 가진다고 한다.

- 구간 [a,b]에서 연속인 함수는 항상 사잇값 성질을 가진다
  - 역은 성립하지 않는다
- 증가/감소하는 함수가 사잇값 성질을 가지면, 함수 $f$는 $[a, b]$에서 연속이다.
  - 증가(increase): $\forall x, y \in A, x < y$이고 $f(x) \leq f(y)$이면 함수 $f$는 $A$에서 증가한다고 한다.

### Thm 3. 연속함수의 합성
$f: A \to \mathbb{R}$와 $g: B \to \mathbb{R}$에 대하여 치역 $f(A) = \{f(x): x \in A\}$가 정의역 $B$에 포함되어 합성함수 $g \circ f(x) = g(f(x))$가 $A$에서 잘 정의된다 가정하자.

$f$가 $c \in A$에서 연속이고 $g$가 $f(c) \in B$에서 연속이면 $g\circ f$ 가 $c$에서 연속이다.


### 단사이며 연속인 함수의 역함수도 연속함수
함수 $f: A \to \mathbb R$가 단사(injective)함수면 $f$의 치역에서 역함수 $f^{-1}$를 자연스럽게 정의할 수 있다. $f$가 구간 $[a,b]$에서 연속이고 단사함수면 $f^{-1}$도 연속함수다.

#### 증명1
$f:[a,b]\to\mathbb R$가 연속이고 단사라고 하자. 

이 증명의 구조는 다음과 같다.

$$
\begin{aligned}
&f\text{가 연속이고 단사}\\
&\quad\Longrightarrow f\text{가 엄격한 단조함수}\\
&\quad\Longrightarrow f^{-1}\text{도 엄격한 단조함수}\\
&\quad\Longrightarrow f^{-1}\text{가 사잇값 성질을 가짐}\\
&\quad\Longrightarrow f^{-1}\text{가 연속}.
\end{aligned}
$$

먼저 $f$가 엄격한 단조함수임을 보인다.

**1. 연속인 단사함수는 엄격한 단조함수다**

임의의 $x_1<x_2<x_3$를 잡는다. $f(x_2)$는 반드시 $f(x_1)$과 $f(x_3)$ 사이에 있어야 한다.

그렇지 않고, 예를 들어 $f(x_2)>\max\{f(x_1),f(x_3)\}$라고 하자. 그러면 $\max\{f(x_1),f(x_3)\}<p<f(x_2)$인 $p$를 선택할 수 있다.

사잇값 정리를 $[x_1,x_2]$에 적용하면 어떤 $u\in(x_1,x_2)$가 존재하여 $f(u)=p$이다. 또한 사잇값 정리를 $[x_2,x_3]$에 적용하면 어떤 $v\in(x_2,x_3)$가 존재하여 $f(v)=p$이다.

그런데 $u\neq v$이면서 $f(u)=f(v)$이므로 $f$의 단사성에 모순이다.

마찬가지로 $f(x_2)<\min\{f(x_1),f(x_3)\}$인 경우에도 단사성에 모순된다. 따라서 $x_1<x_2<x_3$이면 $f(x_2)$는 항상 $f(x_1)$과 $f(x_3)$ 사이에 있다.

한편 단사성에 의해 $f(a)\neq f(b)$이다.
- $f(a)<f(b)$이면 $f$는 엄격히 증가한다.
- $f(a)>f(b)$이면 $f$는 엄격히 감소한다.

이제 두 경우를 나누어 생각한다.

**2. $f$가 엄격히 증가하는 경우**

$f$가 연속이고 $f(a)<f(b)$이므로 사잇값 정리에 의해

$$
 f([a,b])=[f(a),f(b)]
$$

역함수를 $g=f^{-1}:[f(a),f(b)]\to[a,b]$라고 하자.

$g$가 증가함수임을 보이기: 
>$u<v$라고 하자. 만약 $g(u)\ge g(v)$라면 $f$의 증가성에 의해
>
>$$
> u=f(g(u))\ge f(g(v))=v
>$$
>
>이 되어 $u<v$에 모순이다. 따라서
>
>$$
> u<v\implies g(u)<g(v).
>$$
>
>즉, $g$는 엄격히 증가한다.

$g$가 사잇값 성질을 가짐을 보이기
>$u<v$이고 $g(u)<q<g(v)$라고 하자. $f$가 엄격히 증가하므로
>
>$$
> f(g(u))<f(q)<f(g(v)).
>$$
>
>그런데 $f(g(u))=u$, $f(g(v))=v$이므로
>
>$$
> u<f(q)<v.
>$$
>
>이제 $w=f(q)$라고 두면 $w\in(u,v)$이고
>
>$$
> g(w)=g(f(q))=q.
>$$
>
>따라서 $g$는 사잇값 성질을 가진다.

증가함수가 사잇값 성질을 가지면 연속이므로 (따로 증명필요) $g=f^{-1}$ 는 연속이다.

**3. $f$가 엄격히 감소하는 경우**

이 경우 $f([a,b])=[f(b),f(a)]$ 이고, 역함수 $g=f^{-1}:[f(b),f(a)]\to[a,b]$도 엄격히 감소한다.

앞과 같은 방법으로 $g$가 사잇값 성질을 가진다는 것을 보일 수 있다. 이제 $h=-g$ 라고 두면 $h$는 증가함수이고 사잇값 성질을 가진다. 따라서 3번 문제에 의해 $h$는 연속이다. 그러므로 $g=-h$ 도 연속이다.

두 경우 모두 역함수가 연속하므로

$$
 f^{-1}:f([a,b])\to[a,b]
$$

는 연속함수이다.


#### 증명2: 수열을 이용한 증명

함수 $f:[a,b]\to\mathbb{R}$가 연속이고 단사라고 하자. 다음과 같이 둔다: $B=f([a,b]),\ g=f^{-1}:B\to[a,b].$

임의의 $y\in B$를 잡고 $x=g(y)$라고 하자. 그러면 $f(x)=y$이다.

이제 $B$ 안의 임의의 수열 $(y_n)$이 $y_n\to y$를 만족한다고 하자. 다음과 같이 둔다. $x_n=g(y_n)$. 

그러면 역함수의 정의에 의해 $f(x_n)=y_n$이다. 

이제 $x_n\to x$임을 보이면 된다.

**모순을 이용한 증명**

$x_n\not\to x$라고 가정하자. 그러면 어떤 $\varepsilon_0>0$와 부분수열 $(x_{n_k})$가 존재하여 모든 $k$에 대하여 $|x_{n_k}-x|\ge\varepsilon_0$이다.

한편 모든 $x_{n_k}$는 닫힌 유계구간 $[a,b]$에 속한다. 따라서 볼차노–바이어슈트라스 정리에 의해 $(x_{n_k})$에는 수렴하는 부분수열이 존재한다.

이를 다시 $(x_{n_{k_j}})$라고 쓰면, 어떤 $x^*\in[a,b]$가 존재하여 $x_{n_{k_j}}\to x^*$이다.

$f$가 연속이므로 $f(x_{n_{k_j}})\to f(x^*)$.

그런데 $f(x_{n_{k_j}})=y_{n_{k_j}}$이고 $y_n\to y$이므로 $y_{n_{k_j}}\to y$.  
극한의 유일성에 의해 $f(x^*)=y$.

한편 $f(x)=y$이므로 $f(x^*)=f(x)$ 인데 $f$가 단사이므로 $x^*=x$이다.  
따라서 $x_{n_{k_j}}\to x$이다.
하지만 모든 $j$에 대하여 $|x_{n_{k_j}}-x|\ge\varepsilon_0$였으므로 이는 모순이다.

따라서 $x_n\to x$. 즉 $y_n\to y\implies g(y_n)\to g(y)$.

실수에서 수열에 의한 연속성과 $\varepsilon$-$\delta$ 연속성은 동치이므로 $g=f^{-1}$는 $y$에서 연속이다. $y\in B$가 임의적이므로

$f^{-1}:f([a,b])\to[a,b]$는 연속이다.


# 연습문제
1. 다음을 증명하시오.  

(1) 
$f(x)=\begin{cases}
x+1,\ x\ge 0 \\
-x,\ x<0  
\end{cases}$  
일 때,
$\displaystyle \lim_{x\to 0} f(x)=0$

(2) $\displaystyle \lim_{x\to\infty} (2x-1)=\infty$

(3) $\displaystyle \lim_{x\to 0} x\sin\frac{1}{x}=0$

2. 함수 $f(x)=2x-3$는 $x=-2$에서 연속임을 보이시오.

3. 함수 $f(x)=x^2$는 $(0,\infty)$에서 균등연속이 아님을 보이시오.

4. 다음 함수들이 $x=0$에서 왜 불연속인지 설명하시오.

(1)
$f(x)=\begin{cases}
x^2 + 1,\ x\neq 0 \\
0,\ x=0
\end{cases}$

(2)
$g(x)=\begin{cases}
x+1,\ x\ge 0 \\
2x-1,\ x<0
\end{cases}$

(3)
$h(x)=\begin{cases}
\frac{1}{x},\ x\ge 0 \\
\sin\frac{1}{x},\ x<0
\end{cases}$

5. 정의역이 $[-1,1]$이고 치역이 $(-1,1)$인 연속함수가 존재할 수 있는지 설명하시오.

6. 함수 $f(x)=x^{10}+3x-1$가 $(0,1)$에서 실근을 가짐을 증명하시오.
