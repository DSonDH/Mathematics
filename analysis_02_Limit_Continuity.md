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

### Def 1. *(수렴과 극한, Limit of a function at a point)*

$f : D \to \mathbb{R},\ a \in D,\ L \in \mathbb{R}$ 라 하자.

$$ \forall \epsilon>0,\ \exists \delta>0\ \text{s.t.}\ \forall x\in D,\\
0 < |x-a| < \delta \Rightarrow |f(x)-L|<\epsilon $$

이 성립하면 $f$는 $x=a$에서 극한값 $L$로 수렴한다고 하고, 
$\lim_{x\to a} f(x)=L$ 로 표기한다.

- 함숫값 f(x) = a는 고려안함. 따라서 죄부등식에 등호가 없음
- 수렴안하는 경우엔 발산한다고 함

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
\lim_{x\to 0} f(x)=0.
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

### Def 2. *(우극한/좌극한, Right/Left Limits)*
$f:D\to\mathbb{R},\ a\in D',\ L\in\mathbb{R}$ 라 하자.

(좌극한)
$\forall\varepsilon>0,\ \exists\delta>0,$
$0< a-x <\delta \Rightarrow |f(x)-L|<\varepsilon$
→ $\displaystyle \lim_{x\to a^-} f(x)=L=f(a^+)$

(우극한)
$\forall\varepsilon>0,\ \exists\delta>0,$
$0< x-a <\delta \Rightarrow |f(x)-L|<\varepsilon$
→ $\displaystyle \lim_{x\to a^+} f(x)=L=f(a^-)$

### Def 3. *(무한대로의 극한, Limits at infinity)*
극한값이 fix되는 정의. 0보다 크다는 표현이 없으므로, 언저리값이 아님.  

$a, L \in \mathbb{R}$ 이라 하자.

1.
$\forall \varepsilon > 0,\ \exists N\in\mathbb{R}\ \text{s.t.}\ x\geq N \Rightarrow |f(x)-L| < \varepsilon$ 일때 $\displaystyle \lim_{x\to\infty} f(x)=L$
  - ex: $\displaystyle \lim_{x\to\infty} 1/x=0$
  - 증명: 
  임의의 $\varepsilon>0$를 잡는다. 
  $N=\frac{2}{\varepsilon}$ 로 둔다. (즉, $\frac{2}{\varepsilon}\in\mathbb{R}$)  
  $x\ge N$ 이면 $\displaystyle x \ge \frac{2}{\varepsilon}$ 이고, 양변에 역수를 취하면 $\displaystyle \frac{1}{x} \le \frac{\varepsilon}{2}$  
  따라서 $\displaystyle \left|\frac{1}{x}-0\right| = \left|\frac{1}{x}\right| \le \frac{\varepsilon}{2} < \varepsilon$  
  즉, $x\ge N \Rightarrow \left|\frac{1}{x}-0\right|<\varepsilon$  
  그러므로 $\displaystyle \lim_{x\to\infty}\frac{1}{x}=0$ 이다.

2.
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

3.
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
|x-a|<\delta \Rightarrow
|(f(x)+g(x))-(A+B)|
$$
$$
\le |f(x)-A|+|g(x)-B|<\varepsilon.
$$

3.
  - 임의의 $\varepsilon>0$를 잡는다.  삼각부등식에 의해, $$ |f(x)g(x)-AB|=|f(x)g(x)-Ag(x)+Ag(x)-AB| \\ \leq |fg-Ag| + |Ag-AB|= |g||f-A| + |A||g-B| $$

$\lim_{x\to a} g(x)=B$ 이므로  
$\exists\delta_0>0$ s.t. $|x-a|<\delta_0 \Rightarrow |g(x)-B|<1$  
따라서 $|g(x)|\le |B|+1.$  

이제 임의의 $\varepsilon>0$에 대해  
$$|f(x)-A|<\frac{\varepsilon}{2(|B|+1)},\quad
|g(x)-B|<\frac{\varepsilon}{2(|A|+1)}$$
가 되도록 $\delta_1,\delta_2$를 잡는다.  
$\delta=\min\{\delta_0,\delta_1,\delta_2\}$ 이면  
$$|f(x)g(x)-AB| < |g||f-A| + |A||g-B| < \varepsilon/2 + \varepsilon/2$$
 $$\therefore |f(x)g(x)-AB|<\varepsilon.$$

4.
  - 임의의 $\varepsilon>0$를 잡는다.  
$$
\left|\frac{f(x)}{g(x)}-\frac{A}{B}\right|
=\left|\frac{Bf(x)-Ag(x)}{Bg(x)}\right|
\le \frac{|B||f(x)-A|+|A||g(x)-B|}{|B||g(x)|}.
$$

$\lim_{x\to a} g(x)=B\ne0$ 이므로 $\exists\delta_0>0$ s.t.
$$
|x-a|<\delta_0 \Rightarrow |g(x)|>\frac{|B|}{2}.
$$

따라서
$$
\left|\frac{f(x)}{g(x)}-\frac{A}{B}\right|
\le \frac{2}{|B|^2}\Big(|B||f(x)-A|+|A||g(x)-B|\Big).
$$

이제 $|B||f(x)-A|<\frac{|B|^2\varepsilon}{4}$ 가 되도록 $\delta_1>0$ 을 택하고,  
$|A||g(x)-B|<\frac{|B|^2\varepsilon}{4}$ 가 되도록 $\delta_2>0$ 을 택한다.  

$\delta=\min\{\delta_0,\delta_1,\delta_2\}$ 로 두면 $|x-a|<\delta$ 일 때
$$
\left|\frac{f(x)}{g(x)}-\frac{A}{B}\right|
\le \frac{2}{|B|^2}\left(\frac{|B|^2\varepsilon}{4}+\frac{|B|^2\varepsilon}{4}\right)=\varepsilon.
$$

즉,
$$
\lim_{x\to a}\frac{f(x)}{g(x)}=\frac{A}{B}.
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
$$
|L-M|
= |L-f(x)+f(x)-M|
\le |f(x)-L|+|f(x)-M|
< \varepsilon.
$$

임의의 $\varepsilon>0$에 대해 $|L-M|<\varepsilon$이므로 $L=M$이다.

### Thm 2. (샌드위치 정리, *Squeeze Theorem*)
$\forall x\in D,\ f(x)\le g(x)\le h(x)$이고 $L\in\mathbb{R}$에 대하여
$\lim_{x\to a} f(x)=\lim_{x\to a} h(x)=L$이면 $\lim_{x\to a} g(x)=L$이다.

**증명**  
임의의 $\varepsilon>0$를 잡는다.

$\lim_{x\to a} f(x)=L$이므로 $\exists\delta_1>0$ s.t.
$|x-a|<\delta_1 \Rightarrow |f(x)-L|<\varepsilon$, 즉
$L-\varepsilon<f(x)<L+\varepsilon$.

$\lim_{x\to a} h(x)=L$이므로 $\exists\delta_2>0$ s.t.
$|x-a|<\delta_2 \Rightarrow |h(x)-L|<\varepsilon$, 즉
$L-\varepsilon<h(x)<L+\varepsilon$.

$\delta=\min\{\delta_1,\delta_2\}$로 두면 $|x-a|<\delta$에서
$$
L-\varepsilon<f(x)\le g(x)\le h(x)<L+\varepsilon.
$$

따라서 $|g(x)-L|<\varepsilon$가 되어 $\lim_{x\to a} g(x)=L$이다.

**샌드위치 정리 예시**  
$\displaystyle \lim_{x\to 0}\frac{\sin x}{x}=1$ 증명  

$x\neq 0$라 하자. 삼각함수의 기본 부등식에 의해  
$\sin x < x < \tan x \quad (0<x<\tfrac{\pi}{2})$
가 성립한다.

양변을 $x$로 나누면
$\frac{\sin x}{x} < 1 < \frac{\tan x}{x}.$

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
## (1) 연속의 정의 *(Definition of Continuity)*
### Def 1. *(점 연속, Continuity at a point)*

$f:D\to\mathbb{R},\ a\in D$라 하자.

$\forall \varepsilon>0,\ \exists\delta>0$ s.t.
$\forall x\in D, |x-a|<\delta \Rightarrow |f(x)-f(a)|<\varepsilon$  
이 성립하면 $f$는 $x=a$에서 연속이라 한다.  
- $a\in D$: $f(a)$가 정의됨
- $0 < |x-a|$ 조건 없음
- $|f(x)-f(a)|$ 조건이 극한 정의와 달리, 함숫값을 가리킴
- 즉, $f$는 $a$에서 연속 $\iff \lim_{x\to a} f(x)=f(a)$
* 위 정의를 만족하는 모든 대상이 연속이라는 개념을 만족한다.
  - 이산적인 연속도 정의 가능. 아래 예시 참고

예시: D = {0, 1, 2, 3}일때 f(x) = -x + 3이면 f는 x=2에서 연속 증명  
**증명**  
$f(2)=-2+3=1$ 이다. 임의의 $\varepsilon>0$를 잡고, 
$\delta=\frac{1}{2}$로 둔다.  
그러면 $x\in D$에 대해
$|x-2|<\delta$ 를 만족하는 경우는 오직 $x=2$ 뿐이다.
(왜냐하면 $D$의 다른 원소들에 대해 $|x-2|\ge 1$ 이기 때문이다.)

따라서 $|x-2|<\delta$ 이면 반드시 $x=2$이고,
$|f(x)-f(2)| = |1-1| = 0 < \varepsilon$  
즉,
$$
\forall \varepsilon>0,\ \exists\delta>0\ \text{s.t.}\
|x-2|<\delta \Rightarrow |f(x)-f(2)|<\varepsilon.
$$
그러므로 $f$는 $x=2$에서 연속이다.

**개념적으로 중요한 점**  
* 정의역 $D$가 **이산집합**이면
  모든 함수는 정의역의 모든 점에서 연속이다.
* 위 증명은 이를 $\varepsilon$–$\delta$ 정의로 직접 확인한 것이다.
* 이 예시는 연속성은 함수의 식보다 **정의역의 구조**에 크게 의존한다는 점을 보여주는 전형적인 예시다.

### Def 2. *(우연속, 좌연속 — Right/Left continuity)*
$f: D\to \mathbb{R}, a\in D$라 하자.  

(우연속)  
$\forall\epsilon>0, \exists\delta>0$ s.t. $\forall x\in D$  
$0\leq x-a<\delta \Rightarrow |f(x)-f(a)|<\varepsilon$  
이 성립하면 $f$는 $x=a$에서 우연속이라 한다.

(좌연속)  
$\forall\epsilon>0, \exists\delta>0$ s.t. $\forall x\in D$  
$0\leq a-x<\delta \Rightarrow |f(x)-f(a)|<\varepsilon$  
이 성립하면 $f$는 $x=a$에서 좌연속이라 한다.

### Def 3. *(연속함수, Continuous function)*
$f: D\to \mathbb{R}, X \subseteq D$라 하자.  

- $f$가 $X$의 모든 점에서 연속이면 $f$는 $X$에서 연속함수라 한다.

- $f$가 $D$의 모든 점에서 연속이면 $f$는 연속함수라 한다.

- 증명방법
  - x는 a (a는 정의역 원소) 에서 연속임을 보인다
  - a가 원하는 집합의 임의의 점이라고 서술

- 예시: $f(x)=x^2$이 $x>0$에서 연속 증명

임의의 $a\in X$를 잡는다. 즉, $a>0$이다.
임의의 $\varepsilon>0$를 잡는다.
Then $\forall x\in(0, \infty)$, with $|x-a|<\delta$
$$
|f(x)-f(a)| =|x^2-a^2| =|x-a||x+a|
$$

(만약 delta = 1이었으면, 즉 $|x-a|<1$이라고 추가로 가정하면)  
$|x-a|<1$ 이면 $a-1 < x < a+1$ 이므로 $|x+a| < 2a+1$

따라서 $$ |f(x)-f(a)| < (2a+1)|x-a|$$

이제 $\delta = \min\{1, \frac{\varepsilon}{2a+1}\}$
로 두면, $|x-a|<\delta\leq \frac{\varepsilon}{2a+1}$
$$
\therefore |f(x)-f(a)|<\varepsilon.
$$

즉,
$$
\forall \varepsilon>0,\ \exists\delta>0\ \text{s.t.}\ |x-a|<\delta \Rightarrow |f(x)-f(a)|<\varepsilon.
$$
따라서 $f$는 임의의 $a>0$에서 연속이다.  
$a$가 $X$의 임의의 원소였으므로, $f$는 $x>0$에서 연속이다.

### Def 4. *(불연속의 종류, Types of discontinuities)*
연속이 아니면 불연속.  
즉 함숫값 정의가 안되면 무조건 불연속.  
아래는 함숫값 정의되지만 불연속인 경우들.  

$f: D\to \mathbb{R}, a \in D$라 하자. (즉 함숫값이 존재할 때)  
(제 1종 불연속점)  
  - 제거가능(removable discontinuity) 불연속점 $$x=a \quad \text{s.t. }\lim_{x\to a^+}f(x)=\lim_{x\to a^-}f(x)\neq f(a)$$
    - 이 값만 대체하면 연속이 되므로

  - 비약 불연속점 (jump discontinuity) $$x=a \quad \text{s.t. }\lim_{x\to a^+}f(x)\neq\lim_{x\to a^-}f(x)$$
    - 좌극한, 우극한 이지러진 경우

(제 2종 불연속점)  
  - $\displaystyle \lim_{x\to a^+}f(x)$와$\displaystyle \lim_{x\to a^-}f(x)$중에 적어도 하나가 존재하지 않는다.
    - 진동하는 경우 보통은 극한이 없다

## (2) 균등연속 *(Uniform Continuity)*
### Def.
$f: D\to \mathbb{R}$이라 하자.  
$\forall \varepsilon>0,\ \exists \delta>0,\ \forall x,y\in D,$

$|x-y|<\delta \Rightarrow |f(x)-f(y)|<\varepsilon$

이 성립하면 $f$는 $D$에서 균등연속이다.

- f가 균등연속이면 연속.

- 예: $f(x) = x^2$이 $[-2, 2)$에서 균등연속 증명  

정의역을 $D=[-2,2)$라 하자.  
임의의 $\varepsilon>0$를 잡는다.  
임의의 $x,y\in D$에 대해
$|f(x)-f(y)| =|x^2-y^2| =|x-y||x+y|$

$x,y\in[-2,2)$ 이므로
$|x|\le 2, |y|\le 2$
이고, 따라서 $|x+y|\le |x|+|y|\le 4$

그러므로 $|f(x)-f(y)|\le 4|x-y|$

이제 $\delta=\frac{\varepsilon}{4}$
로 두면, $|x-y|<\delta$일 때
$|f(x)-f(y)|<4\cdot\frac{\varepsilon}{4}=\varepsilon$

즉,
$\forall \varepsilon>0,\ \exists\delta>0\ \text{s.t.}\
|x-y|<\delta \Rightarrow |f(x)-f(y)|<\varepsilon
$
가 모든 $x,y\in[-2,2)$에 대해 성립한다.

따라서 $f(x)=x^2$는 $[-2,2)$에서 균등연속이다.

## (3) 연속함수의 연산 *(Operations of continuous functions)*
$a\in D, f,g: D \to \mathbb{R}$가 $x=a$ 에서 연속일때, 다음이 성립한다:
1. $f+g$는 $x=a$ 에서 연속
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
