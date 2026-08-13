# 립시츠 함수 (Lipschitz function)

$\forall x \neq y \in A$에 대해 다음 부등식을 만족하는 $M >0$이 존재하면 함수 $f: A \to \mathbb R$을 립시츠 함수라 한다

$$\left|\frac{f(x)-f(y)}{x-y}\right| \leq M$$

- 기하학적으로 함수 f의 그래프 위 임의의 두 점을 지나는 직선의 기울기를 모은 집합이 유계일 때, f는 립시츠 함수다.

- $f: A \to \mathbb R$이 립시츠면 $A$에서 고른연속이다.
- 역은 성립하지 않는다.

## Lipschitz 연속 (Lipschitz Continuity)
함수 $f: X \to Y$가 **Lipschitz 연속**이라는 것은 다음을 만족하는 상수 $K \geq 0$가 존재하는 것이다:  
($d_X$와 $d_Y$는 각각 공간 $X$와 $Y$의 거리 함수 (metric))  

$$d_Y(f(x_1), f(x_2)) \leq K \cdot d_X(x_1, x_2), \quad \forall x_1, x_2 \in X$$

### 주요 성질
- $K$를 **Lipschitz 상수**라고 한다
- Lipschitz 연속은 함수의 변화율이 상수 $K$로 제한됨을 의미한다
- **함의 관계**: Lipschitz 연속 $\Rightarrow$ 균등연속 $\Rightarrow$ 연속
  - Lipschitz 연속이면 $\delta = \epsilon/K$로 선택하여 균등연속성 증명 가능
  - 균등연속은 각 점에서의 연속성을 함의

- $f$가 닫힌구간 $[a,b]$에서 미분가능하고 $f'$이 $[a,b]$에서 연속이면 $f$가 $[a,b]$에서 립시츠 함수다.

### 예시
1. $f(x) = |x|$는 $K=1$인 Lipschitz 연속
   - $||x_1| - |x_2|| \leq |x_1 - x_2|$ (삼각부등식)
2. $f(x) = x^2$는 유계 구간에서만 Lipschitz 연속
   - $|x_1^2 - x_2^2| = |x_1 + x_2| \cdot |x_1 - x_2| \leq 2M|x_1 - x_2|$ (when $|x_i| \leq M$)
3. $f(x) = \sqrt{x}$는 Lipschitz 연속이 아님 ($x=0$ 근처에서 미분 불가능)


# 측도 (Measure) 
지금까지 길이는 구간 도는 구간의 유한합집합에만 사용했다. 길이 개념을 더 일반적인 집합으로 일반화한 개념이 측도(measure)다.

**측도** 를 주로 $\mu$로 표기한다.
- 정의역: 가측 집합들의 모임 $\mathcal{M}$ (시그마-대수)  
- 치역: $[0, \infty]$

## 정의 [측도 0] *(measure zero)*

집합 $A \subset R$이 다음 성질을 만족할 때 **측도가 0(measure zero)인 집합** 이라 한다:  
임의의 $\epsilon >0$에 대해 열린구간 $O_n$의 가산 모임(countable collection)이 존재하여 $A$가 $O_n$의 합집합에 포함되고, $O_n$의 길이를 모두 합하면 $\epsilon$보다 작거나 같다. 즉 구간 $O_n$의 길이를 $|O_n|$로 나타내면 $$A \subseteq \bigcup_{n=1}^\infty O_n, \quad \sum_{n=1}^\infty |O_n| \leq \epsilon$$

- 참고: 칸토어 집합은 셀 수 없는 집합임에도 측도가 0이다. 즉, 원소의 개수와 집합의 길이는 서로 다른 개념이다.

### 예제 
유한집합 $A = \{a_1, a_2, ..., a_N\}$의 측도가 0임을 보이기 위해 $\epsilon >0$을 임의로 두자.  

각 $1 \leq n \leq N$에 대해 구간을 $G_n = (a_n - \frac\epsilon{2N}, a_n + \frac\epsilon{2N})$으로 잡는다.  
$A$는 구간 $G_n$의 합집합에 포함되며 다음이 성립한다:

$$\sum_{n=1}^N |G_n| = \sum_{n=1}^N\frac\epsilon N = \epsilon$$

## 측도의 성질
1. $\mu(\emptyset) = 0$ (공집합의 측도는 0)
2. **가산 가법성**: 서로소인 가측 집합 $E_1, E_2, \ldots$에 대해
   $$\mu\left(\bigcup_{i=1}^{\infty} E_i\right) = \sum_{i=1}^{\infty} \mu(E_i)$$
   - 두 집합 $A$, $B$의 측도가 각각 0이면 $A \cup B$도 측도가 0이다
   - 측도가 0인 집합의 셀 수 있는 합집합도 측도가 0이다.

예시:
- **르벡 측도**: $\mathbb{R}^n$에서 일반적인 "길이", "넓이", "부피"
- **계수 측도**: 집합의 원소 개수
- **확률 측도**: 전체 공간의 측도가 1인 측도

## 정의. $\alpha$-연속성 *($\alpha$-continuous)*
$f$가 $[a,b]$에서 정의되고 $\alpha >0$이라 하자. 어떤 $\delta >0$이 존재하여 모든 $y, z \in (x-\delta, x+\delta)$에 대해 $|f(y)-f(z)| <\alpha$일때, 함수$f$는 $x\in [a,b]$에서 $\alpha$-연속(continuous) 라고 한다.

- $D^\alpha_f$ 정의: $\mathbb R$위에 함수 $f$가 주어질 때, 이 함수가 $\alpha$-연속이 아닌 점의 집합
  - $D^\alpha_f = \{x\in \mathbb R: f$는 $x$에서 $\alpha$-연속이 아님. $\}$
  - $D_f = \{x\in \mathbb R: f$는 $x$에서 연속이 아님. $\}$
  - $D = \{x\in [a,b]: f$는 $x$에서 연속이 아님. $\}$

**성질**

- $\alpha < \alpha'$이면 $D^{\alpha'} \subseteq D^\alpha$

- $\alpha>0$에 대해 $f$가 $x\in[a,b]$에서 연속이면 $x$에서 $\alpha$-연속이다

- $f$가 $x$에서 연속이 아니면, $f$는 어떤 $\alpha >0$에 대해 $\alpha$연속이 아니다. 이를 활용하면 다음과 같음을 보일 수 있다:

$$ D = \bigcup_{i=1}^{\infty} D^{\alpha_n}, \quad\alpha_n=1/n$$

- 임의의 $\alpha>0$에 대해 $D^\alpha$는 닫힌 집합니다

### 정의. 고른 $\alpha$-연속성 *(uniformly $\alpha$-continuous)*
주어진 $\alpha >0$ 에 대해 어떤 $\delta >0$이 존재하여 $x$와 $y$가 $|x- y| < \delta$를 만족하는 $A$의 점이라 하자. 
이때 $|f(x)-f(y)| <\alpha$이면, 함수$f: A \to \mathbb R$이 $A$에서 고른 $\alpha$-연속 이라고 한다.


## 르베그 정리 (르벡 정리, Lebesgue's theorem)

구간 $[a,b]$ 에서 정의된 유계함수 $f$가 리만적분가능하기 위한 필요충분 조건은 $f$의 불연속점 집합이 측도 0인 것이다.

**증명**

아래에서는 $a<b$이고 $f:[a,b]\to\mathbb R$가 유계라고 가정한다. $M>0$을 $|f(x)|\le M \quad (x\in[a,b])$ 이 되도록 잡는다.  
불연속점 집합과 $\alpha$-불연속점 집합을 각각 $D, D^\alpha$ 로 둔다. 


**정방향: (D)의 측도가 (0)이면 $f$는 리만 적분가능하다**

임의의 $\varepsilon>0$에서 다음과 같이 둔다:

$$\alpha=\frac{\varepsilon}{2(b-a)}$$

문제 9: 다음을 만족하는 유한개의 서로소인 열린 구간 $G_1,\dots,G_N$이 있음을 보여야 한다.

$$D^\alpha\subseteq\bigcup_{n=1}^N G_n, \quad \sum_{n=1}^N|G_n|<\frac{\varepsilon}{4M}.$$

>문제 7에서 $D^\alpha\subseteq D$ 임을 보였다. 가정에 의해 $(D)$의 측도가 $0$이므로 그 부분집합 $D^\alpha$도 측도가 $0$이다.
>
>따라서 $D^\alpha$를 덮는 가산개의 열린 구간 $\{O_j\}$를
>
>$$D^\alpha\subseteq\bigcup_{j=1}^{\infty}O_j,
>\qquad
>\sum_{j=1}^{\infty}|O_j|<\frac{\varepsilon}{4M}$$
>
>이 되도록 선택할 수 있다.
>
>한편 문제 8에 의해 $D^\alpha$는 닫힌 집합이다. 또한 $(D^\alpha\subseteq[a,b])$이므로 $D^\alpha$는 닫히고 유계인 집합, 즉 콤팩트 집합이다. 따라서 열린 덮개에서 유한 부분 덮개를 선택할 수 있다.
>
>$$D^\alpha\subseteq O_1\cup\cdots\cup O_r.$$
>
>이 유한개의 구간이 겹친다면 그 합집합의 연결성분들을 취한다. 유한개의 열린 구간의 합집합은 유한개의 서로소인 열린 구간의 합집합으로 표현된다. 이를 $G_1,\dots,G_N$이라 하면
>
>$$\bigcup_{n=1}^NG_n=\bigcup_{j=1}^rO_j$$
>
>이고,
>
>$$\sum_{n=1}^N|G_n|=\left|\bigcup_{j=1}^rO_j\right|\le\sum_{j=1}^r|O_j|<\frac{\varepsilon}{4M}.$$
>
>따라서
>
>$$\boxed{D^\alpha\subseteq\bigcup_{n=1}^N G_n,\qquad\sum_{n=1}^N|G_n|<\frac{\varepsilon}{4M}}$$
>
>인 유한개의 서로소인 열린 구간이 존재한다.

---

문제 10: 다음과 같이 정의한다. 

$$K=[a,b]\setminus\bigcup_{n=1}^NG_n.$$

$f$가 $K$에서 **고른** $\alpha$-연속임을 증명해야 한다.  

>어떤 $\delta>0$가 존재하여 모든 $y,z\in K$에 대해 $|y-z|<\delta \Rightarrow |f(y)-f(z)|<\alpha$ 임을 보여야 한다.
>
>$\bigcup_{n=1}^NG_n$은 열린 집합이므로 $K$는 $[a,b]$에서 닫힌 집합이다. 따라서 $K$는 콤팩트하다. 또한 $D^\alpha\subseteq\bigcup_{n=1}^NG_n$ 이므로 $K\cap D^\alpha=\varnothing.$
>
>따라서 모든 $x\in K$에서 $f$는 $\alpha$-연속이다. 즉, 각 $x\in K$에 대해 어떤 $r_x>0$가 존재하여 $y,z\in(x-r_x,x+r_x)\cap[a,b]$ 이면 $|f(y)-f(z)|<\alpha$ 이다.
>
>다음 열린 구간들은 $K$를 덮는다.
>
>$$\left\{\left(x-\frac{r_x}{4},x+\frac{r_x}{4}\right):x\in K\right\}.$$
>
>$K$가 콤팩트하므로 유한 부분 덮개를 선택할 수 있다.
>
>$$K\subseteq\bigcup_{i=1}^s\left(x_i-\frac{r_{x_i}}4,x_i+\frac{r_{x_i}}4\right).$$
>
>이제
>
>$$\delta=\min_{1\le i\le s}\frac{r_{x_i}}2>0$$
>
>로 둔다.
>
>$y,z\in K$이고 $|y-z|<\delta$라고 하자. 유한 부분 덮개에 의해 어떤 $i$가 존재하여
>
>$$|y-x_i|<\frac{r_{x_i}}4$$
>
>이다. 그러면
>
>$$|z-x_i|\le |z-y|+|y-x_i|<\frac{r_{x_i}}2+\frac{r_{x_i}}4<r_{x_i}.$$
>
>따라서 $y,z\in(x_i-r_{x_i},x_i+r_{x_i})$이고, $x_i$에서의 $\alpha$-연속성에 의해 $|f(y)-f(z)|<\alpha$ 이다.
>
>그러므로
>
>$$\boxed{f\text{는 }K\text{에서 고른 }\alpha\text{-연속이다}}$$
>
---

문제 11: $U(f,P_\varepsilon)-L(f,P_\varepsilon)\le\varepsilon
$ 인 분할 $P_\varepsilon$를 구성하는 방법을 고안하여 정방향 증명을 완성하라.

>**1. 분할의 구성**
>
>문제 9에서 얻은 구간 $(G_1,\dots,G_N)$의 양 끝점 중 $[a,b]$ 안에 있는 점들을 모두 분할점으로 포함시킨다.  
>문제 10에서 얻은 $\delta>0$에 대해 분할의 메시가 $|P_\varepsilon|<\delta$ 가 되도록 필요한 분할점을 더 추가한다.
>
>분할을 $P_\varepsilon =\{a=x_0<x_1<\cdots <x_m=b\}$ 라 하고 $I_k=[x_{k-1},x_k],\ \Delta x_k=x_k-x_{k-1}$ 로 둔다. 각 부분구간에서 $M_k=\sup_{x\in I_k}f(x), \ m_k=\inf_{x\in I_k}f(x)$ 라 하면
>
>$$
>U(f,P_\varepsilon)-L(f,P_\varepsilon) =\sum_{k=1}^m(M_k-m_k)\Delta x_k
>$$
>
>**2. 열린 구간 $(G_n)$ 안의 부분구간**
>
>$I_k\subseteq\bigcup_{n=1}^NG_n$인 부분구간들을 생각한다. $|f(x)|\le M$ 이므로 $M_k-m_k\le 2M$. 따라서 이 부분구간들이 상합과 하합의 차이에 기여하는 양은
>
>$$
>\sum_{I_k\subseteq\cup G_n}
>(M_k-m_k)\Delta x_k \le 2M\sum_{I_k\subseteq\cup G_n}\Delta x_k
>\le 2M\sum_{n=1}^N|G_n| < 2M\cdot\frac{\varepsilon}{4M} =\frac{\varepsilon}{2}
>$$
>
>**3. 열린 구간 밖의 부분구간**
>
>나머지 부분구간들은 $K$ 안에 들어 있다. 또한 $\Delta x_k<\delta$  
>문제 10의 고른 $\alpha$-연속성에 의해 모든 $y,z\in I_k$에 대해 $|f(y)-f(z)|<\alpha$ 따라서 $M_k-m_k\le\alpha$
>
>그러므로 이 부분구간들의 기여는
>
>$$
>\sum_{I_k\subseteq K}(M_k-m_k)\Delta x_k \le \alpha\sum_{I_k\subseteq K}\Delta x_k
>\le\alpha(b-a) =\frac{\varepsilon}{2(b-a)}(b-a) =\frac{\varepsilon}{2}.
>$$
>
>두 부분을 합하면 $U(f,P_\varepsilon)-L(f,P_\varepsilon) < \frac{\varepsilon}{2}+\frac{\varepsilon}{2} =\varepsilon$  
>따라서 임의의 $\varepsilon>0$에 대해 $U(f,P_\varepsilon)-L(f,P_\varepsilon)<\varepsilon$ 인 분할이 존재한다. 다르부 판정법에 의해
>
>$$
>\boxed{f\text{는 }[a,b]\text{에서 리만 적분가능하다}}
>$$
>
---

"역방향: $f$가 리만 적분가능하면 $D$의 측도는 0이다"  

이제 $f$가 $([a,b])$에서 리만 적분가능하다고 가정한다.

문제 12(a): 모든 $\alpha>0$에 대해 $D^\alpha$의 측도가 0임을 증명하라.

**1. 적분가능성으로부터 분할 선택**

$\alpha>0$ 을 고정하고, 측도 0의 정의에 사용할 임의의 $\varepsilon>0$을 잡는다. $f$가 리만 적분가능하므로 어떤 분할 $P=\{a=x_0<x_1<\cdots <x_N=b\}$ 이 존재하여 $U(f,P)-L(f,P) < \frac{\alpha\varepsilon}{4}$ 을 만족한다. 각 부분구간 $I_k=[x_{k-1},x_k]$에서 진동을 $\omega_k=M_k-m_k$ 로 둔다. 그러면

$$
\sum_{k=1}^N\omega_k\Delta x_k < \frac{\alpha\varepsilon}{4}
$$

**2. 진동이 큰 부분구간**

다음 지표집합을 생각한다: $B=\{k:\omega_k\ge\alpha\}$ 그러면

$$
\alpha\sum_{k\in B}\Delta x_k 
\le \sum_{k\in B}\omega_k\Delta x_k
\le U(f,P)-L(f,P) < \frac{\alpha\varepsilon}{4}.
$$

$\alpha>0$이므로 $\sum_{k\in B}\Delta x_k<\frac{\varepsilon}{4}$ 이다.

**3. $D^\alpha$가 어디에 포함되는가?**

분할점이 아닌 점 $x$가 진동이 작은 부분구간 $I_k$, 즉 $\omega_k<\alpha$ 인 부분구간의 내부에 있다고 하자.  
$x$에서 부분구간의 양 끝점까지의 거리보다 작은 $\delta>0$를 선택하면 $(x-\delta,x+\delta)\subseteq I_k$  

따라서 $y,z \in (x-\delta,x+\delta)$이면 $|f(y)-f(z)| \le M_k-m_k =\omega_k <\alpha$

그러므로 $f$는 $x$에서 $\alpha$-연속이다. 따라서 이러한 $x$는 $D^\alpha$에 속하지 않는다.

결국

$$
D^\alpha \subseteq \{x_0,x_1,\dots,x_N\}
\cup \bigcup_{k\in B}I_k.
$$

$D^\alpha$는 진동이 $\alpha$ 이상인 부분구간들과 유한개의 분할점으로 덮인다.

**4. 유한개의 열린 구간으로 덮기**

각 $k\in B$ 에 대해 $I_k$를 약간 확장하여 열린 구간

$$
G_k= \left( x_{k-1}-\frac{\varepsilon}{8N},\ 
 x_k+\frac{\varepsilon}{8N} \right)
$$

을 잡는다. 그러면 $|G_k| =\Delta x_k+\frac{\varepsilon}{4N}$ 따라서

$$
\sum_{k\in B}|G_k| = \sum_{k\in B}\Delta x_k +\frac{|B|\varepsilon}{4N} < \frac{\varepsilon}{4}+\frac{\varepsilon}{4} =\frac{\varepsilon}{2}.
$$

각 분할점 $x_j$도 길이가 $\frac{\varepsilon}{2(N+1)}$ 인 열린 구간 $H_j$로 덮는다. 그러면 $\sum_{j=0}^N|H_j| =\frac{\varepsilon}{2}$

따라서 $D^\alpha$를 덮는 유한개의 열린 구간들의 전체 길이는

$$
\sum_{k\in B}|G_k|+\sum_{j=0}^N|H_j|
< \frac{\varepsilon}{2}+\frac{\varepsilon}{2} =\varepsilon.
$$

$\varepsilon>0$ 은 임의였으므로

$$
\boxed{D^\alpha\text{의 측도는 }0}
$$

---

문제 12(b): 불연속점 집합 $D$의 측도가 0임을 증명한다. 

>문제 7에서 $D=\bigcup_{n=1}^{\infty}D^{1/n}$ 임을 증명했다.  
>문제 12(a)에 의해 각각의 $D^{1/n}$은 측도가 0이다. 문제 5에 의해 측도 0인 집합들의 가산 합집합도 측도가 0이므로 $D=\bigcup_{n=1}^{\infty}D^{1/n}$ 의 측도는 0이다.
>
---

르베그 정리의 완성: 두 방향을 모두 정리하면 다음과 같다.

**충분조건**: $D$ 의 측도가 0 이면 문제 9~11에 의해 임의의 $\varepsilon>0$ 에 대해 $U(f,P_\varepsilon)-L(f,P_\varepsilon)<\varepsilon$ 인 분할을 구성할 수 있으므로 $f$는 리만 적분가능하다.

**필요조건**: $f$가 리만 적분가능하면 문제 12(a)에 의해 모든 $D^{1/n}$의 측도가 0이고,  $D=\bigcup_{n=1}^{\infty}D^{1/n}$ 이므로 $D$도 측도가 0이다.

따라서

$$
\boxed{
f\text{가 }[a,b]\text{에서 리만 적분가능}
\iff
f\text{의 불연속점 집합 }D\text{의 측도가 }0
}
$$

* 정방향에서는 불연속성이 큰 점들의 집합 $D^\alpha$를 전체 길이가 매우 작은 구간 안에 가둔다.
* 그 구간 밖에서는 콤팩트성을 이용하여 함수의 진동을 일률적으로 작게 만든다.
* 역방향에서는 상합과 하합의 차이가 작다는 사실로부터 진동이 큰 부분구간들의 전체 길이가 작다는 것을 얻는다.


## 적분 불가능한 도함수

미적분학 기본정리에 따르면 적분과 미분은 서로 역연산 관계고, 함수 $f$가 $[a,b]$에서 미분가능할 때 $f'$이 적분가능하면 $\int_a^b f' = f(b) - f(a)$다.

그런데 $f'$은 도함수라는 이유로 적분가능이 당연히 보장되어야 하는게 아니다! 

$\int_a^b f'$이 존재하지 않은 미분가능한 함수 $f$가 존재한다.


# 르벡 적분 (Lebesgue Integration)

## 르벡 적분의 기본 아이디어
리만 적분과 달리, 르벡 적분은 **함숫값(치역)을 갖는 영역의 크기**를 더하는 방식이다.

**기본 구조**:
1. **단순 함수로 시작**: 값이 유한개인 계단 함수 사용
2. **각 함숫값에 대해 계산**:
   $\text{함숫값}) \times (\text{그 값을 갖는 집합의 측도})$
3. **근사의 극한**: 점점 더 정교한 단순 함수로 근사하여 극한을 취함

## 단순 함수 (Simple Function)

$$s(x) = \sum_{i=1}^{n} a_i \chi_{E_i}(x)$$

여기서:
- $a_i \in \mathbb{R}$는 **함수가 가지는 유한개의 상수 값**
- $E_i$는 서로소인 **가측 집합** (measurable set): 측도를 잴 수 있는 집합
- $\chi_{E_i}$는 **특성 함수** (characteristic function): 
  
  $$\chi_{E_i}(x) = \begin{cases} 1 & x \in E_i \\ 0 & x \notin E_i \end{cases}$$

단순 함수의 르벡 적분:

$$\int s \, d\mu = \sum_{i=1}^{n} a_i \mu(E_i)$$

## 비음 가측 함수의 적분
비음 가측 함수 $f \geq 0$에 대해:

$$\int f \, d\mu = \sup \left\{ \int s \, d\mu : 0 \leq s \leq f, \, s \text{ 단순함수} \right\}$$

- 아래에서 근사하는 단순함수들의 적분의 상한으로 정의
- 리만 적분과 달리 정의역을 분할하지 않고 치역을 분할

## 일반 가측 함수의 적분
임의의 가측 함수 $f$를 양의 부분과 음의 부분으로 분해:

$$f^+(x) = \max(f(x), 0), \quad f^-(x) = \max(-f(x), 0$$

이때 $f = f^+ - f^-$이고:

$$\int f \, d\mu = \int f^+ \, d\mu - \int f^- \, d\mu$$

함수 $f$가 **적분가능** (integrable)하려면 $\int f^+ \, d\mu < \infty$ 이고 $\int f^- \, d\mu < \infty$ 이어야 한다.

## 주요 정리
**단조 수렴 정리 (Monotone Convergence Theorem, MCT)**:
- 조건: $0 \leq f_1 \leq f_2 \leq \cdots$, $f_n \uparrow f$ a.e. (almost everywhere, 거의 어디서나)
- 결론: $\lim_{n \to \infty} \int f_n \, d\mu = \int f \, d\mu$
- 의미: 단조증가 수열은 극한과 적분의 순서 교환 가능

**지배 수렴 정리 (Dominated Convergence Theorem, DCT)**:
- 조건: 
  - $f_n \to f$ a.e.
  - $|f_n| \leq g$ a.e. for all $n$, where $\int g \, d\mu < \infty$
- 결론: $\lim_{n \to \infty} \int f_n \, d\mu = \int f \, d\mu$
- 의미: 적분가능한 함수로 지배되면 극한과 적분 순서 교환 가능

**Fatou의 보조정리 (Fatou's Lemma)**:
- 조건: $f_n \geq 0$ a.e.
- 결론: $\int \liminf_{n \to \infty} f_n \, d\mu \leq \liminf_{n \to \infty} \int f_n \, d\mu$
- 의미: 하극한의 적분 $\leq$ 적분의 하극한 (부등식만 성립) Analysis 추가 개념
