# 일반화된 리만 적분

함수 $F$가 미분가능하다면 아래 식이 성립함을 증명할 수 있길 바라지만,

$$\int_a^b F' = F(b) -F(A)$$

$F'$이 리만적분 가능하다는 가정이 필요했다. 실제로, 모든 도함수가 적분가능한건 아니었다.

1960년경 야로슬라프 커츠와일(Jaroslav Kurzweil)과 랄프 헨스톡(Ralph Henstock)이 각각 독자적으로 발견한 '일반화된 리만 적분'이 있다. 이는 르벡적분보다 더 많은 함수를 적분할 수 있고, 추가 가정없이 위 식을 간결하게 증명할 수 있다.

## 극한으로서의 리만적분

$[a,b]$에서의 분할 $P = \{x_0, x_1, x_2, ..., x_n\}$ 이 있다.

**태그된 분할(tagged partition)** 은 분할 $P$에서 각 부분구간 $[x_{k-1}, x_k]$ 마다 점 $c_k$를 하나씩 선택하는 조건을 추가한 것이다.

이 분할로 생성된 리만 합(Riemann sum)은 $R(f,P) = \sum_{k=1}^n f(c_k)(x_k - x_{k-1})$ 이다.

### 정의. $\delta$-세분 ($\delta$-fine)
$\delta >0$ 에 대하여 분할 $P$의 모든 부분구간 $[x_{k-1}, x_k]$이 $x_k- x_{k-1} <\delta$를 만족하면 $P$를 $\delta$-세분 ($\delta$-fine)라 한다.

### 정리. 리만 적분가능성에 대한 극한 판정법

유계함수 $f: [a,b]\to \mathbb R$가 리만적분 가능하고 $\int_a^b f = A$이기 위한 필요충분조건은 임의의 $\epsilon >0$에 대해 다음 조건을 만족하는 $\delta >0$이 존재하는 것이다:

임의의 $\delta$-세분인 태그된 분할 $(P, \{c_k\})$에 대해 $|R(f,P)-A|<\epsilon$

**증명**

($\Rightarrow$)

상합과 하합을 각각

$$U(f,P)=\sum_{k=1}^n M_k\Delta x_k,\qquad L(f,P)=\sum_{k=1}^n m_k\Delta x_k$$

로 나타내며,

$$M_k=\sup_{x\in[x_{k-1},x_k]}f(x),\qquad m_k=\inf_{x\in[x_{k-1},x_k]}f(x)$$

로 둔다.

# 문제 1

## (a) (R(f,P))와 (\int_a^b f)가 (L(f,P)), (U(f,P)) 사이에 있는 이유

태그 (c_k\in[x_{k-1},x_k])에 대하여

$$m_k\le f(c_k)\le M_k$$

이다. (\Delta x_k=x_k-x_{k-1}>0)을 곱하면

$$m_k\Delta x_k \le f(c_k)\Delta x_k \le M_k\Delta x_k$$

이다. 이를 모두 더하면

$$L(f,P)\le R(f,P)\le U(f,P)$$

를 얻는다.

또한 (f)가 리만 적분 가능하면 하적분과 상적분이 모두 (\int_a^b f)와 같으므로, 임의의 분할 (P)에 대하여

$$L(f,P) \le \underline{\int_a^b}f =\int_a^b f =\overline{\int_a^b}f \le U(f,P)$$

이다. 따라서

$$\boxed{L(f,P)\le R(f,P)\le U(f,P)}$$

이고

$$\boxed{L(f,P)\le \int_a^b f\le U(f,P)}$$

이다.

## (b) (U(f,P')-L(f,P')<\epsilon/3)인 이유

(P'=P\cup P_\epsilon)이므로 (P')는 (P_\epsilon)의 세분이다. 분할을 세분하면 상합은 감소하고 하합은 증가하므로

$$U(f,P')\le U(f,P_\epsilon), \qquad L(f,P')\ge L(f,P_\epsilon)$$

이다. 따라서

$$U(f,P')-L(f,P') \le U(f,P_\epsilon)-L(f,P_\epsilon) <\frac{\epsilon}{3}$$

즉,

$$\boxed{U(f,P')-L(f,P')<\frac{\epsilon}{3}}$$

이다.

# 문제 2

(P')는 (P)의 세분이므로

$$U(f,P')\le U(f,P)$$

임을 보이면 된다.

(P)의 한 부분구간 (I=[u,v])가 (P')에 의하여

$$u=y_0<y_1<\cdots<y_r=v$$

로 분할되었다고 하자. (I)에서의 상한을

$$M=\sup_{x\in I}f(x)$$

라 하고, 작은 부분구간 ([y_{j-1},y_j])에서의 상한을 (M_j)라 하자. 작은 부분구간은 (I)에 포함되므로

$$M_j\le M$$

이다. 따라서

$$\sum_{j=1}^r M_j(y_j-y_{j-1}) \le \sum_{j=1}^r M(y_j-y_{j-1}) =M(v-u)$$

즉, 하나의 구간을 세분하면 그 구간에 대응하는 상합은 증가하지 않는다. 모든 부분구간에 이 결과를 적용하면

$$U(f,P')\le U(f,P)$$

이다. 그러므로

$$\boxed{U(f,P)-U(f,P')\ge0}$$

이다.

마찬가지로 작은 구간에서의 하한은 원래 구간에서의 하한보다 크거나 같으므로

$$\boxed{L(f,P')-L(f,P)\ge0}$$

이다.

# 문제 3

(P_\epsilon)의 부분구간 개수가 (n)이므로 내부 분할점의 개수는 (n-1)개이다. 이 점들을 (P)에 추가하여

$$P'=P\cup P_\epsilon$$

을 만든다.

## (a) 상쇄되지 않는 항의 개수

(P_\epsilon)의 내부 분할점 중 이미 (P)에 속하지 않는 점의 개수를 (q)라 하자. 그러면

$$q\le n-1$$

이다.

이 (q)개의 점이 들어 있는 (P)의 부분구간 개수를 (s)라 하자. 하나의 해당 부분구간에 (r_j)개의 새로운 점이 들어 있다면

* (U(f,P))에서는 원래 구간에 대응하는 항이 (1)개이고,
* (U(f,P'))에서는 그 구간이 (r_j+1)개로 나뉘므로 항이 (r_j+1)개이다.

따라서 이 구간에서 상쇄되지 않는 항은 총

$$r_j+2$$

개이다. 모든 해당 구간에 대하여 더하면

$$\sum_{j=1}^s(r_j+2)=q+2s$$

여기서 (s\le q)이므로

$$q+2s\le3q\le3(n-1)<3n$$

따라서 상쇄되지 않는 항의 개수는 최대 (3(n-1))개이며, 필요한 간단한 상계는

$$\boxed{3n}$$

개이다.

## (b) (U(f,P)-U(f,P')<\epsilon/3)의 증명

(M>0)이 (|f|)의 상계이므로 모든 부분구간에서

$$|M_k|\le M$$

이다. 또한 (P)가 (\delta)-세분이므로 (P)의 각 부분구간 길이는 (\delta)보다 작다. (P')의 부분구간도 (P)의 부분구간에 포함되므로 그 길이 역시 (\delta)보다 작다.

따라서 상합을 구성하는 각 항의 절댓값은

$$|M_k\Delta x_k|<M\delta$$

이다.

문제 2에서

$$U(f,P)-U(f,P')\ge0$$

임을 알고 있으며, 문제 3(a)에 의해 상쇄되지 않는 항은 (3n)개 미만이다. 따라서

$$\begin{aligned}
0 &\le U(f,P)-U(f,P') \\
&<3nM\delta.
\end{aligned}$$

교재와 같이

$$\delta=\frac{\epsilon}{9nM}$$

로 놓았으므로

$$3nM\delta=3nM\frac{\epsilon}{9nM}=\frac{\epsilon}{3}$$

따라서

$$\boxed{U(f,P)-U(f,P')<\frac{\epsilon}{3}}$$

이다. 즉,

$$U(f,P)<U(f,P')+\frac{\epsilon}{3}$$

이다.

하합에 대해서도 완전히 같은 방법을 적용하면

$$0\le L(f,P')-L(f,P)<\frac{\epsilon}{3}$$

이고, 따라서

$$L(f,P')-\frac{\epsilon}{3}<L(f,P)$$

이다.

결국

$$\boxed{L(f,P')-\frac{\epsilon}{3}<L(f,P)\le U(f,P)<U(f,P')+\frac{\epsilon}{3}}$$

를 얻는다.

# 문제 4

## (a) (f)가 연속인 경우

각 부분구간

$$I_k=[x_{k-1},x_k]$$

는 닫히고 유계인 구간이다. (f)가 연속이므로 최대·최소 정리에 의해 (f)는 (I_k)에서 최댓값과 최솟값을 갖는다.

따라서 어떤 (c_k^+,c_k^-\in I_k)가 존재하여

$$f(c_k^+)=M_k,\qquad f(c_k^-)=m_k$$

이다.

태그를 (c_k=c_k^+)로 선택하면

$$\begin{aligned}
R(f,P)
&=\sum_{k=1}^n f(c_k^+)\Delta x_k\\
&=\sum_{k=1}^n M_k\Delta x_k
=U(f,P).
\end{aligned}$$

따라서

$$\boxed{R(f,P)=U(f,P)}$$

가 되게 하는 태그가 존재한다.

마찬가지로 (c_k=c_k^-)를 선택하면

$$\boxed{R(f,P)=L(f,P)}$$

가 된다.

## (b) (f)가 연속일 필요가 없는 경우

연속이 아닐 때에는 상한 (M_k)가 실제 함수값으로 달성되지 않을 수 있다. 그러나 상한의 정의에 따라 임의의 (\eta>0)에 대하여 어떤 (c_k\in I_k)가 존재하여

$$M_k-\eta<f(c_k)\le M_k$$

이다.

(\epsilon>0)이 주어졌다고 하고

$$\eta=\frac{\epsilon}{b-a}$$

로 둔다. 각 부분구간에서 위 조건을 만족하도록 태그 (c_k)를 선택하면

$$0\le M_k-f(c_k)<\eta.$$

따라서

$$\begin{aligned}
0
&\le U(f,P)-R(f,P)\\
&=\sum_{k=1}^n\bigl(M_k-f(c_k)\bigr)\Delta x_k\\
&<\eta\sum_{k=1}^n\Delta x_k\\
&=\eta(b-a)=\epsilon.
\end{aligned}$$

그러므로

$$\boxed{U(f,P)-R(f,P)<\epsilon}$$

가 되도록 태그를 선택할 수 있다.

하한에 대해서도 하한의 정의에 따라 태그 (d_k\in I_k)를

$$m_k\le f(d_k)<m_k+\eta$$

가 되게 선택할 수 있다. 그러면

$$\boxed{R(f,P)-L(f,P)<\epsilon}$$

이다.

# 문제 5: 정리 전체의 증명

다음 명제를 증명한다.

> 유계함수 $f:[a,b]\to\mathbb R$가 리만 적분 가능하고
> $$\int_a^b f=A$$
> 이기 위한 필요충분조건은, 임의의 $\epsilon>0$에 대하여 어떤 $\delta>0$가 존재하여 모든 $\delta$-세분 태그된 분할 $(P,\{c_k\})$에 대해
> $$|R(f,P)-A|<\epsilon$$
> 이 성립하는 것이다.

## $(\Rightarrow)$ 리만 적분 가능하면 리만 합이 적분값으로 수렴한다

$f$가 리만 적분 가능하고

$$I:=\int_a^b f=A$$

라고 하자.

임의의 $\epsilon>0$을 택한다. 다르부 적분가능성 판정법에 따라 다음을 만족하는 분할 $P_\epsilon$이 존재한다.

$$U(f,P_\epsilon)-L(f,P_\epsilon)<\frac{\epsilon}{3}.$$

$P_\epsilon$의 부분구간 개수를 $n$이라 하자. 또한 $f$가 유계이므로 어떤 $M>0$이 존재하여

$$|f(x)|\le M\qquad(x\in[a,b])$$

이다.

다음과 같이 둔다.

$$\delta=\frac{\epsilon}{9nM}.$$

이제 $(P,\{c_k\})$를 임의의 $\delta$-세분 태그된 분할이라 하고

$$P'=P\cup P_\epsilon$$

으로 둔다.

문제 1(b)에 의해

$$U(f,P')-L(f,P')<\frac{\epsilon}{3}.$$

문제 2와 문제 3에 의해

$$L(f,P')-\frac{\epsilon}{3}
<L(f,P)$$

이고

$$U(f,P)<U(f,P')+\frac{\epsilon}{3}$$

이다. 문제 1(a)에 의해

$$L(f,P)\le R(f,P)\le U(f,P)$$

이며

$$L(f,P')\le I\le U(f,P')$$

이다.

따라서 $R(f,P)$와 $I$는 모두 다음 열린 구간 안에 들어간다.

$$\left(
L(f,P')-\frac{\epsilon}{3},
U(f,P')+\frac{\epsilon}{3}
\right).$$

이 구간의 길이는

$$\begin{aligned}
\left(U(f,P')+\frac{\epsilon}{3}\right)
-\left(L(f,P')-\frac{\epsilon}{3}\right)
&=U(f,P')-L(f,P')+\frac{2\epsilon}{3}\\
&<\frac{\epsilon}{3}+\frac{2\epsilon}{3}\\
&=\epsilon.
\end{aligned}$$

따라서 두 점 $R(f,P)$와 $I$ 사이의 거리는 $\epsilon$보다 작다. 즉,

$$|R(f,P)-I|<\epsilon.$$

$I=A$이므로

$$\boxed{|R(f,P)-A|<\epsilon}$$

이다.

## $(\Leftarrow)$ 모든 충분히 세분된 리만 합이 $A$에 가까우면 적분 가능하다

다음 조건을 가정한다.

$$\forall\epsilon>0;\exists\delta>0:
\quad
P\text{가 }\delta\text{-세분이면 }
|R(f,P)-A|<\epsilon$$

가 모든 태그 선택에 대해 성립한다.

$f$가 리만 적분 가능함을 보이기 위해 임의의 $\epsilon>0$을 택한다. 가정에서 오차를 $\epsilon/4$로 적용하면 어떤 $\delta>0$이 존재하여 모든 $\delta$-세분 태그된 분할에 대해

$$|R(f,P)-A|<\frac{\epsilon}{4}$$

이다.

망의 크기가 $\delta$보다 작은 분할 $P$를 하나 선택한다. 예를 들어 충분히 큰 자연수 $N$에 대하여

$$\frac{b-a}{N}<\delta$$

가 되게 하고, $[a,b]$을 $N$등분하면 된다.

문제 4(b)에 의해 다음을 만족하는 태그를 선택할 수 있다.

$$U(f,P)-R^+(f,P)<\frac{\epsilon}{4},$$

$$R^-(f,P)-L(f,P)<\frac{\epsilon}{4}.$$

여기서 $R^+$는 상합에 가까운 태그를 이용한 리만 합이고, $R^-$는 하합에 가까운 태그를 이용한 리만 합이다.

두 리만 합은 같은 $\delta$-세분 분할에서 만들어졌으므로 가정에 의해

$$|R^+(f,P)-A|<\frac{\epsilon}{4},$$

$$|R^-(f,P)-A|<\frac{\epsilon}{4}$$

이다. 따라서

$$\begin{aligned}
U(f,P)-L(f,P)
={}&[U(f,P)-R^+(f,P)]\\
&+[R^+(f,P)-A]\\
&+[A-R^-(f,P)]\\
&+[R^-(f,P)-L(f,P)].
\end{aligned}$$

중간의 두 항은 음수일 수도 있으므로 절댓값으로 위에서 추정하면

$$\begin{aligned}
U(f,P)-L(f,P)
\le{}&
[U(f,P)-R^+(f,P)]\\
&+|R^+(f,P)-A|\\
&+|A-R^-(f,P)|\\
&+[R^-(f,P)-L(f,P)]\\
<&\frac{\epsilon}{4}
+\frac{\epsilon}{4}
+\frac{\epsilon}{4}
+\frac{\epsilon}{4}\\
=&\epsilon.
\end{aligned}$$

따라서 임의의 $\epsilon>0$에 대하여

$$U(f,P)-L(f,P)<\epsilon$$

인 분할 $P$가 존재한다. 다르부 적분가능성 판정법에 의해 $f$는 리만 적분 가능하다.

이제 그 적분값을

$$I=\int_a^b f$$

라고 하자. $I=A$임을 보여야 한다.

이미 증명한 $(\Rightarrow)$에 의해 임의의 $\eta>0$에 대하여 충분히 세분된 모든 태그된 분할은

$$|R(f,P)-I|<\frac{\eta}{2}$$

를 만족한다. 한편 처음 가정에 의해서도 충분히 세분된 모든 태그된 분할은

$$|R(f,P)-A|<\frac{\eta}{2}$$

를 만족한다.

두 조건을 동시에 만족할 만큼 세분된 태그된 분할 $P$를 택하면

$$\begin{aligned}
|I-A|
&\le |I-R(f,P)|+|R(f,P)-A|\\
&<\frac{\eta}{2}+\frac{\eta}{2}\\
&=\eta.
\end{aligned}$$

$\eta>0$이 임의이므로

$$I=A.$$

따라서

$$\boxed{\int_a^b f=A}$$

이다. 이로써 필요충분조건의 양방향 증명이 완성된다. $\square$
