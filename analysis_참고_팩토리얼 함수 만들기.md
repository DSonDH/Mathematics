# 지수함수
팩토리얼 $n!$의 n은 자연수에서 주로 얘기된다. 하지만 저 $n$을 $x \in \mathbb R$로 확장하는 것은 직관적이지도 않을 뿐더러, 어떻게 정의해야할 지 불분명하다.

지수함수 $2^n$도 마찬가지. 

일반적으로는 정의역을 순차적으로 확장하여, 역수를 이용해 정의역을 정수로 확장하고, 제곱근으로 유리수로 확장하고 마지막으로 연속성으로 실수로 확장하는 전략이 있다.

또 다른 접근법은, $e^x$를 먼저 적절하게 정의하는 것이다. $E(x) = \sum_{n=0}^\infty \frac{x^n}{n!}$을 정의하고 이게 지수함수와 관련있는 모든 성질을 만족함을 보이는 방법이다.

1. $\forall x, y \in \mathbb R$, $E(x+y) = E(x)E(y)$이다.
2. $E(0)=1, E(-x)=1/E(x)$이고  $\forall x \in \mathbb R$, $E(x)>0$이다.
3. $e = E(1)$이라 정의하면  $\forall m, n \in \mathbb Z$, $E(n) = e^n$이고 $ E(m/n) = (\sqrt[n] e)^m$이다.
4. 함수 $f:[a,\infty] \to \mathbb R$을 생각하자. $\forall \epsilon >0$에 대해 "$x \ge M$이면 항상 $|f(x)-L|<\epsilon$이다" 를 만족하는 $M>a$이 존재하면 $\lim_{x\to\infty}f(x) = L$ 이라 한다.


## 문제 5

모든 (n=0,1,2,\ldots)에 대하여 $\lim_{x\to\infty}x^ne^{-x}=0$임을 증명하라.

$x\geq0$ 이면 지수함수의 급수

$$e^x=1+x+\frac{x^2}{2!}+\cdots$$

의 모든 항이 음이 아니다. 따라서 이 급수에서 $n+1$차 항 하나만 취해도

$$e^x\geq \frac{x^{n+1}}{(n+1)!}$$

이다.

$x>0$일 때 양변이 양수이므로 역수를 취하면

$$e^{-x}=\frac1{e^x} \leq\frac{(n+1)!}{x^{n+1}}.$$

여기에 $x^n$을 곱하면

$$0\leq x^ne^{-x} \leq\frac{(n+1)!}{x}.$$

그런데

$$\lim_{x\to\infty}\frac{(n+1)!}{x}=0$$

이므로 조임정리에 의해

$$\boxed{\lim_{x\to\infty}x^ne^{-x}=0}$$

이다.

이 결과는 지수함수 $e^x$가 어떤 고정된 차수의 다항함수 $x^n$보다도 빠르게 증가한다는 것을 의미한다.

## 문제 6

(a) $e^x$가 역함수 $\log x$를 가지는 이유?

역함수를 가지려면 $e^x:\mathbb R\longrightarrow(0,\infty)$ 가 일대일대응임을 보이면 된다.

1. $e^x$는 일대일함수다

앞의 문제에서 $$e^x$'=e^x$ 이고, 모든 $x\in\mathbb R$에 대해 $e^x>0$임을 확인했다. 따라서 $e^x$'>0$ 이므로 $e^x$는 $\mathbb R$에서 엄격히 증가한다. 엄격히 증가하는 함수는 일대일함수다.

2. $e^x$의 치역은 $(0,\infty)$이다

모든 $x$ 에 대해 $e^x>0$이므로 치역은 $(0,\infty)$의 부분집합이다. 한편 $x\geq0$ 이면 급수 정의에서

$$e^x=1+x+\frac{x^2}{2!}+\cdots\geq 1+x$$

이므로 $\lim_{x\to\infty}e^x=\infty.$ 또한 $e^{-x}=1/e^x$이므로

$$\lim_{x\to-\infty}e^x = \lim_{t\to\infty}e^{-t} = \lim_{t\to\infty}\frac1{e^t} =0.$$

$e^x$는 연속이고 엄격히 증가하면서 0부터 $\infty$까지의 모든 값을 지나므로 치역은 정확히 $(0,\infty)$ 이다. 따라서

$$e^x:\mathbb R\to(0,\infty)$$

는 일대일대응이며 역함수를 가진다. 이 역함수를

$$\log:(0,\infty)\to\mathbb R$$

라고 정의한다.

역함수의 정의에 따라 다음 두 식이 성립한다.

$$\boxed{\log(e^y)=y\quad(y\in\mathbb R)}$$

및

$$\boxed{e^{\log x}=x\quad (x>0)}$$

이다.

두 식의 차이는 정의역에 있다.

* (\log(e^y)=y)에서는 (y)가 임의의 실수다.
* (e^{\log x}=x)에서는 $\log x$가 정의되어야 하므로 $x>0$이어야 한다.

(b) $$$\log x$'=1/x$$임을 증명하라

다음 항등식에서 출발한다. $e^{\log x}=x,\quad x>0.$ 양변을 $x$에 관해 미분하면 연쇄법칙에 의해 $e^{\log x}$\log x$'=1.$ 그런데 $e^{\log x}=x$이므로 $x$\log x$'=1.$ 따라서

$$\boxed{$\log x$'=\frac1x\quad (x>0)}$$

역함수의 미분 공식을 직접 적용해도 같은 결과를 얻는다. $y=e^x$의 역함수가 $x=\log y$이므로

$$(\log y)'
=\frac{1}{(e^x)'\big|_{x=\log y}}
=\frac{1}{e^{\log y}}
=\frac{1}{y}$$

(c) $\log(xy)=\log x+\log y$임을 확인하라

$y>0$을 고정하고 다음 함수를 생각한다. $F(x)=\log(xy)-\log x,\quad (x>0).$ $y$는 상수이므로 연쇄법칙에 따라

$$\frac{d}{dx}\log(xy)
=\frac1{xy}\frac{d}{dx}(xy)
=\frac1{xy}\cdot y
=\frac1x.$$

따라서

$$F'(x)
=\frac{d}{dx}\log(xy)-\frac{d}{dx}\log x
=\frac1x-\frac1x
=0.$$

정의역 $(0,\infty)$에서 도함수가 항상 $0$이므로 $F(x)$는 상수함수다. 그 상수를 구하기 위해 $x=1$을 대입하면 $F(1)=\log y-\log1.$

그런데 $e^0=1$ 이므로 $\log1=0.$ 따라서 $F(1)=\log y.$ 즉, 모든 $x>0$에 대해 $F(x)=\log y$ 이므로 $\log(xy)-\log x=\log y.$ 따라서

$$\boxed{\log(xy)=\log x+\log y\quad(x,y>0)}$$

(d) $t^n=e^{n\log t}$임을 보여라

$t>0$, $n\in\mathbb N$이라고 하자. 문제 3에서 보인 지수법칙 $e^{a+b}=e^ae^b$ 을 반복해서 적용하면

$$e^{n\log t} =e^{\log t+\log t+\cdots+\log t}\\ =\underbrace{e^{\log t}e^{\log t}\cdots e^{\log t}}_{n\text{개}}.$$

역함수 관계에 의해 $e^{\log t}=t$이므로

$$e^{n\log t}
=\underbrace{t\cdot t\cdots t}_{n\text{개}}
=t^n.$$

따라서

$$\boxed{t^n=e^{n\log t}\quad(t>0,\ n\in\mathbb N)} \tag 2$$

이다.

$n=0$을 자연수에 포함하는 정의를 사용해도 $t^0=1=e^0=e^{0\log t}$ 이므로 동일한 식이 성립한다.

>위 6번(d)의 $n$을 $x \in \mathbb R$로 바꿔도 우변의 식은 유효하다. 즉 항등식 (2)로 실수 전체에서 $t^x$를 정의할 수 있다.

## 정의. 일반화된 지수함수
$t >0$이 주어질 때 지수함수를 $t^x = e^{xlogt} \quad (x\in \mathbb R)$로 정의한다.

> 지수함수 예와 같은 논리로 일반화된 팩토리얼 $x!$에 대한 정의를 하기는 더 어렵지만 가능하다.


# 함수방정식과 이상적분

analysis_04_1_RiemannIntegral.md에 '적분 기호 속 미분'에 해당하는 내용 참고.

$$ \frac{n!}{\alpha^{n+1}} = \int_0^\infty t^n e^{-\alpha t}\ dt \quad (\alpha >0)$$

임을 유도할 수 있다. 이 등식에서 $\alpha =1$로 두면

$$ n! = \int_0^\infty t^n e^{-t}\ dt$$

가 되고 여기서 $n$은 실변수 $x$로 바꿔도 된다 ($x \ge 0$)

$$ x! = \int_0^\infty t^x e^{-t}\ dt, \quad x \ge 0$$

이렇게 팩토리얼 함수(factorial function) 또는 계승함수를 정의할 수 있다.


### 실수 팩토리얼 함수의 성질
1. 무한번 미분 가능성: $x>0$에서

$$
\frac{d^n}{dx^n}(x!) = \int_0^\infty t^x(\log t)^ne^{-t}\ dt
$$

이다. 특히

$$
(x!)'' = \int_0^\infty t^x(\log t)^2e^{-t}\ dt>0
$$

이므로 $x!$은 엄격한 볼록함수다.

2. $x!$이 다음 함수방정식을 만족한다:

$$(x+1)! = (x+1)x!$$

>**증명**
>
>1. 무한번 미분 가능성
>
>$t>0$일 때 $t^x=e^{x\log t}$ 이다. 따라서 $x$에 관하여 미분하면
>
>$$
>\frac{\partial}{\partial x}t^x= t^x\log t,\quad \frac{\partial^2}{\partial x^2}t^x= t^x(\log t)^2,\quad \frac{\partial^n}{\partial x^n}t^x= t^x(\log t)^n
>$$
>
>이다. 피적분함수 전체를 미분하면
>
>$$
>\frac{\partial^n}{\partial x^n} \left(t^xe^{-t}\right)= t^x(\log t)^ne^{-t}.
>$$
>
>따라서 적분기호 속 미분이 가능하다면
>
>$$
>\boxed{ \frac{d^n}{dx^n}(x!)= \int_0^\infty t^x(\log t)^ne^{-t}\ dt }
>$$
>
>2. 적분기호 속 미분의 정당화
>
>임의의 콤팩트 구간 $[u,v]\subset(0,\infty)$ 를 생각한다. 즉, $0<u\le x\le v$ 라고 하자. 적분구간을 $(0,1]$과 $[1,\infty)$로 나누어 생각한다.
>
>- $0<t\le1$인 경우
>
>$0<t\le1$ 에서는 지수가 커질수록 $t^x$가 작아진다. $x\ge u$이므로 $t^x\le t^u.$ 따라서
>
>$$
>t^x|\log t|^ne^{-t} \le t^u|\log t|^n.
>$$
>
>그리고
>
>$$
>\int_0^1t^u|\log t|^n\ dt<\infty
>$$
>
>이다. 실제로 $t=e^{-s}$로 치환하면
>
>$$
>\int_0^1t^u|\log t|^n\ dt= \int_0^\infty s^ne^{-(u+1)s},ds,
>$$
>
>이고 문제 19의 결과에 의해 이 적분은 수렴한다.
>
>- $t\ge1$인 경우
>
>$t\ge1$에서는 $x\le v$이므로 $t^x\le t^v$. 따라서
>
>$$
>t^x|\log t|^ne^{-t}
>\le
>t^v(\log t)^ne^{-t}.
>$$
>
>$t\to\infty$일 때 지수함수 $e^t$가 $t^v(\log t)^n$보다 빠르게 증가하므로
>
>$$
>\int_1^\infty t^v(\log t)^ne^{-t}\ dt<\infty.
>$$
>
>따라서 각 $n$에 대해 적분 가능한 지배함수가 존재한다. 바이어슈트라스 $M$-판정법과 정리 8.4.9를 반복해서 적용할 수 있으므로 $x!$은 $(0,\infty)$에서 무한번 미분 가능하다.
>
>결론적으로
>
>$$
>\boxed{(x!)^{(n)}= \int_0^\infty t^x(\log t)^ne^{-t}\ dt }
>$$
>
>3. $(x!)''>0$임을 증명
>
>$n=2$를 대입하면
>
>$$
>(x!)''= \int_0^\infty
>t^x(\log t)^2e^{-t}\ dt.
>$$
>
>$t>0$에서 $t^x>0, \quad e^{-t}>0, \quad(\log t)^2\ge0$ 이므로 피적분함수는 항상 음이 아니다. 또한 $(\log t)^2=0$인 점은 $t=1$ 하나뿐이다. $t\neq1$ 에서는 $t^x(\log t)^2e^{-t}>0$
>
>따라서 양의 길이를 가진 구간에서 피적분함수가 양수이므로 전체 적분은 양수다. 그러므로 팩토리얼 함수 $x!$은 $(0,\infty)$에서 엄격한 볼록함수다.
>
>4. 팩토리얼 함수의 함수방정식
>
>정의에 의해 $(x+1)!= \int_0^\infty t^{x+1}e^{-t}\ dt$  
>먼저 유한한 구간 $[0,b]$에서 부분적분을 적용한다. 다음과 같이 둔다.
>
>$$
>u=t^{x+1}, \quad dv=e^{-t}\ dt.
>$$
>
>그러면 $du=(x+1)t^x\ dt, \quad v=-e^{-t}.$ 따라서
>
>$$
>\int_0^b t^{x+1}e^{-t}\ dt =
>\left[-t^{x+1}e^{-t}\right]_0^b + (x+1)\int_0^b t^xe^{-t}\ dt 
>\\ = -b^{x+1}e^{-b} + (x+1)\int_0^b t^xe^{-t}\ dt.
>$$
>
>여기서 $x\ge0$이므로 $\lim_{t\to0^+}t^{x+1}e^{-t}=0$. 또한 지수함수가 거듭제곱함수보다 빠르게 증가하므로 $\lim_{b\to\infty}b^{x+1}e^{-b}=0.$ 따라서 $b\to\infty$로 보내면
>
>$$
>(x+1)! =
>(x+1)\int_0^\infty t^xe^{-t}\ dt\ =(x+1)x!.
>$$
>
>결론적으로
>
>$$
>\boxed{(x+1)!=(x+1)x!}
>$$
>
>이다.
>
>또한, $0!= \int_0^\infty e^{-t}\ dt =1$ 이므로 이 함수방정식을 반복하면 모든 자연수 $n$에 대해 $n!=n(n-1)\cdots2\cdot1$ 을 얻는다. 따라서 이상적분으로 정의한 실수 팩토리얼 함수는 기존 자연수 팩토리얼의 정의와 정확히 일치한다.

### 단순한 볼록성과 로그볼록성의 차이
양의 함수 $f$가 로그볼록이라는 것은 $\log f$ 가 볼록함수라는 뜻이다. 동치로 $0\le\lambda\le1$일 때

$$
f\bigl((1-\lambda)x+\lambda y\bigr) \le f(x)^{1-\lambda}f(y)^\lambda
$$

가 성립한다. 로그볼록성은 단순한 볼록성보다 강한 성질이다. 양의 볼록함수라고 해서 항상 로그볼록인 것은 아니다.

>**팩토리얼 함수가 로그볼록인 이유**
>
>다음을 놓는다.
>
>$$
>F(x)=x!=\int_0^\infty t^xe^{-t}\ dt.
>$$
>
>도함수는
>
>$$
>F'(x)=\int_0^\infty t^x\log t,e^{-t}\ dt, \quad F''(x)=\int_0^\infty t^x(\log t)^2e^{-t}\ dt
>$$
>
>$\log F(x)$의 이계도함수는
>
>$$
>(\log F(x))'' = \frac{F(x)F''(x)-F'(x)^2}{F(x)^2}
>$$
>
>이다. 이제 가중치 $w_x(t)=t^xe^{-t}>0$ 를 사용하여 코시–슈바르츠 부등식을 적용하면
>
>$$
>\left( \int_0^\infty w_x(t)\log t\ dt \right)^2 \le
>\left( \int_0^\infty w_x(t)\ dt \right)
>\left( \int_0^\infty w_x(t)(\log t)^2\ dt \right).
>$$
>
>즉,
>
>$$
>F'(x)^2\le F(x)F''(x).
>$$
>
>따라서
>
>$$
>(\log F(x))''\ge0.
>$$
>
>그러므로
>
>$$
>\boxed{\log(x!)\text{는 볼록함수다.}}
>$$

### 정리 8.4.11. 보어–몰레럽 정리

$x\ge0$에서 정의된 양의 함수 $f$ 가운데 다음 세 조건을 모두 만족하는 함수는 유일하다.

1. 기본값 조건: $f(0)=1.$

2. 팩토리얼 함수방정식: $f(x+1)=(x+1)f(x).$

3. 로그볼록성: $\log f(x)$ 는 볼록함수다.

이 유일한 함수는

$$
f(x)=x!
$$

이다.

함수방정식과 $f(0)=1$만으로는 자연수가 아닌 점에서 함수의 값을 유일하게 정할 수 없다. 로그볼록성이 그러한 비정상적인 확장들을 제거하는 조건이다.

>**증명**  
>
>양의 함수 $f$가 보어–몰레럽 정리의 조건을 만족한다고 하자. 즉, $f(0)=1$, $f(x+1)=(x+1)f(x)$ 이고 $\log f(x)$가 볼록함수라고 하자. 자연수 $n$과 $x\in(0,1]$를 고정한다.
>
>---
>
>(a) $\log f(x)$가 볼록함수라는 사실과 세 구간 $[n-1,n],\quad[n,n+x],\quad[n,n+1]$ 을 이용하여 다음을 보여라.
>
>$$
>x\log n \le \log f(n+x)-\log(n!) \le x\log(n+1).
>$$
>
>
>먼저 함수방정식과 $f(0)=1$에 의해 모든 자연수 $n$에 대해 $f(n)=n!$ 이다.
>
>다음을 놓는다. $\phi(u)=\log f(u)$
>
>$\phi$는 볼록함수이므로 현의 기울기가 증가한다. 구간 $[n-1,n],\quad[n,n+x],\quad[n,n+1]$ 을 순서대로 비교하면
>
>$$
>\frac{\phi(n)-\phi(n-1)}{n-(n-1)}
>\le \frac{\phi(n+x)-\phi(n)}{x}
>\le \frac{\phi(n+1)-\phi(n)}{(n+1)-n}.
>$$
>
>양 끝의 분모는 $1$이므로
>
>$$
>\phi(n)-\phi(n-1)
>\le \frac{\phi(n+x)-\phi(n)}x
>\le \phi(n+1)-\phi(n).
>$$
>
>$f(n)=n!$을 이용하면
>
>$$
>\phi(n)-\phi(n-1) =\log(n!)-\log((n-1)!)\ =\log n,
>$$
>
>이고
>
>$$
>\phi(n+1)-\phi(n) =\log((n+1)!)-\log(n!)\ =\log(n+1).
>$$
>
>따라서
>
>$$
>\log n
>\le
>\frac{\log f(n+x)-\log f(n)}x
>\le
>\log(n+1).
>$$
>
>양변에 $x>0$을 곱하고 $f(n)=n!$을 사용하면
>
>$$
>\boxed{
>x\log n
>\le
>\log f(n+x)-\log(n!)
>\le
>x\log(n+1)
>}
>$$
>
>---
>
>(b) 다음을 보여라.
>
>$$
>\log f(n+x) = \log f(x) +
>\log\bigl((x+1)(x+2)\cdots(x+n)\bigr).
>$$
>
>함수방정식을 반복해서 적용하면
>
>$$
>f(x+1) =(x+1)f(x),\quad f(x+2) =(x+2)f(x+1),\ =(x+2)(x+1)f(x),\\
>\vdots\\
>f(x+n) =(x+n)(x+n-1)\cdots(x+1)f(x).
>$$
>
>따라서
>
>$$
>f(n+x) = f(x)(x+1)(x+2)\cdots(x+n).
>$$
>
>모든 항이 양수이므로 양변에 로그를 취할 수 있다.
>
>$$
>\boxed{
>\log f(n+x) = \log f(x) + \log\bigl((x+1)(x+2)\cdots(x+n)\bigr)
>}
>$$
>
>---
>
>(c) 다음을 보여라.
>
>$$
>0 \le
>\log f(x) - \log\left( \frac{n^xn!}{(x+1)(x+2)\cdots(x+n)} \right)
>\le x\log\left(1+\frac1n\right).
>$$
>
>다음을 간단히 표기한다. $P_n(x)=(x+1)(x+2)\cdots(x+n).$ 그러면 문제 21(b)에 의해 $\log f(n+x) = \log f(x)+\log P_n(x)$ 이다. 이를 문제 21(a)의 부등식에 대입하면
>
>$$
>x\log n
>\le \log f(x)+\log P_n(x)-\log(n!)
>\le x\log(n+1).
>$$
>
>왼쪽 부등식에서
>
>$$
>\log f(x) \ge x\log n+\log(n!)-\log P_n(x).
>$$
>
>그런데 $x\log n+\log(n!)-\log P_n(x) = \log\left(\frac{n^xn!}{P_n(x)}\right)$ 이므로
>
>$$
>\log f(x) - \log\left(\frac{n^xn!}{P_n(x)}\right)
>\ge0.
>$$
>
>오른쪽 부등식에서는
>
>$$
>\log f(x)
>\le
>x\log(n+1)+\log(n!)-\log P_n(x).
>$$
>
>여기서
>
>$$
>x\log(n+1) = x\log n+x\log\left(1+\frac1n\right)
>$$
>
>이므로
>
>$$
>\log f(x) \le \log\left(\frac{n^xn!}{P_n(x)}\right)
>+ x\log\left(1+\frac1n\right).
>$$
>
>따라서
>
>$$
>\log f(x) - \log\left(\frac{n^xn!}{P_n(x)}\right)
>\le x\log\left(1+\frac1n\right).
>$$
>
>두 부등식을 합하면
>
>$$
>\boxed{
>0 \le
>\log f(x) - \log\left(
>\frac{n^xn!} {(x+1)(x+2)\cdots(x+n)}
>\right)
>\le x\log\left(1+\frac1n\right)
>}
>$$
>
>---
>
>문제 21(d) 다음을 보여라.
>
>$$
>f(x) = \lim_{n\to\infty}
>\frac{n^xn!}
>{(x+1)(x+2)\cdots(x+n)},
>\quad x\in(0,1].
>$$
>
>문제 21(c)에 의해
>
>$$
>0 \le \log f(x) - \log\left(
>\frac{n^xn!}
>{(x+1)(x+2)\cdots(x+n)}
>\right)
>\le x\log\left(1+\frac1n\right).
>$$
>
>$n\to\infty$이면 $\log\left(1+\frac1n\right)\to0$ 이므로 $x\log\left(1+\frac1n\right)\to0.$ 조임정리에 의해
>
>$$
>\log f(x) - \log\left( \frac{n^xn!} {(x+1)(x+2)\cdots(x+n)} \right) \to0.
>$$
>
>따라서
>
>$$
>\log\left(
>\frac{n^xn!}
>{(x+1)(x+2)\cdots(x+n)}
>\right)
>\to\log f(x).
>$$
>
>지수함수의 연속성을 이용하면
>
>$$
>\boxed{
>f(x) = \lim_{n\to\infty} \frac{n^xn!} {(x+1)(x+2)\cdots(x+n)}
>}
>$$
>
>이 식은 우변이 $f$와 무관하게 오직 $x$만으로 결정된다는 점에서 중요하다.
>
>---
>
>문제 21(e): 문제 21(d)의 결과가 모든 $x\ge0$에 대해 성립함을 보여라.
>
>다음을 정의한다.
>
>$$
>A_n(x) = \frac{n^xn!} {(x+1)(x+2)\cdots(x+n)}.
>$$
>
>문제 21(d)에 의해 $A_n(x)\to f(x)$ 가 $x\in(0,1]$에서 성립한다.
>
>먼저 $x=0$이면 $A_n(0) = \frac{n!}{1\cdot2\cdots n} =1=f(0).$ 따라서 $x=0$에서도 성립한다.
>
>이제 $A_n(x+1)$과 $A_n(x)$의 관계를 계산한다.
>
>$$
>A_n(x+1) = \frac{n^{x+1}n!} {(x+2)(x+3)\cdots(x+n+1)}.
>$$
>
>따라서
>
>$$
>\frac{A_n(x+1)}{A_n(x)} =
>n,
>\frac{(x+1)(x+2)\cdots(x+n)}
>{(x+2)(x+3)\cdots(x+n+1)}\ =
>\frac{n(x+1)}{x+n+1}.
>$$
>
>즉,
>
>$$
>A_n(x+1) = \frac{n(x+1)}{x+n+1}A_n(x).
>$$
>
>$n\to\infty$이면 $\frac{n(x+1)}{x+n+1}\to x+1.$ 만약 $A_n(x)\to f(x)$라면
>
>$$
>A_n(x+1) \to(x+1)f(x)\ =f(x+1)
>$$
>
>이다. 마지막 등식은 함수방정식에서 나온다.
>
>따라서 어떤 $x$에서 공식이 성립하면 $x+1$에서도 성립한다. 이미 $x\in[0,1]$에서 성립하므로 이를 반복하면 모든 $x\ge0$에서 성립한다.
>
>결론적으로
>
>$$
>\boxed{
>f(x) = \lim_{n\to\infty}
>\frac{n^xn!}
>{(x+1)(x+2)\cdots(x+n)}
>\quad(x\ge0)
>}
>$$
>
>이다.
>- 이 식은 가우스 곱셈 공식 (Gauss product formula)라 불리는 팩토리얼 함수에 대한 대체 표현이다.

#### 음의 실수로의 확장

적분식

$$
x!=\int_0^\infty t^xe^{-t}\ dt
$$

은 실제로 $-1<x<0$ 에서도 수렴한다.

$t\to0^+$일 때 핵심 부분은 $t^x$이며,

$$
\int_0^1t^x\ dt = \frac1{x+1}
$$

은 정확히 $x>-1$일 때 수렴하기 때문이다. 따라서 적분 정의로 $x!>0,\quad -1<x<0$ 까지 확장할 수 있다.

더 작은 음의 실수에서는 함수방정식 $(x+1)!=(x+1)x!$ 을 역으로 사용한다.

$$
x!=\frac{(x+1)!}{x+1}.
$$

이를 반복하면 양의 정수가 아닌 모든 음의 실수로 팩토리얼 함수를 확장할 수 있다.

다만 $x=-1,-2,-3,\ldots$ 에서는 분모가 $0$이 되므로 함수가 정의되지 않고 수직점근선을 갖는다.

또한 이 확장함수는 절대로 $0$이 되지 않으며 부호가 구간마다 번갈아 나타난다. 일반적으로 $m=0,1,2,\ldots$에 대해

$$
\operatorname{sgn}(x!)=(-1)^m,
\quad
-m-1<x<-m
$$

이다.

이렇게 확장된 팩토리얼 함수를 $x$축 방향으로 한 칸 이동하고 표기법을 바꾸면, 바로 다음 내용인 감마함수로 이어진다.


#### 예제. 문제 22
(a) 다음 함수가 $0$이 되는 곳을 구하고, 같은 근을 가지는 익숙한 함수를 찾아라.

$$
g(x)=\frac{x}{x!(-x)!}.
$$

(b) 가우스 적분

$$
\int_{-\infty}^{\infty}e^{-x^2} dx=\sqrt{\pi}
$$

를 이용하여 $(\frac12)!$ 의 값을 구하라.

(c) (a), (b)의 결과를 이용하여 팩토리얼 함수와 삼각함수 사이의 관계를 추측하라.

(a)

실수로 확장된 팩토리얼 함수 $x!$은 다음 음의 정수에서 수직점근선을 갖는다. $x=-1,-2,-3,\ldots$ 즉, 형식적으로 그 역수는 이 점들에서 $0$이 된다.

양의 정수에서: $x=n$이 양의 정수이면 $(-x)!=(-n)!$은 수직점근선을 갖는다. 따라서 그 역수는 $0$이 되고 $g(n)=0,\quad n=1,2,3,\ldots$ 으로 연속적으로 확장할 수 있다.

음의 정수에서: $x=-n$이 음의 정수이면 $x!=(-n)!$이 수직점근선을 가지므로 $g(-n)=0,\quad n=1,2,3,\ldots$ 으로 확장된다.

$x=0$에서 $0!=1$ 이므로 $g(0)=\frac0{0!0!}=0.$ 따라서 $g$의 근은 모든 정수다. $\boxed{x\in\mathbb Z}$

모든 정수에서 근을 갖는 익숙한 삼각함수는 $\sin(\pi x)$ 이다. 따라서 $g(x)$와 $\sin(\pi x)$ 사이에 상수배 관계가 있을 것으로 추측할 수 있다.

(b) $\left(\frac12\right)!$ 계산

정의에 의해

$$
\left(\frac12\right)!= \int_0^\infty t^{1/2}e^{-t},dt.
$$

다음과 같이 치환한다. $t=u^2,\quad dt=2u,du.$ $t^{1/2}=u$이므로

$$
\left(\frac12\right)!
= \int_0^\infty u e^{-u^2}(2u),du\
= 2\int_0^\infty u^2e^{-u^2}\ du.
$$

이제 $J=\int_0^\infty u^2e^{-u^2}\ du$ 를 계산한다.

다음 관계를 이용한다. $\frac{d}{du}e^{-u^2}=-2ue^{-u^2}.$ 따라서 $ue^{-u^2}\ du=-\frac12d(e^{-u^2}).$  
부분적분하면

$$
J
=\int_0^\infty u\left(ue^{-u^2}\right),du\ =-\frac12\int_0^\infty u,d(e^{-u^2})\ =-\frac12\left[ue^{-u^2}\right]_0^\infty +\frac12\int_0^\infty e^{-u^2}\ du.
$$

지수함수가 다항함수보다 빠르게 감소하므로 $\lim_{u\to\infty}ue^{-u^2}=0.$ 
또한 $u=0$에서도 $ue^{-u^2}=0$이므로 경계항은 $0$이다. 따라서

$$
J=\frac12\int_0^\infty e^{-u^2}\ du.
$$

$e^{-u^2}$은 우함수이고

$$
\int_{-\infty}^{\infty}e^{-u^2}\ du=\sqrt{\pi}
$$

이므로

$$
\int_0^\infty e^{-u^2}\ du=\frac{\sqrt{\pi}}2.
$$

따라서 $J=\frac12\cdot\frac{\sqrt{\pi}}2 = \frac{\sqrt{\pi}}4.$

결국 $\left(\frac12\right)! =2J\ =2\cdot\frac{\sqrt{\pi}}4\ =\frac{\sqrt{\pi}}2.$

(c) 팩토리얼 함수와 삼각함수의 관계 추측

팩토리얼 함수의 함수방정식 $(x+1)!=(x+1)x!$ 에 $x=-1/2$를 대입하면 $(\frac12)!= \frac12 (-\frac12)!$

따라서 $\left(-\frac12\right)!= 2\left(\frac12\right)! = \sqrt{\pi}.$

이제 $g(1/2)$를 계산하면

$$
g\left(\frac12\right)
= \frac{\frac12}
{\left(\frac12\right)!\left(-\frac12\right)!}\
= \frac{\frac12}
{\left(\frac{\sqrt{\pi}}2\right)\sqrt{\pi}}\
= \frac{\frac12}{\frac{\pi}{2}}\
=\frac1\pi. $$

한편 $\sin\left(\pi\cdot\frac12\right)=1.$ 따라서 $g(x)$가 $\sin(\pi x)$의 상수배라고 추측하면 그 상수는 $1/\pi$여야 한다.

즉,

$$
\boxed{
\frac{x}{x!(-x)!}= \frac{\sin(\pi x)}{\pi}
}
$$

라는 관계를 추측할 수 있다.

이를 정리하면

$$
\boxed{
x!(-x)!= \frac{\pi x}{\sin(\pi x)}
}
$$

또는

$$
\boxed{
\frac1{x!(-x)!}= \frac{\sin(\pi x)}{\pi x}
}
$$

이다.

이 식은 감마함수에 대한 오일러의 반사공식에 해당한다.

다만 두 함수가 같은 근을 가지고 $x=1/2$에서 같은 값을 갖는다는 사실만으로 두 함수가 완전히 같다는 것이 증명되지는 않는다. 문제에서 요구하는 것은 이 관계의 추측이며, 엄밀한 증명에는 추가적인 이론이 필요하다.

#### 예제. 문제 23
문제 22에서 구한 $\left(\frac12\right)!=\frac{\sqrt{\pi}}2$ 와 가우스 곱셈 공식 $x!= \lim_{n\to\infty} \frac{n^xn!} {(x+1)(x+2)\cdots(x+n)}$ 을 이용하여 월리스 곱 공식을 유도하라.

$$
\frac{\pi}{2}= \lim_{n\to\infty}
\left(\frac{2\cdot2}{1\cdot3}\right)
\left(\frac{4\cdot4}{3\cdot5}\right)
\left(\frac{6\cdot6}{5\cdot7}\right)
\cdots
\left(
\frac{2n\cdot2n}{(2n-1)(2n+1)}
\right).
$$

**증명**

가우스 곱 공식에 $x=1/2$를 대입하면

$$
\left(\frac12\right)!= \lim_{n\to\infty}
\frac{n^{1/2}n!}
{\left(\frac32\right)
\left(\frac52\right)
\cdots
\left(n+\frac12\right)}.
$$

분모를 정리하면

$$
\left(\frac32\right)
\left(\frac52\right)
\cdots
\left(n+\frac12\right)
= \frac{3\cdot5\cdots(2n+1)}{2^n}.
$$

따라서

$$
\left(\frac12\right)!= \lim_{n\to\infty}
\frac{2^n\sqrt{n},n!}
{3\cdot5\cdots(2n+1)}.
$$

문제 22에서 $\left(\frac12\right)!=\frac{\sqrt{\pi}}2$ 를 얻었으므로

$$
\frac{\sqrt{\pi}}2= \lim_{n\to\infty}
\frac{2^n\sqrt{n},n!}
{3\cdot5\cdots(2n+1)}.
$$

양변을 제곱하면

$$
\frac{\pi}{4}= \lim_{n\to\infty}
\frac{4^n n(n!)^2}
{\left(3\cdot5\cdots(2n+1)\right)^2}.
$$

다음을 놓는다.

$$
A_n= \frac{2^n\sqrt{n},n!} {3\cdot5\cdots(2n+1)}.
$$

그러면

$$
A_n^2= \frac{4^n n(n!)^2}
{\left(3\cdot5\cdots(2n+1)\right)^2}
$$

이고 $A_n^2\longrightarrow\frac{\pi}{4}.$

이제 월리스 곱의 $n$번째 부분곱을 $W_n$이라 하자.

$$
W_n= \prod_{k=1}^n
\frac{(2k)^2}{(2k-1)(2k+1)}.
$$

이를 홀수와 짝수의 곱으로 쓰면

$$
W_n= \frac{(2\cdot4\cdots2n)^2}
{(1\cdot3\cdots(2n-1))(3\cdot5\cdots(2n+1))}.
$$

짝수들의 곱은 $2\cdot4\cdots2n=2^nn!$ 이므로

$$
W_n= \frac{4^n(n!)^2}
{(1\cdot3\cdots(2n-1))(3\cdot5\cdots(2n+1))}.
$$

두 홀수 곱 사이에는 다음 관계가 있다.

$$
3\cdot5\cdots(2n+1)= (2n+1)(1\cdot3\cdots(2n-1)).
$$

따라서

$$
1\cdot3\cdots(2n-1)= \frac{3\cdot5\cdots(2n+1)}{2n+1}.
$$

이를 $W_n$에 대입하면

$$
W_n
= \frac{4^n(n!)^2}
{
\dfrac{
\left(3\cdot5\cdots(2n+1)\right)^2
}{2n+1}
}\
= \frac{4^n(2n+1)(n!)^2}
{\left(3\cdot5\cdots(2n+1)\right)^2}.
$$

$A_n^2$과 비교하면 $W_n= \frac{2n+1}{n}A_n^2.$

이제 $n\to\infty$로 보내면 $\frac{2n+1}{n}\to2$ 이고 $A_n^2\to\frac{\pi}{4}.$

따라서

$$
\lim_{n\to\infty}W_n = 2\cdot\frac{\pi}{4} =\frac{\pi}{2}. $$

결론적으로

$$
\boxed{
\frac{\pi}{2}= \lim_{n\to\infty}
\prod_{k=1}^n
\frac{(2k)^2}{(2k-1)(2k+1)}
}
$$

이다.

이를 전개하면 유명한 월리스 곱 공식이 된다.

$$
\boxed{
\frac{\pi}{2}= \frac{2\cdot2}{1\cdot3}
\cdot
\frac{4\cdot4}{3\cdot5}
\cdot
\frac{6\cdot6}{5\cdot7}
\cdots
}
$$

