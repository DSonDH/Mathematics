# 추가: 순열 조합 공식 정리
## 순열 (Permutation)
$n$개의 서로 다른 원소 중에서 $r$개를 **순서를 고려하여** 선택하는 경우의 수

$$P(n,r) = P_n^r = {}_nP_r = \frac{n!}{(n-r)!} = n(n-1)(n-2)\cdots(n-r+1)$$

**특수한 경우**
- $P(n,0) = 1$ (선택하지 않음)
- $P(n,n) = n!$ (모두 선택)
- $P(n,1) = n$

**같은 원소가 있는 경우**

$n$개의 원소 중 $n_1$개, $n_2$개, $\ldots$, $n_k$개가 각각 같을 때 ($n_1 + n_2 + \cdots + n_k = n$)
$$\frac{n!}{n_1! n_2! \cdots n_k!}$$

## 조합 (Combination)
$n$개의 서로 다른 원소 중에서 $r$개를 **순서를 무시하고** 선택하는 경우의 수
$$
C(n,r) = {}_nC_r = \binom{n}{r} = \frac{n!}{r!(n-r)!} = \frac{n(n-1)(n-2)\cdots(n-r+1)}{r!}
= \frac{P(n,r)}{r!}
= {}_nC_{n-r} = \binom{n}{n-r}
$$

**특수한 경우**
- $\binom{n}{0} = 1$ (선택하지 않음)
- $\binom{n}{n} = 1$ (모두 선택)
- $\binom{n}{1} = n$
- $\binom{n}{r} = \binom{n}{n-r}$ (대칭성)

### 여러 원소가 있는 경우
$n$개의 원소 중 $n_1$개, $n_2$개, $\ldots$, $n_k$개가 각각 같을 때 ($n_1 + n_2 + \cdots + n_k = n$)에서 $r$개를 선택하는 경우의 수는 다음과 같이 계산할 수 있다.    
$$\sum_{\substack{r_1 + r_2 + \cdots + r_k = r \\ 0 \le r_i \le n_i}} \frac{n_1!}{r_1!(n_1-r_1)!} \cdot \frac{n_2!}{r_2!(n_2-r_2)!} \cdots \frac{n_k!}{r_k!(n_k-r_k)!} \\
= \sum_{\substack{r_1 + r_2 + \cdots + r_k = r \\ 0 \le r_i \le n_i}} \binom{n_1}{r_1} \binom{n_2}{r_2} \cdots \binom{n_k}{r_k} \\
= \sum_{\substack{r_1 + r_2 + \cdots + r_k = r \\ 0 \le r_i \le n_i}} \prod_{i=1}^k \binom{n_i}{r_i}$$

### 다항계수
$n$개의 원소를 $k$개의 그룹으로 나누되, 각 그룹에 $n_1, n_2, \ldots, n_k$개씩 배치 ($n_1 + n_2 + \cdots + n_k = n$)하는 경우의 수는 다항계수로 표현할 수 있다.
$$\binom{n}{n_1, n_2, \ldots, n_k} = \frac{n!}{n_1! n_2! \cdots n_k!}$$


## 순열과 조합의 관계
$$P(n,r) = C(n,r) \times r!$$

즉, $n$개 중 $r$개를 조합으로 선택한 후, 그 $r$개를 순열로 배열한다.

## 중복조합 (Combination with Repetition)
$n$개의 원소에서 중복을 허용하여 $r$개를 선택 (순서 무시)

$$H(n,r) = C(n+r-1, r) = \binom{n+r-1}{r} = \frac{(n+r-1)!}{r!(n-1)!}$$

**예시**: 서로 다른 5가지 맛의 아이스크림 중에서 중복을 허용하여 3개를 선택
$$H(5,3) = C(5+3-1,3) = C(7,3) = 35$$

## 파스칼의 항등식 (Pascal's Identity)
$$\binom{n}{r} = \binom{n-1}{r-1} + \binom{n-1}{r}$$

이 관계식은 파스칼의 삼각형(Pascal's Triangle)을 만든다.

## 이항정리 (Binomial Theorem)
$n$이 음이 아닌 정수일 때
$$(x+y)^n = \sum_{r=0}^{n} \binom{n}{r} x^{n-r} y^r$$

**특수한 경우**
- $(1+1)^n = \sum_{r=0}^{n} \binom{n}{r} = 2^n$
- $(1-1)^n = \sum_{r=0}^{n} (-1)^r \binom{n}{r} = 0$ (단, $n > 0$)

## 다항계수 (Multinomial Coefficient)
$n$개의 원소를 $k$개의 그룹으로 나누되, 각 그룹에 $n_1, n_2, \ldots, n_k$개씩 배치 ($n_1 + n_2 + \cdots + n_k = n$)

$$\binom{n}{n_1, n_2, \ldots, n_k} = \frac{n!}{n_1! n_2! \cdots n_k!}$$

**다항정리**
$$(x_1 + x_2 + \cdots + x_k)^n = \sum_{n_1+\cdots+n_k=n} \binom{n}{n_1, \ldots, n_k} x_1^{n_1} \cdots x_k^{n_k}$$

## 원탁순열 (Circular Permutation)
$n$개의 서로 다른 원소를 원탁에 배치하는 경우의 수 (회전으로 같은 것은 하나로 간주)

$$\text{원탁순열} = \frac{n!}{n} = (n-1)!$$

**목걸이 순열 (Necklace Permutation)** - 뒤집기도 같은 것으로 간주
$$\text{목걸이순열} = \frac{(n-1)!}{2} \quad (n \ge 3)$$

## 완전순열 (Derangement) - 앞서 본 짝 짓기 문제에서 사용
$n$개의 서로 다른 원소를 배열할 때, 어떤 원소도 원래 위치에 오지 않는 경우의 수

$$D_n = n! \sum_{k=0}^{n} \frac{(-1)^k}{k!}$$

또는 점화식으로
$$D_n = (n-1)[D_{n-1} + D_{n-2}]$$

초기값: $D_0 = 1$, $D_1 = 0$, $D_2 = 1$

$n$이 충분히 클 때
$$P(\text{완전순열}) = \frac{D_n}{n!} \approx \frac{1}{e} \approx 0.368$$

## 분할과 벨 수 (Partition and Bell Numbers)
$n$개 원소의 집합을 $k$개의 공이 아닌 부분집합으로 나누는 경우의 수: **제2종 스털링 수** $S(n,k)$ 또는 $\left\{\begin{matrix}n\\k\end{matrix}\right\}$

$$S(n,k) = \frac{1}{k!}\sum_{j=0}^{k}(-1)^{k-j}\binom{k}{j}j^n$$

점화식:
$$S(n,k) = k \cdot S(n-1,k) + S(n-1,k-1)$$

**벨 수 (Bell Number)** - 집합을 임의 개수의 부분집합으로 나누는 모든 경우의 수
$$B_n = \sum_{k=0}^{n} S(n,k)$$

초기값: $B_0=1, B_1=1, B_2=2, B_3=5, B_4=15, B_5=52$

## 정리: 빙엄의 정리 (Bing's Theorem) - 초기하분포의 항등식
$$\binom{n}{x}\binom{x}{r} = \binom{n}{r}\binom{n-r}{x-r}$$

또는 다항계수 표기로
$$x(x-1)\cdots(x-r+1)\binom{n}{x} = n(n-1)\cdots(n-r+1)\binom{n-r}{x-r}$$

**증명**  
좌변과 우변을 각각 계산하면

**좌변:**
$$\binom{n}{x}\binom{x}{r} = \frac{n!}{x!(n-x)!} \cdot \frac{x!}{r!(x-r)!} = \frac{n!}{r!(x-r)!(n-x)!}$$

**우변:**
$$\binom{n}{r}\binom{n-r}{x-r} = \frac{n!}{r!(n-r)!} \cdot \frac{(n-r)!}{(x-r)!(n-x)!} = \frac{n!}{r!(x-r)!(n-x)!}$$

두 식이 일치하므로 증명 완료. ∎

**조합론적 해석**  
$n$명 중에서 $r$명의 위원을 선택하고, 그 중에서 다시 $x-r$명을 선택하는 경우의 수는:
1. **좌변의 관점**: 먼저 $n$명 중 $x$명을 선택한 후($\binom{n}{x}$), 그 $x$명 중에서 $r$명을 선택($\binom{x}{r}$)
2. **우변의 관점**: 먼저 $n$명 중 $r$명을 선택한 후($\binom{n}{r}$), 남은 $n-r$명 중에서 $x-r$명을 추가로 선택($\binom{n-r}{x-r}$)

두 방법 모두 같은 대상을 센다.

**Vandermonde의 항등식과의 관계**  
이는 Vandermonde 항등식의 특수한 경우:
$$\binom{m+n}{r} = \sum_{k=0}^{r}\binom{m}{k}\binom{n}{r-k}$$

에서 $m=x$, $n=n-x$, $r=r$로 놓은 것과 관련이 있다.

**초기하분포(Hypergeometric Distribution)에서의 응용**  
모집단 크기 $n$, 성공상태 개수 $K$, 표본크기 $r$일 때, 표본에서 성공이 정확히 $x$번 일어날 확률:
$$P(X=x) = \frac{\binom{K}{x}\binom{n-K}{r-x}}{\binom{n}{r}}$$

이 항등식을 이용하여 다양한 형태의 초기하분포 식을 상호 변환할 수 있다.


# 이항계수
## 정의 (Definition)
$n \choose k$, 서로 다른 $n$개 중에서 순서 없이 $k$개를 고르는 경우의 수의 실수 확장버전.  

실수(또는 복소수) $\alpha$와 음이 아닌 정수 $k\ge 0$에 대해
$$\binom{\alpha}{k}=\frac{\alpha(\alpha-1)(\alpha-2)\cdots(\alpha-k+1)}{k!}$$
로 정의한다. 이는 $\alpha$가 음수인 경우도 포함한다.

또한 편의상
$$\binom{n}{k}=0\quad (k<0 \text{ 또는 } k>n)$$
로 두기도 한다.

## 기본 성질 (Basic identities)

### 대칭성 (Symmetry)
$$\binom{n}{k}=\binom{n}{n-k}$$

### 파스칼 항등식 (Pascal identity)

$$\binom{n}{k}=\binom{n-1}{k}+\binom{n-1}{k-1}\quad (1\le k\le n-1)$$

#### 조합론적 증명 (Combinatorial proof)

$n$개 중 $k$개를 고를 때, 특정 원소(예: $n$번째 원소)를 **포함하지 않는 경우**는 $\binom{n-1}{k}$, **포함하는 경우**는 $\binom{n-1}{k-1}$이므로 합이 전체 경우의 수 $\binom{n}{k}$이다. □

## 이항정리 (Binomial theorem)

### 정리 (정수 지수)

정수 $n\ge 0$에 대해
$$(a+b)^n=\sum_{k=0}^{n}\binom{n}{k}a^{n-k}b^{k}$$

#### 증명 (Proof)

$(a+b)^n$을 $n$개의 괄호 $(a+b)$의 곱으로 보고 전개하면, 각 괄호에서 $b$를 $k$번 선택하고 나머지는 $a$를 선택한 항이 $a^{n-k}b^k$를 만든다.
그러한 선택의 수가 $\binom{n}{k}$이므로 위 식이 성립한다. □

## 합 공식 (Summation identities)

### 전체 합

$$\sum_{k=0}^{n}\binom{n}{k}=2^n$$

이는 이항정리에서 $a=b=1$을 대입하면 된다.

### 교대 합 (Alternating sum)

$$\sum_{k=0}^{n}(-1)^k\binom{n}{k}=0\quad(n\ge 1)$$

이는 이항정리에서 $a=1,b=-1$을 대입하면 된다.

### 가중합 (First moment identity)

$$\sum_{k=0}^{n}k\binom{n}{k}=n2^{n-1}$$

#### 증명

이항정리 $(1+x)^n=\sum_{k=0}^n \binom{n}{k}x^k$를 미분하면
$$n(1+x)^{n-1}=\sum_{k=1}^n k\binom{n}{k}x^{k-1}$$

여기서 $x=1$을 대입하면 결과가 나온다. □

## 곱셈 형태 항등식 (Multiplicative identities)

### 관계식 $k\binom{n}{k}=n\binom{n-1}{k-1}$

$$k\binom{n}{k}=n\binom{n-1}{k-1}$$

이는 양변을 팩토리얼 정의로 직접 계산하면 된다.

### 관계식 $(n-k)\binom{n}{k}=n\binom{n-1}{k}$

$$(n-k)\binom{n}{k}=n\binom{n-1}{k}$$

## 반덜몬드 항등식 (Vandermonde identity)

### 정리

정수 $r,s\ge 0$에 대해
$$\sum_{k=0}^{n}\binom{r}{k}\binom{s}{n-k}=\binom{r+s}{n}$$

#### 조합론적 증명

$r+s$개의 원소가 두 그룹(크기 $r$, 크기 $s$)로 나뉘어 있을 때, 전체에서 $n$개를 고르는 방법 수는 $\binom{r+s}{n}$이다.
한편 첫 그룹에서 $k$개, 둘째 그룹에서 $n-k$개를 고르는 경우의 수는 $\binom{r}{k}\binom{s}{n-k}$이고, $k$에 대해 합하면 전체 경우의 수가 된다. □

## 일반화된 이항계수 (Generalized binomial coefficient)

실수(또는 복소수) $\alpha$와 정수 $n\ge 0$에 대해
$$\binom{\alpha}{n}=\frac{\alpha(\alpha-1)\cdots(\alpha-n+1)}{n!}$$
로 정의한다. 

특히 $\alpha=\frac{1}{2}$인 경우가
$$\sqrt{1+x}=(1+x)^{1/2}$$
테일러급수에 등장한다.

## 일반화된 이항정리 (Generalized binomial series)

$$(1+x)^{\alpha}=\sum_{n=0}^{\infty}\binom{\alpha}{n}x^n,\quad |x|<1$$

이는 해석학에서 테일러 전개로도 증명되며, 수렴반경은 1이다.

### 음수 인수의 이항계수
$\alpha = -r$ (단, $r > 0$)인 경우, $y \ge 0$인 정수에 대해
$$\binom{-r}{y} = \frac{(-r)(-r-1)\cdots(-r-y+1)}{y!} \\ = (-1)^y \frac{r(r+1)\cdots(r+y-1)}{y!} = (-1)^y \binom{r+y-1}{y}$$

이 항등식은 **음이항분포**의 확률질량함수와 일반화된 이항정리 $(1+t)^{-r}$의 전개에서 핵심적인 역할을 한다.

## 통계/확률에서 자주 쓰는 연결

* 이항분포 $B(n,p)$의 pmf에 $\binom{n}{k}$가 등장한다.
* 포아송 근사에서 $\binom{n}{k}\sim \frac{n^k}{k!}$ 같은 형태의 근사가 쓰인다.
* 조합 항등식은 기대값 계산(예: $\sum k\binom{n}{k}$)에 직접 사용된다.


## 이항계수의 재귀적 곱셈 항등식

### 정리
정수 $n, r, x$에 대해 ($0 \le r \le x \le n$)
$$x(x-1)\cdots(x-r+1)\binom{n}{x}=n(n-1)\cdots(n-r+1)\binom{n-r}{x-r}$$

### 증명
**좌변:**
$$x(x-1)\cdots(x-r+1)\binom{n}{x}=x(x-1)\cdots(x-r+1)\cdot\frac{n!}{x!(n-x)!}$$

$$=\frac{x(x-1)\cdots(x-r+1) \cdot n!}{x(x-1)\cdots 1 \cdot (n-x)!}$$

$$=\frac{n!}{(x-r)!(n-x)!}$$

**우변:**
$$n(n-1)\cdots(n-r+1)\binom{n-r}{x-r}=n(n-1)\cdots(n-r+1)\cdot\frac{(n-r)!}{(x-r)!(n-x)!}$$

$$=\frac{n(n-1)\cdots(n-r+1)(n-r)!}{(x-r)!(n-x)!}$$

$$=\frac{n!}{(x-r)!(n-x)!}$$

따라서 좌변 = 우변. □

### 대안 증명 (조합론적)
$n$개의 원소 중 $x$개를 선택하고, 그 중에서 순서를 고려하여 $r$개를 배열하는 경우의 수를 두 가지 방법으로 센다.

- **방법 1**: 먼저 $x$개를 선택($\binom{n}{x}$), 그 중 $r$개를 순서대로 배열($x(x-1)\cdots(x-r+1)$)
- **방법 2**: 먼저 $n$개 중 $r$개를 순서대로 선택($n(n-1)\cdots(n-r+1)$), 나머지 $n-r$개 중 $x-r$개를 선택($\binom{n-r}{x-r}$)

두 방법 모두 같은 결과를 세므로 등식이 성립한다. □

# 특수함수
## 베타 함수
[자세히 보기](https://namu.wiki/w/%EB%B2%A0%ED%83%80%20%ED%95%A8%EC%88%98)

베타 함수 $B(x, y)$는 다음과 같은 적분으로 정의되는 특수 함수입니다.

$$
B(x, y) = \int_0^1 t^{x-1} (1-t)^{y-1} \, dt
$$

여기서 $x$와 $y$는 모두 0보다 큰 값입니다. 베타 함수는 감마 함수와 다음과 같은 관계를 가집니다.

$$
B(x, y) = \frac{\Gamma(x)\,\Gamma(y)}{\Gamma(x+y)}
$$

베타 함수는 베타 분포, 베이지안 통계 등 다양한 분야에서 활용됩니다.

## 감마 함수
[자세히 보기](https://namu.wiki/w/%EA%B0%90%EB%A7%88%20%ED%95%A8%EC%88%98)

감마 함수 $\Gamma(z)$는 다음과 같은 적분으로 정의됩니다.

$$
\Gamma(z) = \int_0^{\infty} e^{-t} t^{z-1} \, dt
$$

여기서 $z$는 0보다 큰 실수 또는 복소수입니다. 감마 함수는 팩토리얼 함수의 확장으로, 자연수 $n$에 대해 $\Gamma(n) = (n-1)!$가 성립합니다. 감마 함수는 확률론, 통계학, 복소 해석 등 다양한 수학 분야에서 중요한 역할을 합니다.

### 스털링의 근사법 (Stirling's approximation)
큰 $n$에 대해 팩토리얼을 근사하는 공식으로, 감마 함수의 점근 전개에서 유도됩니다.

$$n! \approx \sqrt{2\pi n} \left(\frac{n}{e}\right)^n$$

또는 로그 형태로
$$\ln(n!) \approx n\ln(n) - n + \frac{1}{2}\ln(2\pi n)$$

이 공식은 조합론, 확률론, 통계학에서 큰 $n$의 이항계수나 확률을 근사할 때 매우 유용합니다.

#### 증명
Stirling의 공식은 감마 함수의 적분 표현에서 라플라스 방법(Laplace's method)을 적용하여 유도됩니다.

$$m! = \Gamma(m+1) = \int_0^{\infty} x^m e^{-x} \, dx$$

지수를 정리하면
$$\int_0^{\infty} e^{m\ln x - x} \, dx$$

라플라스 방법: 피적분함수 $e^{m\ln x - x}$는 $x = m$에서 최댓값을 가집니다. $z = \sqrt{m}(x - m)$으로 치환하면, $x = m + \frac{z}{\sqrt{m}}$이고

$$m! = \int_{-\infty}^{\infty} e^{m\ln(m + z/\sqrt{m}) - (m + z/\sqrt{m})} \, \frac{dz}{\sqrt{m}}$$

$\ln(m + z/\sqrt{m}) \approx \ln m + \frac{z}{m\sqrt{m}} - \frac{z^2}{2m^2}$ 근처에서 테일러 전개하면

$$m! \approx m^m e^{-m} \sqrt{m} \int_{-\infty}^{\infty} e^{-z^2/(2m)} \, dz = m^m e^{-m} \sqrt{m} \cdot \sqrt{2\pi m}$$

따라서
$$m! \sim \sqrt{2\pi m} \left(\frac{m}{e}\right)^m, \quad m \to \infty$$

이항분포 $B(n, p)$의 누적확률은 정규분포로 근사되며, Stirling의 공식을 이용하면 이 근사의 오차를 정량화할 수 있습니다.

### 이항확률의 정규 근사
**(a) 이항확률의 정규근사**  
$q = 1 - p$라 할 때, $a \le \frac{x-np}{\sqrt{npq}} \le b$인 $x$에 대해

$$P(X=x) = \binom{n}{x} p^x q^{n-x} \sim \frac{1}{\sqrt{2\pi npq}} e^{-\frac{(x-np)^2}{2npq}}, \quad n \to \infty$$

**(b) 누적확률의 정규근사**  
$$\sum_{\substack{x: a \le \frac{x-np}{\sqrt{npq}} \le b}} P(X=x) 
= \sum_{\substack{x: a \le \frac{x-np}{\sqrt{npq}} \le b}} \binom{n}{x} p^x q^{n-x} \sim \int_a^b \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}z^2} \, dz, \quad n \to \infty$$

이는 **중심극한정리**의 직접적인 결과이며, $n$이 충분히 클 때 이항분포 $B(n,p)$를 정규분포 $N(np, npq)$로 근사할 수 있음을 의미합니다.

#### 증명
$X_1, X_2, \ldots, X_n$을 독립적인 베르누이 확률변수라 하고, $X = \sum_{i=1}^n X_i$라 하면 $X \sim B(n,p)$입니다.

중심극한정리에 의해
$$Z_n = \frac{X - np}{\sqrt{npq}} \xrightarrow{d} N(0,1), \quad n \to \infty$$

따라서 충분히 큰 $n$에 대해
$$P\left(a \le \frac{X-np}{\sqrt{npq}} \le b\right) \approx \Phi(b) - \Phi(a) = \int_a^b \frac{1}{\sqrt{2\pi}} e^{-z^2/2} \, dz$$

여기서 $\Phi$는 표준정규분포의 누적분포함수입니다.

(a)의 경우, Stirling 공식을 이용하여 $\binom{n}{x}p^x q^{n-x}$의 합을 리만 합 근사로 적분으로 변환할 수 있습니다. $x$가 $np$ 근처에서 $\sqrt{npq}$ 스케일로 변할 때, 
$$\binom{n}{x}p^x q^{n-x} \approx \frac{1}{\sqrt{2\pi npq}} \exp\left(-\frac{(x-np)^2}{2npq}\right)$$
가 성립하며, 이를 이용하여 (b)의 적분 표현을 얻습니다. □
