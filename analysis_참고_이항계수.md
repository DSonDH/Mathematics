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