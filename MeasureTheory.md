## 📁 File 01. MeasureTheory_01_SetAndSigmaAlgebra.md

### 1. 집합과 $\sigma$-대수 *(Sets and $\sigma$-Algebras)*

#### Def. 1. 집합대수 *(Algebra of Sets)*

집합 $\Omega$의 부분집합족 $\mathcal A \subset 2^\Omega$가 다음을 만족하면 **집합대수**라 한다.

1. $\Omega \in \mathcal A$
2. $A\in\mathcal A \Rightarrow A^c \in \mathcal A$
3. $A,B\in\mathcal A \Rightarrow A\cup B \in \mathcal A$

#### Def. 2. $\sigma$-대수 *(Sigma-Algebra)*

$\mathcal F \subset 2^\Omega$가

1. $\Omega\in\mathcal F$
2. $A\in\mathcal F \Rightarrow A^c\in\mathcal F$
3. ${A_n}\subset\mathcal F \Rightarrow \bigcup_{n=1}^\infty A_n\in\mathcal F$
   를 만족하면 **$\sigma$-대수**라 한다.

#### Lemma.

$\sigma$-대수는 가산 교집합에 대해서도 닫혀 있다.

**증명**
(증명 생략)

---

## 📁 File 02. MeasureTheory_02_Measure.md

### 2. 측도의 정의 *(Definition of Measure)*

#### Def. 1. 측도 *(Measure)*

$(\Omega,\mathcal F)$ 위의 함수
$$
\mu:\mathcal F\to[0,\infty]
$$
가

1. $\mu(\varnothing)=0$
2. 서로소 ${A_n}$에 대해
   $$
   \mu\Big(\bigcup_{n=1}^\infty A_n\Big)=\sum_{n=1}^\infty \mu(A_n)
   $$
   이면 $\mu$를 **측도**라 한다.

#### Def. 2. 측도공간 *(Measure Space)*

삼중쌍 $(\Omega,\mathcal F,\mu)$를 **측도공간**이라 한다.

#### Thm. (Monotonicity)

$A\subset B \Rightarrow \mu(A)\le\mu(B)$

**증명**
(증명 생략)

---

## 📁 File 03. MeasureTheory_03_OuterMeasure.md

### 3. 외측도 *(Outer Measure)*

#### Def. 외측도 *(Outer Measure)*

함수 $\mu^*:2^\Omega\to[0,\infty]$가

1. $\mu^*(\varnothing)=0$
2. 단조성
3. 가산 아집합성
   을 만족하면 **외측도**라 한다.

#### Thm. (Carathéodory Criterion)

집합 $E$가
$$
\mu^*(A)=\mu^*(A\cap E)+\mu^*(A\cap E^c)
$$
를 모든 $A\subset\Omega$에 대해 만족하면 $E$는 $\mu^*$-가측이다.

**증명**
(증명 생략)

---

## 📁 File 04. MeasureTheory_04_LebesgueMeasure.md

### 4. 르베그 측도 *(Lebesgue Measure)*

#### Def. 르베그 외측도 *(Lebesgue Outer Measure)*

$E\subset\mathbb R$에 대해
$$
m^*(E)=\inf\left{\sum_{n=1}^\infty |I_n| \mid E\subset\bigcup I_n\right}
$$

#### Thm.

$m^*$로부터 정의된 가측집합들은 $\sigma$-대수를 이룬다.

**증명**
(증명 생략)

#### Cor.

구간 $[a,b]$의 르베그 측도는 $b-a$이다.

---

## 📁 File 05. MeasureTheory_05_MeasurableFunctions.md

### 5. 가측함수 *(Measurable Functions)*

#### Def. 가측함수 *(Measurable Function)*

$f:(\Omega,\mathcal F)\to(\mathbb R,\mathcal B)$가
$$
f^{-1}((-\infty,a))\in\mathcal F
$$
를 모든 $a\in\mathbb R$에 대해 만족하면 **가측**이라 한다.

#### Thm.

가측함수의 합, 곱, 극한은 가측이다.

**증명**
(증명 생략)

---

## 📁 File 06. MeasureTheory_06_SimpleFunctions.md

### 6. 단순함수 *(Simple Functions)*

#### Def. 단순함수 *(Simple Function)*

$$
\varphi=\sum_{k=1}^n a_k\mathbf 1_{A_k}
$$
형태의 가측함수를 단순함수라 한다.

#### Lemma.

모든 비음수가측함수는 단조증가 단순함수열로 근사된다.

**증명**
(증명 생략)

---

## 📁 File 07. MeasureTheory_07_LebesgueIntegral.md

### 7. 르베그 적분 *(Lebesgue Integral)*

#### Def. 비음수함수의 적분 *(Integral of Nonnegative Functions)*

단순함수 $\varphi$에 대해
$$
\int \varphi,d\mu=\sum a_k\mu(A_k)
$$

#### Def. 일반 함수의 적분 *(Integral of General Functions)*

$f=f^+-f^-$로 분해하여 정의한다.

#### Thm. (Monotone Convergence Theorem, MCT)

$f_n\uparrow f$이면
$$
\int f_n d\mu \uparrow \int f d\mu
$$

**증명**
(증명 생략)

---

## 📁 File 08. MeasureTheory_08_ConvergenceTheorems.md

### 8. 수렴정리들 *(Convergence Theorems)*

#### Thm. (Fatou’s Lemma)

$$
\int \liminf f_n d\mu \le \liminf \int f_n d\mu
$$

#### Thm. (Dominated Convergence Theorem, DCT)

$|f_n|\le g\in L^1$이고 $f_n\to f$ a.e.이면
$$
\lim\int f_n d\mu=\int f d\mu
$$

**증명**
(증명 생략)
