# 1. 곡선 적합 (Curve Fitting)
## (1) 보간법 (Interpolation)
### 1) 개념
주어진 특정 점들을 포함하는 함수를 구하는 방법이다.

정리) 좌표평면에 있는 임의의 서로 다른 $n$개의 점을 지나는 $k$차 다항함수는 유일하게 존재한다.
(단, $k$는 $k < n$인 자연수)

### 2) 사례
네 점 $(1,3), (2,-2), (3,-5), (4,0)$을 모두 지나는 3차 함수
$f(x)=a_0 + a_1 x + a_2 x^2 + a_3 x^3$
를 구한다.

#### Step 1
다음의 방정식을 세운다.

$$
\begin{pmatrix}
1 & x_1 & x_1^2 & x_1^3 \\
1 & x_2 & x_2^2 & x_2^3 \\
1 & x_3 & x_3^2 & x_3^3 \\
1 & x_4 & x_4^2 & x_4^3
\end{pmatrix}
\begin{pmatrix}
a_0\\ a_1\\ a_2\\ a_3
\end{pmatrix}
=
\begin{pmatrix}
y_1\\ y_2\\ y_3\\ y_4
\end{pmatrix}
$$

#### Step 2
네 점을 대입하고 첨가행렬을 만든다.

$$
\begin{pmatrix}
1 & 1 & 1 & 1 & 3 \\
1 & 2 & 4 & 8 & -2 \\
1 & 3 & 9 & 27 & -5 \\
1 & 4 & 16 & 64 & 0
\end{pmatrix}
$$

#### Step 3
가우스-조단 소거법으로 푼다.

$$
\begin{pmatrix}
1 & 1 & 1 & 1 & 3 \\
1 & 2 & 4 & 8 & -2 \\
1 & 3 & 9 & 27 & -5 \\
1 & 4 & 16 & 64 & 0
\end{pmatrix}
\sim
\begin{pmatrix}
1 & 0 & 0 & 0 & 4\\
0 & 1 & 0 & 0 & 3\\
0 & 0 & 1 & 0 & -5\\
0 & 0 & 0 & 1 & 1
\end{pmatrix}
$$

#### Step 4
$a_0=4, a_1=3, a_2=-5, a_3=1$
따라서
$
f(x)=4 + 3x - 5x^2 + x^3.
$

## (2) 최소제곱법 (Least Squares)
### 1) 개념
특정 점들을 **정확히** 포함하는 함수가 존재하지 않을 때, 오차 제곱합이 최소가 되는 근사함수를 구하는 방법이다.

정리) 방정식 $AX=B$은 일반적으로 해가 없을 수 있으나, $A^TAX = A^TB$ 방정식 (이 식을 정규방정식이라 함)을 만족하는 $X$는 항상 존재하며 이를 $AX=B$의 최소제곱해라 한다.

### 2) 사례

네 점 $(0,1),(1,3),(2,4),(3,4)$에 근사하는 일차함수
$
f(x)=a_0+a_1 x
$
를 구한다.

#### Step 1

$$
AX=B
\quad\text{with}\quad
A=
\begin{pmatrix}
1 & x_1\\
1 & x_2\\
1 & x_3\\
1 & x_4
\end{pmatrix},
\quad
B=
\begin{pmatrix}
y_1\\ y_2\\ y_3\\ y_4
\end{pmatrix}
$$

#### Step 2
정규방정식 $A^TAX=A^TB$를 구성한다.
$$
A^TA=
\begin{pmatrix}
4 & 6\\
6 & 14
\end{pmatrix},
\quad
A^TB=
\begin{pmatrix}
12\\
23
\end{pmatrix}
$$

$$
(A^TA)^{-1}
=
\frac1{10}
\begin{pmatrix}
7 & -3\\
-3 & 2
\end{pmatrix}
$$

#### Step 3
$$
X=
\begin{pmatrix}
a_0 \\ a_1
\end{pmatrix}
=

\begin{pmatrix}
\frac{3}{2}\\
1
\end{pmatrix}
$$

따라서
$f(x)=\frac{3}{2} + x$  
완벽히 맞지 않아도 최대한 근접한 식이 완성되더라.  

### 3) $n$차 일반화
$m$개의 데이터 $(x_1,y_1), \ldots, (x_m,y_m)$에 대해  
$n$차 다항식
$y = a_0 + a_1x + \cdots + a_n x^n$
을 최소제곱법으로 구하기 위해서는

$AX=B$

$$
A=
\begin{pmatrix}
1 & x_1 & x_1^2 & \cdots & x_1^n\\
1 & x_2 & x_2^2 & \cdots & x_2^n\\
\vdots & \vdots & \vdots & & \vdots \\
1 & x_m & x_m^2 & \cdots & x_m^n
\end{pmatrix},
\quad
x=
\begin{pmatrix}
a_0\\ a_1\\ \vdots\\ a_n
\end{pmatrix}
\quad
B=
\begin{pmatrix}
y_1\\ y_2\\ \vdots\\ y_m
\end{pmatrix}
$$
으로 설정하면 된다.

## (3) 두 방법의 비교
| 구분 | 보간법 | 최소제곱법 |
| - | - | - |
| 목표 | 데이터를 모두 포함 | 데이터 경향을 가장 잘 설명 |
| 데이터 수 | 적을수록 좋음 | 많아도 가능       |
| 정밀도   | 매우 높음      | 상대적으로 낮음     |
| 신축성   | 조절이 어려움     | 조절이 자유로움     |

# 2. 이차형식의 최적화 (Optimization of Quadratic Forms)
## (1) 이차형식 (Quadratic Form)
가환환 $K$위의 가군 $V$에 대해 다음을 만족하는 함수
$Q:V\to K$ 을 이차형식이라 한다.

1. $Q(kv)=k^2 Q(v)$
2. $Q(u+v+w)=Q(u+v)+Q(v+w)+Q(u,w)-Q(u)-Q(v)-Q(w)$
3. $Q(ku+lv) = k^2Q(u)+l^2Q(v)+klQ(u+v)-klQ(u)-klQ(v)$

ex 1) $ \mathbb{R}^2 $ 상의 일반적인 이차형식:  
$
a_1 x_1^2 + a_2 x_2^2 + 2 a_3 x_1 x_2
\Longleftrightarrow\
(x_1\ \ x_2)
\begin{pmatrix}
a_1 & a_3 \\
a_3 & a_2
\end{pmatrix}
\begin{pmatrix}
x_1 \\
x_2
\end{pmatrix}
$

ex 2) $ \mathbb{R}^3 $ 상의 일반적인 이차형식  
$
a_1 x_1^2 + a_2 x_2^2 + a_3 x_3^2 + 2 a_4 x_1 x_2 + 2 a_5 x_1 x_3 + 2 a_6 x_2 x_3  \\
\Longleftrightarrow\
(x_1\ \ x_2\ \ x_3)
\begin{pmatrix}
a_1 & a_4 & a_5 \\
a_4 & a_2 & a_6 \\
a_5 & a_6 & a_3
\end{pmatrix}
\begin{pmatrix}
x_1 \\
x_2 \\
x_3
\end{pmatrix}
$

## (2) 제약된 극값 (Constrained Extremum)

### 1) 개념

특정 제약 조건 아래에서 함수가 갖는 최댓값 또는 최솟값을 구하는 문제이다.

정리) $n\times n$ 행렬 $A$의 고윳값을 내림차순으로  
$\lambda_1,\dots,\lambda_n,\quad |v|=1$ 라 하면  
$v^TAv$의 최댓값과 최솟값은  
$\lambda_1, \lambda_n$에 대응하는 단위고유벡터에서 존재한다.

### 2) 사례
제약$x^2+y^2=1$에서  
$z=5x^2+4xy+5y^2$의 최댓값과 최솟값을 찾는다.

#### Step 1
2차형식 식에 의해,  
$
z = (x\ y)
\begin{pmatrix}
5 & 2\\
2 & 5
\end{pmatrix}
\begin{pmatrix}
x\\ y
\end{pmatrix}
$

#### Step 2
행렬
$$
A=\begin{pmatrix}
5 & 2\\
2 & 5
\end{pmatrix}
$$
의 고윳값과 고유벡터를 구한다.  
$\lambda_1=7,\quad v_1 = (1,1)$  
$\lambda_2=3,\quad v_2 = (-1,1)$

#### Step 3
고유벡터 정규화  
$
\lambda_1=\frac{1}{\sqrt2}(1,1),
\quad
\lambda_2=\frac{1}{\sqrt2}(-1,1)
$  

#### Step 4
$z_{\max}=7, \quad (x,y)=\frac{1}{\sqrt2}(1,1)$

$z_{\min}=3,\quad (x,y)=\frac{1}{\sqrt2}(-1,1)$

이는, 타원 반경이 길어짐에 따라 원과 곂치는 영역의 최대 최소 지점을 나타낸다.  

# 연습 문제
1. 다음 물음에 답하시오.  
   (1) 세 점 $(0,-1),(1,2),(-1,0)$을 지나는 이차함수를 구하시오.  
   (2) 네 점 $(-1,-2),(0,-4),(1,0),(2,16)$을 지나는 삼차함수를 구하시오.  

2. 최소제곱법을 이용하여 답하시오.  
   (1) 네 점 $(0,0),(2,-1),(3,4)$에 근사한 일차함수를 구하시오.  
   (2) 네 점 $(1,6),(2,1),(-1,5),(-2,2)$에 근사한 이차함수를 구하시오.

3. 제약조건 $x^2+y^2=1$ 하에서 $4x^2+2y^2$의 최댓값과 최솟값을 구하고, 그 때의 $(x,y)$ 값을 각각 구하시오.

4. 제약조건 $x^2+y^2+z^2=1$ 하에서 $x^2-3y^2+8z^2$의 최댓값과 최솟값을 구하고, 그 때의 $(x,y)$ 값을 각각 구하시오.
