# 1. 물리적 벡터

## (1) 평면벡터

$R^2$에서 크기(스칼라)와 방향의 의미를 모두 포함하는 표현 도구이다.

* 점 $A(a_1, a_2)$, $B(b_1, b_2)$에 대해
  $$
  \overrightarrow{AB} = (b_1 - a_1, b_2 - a_2)
  $$
* 원점 $O(0,0)$에서의 위치벡터: $\overrightarrow{OA} = (a_1, a_2)$

예시 그림에서 벡터 $\mathbf{v}$와 같은 벡터는 $\mathbf{d}$이다.

## (2) 공간벡터

$R^3$에서 크기와 방향의 의미를 모두 포함하는 표현 도구이다.

* 점 $A(a_1,a_2,a_3)$, $B(b_1,b_2,b_3)$에 대해
  $$
  \overrightarrow{AB} = (b_1 - a_1, b_2 - a_2, b_3 - a_3)
  $$

## (3) n차원 벡터

$R^n$상의 벡터 $\mathbf{v} = (v_1, v_2, \dots, v_n)$
두 벡터 $\mathbf{v} = (v_1,\dots,v_n)$, $\mathbf{w} = (w_1,\dots,w_n)$가 같을 필요충분조건은
$$
v_i = w_i \quad (\forall i)
$$
이다.

# 2. 벡터의 연산
## (1) 노름 (Norm)
벡터의 크기(또는 길이)는
$$
|\mathbf{v}| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}
$$

단위벡터(unit vector):
$$
\hat{v} = \frac{\mathbf{v}}{|\mathbf{v}|}
$$

표준단위벡터:
$e_1 = (1,0,\dots,0),\ e_2 = (0,1,0,\dots)$

## (2) 선형결합
### ① 덧셈과 뺄셈
$$
\mathbf{v} + \mathbf{w} = (v_1 + w_1, \dots, v_n + w_n)
$$

### ② 실수배
$$
k\mathbf{v} = (kv_1, kv_2, \dots, kv_n)
$$

### ③ 일반형(선형결합)
$$
\mathbf{w} = k_1\mathbf{v}_1 + k_2\mathbf{v}_2 + \cdots + k_r\mathbf{v}_r
$$

## (3) 스칼라곱 (내적, Dot Product, Inner Product)
한 벡터가 다른 벡터의 방향에 대해 가지는 **영의 투영된 크기**이다.

$$
\mathbf{v}\cdot\mathbf{w} = |\mathbf{v}||\mathbf{w}|\cos\theta = v_1w_1 + v_2w_2 + \cdots + v_nw_n
$$

($\theta$는 두 벡터가 이루는 각)

### 벡터의 연산 성질 (Properties of Vector Operations)
$R^n$ 상의 벡터 $\mathbf{u}, \mathbf{v}, \mathbf{w}$와 스칼라 $k, m$에 대하여 다음이 성립한다.

(1) 덧셈과 뺄셈의 성질
| 번호 | 식 | 의미            |  
| - | - | - |
| (a) | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$                               | **교환법칙**    |
| (b) | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ | **결합법칙**    |
| (c) | $\mathbf{u} + \mathbf{0} = \mathbf{u}$                                            | **덧셈의 항등원** |
| (d) | $\mathbf{u} + (-\mathbf{u}) = \mathbf{0}$                                         | **덧셈의 역원**  |

(2) 스칼라곱과 분배법칙
| 번호 | 식 | 의미            |
| - | - | - |
| (e) | $k(\mathbf{u} + \mathbf{v}) = k\mathbf{u} + k\mathbf{v}$  | 스칼라에 대한 **분배법칙** |
| (f) | $(k + m)\mathbf{u} = k\mathbf{u} + m\mathbf{u}$           | 벡터에 대한 **분배법칙**  |
| (g) | $k(m\mathbf{u}) = (km)\mathbf{u}$                         | **스칼라곱의 결합법칙**   |
| (h) | $1\mathbf{u} = \mathbf{u}$                                | 단위 스칼라 곱의 항등성    |
| (i) | $0\mathbf{u} = \mathbf{0},\quad k\mathbf{0} = \mathbf{0}$ | 영벡터의 성질          |

(3) 스칼라곱의 성질
| 번호 | 식 | 의미            |  
| - | - | - |
| (j) | $\mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u}$                                          | **교환법칙**      |
| (k) | $\mathbf{0} \cdot \mathbf{u} = \mathbf{u} \cdot \mathbf{0} = 0$                                      | 영벡터 내적        |
| (l) | $\mathbf{u} \cdot (\mathbf{v} + \mathbf{w}) = \mathbf{u}\cdot\mathbf{v} + \mathbf{u}\cdot\mathbf{w}$ | **분배법칙 (좌측)** |
| (m) | $(\mathbf{u} + \mathbf{v}) \cdot \mathbf{w} = \mathbf{u}\cdot\mathbf{w} + \mathbf{v}\cdot\mathbf{w}$ | **분배법칙 (우측)** |
| (n) | $k(\mathbf{u}\cdot\mathbf{v}) = (k\mathbf{u})\cdot\mathbf{v} = \mathbf{u}\cdot(k\mathbf{v})$         | **스칼라곱의 결합성** |

### (4) 벡터곱 (가위곱, Cross Product)
(외적 이라고도 하지만 tensor product랑 헷갈리므로 지양)  
3차원 한정!!! 두 벡터가 평면상에서 이루는 평행사변형의 면적과 법선방향을 갖는 벡터이다.  
4차원 이상의 고차원 공간에서는 두 벡터에 수직인 방향이 무한히 많아져, 유일한 "벡터" 결과값을 정의하기 어렵다.  
$$
\mathbf{v}\times\mathbf{w} =
\begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
v_1 & v_2 & v_3 \\
w_1 & w_2 & w_3
\end{vmatrix}
$$

성질:
결합법칙이 성립하지 않는다. 먼저 계산한 순서가 영향을 미침.  

| 번호 | 식 | 의미            |  
| - | - | - |
| (a) | $\mathbf{u} \times \mathbf{v} = -(\mathbf{v} \times \mathbf{u})$                                                | **반교환법칙 (Anti-commutativity)**  |
| (b) | $\mathbf{u} \times (\mathbf{v} + \mathbf{w}) = (\mathbf{u} \times \mathbf{v}) + (\mathbf{u} \times \mathbf{w})$ | **분배법칙 (Left Distributivity)**  |
| (c) | $(\mathbf{u} + \mathbf{v}) \times \mathbf{w} = (\mathbf{u} \times \mathbf{w}) + (\mathbf{v} \times \mathbf{w})$ | **분배법칙 (Right Distributivity)** |
| (d) | $k(\mathbf{u} \times \mathbf{v}) = (k\mathbf{u}) \times \mathbf{v} = \mathbf{u} \times (k\mathbf{v})$           | **스칼라곱의 결합성**                   |
| (e) | $\mathbf{u} \times \mathbf{0} = \mathbf{0} \times \mathbf{u} = \mathbf{0}$                                      | **영벡터의 성질**                     |
| (f) | $\mathbf{u} \times \mathbf{u} = \mathbf{0}$                                                                     | **자기 외적은 0**                    |


참고) 외적: 텐서곱  
참고) wedge product, $\wedge$  
$\mathbf{u}\wedge\mathbf{v}$는 벡터가 아니라 2차 텐서(또는 2-형식) 로,
두 벡터로 생성되는 “평면의 방향과 넓이”를 나타낸다.  
이는 외미분기하학(differential forms), 선형대수학, 기하대수(Geometric Algebra) 에서 핵심 개념이다.

# 3. 벡터의 응용
## (1) 직선의 표현
$R^2$ 또는 $R^3$에서 위치벡터가 $\mathbf{a}$인 점 $A$를 지나며 방향벡터가 $\mathbf{v}$인 직선 위 점 $\mathbf{x}$는
$$
\mathbf{x} = \mathbf{a} + k\mathbf{v}, \quad (k \in \mathbb{R})
$$

## (2) 평면의 표현
$R^3$에서 위치벡터가 $\mathbf{a}$인 점 $A$를 지나며 법선벡터가 $\mathbf{v}$인 평면 위 점 $\mathbf{x}$는
$$
(\mathbf{x} - \mathbf{a})\cdot\mathbf{v} = 0
$$

# 연습문제
1. 다음 두 벡터 $\mathbf{u}, \mathbf{v}$의 사잇각 $\theta$에 대하여 $\cos\theta$를 구하시오.  
   (1) $\mathbf{u}=(-3,5), \mathbf{v}=(2,7)$  
   (2) $\mathbf{u}=(2,1,3), \mathbf{v}=(1,2,-4)$  
   (3) $\mathbf{u}=(2,0,1,-2), \mathbf{v}=(1,5,-3,2)$  

2. 다음 두 벡터 $\mathbf{u}, \mathbf{v}$에 의해 결정되는 평행사변형의 넓이를 구하시오.  
   (1) $\mathbf{u}=(2,3,0), \mathbf{v}=(-1,2,-2)$  
   (2) $\mathbf{u}=(-3,1,-3), \mathbf{v}=(6,-2,6)$  

3. 점 $(-1,-1,0), (2,0,1)$을 지나는 직선이 세 점 $(1, -1, 0), (1, 2, 0), (0, 0, 2)$를 지나는 평면과 만나는 교점의 좌표를 구하시오.

4. 두 벡터 $\mathbf{a}=(2,0), \mathbf{b}=(1,3)$으로 구성된 행렬의 $\det([a\ b])$의 기하학적 의미를 설명하시오.

5. 세 벡터 $\mathbf{a}=(2,0,0), \mathbf{b}=(0,3,0), \mathbf{c}=(1,1,1)$로 구성된 행렬
   $$
   \det
   \begin{pmatrix}
   2 & 0 & 1\\
   0 & 3 & 1\\
   0 & 0 & 1
   \end{pmatrix}
   $$
   의 기하학적 의미를 해석하시오.