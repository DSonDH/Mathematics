# Analysis 추가 개념

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

### 예시
1. $f(x) = |x|$는 $K=1$인 Lipschitz 연속
   - $||x_1| - |x_2|| \leq |x_1 - x_2|$ (삼각부등식)
2. $f(x) = x^2$는 유계 구간에서만 Lipschitz 연속
   - $|x_1^2 - x_2^2| = |x_1 + x_2| \cdot |x_1 - x_2| \leq 2M|x_1 - x_2|$ (when $|x_i| \leq M$)
3. $f(x) = \sqrt{x}$는 Lipschitz 연속이 아님 ($x=0$ 근처에서 미분 불가능)


## 르벡 적분 (Lebesgue Integration)

### 측도 (Measure)
**측도** $\mu$는 집합의 "크기"를 재는 함수이다.

정의역: 가측 집합들의 모임 $\mathcal{M}$ (시그마-대수)  
치역: $[0, \infty]$

측도의 성질:
1. $\mu(\emptyset) = 0$ (공집합의 측도는 0)
2. **가산 가법성**: 서로소인 가측 집합 $E_1, E_2, \ldots$에 대해
   $$\mu\left(\bigcup_{i=1}^{\infty} E_i\right) = \sum_{i=1}^{\infty} \mu(E_i)$$

예시:
- **르벡 측도**: $\mathbb{R}^n$에서 일반적인 "길이", "넓이", "부피"
- **계수 측도**: 집합의 원소 개수
- **확률 측도**: 전체 공간의 측도가 1인 측도

### 르벡 적분의 기본 아이디어
리만 적분과 달리, 르벡 적분은 **함숫값(치역)을 갖는 영역의 크기**를 더하는 방식이다.

**기본 구조**:
1. **단순 함수로 시작**: 값이 유한개인 계단 함수 사용
2. **각 함숫값에 대해 계산**:
   $$(\text{함숫값}) \times (\text{그 값을 갖는 집합의 측도})$$
3. **근사의 극한**: 점점 더 정교한 단순 함수로 근사하여 극한을 취함

### 단순 함수 (Simple Function)
$$s(x) = \sum_{i=1}^{n} a_i \chi_{E_i}(x)$$

여기서:
- $a_i \in \mathbb{R}$는 **함수가 가지는 유한개의 상수 값**
- $E_i$는 서로소인 **가측 집합** (measurable set): 측도를 잴 수 있는 집합
- $\chi_{E_i}$는 **특성 함수** (characteristic function): 
  $$\chi_{E_i}(x) = \begin{cases} 1 & x \in E_i \\ 0 & x \notin E_i \end{cases}$$

단순 함수의 르벡 적분:
$$\int s \, d\mu = \sum_{i=1}^{n} a_i \mu(E_i)$$

### 비음 가측 함수의 적분
비음 가측 함수 $f \geq 0$에 대해:

$$\int f \, d\mu = \sup \left\{ \int s \, d\mu : 0 \leq s \leq f, \, s \text{ 단순함수} \right\}$$

- 아래에서 근사하는 단순함수들의 적분의 상한으로 정의
- 리만 적분과 달리 정의역을 분할하지 않고 치역을 분할

### 일반 가측 함수의 적분
임의의 가측 함수 $f$를 양의 부분과 음의 부분으로 분해:

$$f^+(x) = \max(f(x), 0), \quad f^-(x) = \max(-f(x), 0)$$

이때 $f = f^+ - f^-$이고:

$$\int f \, d\mu = \int f^+ \, d\mu - \int f^- \, d\mu$$

함수 $f$가 **적분가능** (integrable)하려면 $\int f^+ \, d\mu < \infty$ 이고 $\int f^- \, d\mu < \infty$ 이어야 한다.

### 주요 정리
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
