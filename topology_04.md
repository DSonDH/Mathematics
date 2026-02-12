# Chapter 4 가산성 (Countability)

## 4.1 제1가산 (First Countable)

### 정의 4.1: 제1가산공간
위상공간 $(X, \mathscr{T})$가 제1가산공간(first countable space)이라는 것은, 각 점이 가산국소기저(countable local base)를 가지는 것을 의미한다. 즉, 모든 $x \in X$에 대해 가산집합 $\{B_n : n \in \mathbb{N}\}$이 존재하여, $x$를 포함하는 모든 열린집합 $G$에 대해 $x \in B_n \subset G$인 $B_n$이 존재한다.

### 정리 4.1: 제1가산은 위상적 성질
제1가산성은 위상적 성질이다. 즉, 위상동형인 두 위상공간에서 한 쪽이 제1가산이면 다른 쪽도 제1가산이다.

## 4.2 제2가산 (Second Countable)

### 정의 4.2: 제2가산공간
위상공간 $(X, \mathscr{T})$가 제2가산공간(second countable space)이라는 것은, 가산기저(countable basis)를 가지는 것을 의미한다. 즉, 가산집합 $\mathcal{B} = \{B_n : n \in \mathbb{N}\}$이 존재하여 $\mathscr{T}$의 모든 열린집합이 $\mathcal{B}$의 원소들의 합집합으로 표현된다.

### 정리 4.2: 제2가산 함의 제1가산
제2가산공간은 제1가산공간이다.

**증명**: 제2가산공간이 가산기저 $\mathcal{B}$를 가지면, 각 점 $x$에 대해 $\mathcal{B}_x = \{B \in \mathcal{B} : x \in B\}$의 가산부분족이 $x$의 국소기저를 이룬다. ∎

### 정의 4.3: 린델뢰프공간 (Lindelöf space)
위상공간 $(X, \mathscr{T})$가 린델뢰프공간(Lindelöf space)이라는 것은, 모든 열린피복이 가산부분피복(countable subcover)을 가지는 것을 의미한다.

### 정리 4.3: 거리화가능 공간에서의 동치성
거리화가능 공간에서는 다음이 모두 동치이다:

1. 제2가산공간이다.
2. 린델뢰프공간이다.
3. 가분공간이다(separable space, 즉 가산 조밀부분집합을 가진다).

**참고**: 여기서 위상공간이 가분(separable)이라는 것은, 가산인 조밀부분집합(dense subset) $D \subset X$가 존재하여 $\overline{D} = X$인 것을 의미한다.