# 라그랑주 승수법 (Lagrange Multiplier Method)

## 1. 기본 목적

라그랑주 승수법은 **제약조건이 있는 최적화 문제를 제약이 없는 형태로 바꾸어 푸는 방법**이다.

$$\max_{x} f(x) \quad \text{subject to } g(x) = 0$$

여기서 $f(x)$는 목적함수, $g(x) = 0$은 제약조건이다.

## 2. 핵심 원리: 왜 $\nabla f = \lambda \nabla g$인가

제약식 $g(x) = 0$은 곡선이나 곡면을 정의한다. 최적점에서는 그 제약면을 따라 조금 움직여도 $f$가 1차적으로 변하지 않아야 한다. 즉, $f$의 증가방향인 $\nabla f$가 제약면과 직교해야 한다.

제약면 $g(x) = 0$의 법선벡터는 $\nabla g$이므로, 최적점에서는:

$$\nabla f = \lambda \nabla g$$

여기서 $\lambda$가 **라그랑주 승수(Lagrange multiplier)** 다.

## 3. 라그랑지안(Lagrangian)

위 조건을 체계적으로 계산하기 위해 라그랑지안을 정의한다:

$$\mathcal{L}(x, \lambda) = f(x) - \lambda g(x)$$

최적해는 다음 연립방정식을 풀어서 구한다:

$$\frac{\partial \mathcal{L}}{\partial x_i} = 0 \quad (i=1,\dots,n), \qquad \frac{\partial \mathcal{L}}{\partial \lambda} = 0$$

마지막 식은 $g(x) = 0$(제약식)과 동치이다.

## 4. 간단한 예시

$f(x,y) = xy$를 최대화하되 $x + y = 10$이라는 제약이 있다고 하자.

라그랑지안: $\mathcal{L}(x,y,\lambda) = xy - \lambda(x + y - 10)$

미분하면:

$$\frac{\partial \mathcal{L}}{\partial x} = y - \lambda = 0, \quad \frac{\partial \mathcal{L}}{\partial y} = x - \lambda = 0, \quad \frac{\partial \mathcal{L}}{\partial \lambda} = -(x + y - 10) = 0$$

따라서 $x = y = 5$이고 최대값은 25이다.

## 5. 제약이 여러 개인 경우

제약이 $m$개이면:

$$\mathcal{L}(x, \lambda_1, \dots, \lambda_m) = f(x) - \sum_{j=1}^m \lambda_j g_j(x)$$

각 $\lambda_j$에 대해 편미분하여 풀면 된다.

## 6. 통계 검정에서의 적용

최적 검정 문제에서는:

**목적:** $E_{\theta_1}[\phi(X)]$ 최대화  
**제약:** $E_{\theta_0}[\phi(X)] = \alpha$ (크기 조건) 및 불편성 조건

라그랑지안은:

$$\mathcal{L}(\phi) = E_{\theta_1}[\phi(X)] - k_1 E_{\theta_0}[\phi(X)] - k_2 E_{\theta_0}\left[\phi(X)\frac{p'_{\theta_0}(X)}{p_{\theta_0}(X)}\right]$$

이를 정리하면:

$$\mathcal{L}(\phi) = E_{\theta_0}\left[\phi(X)\left(\frac{p_{\theta_1}(X)}{p_{\theta_0}(X)} - k_1 - k_2 \frac{p'_{\theta_0}(X)}{p_{\theta_0}(X)}\right)\right]$$

**Pointwise 최적화**: $0 \le \phi(X) \le 1$이므로, 각 점 $x$에서 integrand의 부호에 따라:

- $A(x) > 0$ ⟹ $\phi(x) = 1$ (기각)
- $A(x) < 0$ ⟹ $\phi(x) = 0$ (채택)

따라서 최적 검정은:

$$\phi^*(x) = \begin{cases} 1, & A(x) > 0 \\ 0, & A(x) < 0 \end{cases}$$

## 7. 일반 절차

통계 검정에서 라그랑주 승수법을 적용할 때의 절차:

1. **목적함수 설정:** 대립가설에서의 검정력 $E_{\theta_1}[\phi(X)]$
2. **제약식 설정:** 크기 조건, 불편성 조건 등
3. **라그랑지안 작성:** 목적함수 - (승수) × (제약식)
4. **Pointwise 최적화:** 각 $x$마다 $\phi(x) \in \{0,1\}$ 결정
5. **승수 결정:** 제약식을 실제로 만족하도록 조정
6. **기각역 정리:** 최종 형태를 임계값으로 표현

## 8. 직관적 의미

라그랑주 승수법은 통계 검정에서:

> **제약을 만족하는 범위 안에서 최대한 기각확률을 키우는 방향으로, 각 표본점에 대해 기각/비기각을 배분하는 방법**

즉, 대립가설에서 유리한 점에는 높은 기각확률을, 불리한 점에는 낮은 기각확률을 할당하면서도 크기와 불편성을 동시에 만족시킨다.

## 9. 주의할 점

라그랑주 승수법은 **최적 검정의 후보를 찾는 도구**이지, 항상 완전한 증명은 아니다. 다음이 추가로 필요할 수 있다:

- 미분과 적분 교환 정당화
- 제약을 만족하는 승수의 존재성
- 찾은 해가 실제 최대임을 보이는 논증
- 경계에서의 randomization 처리
