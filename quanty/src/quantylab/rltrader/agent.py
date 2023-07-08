import numpy as np
from quantylab.rltrader import utils


class Agent:
    # 에이전트 상태가 구성하는 값 개수
    # 현재 리소스대비 요청(현재 자기 리소스 상태, 몇퍼센트 쓰고있는지), 스케일링한 값(얼마나 스케일링했는지 unit), ratio_stay
    STATE_DIM = 3

    # 행동
    ACTION_UP = 0  # upscaling
    ACTION_DOWN = 1  # downscaling
    ACTION_STAY = 2  # stay

    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_STAY]
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(self, environment, initial_request, min_scaling_unit, max_scaling_unit):
        self.environment = environment
        self.initial_request = initial_request  # 초기 request
        self.upper_bound = 0.7
        self.lower_bound = 0.3
        # 최소 scaling unit, 최대 scaling unit
        self.min_scaling_unit = min_scaling_unit
        self.max_scaling_unit = max_scaling_unit

        # Agent 클래스의 속성
        self.request = initial_request  # 현재 request
        self.resource = 0 # 첫 사용량
        self.num_scaling = 1  # 처음부터 지금까지 scaling 몇번 했는지 (stay 제외)

        # reward : 리소스 사용량 (퍼센테이지) / num_scaling (스케일링 많이 했을수록 안좋음)
        self.reward = 0
        self.num_up = 0  # upscaling 횟수
        self.num_down = 0  # downscaling 횟수
        self.num_stay = 0  # stay 횟수
        self.num_recent_stay = 0
        # Agent 클래스의 상태
        self.ratio_stay = 1  # stay가 많을수록 좋다
        self.resource_usage = 0  # 현재 리소스대비 요청(현재 자기 리소스 상태, 몇퍼센트 쓰고있는지)
        self.scaling_unit = 0  # 스케일링한 값(얼마나 스케일링했는지 unit)

    def reset(self):
        self.request = self.initial_request
        self.num_scaling = 1
        self.reward = 0 # self.initial_request / self.initial_resource / self.num_scaling
        self.num_up = 0
        self.num_down = 0
        self.num_stay = 1
        self.num_recent_stay = 0
        self.ratio_stay = 0
        self.resource_usage = self.resource / self.initial_request
        self.scaling_unit = 0

    def set_request(self, request):
        self.initial_request = request

    # 현재 리소스대비 요청(현재 자기 리소스 상태, 몇퍼센트 쓰고있는지), 스케일링한 값(얼마나 스케일링했는지 unit) , ratio_stay
    def get_states(self):
        self.ratio_stay = self.num_stay / self.num_scaling
        return (
            self.ratio_stay,
            self.reward,
            self.scaling_unit
        )

    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0.

        pred = pred_policy
        if pred is None:
            pred = pred_value

        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

            # if pred_policy is not None:
            #     if np.max(pred_policy) - np.min(pred_policy) < 0.05:
            #         epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)
        else:
            exploration = False
            action = np.argmax(pred)

        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    def validate_action(self, action):
        if action == Agent.ACTION_DOWN:
            # down 했을때 음수인지 확인
            if (self.environment.get_resource() / self.request) - self.scaling_unit < 0:
                return False
        return True

    def decide_scaling_unit(self, confidence):
        if np.isnan(confidence):
            return (1 - (self.environment.get_resource() / self.request))    
        scaling_unit = max(((self.environment.get_resource() / self.request)-1) * confidence, (1 - (self.environment.get_resource() / self.request)) * confidence)
        # print("confidence: ", confidence, " scaling_unit :", scaling_unit)
        self.scaling_unit = max(round(float(scaling_unit),3), 0)
        return max(round(float(scaling_unit),3), 0)

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_STAY

        # 환경에서 현재 resource 값 얻기
        curr_resource = self.environment.get_resource()

        reward = 0

        # upscaling
        if action == Agent.ACTION_UP:
            # up할 단위를 판단
            # self.scaling_unit = self.decide_scaling_unit(confidence)
            self.scaling_unit = 0.1
            request = self.request
            self.num_scaling += 1 # scaling 횟수 증가
            self.num_up += 1 # upscaling 횟수 증가
            self.num_recent_stay = 0

            usage = curr_resource / request
            scaling_usage = (curr_resource / request)  + self.scaling_unit

            # if usage < self.upper_bound : # upscaling할 필요 없는 상황
            #     reward = -1
            # else:
            if usage > self.upper_bound:
                if (scaling_usage > self.lower_bound) & (scaling_usage < self.upper_bound):
                    reward = 1
                else:
                    reward = 0.5
            # print("[UP] reward : ", reward, " scaling_unit: ", self.scaling_unit, " usage : ", usage, " scaling_usage : ", scaling_usage)
            

        # downscaling
        elif action == Agent.ACTION_DOWN:
            # donw할 단위를 판단
            # self.scaling_unit = self.decide_scaling_unit(confidence)
            self.scaling_unit = 0.1
            request = self.request
            self.num_scaling += 1
            self.num_down = self.num_down + 1
            usage = curr_resource / request
            scaling_usage = (curr_resource / request) - self.scaling_unit

            # if usage > self.upper_bound: # downscaling하면 안되는 상황
            #     self.num_recent_stay = 0
            #     reward = -1
            if usage < self.upper_bound:
                if self.num_recent_stay > 3:
                    reward = 1
                    self.num_recent_stay = 0
                else:
                    reward = 0.5
                    self.num_recent_stay = 0
            # print("[DOWN] reward : ", reward, " scaling_unit: ", self.scaling_unit, " usage : ", usage, " scaling_usage : ", scaling_usage)


        # 관망
        elif action == Agent.ACTION_STAY:
            self.num_stay += 1  # 관망 횟수 증가
            self.num_recent_stay += 1

        self.reward = reward
        return reward
