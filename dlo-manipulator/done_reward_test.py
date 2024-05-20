#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2024/5/19 15:37
# @Author       : Wang Song
# @File         : done_reward_test.py
# @Software     : PyCharm
# @Description  :
import numpy as np
import matplotlib.pyplot as plt

# 定义奖励函数
def calculate_smooth_done_reward(dlo_error):
    # 阶跃奖励的平滑版本，使用指数衰减函数
    done_rewards = [0.4, 0.8, 1.2, 2, 4, 10]
    done_thresholds = [0.07, 0.05, 0.03, 0.02, 0.015, 0.01]
    smooth_done_reward = sum(reward * np.exp(-50 * (dlo_error - threshold))
                             for reward, threshold in zip(done_rewards, done_thresholds))
    return smooth_done_reward

# 生成误差从0.1到0.01的变化
errors = np.linspace(0.1, 0.01, 100)
rewards = [calculate_smooth_done_reward(error) for error in errors]

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(errors, rewards, label='Smooth Done Reward')
plt.xlabel('Error')
plt.ylabel('Smooth Done Reward')
plt.title('Smooth Done Reward vs Error')
plt.legend()
plt.grid(True)
plt.show()