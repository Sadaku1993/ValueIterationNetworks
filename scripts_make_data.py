#coding:utf-8


import numpy as np
import random
import pickle

from obstacle import *
from graph import *

def sample_graph_traj(graph, num_trajs):
    domsize = graph.domsize

    traj = 0
    states_xy = []
    trial = 0

    # 最短軌道をnum_trajs分作成 
    while traj < num_trajs:
        trial += 1
        # 試行回数が一定値以上を超えたら終了
        if trial > num_trajs * 2:
            break
        # 初期位置
        pos = (random.randint(1, domsize[0] - 1), random.randint(1, domsize[1] - 1))
        # 最短軌道を生成
        path = graph.get_shortest_path(pos)
        if path is None:
            continue
        states_xy.append(path)
        traj = traj + 1
        
    return states_xy

def extract_action(traj):
    actions = []
    for i in xrange(len(traj) - 1):
        s0 = traj[i]
        s1 = traj[i+1]
        dx = s1[0] - s0[0]
        dy = s1[1] - s0[1]

        if dx == -1 and dy == -1: actions.append(0)
        if dx == 0 and dy == -1: actions.append(1)
        if dx == 1 and dy == -1: actions.append(2)
        if dx == -1 and dy == 0: actions.append(3)
        if dx == 1 and dy == 0: actions.append(4)
        if dx == -1 and dy == 1: actions.append(5)
        if dx == 0 and dy == 1: actions.append(6)
        if dx == 1 and dy == 1: actions.append(7)

    return actions


def main():
    size_1 = 16
    size_2 = 16
    dom_size = (size_1, size_2)
    max_traj_len = size_1 + size_2
    num_domains = 100       # number of trajectory
    max_obs = 40            # max obstacle number
    max_obs_size = 1.0      # max obstacle size[m]
    num_trajs = 7
    maxSamples = num_domains * num_trajs * max_traj_len / 2

    im_data = np.zeros((maxSamples, size_2, size_1), dtype=np.uint8)
    value_data = np.zeros((maxSamples, size_2, size_1), dtype=np.uint8)
    state_xy_data = np.zeros((maxSamples, 2), dtype=np.int32)
    label_data = np.zeros((maxSamples), dtype=np.int32)

    num_samples = 0
    dom = 1
    while dom <= num_domains:
        goal = (random.randint(1, size_1 -1), random.randint(1, size_2-1))
        obs = Obstacle(dom_size, goal, max_obs_size)
        n_obs = obs.add_n_obs(random.randint(0, max_obs))
        # n_obs = obs.add_n_obs_2(random.randint(0, max_obs))
        
        if n_obs == 0:
            # print ("no obstacles added ")
            continue
        obs.add_border()

        im = obs.getimage()
        
        print ("Goal:", goal[1], goal[0])
        obs.show_image()

        cv2.imshow("test", cv2.resize(255-im*255, (300, 300), interpolation=cv2.INTER_NEAREST))
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        G = Graph(im, goal)
        value_prior = G.get_reward_prior()              # 最適報酬
        states_xy = sample_graph_traj(G, num_trajs)     # 状態履歴

        if len(states_xy) != num_trajs:
            print ('no trajectory added')
            continue

        for i in xrange(len(states_xy)):
            if len(states_xy[i]) > 0:
                actions = extract_action(states_xy[i])
                ns = len(actions)
                im_data[num_samples:num_samples+ns] = im
                value_data[num_samples:num_samples+ns] = value_prior
                state_xy_data[num_samples:num_samples+ns] = np.array(states_xy[i][0:-1])
                label_data[num_samples:num_samples + ns] = np.array(actions)
                num_samples = num_samples + ns

                print (actions)

        data = {}
        data['im'] = im_data[0:num_samples]             # グリッドセル情報
        data['value'] = value_data[0:num_samples]       # 最適報酬
        data['state'] = state_xy_data[0:num_samples]    # 状態履歴
        data['label'] = label_data[0:num_samples]       # 行動履歴
        with open('map_data.pkl', mode='wb') as f:
            pickle.dump(data, f)

        print("")

if __name__ == "__main__":
    main()
