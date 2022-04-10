from template import Agent
import time
from Splendor.splendor_model import *
import pickle

THINKTIME = 0.95





class myAgent(Agent):
    def save_obj(self, obj, name ):
        with open('./agents/SplendorForFun/'+name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        with open('./agents/SplendorForFun/'+name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def recursive_init(self, cur_pos, cur_str):

        if cur_pos>4:
            #self.q_table[cur_str] = [1.5, 2, 3, 50, 25]
            #self.q_table[cur_str] = [1.5, 3, 4.5, 2, 4, 3, 50, 25]
            self.q_table[cur_str] = [0, 0, 0, 0, 0, 0, 0, 0]
            #self.q_table[cur_str] = [0, 10, 20, 0, 10, 10, 50, 25]
            #self.r_table[cur_str] = [3, 2, 1, 10, 5]
            #self.r_table[cur_str] = [1.5, 2, 2, 10, 5]
            return
        for i in range(9):
            temp_str = cur_str+str(i)
            self.recursive_init(cur_pos+1, temp_str)

    def __init__(self, _id):
        super().__init__(_id)
        #black red green blue white: collect_diff collect_same reserve buy_available buy_reserve
        # {'00000':[], '00001':[], ...}
        #number of gems; different color of gems
        #two situation: can buy_available/cannot buy_available
        #enlarge the range of action
        #score
        self.q_table = {}
        self.count = 0
        self.q_table = self.load_obj('q1_table_it')
        self.r_table = [0, 2, 5, 0, 2, 2, 100, 25]
        #self.recursive_init(0, '')
        self.epsilon = 0.9
        self.alpha = 0.1
        self.gamma = 0.8


    def SelectAction(self, actions, gameState):
        startTime = time.time()
        #openQ = PriorityQueue()

        #action_map = {'collect_diff':0, 'collect_same':1, 'reserve':2, 'buy_available':3, 'buy_reserve':4}
        action_map = {'collect_diff1':0, 'collect_diff2':1, 'collect_diff3':2, 'collect_same1':3, 'collect_same2':4, 'reserve':5, 'buy_available':6, 'buy_reserve':7}
        cur_cards = gameState.agents[self.id].cards
        cur_state = str(len(cur_cards['black']))+str(len(cur_cards['red']))+str(len(cur_cards['green']))+str(len(cur_cards['blue']))+str(len(cur_cards['white']))

        self.count += 1
        if random.uniform(0, 1)>self.epsilon or (sum(self.q_table[cur_state])==0):
            gem_flag = 1
            for action in actions:
                if 'collect' in action['type']:
                    num = 0
                    for colour, count in action['collected_gems'].items():
                        num += count
                    if num==2 and gem_flag==1:
                        gem_flag = 2
                    elif num==3 and gem_flag<3:
                        gem_flag = 3
                        break
            cur_action = random.choice(actions)
            flag = False
            if 'collect' in cur_action['type']:
                num = 0
                for colour,count in cur_action['collected_gems'].items():
                    num += count
                if num < gem_flag:
                    flag = True
            while cur_action['type']=='pass' or flag:
                cur_action = random.choice(actions)
                flag = False
                if 'collect' in cur_action['type']:
                    num = 0
                    for colour,count in cur_action['collected_gems'].items():
                        num += count
                    if num < gem_flag:
                        flag = True
            action_type = cur_action['type']
            if 'collect' in action_type:
                num = 0
                for colour,count in cur_action['collected_gems'].items():
                    num += count
                action_type = action_type+str(num)
            cur_simple_action = action_map[action_type]
        else:
            max_q = -1
            if actions[0]['type']!="pass":
                action_type = actions[0]['type']
                if 'collect' in action_type:
                    num = 0
                    for colour,count in actions[0]['collected_gems'].items():
                        num += count
                    action_type = action_type+str(num)
                cur_simple_action = action_map[action_type]
                max_q = self.q_table[cur_state][cur_simple_action]
                cur_action = actions[0]
            else:
                cur_simple_action = -1
                cur_action = None
            for action in actions:
                if action['type']=='pass':
                    continue
                action_type = action['type']
                if 'collect' in action_type:
                    num = 0
                    for colour,count in action['collected_gems'].items():
                        num += count
                    action_type = action_type+str(num)
                temp_action = action_map[action_type]
                if self.q_table[cur_state][temp_action] > max_q:
                    max_q = self.q_table[cur_state][temp_action]
                    cur_simple_action = temp_action
                    cur_action = action
            if cur_action==None:
                cur_action = random.choice(actions)
                while cur_action['type']=='pass':
                    cur_action = random.choice(actions)
                action_type = cur_action['type']
                if 'collect' in action_type:
                    num = 0
                    for colour,count in cur_action['collected_gems'].items():
                        num += count
                    action_type = action_type+str(num)
                cur_simple_action = action_map[action_type]
        gr = SplendorGameRule(len(gameState.agents))
        newState = gr.generateSuccessor(copy.deepcopy(gameState), cur_action, self.id)
        new_cards = newState.agents[self.id].cards
        new_state = str(len(new_cards['black']))+str(len(new_cards['red']))+str(len(new_cards['green']))+str(len(new_cards['blue']))+str(len(new_cards['white']))
        #update
        review = 0
        if cur_simple_action==5:
            review += 1
            review += cur_action['card'].points*self.r_table[cur_simple_action]
        elif cur_simple_action==6 or cur_simple_action==7:
            review += cur_action['card'].points*self.r_table[cur_simple_action]
        elif cur_simple_action>=0 and cur_simple_action<=4:
            review = self.r_table[cur_simple_action]
        else:
            review = 0
        #self.q_table[cur_state][cur_simple_action] += self.alpha * (self.r_table[cur_state][cur_simple_action] + self.gamma * max(self.q_table[new_state]) - self.q_table[cur_state][cur_simple_action])
        self.q_table[cur_state][cur_simple_action] += self.alpha * (review + self.gamma * max(self.q_table[new_state]) - self.q_table[cur_state][cur_simple_action])
        if self.count==1:
            self.save_obj(self.q_table, 'q1_table_step1')
        '''
        gr = SplendorGameRule(len(gameState.agents))
        for action in actions:
            rootState = deepcopy(gameState)
            preState = deepcopy(gameState)
            newState = gr.generateSuccessor(rootState, action, self.id)
            openQ.push(action, SplendorHeuristic(preState, newState, self.id))
            if time.time() - startTime >= THINKTIME:
                break
        retAction = openQ.pop()
        '''
        next_score = newState.agents[self.id].score
        if next_score>=15:
            self.save_obj(self.q_table, 'q1_table_it')

        return cur_action
