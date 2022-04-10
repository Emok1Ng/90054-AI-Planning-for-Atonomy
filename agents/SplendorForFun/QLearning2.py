from copy import deepcopy
#import copy.deepcopy
from template import Agent
import time
import heapq
#from utils import PriorityQueue
from Splendor.splendor_model import *
import pickle

THINKTIME = 0.95

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


def GemHeuristic(rootState, gameState, agentID):
    h_value = 0


def SplendorHeuristic(rootState, gameState, agentId):
    scoreWeight = 1000
    zeroCardWeight = 200
    previousGems = rootState.agents[agentId].gems
    previousScore = rootState.agents[agentId].score
    previousCards = 0
    for i in rootState.agents[agentId].cards.keys():
        if i != 'yellow':
            #previousCards += len(rootState.agents[agentId].cards[i])
            for card in rootState.agents[agentId].cards[i]:
                previousCards += card.points
    currentGems = gameState.agents[agentId].gems
    currentScore = gameState.agents[agentId].score
    currentCards = 0
    for i in gameState.agents[agentId].cards.keys():
        if i != 'yellow':
            currentCards += len(gameState.agents[agentId].cards[i])
    actionGems = {c: 0 for c in previousGems.keys()}
    for i in actionGems.keys():
        actionGems[i] = currentGems[i] - previousGems[i]
    actionScore = currentScore - previousScore
    boardGemCount = {c: 0 for c in actionGems.keys()}
    board = gameState.board
    dealt = board.dealt
    for i in range(len(dealt[0])):
        card = dealt[0][i]
        if card is not None:
            for colour, number in card.cost.items():
                boardGemCount[colour] += number
    gemScore = 0
    for i in actionGems.keys():
        if actionGems[i] != 0:
            gemScore += actionGems[i] * boardGemCount[i]
    return (gemScore + actionScore * scoreWeight + (currentCards - previousCards) * zeroCardWeight)

class myAgent(Agent):
    def save_obj(self, obj, name ):
        with open('./agents/SplendorForFun/'+name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        with open('./agents/SplendorForFun/'+name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def recursive_init(self, cur_pos, cur_str):

        #if cur_pos>10:
        #if cur_pos>5:
        if cur_pos>2:
            #self.q_table[cur_str] = [1.5, 2, 3, 50, 25]
            #self.q_table[cur_str] = [1.5, 3, 4.5, 2, 4, 3, 50, 25]
            #self.q_table[cur_str] = [0, 2, 5, 0, 2, 0, 20, 20, 20, 20, 20, 2]
            #self.q_table[cur_str] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #self.q_table[cur_str] = [0, 2, 5, 0, 2, 2, 100, 25]
            #self.r_table[cur_str] = [3, 2, 1, 10, 5]
            #self.r_table[cur_str] = [1.5, 2, 2, 10, 5]
            if ord(cur_str[-1])-ord('a')<5:
                #self.q_table[cur_str] = [5, 0, 20, 10, 5, 5]
                self.q_table[cur_str] = [0, 0, 0, 0, 0, 0]
                self.q_neg_table[cur_str] = [0, 0, 0, 0, 0, 0]
                self.r_table[cur_str] = [5, 0, 20, 10, 5, 5]
            elif ord(cur_str[-1])-ord('a')<10:
                #self.q_table[cur_str] = [5, 0, 20, 20, 10, 5]
                self.q_table[cur_str] = [0, 0, 0, 0, 0, 0]
                self.q_neg_table[cur_str] = [0, 0, 0, 0, 0, 0]
                self.r_table[cur_str] = [5, 0, 20, 20, 10, 5]
            else:
                #self.q_table[cur_str] = [15, 0, 10, 20, 20, 5]
                self.q_table[cur_str] = [0, 0, 0, 0, 0, 0]
                self.q_neg_table[cur_str] = [0, 0, 0, 0, 0, 0]
                self.r_table[cur_str] = [15, 0, 10, 20, 20, 5]
            return
        '''
        if cur_pos<=4:
            #考虑已经购买的卡牌颜色
            for i in range(13):
                temp_str = cur_str+chr(ord('a')+i)
                self.recursive_init(cur_pos+1, temp_str)
        '''
        if cur_pos==0:
            for i in range(11):
                temp_str = cur_str+chr(ord('a')+i)
                self.recursive_init(cur_pos+1, temp_str)
        elif cur_pos==1:
            for i in range(36):
                if i>25:
                    temp_str = cur_str+chr(ord('A')+i-26)
                else:
                    temp_str = cur_str+chr(ord('a')+i)
                self.recursive_init(cur_pos+1, temp_str)
        #elif cur_pos <=9 and cur_pos>4:
        #    for i in range(7):
        #        temp_str = cur_str+str(i)
        #        self.recursive_init(cur_pos+1, temp_str)
        else:
            for i in range(26):
                temp_str = cur_str+chr(ord('a')+i)
                self.recursive_init(cur_pos+1, temp_str)
    def recursive_init2(self, cur_pos, cur_str):

        #if cur_pos>10:
        #if cur_pos>5:
        if cur_pos>2:
            #self.q_table[cur_str] = [1.5, 2, 3, 50, 25]
            #self.q_table[cur_str] = [1.5, 3, 4.5, 2, 4, 3, 50, 25]
            #self.q_table[cur_str] = [0, 2, 5, 0, 2, 0, 20, 20, 20, 20, 20, 2]
            #self.q_table[cur_str] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #self.q_table[cur_str] = [0, 2, 5, 0, 2, 2, 100, 25]
            #self.r_table[cur_str] = [3, 2, 1, 10, 5]
            #self.r_table[cur_str] = [1.5, 2, 2, 10, 5]
            if ord(cur_str[-1])-ord('a')<5:
                temp = [5, 0, 20, 10, 5, 5]
                if ord(cur_str[0])-ord('a')>7:
                    temp[0] = 0
                temp1 = [temp[i]+self.q_table[cur_str][i] for i in range(6)]
                self.q_table[cur_str] = temp1
                #self.q_table[cur_str] = [0, 0, 0, 0, 0, 0]
                #self.q_neg_table[cur_str] = [0, 0, 0, 0, 0, 0]
                #self.r_table[cur_str] = [5, 0, 20, 10, 5, 5]
            elif ord(cur_str[-1])-ord('a')<10:
                temp = [5, 0, 20, 20, 10, 5]
                if ord(cur_str[0])-ord('a')>7:
                    temp[0] = 0
                temp1 = [temp[i]+self.q_table[cur_str][i] for i in range(6)]
                self.q_table[cur_str] = temp1
                #self.q_table[cur_str] = [0, 0, 0, 0, 0, 0]
                #self.q_neg_table[cur_str] = [0, 0, 0, 0, 0, 0]
                #self.r_table[cur_str] = [5, 0, 20, 20, 10, 5]
            else:
                temp = [15, 0, 10, 20, 20, 5]
                if ord(cur_str[0])-ord('a')>7:
                    temp[0] = 0
                temp1 = [temp[i]+self.q_table[cur_str][i] for i in range(6)]
                self.q_table[cur_str] = temp1
                #self.q_table[cur_str] = [0, 0, 0, 0, 0, 0]
                #self.q_neg_table[cur_str] = [0, 0, 0, 0, 0, 0]
                #self.r_table[cur_str] = [15, 0, 10, 20, 20, 5]
            return
        '''
        if cur_pos<=4:
            #考虑已经购买的卡牌颜色
            for i in range(13):
                temp_str = cur_str+chr(ord('a')+i)
                self.recursive_init(cur_pos+1, temp_str)
        '''
        if cur_pos==0:
            for i in range(11):
                temp_str = cur_str+chr(ord('a')+i)
                self.recursive_init2(cur_pos+1, temp_str)
        elif cur_pos==1:
            for i in range(36):
                if i>25:
                    temp_str = cur_str+chr(ord('A')+i-26)
                else:
                    temp_str = cur_str+chr(ord('a')+i)
                self.recursive_init2(cur_pos+1, temp_str)
        #elif cur_pos <=9 and cur_pos>4:
        #    for i in range(7):
        #        temp_str = cur_str+str(i)
        #        self.recursive_init(cur_pos+1, temp_str)
        else:
            for i in range(26):
                temp_str = cur_str+chr(ord('a')+i)
                self.recursive_init2(cur_pos+1, temp_str)

    def __init__(self, _id):
        super().__init__(_id)
        #black red green blue white: collect_diff collect_same reserve buy_available buy_reserve
        # {'00000':[], '00001':[], ...}
        #number of gems; different color of gems
        #two situation: can buy_available/cannot buy_available
        #enlarge the range of action
        #score
        #self.q_table = {}
        #self.q_neg_table = {}
        self.count = 0
        self.q_table = self.load_obj('q1_table_it7')
        self.q_neg_table = self.load_obj('q_neg_table')
        #self.q_neg_table2 = copy.deepcopy(self.q_neg_table)
        self.r_table = self.load_obj('r_table')
        #self.r_table = [0, 2, 5, 0, 2, 0, 100, 2]
        #self.r_table = [5, 2, 20, 20, 20, 5]
        #self.r_table = {}
        #self.recursive_init(0, '')
        #self.recursive_init2(0, '')
        self.epsilon = 0.5
        self.alpha = 0.1
        self.gamma =  0.8
        #self.gamma = 0.5
        self.beta = 0.8

    def numToChar(self, num):
        if num > 25:
            return chr(ord('A')+num-26)
        else:
            return chr(ord('a')+num)

    def SelectAction(self, actions, gameState):
        startTime = time.time()
        gr = SplendorGameRule(len(gameState.agents))
        #openQ = PriorityQueue()

        #action_map = {'collect_diff':0, 'collect_same':1, 'reserve':2, 'buy_available':3, 'buy_reserve':4}
        #action_map = {'collect_diff1':0, 'collect_diff2':1, 'collect_diff3':2, 'collect_same1':3, 'collect_same2':4, 'reserve':5, 'buy_available':6, 'buy_reserve':7}
        #action_map = {'collect_diff1':0, 'collect_diff2':1, 'collect_diff3':2, 'collect_same1':3, 'collect_same2':4, 'reserve':5, 'buy_available_black':6, 'buy_available_red':7, 'buy_available_green':8, 'buy_available_blue':9, 'buy_available_white':10, 'buy_reserve':11}
        action_map = {'collected_gems':0, 'reserve':1, 'buy_tire1':2, 'buy_tire2':3, 'buy_tire3':4, 'buy_reserve':5}
        #action_map={'collect1_black': 0, 'collect1_red': 1, 'collect1_green': 2, 'collect1_blue': 3, 'collect1_white': 4, 'collect2_black_black': 5, 'collect2_black_red': 6, 'collect2_black_green': 7, 'collect2_black_blue': 8, 'collect2_black_white': 9, 'collect2_red_red': 10, 'collect2_red_green': 11, 'collect2_red_blue': 12, 'collect2_red_white': 13, 'collect2_green_green': 14, 'collect2_green_blue': 15, 'collect2_green_white': 16, 'collect2_blue_blue': 17, 'collect2_blue_white': 18, 'collect2_white_white': 19, 'collect3_black_black_black': 20, 'collect3_black_black_red': 21, 'collect3_black_black_green': 22, 'collect3_black_black_blue': 23, 'collect3_black_black_white': 24, 'collect3_black_red_red': 25, 'collect3_black_red_green': 26, 'collect3_black_red_blue': 27, 'collect3_black_red_white': 28, 'collect3_black_green_green': 29, 'collect3_black_green_blue': 30, 'collect3_black_green_white': 31, 'collect3_black_blue_blue': 32, 'collect3_black_blue_white': 33, 'collect3_black_white_white': 34, 'collect3_red_red_red': 35, 'collect3_red_red_green': 36, 'collect3_red_red_blue': 37, 'collect3_red_red_white': 38, 'collect3_red_green_green': 39, 'collect3_red_green_blue': 40, 'collect3_red_green_white': 41, 'collect3_red_blue_blue': 42, 'collect3_red_blue_white': 43, 'collect3_red_white_white': 44, 'collect3_green_green_green': 45, 'collect3_green_green_blue': 46, 'collect3_green_green_white': 47, 'collect3_green_blue_blue': 48, 'collect3_green_blue_white': 49, 'collect3_green_white_white': 50, 'collect3_blue_blue_blue': 51, 'collect3_blue_blue_white': 52, 'collect3_blue_white_white': 53, 'collect3_white_white_white': 54, 'reserve': 55, 'buy_available_black': 56, 'buy_available_red': 57, 'buy_available_green': 58, 'buy_available_blue': 59, 'buy_available_white': 60, 'but_reserve': 61}

        cur_gems = gameState.agents[self.id].gems
        cur_cards = gameState.agents[self.id].cards
        cur_score = gameState.agents[self.id].score
        #cur_state = str(cur_gems['black'])+str(cur_gems['red'])+str(cur_gems['green'])+str(cur_gems['blue'])+str(cur_gems['white'])+str(chr(ord('a')+cur_score))
        #cur_state = str(chr(ord('a')+cur_gems['black']+len(cur_cards['black'])))+str(chr(ord('a')+cur_gems['red']+len(cur_cards['red'])))+str(chr(ord('a')+cur_gems['green']+len(cur_cards['green'])))+str(chr(ord('a')+cur_gems['blue']+len(cur_cards['blue'])))+str(chr(ord('a')+cur_gems['white']+len(cur_cards['white'])))+str(chr(ord('a')+cur_score))
        cur_gems_num = cur_gems['black']+cur_gems['red']+cur_gems['green']+cur_gems['blue']+cur_gems['white']
        cur_cards_num = len(cur_cards['black'])+len(cur_cards['red'])+len(cur_cards['green'])+len(cur_cards['blue'])+len(cur_cards['white'])
        cur_state = str(self.numToChar(cur_gems_num))+str(self.numToChar(cur_cards_num))+str(chr(ord('a')+cur_score))
        flag = 1
        self.count += 1
        if random.uniform(0, 1)>self.epsilon or (sum(self.q_table[cur_state])==0):
            flag = 0
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
            logic_actions = list(action_map.keys())
            while len(logic_actions)>0:
                max_action = random.choice(logic_actions)
                logic_actions.remove(max_action)
                cur_simple_action = -1
                cur_action = None
                cur_heuristic = -1
                for action in actions:
                    if action['type']=='pass':
                        continue
                    action_type = action['type']
                    if 'collect' in action_type:
                        action_type  = 'collected_gems'
                    if 'buy_available'==action_type:
                        action_type = 'buy_tire'+str(action['card'].deck_id+1)
                    temp_action = action_map[action_type]
                    if action_type!=max_action:
                        continue
                    #if self.q_table[cur_state][temp_action] > max_q:
                    #    max_q = self.q_table[cur_state][temp_action]
                    #    cur_simple_action = temp_action
                    #    cur_action = action
                    #elif self.q_table[cur_state][temp_action] == max_q:
                    if max_action=='collected_gems' or max_action=='reserve' or max_action=='buy_reserve':
                    #if max_action=='collected_gems':
                        num = 0
                        if max_action=='collected_gems':
                            for colour,count in action['collected_gems'].items():
                                num += count
                        rootState = copy.deepcopy(gameState)
                        preState = copy.deepcopy(gameState)
                        newState = gr.generateSuccessor(rootState, action, self.id)
                        #nextState = gr.generateSuccessor(currentState, action, self.id)
                        temp_heuristic = SplendorHeuristic(preState, newState, self.id)+num
                        #if max_action=='collected_gems':
                        #    temp_heuristic += num * 1000
                        #temp_heuristic = num*100
                        if temp_heuristic>cur_heuristic:
                            cur_heuristic = temp_heuristic
                            cur_action = action
                            cur_simple_action = temp_action
                    elif max_action=='buy_tire1' or max_action=='buy_tire2':
                        cur_score = gameState.agents[self.id].score
                        nobles = gameState.board.nobles
                        colour_count = {'black':0, 'red':0, 'green':0, 'blue':0, 'white':0}
                        cur_cards = gameState.agents[self.id].cards
                        for noble in nobles:
                            for key in noble[1].keys():
                                colour_count[key] += 1
                        for key in colour_count.keys():
                            if colour_count[key]<len(cur_cards[key]):
                                colour_count[key] = 0
                            else:
                                colour_count[key] -= len(cur_cards[key])
                        rootState = copy.deepcopy(gameState)
                        preState = copy.deepcopy(gameState)
                        newState = gr.generateSuccessor(rootState, action, self.id)
                        #nextState = gr.generateSuccessor(currentState, action, self.id)
                        #temp_heuristic = SplendorHeuristic(preState, newState, self.id)
                        if action['card'].points + cur_score >= 15:
                            temp_heuristic = 10000000
                        elif action['card'].points >=10:
                            temp_heuristic = SplendorHeuristic(preState, newState, self.id)*action['card'].points
                        elif colour_count[action['card'].colour]>0:
                            temp_heuristic = SplendorHeuristic(preState, newState, self.id)*colour_count[action['card'].colour]
                        else:
                            temp_heuristic = SplendorHeuristic(preState, newState, self.id)*action['card'].points
                        if temp_heuristic>cur_heuristic:
                            cur_heuristic = temp_heuristic
                            cur_action = action
                            cur_simple_action = temp_action
                    elif max_action=='buy_tire3':
                        cur_score = gameState.agents[self.id].score
                        rootState = copy.deepcopy(gameState)
                        preState = copy.deepcopy(gameState)
                        newState = gr.generateSuccessor(rootState, action, self.id)
                        #nextState = gr.generateSuccessor(currentState, action, self.id)
                        #temp_heuristic = SplendorHeuristic(preState, newState, self.id)
                        temp_heuristic = SplendorHeuristic(preState, newState, self.id)*action['card'].points
                        if temp_heuristic>cur_heuristic:
                            cur_heuristic = temp_heuristic
                            cur_action = action
                            cur_simple_action = temp_action
                if cur_action!=None:
                    break



            if cur_action==None:
                cur_action = random.choice(actions)

            if cur_action['type']=='pass':
                return cur_action
            flag = False
            if 'collect' in cur_action['type']:
                num = 0
                for colour,count in cur_action['collected_gems'].items():
                    num += count
                if num < gem_flag:
                    flag = True
            while flag:
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
            #    num = 0
                action_type='collected_gems'
            if 'buy_available'==action_type:
                #action_type = action_type+'_'+cur_action['card'].colour
                action_type = 'buy_tire'+str(cur_action['card'].deck_id+1)
            cur_simple_action = action_map[action_type]

        else:
            #select the logic action with max q value
            temp_max_q = -1
            max_action = ''
            max_action_list = []
            for at in action_map.keys():
                #if at=='reserve':
                #    continue
                temp_simple_action = action_map[at]
                #if self.q_table[cur_state][temp_simple_action]>temp_max_q:
                #    temp_max_q = self.q_table[cur_state][temp_simple_action]
                #    max_action = at
                max_action_list.append((self.q_table[cur_state][temp_simple_action], at))
            max_action_list = sorted(max_action_list, key=lambda x:x[0], reverse=True)

            max_q = -1
            '''
            if actions[0]['type']!="pass":
                action_type = actions[0]['type']
                if 'collect' in action_type:
                    action_type = 'collected_gems'
                if 'buy_available'==action_type:
                    action_type = 'buy_tire'+str(actions[0]['card'].deck_id+1)
                cur_simple_action = action_map[action_type]
                max_q = self.q_table[cur_state][cur_simple_action]
                cur_action = actions[0]
                currentState = copy.deepcopy(gameState)
                nextState = gr.generateSuccessor(currentState, actions[0], self.id)
                cur_heuristic = SplendorHeuristic(currentState, nextState, self.id)
            else:
            '''


            for item in max_action_list:
                cur_simple_action = -1
                cur_action = None
                cur_heuristic = -1
                max_action = item[1]
                for action in actions:
                    if action['type']=='pass':
                        continue
                    action_type = action['type']
                    if 'collect' in action_type:
                        action_type  = 'collected_gems'
                    if 'buy_available'==action_type:
                        action_type = 'buy_tire'+str(action['card'].deck_id+1)
                    temp_action = action_map[action_type]
                    if action_type!=max_action:
                        continue
                    #if self.q_table[cur_state][temp_action] > max_q:
                    #    max_q = self.q_table[cur_state][temp_action]
                    #    cur_simple_action = temp_action
                    #    cur_action = action
                    #elif self.q_table[cur_state][temp_action] == max_q:
                    if max_action=='collected_gems' or max_action=='reserve' or max_action=='buy_reserve':
                    #if max_action=='collected_gems':
                        num = 0
                        if max_action=='collected_gems':
                            for colour,count in action['collected_gems'].items():
                                num += count
                        rootState = copy.deepcopy(gameState)
                        preState = copy.deepcopy(gameState)
                        newState = gr.generateSuccessor(rootState, action, self.id)
                        #nextState = gr.generateSuccessor(currentState, action, self.id)
                        temp_heuristic = SplendorHeuristic(preState, newState, self.id)+num
                        #if max_action=='collected_gems':
                        #    temp_heuristic += num * 1000
                        #temp_heuristic = num*100
                        if temp_heuristic>cur_heuristic:
                            cur_heuristic = temp_heuristic
                            cur_action = action
                            cur_simple_action = temp_action
                    elif max_action=='buy_tire1' or max_action=='buy_tire2':
                        cur_score = gameState.agents[self.id].score
                        nobles = gameState.board.nobles
                        colour_count = {'black':0, 'red':0, 'green':0, 'blue':0, 'white':0}
                        cur_cards = gameState.agents[self.id].cards
                        for noble in nobles:
                            for key in noble[1].keys():
                                colour_count[key] += 1
                        for key in colour_count.keys():
                            if colour_count[key]<len(cur_cards[key]):
                                colour_count[key] = 0
                            else:
                                colour_count[key] -= len(cur_cards[key])
                        rootState = copy.deepcopy(gameState)
                        preState = copy.deepcopy(gameState)
                        newState = gr.generateSuccessor(rootState, action, self.id)
                        #nextState = gr.generateSuccessor(currentState, action, self.id)
                        #temp_heuristic = SplendorHeuristic(preState, newState, self.id)
                        if action['card'].points + cur_score >= 15:
                            temp_heuristic = 10000000
                        elif action['card'].points >=10:
                            temp_heuristic = SplendorHeuristic(preState, newState, self.id)*action['card'].points
                        elif colour_count[action['card'].colour]>0:
                            temp_heuristic = SplendorHeuristic(preState, newState, self.id)*colour_count[action['card'].colour]
                        else:
                            temp_heuristic = SplendorHeuristic(preState, newState, self.id)*action['card'].points
                        if temp_heuristic>cur_heuristic:
                            cur_heuristic = temp_heuristic
                            cur_action = action
                            cur_simple_action = temp_action
                    elif max_action=='buy_tire3':
                        cur_score = gameState.agents[self.id].score
                        rootState = copy.deepcopy(gameState)
                        preState = copy.deepcopy(gameState)
                        newState = gr.generateSuccessor(rootState, action, self.id)
                        #nextState = gr.generateSuccessor(currentState, action, self.id)
                        #temp_heuristic = SplendorHeuristic(preState, newState, self.id)
                        temp_heuristic = SplendorHeuristic(preState, newState, self.id)*action['card'].points
                        if temp_heuristic>cur_heuristic:
                            cur_heuristic = temp_heuristic
                            cur_action = action
                            cur_simple_action = temp_action
                if cur_action!=None:
                    break



            if cur_action==None:
                cur_action = random.choice(actions)
                return cur_action
                '''
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
                '''
        newState = gr.generateSuccessor(deepcopy(gameState), cur_action, self.id)
        new_cards = newState.agents[self.id].cards
        #new_state1 = str(len(new_cards['black']))+str(len(new_cards['red']))+str(len(new_cards['green']))+str(len(new_cards['blue']))+str(len(new_cards['white']))+str(chr(ord('a')+new_score))
        new_gems = newState.agents[self.id].gems
        new_score = newState.agents[self.id].score
        #new_state0 = str(new_gems['black'])+str(new_gems['red'])+str(new_gems['green'])+str(new_gems['blue'])+str(new_gems['white'])
        #new_state = new_state0+new_state1
        #new_state = str(chr(ord('a')+new_gems['black']+len(new_cards['black'])))+str(chr(ord('a')+new_gems['red']+len(new_cards['red'])))+str(chr(ord('a')+new_gems['green']+len(new_cards['green'])))+str(chr(ord('a')+new_gems['blue']+len(new_cards['blue'])))+str(chr(ord('a')+new_gems['white']+len(new_cards['white'])))+str(chr(ord('a')+new_score))
        new_gems_num = new_gems['black']+new_gems['red']+new_gems['green']+new_gems['blue']+new_gems['white']
        new_cards_num = len(new_cards['black'])+len(new_cards['red'])+len(new_cards['green'])+len(new_cards['blue'])+len(new_cards['white'])
        new_state = str(self.numToChar(new_gems_num))+str(self.numToChar(new_cards_num))+str(chr(ord('a')+new_score))
        #update
        nobles = gameState.board.nobles
        colour_count = {'black':0, 'red':0, 'green':0, 'blue':0, 'white':0}
        for noble in nobles:
            for key in noble[1].keys():
                colour_count[key] += 1
        for key in colour_count.keys():
            if colour_count[key]<len(new_cards[key]):
                colour_count[key] = 0
            else:
                colour_count[key] -= len(new_cards[key])

        review = 0
        if cur_simple_action==4:
            review += 1
            review += (cur_action['card'].points+1)*self.r_table[cur_state][cur_simple_action]
        elif (cur_simple_action>=2 and cur_simple_action<=3) or cur_simple_action==5:
            review += self.beta*((cur_action['card'].points+1)*self.r_table[cur_state][cur_simple_action])+(1-self.beta)*((colour_count[cur_action['card'].colour])*self.r_table[cur_state][cur_simple_action])
        elif cur_simple_action==0:
            review = self.r_table[cur_state][cur_simple_action]
        else:
            review = 0
        #self.q_table[cur_state][cur_simple_action] += self.alpha * (self.r_table[cur_state][cur_simple_action] + self.gamma * max(self.q_table[new_state]) - self.q_table[cur_state][cur_simple_action])
        if flag==0:
            self.q_neg_table[cur_state][cur_simple_action] = self.alpha * (review + self.gamma * max(self.q_table[new_state]) - self.q_table[cur_state][cur_simple_action])
        self.q_table[cur_state][cur_simple_action] += self.alpha * (review + self.gamma * max(self.q_table[new_state]) - self.q_table[cur_state][cur_simple_action])

        #if self.count==1:
        #    self.save_obj(self.q_table, 'q1_table_step1')

        next_score = newState.agents[self.id].score
        if next_score>=15:
            self.save_obj(self.q_table, 'q1_table_it7')
            #self.save_obj(self.r_table, 'r_table')
            #self.save_obj(self.q_neg_table, 'q_neg_table')

        other_agent_score = 0
        if self.id==0:
            other_agent_score = newState.agents[1].score
        else:
            other_agent_score = newState.agents[0].score
        if other_agent_score>=10 and new_score<=3:
            for key in self.q_table.keys():
                if sum(self.q_neg_table[key])!=0:
                    for i in range(6):
                        self.q_table[key][i] -= 2*self.q_neg_table[key][i]
                    self.q_neg_table[key] = [0, 0, 0, 0, 0, 0]
            self.save_obj(self.q_table, 'q1_table_it7')


        return cur_action
