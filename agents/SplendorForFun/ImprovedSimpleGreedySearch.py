from template import Agent
import time
import heapq
from Splendor.splendor_model import *

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


def RankTheCards(dealt):
    tier1 = {}
    for i in range(len(dealt[0])):
        card = dealt[0][i]
        if card is not None:
            if card.points == 1 and len(card.cost.keys()) == 1:
                tier1[i] = 10
            else:
                tier1[i] = 0
        else:
            tier1[i] = -10

    tier2 = {}
    for i in range(len(dealt[1])):
        card = dealt[1][i]
        if card is not None:
            if card.points == 3 and len(card.cost.keys()) == 1:
                tier2[i] = 40
            elif card.points == 2 and len(card.cost.keys()) == 1:
                tier2[i] = 30
            elif card.points == 2 and len(card.cost.keys()) == 3:
                tier2[i] = 20
            elif card.points == 2 and len(card.cost.keys()) == 2:
                tier2[i] = 10
            else:
                tier2[i] = 0
        else:
            tier2[i] = -10

    tier3 = {}
    for i in range(len(dealt[2])):
        card = dealt[2][i]
        if card is not None:
            if card.points == 4 and len(card.cost.keys()) == 1:
                tier3[i] = 30
            elif card.points == 5 and len(card.cost.keys()) == 2:
                tier3[i] = 20
            elif card.points == 4 and len(card.cost.keys()) == 3:
                tier3[i] = 10
            else:
                tier3[i] = 0
        else:
            tier3[i] = -0

    t1 = sorted(tier1.items(), key=lambda item: item[1], reverse=True)
    t2 = sorted(tier2.items(), key=lambda item: item[1], reverse=True)
    t3 = sorted(tier3.items(), key=lambda item: item[1], reverse=True)
    cards = [dealt[0][t1[0][0]], dealt[0][t1[1][0]], dealt[0][t1[2][0]], dealt[1][t2[0][0]], dealt[1][t2[1][0]], dealt[2][t3[0][0]]]
    return cards


def SplendorHeuristic(rootState, gameState, agentId):
    nobleWeight = 100
    scoreWeight = 500
    zeroCardWeight = 1000
    previousGems = rootState.agents[agentId].gems
    previousScore = rootState.agents[agentId].score
    previousCards = 0
    for i in rootState.agents[agentId].cards.keys():
        if i != 'yellow':
            previousCards += len(rootState.agents[agentId].cards[i])
    currentGems = gameState.agents[agentId].gems
    currentScore = gameState.agents[agentId].score
    currentCards = 0
    nobleBonus = 0
    for i in gameState.agents[agentId].cards.keys():
        if i != 'yellow':
            currentCards += len(gameState.agents[agentId].cards[i])
            if len(gameState.agents[agentId].cards[i]) != len(rootState.agents[agentId].cards[i]):
                for noble in gameState.board.nobles:
                    if i in noble[1].keys() and len(gameState.agents[agentId].cards[i]) < noble[1][i]:
                        nobleBonus += 1
    actionGems = {c: 0 for c in previousGems.keys()}
    for i in actionGems.keys():
        actionGems[i] = currentGems[i] - previousGems[i]
    actionScore = currentScore - previousScore
    boardGemCount = {c: 0 for c in actionGems.keys()}
    board = gameState.board
    dealt = board.dealt
    cards = RankTheCards(dealt)
    for i in range(len(cards)):
        card = cards[i]
        if card is not None:
            for colour, number in card.cost.items():
                boardGemCount[colour] += number
    gemScore = 0
    for i in actionGems.keys():
        if actionGems[i] != 0:
            gemScore += actionGems[i] * boardGemCount[i]
    return - (gemScore + actionScore * scoreWeight + (currentCards - previousCards) * zeroCardWeight + nobleBonus * nobleWeight)


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)

    def SelectAction(self, actions, gameState):
        startTime = time.time()
        openQ = PriorityQueue()
        gr = SplendorGameRule(len(gameState.agents))
        for action in actions:
            rootState = copy.deepcopy(gameState)
            preState = copy.deepcopy(gameState)
            newState = gr.generateSuccessor(rootState, action, self.id)
            openQ.push(action, SplendorHeuristic(preState, newState, self.id))
            if time.time() - startTime >= THINKTIME:
                break
        retAction = openQ.pop()
        return retAction
