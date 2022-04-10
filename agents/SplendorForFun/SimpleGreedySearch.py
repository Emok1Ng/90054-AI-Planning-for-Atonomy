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


def SplendorHeuristic(rootState, gameState, agentId):
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
    return - (gemScore + actionScore * scoreWeight + (currentCards - previousCards) * zeroCardWeight)


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
