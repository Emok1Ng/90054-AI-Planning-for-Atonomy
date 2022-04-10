from copy import deepcopy

from template import Agent
import time
from utils import PriorityQueue
from Splendor.splendor_model import *


THINKTIME = 0.95


def SplendorHeuristic(rootState, gameState, agentId):

    # Priority
    # 1st get > 0 score this round
    # 2nd get > 0 score next round
    # 3rd get = 0 score card this round
    # 4th get = 0 score next round
    # 5th get gems of majority color
    scoreWeight = 2000
    nextScoreWeight = 300
    cardWeight = 1000

    # Calculate difference in score and cards after the action
    previousGems = rootState.agents[agentId].gems
    previousScore = rootState.agents[agentId].score
    previousCards = rootState.agents[agentId].cards
    previousCardsCount = 0
    for i in previousCards.keys():
        if i != 'yellow':
            previousCardsCount += len(previousCards[i])
    currentGems = gameState.agents[agentId].gems
    currentScore = gameState.agents[agentId].score
    currentCards = gameState.agents[agentId].cards
    currentCardsCount = 0
    for i in currentCards.keys():
        if i != 'yellow':
            currentCardsCount += len(currentCards[i])
    scoreDiff = currentScore - previousScore
    cardsDiff = currentCardsCount - previousCardsCount

    # Calculate whether there are cards can be afford next round
    board = gameState.board
    dealt = board.dealt
    currentGemTotal = {}
    wild = currentGems['yellow']
    for i in currentGems.keys():
        if i != 'yellow':
            currentGemTotal[i] = currentGems[i] + len(currentCards[i])
    potentialScore = 0
    for i in range(len(dealt)):
        for j in range(len(dealt[i])):
            card = dealt[i][j]
            shortFall = 0
            if card is not None:
                for colour, number in card.cost.items():
                    shortFall += max(number - currentGemTotal[colour], 0)
                if shortFall <= wild:
                    potentialScore += card.points
    for i in range(len(currentCards['yellow'])):
        card = currentCards['yellow'][i]
        shortFall = 0
        if card is not None:
            for colour, number in card.cost.items():
                shortFall += max(number - currentGemTotal[colour], 0)
            if shortFall <= wild:
                potentialScore += card.points
    if potentialScore != 0:
        print("sth")
    # Calculate difference in gems after the action
    # Focus on the easiest card
    # Only dominate when none of the card can be bought this round and next round
    actionGems = {c: 0 for c in previousGems.keys()}
    for i in actionGems.keys():
        actionGems[i] = currentGems[i] - previousGems[i]
    boardGemCount = {c: 0 for c in actionGems.keys()}
    for i in range(len(dealt[0])):
        card = dealt[0][i]
        if card is not None:
            for colour, number in card.cost.items():
                boardGemCount[colour] += number
    gemScore = 0
    for i in actionGems.keys():
        if actionGems[i] != 0:
            gemScore += actionGems[i] * boardGemCount[i]

    return - (gemScore + scoreDiff * scoreWeight + max(cardsDiff * cardWeight, potentialScore * nextScoreWeight))


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)

    def SelectAction(self, actions, gameState):
        startTime = time.time()
        openQ = PriorityQueue()
        gr = SplendorGameRule(len(gameState.agents))
        for action in actions:
            rootState = deepcopy(gameState)
            preState = deepcopy(gameState)
            newState = gr.generateSuccessor(rootState, action, self.id)
            openQ.push(action, SplendorHeuristic(preState, newState, self.id))
            if time.time() - startTime >= THINKTIME:
                break
        retAction = openQ.pop()
        return retAction
