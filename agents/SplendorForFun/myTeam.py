import pickle
import time
import numpy as np

from template import Agent
from Splendor.splendor_model import *

THINKTIME = 0.95


def getLegalActions(gr, game_state, agent_id):
    actions = []
    agent, board = game_state.agents[agent_id], game_state.board
    potential_nobles = []
    for noble in board.nobles:
        if gr.noble_visit(agent, noble):
            potential_nobles.append(noble)
    if len(potential_nobles) == 0:
        potential_nobles = [None]
    # Generate actions (collect up to 3 different gems). Work out all legal combinations. Theoretical max is 10.
    available_colours = [colour for colour, number in board.gems.items() if colour != 'yellow' and number > 0]
    combo_length = min(len(available_colours), 3)
    for combo in itertools.combinations(available_colours, combo_length):
        collected_gems = {colour: 1 for colour in combo}
        return_combos = gr.generate_return_combos(agent.gems, collected_gems)
        for returned_gems in return_combos:
            for noble in potential_nobles:
                actions.append({'type': 'collect_diff',
                                'collected_gems': collected_gems,
                                'returned_gems': returned_gems,
                                'noble': noble})
    # Generate actions (collect 2 identical gems). Theoretical max is 5.
    available_colours = [colour for colour, number in board.gems.items() if colour != 'yellow' and number >= 4]
    for colour in available_colours:
        collected_gems = {colour: 2}
        return_combos = gr.generate_return_combos(agent.gems, collected_gems)
        for returned_gems in return_combos:
            for noble in potential_nobles:
                actions.append({'type': 'collect_same',
                                'collected_gems': collected_gems,
                                'returned_gems': returned_gems,
                                'noble': noble})
    # Generate actions (reserve card). Agent can reserve only if it possesses < 3 cards currently reserved.
    # if len(agent.cards['yellow']) < 3:
    #     collected_gems = {'yellow': 1} if board.gems['yellow'] > 0 else {}
    #     return_combos = gr.generate_return_combos(agent.gems, collected_gems)
    #     for returned_gems in return_combos:
    #         for card in board.dealt_list():
    #             if card:
    #                 for noble in potential_nobles:
    #                     actions.append({'type': 'reserve',
    #                                     'card': card,
    #                                     'collected_gems': collected_gems,
    #                                     'returned_gems': returned_gems,
    #                                     'noble': noble})
    # Generate actions (buy card). Agents can buy cards if they can cover its resource cost. Resources can come from
    for card in board.dealt_list() + agent.cards['yellow']:
        if not card or len(agent.cards[card.colour]) == 7:
            continue
        returned_gems = gr.resources_sufficient(agent, card.cost)  # Check if this card is affordable.
        if type(returned_gems) == dict:  # If a dict was returned, this means the agent possesses sufficient resources.
            # Check to see if the acquisition of a new card has meant new nobles becoming candidates to visit.
            new_nobles = []
            for noble in board.nobles:
                a = time.time()
                dumped = pickle.dumps(agent, -1)
                agent_post_action = pickle.loads(dumped)
                b = time.time()
                # Give the card featured in this action to a copy of the agent.
                agent_post_action.cards[card.colour].append(card)
                # Use this copied agent to check whether this noble can visit.
                if gr.noble_visit(agent_post_action, noble):
                    new_nobles.append(noble)  # If so, add noble to the new list.
            if not new_nobles:
                new_nobles = [None]
            for noble in new_nobles:
                actions.append({'type': 'buy_reserve' if card in agent.cards['yellow'] else 'buy_available',
                                'card': card,
                                'returned_gems': returned_gems,
                                'noble': noble})
    # Return list of actions. If there are no actions (almost impossible), all this player can do is pass.
    # A noble is still permitted to visit if conditions are met.
    if not actions:
        for noble in potential_nobles:
            actions.append({'type': 'pass', 'noble': noble})
    return actions


def IgnoreActions(actions):
    maxDiff = 0
    for action in actions:
        if action['type'] == 'collect_diff' and sum(action['collected_gems'].values()) == 3:
            maxDiff = 3
            break
        elif action['type'] == 'collect_diff' and sum(action['collected_gems'].values()) == 2:
            maxDiff = 2
    newActions = []
    for action in actions:
        if action['type'] == 'collect_diff' and sum(action['collected_gems'].values()) < maxDiff:
            continue
        if action['type'] == 'reserve' and maxDiff == 3:
            continue
        newActions.append(action)
    return newActions


def Selection(root):
    weight = 60
    maxPie = -99999
    node = None
    for child in root.children.keys():
        if root.children[child] == 0:
            node = child
            break
        delta = np.sqrt(2 * np.log(root.visited) / root.children[child])
        pie = child.value + weight * delta
        if pie > maxPie:
            node = child
            maxPie = pie
    root.children[node] += 1
    root.visited += 1
    return node


def Expansion(node, gr):
    if node.children == {}:
        actions = getLegalActions(gr, node.state, node.id)
        dumped = pickle.dumps(node.state, -1)
        for action in actions:
            tempState = pickle.loads(dumped)
            childState = gr.generateSuccessor(tempState, action, node.id)
            child = Node(childState, action, 1 - node.id)
            child.parent = node
            node.children[child] = 0
    retNode = random.choice(list(node.children.keys()))
    node.children[retNode] += 1
    return retNode


def Simulation(node, gr):
    aa = time.time()
    p = pickle.dumps(node.state, -1)
    tempState = pickle.loads(p)
    turn = node.id
    count = 0
    ab = time.time()
    while count < 3:
        a = time.time()
        actions = getLegalActions(gr, tempState, turn)
        b = time.time()
        action = random.choice(actions)
        c = time.time()
        tempState = gr.generateSuccessor(tempState, action, turn)
        d = time.time()
        turn = 1 - turn
        count += 1
    myCards = 0
    for i in tempState.agents[node.id].cards.keys():
        if i != 'yellow':
            myCards += len(tempState.agents[node.id].cards[i])
    # opCards = 0
    # for i in tempState.agents[1-node.id].cards.keys():
    #     if i != 'yellow':
    #         opCards += len(tempState.agents[1-node.id].cards[i])
    reward = tempState.agents[node.id].score * 80 + (tempState.agents[node.id].score - tempState.agents[1-node.id].score) * 30 + myCards * 40
    node.value = (node.value * node.visited + reward) / (node.visited + 1)
    node.visited += 1
    return node


def Backpropagation(rootNode, startNode):
    tempNode = startNode
    discount = 0.9
    while True:
        parentNode = tempNode.parent
        parentNode.value = 0
        for child in parentNode.children.keys():
            if child.value * discount > parentNode.value:
                parentNode.value = child.value * discount
        discount *= 0.9
        if tempNode == rootNode:
            break
        else:
            tempNode = tempNode.parent
    return


class Node:
    def __init__(self, state, fromAction, agentId):
        self.id = agentId
        self.action = fromAction
        self.state = state
        self.parent = None
        self.children = {}
        self.value = 0
        self.visited = 0


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)

    def BuildTree(self, gameState, actions, gr):
        root = Node(gameState, None, self.id)
        for action in actions:
            tempState = pickle.loads(pickle.dumps(gameState, -1))
            childState = gr.generateSuccessor(tempState, action, self.id)
            child = Node(childState, action, 1 - self.id)
            child.parent = root
            root.children[child] = 0
        return root

    def SelectAction(self, actions, gameState):
        a = time.time()
        gr = SplendorGameRule(len(gameState.agents))
        actions = IgnoreActions(actions)
        b = time.time()
        root = self.BuildTree(gameState, actions, gr)
        startTime = time.time()
        count = 0
        c = time.time()
        while time.time() - startTime <= THINKTIME:
            d = time.time()
            expandNode = Selection(root)
            e = time.time()
            child = Expansion(expandNode, gr)
            f = time.time()
            simulateNode = Simulation(child, gr)
            g = time.time()
            Backpropagation(expandNode, simulateNode)
            h = time.time()
            count += 1
        maxValue = -99999
        retNode = None
        for child in root.children:
            print(child.visited)
            if child.value > maxValue:
                maxValue = child.value
                retNode = child
        return retNode.action
