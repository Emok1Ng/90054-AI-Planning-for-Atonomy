import time
import numpy as np

from template import Agent
from Splendor.splendor_model import *

THINKTIME = 0.97


def resources_sufficient(agentCards, agentGems, costs):
    wild = agentGems['yellow']
    return_combo = {c: 0 for c in COLOURS.values()}
    for colour, cost in costs.items():
        # If a shortfall is found, see if the difference can be made with wild/seal/yellow gems.
        available = agentGems[colour] + len(agentCards[colour])
        shortfall = max(cost - available, 0)  # Shortfall shouldn't be negative.
        wild -= shortfall
        # If wilds are expended, the agent cannot make the purchase.
        if wild < 0:
            return False
        # Else, increment return_combo accordingly. Note that the agent should never return gems if it can afford
        # to pay using its card stacks, and should never return wilds if it can return coloured gems instead.
        # Although there may be strategic instances where holding on to coloured gems is beneficial (by virtue of
        # shorting players from resources), in this implementation, this edge case is not worth added complexity.
        gem_cost = max(cost - len(agentCards[colour]), 0)  # Gems owed.
        gem_shortfall = max(gem_cost - agentGems[colour], 0)  # Wilds required.
        return_combo[colour] = gem_cost - gem_shortfall  # Coloured gems to be returned.
        return_combo['yellow'] += gem_shortfall  # Wilds to be returned.

    # Filter out unnecessary colours and return dict specifying combination of gems.
    return dict({i for i in return_combo.items() if i[-1] > 0})


def dealt_list(dealt):
    return [card for deck in dealt for card in deck if card]


def noble_visit(agentCards, noble):
    _, costs = noble
    for colour, cost in costs.items():
        if not len(agentCards[colour]) >= cost:
            return False
    return True


def deal(decks, deck_id):
    if len(decks[deck_id]):
        random.shuffle(decks[deck_id])
        return decks[deck_id].pop()
    return None


def generateSuccessor(state, action, agent_id):
    stateList = copy(state)
    if agent_id == 0:
        agent_index = 0
    else:
        agent_index = 7
    stateList[agent_index + 6] = action
    score = 0
    if 'card' in action:
        card = action['card']

    if 'collect' in action['type'] or action['type'] == 'reserve':
        # Decrement board gem stacks by collected_gems. Increment player gem stacks by collected_gems.

        for colour, count in action['collected_gems'].items():
            stateList[16][colour] -= count
            stateList[agent_index + 1][colour] += count
        # Decrement player gem stacks by returned_gems. Increment board gem stacks by returned_gems.
        for colour, count in action['returned_gems'].items():
            stateList[16][colour] -= count
            stateList[agent_index + 1][colour] += count

        if action['type'] == 'reserve':
            # Remove card from dealt cards by locating via unique code (cards aren't otherwise hashable).
            # Since we want to retain the positioning of dealt cards, set removed card slot to new dealt card.
            # Since the board may have None cards (empty slots that cannot be filled), check cards first.
            # Add card to player's yellow stack.
            for i in range(len(stateList[15][card.deck_id])):
                if stateList[15][card.deck_id][i] and stateList[15][card.deck_id][i].code == card.code:
                    stateList[15][card.deck_id][i] = deal(stateList[14], card.deck_id)
                    stateList[agent_index + 2]['yellow'].append(card)
                    break

    elif 'buy' in action['type']:
        # Decrement player gem stacks by returned_gems. Increment board gem stacks by returned_gems.
        for colour, count in action['returned_gems'].items():
            stateList[agent_index + 1][colour] -= count
            stateList[16][colour] += count
        # If buying one of the available cards on the board, set removed card slot to new dealt card.
        # Since the board may have None cards (empty slots that cannot be filled), check cards first.
        if 'available' in action['type']:
            for i in range(len(stateList[15][card.deck_id])):
                if stateList[15][card.deck_id][i] and stateList[15][card.deck_id][i].code == card.code:
                    stateList[15][card.deck_id][i] = deal(stateList[14], card.deck_id)
                    break
        # Else, agent is buying a reserved card. Remove card from player's yellow stack.
        else:
            for i in range(len(stateList[agent_index + 2]['yellow'])):
                if stateList[agent_index + 2]['yellow'][i].code == card.code:
                    del stateList[agent_index + 2]['yellow'][i]
                    break

                    # Add card to player's stack of matching colour, and increment agent's score accordingly.
        stateList[agent_index + 2][card.colour].append(card)
        score += card.points

    if action['noble']:
        # Remove noble from board. Add noble to player's stack. Like cards, nobles aren't hashable due to possessing
        # dictionaries (i.e. resource costs). Therefore, locate and delete the noble via unique code.
        # Add noble's points to agent score.
        for i in range(len(stateList[17])):
            if stateList[17][i][0] == action['noble'][0]:
                del stateList[17][i]
                stateList[agent_index + 3].append(action['noble'])
                score += 3
                break

    # Log this turn's action and any resultant score. Return updated gamestate.
    stateList[agent_index + 5].append((action, score))
    stateList[agent_index] += score
    stateList[agent_index + 4] = action['type'] == 'pass'
    return stateList


def getLegalActions(gr, stateList, agent_id):
    actions = []
    if agent_id == 0:
        agent_index = 0
    else:
        agent_index = 7
    potential_nobles = []
    for noble in stateList[17]:
        if noble_visit(stateList[agent_index+2], noble):
            potential_nobles.append(noble)
    if len(potential_nobles) == 0:
        potential_nobles = [None]
    available_colours = [colour for colour, number in stateList[16].items() if colour != 'yellow' and number > 0]
    combo_length = min(len(available_colours), 3)
    for combo in itertools.combinations(available_colours, combo_length):
        collected_gems = {colour: 1 for colour in combo}
        return_combos = gr.generate_return_combos(stateList[agent_index+1], collected_gems)
        for returned_gems in return_combos:
            for noble in potential_nobles:
                actions.append({'type': 'collect_diff',
                                'collected_gems': collected_gems,
                                'returned_gems': returned_gems,
                                'noble': noble})
    available_colours = [colour for colour, number in stateList[16].items() if colour != 'yellow' and number >= 4]
    for colour in available_colours:
        collected_gems = {colour: 2}
        return_combos = gr.generate_return_combos(stateList[agent_index+1], collected_gems)
        for returned_gems in return_combos:
            for noble in potential_nobles:
                actions.append({'type': 'collect_same',
                                'collected_gems': collected_gems,
                                'returned_gems': returned_gems,
                                'noble': noble})
    for card in dealt_list(stateList[15]) + stateList[agent_index+2]['yellow']:
        if not card or len(stateList[agent_index+2][card.colour]) == 7:
            continue
        returned_gems = resources_sufficient(stateList[agent_index+2], stateList[agent_index+1], card.cost)  # Check if this card is affordable.
        if type(returned_gems) == dict:  # If a dict was returned, this means the agent possesses sufficient resources.
            # Check to see if the acquisition of a new card has meant new nobles becoming candidates to visit.
            new_nobles = []
            for noble in stateList[17]:
                agentCard = stateList[agent_index+2]
                # Give the card featured in this action to a copy of the agent.
                agentCard[card.colour].append(card)
                # Use this copied agent to check whether this noble can visit.
                if noble_visit(agentCard, noble):
                    new_nobles.append(noble)  # If so, add noble to the new list.
                agentCard[card.colour].remove(card)
            if not new_nobles:
                new_nobles = [None]
            for noble in new_nobles:
                actions.append({'type': 'buy_reserve' if card in stateList[agent_index+2]['yellow'] else 'buy_available',
                                'card': card,
                                'returned_gems': returned_gems,
                                'noble': noble})
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


def copy(state):
    cards0 = {key: value[:] for key, value in state[2].items()}
    cards1 = {key: value[:] for key, value in state[9].items()}
    gems0 = {key: value for key, value in state[1].items()}
    gems1 = {key: value for key, value in state[8].items()}
    nobles0 = state[3][:]
    nobles1 = state[10][:]
    action_reward0 = state[5][:]
    action_reward1 = state[12][:]
    decks = [state[14][0][:], state[14][1][:], state[14][2][:]]
    dealt = [state[15][0][:], state[15][1][:], state[15][2][:]]
    noblesb = state[17]
    gemsb = {key: value for key, value in state[16].items()}
    ret = [state[0], gems0, cards0, nobles0,
           state[4], action_reward0, state[6],
           state[7], gems1, cards1, nobles1,
           state[1], action_reward1, state[13],
           decks, dealt, gemsb, noblesb
           ]
    return ret


def toStateList(state):
    retList = [state.agents[0].score, state.agents[0].gems, state.agents[0].cards, state.agents[0].nobles,
               state.agents[0].passed, state.agents[0].agent_trace.action_reward, state.agents[0].last_action,
               state.agents[1].score, state.agents[1].gems, state.agents[1].cards, state.agents[1].nobles,
               state.agents[1].passed, state.agents[1].agent_trace.action_reward, state.agents[1].last_action,
               state.board.decks, state.board.dealt, state.board.gems, state.board.nobles]
    return retList


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
        for action in actions:
            childState = generateSuccessor(node.state, action, node.id)
            child = Node(childState, action, 1 - node.id)
            child.parent = node
            node.children[child] = 0
    retNode = random.choice(list(node.children.keys()))
    node.children[retNode] += 1
    return retNode


def Simulation(node, gr):
    if node.id == 0:
        agent_index = 0
        op_index = 7
    else:
        agent_index = 7
        op_index = 0
    turn = node.id
    count = 0
    actions = getLegalActions(gr, node.state, turn)
    action = random.choice(actions)
    tempState = generateSuccessor(node.state, action, turn)
    turn = 1 - turn
    count += 1
    while count < 3:
        actions = getLegalActions(gr, tempState, turn)
        action = random.choice(actions)
        tempState = generateSuccessor(tempState, action, turn)
        turn = 1 - turn
        count += 1
    myCards = 0
    for i in tempState[agent_index+2].keys():
        if i != 'yellow':
            myCards += len(tempState[agent_index+2][i])
    # opCards = 0
    # for i in tempState.agents[1-node.id].cards.keys():
    #     if i != 'yellow':
    #         opCards += len(tempState.agents[1-node.id].cards[i])
    if tempState[agent_index] >= 15:
        win = 1000
    else:
        win = 0
    reward = win + tempState[agent_index] * 80 + (tempState[agent_index] - tempState[op_index]) * 30 + myCards * 40
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

    def BuildTree(self, gameState, actions):
        stateList = toStateList(gameState)
        root = Node(stateList, None, self.id)
        for action in actions:
            childState = generateSuccessor(root.state, action, self.id)
            child = Node(childState, action, 1 - self.id)
            child.parent = root
            root.children[child] = 0
        return root

    def SelectAction(self, actions, gameState):
        gr = SplendorGameRule(len(gameState.agents))
        actions = IgnoreActions(actions)
        root = self.BuildTree(gameState, actions)
        startTime = time.time()
        count = 0
        while time.time() - startTime <= THINKTIME:
            expandNode = Selection(root)
            child = Expansion(expandNode, gr)
            simulateNode = Simulation(child, gr)
            Backpropagation(expandNode, simulateNode)
            count += 1
        maxValue = -99999
        retNode = None
        for child in root.children:
            print(child.visited)
            if child.value > maxValue:
                maxValue = child.value
                retNode = child
        return retNode.action
