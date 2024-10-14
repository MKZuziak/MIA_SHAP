import itertools
import copy
import random
from typing import Any
from collections import OrderedDict


def form_coalitions(
    elements: list,
    return_dict: bool = True,
    amplification: int = 5,
    ) -> list[list] | dict[tuple : None]:
    """Given a list of elements of a length N, forms a set of lists, each one with K copies
    of the selected client. Returned set is of cardinality |N|, where each member of the set
    is an array of length N+K.
    
    -------------
    Args
        elements (list): a list containing all the elements from which we want to form a superset.
        return_dict (bool): if True, will return the superset in a form {tuple: None} rather than [list].
        amplificaiton (int): amplification of the coalition, number of clients appended to the list.
    -------------
        Returns
        list[list] | dict[tuple : None]"""
    newset = list() # New empty set (list)
    for l in list(elements):
        new_list = copy.deepcopy(list(elements))
        for _ in range(amplification):
            new_list.append(l)
            newset.append(new_list) # Alpha-Amplified Coalitions
    if return_dict == True:
        newset = {tuple(coalition): None for coalition in newset}
        return newset
    else:
        return newset


def select_subsets(
    coalitions: dict | list,
    searched_node: int
    ) -> dict[tuple : Any]:
        """Given a dict or list of possible coalitions and a searched node, will return
        every possible coalition which DO NOT CONTAIN the searche node.
        -------------
        Args
            coalitions (dict): a superset or a set of all possible coalitions, mapping
                each coalition to some value, e.g. {(1, 2, 3,): None}
            searched_node (int): an id of the searched node.
       -------------
         Returns
            dict[tuple : Any]"""
        if type(coalitions) == dict:
            subsets = {nodes: model for nodes, model in coalitions.items() 
                    if searched_node not in nodes}
        if type(coalitions) == list:
            subsets = [nodes for nodes in coalitions if searched_node not in nodes]
        return subsets


def select_gradients(
    gradients: OrderedDict,
    query: list,
    in_place: bool = False
    ):
    """Helper function for selecting gradients that of nodes that are
    in the query.

    Parameters
    ----------
    sqe: OrderedDict
        An OrderedDict containing all the gradients.
    query: list
        A size containing ids of the searched nodes.
    in_place: bool, default to False
        No description
    
    Returns
    -------
    Generator
        A generator object that can be iterated to obtain chunks of the original iterable.
    """
    selected_gradients = {}
    if in_place == False:
        for node_id in query:
            if node_id not in selected_gradients:
                selected_gradients[node_id] = copy.deepcopy(gradients[node_id])
            else:
                selected_gradients[f'copy_{random.randint(0, 100000000)}_{node_id}'] = copy.deepcopy(gradients[node_id])
    return selected_gradients