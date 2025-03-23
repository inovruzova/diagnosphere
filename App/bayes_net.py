import random
import numpy as np
from collections import defaultdict

def extend(s, var, val):
    """Create a copy of dictionary `s` and add a new key-value pair where `var` is set to `val`. Return the updated copy."""
    return {**s, var: val}

class ProbDist:
    """
    Represents a discrete probability distribution for a single random variable. 
    You can initialize it with a variable name and an optional frequency dictionary.
    Probabilities are normalized automatically if frequencies are provided.
​
    Example:
    >>> P = ProbDist('Flip'); P['H'], P['T'] = 0.25, 0.75; P['H']
    0.25
    >>> P = ProbDist('X', {'lo': 125, 'med': 375, 'hi': 500})
    >>> P['lo'], P['med'], P['hi']
    (0.125, 0.375, 0.5)
    """
    def __init__(self, varname='?', freqs=None):
        """
        Initialize the distribution. If `freqs` is given, it must be a dictionary 
        with values as keys and their frequencies as values. The distribution is normalized.
        """
        self.prob = {}
        self.varname = varname
        self.values = []
        if freqs:
            for (v, p) in freqs.items():
                self[v] = p
            self.normalize()
    
    def __getitem__(self, val):
        """Retrieve the probability of `val` if it exists, otherwise return 0."""
        try:
            return self.prob[val]
        except KeyError:
            return 0
    
    def __setitem__(self, val, p):
        """Assign probability `p` to the value `val`."""
        if val not in self.values:
            self.values.append(val)
        self.prob[val] = p
        
    def normalize(self):
        """
        Ensure that the probabilities of all values sum up to 1. 
        If the sum of values is 0, a ZeroDivisionError is raised.
        """
        total = sum(self.prob.values())
        if not np.isclose(total, 1.0):
            for val in self.prob:
                self.prob[val] /= total
        return self

    def show_approx(self, numfmt='{:.3g}'):
        """
        Display the probabilities rounded to a specified format, sorted by their keys. 
        Useful for readability in doctests.
        """
        return ', '.join([('{}: ' + numfmt).format(v, p)
                          for (v, p) in sorted(self.prob.items())])

    def __repr__(self):
        """Return a string representation of the distribution."""
        return "P({})".format(self.varname)

def probability_sampling(probabilities):
    """
    Perform random sampling based on the given probability distribution. 
    Returns an outcome based on the probabilities.
    """
    total = sum(probabilities.values())
    r = random.uniform(0, total)
    cumulative = 0
    for outcome, prob in probabilities.items():
        cumulative += prob
        if r <= cumulative:
            return outcome
    return None  # This should not occur if probabilities are normalized.

class MultiClassBayesNode:
    """
    Represents a node in a Bayesian network for multi-class variables. 
    It contains the variable, its parents, and the conditional probability table (CPT).
    """
    def __init__(self, X, parents, cpt):
        """
        Initialize the node with:
        - `X`: Variable name.
        - `parents`: List of parent variable names.
        - `cpt`: A dictionary representing the conditional probability table.
        """
        if isinstance(parents, str):
            parents = parents.split()
        self.variable = X
        self.parents = parents
        self.cpt = cpt
        self.children = []

    def p(self, value, event):
        """
        Compute the conditional probability of `X=value` given the parent values in `event`.
        """
        parent_values = tuple(event.get(p, None) for p in self.parents)
        probabilities = self.cpt.get(parent_values, {})
        return probabilities.get(value, 0)  # Defaults to 0 if `value` is not found.

    def sample(self, event):
        """
        Sample a value for the variable given parent values in `event`. 
        Sampling is based on the conditional probability distribution.
        """
        parent_values = tuple(event.get(p, None) for p in self.parents)
        probabilities = self.cpt.get(parent_values, {})
        return probability_sampling(probabilities)

    def __repr__(self):
        """Return a string representation of the node."""
        return repr((self.variable, ' '.join(self.parents)))

class BayesNet:
    """
    Represents a Bayesian network consisting of nodes (variables) and their dependencies.
    Supports multi-class nodes.
    """
    def __init__(self, node_specs=None):
        """
        Initialize the network. Nodes must be added in topological order 
        (parents must be added before their children).
        """
        self.nodes = []
        self.variables = []
        node_specs = node_specs or []
        for node_spec in node_specs:
            self.add(node_spec)
            
    def add(self, node_spec):
        """
        Add a node to the network. Accepts either a pre-constructed node 
        or the specifications for a new node.
        """
        if isinstance(node_spec, MultiClassBayesNode):
            node = node_spec
        else:
            node = MultiClassBayesNode(*node_spec)

        assert node.variable not in self.variables
        assert all((parent in self.variables) for parent in node.parents)
        
        self.nodes.append(node)
        self.variables.append(node.variable)
        
        # Register this node as a child for its parent nodes
        for parent in node.parents:
            self.variable_node(parent).children.append(node)
            
    def variable_node(self, var):
        """Retrieve the node corresponding to the variable `var`."""
        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception(f"No such variable: {var}")
    
    def variable_values(self, var):
        """Retrieve the domain of `var` (default is `[True, False]`)."""
        return [True, False]
    
    def __repr__(self):
        """Return a string representation of the network."""
        return f"BayesNet({self.nodes!r})"

class Factor:
    """Represents a factor in a joint distribution."""
    def __init__(self, variables, cpt):
        """
        Initialize the factor with:
        - `variables`: List of variables involved in the factor.
        - `cpt`: Conditional probability table.
        """
        self.variables = variables
        self.cpt = cpt
        
    def normalize(self):
        """
        Normalize the factor and return a `ProbDist` for the remaining variable.
        This is only valid if the factor has one variable left.
        """
        assert len(self.variables) == 1
        return ProbDist(self.variables[0], {k: v for ((k,), v) in self.cpt.items()})
    
    def p(self, e):
        """Retrieve the probability for the event `e` from the factor's CPT."""
        return self.cpt[event_values(e, self.variables)]

def enumerate_all(variables, e, bn):
    """
    Calculate the sum of all entries in the joint probability distribution 
    for `variables` consistent with the evidence `e` in network `bn`.
    """
    if not variables:
        return 1.0
    Y, rest = variables[0], variables[1:]
    Ynode = bn.variable_node(Y)
    if Y in e:
        return Ynode.p(e[Y], e) * enumerate_all(rest, e, bn)
    else:
        return sum(Ynode.p(y, e) * enumerate_all(rest, extend(e, Y, y), bn)
                   for y in bn.variable_values(Y))

def enumeration_ask(X, e, bn):
    """
    Compute the conditional probability distribution for the query variable `X` 
    given evidence `e` in the Bayesian network `bn`.
    """
    assert X not in e, "Query variable must not overlap with the evidence."
    Q = ProbDist(X)
    for xi in bn.variable_values(X):
        Q[xi] = enumerate_all(bn.variables, extend(e, X, xi), bn)
    return Q.normalize()

def event_values(event, variables):
    """
    Generate a tuple containing the values of the specified variables from the event.
    
    Examples:
    >>> event_values({'A': 10, 'B': 9, 'C': 8}, ['C', 'A'])
    (8, 10)
    >>> event_values((1, 2), ['C', 'A'])
    (1, 2)
    """
    if isinstance(event, tuple) and len(event) == len(variables):
        return event
    else:
        return tuple(event[var] for var in variables)