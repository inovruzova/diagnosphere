import random
import numpy as np
from collections import defaultdict

def probability(p):
    """Return true with probability p."""
    return p > random.uniform(0.0, 1.0)

def probability_sampling(probabilities):
    """Randomly sample an outcome from a probability distribution."""
    total = sum(probabilities.values())
    r = random.uniform(0, total)
    cumulative = 0
    for outcome, prob in probabilities.items():
        cumulative += prob
        if r <= cumulative:
            return outcome
    return None  # Should not reach here if probabilities are normalize

class ProbDist:
    """A discrete probability distribution. You name the random variable
    in the constructor, then assign and query probability of values.
    >>> P = ProbDist('Flip'); P['H'], P['T'] = 0.25, 0.75; P['H']
    0.25
    >>> P = ProbDist('X', {'lo': 125, 'med': 375, 'hi': 500})
    >>> P['lo'], P['med'], P['hi']
    (0.125, 0.375, 0.5)
    """

    def __init__(self, varname='?', freqs=None):
        """If freqs is given, it is a dictionary of values - frequency pairs,
        then ProbDist is normalized."""
        self.prob = {}
        self.varname = varname
        self.values = []
        if freqs:
            for (v, p) in freqs.items():
                self[v] = p
            self.normalize()

    def __getitem__(self, val):
        """Given a value, return P(value)."""
        try:
            return self.prob[val]
        except KeyError:
            return 0

    def __setitem__(self, val, p):
        """Set P(val) = p."""
        if val not in self.values:
            self.values.append(val)
        self.prob[val] = p

    def normalize(self):
        """Make sure the probabilities of all values sum to 1.
        Returns the normalized distribution.
        Raises a ZeroDivisionError if the sum of the values is 0."""
        total = sum(self.prob.values())
        if not np.isclose(total, 1.0):
            for val in self.prob:
                self.prob[val] /= total
        return self

    def show_approx(self, numfmt='{:.3g}'):
        """Show the probabilities rounded and sorted by key, for the
        sake of portable doctests."""
        return ', '.join([('{}: ' + numfmt).format(v, p)
                          for (v, p) in sorted(self.prob.items())])

    def __repr__(self):
        return "P({})".format(self.varname)

class BayesNode:
    """A conditional probability distribution for a boolean variable,
    P(X | parents). Part of a BayesNet."""

    def __init__(self, X, parents, cpt):
        """X is a variable name, and parents a sequence of variable
        names or a space-separated string.  cpt, the conditional
        probability table, takes one of these forms:

        * A number, the unconditional probability P(X=true). You can
          use this form when there are no parents.

        * A dict {v: p, ...}, the conditional probability distribution
          P(X=true | parent=v) = p. When there's just one parent.

        * A dict {(v1, v2, ...): p, ...}, the distribution P(X=true |
          parent1=v1, parent2=v2, ...) = p. Each key must have as many
          values as there are parents. You can use this form always;
          the first two are just conveniences.

        In all cases the probability of X being false is left implicit,
        since it follows from P(X=true).

        >>> X = BayesNode('X', '', 0.2)
        >>> Y = BayesNode('Y', 'P', {T: 0.2, F: 0.7})
        >>> Z = BayesNode('Z', 'P Q',
        ...    {(T, T): 0.2, (T, F): 0.3, (F, T): 0.5, (F, F): 0.7})
        """
        if isinstance(parents, str):
            parents = parents.split()

        # We store the table always in the third form above.
        if isinstance(cpt, (float, int)):  # no parents, 0-tuple
            cpt = {(): cpt}
        elif isinstance(cpt, dict):
            # one parent, 1-tuple
            if cpt and isinstance(list(cpt.keys())[0], bool):
                cpt = {(v,): p for v, p in cpt.items()}

        assert isinstance(cpt, dict)
        for vs, p in cpt.items():
            assert isinstance(vs, tuple) and len(vs) == len(parents)
            assert all(isinstance(v, bool) for v in vs)
            assert 0 <= p <= 1

        self.variable = X
        self.parents = parents
        self.cpt = cpt
        self.children = []

    def p(self, value, event):
        """Return the conditional probability
        P(X=value | parents=parent_values), where parent_values
        are the values of parents in event. (event must assign each
        parent a value.)
        >>> bn = BayesNode('X', 'Burglary', {T: 0.2, F: 0.625})
        >>> bn.p(False, {'Burglary': False, 'Earthquake': True})
        0.375"""
        assert isinstance(value, bool)
        ptrue = self.cpt[event_values(event, self.parents)]
        return ptrue if value else 1 - ptrue

    def sample(self, event):
        """Sample from the distribution for this variable conditioned
        on event's values for parent_variables. That is, return True/False
        at random according with the conditional probability given the
        parents."""
        return probability(self.p(True, event))

    def __repr__(self):
        return repr((self.variable, ' '.join(self.parents)))

class BayesNet:
    """Bayesian network containing only boolean-variable or multi-class nodes."""
    
    def __init__(self, node_specs=None):
        """Nodes must be ordered with parents before children."""
        self.nodes = []
        self.variables = []
        node_specs = node_specs or []
        for node_spec in node_specs:
            self.add(node_spec)

    def add(self, node_spec):
        """Add a node to the net. Supports both pre-constructed nodes and node specs."""
        if isinstance(node_spec, (BayesNode, MultiClassBayesNode)):
            # If already a node, add it directly
            node = node_spec
        else:
            # Otherwise, initialize a new node
            node = BayesNode(*node_spec)

        print(f"Node: {node_spec.variable}, Parents: {node_spec.parents}")
        assert all((parent in [n.variable for n in self.nodes]) for parent in node_spec.parents), \
        f"Missing parent for node {node_spec.variable}"

        assert node.variable not in self.variables
        assert all((parent in self.variables) for parent in node.parents)

        self.nodes.append(node)
        self.variables.append(node.variable)

        # Register children for the parent nodes
        for parent in node.parents:
            self.variable_node(parent).children.append(node)

    def variable_node(self, var):
        """Return the node for the variable named var."""
        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception(f"No such variable: {var}")

    def variable_values(self, var):
        """Return the domain of var."""
        return [True, False]

    def __repr__(self):
        return f"BayesNet({self.nodes!r})"
    
class MultiClassBayesNode:
    """A Bayesian node for multi-class variables."""
    def __init__(self, X, parents, cpt):
        """
        X: Name of the variable.
        parents: List of parent variable names.
        cpt: Conditional probability table, mapping tuples of parent values to
             dictionaries of target probabilities.
        """
        if isinstance(parents, str):
            parents = parents.split()
        self.variable = X
        self.parents = parents
        self.cpt = cpt
        self.children = []

    def p(self, value, event):
        """Return the conditional probability P(X=value | parents=parent_values)."""
        parent_values = tuple(event.get(p, None) for p in self.parents)
        probabilities = self.cpt.get(parent_values, {})
        return probabilities.get(value, 0)  # Default to 0 if value not found

    def sample(self, event):
        """Sample from the distribution for this variable given parent values."""
        parent_values = tuple(event.get(p, None) for p in self.parents)
        probabilities = self.cpt.get(parent_values, {})
        return probability_sampling(probabilities)

    def __repr__(self):
        return repr((self.variable, ' '.join(self.parents)))
    
def extend(s, var, val):
    """Copy dict s and extend it by setting var to val; return copy."""
    return {**s, var: val}
    
def enumerate_all(variables, e, bn):
    """Return the sum of those entries in P(variables | e{others})
    consistent with e, where P is the joint distribution represented
    by bn, and e{others} means e restricted to bn's other variables
    (the ones other than variables). Parents must precede children in variables."""
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
    """Return the conditional probability distribution of variable X
    given evidence e, from BayesNet bn. [Figure 14.9]
    >>> enumeration_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary
    ...  ).show_approx()
    'False: 0.716, True: 0.284'"""
    assert X not in e, "Query variable must be distinct from evidence"
    Q = ProbDist(X)
    for xi in bn.variable_values(X):
        Q[xi] = enumerate_all(bn.variables, extend(e, X, xi), bn)
    return Q.normalize()

def compute_cpt(data, target, parents, alpha=1):
    """
    Compute CPT with Laplace smoothing.
    
    Args:
        data: pandas DataFrame (training data)
        target: str, target variable
        parents: list of parent variable names
        alpha: smoothing parameter (default=1)
    
    Returns:
        cpt: dict { parent_values_tuple: { target_value: probability } }
    """
    target_values = data[target].cat.categories

    if not parents:
        # Marginal distribution of target
        counts = defaultdict(lambda: alpha)
        for val in data[target]:
            counts[val] += 1
        total = sum(counts.values())
        cpt = {(): {tv: counts[tv]/total for tv in counts}}
        return cpt

    # Determine possible parent combinations
    from itertools import product
    parent_values_list = [data[p].cat.categories for p in parents]
    parent_combinations = list(product(*parent_values_list)) if parents else [()]

    # Initialize counts with alpha
    counts = {pc: defaultdict(lambda: alpha) for pc in parent_combinations}

    # Count occurrences
    for _, row in data.iterrows():
        pv = tuple(row[p] for p in parents) if parents else ()
        tv = row[target]
        counts[pv][tv] += 1

    # Compute probabilities
    cpt = {}
    for pc in parent_combinations:
        total = sum(counts[pc].values())
        cpt[pc] = {tv: (counts[pc][tv] / total) for tv in counts[pc]}
        
    return cpt

def evaluate_bayes_net(bn, test_data, query_var):
    """
    Evaluate the Bayesian Network on a test dataset.
    
    Args:
        bn: Bayesian Network (BayesNet instance).
        test_data: Test dataset (pandas DataFrame).
        query_var: The target variable to predict.
    
    Returns:
        Accuracy of the predictions as a float.
    """
    correct = 0  # Count of correct predictions

    for _, row in test_data.iterrows():
        # Build evidence dictionary from test row
        evidence = {
            "Gender": row["Gender"],
            "Physical_Activity_Hours": row["Physical_Activity_Hours"],
#             "Consultation_History": row["Consultation_History"],
#             "Severity": row["Severity"],
            "Country": row["Country"],
#             "Stress_Level": row["Stress_Level"],
#             "Work_Stress_Index": row["Work_Stress_Index"],
            "Age": row["Age"],
#             "Work_Hours": row["Work_Hours"],
#             "Physical_Activity_Stress_Index": row["Physical_Activity_Stress_Index"],
#             "Sleep_Hours": row["Sleep_Hours"],
            "Occupation": row["Occupation"],
        }

        # Predict the target variable
        prediction = predict_bayes_net(bn, evidence, query_var)
        
        
        # Check if the prediction matches the actual value
        if prediction == row[query_var]:
            correct += 1

    # Calculate accuracy
    accuracy = correct / len(test_data)
    return accuracy

def event_values(event, variables):
    """Return a tuple of the values of variables in event.
    >>> event_values ({'A': 10, 'B': 9, 'C': 8}, ['C', 'A'])
    (8, 10)
    >>> event_values ((1, 2), ['C', 'A'])
    (1, 2)
    """
    if isinstance(event, tuple) and len(event) == len(variables):
        return event
    else:
        return tuple([event[var] for var in variables])
    
def predict_bayes_net(bn, evidence, query_var):
    """
    Predict the most likely value of a query variable given evidence using the Bayesian Network.
    
    Args:
        bn: Bayesian network.
        evidence: Dictionary of evidence variables and their values.
        query_var: Variable to predict.
    
    Returns:
        The most likely value of the query variable.
    """
    result = enumeration_ask(query_var, evidence, bn)
    return max(result.prob, key=lambda k: result.prob[k])