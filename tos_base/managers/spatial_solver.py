from dataclasses import dataclass, field
from typing import Set, Dict, List, Tuple, Optional, Union, Iterable
from collections import deque
import random
import copy

# Import the relationship classes from the existing codebase
from ragen.env.spatial.Base.tos_base.core.relationship import (
    PairwiseRelationship, PairwiseRelationshipDiscrete,
    LocalRelationship, RelationTriple, relationship_applies
)


@dataclass
class Variable:
    name: str
    domain: Set[Tuple[int, int]] = field(default_factory=set)


@dataclass
class Constraint:
    var1_name: str
    var2_name: str
    relation: Union[PairwiseRelationship, PairwiseRelationshipDiscrete, LocalRelationship]
    anchor_pos: Optional[Tuple[int, int]] = None
    anchor_ori: Tuple[int, int] = (0, 1)


class AC3Solver:
    def __init__(self, variables: Dict[str, Variable], constraints: List[Constraint]):
        self.variables = variables
        self.constraints = constraints
        self.adjacency = self._build_adjacency()

    def _build_adjacency(self) -> Dict[str, Set[str]]:
        """Build adjacency table for efficient neighbor lookup."""
        adj = {name: set() for name in self.variables}
        for c in self.constraints:
            adj[c.var1_name].add(c.var2_name)
            adj[c.var2_name].add(c.var1_name)
        return adj

    def add_constraint(self, constraint: Constraint) -> None:
        """Add new constraint and update adjacency."""
        self.constraints.append(constraint)
        self.adjacency[constraint.var1_name].add(constraint.var2_name)
        self.adjacency[constraint.var2_name].add(constraint.var1_name)

    def solve(self) -> bool:
        queue = deque(self._get_all_arcs())
        while queue:
            var1_name, var2_name = queue.popleft()
            if self._revise(var1_name, var2_name):
                if not self.variables[var1_name].domain:
                    raise ValueError(f"Domain is empty for {var1_name}")
                # Use adjacency for efficient neighbor lookup
                for neighbor in self.adjacency[var1_name]:
                    if neighbor != var2_name:
                        queue.append((neighbor, var1_name))
        return True

    def solve_incremental(self, new_constraint: Constraint) -> bool:
        """Add constraint and solve incrementally."""
        self.add_constraint(new_constraint)
        # Only propagate constraints involving the new constraint variables
        queue = deque([(new_constraint.var1_name, new_constraint.var2_name),
                       (new_constraint.var2_name, new_constraint.var1_name)])
        while queue:
            var1_name, var2_name = queue.popleft()
            if self._revise(var1_name, var2_name):
                if not self.variables[var1_name].domain:
                    raise ValueError(f"Domain is empty for {var1_name}")
                for neighbor in self.adjacency[var1_name]:
                    if neighbor != var2_name:
                        queue.append((neighbor, var1_name))
        return True

    def _get_all_arcs(self) -> List[Tuple[str, str]]:
        arcs = []
        for c in self.constraints:
            arcs.append((c.var1_name, c.var2_name))
            arcs.append((c.var2_name, c.var1_name))
        return arcs

    def _revise(self, var1_name: str, var2_name: str) -> bool:
        revised = False
        var1_domain = self.variables[var1_name].domain.copy()
        for pos1 in var1_domain:
            if not self._has_support(var1_name, var2_name, pos1):
                self.variables[var1_name].domain.remove(pos1)
                revised = True
        return revised

    def _constraints_between(self, a: str, b: str) -> List[Constraint]:
        return [c for c in self.constraints if (c.var1_name == a and c.var2_name == b) or (c.var1_name == b and c.var2_name == a)]

    def _has_support(self, var1_name: str, var2_name: str, pos1: Tuple[int, int]) -> bool:
        rules = self._constraints_between(var1_name, var2_name)
        if not rules:
            return True
        var2_domain = self.variables[var2_name].domain
        for pos2 in var2_domain:
            ok_all = True
            for rule in rules:
                # Skip LocalRelationship without known anchor position
                if isinstance(rule.relation, LocalRelationship) and rule.anchor_pos is None:
                    continue
                if rule.var1_name == var1_name:
                    ok = relationship_applies(pos1, pos2, rule.relation, rule.anchor_pos, rule.anchor_ori)
                else:
                    ok = relationship_applies(pos2, pos1, rule.relation, rule.anchor_pos, rule.anchor_ori)
                if not ok:
                    ok_all = False
                    break
            if ok_all:
                return True
        return False


class SpatialSolver:
    """Spatial constraint solver with AC-3 and simple metrics."""

    def __init__(self, all_object_names: List[str], grid_size: int):
        self.grid_size = int(grid_size)
        self.variables: Dict[str, Variable] = {}
        self.constraints: List[Constraint] = []
        for name in all_object_names:
            self.variables[name] = Variable(name=name, domain=set())
        self.solver: Optional[AC3Solver] = None

    def set_initial_position(self, name: str, position: Tuple[int, int]):
        if name not in self.variables:
            self.variables[name] = Variable(name=name, domain=set())
        self.variables[name].domain = {tuple(position)}

    def add_observation(self, relation_triples: List[RelationTriple]):
        """Add observations using incremental constraint solving."""
        new_constraints = []
        for triple in relation_triples:
            anchor_pos = None
            if triple.anchor_name and triple.anchor_name in self.variables:
                a_dom = self.variables[triple.anchor_name].domain
                if len(a_dom) == 1:
                    anchor_pos = next(iter(a_dom))
            
            constraint = Constraint(
                var1_name=triple.obj_a,
                var2_name=triple.obj_b,
                relation=triple.relation,
                anchor_pos=anchor_pos,
                anchor_ori=triple.anchor_ori or (0, 1),
            )
            new_constraints.append(constraint)
            self._ensure_domain_initialized(triple.obj_a)
            self._ensure_domain_initialized(triple.obj_b)

        # Use incremental solving if solver exists, otherwise full solve
        if self.solver is None:
            self.constraints.extend(new_constraints)
            self.solver = AC3Solver(self.variables, self.constraints)
            self.solver.solve()
        else:
            for constraint in new_constraints:
                self.solver.solve_incremental(constraint)

    def get_possible_positions(self) -> Dict[str, Set[Tuple[int, int]]]:
        # Ensure unconstrained variables have full domain
        for name in self.variables:
            self._ensure_domain_initialized(name)
        return {name: var.domain.copy() for name, var in self.variables.items()}

    def get_relationship(self, obj1_name: str, obj2_name: str,
                         anchor_ori: Tuple[int, int] = (0, 1),
                         discrete: bool = False) -> Optional[PairwiseRelationship]:
        if (obj1_name not in self.variables or obj2_name not in self.variables or
                len(self.variables[obj1_name].domain) != 1 or
                len(self.variables[obj2_name].domain) != 1):
            return None
        pos1 = next(iter(self.variables[obj1_name].domain))
        pos2 = next(iter(self.variables[obj2_name].domain))
        if discrete:
            return PairwiseRelationshipDiscrete.relationship(pos1, pos2, anchor_ori)
        return PairwiseRelationship.relationship(pos1, pos2, anchor_ori, full=True)

    def _ensure_domain_initialized(self, name: str):
        if not self.variables[name].domain:
            g = int(self.grid_size)
            self.variables[name].domain = {(x, y) for x in range(-g, g) for y in range(-g, g)}

    def copy(self) -> 'SpatialSolver':
        """Create a deep copy of the solver for simulation."""
        new_solver = SpatialSolver([], self.grid_size)
        # Deep copy variables with their domains
        new_solver.variables = {
            name: Variable(name=name, domain=set(var.domain))
            for name, var in self.variables.items()
        }
        # Deep copy constraints and solver state
        new_solver.constraints = copy.deepcopy(self.constraints)
        if self.solver is not None:
            new_solver.solver = AC3Solver(new_solver.variables, new_solver.constraints)
        return new_solver

    # ---- Metrics ----
    def compute_metrics(self, mode: str = 'dir', max_samples_per_var: int = 50,
                        anchor_ori: Tuple[int, int] = (0, 1)) -> tuple[
        Dict[str, int], int, Dict[Tuple[str, str], Set[str]], int
    ]:
        """
        mode: 'disc' for PairwiseRelationshipDiscrete; 'dir' for direction-only (continuous, full=False).
        Returns: (domain_sizes, total_positions, pair_rel_sets, total_relationships)
        """
        # Domain sizes
        domain_sizes: Dict[str, int] = {}
        for name in self.variables:
            self._ensure_domain_initialized(name)
            domain_sizes[name] = len(self.variables[name].domain)
        total_positions = sum(domain_sizes.values())

        # Possible relationship sets per unordered pair
        names = sorted(self.variables.keys())
        rel_sets: Dict[Tuple[str, str], Set[str]] = {}

        def _sample(domain: Set[Tuple[int, int]], k: int) -> List[Tuple[int, int]]:
            if len(domain) <= k:
                return list(domain)
            return random.sample(list(domain), k)

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                da = _sample(self.variables[a].domain, max_samples_per_var)
                db = _sample(self.variables[b].domain, max_samples_per_var)
                s: Set[str] = set()
                for pa in da:
                    for pb in db:
                        if mode == 'disc':
                            rel = PairwiseRelationshipDiscrete.relationship(pa, pb, anchor_ori)
                            s.add(rel.to_string('allo'))
                        else:
                            rel = PairwiseRelationship.relationship(pa, pb, anchor_ori, full=False)
                            s.add(rel.to_string('allo'))
                rel_sets[(a, b)] = s

        total_relationships = sum(len(v) for v in rel_sets.values())
        return domain_sizes, total_positions, rel_sets, total_relationships


# Convenience API for callers
def make_solver_with_initial(objects: Iterable[str], grid_size: int,
                             set_initial_anchor: bool = True) -> SpatialSolver:
    s = SpatialSolver(list(objects), grid_size)
    if set_initial_anchor and 'initial_pos' in s.variables:
        s.set_initial_position('initial_pos', (0, 0))
    return s


