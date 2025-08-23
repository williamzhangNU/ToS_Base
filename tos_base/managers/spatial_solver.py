from dataclasses import dataclass, field
from typing import Set, Dict, List, Tuple, Optional, Union, Iterable
import random
import copy

# Import the relationship classes from the existing codebase
from ragen.env.spatial.Base.tos_base.core.relationship import (
    PairwiseRelationship, PairwiseRelationshipDiscrete, ProximityRelationship,
    RelationTriple, CardinalBins, StandardDistanceBins
)
from ragen.env.spatial.Base.tos_base.utils.relationship_utils import relationship_applies, generate_points_for_relationship


@dataclass
class Variable:
    name: str
    domain: Set[Tuple[int, int]] = field(default_factory=set)


@dataclass
class Constraint:
    var1_name: str
    var2_name: str
    relation: Union[PairwiseRelationship, PairwiseRelationshipDiscrete, ProximityRelationship]
    orientation: Tuple[int, int] = (0, 1)
    
    def __eq__(self, other):
        if not isinstance(other, Constraint):
            return False
        return (self.var1_name == other.var1_name and 
                self.var2_name == other.var2_name and
                self.relation == other.relation and
                self.orientation == other.orientation)
    
    def __hash__(self):
        return hash((self.var1_name, self.var2_name, self.relation, self.orientation))


class AC3Solver:
    def __init__(self, variables: Dict[str, Variable], constraints: List[Constraint]):
        self.variables = variables
        self.adjacency: Dict[str, Set[str]] = {name: set() for name in self.variables}
        # Group constraints by variable pairs (undirected)
        self.arc_constraints: Dict[frozenset, Set[Constraint]] = {}
        
        for c in constraints:
            self.add_constraint(c)

    def add_constraint(self, constraint: Constraint) -> bool:
        """Add constraint to arc predicates. Returns True if arc is new or predicate changed."""
        var1, var2 = constraint.var1_name, constraint.var2_name
        # Use frozenset for undirected arc storage
        arc = frozenset({var1, var2})
        
        # Update adjacency
        self.adjacency.setdefault(var1, set()).add(var2)
        self.adjacency.setdefault(var2, set()).add(var1)
        
        # Add constraint to arc predicate
        old_constraints = self.arc_constraints.get(arc, set()).copy()
        self.arc_constraints.setdefault(arc, set()).add(constraint)
        
        # Return True if predicate changed (new constraint added)
        return old_constraints != self.arc_constraints[arc]

    def copy(self) -> 'AC3Solver':
        new_variables = {
            name: Variable(name=name, domain=set(var.domain))
            for name, var in self.variables.items()
        }

        flat = {c for bucket in self.arc_constraints.values() for c in bucket}
        new_constraints = [copy.deepcopy(c) for c in flat]
        return AC3Solver(new_variables, new_constraints)

    def __deepcopy__(self, memo):
        return self.copy()

    def propagate(self, changed_arcs: Optional[Set[Tuple[str, str]]] = None) -> bool:
        """AC-3 propagation from changed arcs or all arcs."""
        if changed_arcs is None:
            # All directed arcs
            work: Set[Tuple[str, str]] = {(a, b) for a, ns in self.adjacency.items() for b in ns}
        else:
            work = changed_arcs.copy()
                
        while work:
            var1_name, var2_name = work.pop()
            if self._revise(var1_name, var2_name):
                if not self.variables[var1_name].domain:
                    raise ValueError(f"Domain is empty for {var1_name}")
                # Add all incoming arcs to var1 (except the one we just processed)
                for neighbor in self.adjacency[var1_name]:
                    if neighbor != var2_name:
                        work.add((neighbor, var1_name))
        return True

    def _revise(self, var1_name: str, var2_name: str) -> bool:
        revised = False
        var1_domain = self.variables[var1_name].domain.copy()
        for pos1 in var1_domain:
            if not self._has_support(var1_name, var2_name, pos1):
                self.variables[var1_name].domain.remove(pos1)
                revised = True
        return revised

    def _constraints_between(self, a: str, b: str) -> Set[Constraint]:
        """Get all constraints between variables a and b (undirected)."""
        arc = frozenset({a, b})
        return self.arc_constraints.get(arc, set())

    def _has_support(self, var1_name: str, var2_name: str, pos1: Tuple[int, int]) -> bool:
        """
        Check if pos1 from var1 has support in var2's domain.
        For each value pos2 in var2's domain, check if ALL constraints are satisfied (conjunction).
        """
        constraints = self._constraints_between(var1_name, var2_name)
        if not constraints:
            return True
            
        var2_domain = self.variables[var2_name].domain
        for pos2 in var2_domain:
            # Check if ALL constraints are satisfied for this pos2 (conjunction)
            all_satisfied = True
            for constraint in constraints:
                # Determine the correct order based on constraint definition
                if constraint.var1_name == var1_name:
                    # Constraint is defined as var1 -> var2, so check pos1 -> pos2
                    satisfied = relationship_applies(pos1, pos2, constraint.relation, constraint.orientation)
                else:
                    # Constraint is defined as var2 -> var1, so check pos2 -> pos1
                    satisfied = relationship_applies(pos2, pos1, constraint.relation, constraint.orientation)
                
                if not satisfied:
                    all_satisfied = False
                    break
            
            # If ALL constraints are satisfied for this pos2, then pos1 has support
            if all_satisfied:
                return True
                
        # No value in var2's domain satisfies all constraints
        return False


class SpatialSolver:
    """Spatial constraint solver with AC-3 and simple metrics."""

    def __init__(self, all_object_names: List[str], grid_size: int):
        self.grid_size = int(grid_size)
        variables = {name: Variable(name=name, domain=set()) for name in all_object_names}
        self.solver = AC3Solver(variables, [])
        self.set_initial_position('initial_pos', (0, 0))

    def set_initial_position(self, name: str, position: Tuple[int, int]):
        if name not in self.solver.variables:
            self.solver.variables[name] = Variable(name=name, domain=set())
        self.solver.variables[name].domain = {tuple(position)}

    def add_observation(self, relation_triples: List[RelationTriple]):
        """Add observations: seed domains cheaply, then solve all constraints."""
        new_constraints: List[Constraint] = []
        # 1) seeding from pairwise constraints only
        g = int(self.grid_size)
        x_rng, y_rng = (-g, g), (-g, g)
        for t in relation_triples:
            cons = Constraint(t.subject, t.anchor, t.relation, t.orientation or (0, 1))
            new_constraints.append(cons)
            # seed subject via anchor for pairwise only
            if isinstance(t.relation, (PairwiseRelationship, PairwiseRelationshipDiscrete)):
                s_dom = self.solver.variables[t.subject].domain
                a_dom = self.solver.variables[t.anchor].domain
                if not s_dom and 1 <= len(a_dom) <= 10:
                    domain = set()
                    for anchor_pt in a_dom:
                        domain |= generate_points_for_relationship(anchor_pt, t.relation, x_rng, y_rng, t.orientation or (0, 1))
                    self.solver.variables[t.subject].domain = domain
            
        for t in relation_triples:
            # ensure domains exist
            self._ensure_domain_initialized(t.subject)
            self._ensure_domain_initialized(t.anchor)

        # 2) install constraints and propagate from changed arcs only
        changed_arcs = set()
        for c in new_constraints:
            if self.solver.add_constraint(c):
                # Track arcs that need propagation (both directions)
                changed_arcs.add((c.var1_name, c.var2_name))
                changed_arcs.add((c.var2_name, c.var1_name))
        
        if changed_arcs:
            self.solver.propagate(changed_arcs)  # Only propagate from changed arcs

    def get_possible_positions(self) -> Dict[str, Set[Tuple[int, int]]]:
        # Ensure unconstrained variables have full domain
        for name in self.solver.variables:
            self._ensure_domain_initialized(name)
        return {name: var.domain.copy() for name, var in self.solver.variables.items()}

    def get_relationship(self, obj1_name: str, obj2_name: str,
                         perspective: Tuple[int, int] = (0, 1),
                         discrete: bool = True, bin_system=None, distance_bin_system=None,
                         perspective_type: str = 'allo') -> Optional[PairwiseRelationship]:
        if (obj1_name not in self.solver.variables or obj2_name not in self.solver.variables or
                len(self.solver.variables[obj1_name].domain) != 1 or
                len(self.solver.variables[obj2_name].domain) != 1):
            return None
        pos1 = next(iter(self.solver.variables[obj1_name].domain))
        pos2 = next(iter(self.solver.variables[obj2_name].domain))
        if discrete:
            bin_system = bin_system or CardinalBins()
            distance_bin_system = distance_bin_system or StandardDistanceBins()
            return PairwiseRelationshipDiscrete.relationship(pos1, pos2, perspective, bin_system, distance_bin_system, perspective_type)
        return PairwiseRelationship.relationship(pos1, pos2, perspective, full=True)

    def _ensure_domain_initialized(self, name: str):
        if not self.solver.variables[name].domain:
            g = int(self.grid_size)
            self.solver.variables[name].domain = {(x, y) for x in range(-g, g + 1) for y in range(-g, g + 1)}

    def copy(self) -> 'SpatialSolver':
        new_spatial = object.__new__(SpatialSolver)
        new_spatial.grid_size = self.grid_size
        new_spatial.solver = self.solver.copy()
        return new_spatial

    # ---- Metrics ----
    def compute_metrics(self, mode: str = 'dir', max_samples_per_var: int = 50,
                        perspective: Tuple[int, int] = (0, 1), bin_system=None, distance_bin_system=None) -> tuple[
        Dict[str, int], int, Dict[Tuple[str, str], Set[str]], int
    ]:
        """
        mode: 'disc' for PairwiseRelationshipDiscrete; 'dir' for direction-only (continuous, full=False).
        Returns: (domain_sizes, total_positions, pair_rel_sets, total_relationships)
        """
        # Domain sizes
        domain_sizes: Dict[str, int] = {}
        for name in self.solver.variables:
            self._ensure_domain_initialized(name)
            domain_sizes[name] = len(self.solver.variables[name].domain)
        total_positions = sum(domain_sizes.values())

        # Possible relationship sets per unordered pair
        names = sorted(self.solver.variables.keys())
        rel_sets: Dict[Tuple[str, str], Set[str]] = {}

        def _sample(domain: Set[Tuple[int, int]], k: int) -> List[Tuple[int, int]]:
            if len(domain) <= k:
                return list(domain)
            return random.sample(list(domain), k)

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                da = _sample(self.solver.variables[a].domain, max_samples_per_var)
                db = _sample(self.solver.variables[b].domain, max_samples_per_var)
                s: Set[str] = set()
                for pa in da:
                    for pb in db:
                        if mode == 'disc':
                            bin_system = bin_system or CardinalBins()
                            distance_bin_system = distance_bin_system or StandardDistanceBins()
                            rel = PairwiseRelationshipDiscrete.relationship(pa, pb, perspective, bin_system, distance_bin_system, 'allo')
                            s.add(rel.to_string('allo'))
                        else:
                            rel = PairwiseRelationship.relationship(pa, pb, perspective, full=False)
                            s.add(rel.to_string('allo'))
                rel_sets[(a, b)] = s

        total_relationships = sum(len(v) for v in rel_sets.values())
        return domain_sizes, total_positions, rel_sets, total_relationships
    
    def compute_allocentric_bin_metrics(self, max_samples_per_var: int = 50,
                                        perspective: Tuple[int, int] = (0, 1), bin_system=None) -> tuple[
        Dict[str, int], int, Dict[Tuple[str, str], Set[str]], int
    ]:
        """Deprecated: Use compute_metrics(mode='disc') instead."""
        return self.compute_metrics('disc', max_samples_per_var, perspective, bin_system)
