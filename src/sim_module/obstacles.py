# obstacles.py
"""
Obstacle definitions and management for 2D obstacle avoidance
Supports multiple obstacle types: circles, ellipses, rectangles
"""

from dataclasses import dataclass, field
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass
class Obstacle:
    """
    Generic obstacle class for 2D workspace
    """
    center: np.ndarray  # [x, y] center position
    radius: float       # Radius for circle, or equivalent safety radius
    type: str = 'circle'  # 'circle', 'ellipse', 'rectangle'
    
    # Additional parameters for different types
    semi_axes: Tuple[float, float] = None  # (a, b) for ellipse
    orientation: float = 0.0  # Rotation angle in radians
    dimensions: Tuple[float, float] = None  # (width, height) for rectangle
    
    # Visualization
    color: str = 'red'
    alpha: float = 0.3
    
    def __post_init__(self):
        """Initialize derived parameters"""
        if isinstance(self.center, (list, tuple)):
            self.center = np.array(self.center, dtype=float)
    
    def distance_to_point(self, p: np.ndarray) -> float:
        """
        Calculate signed distance from point to obstacle surface
        Positive = safe (outside), Negative = collision (inside)
        
        Args:
            p: Point position [x, y]
            
        Returns:
            Signed distance to obstacle surface
        """
        # Only circle type supported for dual obstacle scenario
        return np.linalg.norm(p - self.center) - self.radius
    
    def get_gradient(self, p: np.ndarray) -> np.ndarray:
        """
        Get gradient of distance function at point p
        Used for CBF constraint computation
        
        Args:
            p: Point position [x, y]
            
        Returns:
            Gradient vector [2]
        """
        # Only circle type supported for dual obstacle scenario
        p_rel = p - self.center
        dist = np.linalg.norm(p_rel)
        if dist < 1e-6:
            return np.zeros(2)
        return p_rel / dist
    
    def compute_cbf_constraint(self, p: np.ndarray, v: np.ndarray, 
                              Mpinv: np.ndarray, Dp: np.ndarray, 
                              Kp: np.ndarray, pref: np.ndarray,
                              alpha: float) -> Tuple[np.ndarray, float]:
        """
        Compute CBF constraint for this obstacle
        Returns A, b such that A @ u >= b ensures safety
        
        Args:
            p: Current position [2]
            v: Current velocity [2]
            Mpinv: Inverse mass matrix [2x2]
            Dp: Damping matrix [2x2]
            Kp: Stiffness matrix [2x2]
            pref: Reference position [2]
            alpha: CBF gain
            
        Returns:
            A: Constraint matrix coefficient [2]
            b: Constraint RHS scalar
        """
        # Barrier function: h = distance^2 to obstacle
        dist = self.distance_to_point(p)
        h = dist**2
        
        # Gradient of h
        grad_dist = self.get_gradient(p)
        grad_h = 2 * dist * grad_dist
        
        # Time derivative: ḣ = 2*dist*grad_dist^T*v
        h_dot = 2 * dist * np.dot(grad_dist, v)
        
        # For control affine system: ẍ = M^{-1}*(u - D*v - K*(p-pref))
        # ḧ needs to account for this
        
        # Simplified: use first-order CBF on velocity level
        # ḣ + α*h >= 0
        # grad_h^T * ẋ + α*h >= 0
        # grad_h^T * v + α*h >= 0
        
        # For control: grad_h^T * M^{-1} * u >= ...
        A = grad_h @ Mpinv
        
        # RHS: -grad_h^T * M^{-1} * (D*v + K*(p-pref)) - α*h
        rhs = -(grad_h @ Mpinv @ (Dp @ v + Kp @ (p - pref)) + alpha * h)
        
        return A, rhs
    
    def plot(self, ax: plt.Axes, **kwargs):
        """
        Plot obstacle on matplotlib axes
        
        Args:
            ax: Matplotlib axes object
            **kwargs: Additional plot parameters
        """
        color = kwargs.get('color', self.color)
        alpha = kwargs.get('alpha', self.alpha)
        label = kwargs.get('label', 'Obstacle')
        
        # Only circle type supported for dual obstacle scenario
        circle = plt.Circle(self.center, self.radius, 
                          color=color, alpha=alpha, label=label)
        ax.add_patch(circle)


class ObstacleScene:
    """
    Manages multiple obstacles in the workspace
    """
    
    def __init__(self, obstacles: Optional[List[Obstacle]] = None):
        self.obstacles = obstacles if obstacles is not None else []
    
    def add_obstacle(self, obstacle: Obstacle):
        """Add an obstacle to the scene"""
        self.obstacles.append(obstacle)
    
    def remove_obstacle(self, index: int):
        """Remove obstacle by index"""
        if 0 <= index < len(self.obstacles):
            self.obstacles.pop(index)
    
    def clear(self):
        """Remove all obstacles"""
        self.obstacles = []
    
    def is_collision_free(self, p: np.ndarray, safety_margin: float = 0.0) -> bool:
        """
        Check if point is collision-free with all obstacles
        
        Args:
            p: Point position [x, y]
            safety_margin: Additional safety margin (meters)
            
        Returns:
            True if collision-free, False if collision
        """
        for obs in self.obstacles:
            if obs.distance_to_point(p) < safety_margin:
                return False
        return True
    
    def get_closest_obstacle(self, p: np.ndarray) -> Tuple[Optional[Obstacle], float]:
        """
        Find closest obstacle to point
        
        Args:
            p: Point position [x, y]
            
        Returns:
            (closest_obstacle, distance) or (None, inf) if no obstacles
        """
        if not self.obstacles:
            return None, float('inf')
        
        min_dist = float('inf')
        closest_obs = None
        
        for obs in self.obstacles:
            dist = obs.distance_to_point(p)
            if dist < min_dist:
                min_dist = dist
                closest_obs = obs
        
        return closest_obs, min_dist
    
    def get_all_cbf_constraints(self, p: np.ndarray, v: np.ndarray,
                               Mpinv: np.ndarray, Dp: np.ndarray,
                               Kp: np.ndarray, pref: np.ndarray,
                               alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all CBF constraints from all obstacles

        Returns:
            A: Constraint matrix [n_obstacles, 2]
            b: Constraint vector [n_obstacles]
        """
        if not self.obstacles:
            return np.empty((0, 2)), np.empty(0)
        
        A_list = []
        b_list = []
        
        for obs in self.obstacles:
            A_i, b_i = obs.compute_cbf_constraint(p, v, Mpinv, Dp, Kp, pref, alpha)
            A_list.append(A_i)
            b_list.append(b_i)
        
        return np.array(A_list), np.array(b_list)
    
    def get_all_cbf_constraints_for_points(self, points: list, v: np.ndarray,
                                           Mpinv: np.ndarray, Dp: np.ndarray,
                                           Kp: np.ndarray, pref: np.ndarray,
                                           alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CBF constraints for multiple evaluation points (e.g., joint1 and EE).
        Returns stacked A rows and corresponding b entries for all (point, obstacle).
        """
        if not self.obstacles or len(points) == 0:
            return np.empty((0, 2)), np.empty(0)

        A_rows = []
        b_vals = []
        # For each point (e.g., joint1 then EE), compute constraints with each obstacle
        for p_point in points:
            # note: we reuse the same velocity vector v as approximation for each point
            for obs in self.obstacles:
                A_i, b_i = obs.compute_cbf_constraint(p_point, v, Mpinv, Dp, Kp, pref, alpha)
                # ensure A_i is 1D array length 2
                A_rows.append(np.atleast_1d(A_i).astype(float))
                b_vals.append(float(b_i))

        if len(A_rows) == 0:
            return np.empty((0, 2)), np.empty(0)

        return np.vstack(A_rows), np.array(b_vals)

    def plot_all(self, ax: plt.Axes, **kwargs):
        """Plot all obstacles"""
        for i, obs in enumerate(self.obstacles):
            label = kwargs.get('label', 'Obstacle') if i == 0 else None
            obs.plot(ax, label=label, **kwargs)
    
    def get_workspace_bounds(self, margin: float = 0.1) -> Tuple[float, float, float, float]:
        """
        Get bounding box of all obstacles
        
        Returns:
            (x_min, x_max, y_min, y_max)
        """
        if not self.obstacles:
            return -0.5, 0.5, -0.5, 0.5
        
        x_min = min(obs.center[0] - obs.radius for obs in self.obstacles) - margin
        x_max = max(obs.center[0] + obs.radius for obs in self.obstacles) + margin
        y_min = min(obs.center[1] - obs.radius for obs in self.obstacles) - margin
        y_max = max(obs.center[1] + obs.radius for obs in self.obstacles) + margin
        
        return x_min, x_max, y_min, y_max


# ==================== Utility Functions ====================
# Note: Predefined obstacle scenarios removed - obstacles are defined in scenario_setup.py

def check_path_collision(path: np.ndarray, scene: ObstacleScene, 
                        safety_margin: float = 0.0) -> bool:
    """
    Check if a path collides with obstacles
    
    Args:
        path: Path as [N, 2] array
        scene: Obstacle scene
        safety_margin: Additional safety margin
        
    Returns:
        True if collision-free, False if collision
    """
    for p in path:
        if not scene.is_collision_free(p, safety_margin):
            return False
    return True


def get_collision_points(path: np.ndarray, scene: ObstacleScene) -> List[int]:
    """
    Get indices of path points that are in collision
    
    Args:
        path: Path as [N, 2] array
        scene: Obstacle scene
        
    Returns:
        List of collision point indices
    """
    collision_indices = []
    for i, p in enumerate(path):
        if not scene.is_collision_free(p):
            collision_indices.append(i)
    return collision_indices