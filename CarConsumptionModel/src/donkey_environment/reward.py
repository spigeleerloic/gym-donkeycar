import numpy as np



def calc_reward(self, done: bool) -> float:
    if done and self.hit == "none":
        return self.done_reward
    
    if len(self.markers) <= 0:
        return 0.0
    
    position = np.array([self.x, self.y, self.z])        
    distance_to_next_marker = np.linalg.norm(position - self.markers[self.next_marker])

    max_distance = 0.0 
    distance_towards_objective = distance_to_next_marker
    distance_marker = []
    for i in range(len(self.markers)):
        distance = np.linalg.norm(self.markers[i] - self.markers[(i + 1) % len(self.markers)])
        
        max_distance += distance
        distance_marker.append(distance)
        # next_marker can be the end of the circuit which means no more distance should be added
        if i >= self.next_marker and self.next_marker != 0:
            distance_towards_objective += distance
    
    self.maximal_distance = max_distance
    self.distance_towards_objective = distance_towards_objective
    self.distance_to_next_marker = distance_to_next_marker

    threshold = 1.0
    if distance_to_next_marker < threshold:
        self.next_marker = (self.next_marker + 1) % len(self.markers)
            
    normalized_distance = distance_towards_objective / max_distance
    # ensure 0 <= normalized_distance <= 1
    if normalized_distance > 1.0:
        normalized_distance = 1.0

    if self.hit != "none":
        return self.collision_reward - self.coef_collision_reward * normalized_distance
    
    return - normalized_distance 


def calc_reward(self, done: bool) -> float:

    if self.hit != "none":
        return self.collision_reward

    if done:
        return self.done_reward
    
    if len(self.markers) <= 0:
        return 0.0
    
    position = np.array([self.x, self.y, self.z])        
    distance_to_next_marker = np.linalg.norm(position - self.markers[self.next_marker])

    max_distance = 0.0 
    distance_towards_objective = distance_to_next_marker
    distance_marker = []
    for i in range(len(self.markers)):
        distance = np.linalg.norm(self.markers[i] - self.markers[(i + 1) % len(self.markers)])
        
        max_distance += distance
        distance_marker.append(distance)
        # next_marker can be the end of the circuit which means no more distance should be added
        if i >= self.next_marker and self.next_marker != 0:
            distance_towards_objective += distance
    
    self.maximal_distance = max_distance
    self.distance_towards_objective = distance_towards_objective
    self.distance_to_next_marker = distance_to_next_marker

    threshold = 1.0
    if distance_to_next_marker < threshold:
        self.next_marker = (self.next_marker + 1) % len(self.markers)
            
    normalized_distance = distance_towards_objective / max_distance
    # ensure 0 <= normalized_distance <= 1
    if normalized_distance > 1.0:
        normalized_distance = 1.0

    return (1.0 - normalized_distance)