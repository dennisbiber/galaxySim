import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

class System:
    def __init__(self, minDistance, maxDistance):
        self.angle = random.uniform(0, 2 * np.pi)
        self.eccentricity = random.uniform(-0.2, 0.8)
        self.radius = random.uniform(minDistance, maxDistance)
        self.inclination = random.uniform(0, np.pi/1200)
        x = self.radius * np.cos(self.angle)
        y = self.radius * np.sin(self.angle)
        self.position = [x, y, 0]
        self.velocity = [0, 0, 0]  # Initial velocity vector

    def updateRadius(self):
        self.radius = np.sqrt(self.position[0]**2 + self.position[1]**2 + self.position[2]**2)

class SubSystem:
    def __init__(self, systems, center_of_orbit):
        self.systems = systems
        self.inclination = systems[0].inclination
        self.angle = systems[0].angle
        self.position = systems[0].position
        self.radius = systems[0].radius
        self.eccentricity = systems[0].eccentricity
        self.lastPosition = systems[0].position
        self.center_of_orbit = center_of_orbit

    def centerOfSubSystem(self):
        positions = []
        for system in self.systems:
            positions.append(system.position)
        positions_array = np.array(positions)
        centroid = np.mean(positions_array, axis=0)
        self.position = centroid.tolist()
        system.angle = self.angle
        # system.updateRadius()
        self.updateEccentricity()
    
    def update_centroid(self):
        centroid_change = np.array(self.position) - np.array(self.lastPosition)
        # Forward the change to all positions
        for system in self.systems:
            system.position += centroid_change
        # Update the centroid to the new value
        self.lastPosition = self.position

    def updateRadius(self):
        reference_new = np.array(self.center_of_orbit)

        # Calculate the normalized direction vector based on the angle
        direction_vector = np.array([np.cos(self.angle), np.sin(self.angle), 0])
        # Scale the direction vector by the distance
        offset_vector = direction_vector * self.radius
        
        # Calculate the new known point position maintaining the distance and angle
        self.position = reference_new + offset_vector
        

    def updateEccentricity(self):
        e = []
        for system in self.systems:
            e.append(system.eccentricity)
        self.eccentricity = np.mean(e)


def checkSubsystemProximity(subSystems):
    merge_pairs = []
    
    # Step 1: Identify pairs of subsystems to merge
    for i in range(len(subSystems)):
        for j in range(i + 1, len(subSystems)):  # Only check pairs (i, j) where i < j
            distance = np.sqrt(
                (subSystems[i].position[0] - subSystems[j].position[0])**2 +
                (subSystems[i].position[1] - subSystems[j].position[1])**2 +
                (subSystems[i].position[2] - subSystems[j].position[2])**2
            )
            if distance < 2:
                merge_pairs.append((i, j))
    
    # Step 2: Merge identified pairs
    merged_indices = set()  # To keep track of already merged subsystems
    new_subSystems = []
    
    for i, j in merge_pairs:
        if i not in merged_indices and j not in merged_indices:
            iSystems = subSystems[i].systems
            jSystems = subSystems[j].systems
            new_subSys = SubSystem([*iSystems, *jSystems], subSystems[i].center_of_orbit)
            new_subSystems.append(new_subSys)
            merged_indices.add(i)
            merged_indices.add(j)
    
    # Add subsystems that were not merged
    for idx in range(len(subSystems)):
        if idx not in merged_indices:
            new_subSystems.append(subSystems[idx])
    
    # Update the global subSystems list
    subSystems[:] = new_subSystems


def splitDivergingSubsystem(subsystem):
    new_subsystems = []
    systems_to_split = []

    # Find systems that need to be split
    for i, system1 in enumerate(subsystem.systems):
        for system2 in subsystem.systems:
            if system1 != system2:
                distance = np.sqrt(
                    (system1.position[0] - system2.position[0])**2 +
                    (system1.position[1] - system2.position[1])**2 +
                    (system1.position[2] - system2.position[2])**2
                )
                if distance > 3:
                    if system1 not in systems_to_split:
                        systems_to_split.append(system1)
                    if system2 not in systems_to_split:
                        systems_to_split.append(system2)

    # Create new subsystems based on the systems_to_split
    if systems_to_split:
        systems_remaining = [sys for sys in subsystem.systems if sys not in systems_to_split]
        
        if systems_remaining:
            new_subsystems.append(SubSystem(systems_remaining, subsystem.center_of_orbit))
        
        if len(systems_to_split) > 1:
            new_subsystems.append(SubSystem(systems_to_split, subsystem.center_of_orbit))
        elif len(systems_to_split) == 1:
            new_subsystems.append(SubSystem(systems_to_split, subsystem.center_of_orbit))

    return new_subsystems


def removeSubsystemInBlackhole(subsystems):
    to_remove = []
    
    for subSystem in subsystems:
        distance = np.sqrt(
            subSystem.position[0]**2 +
            subSystem.position[1]**2 +
            subSystem.position[2]**2
        )
        if distance < 0.5:
            print("HERE")
            to_remove.append(subSystem)
    
    for sub in to_remove:
        subsystems.remove(sub)


def updateSubSystems(subSystem, dt):
    for system in subSystem:
        a = system.radius  # Semi-major axis
        e = system.eccentricity  # Eccentricity
        r = a * (1 - e**2) / (1 + e * np.cos(system.angle))

        mu = 1.0
        # Calculate the new angle
        drdt = np.sqrt(mu * (2/r - 1/a))
        delta_angle = drdt * dt / r

        system.angle += delta_angle
        system.angle %= 2 * np.pi

        # Update the position
        system.position[0] = r * np.cos(system.angle)
        system.position[1] = r * np.sin(system.angle)
        system.position[2] = 0  # Keep in the XY plane for simplicity
                    

def updateSystemPosition(subSystem, dt):
    # Calculate the new radius based on the eccentricity
    # if len(subSystem.systems) > 1:
    #     updateSubSystems(subSystem.systems, dt)
    subSystem.updateRadius()
    a = subSystem.radius  # Semi-major axis
    e = subSystem.eccentricity  # Eccentricity
    i = subSystem.inclination  # Inclination angle (in radians)
    r = a * (1 - e**2) / (1 + e * np.cos(subSystem.angle))

    mu = 1.0
    # Calculate the new angle
    drdt = np.sqrt(mu * (2/r - 1/a))
    delta_angle = drdt * dt / r

    subSystem.angle += delta_angle
    subSystem.angle %= 2 * np.pi
    # subSystem.inclination += drdt * dt / r_incline

    # Calculate new x, y, z position based on orbital parameters
    x = r * np.cos(subSystem.angle)
    y = r * np.sin(subSystem.angle)
    z = y * np.sin(i)  # z varies with the inclination angle

    # Update position relative to the center of orbit
    subSystem.position[0] = x + subSystem.center_of_orbit[0]
    subSystem.position[1] = y + subSystem.center_of_orbit[1]
    subSystem.position[2] = z + subSystem.center_of_orbit[2]

    subSystem.update_centroid()

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
maxDistance = 1000
minDistance = 15
numSystems = 350
ax.set_xlim(-maxDistance, maxDistance)
ax.set_ylim(-maxDistance, maxDistance)
ax.set_zlim(-2, 5)
central = [0, 0, 0]

# Initialize plot elements
central_dot = ax.scatter([], [], [], 'ko', color="black", s=10)  # Central dot
moving_dot = ax.scatter([], [], [], 'ro', color="blue", s=5)       # Moving dot

subSystems = []
for idx in range(numSystems):
    newSystem = System(minDistance, maxDistance)
    newSubSystem = SubSystem([newSystem], central)
    subSystems.append(newSubSystem)

# Function to initialize the animation
def init():
    central_dot._offsets3d = ([0], [0], [0])
    moving_dot._offsets3d = ([], [], [])
    return central_dot, moving_dot
z_adjust = 0.0002
# Function to update the animation
def update(frame):
    global central
    # checkSubsystemProximity(subSystems)
    for sub in subSystems:
        sub.centerOfSubSystem()
        updateSystemPosition(sub, 2.0)
        removeSubsystemInBlackhole(sub.systems)
    central[2] += z_adjust
    # # Split diverging subsystems
    # new_subsystems = []
    # for sub in subSystems:
    #     new_subsystems.extend(splitDivergingSubsystem(sub))
    # subSystems[:] = new_subsystems
    
    # Flatten the list of lists into a single list of tuples
    positions = [pos for system in subSystems for pos in system.systems]
    
    # Extract the positions into separate lists for x, y, and z coordinates
    if positions:
        x, y, z = zip(*[(s.position[0], s.position[1], s.position[2]) for s in positions])
    else:
        x, y, z = [], [], []
    central_dot._offsets3d = ([central[0]], [central[1]], [central[2]])
    # Update plot elements
    moving_dot._offsets3d = (x, y, z)
    return central_dot, moving_dot

# Create the animation
ani = animation.FuncAnimation(fig, update, init_func=init, frames=2000, interval=50, blit=False)

# Show the plot
plt.show()
