import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

class MovingObject:
    def __init__(self, position, velocity, angle):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([velocity * np.cos(angle), velocity * np.sin(angle), velocity * np.sin(angle)**2], dtype=float)
        self.acceleration = np.array([0.0, 0.0, 0.0], dtype=float)

    def update_velocity(self):
        self.velocity += self.acceleration

    def update_position(self):
        self.position += self.velocity

    def update_acceleration(self, new_acceleration):
        self.acceleration = np.array(new_acceleration, dtype=float)

    def update_trajectory(self, angle):
        speed = np.linalg.norm(self.velocity)
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle), speed], dtype=float)

class Field:
    def __init__(self, position, radius, intensity_dt=2, energy=1):
        self.position = np.array(position, dtype=float)
        self.radius = radius
        self.dldt = intensity_dt
        self.energy = energy 

    def is_inside(self, obj_position):
        distance = np.linalg.norm(obj_position - self.position)
        return distance <= self.radius

    def field_effect(self, obj_position):
        displacement = self.position - obj_position
        distance = np.linalg.norm(displacement)
        if distance == 0:
            return np.array([0.0, 0.0, 0.0])  # Avoid division by zero
        effect_magnitude = self.dldt * self.energy / distance**2  # Example effect: inversely proportional to distance
        if effect_magnitude > 3:
            effect_magnitude = 3
        direction =  displacement / distance
        return effect_magnitude * direction


class Animation:
    def __init__(self, size, moving_object, field, time_step=0.01, total_time=100):
        self.moving_object = moving_object
        self.field = field
        self.time_step = time_step
        self.total_time = total_time
        fig = plt.figure()
        self.fig = fig
        ax = fig.add_subplot(111, projection="3d")
        markerPos = [pos for pos in field]
        X, Y, Z = zip(*[(s.position[0], s.position[1], s.position[2]) for s in markerPos])
        field_marker = ax.scatter([X], [Y], [Z], color='red')
        self.field_marker = field_marker
        ax.set_xlim(-size, size)
        ax.set_ylim(-size, size)
        ax.set_zlim(-size, size)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Object Trajectory and Field Influence')
        self.ax = ax
        self.time = 0
        self.dot = ax.scatter([], [], [], color='blue', s=10)
        self.anim = animation.FuncAnimation(self.fig, self.update, frames=int(self.total_time / self.time_step), interval=50, blit=False)

    def update(self, frame):
        for field in self.field:
            if field.is_inside(self.moving_object.position):
                effect = field.field_effect(self.moving_object.position)
                self.moving_object.update_acceleration(effect)
            else:
                pass

        self.moving_object.update_velocity()
        self.moving_object.update_position()
        X, Y, Z = self.moving_object.position
        self.dot.remove()  # Remove the previous dot
        self.dot = self.ax.scatter(X, Y, Z, color='blue', s=10)  # Plot the new dot
        return self.dot, self.field_marker

    def show(self):
        plt.show()

size = 30
# Initialize the moving object and field
moving_object = MovingObject(position=[-size, 0, 0], velocity=0.5, angle=np.pi/20)
num_dots = 200
coords = []
radii = []
energy = []
intensity = []
for i in range(num_dots):
    coords.append([random.uniform(-size*0.95, size*0.95), random.uniform(-size*0.6, size*0.6), random.uniform(-size*0.6, size*0.6)])
    radii.append(random.uniform(random.uniform(2, 10), random.uniform(11, 30)))
    energy.append(random.uniform(random.uniform(0.25, 2), random.uniform(2.0, 3.0)))
    intensity.append(random.uniform(1, 2)) 

fields = []
for coord, radius, nrg, intenseVal in zip(coords, radii, energy, intensity):
    fields.append(Field(position=coord, radius=radius, energy=nrg, intensity_dt=intenseVal))

# Create and run the animation
animation = Animation(size, moving_object, fields, time_step=0.1, total_time=10)
animation.show()
