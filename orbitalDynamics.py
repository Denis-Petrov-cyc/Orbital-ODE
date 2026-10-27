import numpy as np
import matplotlib.pyplot as plt

SunGM = 1.32712440018e11

MercuryGM = 22031.86855
VenusGM = 324858.592
EarthGM = 398600.4418
MarsGM = 42828.375214
JupiterGM = 1.26686534e8
SaturnGM = 3.7931187e7
UranusGM = 5.793939e6
NeptuneGM = 6.836529e6
PlutoGM = 869.61

MoonGM = 4902.800118

bodies = {
    "sun": {
        "GM": SunGM,
        "parent": None,
        "position": np.array([0.0, 0.0, 0.0]),
        "velocity": np.array([0.0, 0.0, 0.0])
    },

    "mercury": {
        "GM": MercuryGM,
        "parent": "sun",
        "position": np.array([57_909_050.0, 0.0, 0.0]),
        "velocity": np.array([0.0, 47.36, 0.0])
    },

    "venus": {
        "GM": VenusGM,
        "parent": "sun",
        "position": np.array([108_208_000.0, 0.0, 0.0]),
        "velocity": np.array([0.0, 35.02, 0.0])
    },

    "earth": {
        "GM": EarthGM,
        "parent": "sun",
        "position": np.array([149_597_870.7, 0.0, 0.0]),
        "velocity": np.array([0.0, 29.78, 0.0])
    },

    "mars": {
        "GM": MarsGM,
        "parent": "sun",
        "position": np.array([227_939_200.0, 0.0, 0.0]),
        "velocity": np.array([0.0, 24.07, 0.0])
    },

    "jupiter": {
        "GM": JupiterGM,
        "parent": "sun",
        "position": np.array([778_570_000.0, 0.0, 0.0]),
        "velocity": np.array([0.0, 13.07, 0.0])
    },

    "saturn": {
        "GM": SaturnGM,
        "parent": "sun",
        "position": np.array([1_433_529_000.0, 0.0, 0.0]),
        "velocity": np.array([0.0, 9.69, 0.0])
    },

    "uranus": {
        "GM": UranusGM,
        "parent": "sun",
        "position": np.array([2_872_463_000.0, 0.0, 0.0]),
        "velocity": np.array([0.0, 6.81, 0.0])
    },

    "neptune": {
        "GM": NeptuneGM,
        "parent": "sun",
        "position": np.array([4_495_060_000.0, 0.0, 0.0]),
        "velocity": np.array([0.0, 5.43, 0.0])
    },

    "pluto": {
        "GM": PlutoGM,
        "parent": "sun",
        "position": np.array([5_906_380_000.0, 0.0, 0.0]),
        "velocity": np.array([0.0, 4.74, 0.0])
    },

    "moon": {
        "GM": MoonGM,
        "parent": "earth",
        "position": np.array([149_982_270.7, 0.0, 0.0]),
        "velocity": np.array([0.0, 30.802, 0.0])
    }
}

def derivatives(state):
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)

    ax = -(EarthGM * x) / (r**3)

    ay = -(EarthGM * y) / (r**3)

    az = -(EarthGM * z) / (r**3)

    return np.array([vx, vy, vz, ax, ay, az])

def nBodyDerivatives(positions, velocities, GMs):
    bodiesNumber = len(GMs)
    accelerations = np.zeros_like(positions)

    for i in range (bodiesNumber):
        for j in range(bodiesNumber):
            if i == j:
                continue
            displacement = positions[j] - positions[i]
            distance = np.linalg.norm(displacement)
            accelerations[i] += (
                GMs[j] * displacement / distance ** 3
            )
    return velocities, accelerations


def eulerStep(state, dt):
    change = derivatives(state) * dt
    newState = state + change
    return newState


def rk4Step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + (dt / 2) * k1)
    k3 = derivatives(state + (dt / 2) * k2)
    k4 = derivatives(state + dt * k3)
    newState = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return newState


def nBodyRK4Step(positions, velocities, GMs, dt):
    k1Position, k1Velocity = nBodyDerivatives(
        positions,
        velocities,
        GMs
    )
    k2Position, k2Velocity = nBodyDerivatives(
        positions + (dt / 2) * k1Position,
        velocities + (dt / 2) * k1Velocity,
        GMs
    )
    k3Position, k3Velocity = nBodyDerivatives(
        positions + (dt / 2) * k2Position,
        velocities + (dt / 2) * k2Velocity,
        GMs
    )
    k4Position, k4Velocity = nBodyDerivatives(
        positions + dt * k3Position,
        velocities + dt * k3Velocity,
        GMs
    )
    newPositions = positions + (dt / 6) * (
        k1Position +
        2 * k2Position +
        2 * k3Position +
        k4Position
    )
    newVelocities = velocities + (dt / 6) * (
        k1Velocity +
        2 * k2Velocity +
        2 * k3Velocity +
        k4Velocity
    )
    return newPositions, newVelocities


def nBodyEnergy(positions, velocities, GMs):
    KE = 0.0
    PE = 0.0

    bodiesNumber = len(GMs)
    for i in range (bodiesNumber):
        speedSquared = np.dot(
            velocities[i],
            velocities[i]
        )
        KE += (
            0.5 * GMs[i] * speedSquared
        )
        for j in range(i + 1, bodiesNumber):
            difference = positions[j] - positions[i]
            distance = np.linalg.norm(difference)

            PE -= (
                GMs[i] * GMs[j] / distance
            )
    return KE + PE


def simulate(initialState, dt, totalTime, stepMethod):
    numSteps = int(totalTime / dt)

    state = initialState.copy()
    history = [state.copy()]

    for i in range(numSteps):
        state = stepMethod(state, dt)
        history.append(state.copy())

    return np.array(history)


def nBodySimulate(positions, velocities, GMs, dt, totalTime):
    numberOfSteps = int(totalTime / dt)

    positionHistory = [positions.copy()]
    velocityHistory = [velocities.copy()]

    for i in range(numberOfSteps):

        positions, velocities = nBodyRK4Step(
            positions,
            velocities,
            GMs,
            dt
        )

        positionHistory.append(positions.copy())
        velocityHistory.append(velocities.copy())

    return (
        np.array(positionHistory),
        np.array(velocityHistory)
    )


def orbitalEnergy(state):
    x, y, z, vx, vy, vz = state

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    vSquared = vx ** 2 + vy ** 2 + vz ** 2

    energy = (vSquared / 2) - (EarthGM / r)

    return energy


initialState = np.array([
    7000.0, 
    0.0,
    0.0,
    0.0,
    6.535,
    3.773
])

dt = 10
totalTime = 6000

eulerHistory = simulate(initialState, dt, totalTime, eulerStep)
rk4History = simulate(initialState, dt, totalTime, rk4Step)

eulerEnergy = np.array([
    orbitalEnergy(state) for state in eulerHistory
    ])

rk4Energy = np.array([
    orbitalEnergy(state) for state in rk4History
])

eulerEnergyError = np.abs(
    (eulerEnergy - eulerEnergy[0]) / eulerEnergy[0]
)

rk4EnergyError = np.abs(
    (rk4Energy - rk4Energy[0]) / rk4Energy[0]
)

eulerX = eulerHistory[:, 0]
eulerY = eulerHistory[:, 1]

rk4X = rk4History[:, 0]
rk4Y = rk4History[:, 1]

plt.plot(eulerX, eulerY, label="Euler")
plt.plot(rk4X, rk4Y, label="RK4")

plt.scatter(0, 0, label="Earth")

plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.title("Euler vs RK4 Orbital Simulation")
plt.axis("equal")
plt.legend()

plt.show()


timeEuler = np.arange(len(eulerEnergy)) * dt
timerRk4 = np.arange(len(rk4Energy)) * dt

plt.plot(timeEuler, eulerEnergy, label = "Euler")
plt.plot(timerRk4, rk4Energy, label = "RK4")

plt.xlabel("Time (seconds)")
plt.ylabel("Specific Orbital Energy (km^2/s^2)")
plt.title("Orbital Energy Conservation")
plt.legend()

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection ="3d")

rk4x = rk4History[:, 0]
rk4y = rk4History[:, 1]
rk4z = rk4History[:, 2]

ax.plot(rk4x, rk4y, rk4z, label = "RK$ Orbit")
ax.scatter(0, 0, 0, label = "Earth")

ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.set_zlabel("z (km)")

ax.set_title("3D Orbital Simulation")
ax.legend()

plt.show()

print("Euler final relative energy error:", eulerEnergyError[-1])
print("RK4 final relative energy error:", rk4EnergyError[-1])


userInput = input("Enter bodies to simulate, seperated by commas: ")

UsersChosen = [
    body.strip().lower()
    for body in userInput.split(",")
    ]

chosenGM = []
chosenPositions = []
chosenVelocities = []
validBodies = []

for body in UsersChosen:
    if body in bodies:
        validBodies.append(body)
        chosenGM.append(bodies[body]["GM"])
        chosenPositions.append(bodies[body]["position"])
        chosenVelocities.append(bodies[body]["velocity"])
        print(body, bodies[body]["GM"])
    else:
        print("Unknown body:", body)

chosenGM = np.array(chosenGM)
chosenPositions = np.array(chosenPositions)
chosenVelocities = np.array(chosenVelocities)
if len(chosenGM) == 0:
    print("No bodies selected.")
    quit()


velocityDerivatives, accelerations = nBodyDerivatives(
    chosenPositions,
    chosenVelocities,
    chosenGM
)

print("Accelerations:")
print(accelerations)

nBodyDt = 600
nBodyTotalTime = 365.25 * 24 * 3600

positionHistory, velocityHistory = nBodySimulate(
    chosenPositions,
    chosenVelocities,
    chosenGM,
    nBodyDt,
    nBodyTotalTime
)

nBodyEnergyHistory = []

for i in range(len(positionHistory)):
    energy = nBodyEnergy(
        positionHistory[i],
        velocityHistory[i],
        chosenGM
    )
    nBodyEnergyHistory.append(energy)
nBodyEnergyHistory = np.array(nBodyEnergyHistory)


nBodyEnergyError = np.abs(
    (nBodyEnergyHistory - nBodyEnergyHistory[0])
    / nBodyEnergyHistory[0]
)
print(
    "N-body final relative energy error:",
    nBodyEnergyError[-1]
)
print(
    "N-body maximum relative energy error:",
    np.max(nBodyEnergyError)
)


print("Starting N-body energy:", nBodyEnergyHistory[0])
print("Ending N-body energy:", nBodyEnergyHistory[-1])
print("Minimum N-body energy:", np.min(nBodyEnergyHistory))
print("Maximum N-body energy:", np.max(nBodyEnergyHistory))


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for i, body in enumerate(validBodies):
    ax.plot(
        positionHistory[:, i, 0],
        positionHistory[:, i, 1],
        positionHistory[:, i, 2],
        label=body.capitalize()
    )

ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.set_zlabel("z (km)")
ax.set_title("3D N-Body Orbital Simulation")
ax.legend()

plt.show()

if "earth" in validBodies and "moon" in validBodies:
    earthIndex = validBodies.index("earth")
    moonIndex = validBodies.index("moon")

    earthPosition = positionHistory[:, earthIndex, :]
    moonPosition = positionHistory[:, moonIndex, :]
    moonToEarth = moonPosition - earthPosition

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    ax.plot(
        moonToEarth[:, 0],
        moonToEarth[:, 1],
        moonToEarth[:, 2],
        label = "Moon"
    )

    ax.scatter(
        0,
        0,
        0,
        label = "Earth"
    )

    ax.set_xlabel("x relative to Earth (km)")
    ax.set_ylabel("y relative to Earth (km)")
    ax.set_zlabel("z relative to Earth (km)")
    ax.set_title("Moon Orbit Relative to Earth")
    ax.legend()

    plt.show()


if "earth" in validBodies and "sun" in validBodies:
    earthIndex = validBodies.index("earth")
    sunIndex = validBodies.index("sun")
    earthPosition = positionHistory[:, earthIndex, :]
    sunPosition = positionHistory[:, sunIndex, :]
    sunToEarth = earthPosition - sunPosition
    EDistance = np.linalg.norm(
        sunToEarth,
        axis = 1
    )

    print("Starting Earth-Sun distance:", EDistance[0], "km")
    print("Ending Earth-Sun distance:", EDistance[-1], "km")
    print("Minimum Earth-Sun distance:", np.min(EDistance), "km")
    print("Maximum Earth-Sun distance:", np.max(EDistance), "km")