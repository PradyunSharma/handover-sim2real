#!/usr/bin/env python3
import time
import copy
import math
import roslibpy

ROSBRIDGE_HOST = "172.16.0.7"
ROSBRIDGE_PORT = 9090

CURRENT_POSE_TOPIC = "/cartesian_pose"
TARGET_POSE_TOPIC = "/equilibrium_pose"
POSE_MSG_TYPE = "geometry_msgs/PoseStamped"

# Small safe translation offsets in meters
DX = 0.02
DY = 0.00
DZ = 0.00

# Small safe relative rotation offsets in degrees
# Applied in the current end-effector frame
DROLL_DEG  = 0.0   # around local X
DPITCH_DEG = 0.0   # around local Y
DYAW_DEG   = 0.0   # around local Z

current_msg = None

def pose_cb(msg):
    global current_msg
    current_msg = msg

def quat_normalize(q):
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0:
        raise ValueError("Zero-norm quaternion")
    return [x / n, y / n, z / n, w / n]

def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return [x, y, z, w]

def quat_from_axis_angle(ax, ay, az, angle_rad):
    half = angle_rad / 2.0
    s = math.sin(half)
    return quat_normalize([ax * s, ay * s, az * s, math.cos(half)])

client = roslibpy.Ros(host=ROSBRIDGE_HOST, port=ROSBRIDGE_PORT)
client.run()

for _ in range(50):
    if client.is_connected:
        break
    time.sleep(0.1)

if not client.is_connected:
    raise RuntimeError(f"Could not connect to rosbridge at {ROSBRIDGE_HOST}:{ROSBRIDGE_PORT}")

sub = roslibpy.Topic(client, CURRENT_POSE_TOPIC, POSE_MSG_TYPE)
pub = roslibpy.Topic(client, TARGET_POSE_TOPIC, POSE_MSG_TYPE)

sub.subscribe(pose_cb)
pub.advertise()

print(f"Connected to rosbridge at {ROSBRIDGE_HOST}:{ROSBRIDGE_PORT}")
print(f"Subscribed to {CURRENT_POSE_TOPIC}")
print(f"Publishing to {TARGET_POSE_TOPIC}")
print("Waiting for current pose...")

timeout_s = 10.0
t0 = time.time()
while current_msg is None and (time.time() - t0) < timeout_s:
    time.sleep(0.05)

if current_msg is None:
    sub.unsubscribe()
    pub.unadvertise()
    client.terminate()
    raise RuntimeError(f"No message received on {CURRENT_POSE_TOPIC} through rosbridge")

target = copy.deepcopy(current_msg)

# Translation
target["pose"]["position"]["x"] += DX
target["pose"]["position"]["y"] += DY
target["pose"]["position"]["z"] += DZ

# Current orientation
q_current = [
    current_msg["pose"]["orientation"]["x"],
    current_msg["pose"]["orientation"]["y"],
    current_msg["pose"]["orientation"]["z"],
    current_msg["pose"]["orientation"]["w"],
]
q_current = quat_normalize(q_current)

# Relative rotation quaternions
qx = quat_from_axis_angle(1.0, 0.0, 0.0, math.radians(DROLL_DEG))
qy = quat_from_axis_angle(0.0, 1.0, 0.0, math.radians(DPITCH_DEG))
qz = quat_from_axis_angle(0.0, 0.0, 1.0, math.radians(DYAW_DEG))

# Apply relative rotation in local tool frame
q_target = quat_multiply(q_current, qx)
q_target = quat_multiply(q_target, qy)
q_target = quat_multiply(q_target, qz)
q_target = quat_normalize(q_target)

target["pose"]["orientation"]["x"] = q_target[0]
target["pose"]["orientation"]["y"] = q_target[1]
target["pose"]["orientation"]["z"] = q_target[2]
target["pose"]["orientation"]["w"] = q_target[3]

print("Current pose:")
print(current_msg["pose"])
print("Target pose:")
print(target["pose"])

# Publish target repeatedly for 2 seconds at 10 Hz
for i in range(20):
    now = time.time()
    secs = int(now)
    nsecs = int((now - secs) * 1e9)

    target["header"]["seq"] = i
    target["header"]["stamp"] = {"secs": secs, "nsecs": nsecs}

    pub.publish(roslibpy.Message(target))
    time.sleep(0.1)

print("Target sent.")

sub.unsubscribe()
pub.unadvertise()
client.terminate()
print("Done.")