# Virtual RCM Controller for Surgical Robots

## 1. Overview

This method implements a **Virtual Remote Center of Motion (V-RCM)** controller for a surgical instrument or endoscope mounted on a general-purpose robot such as a UR5e.

The key idea is:

> The instrument shaft must continuously pass through a fixed RCM/trocar point, while allowing rotation about the RCM and insertion/retraction along the instrument axis.

The controller is implemented as a task-space velocity controller with a constrained quadratic program (QP). It can naturally incorporate:

- RCM constraints
- viewpoint/orientation control
- insertion control
- joint limits
- joint-velocity limits
- collision avoidance

For an autonomous endoscope system, the upper-level planner only needs to output a desired viewpoint change and insertion. The V-RCM controller converts this into robot joint velocities.

---

## 2. Robot and Instrument Model

Let the robot have joint configuration

\[
q \in \mathbb{R}^n,
\]

with end-effector pose

\[
T_e(q) =
\begin{bmatrix}
R_e & p_e\\
0 & 1
\end{bmatrix}.
\]

The instrument axis is defined by the end-effector \(z\)-axis:

\[
d = R_e e_z,
\]

where

\[
e_z = [0,0,1]^T.
\]

The end-effector spatial velocity is

\[
\begin{bmatrix}
v_e\\
\omega_e
\end{bmatrix}
=
J(q)\dot q
=
\begin{bmatrix}
J_v\\
J_\omega
\end{bmatrix}\dot q.
\]

Here:

- \(p_e\): reference point on the instrument/end-effector
- \(d\): instrument shaft direction
- \(v_e\): translational velocity
- \(\omega_e\): angular velocity
- \(J_v,J_\omega\): translational and rotational Jacobians

---

## 3. RCM Constraint

Let

\[
p_r
\]

be the fixed RCM/trocar point in the robot/world frame.

The physical requirement is **not** that \(p_e\) remains fixed.

Instead:

\[
\boxed{
p_r \text{ must remain on the instrument axis}
}
\]

or equivalently,

\[
(p_e-p_r)\times d=0.
\]

Define

\[
r=p_e-p_r.
\]

Ideally,

\[
r=ld,
\]

where \(l\) is the signed distance from the RCM to the reference point.

Therefore, insertion is allowed because \(l\) can change:

\[
\dot l \neq 0.
\]

The RCM point remains fixed in space while the instrument slides through it.

---

## 4. Velocity-Level RCM Constraint

Because

\[
p_e=p_r+ld,
\]

differentiating gives

\[
v_e=\dot l d+l\dot d.
\]

The instrument direction satisfies

\[
\dot d=\omega_e\times d.
\]

Therefore,

\[
v_e=\dot l d+l(\omega_e\times d).
\]

The component perpendicular to the instrument axis must satisfy

\[
(I-dd^T)
\left[
v_e-l(\omega_e\times d)
\right]=0.
\]

Using

\[
v_e=J_v\dot q,
\qquad
\omega_e=J_\omega\dot q,
\]

we obtain

\[
\boxed{
J_{RCM}\dot q=0
}
\]

with

\[
J_{RCM}
=
(I-dd^T)
\left[
J_v+l[d]_\times J_\omega
\right].
\]

Here \([d]_\times\) is the skew-symmetric matrix satisfying

\[
[d]_\times x=d\times x.
\]

Although \(J_{RCM}\) is a \(3\times n\) matrix, the RCM constraint contains only two independent constraints because motion along \(d\) is allowed.

---

## 5. Insertion Control

Insertion corresponds to motion along the instrument axis.

The desired insertion velocity is

\[
v_{ins}.
\]

Therefore,

\[
d^T v_e=v_{ins}.
\]

Using the robot Jacobian:

\[
\boxed{
d^T J_v\dot q=v_{ins}.
}
\]

Examples:

- \(v_{ins}>0\): insertion
- \(v_{ins}<0\): retraction
- \(v_{ins}=0\): no insertion

Thus, the RCM constraint and insertion constraint are compatible:

\[
J_{RCM}\dot q=0,
\]

while simultaneously

\[
d^T J_v\dot q=v_{ins}.
\]

---

## 6. Viewpoint / Orientation Control

For an endoscope, the main viewpoint variable is the instrument viewing direction \(d\).

Let the desired viewing direction be

\[
d_{des}.
\]

A simple orientation error is

\[
e_d=d_{des}\times d.
\]

The desired angular velocity can be defined as

\[
\boxed{
\omega_{des}
=
k_R(d_{des}\times d)
+
\omega_{roll}d.
}
\]

where:

- \(k_R\): orientation feedback gain
- \(\omega_{roll}\): optional roll velocity

If roll is not required,

\[
\omega_{roll}=0.
\]

This gives a simple interface for an upper-level viewpoint planner.

---

## 7. QP-Based Virtual RCM Controller

At each control cycle, solve for the robot joint velocity

\[
\dot q.
\]

A basic QP is

\[
\begin{aligned}
\min_{\dot q}\quad
&
\frac{1}{2}
\left\|
J_\omega\dot q-\omega_{des}
\right\|^2
+
\frac{\lambda}{2}\|\dot q\|^2
\\
\text{s.t.}\quad
&
J_{RCM}\dot q=0,
\\
&
d^TJ_v\dot q=v_{ins},
\\
&
\dot q_{min}\leq\dot q\leq\dot q_{max},
\\
&
q_{min}+\epsilon
\leq
q+\dot q\Delta t
\leq
q_{max}-\epsilon.
\end{aligned}
\]

The first term tracks the desired viewpoint rotation. The second term regularizes joint motion.

The constraints guarantee that the resulting robot motion respects the RCM and insertion requirements.

---

## 8. Practical RCM Error Correction

In a real system, calibration and perception errors mean that

\[
(p_e-p_r)\times d
\]

will not be exactly zero.

Define the instantaneous RCM position error as

\[
e_{RCM}
=
(I-dd^T)(p_e-p_r).
\]

Instead of enforcing

\[
J_{RCM}\dot q=0,
\]

a more robust controller can use

\[
\boxed{
J_{RCM}\dot q
=
-k_{RCM}e_{RCM}.
}
\]

This provides feedback correction toward the desired RCM manifold.

The resulting controller therefore simultaneously:

1. keeps the instrument axis passing through the RCM;
2. corrects accumulated RCM error;
3. tracks the desired viewing direction;
4. performs insertion/retraction.

---

## 9. Collision Avoidance

Collision constraints can be added to the same QP.

Let the distance between the robot/environment and obstacle \(i\) be

\[
d_i(q).
\]

Require

\[
d_i(q)\geq d_{safe}.
\]

Linearizing over one control step:

\[
d_i(q+\dot q\Delta t)
\approx
d_i(q)
+
\nabla d_i^T\dot q\Delta t.
\]

Therefore,

\[
\boxed{
\nabla d_i^T\dot q
\geq
\frac{d_{safe}-d_i(q)}{\Delta t}.
}
\]

The complete QP can then be written as

\[
\begin{aligned}
\min_{\dot q}\quad
&
\|J_\omega\dot q-\omega_{des}\|^2
+
\lambda\|\dot q\|^2
\\
\text{s.t.}\quad
&
J_{RCM}\dot q=-k_{RCM}e_{RCM},
\\
&
d^TJ_v\dot q=v_{ins},
\\
&
\nabla d_i^T\dot q\geq b_i,
\\
&
q_{min}\leq q+\dot q\Delta t\leq q_{max},
\\
&
\dot q_{min}\leq\dot q\leq\dot q_{max}.
\end{aligned}
\]

This provides a unified constrained controller for surgical manipulation.

---

## 10. Interface to Active Endoscopic Viewpoint Planning

For autonomous endoscopic viewpoint adjustment, the upper-level planner does not need to operate in the 6-DoF robot joint space.

Instead, define a compact surgical viewpoint action:

\[
\boxed{
a_t=
[
\Delta\theta_{pitch},
\Delta\theta_{yaw},
\Delta d_{ins}
].
}
\]

For example,

\[
a_t=
[
5^\circ,-3^\circ,8\text{ mm}
]
\]

means:

- pitch the endoscope by \(5^\circ\);
- yaw by \(-3^\circ\);
- insert by \(8\) mm.

The planner converts this action into desired orientation and insertion velocities. The V-RCM controller then computes the corresponding robot joint velocities.

This creates a clean separation:

```text
Active Viewpoint Planner
        |
        | desired viewpoint + insertion
        v
Virtual RCM Controller
        |
        | constrained joint velocity
        v
QP Solver
        |
        v
Robot
        |
        v
Endoscope
```

---

## 11. Stop-and-Go Control Loop

For a coarse-to-fine active perception system, the controller can operate in short motion segments:

```text
RGB-D observation
        |
        v
Target / scene perception
        |
        v
Visibility evaluation
        |
        v
Viewpoint planner
        |
        | Δpitch, Δyaw, Δinsertion
        v
Virtual RCM controller
        |
        v
QP
        |
        v
Execute short motion
        |
        v
New RGB-D observation
        |
        +------> repeat
```

The high-level planner therefore performs discrete **Stop-and-Go** decisions, while the low-level V-RCM controller continuously enforces the physical constraints during each motion segment.

---

## 12. Controller Pseudocode

```python
while robot_running:

    # 1. Robot state
    q = robot.get_joint_position()

    # 2. Forward kinematics
    T = robot.forward_kinematics(q)
    p_e = T.position
    R_e = T.rotation

    # 3. Instrument direction
    d = R_e @ ez

    # 4. Robot Jacobian
    Jv, Jw = robot.jacobian(q)

    # 5. RCM geometry
    l = dot(p_e - p_rcm, d)

    # 6. RCM error
    e_rcm = (I - outer(d, d)) @ (p_e - p_rcm)

    # 7. RCM Jacobian
    J_rcm = (I - outer(d, d)) @ (
        Jv + l * skew(d) @ Jw
    )

    # 8. Desired viewpoint
    d_des = planner.get_desired_direction()
    v_ins = planner.get_insertion_velocity()

    # 9. Desired angular velocity
    omega_des = k_R * cross(d_des, d)

    # 10. Solve constrained QP
    qdot = solve_qp(
        objective=tracking_cost(Jw, omega_des)
                  + regularization(qdot),

        equality=[
            J_rcm @ qdot + k_rcm * e_rcm == 0,
            d.T @ Jv @ qdot == v_ins,
        ],

        inequality=[
            joint_velocity_limits(qdot),
            joint_position_limits(q, qdot),
            collision_constraints(q, qdot),
        ],
    )

    # 11. Execute
    robot.send_joint_velocity(qdot)
```

---

## 13. Recommended System Architecture

For the complete surgical active-perception system:

```text
                    RGB-D Camera
                         |
                         v
              +---------------------+
              | Scene / Target       |
              | Perception           |
              +---------------------+
                         |
                         v
              +---------------------+
              | Visibility / Utility |
              | Evaluation           |
              +---------------------+
                         |
                         v
              +---------------------+
              | Coarse-to-Fine       |
              | Viewpoint Planner    |
              +---------------------+
                         |
              Δpitch, Δyaw, Δinsert
                         |
                         v
              +---------------------+
              | Virtual RCM          |
              | Controller           |
              +---------------------+
                         |
                         v
              +---------------------+
              | Constrained QP       |
              | RCM + Collision      |
              | + Joint Limits       |
              +---------------------+
                         |
                         v
                       UR5e
                         |
                         v
                    Endoscope
```

The key design principle is:

> **The planner reasons in a low-dimensional surgical viewpoint space, while the V-RCM controller handles robot-level kinematic feasibility and safety constraints.**

This separation makes the method suitable for both simulation and physical deployment.
