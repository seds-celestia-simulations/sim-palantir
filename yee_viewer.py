"""Interactive 3D viewer for the Yee-FDTD physics module."""

import cupy as cp
import numpy as np
import pygame
import time
from ctypes import c_void_p
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective, gluProject, gluUnProject
from OpenGL.GL.shaders import compileProgram, compileShader
import random

import yee_physics as sim
print("E type:", type(sim.E))
print("E device:", sim.E.device)

# Window and camera settings.
WIDTH, HEIGHT = 1000, 800
camera_distance = 200.0
rotation_x, rotation_y = 25.0, 35.0
paused = False
step_number = 0
FIELD_OPACITY = 0.009
OPACITY_MIN = 0.001
OPACITY_MAX = 0.05
display_peak = 1e-12


# Program state for user input (when placing current sources).
input_state = {
    "stage": 0,
    "shape": None,
    "behaviour": None,
    "axis": None,
    "position": [sim.N // 2, sim.N // 2, sim.N // 2]
}

# Grid coordinates centered on the simulation's source.
coordinates = np.indices((sim.N, sim.N, sim.N), dtype=np.float32)
coordinates = coordinates.reshape(3, -1).T
coordinates -= sim.N / 2

# Keep a copy of the coordinate array on the GPU.
gpu_coordinates = cp.asarray(coordinates, dtype=cp.float32)
shader_program = None
position_vbo = None
value_vbo = None

# OpenGL buffer, initialized after the window is created.
field_vbo = None

def draw_preview_source():
    if input_state["stage"] != 3:
        return

    glDisable(GL_DEPTH_TEST)
    shape = input_state["shape"]
    axis = input_state["axis"]
    px, py, pz = input_state["position"]
    ox, oy, oz = px - sim.N / 2, py - sim.N / 2, pz - sim.N / 2

    color = (1.0, 0.4, 0.0)  # Orange indicator
    if shape == "Wi-Fi Router":
        render_router_geometry(ox, oy, oz, axis, color)
    elif shape == "Phone":
        render_phone_geometry(ox, oy, oz, axis, color)

    glLineWidth(1.0)
    glEnable(GL_DEPTH_TEST)

def render_router_geometry(ox, oy, oz, axis, color):
    glColor3fv(color)
    
    # 1. Router Body (rectangular slab)
    glLineWidth(2.0)
    
    # Orient the router body based on the antenna axis
    if axis == 0:
        bw, bh, bd = 6.0, 2.5, 4.0
    elif axis == 1:
        bw, bh, bd = 2.5, 6.0, 4.0
    else: # axis == 2
        bw, bh, bd = 4.0, 2.5, 6.0

    corners = [
        (ox - bw, oy - bh, oz - bd), (ox + bw, oy - bh, oz - bd),
        (ox + bw, oy + bh, oz - bd), (ox - bw, oy + bh, oz - bd),
        (ox - bw, oy - bh, oz + bd), (ox + bw, oy - bh, oz + bd),
        (ox + bw, oy + bh, oz + bd), (ox - bw, oy + bh, oz + bd),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    
    glBegin(GL_LINES)
    for a, b in edges:
        glVertex3f(*corners[a])
        glVertex3f(*corners[b])
    glEnd()

    # 2. Antennas (Two dipoles at distance)
    glLineWidth(3.0)
    antenna_spacing = 4.0
    half_len = 5.0
    glBegin(GL_LINES)
    for offset in [-antenna_spacing, antenna_spacing]:
        if axis == 0:
            glVertex3f(ox - half_len, oy + offset, oz)
            glVertex3f(ox + half_len, oy + offset, oz)
        elif axis == 1:
            glVertex3f(ox + offset, oy - half_len, oz)
            glVertex3f(ox + offset, oy + half_len, oz)
        elif axis == 2:
            glVertex3f(ox + offset, oy, oz - half_len)
            glVertex3f(ox + offset, oy, oz + half_len)
    glEnd()


def render_phone_geometry(ox, oy, oz, axis, color):
    glColor3fv(color)
    
    # Phone Body (handset outline)
    pw, ph = 2.5, 5.0
    glLineWidth(2.0)
    glBegin(GL_LINE_LOOP)
    if axis == 0:
        glVertex3f(ox, oy - ph, oz - pw)
        glVertex3f(ox, oy + ph, oz - pw)
        glVertex3f(ox, oy + ph, oz + pw)
        glVertex3f(ox, oy - ph, oz + pw)
    elif axis == 1:
        glVertex3f(ox - pw, oy, oz - ph)
        glVertex3f(ox + pw, oy, oz - ph)
        glVertex3f(ox + pw, oy, oz + ph)
        glVertex3f(ox - pw, oy, oz + ph)
    elif axis == 2:
        glVertex3f(ox - pw, oy - ph, oz)
        glVertex3f(ox + pw, oy - ph, oz)
        glVertex3f(ox + pw, oy + ph, oz)
        glVertex3f(ox - pw, oy + ph, oz)
    glEnd()

    # Compact Internal Loop Antenna
    radius = 1.2
    segments = 16
    glLineWidth(3.0)
    glBegin(GL_LINE_LOOP)
    for i in range(segments):
        theta = 2.0 * np.pi * i / segments
        c, s = radius * np.cos(theta), radius * np.sin(theta)
        if axis == 0:
            glVertex3f(ox, oy + c, oz + s)
        elif axis == 1:
            glVertex3f(ox + c, oy, oz + s)
        elif axis == 2:
            glVertex3f(ox + c, oy + s, oz)
    glEnd()


def get_raycast_grid_coordinate(mouse_x, mouse_y):
    """Cast a ray from the mouse position and find its intersection with the simulation grid's Z=0 plane."""
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)
    
    # OpenGL's Y axis starts from the bottom, Pygame's starts from the top.
    opengl_y = viewport[3] - mouse_y
    
    # Get the 3D points at the near and far clipping planes
    near_point = gluUnProject(mouse_x, opengl_y, 0.0, modelview, projection, viewport)
    far_point = gluUnProject(mouse_x, opengl_y, 1.0, modelview, projection, viewport)
    
    # Calculate the ray direction vector
    direction = np.array(far_point) - np.array(near_point)
    direction /= np.linalg.norm(direction)
    
    # For a 3D grid, calculate intersection with the local Z=0 plane (center of the cube).
    if direction[2] != 0:
        t = -near_point[2] / direction[2]
        intersection = np.array(near_point) + t * direction
        
        # Offset back to grid indices (0 to N)
        grid_x = int(round(intersection[0] + sim.N / 2))
        grid_y = int(round(intersection[1] + sim.N / 2))
        grid_z = int(sim.N / 2) # Intersected the Z=0 slice
        
        # Clamp to bounds
        grid_x = max(0, min(sim.N - 1, grid_x))
        grid_y = max(0, min(sim.N - 1, grid_y))
        
        return (grid_x, grid_y, grid_z)
    
    return (sim.N // 2, sim.N // 2, sim.N // 2)


def draw_box(size):
    """Draw a wireframe cube showing the simulation domain."""
    h = size / 2
    corners = [
        (-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
        (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    glColor3f(0.35, 0.38, 0.43)
    glBegin(GL_LINES)
    for a, b in edges:
        glVertex3fv(corners[a])
        glVertex3fv(corners[b])
    glEnd()


def draw_axes(size):
    """Draw X, Y, and Z axes along three edges from one cube corner."""
    h = size / 2
    corner = (-h, -h, -h)
    axes = [
        (corner, (h, -h, -h), (1.0, 0.25, 0.25)),  # X: red
        (corner, (-h, h, -h), (0.25, 1.0, 0.35)),  # Y: green
        (corner, (-h, -h, h), (0.3, 0.55, 1.0)),   # Z: blue
    ]

    # Draw axes on top of the field so the coordinate frame stays visible.
    glDisable(GL_DEPTH_TEST)
    glLineWidth(3.0)
    glBegin(GL_LINES)
    for start, end, color in axes:
        glColor3fv(color)
        glVertex3fv(start)
        glVertex3fv(end)
    glEnd()

    # Arrowheads point toward the positive end of each axis.
    arrow = 2.2
    wing = 1.2
    arrowheads = [
        ((h - arrow, -h + wing, -h), (h, -h, -h)),
        ((h - arrow, -h, -h + wing), (h, -h, -h)),
        ((-h + wing, h - arrow, -h), (-h, h, -h)),
        ((-h, h - arrow, -h + wing), (-h, h, -h)),
        ((-h + wing, -h, h - arrow), (-h, -h, h)),
        ((-h, -h + wing, h - arrow), (-h, -h, h)),
    ]
    glBegin(GL_LINES)
    for i, (start, end) in enumerate(arrowheads):
        glColor3fv(axes[i // 2][2])
        glVertex3fv(start)
        glVertex3fv(end)
    glEnd()
    glLineWidth(1.0)
    glEnable(GL_DEPTH_TEST)



def draw_room_tiles(size, tile_size=10):
    """Draw six tiled faces with camera-dependent transparency."""
    h = size / 2.0
    count = max(1, int(np.ceil(size / tile_size)))
    tile = size / count

    surface_offset = 0.05
    line_offset = 0.08

    # Direction from the cube centre toward the camera,
    # expressed in the cube's local coordinates.
    rx = np.radians(rotation_x)
    ry = np.radians(rotation_y)

    camera_dir = np.array([
        -np.sin(ry) * np.cos(rx),
        np.sin(rx),
        np.cos(ry) * np.cos(rx)
    ])

    # (outward normal, U direction, V direction, base colour)
    faces = [
        # Floor and ceiling
        ((0, -1, 0), (1, 0, 0), (0, 0, 1), (0.16, 0.18, 0.21)),
        ((0,  1, 0), (1, 0, 0), (0, 0, 1), (0.16, 0.18, 0.21)),

        # Back and front walls
        ((0, 0, -1), (1, 0, 0), (0, 1, 0), (0.12, 0.14, 0.18)),
        ((0, 0,  1), (1, 0, 0), (0, 1, 0), (0.12, 0.14, 0.18)),

        # Right and left walls
        (( 1, 0, 0), (0, 1, 0), (0, 0, 1), (0.14, 0.16, 0.20)),
        ((-1, 0, 0), (0, 1, 0), (0, 0, 1), (0.14, 0.16, 0.20))
    ]

    def facing(face):
        return float(np.dot(face[0], camera_dir))

    def vertex(normal, u_axis, v_axis, u, v, inset):
        return tuple(
            normal[k] * (h - inset)
            + u_axis[k] * u
            + v_axis[k] * v
            for k in range(3)
        )

    # Draw the opaque faces first, then the transparent faces.
    # Transparent faces are drawn back-to-front.
    opaque_faces = [f for f in faces if facing(f) <= 0]
    transparent_faces = sorted(
        [f for f in faces if facing(f) > 0],
        key=facing
    )

    for face in opaque_faces + transparent_faces:
        normal, u_axis, v_axis, base_color = face

        # A face becomes transparent as it turns toward the camera.
        visibility = max(0.0, facing(face))
        alpha = 1.0 - 0.85 * min(1.0, 3.0 * visibility)

        if alpha < 1.0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(GL_FALSE)
        else:
            glDisable(GL_BLEND)
            glDepthMask(GL_TRUE)

        # Draw tiled surfaces.
        glBegin(GL_QUADS)

        for i in range(count):
            for j in range(count):
                a = -h + i * tile
                b = -h + j * tile

                shade = 0.015 if (i + j) % 2 else 0.0

                glColor4f(
                    base_color[0] + shade,
                    base_color[1] + shade,
                    base_color[2] + shade,
                    alpha
                )

                for u, v in [
                    (a, b),
                    (a + tile, b),
                    (a + tile, b + tile),
                    (a, b + tile)
                ]:
                    glVertex3fv(
                        vertex(normal, u_axis, v_axis,
                               u, v, surface_offset)
                    )

        glEnd()

        # Draw grout lines, respecting face transparency.
        glColor4f(0.28, 0.30, 0.34, alpha)
        glLineWidth(1.0)
        glBegin(GL_LINES)

        for i in range(count + 1):
            p = -h + i * tile

            glVertex3fv(vertex(normal, u_axis, v_axis,
                                p, -h, line_offset))
            glVertex3fv(vertex(normal, u_axis, v_axis,
                                p, h, line_offset))

            glVertex3fv(vertex(normal, u_axis, v_axis,
                                -h, p, line_offset))
            glVertex3fv(vertex(normal, u_axis, v_axis,
                                h, p, line_offset))

        glEnd()

    # Restore OpenGL state for the rest of the viewer.
    glDepthMask(GL_TRUE)
    glDisable(GL_BLEND)



VERTEX_SHADER = """
#version 120
attribute vec3 a_position;
attribute float a_field;
uniform float u_peak;
uniform float u_threshold;
varying vec3 v_color;
varying float v_strength;

void main()
{
    
    float raw_strength = abs(a_field) / max(u_peak, 1e-30);
    // Gamma correction: make weaker fields much more visible.
    float strength = pow(clamp(raw_strength, 0.0, 1.0), 0.40);
    v_strength = strength;


    // Hide weak field points without CPU filtering.
    if (raw_strength <= u_threshold) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    } else {
        gl_Position = gl_ModelViewProjectionMatrix * vec4(a_position, 1.0);
    }

    
    // Thermal heatmap based on normalized E-field magnitude.
    float s = v_strength;

    if (s < 0.15) {
        // Black to dark purple
        v_color = mix(
            vec3(0.0, 0.0, 0.0),
            vec3(0.25, 0.0, 0.45),
            s / 0.15
        );
    }
    else if (s < 0.35) {
        // Purple to red
        v_color = mix(
            vec3(0.25, 0.0, 0.45),
            vec3(0.8, 0.0, 0.1),
            (s - 0.15) / 0.20
        );
    }
    else if (s < 0.60) {
        // Red to orange
        v_color = mix(
            vec3(0.8, 0.0, 0.1),
            vec3(1.0, 0.45, 0.0),
            (s - 0.35) / 0.25
        );
    }
    else if (s < 0.80) {
        // Orange to yellow
        v_color = mix(
            vec3(1.0, 0.45, 0.0),
            vec3(1.0, 0.9, 0.0),
            (s - 0.60) / 0.20
        );
    }
    else {
        // Yellow to white
        v_color = mix(
            vec3(1.0, 0.9, 0.0),
            vec3(1.0, 1.0, 1.0),
            (s - 0.80) / 0.20
        );
    }

}
"""

FRAGMENT_SHADER = """
#version 120
varying vec3 v_color;
varying float v_strength;
uniform float u_opacity;

void main()
{
    float alpha = min(0.5, 3.0 * u_opacity * sqrt(v_strength));
    gl_FragColor = vec4(v_color, alpha);
}
"""

def draw_field():
    # Calculate field peak on CUDA.
    global display_peak

    field_gpu = cp.sqrt(sim.E[0]**2 + sim.E[1]**2 + sim.E[2]**2)
    peak = float(cp.max(cp.abs(field_gpu)).item())

    if peak < 1e-12:
        return 0

    # Transfer ONLY field values, as float32.
    field_cpu = np.ascontiguousarray(
        cp.asnumpy(field_gpu.astype(cp.float32)).ravel()
    )

    # Update the existing field buffer.
    glBindBuffer(GL_ARRAY_BUFFER, value_vbo)
    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        field_cpu.nbytes,
        field_cpu
    )

    glUseProgram(shader_program)

    glUniform1f(
        glGetUniformLocation(shader_program, "u_peak"),
        peak  # Modified: Pass instantaneous peak instead of display_peak
    )
    glUniform1f(
        glGetUniformLocation(shader_program, "u_threshold"),
        0.001
    )
    glUniform1f(
        glGetUniformLocation(shader_program, "u_opacity"),
        FIELD_OPACITY
    )

    # Static positions.
    position_location = glGetAttribLocation(
        shader_program, "a_position"
    )
    glBindBuffer(GL_ARRAY_BUFFER, position_vbo)
    glEnableVertexAttribArray(position_location)
    glVertexAttribPointer(
        position_location, 3, GL_FLOAT,
        GL_FALSE, 0, c_void_p(0)
    )

    # Dynamic Ey values.
    field_location = glGetAttribLocation(
        shader_program, "a_field"
    )
    glBindBuffer(GL_ARRAY_BUFFER, value_vbo)
    glEnableVertexAttribArray(field_location)
    glVertexAttribPointer(
        field_location, 1, GL_FLOAT,
        GL_FALSE, 0, c_void_p(0)
    )

    # Blend contributions from all field points.
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE)

    # Allow points inside the volume to remain visible.
    glDisable(GL_DEPTH_TEST)
    glDepthMask(GL_FALSE)

    glPointSize(4.0)
    glDrawArrays(GL_POINTS, 0, sim.N**3)

    # Restore OpenGL state for the other objects.
    glDepthMask(GL_TRUE)
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_BLEND)

    glDisableVertexAttribArray(position_location)
    glDisableVertexAttribArray(field_location)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glUseProgram(0)

    return sim.N**3


def draw_source():
    if not sim.current_sources:
        return

    glDisable(GL_DEPTH_TEST)
    for src in sim.current_sources:
        shape = src.get("shape_name")
        axis = src.get("axis")
        px, py, pz = src.get("position")
        ox, oy, oz = px - sim.N / 2, py - sim.N / 2, pz - sim.N / 2

        color = (1.0, 0.85, 0.1)  # Yellow
        if shape == "Wi-Fi Router":
            render_router_geometry(ox, oy, oz, axis, color)
        elif shape == "Phone":
            render_phone_geometry(ox, oy, oz, axis, color)

    glLineWidth(1.0)
    glEnable(GL_DEPTH_TEST)

def draw_text(font, x, y, text, color=(235, 238, 245)):
    """Draw transparent Pygame text at screen coordinates (bottom-left origin)."""
    surface = font.render(text, True, color)
    pixels = pygame.image.tobytes(surface, "RGBA", True)

    # Blend the transparent background away; without this, text can appear as bars.
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glRasterPos2f(x, y)
    glDrawPixels(surface.get_width(), surface.get_height(),
                 GL_RGBA, GL_UNSIGNED_BYTE, pixels)
    glDisable(GL_BLEND)


def draw_overlay(small_font, large_font, axis_labels):
    """Draw axis labels and a small legend over the 3D scene."""
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WIDTH, 0, HEIGHT, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)

    # Labels follow the positive ends of the rotated 3D axes.
    for x, y, label, color in axis_labels:
        draw_text(small_font, x, y, label, color)

    draw_text(small_font, 15, 68, "Yellow points: current-source region", (255, 220, 70))
    draw_text(small_font, 15, 43, "Electric field magnitude: black (weak) to white (strong)")

    # --- NEW CONTROLS PANE ---
    pane_width = 280
    pane_height = 220
    pane_x = WIDTH - pane_width - 20
    pane_y_top = HEIGHT - 20
    pane_y_bottom = pane_y_top - pane_height

    # Draw semi-transparent background for the pane
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0.1, 0.1, 0.15, 0.8)
    glBegin(GL_QUADS)
    glVertex2f(pane_x, pane_y_bottom)
    glVertex2f(pane_x + pane_width, pane_y_bottom)
    glVertex2f(pane_x + pane_width, pane_y_top)
    glVertex2f(pane_x, pane_y_top)
    glEnd()
    glDisable(GL_BLEND)

    # Control layout
    controls = [
        ("Rotate", "Click and drag"),
        ("Zoom", "Scroll"),
        ("Change opacity", "- / +"),
        ("Pause simulation", "Space"),
        ("Reset simulation", "C"),
        ("Quit", "Esc")
    ]
    
    start_y = pane_y_top - 30
    for i, (action, key) in enumerate(controls):
        item_y = start_y - i * 30 - 19

        draw_text(small_font, pane_x + 20, item_y, f"{key}: {action}")
    
    param_pane_width = 400
    param_pane_height = 240
    param_pane_x = WIDTH - param_pane_width - 20
    param_pane_y_bottom = 20
    param_pane_y_top = param_pane_y_bottom + param_pane_height

    # Draw semi-transparent background for the parameters pane
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0.1, 0.1, 0.15, 0.8)
    glBegin(GL_QUADS)
    glVertex2f(param_pane_x, param_pane_y_bottom)
    glVertex2f(param_pane_x + param_pane_width, param_pane_y_bottom)
    glVertex2f(param_pane_x + param_pane_width, param_pane_y_top)
    glVertex2f(param_pane_x, param_pane_y_top)
    glEnd()
    glDisable(GL_BLEND)

    # Parameters layout using Unicode symbols (ε, μ, Δ)
    parameters = [
        f"Grid Size: {sim.N}³ cells",
        f"Spatial Step (Δx): {sim.dx} m",
        f"Time Step (Δt): {sim.dt:.3e} s",
        f"Permittivity (ε): {sim.eps_fixed:.3e} F/m",
        f"Permeability (μ): {sim.mu_fixed:.3e} H/m",
        f"Opacity: {FIELD_OPACITY}"
    ]

    param_start_y = param_pane_y_top - 30
    draw_text(small_font, param_pane_x + 20, param_start_y - 40, "Simulation Parameters:\n\n", (255, 230, 80))
    for i, text in enumerate(parameters):
        item_y = param_start_y - (i + 1) * 25 - 40
        draw_text(small_font, param_pane_x + 20, item_y, text)

    prompt_color = (255, 230, 80)
    prompt_y = HEIGHT - 60  # Moved up above the simulation cube
    
    prompt_text = None
    if input_state["stage"] == 0:
        prompt_text = "Press 'N' to add a new source"
    elif input_state["stage"] == 1:
        prompt_text = "Select source device: [1] Wi-Fi Router   [2] Phone"
    elif input_state["stage"] == 2:
        prompt_text = "Select behaviour: [1] Gaussian Pulse  [2] Sine  [3] Packet Burst"
    elif input_state["stage"] == 3:
        px, py, pz = input_state["position"]
        axis_name = {0: "X", 1: "Y", 2: "Z"}.get(input_state["axis"], "X")
        prompt_text = f"Move: Arrows (X/Y) | Shift+Arrows (Z) | Rotate: R (Axis: {axis_name}) | Pos: ({px}, {py}, {pz}) | [Enter] to place"

    if prompt_text:
        text_surface = large_font.render(prompt_text, True, (255, 255, 255))
        prompt_x = max(0, (WIDTH - text_surface.get_width()) // 2)
        draw_text(large_font, prompt_x, prompt_y, prompt_text, prompt_color)

    glEnable(GL_DEPTH_TEST)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def projected_axis_labels(size):
    """Project X, Y, and Z labels near the positive ends of the edge axes."""
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)
    h = size / 2
    axes = [
        ((h + 1, -h, -h), "X", (255, 90, 90)),
        ((-h, h + 1, -h), "Y", (90, 255, 110)),
        ((-h, -h, h + 1), "Z", (100, 155, 255)),
    ]

    labels = []
    for point, label, color in axes:
        x, y, depth = gluProject(*point, modelview, projection, viewport)
        if 0 <= x < WIDTH and 0 <= y < HEIGHT and 0 <= depth <= 1:
            labels.append((x + 5, y + 5, label, color))
    return labels

def initialize_field_renderer():
    global shader_program, position_vbo, value_vbo

    shader_program = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )

    position_vbo, value_vbo = [
        int(v) for v in glGenBuffers(2)
    ]

    # Coordinates never change: upload them only once.
    positions = np.ascontiguousarray(
        coordinates, dtype=np.float32
    )

    glBindBuffer(GL_ARRAY_BUFFER, position_vbo)
    glBufferData(
        GL_ARRAY_BUFFER,
        positions.nbytes,
        positions,
        GL_STATIC_DRAW
    )

    # Allocate a buffer for one float32 Ey value per cell.
    glBindBuffer(GL_ARRAY_BUFFER, value_vbo)
    glBufferData(
        GL_ARRAY_BUFFER,
        sim.N**3 * 4,
        None,
        GL_STREAM_DRAW
    )

    glBindBuffer(GL_ARRAY_BUFFER, 0)


def get_random_valid_position(padding=10, min_dist=15.0):
    """Generate a random (x,y,z) coordinate avoiding boundaries and existing sources."""
    while True:
        # Constraint 1: Don't cross the simulation boundary
        x = random.randint(padding, sim.N - padding)
        y = random.randint(padding, sim.N - padding)
        z = random.randint(padding, sim.N - padding)
        
        # Constraint 2: Don't overlap with pre-existing sources
        overlap = False
        for src in sim.current_sources:
            sx, sy, sz = src["position"]
            dist = ((x - sx)**2 + (y - sy)**2 + (z - sz)**2)**0.5
            if dist < min_dist:
                overlap = True
                break
                
        if not overlap:
            return (x, y, z)


def main():
    """Run the interactive 3D viewer."""
    global camera_distance, rotation_x, rotation_y, paused, step_number, field_vbo
    global WIDTH, HEIGHT  # Add this to modify the global resolution variables
    global FIELD_OPACITY

    pygame.init()
    pygame.key.set_repeat(200, 30)
    # small_font = pygame.font.Font(None, 23)
    small_font = pygame.font.Font(None, 30)
    large_font = pygame.font.Font(None, 39)
    pygame.display.set_caption("3D Yee FDTD | Ey field and source")
    
    # Retrieve the native screen resolution
    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h
    
    # Enable fullscreen mode with the native resolution
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL | FULLSCREEN)
    
    initialize_field_renderer()
    field_vbo = glGenBuffers(1)
    print("OpenGL GPU:", glGetString(GL_RENDERER).decode())

    glViewport(0, 0, WIDTH, HEIGHT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, WIDTH / HEIGHT, 0.1, 500)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.035, 0.04, 0.055, 1.0)

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_SPACE:
                    paused = not paused
                elif event.key in (K_MINUS, K_KP_MINUS):
                    FIELD_OPACITY = max(OPACITY_MIN, FIELD_OPACITY - 0.002)
                elif event.key in (K_EQUALS, K_PLUS, K_KP_PLUS):
                    FIELD_OPACITY = min(OPACITY_MAX, FIELD_OPACITY + 0.002)
                elif event.key == K_c:
                    # Reset the simulation and viewer states
                    sim.reset()
                    step_number = 0
                    input_state["stage"] = 0
                    input_state["shape"] = None
                    input_state["behaviour"] = None
                    input_state["axis"] = None
                    input_state["position"] = [sim.N // 2, sim.N // 2, sim.N // 2]
                
                elif input_state["stage"] == 0 and event.key == K_n:
                    input_state["stage"] = 1
                
                elif input_state["stage"] == 1:
                    if event.key == K_1:
                        input_state["shape"] = "Wi-Fi Router"
                        input_state["stage"] = 2
                    elif event.key == K_2:
                        input_state["shape"] = "Phone"
                        input_state["stage"] = 2
                
                elif input_state["stage"] == 2:
                    if event.key == K_1:
                        input_state["behaviour"] = "Gaussian Pulse"
                        input_state["stage"] = 3
                    elif event.key == K_2:
                        input_state["behaviour"] = "Sine"
                        input_state["stage"] = 3
                    elif event.key == K_3:
                        input_state["behaviour"] = "Packet Burst"
                        input_state["stage"] = 3

                    if input_state["stage"] == 3:
                        input_state["axis"] = 0
                        input_state["position"] = [sim.N // 2, sim.N // 2, sim.N // 2]
                
                elif input_state["stage"] == 3:
                    keys = pygame.key.get_pressed()
                    shift_pressed = keys[K_LSHIFT] or keys[K_RSHIFT]

                    if event.key == K_r:
                        input_state["axis"] = (input_state["axis"] + 1) % 3
                    elif shift_pressed:
                        if (event.key == K_LEFT or event.key == K_DOWN) and input_state["position"][2] > 10:
                            input_state["position"][2] = max(0, input_state["position"][2] - 1)
                        elif (event.key == K_RIGHT or event.key == K_UP) and input_state["position"][2] < 90:
                            input_state["position"][2] = min(sim.N - 1, input_state["position"][2] + 1)
                    else:
                        if event.key == K_LEFT and input_state["position"][0] > 10:
                            input_state["position"][0] = max(0, input_state["position"][0] - 1)
                        elif event.key == K_RIGHT and input_state["position"][0] < 90:
                            input_state["position"][0] = min(sim.N - 1, input_state["position"][0] + 1)
                        elif event.key == K_DOWN and input_state["position"][1] > 10:
                            input_state["position"][1] = max(0, input_state["position"][1] - 1)
                        elif event.key == K_UP and input_state["position"][1] < 90:
                            input_state["position"][1] = min(sim.N - 1, input_state["position"][1] + 1)

                    if event.key == K_RETURN or event.key == K_KP_ENTER:
                        # Finalize dynamic source placement
                        sim.add_dynamic_source(
                            input_state["shape"],
                            input_state["behaviour"],
                            input_state["axis"],
                            tuple(input_state["position"]),
                            step_number
                        )
                        
                        # Reset input state
                        input_state["stage"] = 0
                        input_state["shape"] = None
                        input_state["behaviour"] = None
                        input_state["axis"] = None
                        input_state["position"] = [sim.N // 2, sim.N // 2, sim.N // 2]

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 4:
                    camera_distance = max(35, camera_distance - 4)
                elif event.button == 5:
                    camera_distance = min(180, camera_distance + 4)

            elif event.type == MOUSEMOTION:
                if event.buttons[0]:  # Left mouse button held
                    # Rotate the cube.
                    dx_mouse, dy_mouse = event.rel
                    rotation_y += dx_mouse * 0.5
                    rotation_x += dy_mouse * 0.5

            elif event.type == MOUSEWHEEL:
                camera_distance = max(
                    35,
                    min(180, camera_distance - event.y * 4)
                )

        physics_ms = 0.0
        if not paused:
            start = time.perf_counter()
            sim.step(step_number)
            # Wait for GPU physics calculations to finish.
            cp.cuda.get_current_stream().synchronize()
            physics_ms = (time.perf_counter() - start) * 1000
            step_number += 1

        render_start = time.perf_counter()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, -camera_distance)
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)
                
        draw_room_tiles(sim.N)
        point_count = draw_field()
        draw_box(sim.N)
        draw_source()
        draw_preview_source()  # NEW: Show source while positioning
        draw_axes(sim.N)
        draw_overlay(small_font, large_font, projected_axis_labels(sim.N))

        pygame.display.flip()

        render_ms = (time.perf_counter() - render_start) * 1000

        clock.tick(60)

        if step_number % 60 == 0 and not paused:
            print(
                f"Physics: {physics_ms:.1f} ms | "
                f"Viewer: {render_ms:.1f} ms | "
                f"Points: {point_count} | "
                f"FPS: {clock.get_fps():.1f}"
            )

    glDeleteBuffers(1, [field_vbo])
    pygame.quit()

if __name__ == "__main__":
    main()