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

import yee_physics as sim
print("E type:", type(sim.E))
print("E device:", sim.E.device)

# Window and camera settings.
WIDTH, HEIGHT = 1000, 800
camera_distance = 200.0
rotation_x, rotation_y = 25.0, 35.0
paused = False
step_number = 0
FIELD_OPACITY = 0.05
OPACITY_MIN = 0.025
OPACITY_MAX = 0.10

SLIDER_X = 20
SLIDER_Y = HEIGHT - 132
SLIDER_HEIGHT = 200

# Program state for user input (when placing current sources).
input_state = {
    "stage": 0,
    "shape": None,
    "behaviour": None,
    "axis": None
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

source_coordinates = np.array([], dtype=np.float32)

def update_source_display():
    """Updates the spatial mask tracking where sources are for rendering."""
    global source_coordinates
    source_strength = cp.sqrt(cp.sum(sim.J_shape**2, axis=0))
    if source_strength.max() > 0:
        source_mask_gpu = (source_strength > 0.25 * source_strength.max())
        source_mask = cp.asnumpy(source_mask_gpu)
        source_coordinates = np.ascontiguousarray(coordinates[source_mask.ravel()], dtype=np.float32)

update_source_display()

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
    float strength = abs(a_field) / max(u_peak, 1e-30);
    v_strength = clamp(strength, 0.0, 1.0);

    // Hide weak field points without CPU filtering.
    if (strength <= u_threshold) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    } else {
        gl_Position = gl_ModelViewProjectionMatrix * vec4(a_position, 1.0);
    }

    if (a_field >= 0.0) {
        v_color = vec3(1.0, 0.05, 0.0);  // Positive: Red
    } else {
        v_color = vec3(0.0, 0.05, 1.0);  // Negative: Blue
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
    float alpha = u_opacity * sqrt(v_strength);
    gl_FragColor = vec4(v_color, alpha);
}
"""

def draw_field():
    # Calculate field peak on CUDA.
    field_gpu = sim.E[1]
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
        peak
    )
    glUniform1f(
        glGetUniformLocation(shader_program, "u_threshold"),
        0.01
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
    """Show the source region as bright yellow points."""
    if len(source_coordinates) == 0:
        return

    # Draw on top of the field cloud so the source remains easy to locate.
    glDisable(GL_DEPTH_TEST)
    glColor3f(1.0, 0.85, 0.1)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, source_coordinates)
    glPointSize(7.0)
    glDrawArrays(GL_POINTS, 0, len(source_coordinates))
    glDisableClientState(GL_VERTEX_ARRAY)
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

def update_opacity_from_mouse(mouse_y):
    global FIELD_OPACITY

    # Convert Pygame Y to OpenGL Y.
    y = HEIGHT - mouse_y

    fraction = (y - (SLIDER_Y - SLIDER_HEIGHT)) / SLIDER_HEIGHT
    fraction = max(0.0, min(1.0, fraction))

    FIELD_OPACITY = OPACITY_MIN + fraction * (OPACITY_MAX - OPACITY_MIN)


def mouse_over_opacity_slider(mouse_pos):
    x, y = mouse_pos

    slider_top = HEIGHT - SLIDER_Y
    slider_bottom = slider_top + SLIDER_HEIGHT

    return (
        abs(x - SLIDER_X) <= 14
        and slider_top <= y <= slider_bottom
    )


def draw_opacity_slider(font):
    x = SLIDER_X
    y = SLIDER_Y

    fraction = (FIELD_OPACITY - OPACITY_MIN) / (OPACITY_MAX - OPACITY_MIN)
    knob_y = y - SLIDER_HEIGHT + fraction * SLIDER_HEIGHT

    draw_text(font, x, y + 15, f"Field opacity: {FIELD_OPACITY:.3f}")

    # Background track.
    glColor3f(0.25, 0.28, 0.35)
    glBegin(GL_QUADS)
    glVertex2f(x - 3, y)
    glVertex2f(x + 3, y)
    glVertex2f(x + 3, y - SLIDER_HEIGHT)
    glVertex2f(x - 3, y - SLIDER_HEIGHT)
    glEnd()

    # Filled portion, from bottom to knob.
    glColor3f(0.95, 0.55, 0.15)
    glBegin(GL_QUADS)
    glVertex2f(x - 3, y - SLIDER_HEIGHT)
    glVertex2f(x + 3, y - SLIDER_HEIGHT)
    glVertex2f(x + 3, knob_y)
    glVertex2f(x - 3, knob_y)
    glEnd()

    # Draggable knob.
    glColor3f(1.0, 0.9, 0.7)
    glBegin(GL_QUADS)
    glVertex2f(x - 9, knob_y - 5)
    glVertex2f(x + 9, knob_y - 5)
    glVertex2f(x + 9, knob_y + 5)
    glVertex2f(x - 9, knob_y + 5)
    glEnd()

def draw_overlay(font, axis_labels):
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
        draw_text(font, x, y, label, color)

    # Legend: screen-space text remains fixed while the scene rotates.
    draw_text(font, 15, HEIGHT - 53, "Yellow points: current-source region", (255, 220, 70))
    draw_text(font, 15, HEIGHT - 78, "Ey field: red positive, blue negative")
    draw_text(font, 15, 18, "Drag: rotate | Wheel: zoom | Space: pause | Esc: quit")
    draw_opacity_slider(font)


    prompt_color = (255, 230, 100)
    prompt_y = HEIGHT - 110
    
    if input_state["stage"] == 0:
        draw_text(font, 15, prompt_y, "Press 'N' to configure a new source")
    elif input_state["stage"] == 1:
        draw_text(font, 15, prompt_y, "Select current source shape: [1] Vortex Loop   [2] Dipole", prompt_color)
    elif input_state["stage"] == 2:
        draw_text(font, 15, prompt_y, "Select current source behaviour: [1] Gaussian Pulse   [2] Continuous Wave   [3] Packet Burst", prompt_color)
    elif input_state["stage"] == 3:
        draw_text(font, 15, prompt_y, "Select current source orientation direction: [1] X   [2] Y   [3] Z", prompt_color)
    elif input_state["stage"] == 4:
        draw_text(font, 15, prompt_y, "Click anywhere on the simulation grid to place the source.", (100, 255, 100))

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

def main():
    """Run the interactive 3D viewer."""
    global camera_distance, rotation_x, rotation_y, paused, step_number, field_vbo

    pygame.init()
    font = pygame.font.Font(None, 23)
    pygame.display.set_caption("3D Yee FDTD | Ey field and source")
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
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
    opacity_dragging = False
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
                
                elif input_state["stage"] == 0 and event.key == K_n:
                    input_state["stage"] = 1
                
                elif input_state["stage"] == 1:
                    if event.key == K_1:
                        input_state["shape"] = "Vortex Loop"
                        input_state["stage"] = 2
                    elif event.key == K_2:
                        input_state["shape"] = "Dipole"
                        input_state["stage"] = 2
                
                elif input_state["stage"] == 2:
                    if event.key == K_1:
                        input_state["behaviour"] = "Gaussian Pulse"
                        input_state["stage"] = 3
                    elif event.key == K_2:
                        input_state["behaviour"] = "Continuous Wave"
                        input_state["stage"] = 3
                    elif event.key == K_3:
                        input_state["behaviour"] = "Packet Burst"
                        input_state["stage"] = 3
                
                elif input_state["stage"] == 3:
                    if event.key == K_1:
                        input_state["axis"] = 0
                        input_state["stage"] = 4
                    elif event.key == K_2:
                        input_state["axis"] = 1
                        input_state["stage"] = 4
                    elif event.key == K_3:
                        input_state["axis"] = 2
                        input_state["stage"] = 4

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    opacity_dragging = mouse_over_opacity_slider(event.pos)
                    if opacity_dragging:
                        update_opacity_from_mouse(event.pos[1])

                    elif input_state["stage"] == 4:
                        mouse_x, mouse_y = event.pos
                        target_coord = get_raycast_grid_coordinate(mouse_x, mouse_y)
                        
                        # Add new active source locally to the simulation engine
                        sim.add_dynamic_source(
                            input_state["shape"],
                            input_state["behaviour"],
                            input_state["axis"],
                            target_coord,
                            step_number  # Pass the current step number here
                        )

                        # Refresh the buffer coordinates for drawing the source location
                        update_source_display()
                        
                        # Reset input state to exit placement mode
                        input_state["stage"] = 0
                        input_state["shape"] = None
                        input_state["behaviour"] = None
                        input_state["axis"] = None

                elif event.button == 4:
                    camera_distance = max(35, camera_distance - 4)
                elif event.button == 5:
                    camera_distance = min(180, camera_distance + 4)

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    opacity_dragging = False

            elif event.type == MOUSEMOTION:

                if event.buttons[0]:  # Left mouse button held

                    if opacity_dragging:
                        # Move the slider vertically.
                        update_opacity_from_mouse(event.pos[1])
                    else:
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

        point_count = draw_field()
        draw_box(sim.N)
        draw_source()
        draw_axes(sim.N)
        draw_overlay(font, projected_axis_labels(sim.N))

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