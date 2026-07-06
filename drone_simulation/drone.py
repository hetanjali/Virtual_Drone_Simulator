import cv2
import mediapipe as mp
import pygame
import time
import os
import math

# -------- PYGAME SETUP --------
pygame.init()
WIDTH, HEIGHT = 900, 650
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture Drone Simulator")
clock = pygame.time.Clock()

font_large = pygame.font.SysFont("monospace", 28, bold=True)
font_small = pygame.font.SysFont("monospace", 18)

drone_x, drone_y = WIDTH // 2, HEIGHT // 2
speed = 4
drone_angle = 0.0
is_airborne = False
frame_count = 0

# -------- MEDIAPIPE SETUP --------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'hand_landmarker.task')

try:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1
    )
    landmarker = HandLandmarker.create_from_options(options)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

cap = cv2.VideoCapture(0)
prev_time = 0
DELAY = 0.25
command = "LANDED"
fingers_display = []


# -------- FINGER DETECTION --------
def count_fingers(lm_list, handedness):
    if len(lm_list) < 21:
        return []
    fingers = []
    if handedness == 'Right':
        fingers.append(1 if lm_list[4][1] > lm_list[3][1] else 0)
    else:
        fingers.append(1 if lm_list[4][1] < lm_list[3][1] else 0)
    for tip_id in [8, 12, 16, 20]:
        pip_id = tip_id - 2
        fingers.append(1 if lm_list[tip_id][2] < lm_list[pip_id][2] else 0)
    return fingers


# -------- DRAW DRONE --------
def draw_drone(surface, cx, cy, angle, frame):
    """
    Draws a top-down quadcopter drone.
    cx, cy  = center position
    angle   = rotation in radians
    frame   = frame counter for propeller spin animation
    """

    def rp(px, py):
        """Rotate a local point and return screen coords."""
        c, s = math.cos(angle), math.sin(angle)
        return (int(cx + px * c - py * s),
                int(cy + px * s + py * c))

    def rpoly(pts):
        return [rp(px, py) for px, py in pts]

    # ── arm directions (45° diagonals) ──────────────────────────────────
    # index:  0=front-right  1=front-left  2=back-left  3=back-right
    arm_dirs = [
        angle - math.pi / 4,
        angle + math.pi / 4,
        angle + 3 * math.pi / 4,
        angle - 3 * math.pi / 4,
    ]
    ARM_LEN  = 85   # center → motor tip
    ARM_W    = 11   # arm half-width
    PROP_R   = 38   # propeller disc radius

    motor_centers = [
        (int(cx + ARM_LEN * math.cos(a)), int(cy + ARM_LEN * math.sin(a)))
        for a in arm_dirs
    ]

    # ── 1. PROPELLER DISCS (behind everything) ───────────────────────────
    prop_colors = [
        (160, 140, 255),   # front-right  purple CW
        ( 80, 210, 170),   # front-left   teal   CCW
        (160, 140, 255),   # back-left    purple CW
        ( 80, 210, 170),   # back-right   teal   CCW
    ]
    for i, (mx, my) in enumerate(motor_centers):
        pc  = prop_colors[i]
        spin_dir = 1 if i % 2 == 0 else -1
        blade_angle = frame * 0.15 * spin_dir

        ps = pygame.Surface((PROP_R * 2 + 6, PROP_R * 2 + 6), pygame.SRCALPHA)
        pcx2, pcy2 = PROP_R + 3, PROP_R + 3

        # 2 blades
        for b in range(2):
            ba = blade_angle + b * math.pi
            cos_b, sin_b = math.cos(ba), math.sin(ba)
            perp_cos, perp_sin = math.cos(ba + math.pi / 2), math.sin(ba + math.pi / 2)
            blen = PROP_R - 2
            bw   = 9
            tip1  = (pcx2 + int(blen * cos_b),       pcy2 + int(blen * sin_b))
            tip2  = (pcx2 + int(blen * (-cos_b)),     pcy2 + int(blen * (-sin_b)))
            def blade_poly(tip):
                return [
                    (tip[0] + int(bw * perp_cos), tip[1] + int(bw * perp_sin)),
                    (tip[0] - int(bw * perp_cos), tip[1] - int(bw * perp_sin)),
                    (pcx2   - int(bw * perp_cos), pcy2   - int(bw * perp_sin)),
                    (pcx2   + int(bw * perp_cos), pcy2   + int(bw * perp_sin)),
                ]
            pygame.draw.polygon(ps, (*pc, 200), blade_poly(tip1))
            pygame.draw.polygon(ps, (*pc, 200), blade_poly(tip2))
            pygame.draw.polygon(ps, (*pc, 255), blade_poly(tip1), 1)
            pygame.draw.polygon(ps, (*pc, 255), blade_poly(tip2), 1)

        # faint motion-blur ring
        pygame.draw.circle(ps, (*pc, 40), (pcx2, pcy2), PROP_R,     4)
        pygame.draw.circle(ps, (*pc, 20), (pcx2, pcy2), PROP_R - 9, 2)

        surface.blit(ps, (mx - PROP_R - 3, my - PROP_R - 3))

    # ── 2. ARMS ──────────────────────────────────────────────────────────
    ARM_INNER = 24   # where arm starts (from center)
    ARM_OUTER = ARM_LEN - 14

    for i, a in enumerate(arm_dirs):
        cos_a, sin_a = math.cos(a), math.sin(a)
        perp_cos = math.cos(a + math.pi / 2)
        perp_sin = math.sin(a + math.pi / 2)
        hw = ARM_W / 2

        arm_poly = [
            (int(cx + ARM_INNER * cos_a + hw * perp_cos),
             int(cy + ARM_INNER * sin_a + hw * perp_sin)),
            (int(cx + ARM_INNER * cos_a - hw * perp_cos),
             int(cy + ARM_INNER * sin_a - hw * perp_sin)),
            (int(cx + ARM_OUTER * cos_a - hw * perp_cos),
             int(cy + ARM_OUTER * sin_a - hw * perp_sin)),
            (int(cx + ARM_OUTER * cos_a + hw * perp_cos),
             int(cy + ARM_OUTER * sin_a + hw * perp_sin)),
        ]
        pygame.draw.polygon(surface, (65, 64, 60),   arm_poly)
        pygame.draw.polygon(surface, (120, 118, 110), arm_poly, 1)

        # center stripe
        pygame.draw.line(surface, (140, 138, 130),
                         (int(cx + (ARM_INNER + 2) * cos_a), int(cy + (ARM_INNER + 2) * sin_a)),
                         (int(cx + (ARM_OUTER - 2) * cos_a), int(cy + (ARM_OUTER - 2) * sin_a)), 1)

    # ── 3. MOTOR MOUNTS ──────────────────────────────────────────────────
    for i, (mx, my) in enumerate(motor_centers):
        pygame.draw.circle(surface, (75, 66, 175), (mx, my), 15)   # purple ring
        pygame.draw.circle(surface, (35, 30, 88),  (mx, my), 10)   # dark core
        pygame.draw.circle(surface, (130, 120, 210), (mx - 3, my - 3), 4)  # shine

        # 4 screw holes
        for sc in range(4):
            sa = angle + sc * math.pi / 2 + math.pi / 4
            pygame.draw.circle(surface,
                               (55, 53, 50),
                               (int(mx + 12 * math.cos(sa)), int(my + 12 * math.sin(sa))),
                               2)

    # ── 4. LEDs on arm tips ───────────────────────────────────────────────
    led_colors = [
        (239, 159,  39),   # front-right  amber
        (239, 159,  39),   # front-left   amber
        (226,  75,  74),   # back-left    red
        (226,  75,  74),   # back-right   red
    ]
    for i, a in enumerate(arm_dirs):
        lx = int(cx + (ARM_LEN + 4) * math.cos(a))
        ly = int(cy + (ARM_LEN + 4) * math.sin(a))
        lc = led_colors[i]
        pygame.draw.circle(surface, lc, (lx, ly), 5)
        gs = pygame.Surface((18, 18), pygame.SRCALPHA)
        pygame.draw.circle(gs, (*lc, 80), (9, 9), 9)
        surface.blit(gs, (lx - 9, ly - 9))

    # ── 5. OCTAGONAL BODY ────────────────────────────────────────────────
    body_pts = rpoly([
        (28 * math.cos(math.pi / 8 + i * math.pi / 4),
         28 * math.sin(math.pi / 8 + i * math.pi / 4))
        for i in range(8)
    ])
    pygame.draw.polygon(surface, (38, 37, 34), body_pts)
    pygame.draw.polygon(surface, (110, 108, 100), body_pts, 2)

    inner_pts = rpoly([
        (17 * math.cos(math.pi / 8 + i * math.pi / 4),
         17 * math.sin(math.pi / 8 + i * math.pi / 4))
        for i in range(8)
    ])
    pygame.draw.polygon(surface, (55, 54, 50), inner_pts)
    pygame.draw.polygon(surface, (90, 88, 82), inner_pts, 1)

    # cross brace
    for ang_offset in [0, math.pi / 4]:
        pygame.draw.line(surface, (90, 88, 82),
                         rp(-17 * math.cos(ang_offset), -17 * math.sin(ang_offset)),
                         rp( 17 * math.cos(ang_offset),  17 * math.sin(ang_offset)), 1)

    # ── 6. CAMERA (small square, NOT a circle) ───────────────────────────
    cam_pts = rpoly([(-7, -7), (7, -7), (7, 7), (-7, 7)])
    pygame.draw.polygon(surface, (12, 90, 68),  cam_pts)
    pygame.draw.polygon(surface, (70, 190, 150), cam_pts, 1)

    # tiny lens dot (3px — clearly not a "circle", just a lens detail)
    pygame.draw.circle(surface, (70, 190, 150), (cx, cy), 3)


# -------- GRID BACKGROUND --------
def draw_grid(surface):
    surface.fill((18, 18, 20))
    for gx in range(0, WIDTH, 40):
        pygame.draw.line(surface, (26, 26, 30), (gx, 0), (gx, HEIGHT))
    for gy in range(0, HEIGHT, 40):
        pygame.draw.line(surface, (26, 26, 30), (0, gy), (WIDTH, gy))


# -------- HUD --------
def draw_hud(surface, cmd, fingers_up, airborne):
    hud = pygame.Surface((270, 240), pygame.SRCALPHA)
    pygame.draw.rect(hud, (10, 10, 10, 170), (0, 0, 270, 240), border_radius=12)
    surface.blit(hud, (14, 14))

    cmd_colors = {
        "UP":      (80,  210, 170),
        "DOWN":    (160, 140, 255),
        "LEFT":    (239, 159,  39),
        "RIGHT":   (226,  75,  74),
        "TAKEOFF": (80,  210, 170),
        "LANDED":  (136, 135, 128),
        "STOP":    (239, 159,  39),
    }
    col = cmd_colors.get(cmd, (200, 200, 200))
    surface.blit(font_small.render("COMMAND", True, (100, 100, 100)), (26, 22))
    surface.blit(font_large.render(cmd, True, col), (26, 42))

    fc = sum(fingers_up) if fingers_up else 0
    surface.blit(font_small.render(f"Fingers up: {fc}", True, (150, 150, 150)), (26, 82))

    guide = [
        ("0", "STOP"),
        ("1", "UP"),
        ("2", "DOWN"),
        ("3", "LEFT"),
        ("4", "RIGHT"),
        ("5", "TAKEOFF"),
    ]
    for i, (g, c) in enumerate(guide):
        active = (c == cmd)
        gc = (210, 210, 210) if active else (70, 70, 70)
        surface.blit(font_small.render(f"  {g} finger{'s' if g!='1' else ''} → {c}", True, gc),
                     (26, 108 + i * 19))

    status_col = (80, 210, 170) if airborne else (226, 75, 74)
    surface.blit(font_small.render(
        "● AIRBORNE" if airborne else "● LANDED", True, status_col), (26, 230))


# -------- MAIN LOOP --------
running = True

while running:
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # ── Camera ──────────────────────────────────────────────────────────
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect_for_video(mp_image, frame_count * 33)
    frame_count += 1

    if results.hand_landmarks:
        for hi, hand_landmarks in enumerate(results.hand_landmarks):
            handedness = 'Right'
            if results.handedness and len(results.handedness) > hi:
                handedness = results.handedness[hi][0].category_name

            h, w, _ = img.shape
            lm_list = [[lid, int(lm.x * w), int(lm.y * h), lm.z]
                       for lid, lm in enumerate(hand_landmarks)]

            for _, lmx, lmy, _ in lm_list:
                cv2.circle(img, (lmx, lmy), 4, (80, 210, 170), -1)

            fingers_display = count_fingers(lm_list, handedness)
            total = sum(fingers_display)

            current_time = time.time()
            if current_time - prev_time > DELAY:
                if   total == 0: command = "STOP"
                elif total == 1: command = "UP"
                elif total == 2: command = "DOWN"
                elif total == 3: command = "LEFT"
                elif total == 4: command = "RIGHT"
                elif total == 5:
                    command = "TAKEOFF"
                    is_airborne = True
                prev_time = current_time

            cv2.putText(img, f'{command} [{total}]', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 210, 170), 2)
    else:
        if time.time() - prev_time > DELAY:
            if command not in ("LANDED", "TAKEOFF"):
                command = "STOP"
            prev_time = time.time()

    cv2.imshow("Hand Tracking", img)

    # ── Move drone ───────────────────────────────────────────────────────
    if is_airborne:
        drone_angle += 0.015
        if command == "UP":    drone_y -= speed
        elif command == "DOWN":  drone_y += speed
        elif command == "LEFT":  drone_x -= speed
        elif command == "RIGHT": drone_x += speed

    drone_x = max(100, min(WIDTH  - 100, drone_x))
    drone_y = max(100, min(HEIGHT - 100, drone_y))

    # ── Draw ─────────────────────────────────────────────────────────────
    draw_grid(win)
    draw_drone(win, drone_x, drone_y, drone_angle, frame_count)
    draw_hud(win, command, fingers_display, is_airborne)

    fps_txt = font_small.render(f"FPS: {int(clock.get_fps())}", True, (55, 55, 65))
    win.blit(fps_txt, (WIDTH - 100, HEIGHT - 28))

    pygame.display.update()

    if cv2.waitKey(1) & 0xFF == 27:
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
pygame.quit()